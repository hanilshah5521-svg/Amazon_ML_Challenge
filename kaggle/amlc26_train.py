# %% [markdown]
# # AMLC 2026: two-stage matcher on exported features (Kaggle GPU)
# Inputs (private datasets): `amlc26-feat-train` (feat parts: features + s1, cand, country, rank, y, is_val) and
# `amlc26-feat-test` (`<country>_part_*.parquet`, `val_s1.parquet`, `val_truth.parquet`).
#
# * Stage 1: XGBoost + CatBoost (GPU), K-fold out-of-fold grouped by S1 (fold = hash of s1), early stopping on a
#   5% S1 slice of the training folds. Test = mean of the fold models.
# * Stage 2: stage-1 group features of the S1 (`stage2_features`, as in model_v2.py) plus **owner competition**:
#   how strongly the OTHER S1 records claim the same S2/S3 candidate (`cand_features`).
# * Decision: per-country threshold vs per-record expected-F0.5 picked on half A of the 200k held-out S1 (same
#   halves as model_v2: `np.random.default_rng(0)`), F0.5 reported on half B. Countries without labels (France)
#   get the stricter labelled setting.
# * Output (/kaggle/working): `test_scores.parquet` (every variant's score), `config.json` (decisions + report),
#   `val_oof.parquet`. TSVs are written locally by `finalize_kaggle.py` (validator + versioned output folder).

# %%
import gc
import glob
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

T0 = time.time()
def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

SMOKE = int(os.environ.get("SMOKE", 0))            # local code check: few parts, few rounds
GPU = os.environ.get("GPU", "1") == "1"
MODELS = os.environ.get("MODELS", "xgb,cat").split(",")
FOLDS = int(os.environ.get("FOLDS", 4))
ABLATE_CAND = os.environ.get("ABLATE_CAND", "1") == "1"   # extra XGB stage 2 without owner-competition features
K = 8
OUT = os.environ.get("OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)

def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"nothing matches {pattern}"
    return hits

IN = os.environ.get("IN", "/kaggle/input")
TRAIN_FILES = find(f"{IN}/**/part_*.parquet")
TRAIN_FILES = [f for f in TRAIN_FILES if os.path.basename(f).startswith("part_")]
TEST_FILES = {c: find(f"{IN}/**/{c}_part_*.parquet") for c in ("France", "India", "US")}
VAL_S1 = find(f"{IN}/**/val_s1.parquet")[0]
VAL_TRUTH = find(f"{IN}/**/val_truth.parquet")[0]
if SMOKE:
    TRAIN_FILES = TRAIN_FILES[:1] + TRAIN_FILES[-1:]
    TEST_FILES = {c: v[:1] for c, v in TEST_FILES.items()}
log(f"train parts {len(TRAIN_FILES)}, test parts {sum(map(len, TEST_FILES.values()))}, models {MODELS}, folds {FOLDS}, gpu {GPU}")

# %%
P1 = ["name_sim", "addr_sim", "score", "rank", "via", "legal_rel", "num_rel", "core_ratio", "core_tsort", "core_tset",
      "core_partial", "key_equal", "key_ratio", "nospace_ratio", "addr_tset", "addr_empty", "len_diff", "gap_to_best",
      "score_over_best", "n_close", "name_rank", "addr_rank", "best_key", "ent_added", "ent_dropped", "ent_changed",
      "private_diff", "legal_equal", "cnum_exact", "cnum_subset", "cnum_shared", "cnum_n1", "cnum_n2", "cnum_part_changed",
      "cnum_trunc", "cnum_close", "cnum_range_in", "first_num_logdiff", "cnum_off_4_20", "cnum_unrelated", "first_tok_equal",
      "invented_name", "acronym", "web_name", "web_contains", "place_in_name", "decoy_added", "noise_added", "decoy_dropped",
      "n_decoy_added", "legal_trans_logcount", "legal_trans_unseen", "alias_marker", "tokens_identical", "reordered_only",
      "addr_parts_reordered", "nm_shared_idf", "nm_shared_max", "nm_un1_idf", "nm_un1_max", "nm_un2_idf", "nm_un2_max",
      "ad_shared_idf", "ad_shared_max", "ad_un1_idf", "ad_un1_max", "ad_un2_idf", "ad_un2_max", "gen_un1", "gen_un2",
      "state_rel", "s1_key_n", "pool_key_n1", "pool_key_n2", "s1_addr_n", "pool_addr_n", "grp_same_key", "grp_max_addr",
      "grp_n", "is_s2", "best_addr_among_namesakes", "addr_gap_to_best", "grp_any_num_exact", "num_shift_vs_sibling",
      "grp_any_legal_equal"]
S2_FEATURES = ["p1", "p1_rank", "p1_max", "p1_second", "p1_gap", "n_p1_05", "n_p1_08", "src_strong", "p1_sum",
               "top_cnum_subset", "top_addr_empty", "top_ent_changed", "dev_legal", "dev_num", "dev_key"]
CAND_FEATURES = ["c_n_claim", "c_p1_rank", "c_other_max", "c_gap", "c_n_strong_other", "c_mutual_best"]
N1, N2, N3 = len(P1), len(P1) + len(S2_FEATURES), len(P1) + len(S2_FEATURES) + len(CAND_FEATURES)
GROUP_COLS = ["is_s2", "cnum_subset", "addr_empty", "ent_changed", "legal_equal", "key_equal"]   # used by stage2_features

# %%
def load(files, meta):
    """Features straight into one preallocated float32 matrix (room for the stage-2 columns), plus a slim frame."""
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    X = np.zeros((n, N3), dtype=np.float32)
    frames, i = [], 0
    for f in files:
        d = pl.read_parquet(f, columns=list(dict.fromkeys(meta + P1)))
        X[i:i + d.height, :N1] = d.select([pl.col(c).cast(pl.Float32) for c in P1]).to_numpy()
        frames.append(d.select(meta + [pl.col(c).cast(pl.Float32) for c in GROUP_COLS]))
        i += d.height
        del d
    return X, pl.concat(frames)

tr_X, tr = load(TRAIN_FILES, ["s1", "cand", "country", "rank", "y", "is_val"])
y = tr["y"].to_numpy().astype(np.float32)
te_X, te = [], []
for c, files in TEST_FILES.items():
    Xc, dc = load(files, ["s1", "cand", "country", "rank"])
    te_X.append(Xc); te.append(dc)
te_X, te = np.concatenate(te_X), pl.concat(te)
gc.collect()
log(f"train {tr_X.shape} ({tr_X.nbytes / 1e9:.1f} GB), positives {int(y.sum()):,}; test {te_X.shape} ({te_X.nbytes / 1e9:.1f} GB)")

fold = (tr["s1"].hash(seed=11) % FOLDS).to_numpy()
es_mask = (tr["s1"].hash(seed=23) % 20 == 0).to_numpy()          # 5% of S1: early-stopping set inside each fold

# %%
XGB = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda" if GPU else "cpu",
           learning_rate=0.08, grow_policy="lossguide", max_leaves=127, max_depth=0, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, max_bin=256, seed=7)
CAT = dict(loss_function="Logloss", iterations=4000, learning_rate=0.1, depth=8, border_count=128, l2_leaf_reg=3,
           od_type="Iter", od_wait=100, random_seed=7, verbose=500, task_type="GPU" if GPU else "CPU")
ROUNDS = 20 if SMOKE else 4000

def predict_chunks(fn, X, rows, ncols, chunk=2_000_000):
    """Predict in row chunks; each chunk is one contiguous copy of the first ncols columns."""
    n = len(X) if rows is None else len(rows)
    take = (lambda s: X[s:s + chunk, :ncols]) if rows is None else (lambda s: X[rows[s:s + chunk], :ncols])
    return np.concatenate([fn(np.ascontiguousarray(take(s))) for s in range(0, n, chunk)]).astype(np.float32)

def fit_predict(model, ncols, trn, es, targets):
    """Fit on rows trn using the first ncols feature columns (early stopping on es); predict each (X, rows) target."""
    if model == "xgb":
        dtr = xgb.QuantileDMatrix(tr_X[trn, :ncols], y[trn])
        des = xgb.QuantileDMatrix(tr_X[es, :ncols], y[es], ref=dtr)
        b = xgb.train(XGB, dtr, ROUNDS, evals=[(des, "es")], early_stopping_rounds=100, verbose_eval=False)
        del dtr, des; gc.collect()
        it = (0, b.best_iteration + 1)
        fn = lambda A: b.inplace_predict(A, iteration_range=it)
        best = b.best_iteration
    else:
        ptr = Pool(tr_X[trn, :ncols], y[trn]); pes = Pool(tr_X[es, :ncols], y[es])
        m = CatBoostClassifier(**{**CAT, "iterations": ROUNDS}).fit(ptr, eval_set=pes, use_best_model=True)
        del ptr, pes; gc.collect()
        fn = lambda A: m.predict_proba(A)[:, 1]
        best = m.get_best_iteration()
    out = [predict_chunks(fn, X, rows, ncols) for X, rows in targets]
    return out, best

def run_stage(ncols, models, tag):
    """Out-of-fold train scores and fold-mean test scores per model, plus their mean ('ens')."""
    oof = {m: np.zeros(len(y), np.float32) for m in models}
    tst = {m: np.zeros(len(te_X), np.float32) for m in models}
    for f in range(FOLDS):
        hold = fold == f
        trn, es = np.flatnonzero(~hold & ~es_mask), np.flatnonzero(~hold & es_mask)
        for m in models:
            (p_hold, p_test), best = fit_predict(m, ncols, trn, es, [(tr_X, np.flatnonzero(hold)), (te_X, None)])
            oof[m][hold] = p_hold
            tst[m] += p_test / FOLDS
            log(f"{tag} fold {f} {m}: best iteration {best}")
    oof["ens"] = np.mean([oof[m] for m in models], axis=0)
    tst["ens"] = np.mean([tst[m] for m in models], axis=0)
    return oof, tst

# %%
def stage2_features(df):
    """Group features from stage-1 probabilities (p1) of the record's other candidates (copy of model_v2.py)."""
    strong = pl.col("p1") >= 0.5
    df = df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("p1_rank"),
        pl.col("p1").max().over("s1").alias("p1_max"),
        pl.col("p1").sort(descending=True).get(1).over("s1").fill_null(0).alias("p1_second"),
        strong.sum().over("s1").cast(pl.Float32).alias("n_p1_05"),
        (pl.col("p1") >= 0.8).sum().over("s1").cast(pl.Float32).alias("n_p1_08"),
        pl.col("is_s2").filter(strong).n_unique().over("s1").cast(pl.Float32).alias("src_strong"),
        pl.col("p1").sum().over("s1").alias("p1_sum"))
    top = pl.col("p1") == pl.col("p1_max")
    def others_mean(col):
        tot = pl.col(col).filter(strong).sum().over("s1") - pl.when(strong).then(pl.col(col)).otherwise(0)
        cnt = pl.col("n_p1_05") - strong.cast(pl.Float32)
        return (tot / cnt).fill_nan(-1).fill_null(-1)
    return df.with_columns(
        (pl.col("p1_max") - pl.col("p1")).alias("p1_gap"),
        pl.col("cnum_subset").filter(top).first().over("s1").alias("top_cnum_subset"),
        pl.col("addr_empty").filter(top).first().over("s1").alias("top_addr_empty"),
        pl.col("ent_changed").filter(top).first().over("s1").alias("top_ent_changed"),
        (others_mean("legal_equal") - pl.col("legal_equal")).alias("dev_legal"),
        (others_mean("cnum_subset") - pl.col("cnum_subset")).alias("dev_num"),
        (others_mean("key_equal") - pl.col("key_equal")).alias("dev_key"))

def cand_features(df):
    """Owner competition: how strongly the OTHER S1 records claim this S2/S3 candidate (stage-1 p1).
    One S2/S3 record has at most one S1 owner, so a candidate that another S1 claims more strongly is likely a
    decoy for this S1. Train: over all 2.2M training S1; test: over all test S1."""
    strong = (pl.col("p1") >= 0.5).cast(pl.Float32)
    df = df.with_columns(
        pl.len().over("cand").cast(pl.Float32).alias("c_n_claim"),
        pl.col("p1").rank("ordinal", descending=True).over("cand").cast(pl.Float32).alias("c_p1_rank"),
        pl.col("p1").max().over("cand").alias("_c_first"),
        pl.when(pl.len().over("cand") > 1).then(pl.col("p1").top_k(2).min().over("cand")).otherwise(0.0)
          .alias("_c_second"),                       # second-highest claim (0 when this S1 is the only claimant)
        (strong.sum().over("cand") - strong).alias("c_n_strong_other"))
    df = df.with_columns(pl.when(pl.col("c_p1_rank") == 1).then(pl.col("_c_second")).otherwise(pl.col("_c_first"))
                           .alias("c_other_max"))
    return df.with_columns((pl.col("p1") - pl.col("c_other_max")).alias("c_gap"),
                           ((pl.col("c_p1_rank") == 1) & (pl.col("p1_rank") == 1)).cast(pl.Float32).alias("c_mutual_best")
                           ).drop("_c_first", "_c_second")

def fill_stage2(X, df, p1):
    df = cand_features(stage2_features(df.with_columns(pl.Series("p1", p1))))
    X[:, N1:N3] = df.select([pl.col(c).cast(pl.Float32) for c in S2_FEATURES + CAND_FEATURES]).to_numpy()

# %%
def f05_macro(pred, truth, s1_ids):
    p = pred.group_by("s1").agg(pl.col("cand").alias("p"))
    t = truth.group_by("s1").agg(pl.col("cand").alias("t"))
    df = (pl.DataFrame({"s1": s1_ids}).join(p, on="s1", how="left").join(t, on="s1", how="left")
            .with_columns(pl.col("p").fill_null([]), pl.col("t").fill_null([])))
    df = df.with_columns(pl.col("p").list.len().alias("np"), pl.col("t").list.len().alias("nt"),
                         pl.col("p").list.set_intersection(pl.col("t")).list.len().alias("tp"))
    prec, rec = pl.col("tp") / pl.col("np"), pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
           .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    return df.select(f.mean()).item()

def decide(cand, k, thr):
    m = cand.filter((pl.col("rank") <= k) & (pl.col("score") >= thr))
    return m.sort("score", descending=True).unique("cand", keep="first").select("s1", "cand")

def decide_expected_f(c, miss_rate):
    c = c.sort(["s1", "score"], descending=[False, True]).with_columns(
        pl.col("score").cum_sum().over("s1").alias("tp"), pl.int_range(1, pl.len() + 1).over("s1").alias("n"),
        (pl.col("score").sum().over("s1") * (1 + miss_rate)).alias("T"),
        ((1 - pl.col("score")).log().sum().over("s1")).exp().alias("p_none"))
    c = c.with_columns((1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("n"))).alias("ef"))
    best = c.group_by("s1").agg(pl.col("ef").max().alias("ef_max"),
                                pl.col("n").filter(pl.col("ef") == pl.col("ef").max()).first().alias("n_best"),
                                pl.col("p_none").first())
    keep = best.filter(pl.col("ef_max") > pl.col("p_none")).select("s1", "n_best")
    m = c.join(keep, on="s1").filter(pl.col("n") <= pl.col("n_best")).select("s1", "cand", "score")
    return m.sort("score", descending=True).unique("cand", keep="first").select("s1", "cand")

val = pl.read_parquet(VAL_S1)
val = val.filter(pl.col("entity_id").is_in(tr.filter(pl.col("is_val") == 1)["s1"].unique()))
truth = pl.read_parquet(VAL_TRUTH).filter(pl.col("s1").is_in(val["entity_id"]))
ids = {c: val.filter(pl.col("country") == c)["entity_id"].to_list() for c in sorted(val["country"].unique().to_list())}
rng = np.random.default_rng(0)
halves = {i: int(rng.integers(2)) for c in ids.values() for i in c}
WEIGHT = {c: te.filter(pl.col("country") == c).select(pl.col("s1").n_unique()).item() for c in ids}   # test share
vmask = (tr["is_val"] == 1).to_numpy()
vframe = tr.filter(pl.col("is_val") == 1).select("s1", "cand", "country", "rank", "y")

def evaluate(score):
    """Decision rule/threshold picked on half A, F0.5 on half B, per labelled country."""
    df = vframe.with_columns(pl.Series("score", score[vmask]))
    res = {}
    for country, cids in ids.items():
        a, b = [i for i in cids if halves[i] == 0], [i for i in cids if halves[i] == 1]
        c = df.filter(pl.col("country") == country)
        ca, cb = c.filter(pl.col("s1").is_in(a)), c.filter(pl.col("s1").is_in(b))
        ta, tb = truth.filter(pl.col("s1").is_in(a)), truth.filter(pl.col("s1").is_in(b))
        thr = max(np.arange(0.3, 0.96, 0.025), key=lambda t: f05_macro(decide(ca, K, t), ta, a))
        miss = max((0.0, 0.05, 0.1, 0.2), key=lambda m: f05_macro(decide_expected_f(ca, m), ta, a))
        f_thr, f_ef = f05_macro(decide(cb, K, thr), tb, b), f05_macro(decide_expected_f(cb, miss), tb, b)
        res[country] = dict(thr=float(thr), miss=miss, f_thr=f_thr, f_ef=f_ef,
                            rule="expected_f" if f_ef > f_thr else "threshold", f=max(f_thr, f_ef))
    res["_weighted"] = sum(res[c]["f"] * WEIGHT[c] for c in ids) / sum(WEIGHT.values())
    return res

# %%
report, val_cols, test_cols = {}, {}, {}
oof1, tst1 = run_stage(N1, MODELS, "stage1")
for m, p in oof1.items():
    report[f"s1_{m}"] = evaluate(p); log(f"stage 1 {m}:", json.dumps(report[f"s1_{m}"]))
    val_cols[f"s1_{m}"], test_cols[f"s1_{m}"] = p[vmask], tst1[m]

fill_stage2(tr_X, tr, oof1["ens"])
fill_stage2(te_X, te, tst1["ens"])
log("stage-2 features filled (group + owner competition)")
oof2, tst2 = run_stage(N3, MODELS, "stage2")
for m, p in oof2.items():
    report[f"s2_{m}"] = evaluate(p); log(f"stage 2 {m}:", json.dumps(report[f"s2_{m}"]))
    val_cols[f"s2_{m}"], test_cols[f"s2_{m}"] = p[vmask], tst2[m]
if ABLATE_CAND and "xgb" in MODELS:                  # does owner competition help? same XGB stage 2 without it
    oof2b, tst2b = run_stage(N2, ["xgb"], "stage2-no-owner")
    report["s2nc_xgb"] = evaluate(oof2b["xgb"]); log("stage 2 xgb WITHOUT owner competition:", json.dumps(report["s2nc_xgb"]))
    val_cols["s2nc_xgb"], test_cols["s2nc_xgb"] = oof2b["xgb"][vmask], tst2b["xgb"]

# %%
best = max(report, key=lambda v: report[v]["_weighted"])
decision = {c: report[best][c] for c in ids}
decision["_default"] = max(decision.values(), key=lambda r: r["thr"])     # countries without labels (France)
log("half-B weighted F0.5 per variant:", {v: round(r["_weighted"], 5) for v, r in report.items()})
log(f"chosen: {best}", json.dumps(decision))

te.select("s1", "cand", "country", "rank").with_columns([pl.Series(k, v) for k, v in test_cols.items()]) \
  .write_parquet(f"{OUT}/test_scores.parquet")
vframe.with_columns([pl.Series(k, v) for k, v in val_cols.items()]).write_parquet(f"{OUT}/val_oof.parquet")
json.dump(dict(variant=best, decision=decision, report=report, models=MODELS, folds=FOLDS, k=K),
          open(f"{OUT}/config.json", "w"), indent=1)
log("wrote test_scores.parquet, val_oof.parquet, config.json")
