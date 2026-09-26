"""Two-stage LightGBM matcher (v2).

stage 1  pair model on the 24 base features + features_v2 (legal/entity type, compound numbers,
         word categories, address parts, namesakes/crowding, group-without-model)
stage 2  pair model re-scoring with the stage-1 probabilities of the record's other candidates:
         strong-candidate counts, both sources present, second-best, gap, the top candidate's
         warning flags, and whether this candidate deviates from its group's majority
decision per-country threshold vs per-record expected-F0.5 choice (incl. predicting nothing),
         picked on half A of the held-out records and reported on half B; one owner per S2/S3.

Usage: python model_v2.py train     # features, ablations, final models -> work/v2/
       python model_v2.py predict   # test features in chunks -> output/, validator
"""
import json
import subprocess
import sys
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import DATA, ROOT, WORK, read_pairs, read_source
from evaluation import decide, f05_macro, new_output_dir, write_tsv
from features_v2 import NEW_FEATURES, country_features
from lgbm_matcher import FEATURES as BASE_FEATURES

V2 = WORK / "v2"
V2.mkdir(exist_ok=True)
MODEL_K = 15
P1 = BASE_FEATURES + NEW_FEATURES
S2_FEATURES = ["p1", "p1_rank", "p1_max", "p1_second", "p1_gap", "n_p1_05", "n_p1_08", "src_strong", "p1_sum",
               "top_cnum_subset", "top_addr_empty", "top_ent_changed", "dev_legal", "dev_num", "dev_key"]
P2 = P1 + S2_FEATURES
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=16, seed=7, deterministic=True, force_row_wise=True)
ROUNDS = 600


def features_chunked(cand, split, chunk_s1=60_000, on_chunk=None):
    """Features per country in S1 chunks; on_chunk(frame) can shrink each chunk right away."""
    out = []
    for country in cand["country"].unique().sort():
        c = cand.filter(pl.col("country") == country)
        ids = c["s1"].unique().sort()
        for s in range(0, len(ids), chunk_s1):
            part = c.filter(pl.col("s1").is_in(ids[s:s + chunk_s1]))
            f = country_features(part, split, country).with_columns(
                (pl.col("country") == "India").cast(pl.Float32).alias("is_india"))
            out.append(on_chunk(f) if on_chunk else f)
            del f
            print(f"   {country}: features for {min(s + chunk_s1, len(ids)):,}/{len(ids):,} S1", flush=True)
    return pl.concat(out, how="diagonal_relaxed")


def stage2_features(df):
    """Group features from stage-1 probabilities (p1) of the record's other candidates."""
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
    # majority of the OTHER strong candidates (excluding self)
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


def oof(df, feats, label="y", folds=5):
    X, y, g = df.select(feats).to_numpy(), df[label].to_numpy(), df["s1"].to_numpy()
    out = np.zeros(len(df), dtype=np.float32)
    for tr, te in GroupKFold(folds).split(X, y, g):
        out[te] = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS).predict(X[te])
    return out


def decide_expected_f(c, miss_rate):
    """Per record: accept the top-n candidates maximising expected F0.5 (n may be 0).
    E[F(n)] ~= 1.25*TP / (0.25*T + n) with TP = sum of top-n p, T = sum of all p + expected
    trues missed by candidate search; E[F(0)] = P(no true match)."""
    c = c.sort(["s1", "score"], descending=[False, True]).with_columns(
        pl.col("score").cum_sum().over("s1").alias("tp"), pl.int_range(1, pl.len() + 1).over("s1").alias("n"),
        (pl.col("score").sum().over("s1") * (1 + miss_rate)).alias("T"),
        ((1 - pl.col("score")).log().sum().over("s1")).exp().alias("p_none"))
    c = c.with_columns((1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("n"))).alias("ef"))
    best = c.group_by("s1").agg(pl.col("ef").max().alias("ef_max"), pl.col("n").filter(pl.col("ef") == pl.col("ef").max()).first().alias("n_best"),
                                pl.col("p_none").first())
    keep = best.filter(pl.col("ef_max") > pl.col("p_none")).select("s1", "n_best")
    m = c.join(keep, on="s1").filter(pl.col("n") <= pl.col("n_best")).select("s1", "cand", "score")
    return m.sort("score", descending=True).unique("cand", keep="first").select("s1", "cand")


def loss_breakdown(pred, truth, cand, s1_ids):
    """Split the macro F0.5 loss into causes. pred/truth/cand: DataFrames with s1, cand."""
    P, T, C = defaultdict(set), defaultdict(set), defaultdict(set)
    for d, df in ((P, pred), (T, truth), (C, cand)):
        for s, c in df.select("s1", "cand").rows():
            d[s].add(c)
    f = lambda tp, nt, npred: 1.0 if nt == 0 and npred == 0 else (0.0 if tp == 0 else 1.25 * tp / (0.25 * nt + npred))
    loss = Counter()
    for s in s1_ids:
        p, t, c = P[s], T[s], C[s]
        tp, nt = len(p & t), len(t)
        now = f(tp, nt, len(p))
        if nt == 0:
            loss["no-match record given a match"] += 1 - now
            continue
        no_fp = f(tp, nt, tp)                           # same true matches, false ones removed
        all_cand = f(len(t & c), nt, len(t & c))        # every true candidate accepted
        loss["false accepts"] += no_fp - now
        loss["true candidates rejected"] += all_cand - no_fp
        loss["true match not in top-K"] += 1 - all_cand
    n = len(s1_ids)
    for k, v in loss.most_common():
        print(f"   {k:32s} {v / n:.4f}")
    print(f"   {'total loss':32s} {sum(loss.values()) / n:.4f}")


def evaluate(df, col, truth, ids_by_country, halves, verbose=False):
    """Pick the decision rule/threshold on half A, report F0.5 on half B, per country."""
    res = {}
    for country, ids in ids_by_country.items():
        a, b = [i for i in ids if halves[i] == 0], [i for i in ids if halves[i] == 1]
        c = df.filter(pl.col("country") == country).with_columns(pl.col(col).alias("score"))
        ca, cb = c.filter(pl.col("s1").is_in(a)), c.filter(pl.col("s1").is_in(b))
        ta, tb = truth.filter(pl.col("s1").is_in(a)), truth.filter(pl.col("s1").is_in(b))
        thr = max(np.arange(0.3, 0.96, 0.025), key=lambda t: f05_macro(decide(ca, MODEL_K, t), ta, a))
        miss = max((0.0, 0.05, 0.1, 0.2), key=lambda m: f05_macro(decide_expected_f(ca, m), ta, a))
        f_thr = f05_macro(decide(cb, MODEL_K, thr), tb, b)
        f_ef = f05_macro(decide_expected_f(cb, miss), tb, b)
        if verbose:
            print(f"loss breakdown {country}, half B ({col}):")
            loss_breakdown(decide_expected_f(cb, miss) if f_ef > f_thr else decide(cb, MODEL_K, thr), tb, cb, b)
        res[country] = dict(thr=float(thr), miss=miss, f_thr=f_thr, f_ef=f_ef,
                            rule="expected_f" if f_ef > f_thr else "threshold", f=max(f_thr, f_ef))
    return res


def train():
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"}).filter(pl.col("s1").is_in(val["entity_id"]))
    cand = (pl.read_parquet(WORK / "val_candidates.parquet").filter(pl.col("rank") <= MODEL_K)
              .drop("country").join(val.select(pl.col("entity_id").alias("s1"), "country"), on="s1"))
    feat_path = V2 / "train_features.parquet"
    if feat_path.exists():
        df = pl.read_parquet(feat_path)
    else:
        df = features_chunked(cand, "train").join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "cand"], how="left") \
               .with_columns(pl.col("y").fill_null(0))
        df.write_parquet(feat_path)
    print(f"training pairs: {df.height:,} from {df['s1'].n_unique():,} S1; positives {int(df['y'].sum()):,}", flush=True)
    ids = {c: val.filter(pl.col("country") == c)["entity_id"].to_list() for c in ("India", "US")}
    rng = np.random.default_rng(0)
    halves = {i: int(rng.integers(2)) for c in ids.values() for i in c}
    report = {}
    df = df.with_columns(pl.Series("p_base", oof(df, BASE_FEATURES)))
    report["base_24_features"] = evaluate(df, "p_base", truth, ids, halves)
    print("A) base features:", report["base_24_features"], flush=True)
    df = df.with_columns(pl.Series("p1", oof(df, P1)))
    report["v2_stage1"] = evaluate(df, "p1", truth, ids, halves)
    print("B) + v2 features:", report["v2_stage1"], flush=True)
    df = stage2_features(df)
    df = df.with_columns(pl.Series("p2", oof(df, P2)))
    report["v2_stage2"] = evaluate(df, "p2", truth, ids, halves, verbose=True)
    print("C) + stage 2:", report["v2_stage2"], flush=True)
    keep = [c for c in ("s1", "cand", "country", "rank", "name_sim", "addr_sim", "via", "y", "p_base", "p1", "p2") if c in df.columns]
    df.select(keep).write_parquet(V2 / "oof.parquet")          # out-of-fold scores: decision experiments run without retraining
    pl.DataFrame({"s1": list(halves), "half": list(halves.values())}).write_parquet(V2 / "halves.parquet")
    # choose the final configuration per country from half B results
    use_stage2 = all(report["v2_stage2"][c]["f"] >= report["v2_stage1"][c]["f"] for c in ids)
    final = report["v2_stage2" if use_stage2 else "v2_stage1"]
    # France: no labels -> the stricter (higher-threshold) of the two countries' settings
    final["France"] = max(final.values(), key=lambda r: r["thr"])
    X1, y = df.select(P1).to_numpy(), df["y"].to_numpy()
    lgb.train(PARAMS, lgb.Dataset(X1, y), ROUNDS).save_model(str(V2 / "stage1.txt"))
    if use_stage2:
        lgb.train(PARAMS, lgb.Dataset(df.select(P2).to_numpy(), y), ROUNDS).save_model(str(V2 / "stage2.txt"))
    (V2 / "config.json").write_text(json.dumps(dict(use_stage2=use_stage2, decision=final, report=report), indent=1))
    print("saved models and config:", json.dumps(final, indent=1), flush=True)


def predict():
    cfg = json.loads((V2 / "config.json").read_text())
    m1 = lgb.Booster(model_file=str(V2 / "stage1.txt"))
    m2 = lgb.Booster(model_file=str(V2 / "stage2.txt")) if cfg["use_stage2"] else None
    s1 = read_source("test", 1, ["entity_id", "country"])
    kept, matches = [], []
    for country in ("France", "India", "US"):
        cand = (pl.read_parquet(WORK / f"test_candidates_{country}.parquet").filter(pl.col("rank") <= MODEL_K)
                  .with_columns(pl.lit(country).alias("country")))
        def score_chunk(f):          # score each S1 chunk immediately, keep only what decisions need
            f = f.with_columns(pl.Series("p1", m1.predict(f.select(P1).to_numpy()).astype(np.float32)))
            if m2 is not None:
                f = stage2_features(f)
                f = f.with_columns(pl.Series("p", m2.predict(f.select(P2).to_numpy()).astype(np.float32)))
            else:
                f = f.with_columns(pl.col("p1").alias("p"))
            return f.select("s1", "cand", pl.col("p").alias("score"), "rank")
        c = features_chunked(cand, "test", on_chunk=score_chunk)
        del cand
        d = cfg["decision"][country]
        matches.append(decide_expected_f(c, d["miss"]) if d["rule"] == "expected_f" else decide(c, MODEL_K, d["thr"]))
        kept.append(c.select("s1", "cand"))
        print(f"{country}: {c.height:,} pairs scored, {matches[-1].height:,} matches", flush=True)
        del c
    kept, matches = pl.concat(kept), pl.concat(matches).unique("cand", keep="first")
    ids = s1["entity_id"]
    OUT = new_output_dir("lgbm_two_stage")
    write_tsv(kept, "candidate_entity_ids", OUT / "candidate_pairs.tsv", ids)
    write_tsv(matches, "matched_entity_ids", OUT / "matching_results.tsv", ids)
    print(f"matches {matches.height:,}, S1 with no match {1 - matches['s1'].n_unique() / len(ids):.1%}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(OUT / "matching_results.tsv"), "--candidate", str(OUT / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test"), "--check-ids"])


if __name__ == "__main__":
    train() if sys.argv[1] == "train" else predict()
