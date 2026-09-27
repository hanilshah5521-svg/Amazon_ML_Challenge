"""EXP-3/EXP-4 test-shift stress test (TRAIN_HIDE=h: EXP-4 = train half the rows on owner lists with h of the S1 hidden): test has ~2x more unowned S2/S3 per S1 than train, which is exactly what owner-competition
features could over-trust (a decoy whose real owner is absent looks 'unclaimed'). Simulate it: hide a share h of ALL training S1,
drop them from every owner list, recompute oc_* features, and score held-out half B (hidden S1 excluded) with EXP-3 models trained
on non-half-B S1 only. EXP-2 (no oc features) on the same kept S1 is the reference.
Usage: WORK_DIR=../../../work/exp3 PATTERN_FEATS=1 OWNER_FEATS=1 MODEL_K=8 OUR_CANDS_DIR=... python stress_exp3.py"""
import json, os
from pathlib import Path
import numpy as np
import polars as pl
import model_v2 as M
from common import read_pairs

W = Path(os.environ["WORK_DIR"]); E2 = W.parent / "exp2"
cfg3 = json.loads((W / "v2" / "config.json").read_text())["decision"]
cfg2 = json.loads((E2 / "v2" / "config.json").read_text())["decision"]
truth = read_pairs().rename({"m": "cand"})
halves = pl.read_parquet(W / "v2" / "halves.parquet")
B = set(halves.filter(pl.col("half") == 1)["s1"].to_list())

df = pl.read_parquet(W / "v2" / "train_features.parquet")
if M.X5:                                            # EXP-5 features ride along unchanged (they do not depend on owner lists)
    df = M.add_x5(df, "train")
p1_oof = pl.read_parquet(W / "v2" / "oof.parquet", columns=["s1", "cand", "p1"])
o2 = pl.read_parquet(E2 / "v2" / "oof.parquet", columns=["s1", "cand", "country", "p2"])
d = Path(os.environ["OUR_CANDS_DIR"]) / "train"
ids = pl.read_parquet(d / "records.parquet", columns=["rid", "entity_id"])
raw = (pl.read_parquet(d / "cands.parquet", columns=["q", "s1", "rank", "rr"])
         .join(ids.rename({"rid": "s1", "entity_id": "s1_eid"}), on="s1").join(ids.rename({"rid": "q", "entity_id": "cand"}), on="q")
         .select(pl.col("s1_eid").alias("s1"), "cand", pl.col("rank").cast(pl.Float32).alias("oc_rank"), pl.col("rr").cast(pl.Float32).alias("oc_rr")))


def oc_tables(c):
    rec = c.group_by("cand").agg(pl.len().cast(pl.Float32).alias("oc_n"), pl.col("oc_rr").max().alias("oc_rr1"),
                                 pl.col("oc_rr").top_k(2).min().alias("_rr2"), pl.len().alias("_len"))
    rec = rec.with_columns(pl.when(pl.col("_len") > 1).then(pl.col("_rr2")).otherwise(0.0).alias("oc_rr2")).drop("_rr2", "_len")
    pairs = c.join(rec, on="cand").with_columns(
        pl.when(pl.col("oc_rr") >= pl.col("oc_rr1")).then(pl.col("oc_rr2")).otherwise(pl.col("oc_rr1")).alias("oc_rr_other"))
    return pairs.select("s1", "cand", "oc_rank", "oc_rr", "oc_rr_other"), rec.select("cand", "oc_n", "oc_rr1")


def add_oc(f, p, r):                                   # same as owner_competition.add_oc, with given tables
    f = f.join(p, on=["s1", "cand"], how="left").join(r, on="cand", how="left")
    return f.with_columns(pl.col("oc_rr").is_not_null().cast(pl.Float32).alias("oc_in"), pl.col("oc_rr_other").fill_null(pl.col("oc_rr1")).fill_null(0.0),
                          pl.col("oc_n").fill_null(0.0)).with_columns((pl.col("oc_rr") - pl.col("oc_rr_other")).alias("oc_gap")).drop("oc_rr1")


# models on non-half-B S1 only (stage 2 trained with the OOF stage-1 scores, as in model_v2.train)
p0, r0 = oc_tables(raw)
TRAIN_HIDE = float(os.environ.get("TRAIN_HIDE", 0))
base = df.filter(~pl.col("s1").is_in(list(B)))
if TRAIN_HIDE:        # EXP-4: half of the training S1 see owner lists with TRAIN_HIDE of the other S1 removed (their own rows kept)
    rng = np.random.default_rng(5)
    s1s = raw["s1"].unique().sort()
    gone = set(s1s.filter(pl.Series(rng.random(len(s1s)) < TRAIN_HIDE)).to_list())
    ph, rh = oc_tables(raw.filter(~pl.col("s1").is_in(list(gone))))
    tr_ids = base["s1"].unique().sort()
    sim = set(tr_ids.filter(pl.Series(np.random.default_rng(6).random(len(tr_ids)) < 0.5)).to_list()) - gone
    tr = pl.concat([add_oc(base.filter(~pl.col("s1").is_in(list(sim))), p0, r0),
                    add_oc(base.filter(pl.col("s1").is_in(list(sim))), ph, rh)], how="diagonal_relaxed")
    print(f"EXP-4 training: {len(sim):,} S1 on owner lists with {TRAIN_HIDE:.0%} of S1 hidden", flush=True)
else:
    tr = add_oc(base, p0, r0)
tr = tr.join(p1_oof, on=["s1", "cand"])
m1 = M.lgb.train(M.PARAMS, M.lgb.Dataset(tr.select(M.P1).to_numpy(), tr["y"].to_numpy()), M.ROUNDS)
tr = M.stage2_features(tr)
m2 = M.lgb.train(M.PARAMS, M.lgb.Dataset(tr.select(M.P2).to_numpy(), tr["y"].to_numpy()), M.ROUNDS)
if os.environ.get("SAVE_MODELS"):
    m1.save_model(os.environ["SAVE_MODELS"] + "/stress_stage1.txt"); m2.save_model(os.environ["SAVE_MODELS"] + "/stress_stage2.txt")
del tr
evb = df.filter(pl.col("s1").is_in(list(B))).drop([c for c in M.OC_FEATURES if c in df.columns])
all_s1 = raw["s1"].unique().sort()
for h in (0.0, 0.1, 0.2, 0.3):
    rng = np.random.default_rng(11)
    hidden = set(all_s1.filter(pl.Series(rng.random(len(all_s1)) < h)).to_list())
    p, r = oc_tables(raw.filter(~pl.col("s1").is_in(list(hidden)))) if h else (p0, r0)
    f = add_oc(evb.filter(~pl.col("s1").is_in(list(hidden))), p, r)
    f = f.with_columns(pl.Series("p1", m1.predict(f.select(M.P1).to_numpy()).astype(np.float32)))
    f = M.stage2_features(f)
    f = f.with_columns(pl.Series("score", m2.predict(f.select(M.P2).to_numpy()).astype(np.float32)))
    out = []
    for country in ("India", "US"):
        keep = f.filter(pl.col("country") == country)["s1"].unique().to_list()
        t = truth.filter(pl.col("s1").is_in(keep))
        res = []
        for name, c, dec in (("EXP-2", o2.filter(pl.col("s1").is_in(keep)).rename({"p2": "score"}), cfg2[country]),
                             ("EXP-3", f.filter(pl.col("country") == country).select("s1", "cand", "score"), cfg3[country])):
            pred = M.decide_expected_f(c, dec["miss"])
            tp = pred.join(t, on=["s1", "cand"]).height
            res.append((name, M.f05_macro(pred, t, keep), pred.height - tp, t.height - tp))
        out.append(f"{country}: " + "  ".join(f"{n} {s:.5f} (FP {fp:,} FN {fn:,})" for n, s, fp, fn in res) + f"  delta {res[1][1] - res[0][1]:+.5f}")
    print(f"hide {h:.0%} of train S1 | " + " | ".join(out), flush=True)
