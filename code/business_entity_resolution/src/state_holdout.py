"""Whole-state holdout: hold out 20% of the states of each country; train a stage-1 model on training S1 of the OTHER states,
score the held-out states' evaluation S1 (val_s1), expected-F decision (miss 0), macro F0.5. Run once per feature set with that
experiment's flags / WORK_DIR; the split is the same for every run (seeded by state name).
Usage: WORK_DIR=... [PATTERN_FEATS=1] [OWNER_FEATS=1] [X5_FEATS=1] MODEL_K=8 python state_holdout.py <label>"""
import hashlib, os, sys
import numpy as np
import polars as pl
import model_v2 as M
from blocking import _state_of
from common import read_pairs

W = M.WORK
df = pl.read_parquet(W / "v2" / "train_features.parquet")
if M.OWNER:
    df = M.add_oc(df, "train")
if M.X5:
    df = M.add_x5(df, "train")
s1 = pl.read_parquet(W / "train_s1_norm.parquet", columns=["entity_id", "addr", "country"])
st = pl.concat([s1.filter(pl.col("country") == c).with_columns(pl.Series("state", _state_of(s1.filter(pl.col("country") == c)["addr"].to_list(), c)))
                for c in ("India", "US")]).select(pl.col("entity_id").alias("s1"), "state", "country")
held_state = lambda s: s != "" and int(hashlib.md5(s.encode()).hexdigest(), 16) % 5 == 0
st = st.with_columns(pl.col("state").map_elements(held_state, return_dtype=pl.Boolean).alias("held_state"))
val = set(pl.read_parquet(W / "val_s1.parquet")["entity_id"].to_list())
df = df.join(st.select("s1", "held_state", "state"), on="s1", how="left").with_columns(pl.col("held_state").fill_null(False))
tr = df.filter(~pl.col("held_state"))
ev = df.filter(pl.col("held_state") & pl.col("s1").is_in(list(val)))
m = M.lgb.train(M.PARAMS, M.lgb.Dataset(tr.select(M.P1).to_numpy(), tr["y"].to_numpy()), M.ROUNDS)
ev = ev.with_columns(pl.Series("score", m.predict(ev.select(M.P1).to_numpy()).astype(np.float32)))
truth = read_pairs().rename({"m": "cand"})
res = []
for c in ("India", "US"):
    e = ev.filter(pl.col("country") == c)
    ids = e["s1"].unique().to_list()
    pred, t = M.decide_expected_f(e.select("s1", "cand", "score"), 0.0), truth.filter(pl.col("s1").is_in(ids))
    tp = pred.join(t, on=["s1", "cand"]).height
    res.append(f"{c} states {e['state'].n_unique()} S1 {len(ids):,}: F0.5 {M.f05_macro(pred, t, ids):.5f} FP {pred.height - tp:,} FN {t.height - tp:,}")
print(f"{sys.argv[1]:8s} (train {tr['s1'].n_unique():,} S1, {len(M.P1)} features) | " + " | ".join(res), flush=True)
