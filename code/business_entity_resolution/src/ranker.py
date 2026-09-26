"""Learned candidate cut. Blocking keeps the top-40 per S1 by name_sim + addr_sim; a small LightGBM on cheap pair features
re-ranks them and only the best K go to the matcher (candidate_pairs.tsv = exactly these). No text-distance features and no
per-pair Python loops, so it is cheap enough to run on all 40 candidates.

  python ranker.py fit                 # OOF recall@K table (learned cut vs raw score cut), saves <work>/<V2_NAME>/ranker.txt
  python ranker.py cut train <K>       # -> <work>/val_candidates_cut.parquet
  python ranker.py cut test  <K>       # -> <work>/test_candidates_cut_<country>.parquet
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import WORK, read_pairs

V2 = WORK / os.environ.get("V2_NAME", "v2")
V2.mkdir(exist_ok=True)
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=7)
ROUNDS = 300
SUF = os.environ.get("CUT_SUFFIX", "")            # e.g. _k10
F32 = pl.Float32
FEATS = ["name_sim", "addr_sim", "score", "via", "rank", "c_addr_empty", "q_addr_empty", "state_rel", "key_equal", "core_equal",
         "num_shared", "num_n1", "num_n2", "first_num_eq", "name_jac", "addr_jac", "len1", "len2",
         "max_name", "max_addr", "max_score", "gap_name", "gap_addr", "gap_score", "rk_name", "rk_addr", "n_key_equal",
         "n_first_num_eq"]


def tables(split, country):
    cols = ["entity_id", "name_core", "name_key", "addr"]
    read = lambda i: pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet", columns=cols + ["country"]) \
                       .filter(pl.col("country") == country).drop("country")
    return read(1), pl.concat([read(2), read(3)])


def cheap_features(cand, s1n, pool):
    """cand: blocking output rows (s1, cand, name_sim, addr_sim, score, via, rank) of ONE country."""
    ren = lambda d, sfx, idc: d.select(pl.col("entity_id").alias(idc), *[pl.col(c).alias(c + sfx) for c in ("name_core", "name_key", "addr")])
    df = cand.join(ren(s1n, "1", "s1"), on="s1", how="left").join(ren(pool, "2", "cand"), on="cand", how="left")
    st = lambda c: pl.col(c).str.extract(r"\b(st[a-z]{2})\b", 1)
    nums = lambda c: pl.col(c).str.extract_all(r"\d+")
    t1, t2 = pl.col("name_core1").str.split(" "), pl.col("name_core2").str.split(" ")
    a1, a2 = pl.col("addr1").str.split(" "), pl.col("addr2").str.split(" ")
    df = df.with_columns(
        (pl.col("addr2") == "").cast(F32).alias("c_addr_empty"), (pl.col("addr1") == "").cast(F32).alias("q_addr_empty"),
        pl.when(st("addr1").is_null() | st("addr2").is_null()).then(0.0).when(st("addr1") == st("addr2")).then(1.0)
          .otherwise(-1.0).cast(F32).alias("state_rel"),
        (pl.col("name_key1") == pl.col("name_key2")).cast(F32).alias("key_equal"),
        (pl.col("name_core1") == pl.col("name_core2")).cast(F32).alias("core_equal"),
        nums("addr1").list.set_intersection(nums("addr2")).list.len().cast(F32).alias("num_shared"),
        nums("addr1").list.len().cast(F32).alias("num_n1"), nums("addr2").list.len().cast(F32).alias("num_n2"),
        (nums("addr1").list.first() == nums("addr2").list.first()).fill_null(False).cast(F32).alias("first_num_eq"),
        (t1.list.set_intersection(t2).list.len() / t1.list.set_union(t2).list.len()).cast(F32).alias("name_jac"),
        (a1.list.set_intersection(a2).list.len() / a1.list.set_union(a2).list.len()).fill_null(0.0).cast(F32).alias("addr_jac"),
        pl.col("name_core1").str.len_chars().cast(F32).alias("len1"), pl.col("name_core2").str.len_chars().cast(F32).alias("len2"),
        pl.col("name_sim").max().over("s1").alias("max_name"), pl.col("addr_sim").max().over("s1").alias("max_addr"),
        pl.col("score").max().over("s1").alias("max_score"),
        pl.col("name_sim").rank("ordinal", descending=True).over("s1").cast(F32).alias("rk_name"),
        pl.col("addr_sim").rank("ordinal", descending=True).over("s1").cast(F32).alias("rk_addr"))
    df = df.with_columns(
        (pl.col("name_sim") - pl.col("max_name")).alias("gap_name"), (pl.col("addr_sim") - pl.col("max_addr")).alias("gap_addr"),
        (pl.col("score") - pl.col("max_score")).alias("gap_score"), pl.col("rank").cast(F32),
        pl.col("key_equal").sum().over("s1").alias("n_key_equal"), pl.col("first_num_eq").sum().over("s1").alias("n_first_num_eq"))
    return df.select("s1", "cand", *FEATS)


def features_all(cand, split, chunk=60_000):
    """Cheap features per country and S1 chunk (bounded memory)."""
    out = []
    for country in cand["country"].unique().sort():
        c = cand.filter(pl.col("country") == country)
        s1n, pool = tables(split, country)
        ids = c["s1"].unique().sort()
        for s in range(0, len(ids), chunk):
            part = c.filter(pl.col("s1").is_in(ids[s:s + chunk]))
            out.append(cheap_features(part, s1n, pool).with_columns(pl.lit(country).alias("country")))
            print(f"   {country}: cheap features for {min(s + chunk, len(ids)):,}/{len(ids):,} S1", flush=True)
        del s1n, pool
    return pl.concat(out)


def recall_table(df, truth_n, label, ks=(4, 5, 6, 8, 10, 12, 15, 20, 30, 40)):
    """recall of TRUE matches among the top-K of the learned score and of the raw score, per country."""
    for country in df["country"].unique().sort():
        d = df.filter(pl.col("country") == country)
        n = truth_n[country]
        lr = d.with_columns(pl.col("p_rank").rank("ordinal", descending=True).over("s1").alias("lrank"))
        rows = []
        for k in ks:
            learned = lr.filter(pl.col("lrank") <= k)["y"].sum() / n
            raw = d.filter(pl.col("rank") <= k)["y"].sum() / n
            rows.append(f"K={k}: raw {raw:.4f} learned {learned:.4f}")
        print(f"{label} {country}: " + " | ".join(rows), flush=True)


def fit():
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"}).filter(pl.col("s1").is_in(val["entity_id"]))
    cand = pl.read_parquet(WORK / "val_candidates.parquet")
    df = features_all(cand, "train").join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "cand"], how="left") \
                                   .with_columns(pl.col("y").fill_null(0))
    ctry = val.select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c2"))
    n_true = {c: truth.join(ctry.filter(pl.col("c2") == c), on="s1").height for c in sorted(val["country"].unique().to_list())}
    X, y, g = df.select(FEATS).to_numpy(), df["y"].to_numpy(), df["s1"].to_numpy()
    oofp = np.zeros(len(df), dtype=np.float32)
    for tr, te in GroupKFold(5).split(X, y, g):
        oofp[te] = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS).predict(X[te])
    df = df.with_columns(pl.Series("p_rank", oofp))
    recall_table(df, n_true, "OOF")
    lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS).save_model(str(V2 / "ranker.txt"))
    print("saved ranker", flush=True)


def cut(split, k):
    m = lgb.Booster(model_file=str(V2 / "ranker.txt"))

    def one(cand):
        df = features_all(cand, split)
        p = m.predict(df.select(FEATS).to_numpy()).astype(np.float32)
        df = df.with_columns(pl.Series("p_rank", p)).with_columns(
            pl.col("p_rank").rank("ordinal", descending=True).over("s1").alias("new_rank")).filter(pl.col("new_rank") <= k)
        return cand.join(df.select("s1", "cand", "new_rank"), on=["s1", "cand"]).with_columns(
            pl.col("new_rank").alias("rank")).drop("new_rank")

    if split == "train":
        one(pl.read_parquet(WORK / "val_candidates.parquet")).write_parquet(WORK / f"val_candidates_cut{SUF}.parquet")
        if (WORK / "extra_candidates.parquet").exists():
            one(pl.read_parquet(WORK / "extra_candidates.parquet")).write_parquet(WORK / f"extra_candidates_cut{SUF}.parquet")
    else:
        for country in sorted(p.stem[len("test_candidates_"):] for p in WORK.glob("test_candidates_*.parquet")
                              if "cut" not in p.stem and p.stem != "test_candidates"):
            c = pl.read_parquet(WORK / f"test_candidates_{country}.parquet").with_columns(pl.lit(country).alias("country"))
            one(c).write_parquet(WORK / f"test_candidates_cut{SUF}_{country}.parquet")
            print("cut", country, flush=True)


if __name__ == "__main__":
    fit() if sys.argv[1] == "fit" else cut(sys.argv[2], int(sys.argv[3]))
