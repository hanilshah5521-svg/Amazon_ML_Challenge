"""EXP-3: owner-competition features from the 'patel' S2/S3->S1 blocker, which ranks candidate owners for EVERY S2/S3 record
against ALL S1 of the split (train 2.2M S1, test 1.73M), so the features mean the same thing on train and test.
(The matcher's own scores exist only for the 400k sampled training S1, so competition built on them would be 5x sparser in train.)

Per (s1, cand): oc_in (this S1 is in cand's owner list), oc_rank, oc_rr (this S1's re-rank score), oc_n (list length),
oc_rr_other (best score of any OTHER S1 for cand), oc_gap = oc_rr - oc_rr_other.
Output: <WORK>/oc_{split}_pairs.parquet (s1, cand, feats) and oc_{split}_records.parquet (cand, oc_n, oc_rr1).
Usage: OUR_CANDS_DIR=... python owner_competition.py
"""
import os
from functools import lru_cache
from pathlib import Path

import polars as pl

from common import WORK

OC_FEATURES = ["oc_in", "oc_rank", "oc_rr", "oc_n", "oc_rr_other", "oc_gap"]


def build(split):
    d = Path(os.environ["OUR_CANDS_DIR"]) / split
    ids = pl.read_parquet(d / "records.parquet", columns=["rid", "entity_id"])
    c = (pl.read_parquet(d / "cands.parquet", columns=["q", "s1", "rank", "rr"])
           .join(ids.rename({"rid": "s1", "entity_id": "s1_eid"}), on="s1")
           .join(ids.rename({"rid": "q", "entity_id": "cand"}), on="q")
           .select(pl.col("s1_eid").alias("s1"), "cand", pl.col("rank").cast(pl.Float32).alias("oc_rank"), pl.col("rr").cast(pl.Float32).alias("oc_rr")))
    rec = c.group_by("cand").agg(pl.len().cast(pl.Float32).alias("oc_n"), pl.col("oc_rr").max().alias("oc_rr1"),
                                 pl.col("oc_rr").top_k(2).min().alias("_rr2"), pl.len().alias("_len"))
    rec = rec.with_columns(pl.when(pl.col("_len") > 1).then(pl.col("_rr2")).otherwise(0.0).alias("oc_rr2")).drop("_rr2", "_len")
    pairs = c.join(rec, on="cand").with_columns(
        pl.when(pl.col("oc_rr") >= pl.col("oc_rr1")).then(pl.col("oc_rr2")).otherwise(pl.col("oc_rr1")).alias("oc_rr_other"))
    pairs.select("s1", "cand", "oc_rank", "oc_rr", "oc_rr_other").write_parquet(WORK / f"oc_{split}_pairs.parquet")
    rec.select("cand", "oc_n", "oc_rr1").write_parquet(WORK / f"oc_{split}_records.parquet")
    print(f"{split}: {pairs.height:,} owner-list pairs over {rec.height:,} records", flush=True)


@lru_cache(maxsize=2)
def _tables(split):                     # read once per split; test predict calls add_oc once per S1 chunk
    return pl.read_parquet(WORK / f"oc_{split}_pairs.parquet"), pl.read_parquet(WORK / f"oc_{split}_records.parquet")


def add_oc(df, split):
    """Join the owner-competition features onto candidate pairs (s1, cand). Pairs outside the owner list get oc_in 0,
    null rank/score/gap, and the record's best S1 score as oc_rr_other."""
    p, r = _tables(split)
    p = p.filter(pl.col("s1").is_in(df["s1"].unique().implode()))
    r = r.filter(pl.col("cand").is_in(df["cand"].unique().implode()))
    df = df.drop([c for c in OC_FEATURES if c in df.columns]).join(p, on=["s1", "cand"], how="left").join(r, on="cand", how="left")
    return df.with_columns(pl.col("oc_rr").is_not_null().cast(pl.Float32).alias("oc_in"),
                           pl.col("oc_rr_other").fill_null(pl.col("oc_rr1")).fill_null(0.0),
                           pl.col("oc_n").fill_null(0.0)).with_columns(
                           (pl.col("oc_rr") - pl.col("oc_rr_other")).alias("oc_gap")).drop("oc_rr1")


if __name__ == "__main__":
    for s in ("train", "test"):
        build(s)
    t = pl.DataFrame({"s1": ["a", "a", "b"], "cand": ["x", "y", "x"]})
    print("add_oc columns:", add_oc(t, "test").columns)
