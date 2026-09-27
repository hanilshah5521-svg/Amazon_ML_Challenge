"""Memory-safe version of `ranker.py cut test K`: same learned cut, applied in S1 chunks so the
cheap features of a whole country (32M rows for India) are never held at once.
  python cut_test_chunked.py [K=8] [chunk_s1=100000]   (countries already cut are skipped)
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import WORK
from ranker import FEATS, V2, features_all

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000


def main():
    m = lgb.Booster(model_file=str(V2 / "ranker.txt"))
    for country in ("France", "India", "US"):
        out = WORK / f"test_candidates_cut_{country}.parquet"
        if out.exists():
            print("already cut", country, flush=True)
            continue
        cand = pl.read_parquet(WORK / f"test_candidates_{country}.parquet").with_columns(pl.lit(country).alias("country"))
        ids = cand["s1"].unique().sort()
        parts = []
        for s in range(0, len(ids), CHUNK):
            c = cand.filter(pl.col("s1").is_in(ids[s:s + CHUNK]))
            df = features_all(c, "test")
            p = m.predict(df.select(FEATS).to_numpy()).astype(np.float32)
            keep = (df.with_columns(pl.Series("p_rank", p))
                      .with_columns(pl.col("p_rank").rank("ordinal", descending=True).over("s1").alias("new_rank"))
                      .filter(pl.col("new_rank") <= K).select("s1", "cand", "new_rank"))
            parts.append(c.join(keep, on=["s1", "cand"]).with_columns(pl.col("new_rank").alias("rank")).drop("new_rank"))
            print(f"{country}: cut {min(s + CHUNK, len(ids)):,}/{len(ids):,} S1", flush=True)
            del c, df, keep
        pl.concat(parts).write_parquet(out)
        print("cut", country, flush=True)
        del cand, parts


if __name__ == "__main__":
    main()
