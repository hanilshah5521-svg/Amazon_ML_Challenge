"""All remaining training S1 (everything not in the 200k held-out set): candidate search + learned
cut to K, in chunks, so only K rows per S1 are ever stored (not the top-40).

  python extra_all.py [K=8] [chunk=200000]
  -> <work>/extra_all/<country>_<i>.parquet   (resumable: finished chunks are skipped)
  -> <work>/extra_candidates_cut.parquet      (all chunks concatenated at the end)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import block_country, load
from common import WORK
from ranker import FEATS, V2, features_all

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
OUT = WORK / "extra_all"
OUT.mkdir(exist_ok=True)


def main():
    m = lgb.Booster(model_file=str(V2 / "ranker.txt"))
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet")
    val_ids = pl.read_parquet(WORK / "val_s1.parquet")["entity_id"]
    rest = s1_all.filter(~pl.col("entity_id").is_in(val_ids)).sort("entity_id")
    for country in rest["country"].unique().sort():
        q_all = rest.filter(pl.col("country") == country)
        pool = pl.concat([load("train", 2, country), load("train", 3, country)])
        for i, s in enumerate(range(0, q_all.height, CHUNK)):
            path = OUT / f"{country}_{i:02d}.parquet"
            if path.exists():
                continue
            t0 = time.time()
            cand = block_country(q_all.slice(s, CHUNK), pool, k_final=40).with_columns(pl.lit(country).alias("country"))
            df = features_all(cand, "train")
            p = m.predict(df.select(FEATS).to_numpy()).astype(np.float32)
            keep = (df.with_columns(pl.Series("p_rank", p))
                      .with_columns(pl.col("p_rank").rank("ordinal", descending=True).over("s1").alias("new_rank"))
                      .filter(pl.col("new_rank") <= K).select("s1", "cand", "new_rank"))
            cand.join(keep, on=["s1", "cand"]).with_columns(pl.col("new_rank").alias("rank")).drop("new_rank").write_parquet(path)
            print(f"{country} chunk {i}: {min(s + CHUNK, q_all.height):,}/{q_all.height:,} S1 ({time.time() - t0:.0f}s)", flush=True)
            del cand, df, keep
        del pool
    pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))]).write_parquet(WORK / "extra_candidates_cut.parquet")
    ex = rest.select("entity_id", "country")
    ex.write_parquet(WORK / "extra_s1.parquet")
    print(f"extra_candidates_cut.parquet written for {ex.height:,} S1", flush=True)


if __name__ == "__main__":
    main()
