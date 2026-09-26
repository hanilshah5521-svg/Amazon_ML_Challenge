"""Split the shared union blocking output into the two experiment candidate sets.

  exp0 = exactly the Harsh V top-40 (pairs from its own retrievers, rank = rank_hv)
  exp1 = Harsh V top-40 + every 'patel' S2/S3->S1 candidate (rank = rank in the union, our_rank / our_rr kept)

Usage: python derive_candidates.py <shared_work_dir> <exp0_dir> <exp1_dir>
"""
import sys
from pathlib import Path

import polars as pl

LINK = ["indic_dict.json", "train_s1_norm.parquet", "train_s2_norm.parquet", "train_s3_norm.parquet",
        "test_s1_norm.parquet", "test_s2_norm.parquet", "test_s3_norm.parquet", "val_s1.parquet", "extra_s1.parquet"]


def main():
    shared, e0, e1 = (Path(p).resolve() for p in sys.argv[1:4])
    files = ["val_candidates.parquet", "extra_candidates.parquet"] + sorted(
        p.name for p in shared.glob("test_candidates_*.parquet") if "cut" not in p.stem)
    for d in (e0, e1):
        d.mkdir(parents=True, exist_ok=True)
        for f in LINK:
            if not (d / f).exists():
                (d / f).symlink_to(shared / f)
    for f in files:
        c = pl.read_parquet(shared / f)
        x0 = (c.filter(((pl.col("via") & 3) > 0) & (pl.col("rank_hv") <= 40))
                .with_columns(pl.col("rank_hv").alias("rank"), (pl.col("via") & 3).alias("via"))
                .drop("rank_hv", "our_rank", "our_rr"))
        x1 = c.drop("rank_hv")
        x0.write_parquet(e0 / f)
        x1.write_parquet(e1 / f)
        n = c["s1"].n_unique()
        print(f"{f}: exp0 {x0.height / n:.1f} pairs per S1 | exp1 {x1.height / n:.1f} pairs per S1 "
              f"(patel-only pairs {c.filter(pl.col('rank_hv').is_null()).height:,})", flush=True)


if __name__ == "__main__":
    main()
