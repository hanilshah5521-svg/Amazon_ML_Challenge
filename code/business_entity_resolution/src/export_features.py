"""Build features in S1 chunks and write each chunk straight to disk (for training on Kaggle).

  python export_features.py train   # held-out 200k (val_candidates_cut) + all other training S1
                                     #   (extra_candidates_cut) -> <work>/feat_train/part_*.parquet
                                     #   columns: features + s1, cand, country, rank, y, is_val
  python export_features.py test    # test_candidates_cut_<country> -> <work>/feat_test/part_*.parquet
Parts are written as each chunk finishes (a crash loses only the chunk in progress; rerun overwrites).
"""
import sys

import polars as pl

from common import WORK, read_pairs
from model_v2 import MODEL_K, features_chunked


def export(cand, split, out_dir, truth=None, val_ids=None):
    out_dir.mkdir(exist_ok=True)
    counter = {"i": 0}

    def write(f):
        path = out_dir / f"part_{counter['i']:04d}.parquet"
        counter["i"] += 1
        if truth is not None:
            f = (f.join(truth, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0).cast(pl.Int8),
                 pl.col("s1").is_in(val_ids).cast(pl.Int8).alias("is_val")))
        f.write_parquet(path, compression="zstd")
        print(f"   wrote {path.name}: {f.height:,} rows", flush=True)
        return f.head(0)

    features_chunked(cand, split, on_chunk=write)


def main():
    if sys.argv[1] == "train":
        val = pl.read_parquet(WORK / "val_s1.parquet")
        ex = pl.read_parquet(WORK / "extra_s1.parquet")
        s1c = pl.concat([val, ex]).select(pl.col("entity_id").alias("s1"), "country")
        cand = (pl.concat([pl.read_parquet(WORK / "val_candidates_cut.parquet"), pl.read_parquet(WORK / "extra_candidates_cut.parquet")],
                          how="diagonal_relaxed")
                  .filter(pl.col("rank") <= MODEL_K).drop("country", strict=False).join(s1c, on="s1"))
        truth = read_pairs().rename({"m": "cand"}).with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
        export(cand, "train", WORK / "feat_train", truth, val["entity_id"])
    else:
        for country in ("France", "India", "US"):
            cand = (pl.read_parquet(WORK / f"test_candidates_cut_{country}.parquet").filter(pl.col("rank") <= MODEL_K)
                      .with_columns(pl.lit(country).alias("country")))
            export(cand, "test", WORK / f"feat_test_{country}")
            del cand


if __name__ == "__main__":
    main()
