"""Build CE text pairs for one already-cut candidate parquet (no labels needed -- scoring only).
Usage: python prep_extra.py <candidates_cut.parquet> <out.parquet>
"""
import sys
import unicodedata

import polars as pl

from common import DATA, WORK
from prep_data import texts_for  # reuses the same text-building logic as the original prep


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    cand = pl.read_parquet(WORK / in_path).select("s1", "cand")
    out = texts_for("train", cand)
    out.write_parquet(WORK / out_path)
    print(f"wrote {out_path}: {out.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
