"""Shared paths and I/O helpers."""
import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "student_resource" / "dataset"
WORK = Path(os.environ["WORK_DIR"]) if os.environ.get("WORK_DIR") else ROOT / "work"
WORK.mkdir(exist_ok=True)


def read_tsv(path, columns=None):
    """Read a challenge TSV as strings; nulls become empty strings."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       columns=columns).fill_null("")


def read_source(split, i, columns=None):
    return read_tsv(DATA / split / f"{split}_source{i}.tsv", columns)


def read_pairs():
    """Training ground truth exploded to one (s1, match) row per positive pair."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}))
