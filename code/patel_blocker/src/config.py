"""Shared defaults + CLI plumbing. Every path is overridable from the command line."""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parents[1] / "student_resource" / "dataset"  # repo_root/student_resource/dataset
DEFAULT_WORK = ROOT.parents[1] / "work" / "patel_blocker"   # repo_root/work/patel_blocker (git-ignored)
SEED = 42

# Blocking
TOP_K = 3              # S1 candidates kept per confident S2/S3 record
REL_MIN = 0.8          # keep candidate only if rerank score >= REL_MIN * best rerank score of that record
DEEP_BELOW = 52.0      # best IDF score below this = low-confidence record (~lowest third in train)
DEEP_K = 200           # IDF pool size re-ranked by string similarity for low-confidence records (0 = legacy)
TOP_K_DEEP = 8         # S1 candidates kept per low-confidence record after re-ranking
NUMW = 0               # 1 = add (house number x address word) blocking keys
MAX_DF_FRAC = 0.0005   # tokens in more than this share of a country's S1 records are not indexed
MIN_MAX_DF = 50        # ...but never cap below this many records (keeps small samples sensible)
STOP_FRAC = 0.002      # name tokens above this S1 share are dropped from the "core" compact key
QUERY_CHUNK = 50_000   # S2/S3 records per blocking join chunk (RAM knob: lower if you run out of memory)
FEATURE_CHUNK = 2_000_000  # candidate pairs per feature chunk (RAM knob)

# Training
N_FOLDS = 2
MAX_TRAIN_ROWS = 6_000_000
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=0, seed=SEED, deterministic=True, force_col_wise=True, verbose=-1)
NUM_ROUNDS = 600
EARLY_STOP = 50


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="folder with train/ and test/")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="cache/artefact folder")
    return p


def split_dir(work: Path, split: str) -> Path:
    d = work / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup(args) -> logging.Logger:
    random.seed(SEED)
    np.random.seed(SEED)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.work_dir / "run.log")])
    log = logging.getLogger("ber")
    log.info("args: %s", {k: str(v) for k, v in vars(args).items()})
    return log


class Timer:
    def __init__(self, log, what):
        self.log, self.what = log, what

    def __enter__(self):
        self.t = time.time()
        self.log.info("%s ...", self.what)

    def __exit__(self, *exc):
        self.log.info("%s done in %.1fs", self.what, time.time() - self.t)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))
