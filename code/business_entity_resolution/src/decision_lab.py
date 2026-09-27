"""Decision-layer experiments on the saved v6 Kaggle scores (no training).

Uses <kaggle_out>/val_oof.parquet (200k held-out S1, labels) and the same half A / half B split as every earlier
experiment (np.random.default_rng(0) over val_s1 ids, India then US). Choices are made on half A, reported on half B.

  python decision_lab.py <kaggle_out_dir>
Prints half-B F0.5 per country for: v6 as submitted, finer grids, joint ownership (assign every S2/S3 record to its
best claimant BEFORE choosing how many candidates each S1 keeps), each model variant; writes decision_lab.json.
Also prints label-free statistics (matches per S1, share of empty S1) used to check France on test.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from common import WORK
from evaluation import decide, f05_macro
from model_v2 import decide_expected_f

GRID = np.round(np.arange(0.50, 0.901, 0.0125), 4)
MISS = (0.0, 0.025, 0.05, 0.1)


def halves_split():
    val = pl.read_parquet(WORK / "val_s1.parquet")
    rng = np.random.default_rng(0)
    ids = {c: val.filter(pl.col("country") == c)["entity_id"].to_list() for c in ("India", "US")}
    h = {i: int(rng.integers(2)) for c in ids.values() for i in c}
    return ids, h


def joint(c):
    """Keep only pairs where this S1 is the candidate's best claimant (ownership decided first)."""
    return c.sort("score", descending=True).unique("cand", keep="first")


def best_rule(ca, ta, a, own_first):
    x = joint(ca) if own_first else ca
    thr = max(GRID, key=lambda t: f05_macro(decide(x, 8, float(t)), ta, a))
    miss = max(MISS, key=lambda m: f05_macro(decide_expected_f(x, m), ta, a))
    f_thr = f05_macro(decide(x, 8, float(thr)), ta, a)
    f_ef = f05_macro(decide_expected_f(x, miss), ta, a)
    return ("threshold", float(thr)) if f_thr >= f_ef else ("expected_f", miss)


def apply(c, rule, own_first):
    x = joint(c) if own_first else c
    return decide(x, 8, rule[1]) if rule[0] == "threshold" else decide_expected_f(x, rule[1])


def main():
    k = Path(sys.argv[1])
    oof = pl.read_parquet(k / "val_oof.parquet")
    truth = oof.filter(pl.col("y") == 1).select("s1", "cand")
    ids, h = halves_split()
    variants = [v for v in ("s2_xgb", "s2_cat", "s2_ens") if v in oof.columns]
    out = {}
    for country, cids in ids.items():
        a = [i for i in cids if h[i] == 0]; b = [i for i in cids if h[i] == 1]
        ta, tb = truth.filter(pl.col("s1").is_in(a)), truth.filter(pl.col("s1").is_in(b))
        for v in variants:
            c = oof.filter(pl.col("country") == country).with_columns(pl.col(v).alias("score")).select("s1", "cand", "score", "rank")
            ca, cb = c.filter(pl.col("s1").is_in(a)), c.filter(pl.col("s1").is_in(b))
            for own_first in (False, True):
                rule = best_rule(ca, ta, a, own_first)
                pb = apply(cb, rule, own_first)
                f = f05_macro(pb, tb, b)
                key = f"{country}|{v}|{'joint' if own_first else 'post'}"
                out[key] = dict(rule=rule[0], param=rule[1], f_halfB=f, matches_per_s1=pb.height / len(b),
                                empty=1 - pb["s1"].n_unique() / len(b))
                print(f"{key:28} {rule[0]:10} {rule[1]:<6} half-B F0.5 {f:.5f}  matches/S1 {pb.height / len(b):.3f}  "
                      f"empty {out[key]['empty']:.3%}", flush=True)
    (k / "decision_lab.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
