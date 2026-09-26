"""Discover per-country decoy / noise words without labels.

Same name_key, same street, house numbers on both sides, and the S2/S3 name adds exactly one word: if the house
numbers then disagree the word marks a different business ("decoy"), if they agree it is noise (measured on US
train: partners/holdings/group are decoys, center/service/services noise). India/US lists come from train
candidates, the France list from the unlabelled test candidates.

Usage: python word_lists.py            # writes <work>/<V2_NAME>/word_lists.json
"""
import json
import os
from collections import defaultdict

import polars as pl

from common import WORK
from features_v2 import WORD_LIST_PATH

MODEL_K = 15
COLS = ("name_core", "name_key", "addr")


def discover_words(cand, s1n, pool, min_n=30, lo=0.3, hi=0.7):
    """cand: (s1, cand) pairs of ONE country; s1n, pool: normalized records of that country."""
    txt = lambda d, sfx, idcol: d.select(pl.col("entity_id").alias(idcol), *[pl.col(c).alias(c + sfx) for c in COLS])
    df = cand.select("s1", "cand").join(txt(s1n, "1", "s1"), on="s1").join(txt(pool, "2", "cand"), on="cand")
    stats = defaultdict(lambda: [0, 0])
    for c1, k1, a1, c2, k2, a2 in df.select("name_core1", "name_key1", "addr1", "name_core2", "name_key2", "addr2").iter_rows():
        if k1 != k2 or not a2:
            continue
        t1, t2 = set(c1.split()), set(c2.split())
        added, dropped = t2 - t1, t1 - t2
        if len(added) != 1 or dropped:
            continue
        w1 = {t for t in a1.split() if not t.isdigit()}
        w2 = {t for t in a2.split() if not t.isdigit()}
        n1 = {t for t in a1.split() if t.isdigit()}
        n2 = {t for t in a2.split() if t.isdigit()}
        if not n1 or not n2 or not (w1 <= w2 or w2 <= w1):
            continue                                  # same street, both have numbers
        s = stats[next(iter(added))]
        s[0] += 1
        s[1] += (n1 <= n2 or n2 <= n1)
    decoy = sorted(w for w, (n, a) in stats.items() if n >= min_n and a / n < lo)
    noise = sorted(w for w, (n, a) in stats.items() if n >= min_n and a / n > hi)
    return decoy, noise, {w: v for w, v in stats.items() if v[0] >= min_n}


def tables(split, country):
    read = lambda i: pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet").filter(pl.col("country") == country)
    return read(1), pl.concat([read(2), read(3)])


def main():
    out = {}
    train_c = sorted(pl.read_parquet(WORK / "val_s1.parquet")["country"].unique().to_list())
    jobs = [("train", c, WORK / "val_candidates.parquet") for c in train_c]
    for p in sorted(WORK.glob("test_candidates_*.parquet")):             # countries without labels: lists from unlabelled test candidates
        c = p.stem[len("test_candidates_"):]
        if "cut" not in p.stem and c not in train_c and p.stem != "test_candidates":
            jobs.append(("test", c, p))
    for split, country, path in jobs:
        cand = pl.read_parquet(path)
        if "country" in cand.columns:
            cand = cand.filter(pl.col("country") == country)
        cand = cand.filter(pl.col("rank") <= MODEL_K)
        s1n, pool = tables(split, country)
        decoy, noise, stats = discover_words(cand, s1n, pool)
        out[country] = {"decoy": decoy, "noise": noise, "counts": stats}
        print(f"{country}: decoy {decoy}\n{' ' * len(country)}  noise {noise}", flush=True)
    WORD_LIST_PATH.parent.mkdir(exist_ok=True)
    WORD_LIST_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", WORD_LIST_PATH)


if __name__ == "__main__":
    main()
