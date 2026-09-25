"""Mine true pairs for systematic normalization gaps.

For each true pair, compares normalized tokens on both sides and aggregates:
  * 1-to-1 substitutions that are not simple typos (word A in S1 <-> word B in match)
  * tokens that are dropped / added far more often than chance
  * examples of the lowest-similarity true pairs
Usage: python edge_miner.py [n_s1=60000]
"""
import sys
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

from common import DATA, read_source, read_tsv
from normalize import normalize_address, normalize_name


def sample_pairs(n):
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv").sample(n, seed=11)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "")
               .rename({"source1_entity_id": "a", "matched_entity_ids": "b"}))
    s1 = read_source("train", 1).rename({"entity_id": "a"})
    other = pl.concat([read_source("train", 2), read_source("train", 3)]).rename({"entity_id": "b"})
    other = other.filter(pl.col("b").is_in(pairs["b"]))
    return pairs.join(s1, on="a").join(other, on="b", suffix="_m")


def mine(rows, field):
    subs, typo, dropped, added, seen1, seen2 = (Counter() for _ in range(6))
    worst = []
    for x, y, raw_x, raw_y, country in rows:
        t1, t2 = set(x.split()), set(y.split())
        seen1.update(t1); seen2.update(t2)
        a, b = t1 - t2, t2 - t1
        dropped.update(a); added.update(b)
        if len(a) == 1 and len(b) == 1:
            (p,), (q,) = a, b
            (typo if fuzz.ratio(p, q) >= 75 else subs)[(p, q)] += 1
        if t1 and t2:
            j = len(t1 & t2) / len(t1 | t2)
            if j < 0.2:
                worst.append((j, country, raw_x, raw_y))
    print(f"\n######## {field.upper()} ########")
    print("-- top non-typo 1:1 substitutions (S1 word -> match word):")
    for (p, q), c in subs.most_common(40):
        print(f"   {c:6}  {p!r:22} -> {q!r}")
    print("-- tokens most often missing from the match (drop rate = dropped / occurrences in S1):")
    rates = [(dropped[t] / seen1[t], t, seen1[t]) for t in seen1 if seen1[t] >= 300]
    for r, t, n in sorted(rates, reverse=True)[:30]:
        print(f"   {t!r:18} dropped {r:5.1%} of {n}")
    print("-- tokens most often added in the match (not in S1):")
    for t, c in added.most_common(40):
        if seen2[t] >= 300 and c / seen2[t] > 0.5:
            print(f"   {t!r:18} added {c} ({c / seen2[t]:.0%} of its occurrences)")
    print(f"-- {len(worst):,} pairs with token Jaccard < 0.2; sample:")
    for j, country, rx, ry in sorted(worst)[:: max(1, len(worst) // 25)][:25]:
        print(f"   [{country}] {rx[:60]!r}  <->  {ry[:60]!r}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    p = sample_pairs(n)
    names, addrs = [], []
    for r in p.select("business_name", "business_name_m", "business_address",
                      "business_address_m", "country").iter_rows():
        n1, n2, a1, a2, c = r
        names.append((normalize_name(n1, c)[1], normalize_name(n2, c)[1], n1, n2, c))
        if a1 and a2:
            addrs.append((normalize_address(a1, c), normalize_address(a2, c), a1, a2, c))
    print(f"{len(names):,} true pairs ({len(addrs):,} with both addresses)")
    mine(names, "name_core")
    mine(addrs, "address")


if __name__ == "__main__":
    main()
