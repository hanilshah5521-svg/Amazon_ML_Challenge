"""France (test only, no labels): address patterns from high-confidence pseudo-pairs.

Pseudo-pairs = S1 x S2/S3 records sharing an identical, rare name_core (<= 3 S1 records carry it).
Reports how their addresses differ (house number agreement, street agreement), the top shared
addresses that look like the same business, and crowded addresses (many businesses, one address).
Writes work/explore/france_patterns.md
"""
import re

import polars as pl
from rapidfuzz import fuzz

from common import WORK, read_source

NUM = re.compile(r"\b\d+\b")
OUT = WORK / "explore" / "france_patterns.md"


def street_key(addr):
    return " ".join(t for t in addr.split() if not t.isdigit() and t not in ("bis", "ter", "no"))


def main():
    frames = [pl.read_parquet(WORK / f"test_s{i}_norm.parquet").filter(pl.col("country") == "France")
                .with_columns(pl.lit(i).alias("src")) for i in (1, 2, 3)]
    fr = pl.concat(frames)
    raw = pl.concat([read_source("test", i).filter(pl.col("country") == "France") for i in (1, 2, 3)])
    raw_map = {r[0]: (r[1], r[2]) for r in raw.rows()}
    s1, oth = fr.filter(pl.col("src") == 1), fr.filter(pl.col("src") != 1)
    lines = [f"# France patterns (test): {s1.height:,} S1, {oth.height:,} S2+S3", ""]

    # ---- pseudo-pairs on rare identical name_core ------------------------------------------
    # Unique in S1 and at most 6 records in S2+S3 (a real match group averages ~3.5 records),
    # otherwise generic French names ("club ecole") create false pseudo-pairs.
    rare1 = s1.group_by("name_core").len().filter(pl.col("len") == 1)["name_core"]
    rare2 = oth.group_by("name_core").len().filter(pl.col("len") <= 6)["name_core"]
    pp = (s1.filter(pl.col("name_core").is_in(rare1) & pl.col("name_core").is_in(rare2)
                    & (pl.col("name_core").str.len_chars() >= 6))
            .join(oth, on="name_core", suffix="_m"))
    stats = {"pairs": 0, "addr_empty": 0, "same_addr": 0, "same_street": 0, "num_both": 0,
             "num_same": 0, "num_missing_one": 0}
    examples = []
    for r in pp.select("entity_id", "entity_id_m", "addr", "addr_m").iter_rows():
        a, b = r[2], r[3]
        stats["pairs"] += 1
        if not b:
            stats["addr_empty"] += 1
            continue
        stats["same_addr"] += a == b
        stats["same_street"] += street_key(a) == street_key(b)
        na, nb = set(NUM.findall(a)), set(NUM.findall(b))
        if na and nb:
            stats["num_both"] += 1
            stats["num_same"] += bool(na & nb)
            if not (na & nb) and len(examples) < 12:
                examples.append((r[0], r[1]))
        else:
            stats["num_missing_one"] += 1
    n = max(stats["pairs"], 1)
    lines += ["## 1. Pseudo-pairs (identical rare name) — how addresses differ", "",
              f"- pseudo-pairs: {stats['pairs']:,}",
              f"- match has empty address: {stats['addr_empty'] / n:.1%}",
              f"- identical normalized address: {stats['same_addr'] / n:.1%}",
              f"- same street (numbers ignored): {stats['same_street'] / n:.1%}",
              f"- both have a number: {stats['num_both']:,}; of those numbers agree: "
              f"{stats['num_same'] / max(stats['num_both'], 1):.1%}",
              f"- one side lacks a number: {stats['num_missing_one'] / n:.1%}", "",
              "Pseudo-pairs whose numbers disagree:", ""]
    for a, b in examples:
        lines.append(f"- `{raw_map[a][0]}` | {raw_map[a][1]}  \n  vs `{raw_map[b][0]}` | {raw_map[b][1]}")

    # ---- addresses shared across sources --------------------------------------------------
    grp = (fr.filter(pl.col("addr") != "").group_by("addr")
             .agg(pl.len().alias("records"), pl.col("src").n_unique().alias("sources"),
                  pl.col("name_key").n_unique().alias("distinct_names"),
                  pl.col("entity_id"), pl.col("name_core"), (pl.col("src") == 1).sum().alias("n_s1")))
    same = (grp.filter((pl.col("sources") == 3) & (pl.col("distinct_names") <= 2) & (pl.col("n_s1") == 1))
               .sort("records", descending=True).head(10))
    lines += ["", "## 2. Top 10 addresses: one S1 record + S2/S3 records, same address, near-identical names",
              "(likely the same business: the easy, high-confidence matches)", ""]
    for r in same.iter_rows(named=True):
        lines.append(f"### `{r['addr']}` — {r['records']} records, {r['distinct_names']} distinct name keys")
        for eid in r["entity_id"]:
            nm, ad = raw_map[eid]
            lines.append(f"- {eid}: **{nm}** | {ad}")
        lines.append("")
    crowded = grp.filter(pl.col("n_s1") >= 2).sort("distinct_names", descending=True).head(10)
    lines += ["## 3. Top 10 crowded addresses: several S1 businesses at one address (false-match traps)", ""]
    for r in crowded.iter_rows(named=True):
        names = sorted(set(raw_map[e][0] for e in r["entity_id"]))
        lines.append(f"- `{r['addr']}`: {r['records']} records, {r['n_s1']} S1, "
                     f"{r['distinct_names']} distinct names — e.g. {'; '.join(names[:6])}")
    lines += ["", "## 4. Crowding overall", "",
              f"- distinct non-empty addresses: {grp.height:,}",
              f"- addresses carrying >= 2 different S1 businesses: {grp.filter(pl.col('n_s1') >= 2).height:,} "
              f"({grp.filter(pl.col('n_s1') >= 2)['n_s1'].sum():,} S1 records)"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"... full report: {OUT}")


if __name__ == "__main__":
    main()
