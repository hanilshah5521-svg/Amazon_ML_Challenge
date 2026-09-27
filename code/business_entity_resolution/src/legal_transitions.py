"""Legal-type transitions seen in TRUE training pairs, per country (from the 0.976-Hanil branch).

Leak fix vs the original: pairs of the held-out S1 (val_s1.parquet, the records F0.5 is reported on) are excluded, so a rare
true transition of a held-out pair cannot make its own `legal_trans_unseen` = 0.
Output: <work>/<V2_NAME>/legal_transitions.json = {country: {"<s1 types> > <cand types>": count}}
Usage: python legal_transitions.py
"""
import json
import os

import polars as pl

from common import WORK, read_pairs
from features_v2_pattern import ENT

OUT = WORK / os.environ.get("V2_NAME", "v2") / "legal_transitions.json"


def ent_key(col):
    keys, vals = list(ENT), list(ENT.values())
    return (pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().is_in(keys))
            .replace_strict(keys, vals)).list.unique().list.sort().list.join("|"))


def main():
    held_out = pl.read_parquet(WORK / "val_s1.parquet")["entity_id"]
    if os.environ.get("EXP5") == "1":     # also drop the 200k extra training S1: no row of the 3.2M training pairs counts its own label
        held_out = pl.concat([held_out, pl.read_parquet(WORK / "extra_s1.parquet")["entity_id"]])
    pairs = read_pairs().rename({"m": "cand"}).filter(~pl.col("s1").is_in(held_out.implode()))
    s1 = pl.scan_parquet(WORK / "train_s1_norm.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("n1"), "country")
    oth = pl.concat([pl.scan_parquet(WORK / f"train_s{i}_norm.parquet").select(pl.col("entity_id").alias("cand"), pl.col("name").alias("n2"))
                     for i in (2, 3)])
    t = (pairs.lazy().join(s1, on="s1").join(oth, on="cand")
              .select("country", ent_key("n1").alias("e1"), ent_key("n2").alias("e2"))
              .group_by("country", "e1", "e2").len().collect())
    table = {c: {f"{r['e1']}>{r['e2']}": r["len"] for r in t.filter(pl.col("country") == c).iter_rows(named=True)}
             for c in t["country"].unique().to_list()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(table, indent=0), encoding="utf-8")
    print(f"pairs used: {pairs.height:,} (held-out S1 excluded: {held_out.len():,})", flush=True)
    for c, d in table.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:8]
        print(c, f"{len(d)} transitions;", top, flush=True)


if __name__ == "__main__":
    main()
