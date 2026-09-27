"""France: infer the missing region of S2/S3 addresses from the city (label-free, test data only).

31.9% of French S2/S3 records have a full address but no region token. In training (US/India) a missing state
only ever co-occurs with an EMPTY address, so the model treats these French pairs as out-of-distribution
(33% of accepted French pairs vs 2-3% in India/US). France has ~20 city/locality tokens that each belong to one
region; the map is learned from French S1 records (they always carry city + region): tokens with >=99% purity and
>=500 occurrences. Checked on records that do carry a region: 99.95% agreement.

  python fr_region_fill.py     # rewrites the France rows of <work>/test_s2_norm.parquet and test_s3_norm.parquet
Files are written to a temp path and swapped in with os.replace, so hard-linked copies elsewhere stay unchanged.
The original files are kept as *_norm.pre_frfill.parquet.
"""
import json
import os

import polars as pl

from common import WORK
from normalize import STATE_COMPONENT

REGIONS = set(STATE_COMPONENT["France"].values())


def learn_map():
    s1 = pl.read_parquet(WORK / "test_s1_norm.parquet", columns=["addr", "country"]).filter(pl.col("country") == "France")
    t = (s1.with_columns(pl.col("addr").str.split(" ").alias("tok"))
           .with_columns(pl.col("tok").list.eval(pl.element().filter(pl.element().is_in(list(REGIONS)))).list.first().alias("reg"))
           .filter(pl.col("reg").is_not_null()).explode("tok")
           .filter(~pl.col("tok").is_in(list(REGIONS)) & ~pl.col("tok").str.contains(r"\d") & (pl.col("tok").str.len_chars() >= 3))
           .group_by("tok", "reg").len())
    tot = t.group_by("tok").agg(pl.col("len").sum().alias("n"))
    city = (t.join(tot, on="tok").with_columns((pl.col("len") / pl.col("n")).alias("purity"))
              .filter((pl.col("purity") >= 0.99) & (pl.col("n") >= 500)))
    return dict(zip(city["tok"].to_list(), city["reg"].to_list()))


def fill(addr, m):
    toks = addr.split()
    if not toks or any(t in REGIONS for t in toks):
        return addr
    reg = next((m[t] for t in toks if t in m), None)
    return f"{addr} {reg}" if reg else addr


def main():
    m = learn_map()
    (WORK / "fr_city_region.json").write_text(json.dumps(m, indent=0))
    print(f"learned {len(m)} city->region tokens", flush=True)
    for i in (2, 3):
        path = WORK / f"test_s{i}_norm.parquet"
        df = pl.read_parquet(path)
        fr = df["country"] == "France"
        new = [fill(a, m) if f else a for a, f in zip(df["addr"].to_list(), fr.to_list())]
        changed = sum(a != b for a, b in zip(new, df["addr"].to_list()))
        out = df.with_columns(pl.Series("addr", new))
        backup = WORK / f"test_s{i}_norm.pre_frfill.parquet"
        if not backup.exists():
            os.link(path, backup)                      # keep the original (same inode as before)
        tmp = WORK / f"test_s{i}_norm.tmp.parquet"
        out.write_parquet(tmp)
        os.replace(tmp, path)                          # new inode: hard links elsewhere keep the old data
        print(f"test_s{i}: region filled for {changed:,} French records", flush=True)


if __name__ == "__main__":
    main()
