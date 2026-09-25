"""Unsupervised mining of France (test only, no labels).

Compares token document-frequency between S1 and S2/S3, before and after normalization:
a token much more common on one side is a source-specific format or injected noise that the
normalizer does not yet reconcile. Also counts French address formats.
Usage: python france_miner.py
"""
import re
from collections import Counter

import polars as pl

from common import WORK, read_source

FORMATS = {
    "5-digit postcode": r"\b\d{5}\b",
    "number+bis/ter glued (12bis)": r"\b\d+(?:bis|ter|b|t)\b",
    "number range (12-14)": r"\b\d+\s*-\s*\d+\b",
    "cedex": r"\bcedex\b",
    "bp / cs (PO box)": r"\b(?:bp|cs)\s*\d+",
    "batiment / bat": r"\bb[aâ]t(?:iment)?\b",
    "appartement / appt": r"\bappt?\b|\bappartement\b",
    "residence": r"\br[ée]sidence\b",
    "zone (za/zi/zac)": r"\bz\.?\s?[aic]\b|\bzac\b",
    "lieu-dit": r"\blieu[- ]dit\b",
    "leading # on number": r"#\s*\d",
    "n° marker": r"\bn[°o]\.?\s*\d",
    "saint / st / ste in address": r"\b(?:saint|sainte|st|ste)\b",
}


def df_by_source(frames, col, tokenizer):
    out = {}
    for src, df in frames.items():
        c = Counter()
        for v in df[col].to_list():
            c.update(set(tokenizer(v)))
        out[src] = (c, df.height)
    return out


def shift_table(stats, min_count=500, top=30):
    (c1, n1), (c2, n2), (c3, n3) = stats[1], stats[2], stats[3]
    rows = []
    for tok in set(c1) | set(c2) | set(c3):
        f1, f23 = c1[tok] / n1, (c2[tok] + c3[tok]) / (n2 + n3)
        if c1[tok] + c2[tok] + c3[tok] < min_count:
            continue
        ratio = (f23 + 1e-5) / (f1 + 1e-5)
        rows.append((tok, f1, c2[tok] / n2, c3[tok] / n3, ratio))
    df = pl.DataFrame(rows, schema=["token", "S1", "S2", "S3", "S23_over_S1"], orient="row")
    return (df.sort("S23_over_S1", descending=True).head(top),
            df.sort("S23_over_S1").head(top))


def main():
    raw = {i: read_source("test", i).filter(pl.col("country") == "France") for i in (1, 2, 3)}
    norm = {i: pl.read_parquet(WORK / f"test_s{i}_norm.parquet").filter(pl.col("country") == "France")
            for i in (1, 2, 3)}
    rawtok = lambda s: re.findall(r"[^\s,]+", s.lower())
    cfg = pl.Config(tbl_rows=40, float_precision=4, fmt_str_lengths=40)
    with cfg:
        print("== French address formats (share of records per source)")
        rows = []
        for name, pat in FORMATS.items():
            rx = re.compile(pat, re.I)
            rows.append((name, *[sum(bool(rx.search(a)) for a in raw[i]["business_address"].to_list())
                                 / raw[i].height for i in (1, 2, 3)]))
        print(pl.DataFrame(rows, schema=["format", "S1", "S2", "S3"], orient="row"))
        for label, frames, col, tok in [
            ("RAW address tokens", raw, "business_address", rawtok),
            ("RAW name tokens", raw, "business_name", rawtok),
            ("NORMALIZED address tokens (residual gaps)", norm, "addr", str.split),
            ("NORMALIZED name_core tokens (residual gaps)", norm, "name_core", str.split)]:
            more23, more1 = shift_table(df_by_source(frames, col, tok))
            print(f"\n== {label}: much more common in S2/S3 than S1")
            print(more23)
            print(f"== {label}: much more common in S1 than S2/S3")
            print(more1)


if __name__ == "__main__":
    main()
