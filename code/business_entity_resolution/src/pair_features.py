"""Pair-level signals that separate true matches from planted decoys.

Decoys share the business name and street but differ in legal form ("Pvt Ltd" vs "LLP") or
unit number ("No.9" vs "No.14" on the same survey number). Among candidates scoring >= 1.5 on
held-out train: legal-form conflict 0.3% of true pairs vs 16-17% of false ones; "every number
of the shorter address also appears in the other" 85% of true vs 12-13% of false.
"""
import re

import polars as pl

from common import WORK

NUM = re.compile(r"\b\d+\b")
# Families the noise swaps freely inside true pairs (Private Limited ~ Ltd ~ Pvt).
LEGAL_FAMILY = {"pvt": "company", "ltd": "company", "plc": "company", "opc": "company",
                "llp": "llp", "llc": "llc", "inc": "inc", "corp": "inc", "co": "co",
                "trust": "trust", "pc": "pc", "pllc": "pllc", "lp": "lp",
                "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl",
                "sci": "sci", "snc": "snc", "ei": "ei"}


def legal_families(name):
    return {LEGAL_FAMILY[t] for t in name.split() if t in LEGAL_FAMILY}


def legal_relation(n1, n2):
    """1 = same family, 0.5 = missing on one side, 0 = conflicting forms."""
    a, b = legal_families(n1), legal_families(n2)
    if not a or not b:
        return 0.5
    return 1.0 if a & b else 0.0


def number_relation(a1, a2):
    """1 = the smaller number set is contained in the other, 0.5 = one side has no number,
    0 = partial or no overlap (e.g. same survey no. but different unit no.)."""
    x, y = set(NUM.findall(a1)), set(NUM.findall(a2))
    if not x or not y:
        return 0.5
    return 1.0 if (x <= y or y <= x) else 0.0


def add_pair_features(cand, split):
    """Join normalized name/address for both sides and add legal_rel and num_rel columns."""
    s1 = pl.read_parquet(WORK / f"{split}_s1_norm.parquet").select(
        pl.col("entity_id").alias("s1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"))
    oth = pl.concat([pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet").select(
        pl.col("entity_id").alias("cand"), pl.col("name").alias("n2"), pl.col("addr").alias("a2"))
        for i in (2, 3)])
    oth = oth.filter(pl.col("cand").is_in(cand["cand"].unique()))
    c = cand.join(s1, on="s1", how="left").join(oth, on="cand", how="left")
    legal = [legal_relation(a, b) for a, b in c.select("n1", "n2").iter_rows()]
    nums = [number_relation(a, b) for a, b in c.select("a1", "a2").iter_rows()]
    return c.with_columns(pl.Series("legal_rel", legal, dtype=pl.Float32),
                          pl.Series("num_rel", nums, dtype=pl.Float32)).drop("n1", "n2", "a1", "a2")
