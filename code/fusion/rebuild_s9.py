"""Reconstruct S9 (0.9796) = E1 France minus pairs of S10's removal set whose first house numbers differ (teammate's rule).
Usage: python rebuild_s9.py"""
import re, polars as pl
R = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/COMPETITIONS/AMAZON_ML/"
T = R + "student_resource/dataset/test/"

def first_num(a):
    a = re.sub(r"n°|no\.", " ", (a or "").lower()).replace("(", " ").replace(")", " ")
    for i, tok in enumerate(re.split(r"[\s,]+", a.strip())):
        m = re.search(r"\d+", tok)
        if m:
            if i > 0 and len(m.group()) in (5, 6):
                continue                      # postal code
            return str(int(m.group()))
    return None

raw = pl.concat([pl.read_csv(T + f"test_source{i}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
                 .select("entity_id", "business_address") for i in (1, 2, 3)])
fn = dict(zip(raw["entity_id"], map(first_num, raw["business_address"])))
fr = pl.read_parquet("fr_pairs_e1.parquet")          # E1 France accepts, kept = in S10
rm = fr.filter(pl.col("kept") == 0).with_columns(
    pl.col("s1").replace_strict(fn, default=None).alias("h1"), pl.col("m").replace_strict(fn, default=None).alias("h2"))
rm = rm.with_columns((pl.col("h1").is_not_null() & pl.col("h2").is_not_null() & (pl.col("h1") != pl.col("h2"))).alias("s9_rm"))
print(f"S10 removals {rm.height:,}; with first house number different (S9 rule): {rm['s9_rm'].sum():,}  (teammate: 16,642)")
rm.select("s1", "m", "s9_rm", "n1", "n2", "a1", "a2").write_parquet("s10_removed_split.parquet")
assert first_num("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac") == "5"
assert first_num("0028 rue x, 44000 Nantes") == "28" and first_num("Nantes 44000, N° 12 rue y") == "12"
