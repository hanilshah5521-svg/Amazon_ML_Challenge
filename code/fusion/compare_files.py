"""Pair-level comparison of submission files (test, no labels). Usage: python compare_files.py > 01_compare.txt"""
import polars as pl
R = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/COMPETITIONS/AMAZON_ML/"
FILES = {"H9796": "/Users/patel/Downloads/matching_results (1).tsv",
         "E2": R + "hv_repo/output/v3_lgbm_two_stage/matching_results.tsv",
         "E1_977": R + "hv_repo/output/v2_lgbm_two_stage/matching_results.tsv",
         "E0": R + "hv_repo/output/v1_lgbm_two_stage/matching_results.tsv", "SA": R + "hv_repo/output/v4_fusion_e2_h9796fr/matching_results.tsv"}
CAND = R + "hv_repo/output/v3_lgbm_two_stage/candidate_pairs.tsv"

def pairs(path, col="matched_entity_ids"):
    d = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8} if col else None, infer_schema_length=0)
    c = d.columns[1]
    return (d.rename({d.columns[0]: "s1", c: "m"}).with_columns(pl.col("m").fill_null("").str.split(","))
             .explode("m").filter(pl.col("m") != "").unique())

country = pl.read_parquet(R + "hv_repo/work/exp2/test_s1_norm.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
P = {k: pairs(v).join(country, on="s1", how="left") for k, v in FILES.items()}
cand = pairs(CAND).select("s1", "m").with_columns(pl.lit(True).alias("in_cand"))
H = P["H9796"]
print("pairs per file and country:")
for k, d in P.items():
    print(f"  {k:7s}", d.group_by("country").len().sort("country").to_dicts(), "total", d.height)
print("\nH9796 vs others (pairs): both / only H / only other")
for k, d in P.items():
    if k == "H9796": continue
    for c in ["France", "India", "US"]:
        h, o = H.filter(pl.col("country") == c).select("s1", "m"), d.filter(pl.col("country") == c).select("s1", "m")
        b = h.join(o, on=["s1", "m"]).height
        print(f"  vs {k:7s} {c:7s} both {b:>9,}  onlyH {h.height-b:>7,}  only{k} {o.height-b:>7,}")
hc = H.join(cand, on=["s1", "m"], how="left")
print("\nH9796 pairs inside our EXP-2 candidate set (8/S1):", hc.group_by("country").agg(pl.col("in_cand").is_not_null().mean()).sort("country").to_dicts())
onlyH = H.join(P["E2"], on=["s1", "m"], how="anti").join(cand, on=["s1", "m"], how="left")
print("H-only (vs E2) pairs inside our candidates:", onlyH.group_by("country").agg(pl.len(), pl.col("in_cand").is_not_null().mean()).sort("country").to_dicts())
# S1-level
def sets(d): return d.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("ms"))
s = country.join(sets(H).rename({"ms": "h"}), on="s1", how="left").join(sets(P["E2"]).rename({"ms": "e"}), on="s1", how="left").fill_null("")
print("\nS1 with identical match sets H vs E2:", s.group_by("country").agg((pl.col("h") == pl.col("e")).mean(), (pl.col("h") == "").mean().alias("H_empty"), (pl.col("e") == "").mean().alias("E2_empty")).sort("country").to_dicts())
onlyH.write_parquet(R + "experiments/FUSION/onlyH_vs_E2.parquet")
H.join(P["E2"], on=["s1", "m"], how="anti").write_parquet(R + "experiments/FUSION/onlyH_vs_E2.parquet")
P["E2"].join(H, on=["s1", "m"], how="anti").write_parquet(R + "experiments/FUSION/onlyE2_vs_H.parquet")
