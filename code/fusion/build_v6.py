"""India/US from one experiment's output + S9 France (E1 France minus the 16,641 reconstructed S9 removals).
Usage: python build_v6.py <out_dir> [india_us_output_dir; default hv_repo/output/v3_lgbm_two_stage = EXP-2]"""
import sys, pathlib, shutil, polars as pl
R = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/COMPETITIONS/AMAZON_ML/"
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SRC = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else R + "hv_repo/output/v3_lgbm_two_stage")
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0)
E1, EX = dict(rd(R + "hv_repo/output/v2_lgbm_two_stage/matching_results.tsv").iter_rows()), rd(SRC / "matching_results.tsv")
fr = set(pl.read_parquet(R + "hv_repo/work/exp2/test_s1_norm.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"])
rm = pl.read_parquet("s10_removed_split.parquet").filter("s9_rm")
bad = set(zip(rm["s1"], rm["m"]))
n_fr = 0
with open(out / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1, ms in EX.iter_rows():
        if s1 in fr:
            ms = ",".join(m for m in (E1[s1] or "").split(",") if m and (s1, m) not in bad)
            n_fr += len([m for m in ms.split(",") if m])
        f.write(f"{s1}\t{ms or ''}\n")
shutil.copyfile(SRC / "candidate_pairs.tsv", out / "candidate_pairs.tsv")     # same learned cut (8 per S1) in EXP-1/2/3
print(f"India/US from {SRC.name}; France pairs {n_fr:,} (E1 865,515 - {len(bad):,} = {865515 - len(bad):,})")
