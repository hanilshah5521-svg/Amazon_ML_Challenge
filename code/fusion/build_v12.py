"""v12 = v10 (EXP-5 India/US + S9 France) minus the India/US pairs that the teammate's 0.9822 file removed from the 0.977 file
(teammate_iu_removed.parquet, in_e5 = still accepted by EXP-5). Usage: python build_v12.py <out_dir>"""
import sys, pathlib, shutil, polars as pl
R = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/COMPETITIONS/AMAZON_ML/hv_repo/output/"
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rm = pl.read_parquet("teammate_iu_removed.parquet").filter("in_e5")
bad = set(zip(rm["s1"], rm["cand"]))
v10 = pl.read_csv(R + "v10_e5_s9france/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
n = 0
with open(out / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1, ms in v10.iter_rows():
        keep = [m for m in (ms or "").split(",") if m and (s1, m) not in bad]
        n += len((ms or "").split(",")) - len(keep) if ms else 0
        f.write(f"{s1}\t{','.join(keep)}\n")
shutil.copyfile(R + "v10_e5_s9france/candidate_pairs.tsv", out / "candidate_pairs.tsv")
print(f"removed {n:,} pairs (expected {len(bad):,})")
