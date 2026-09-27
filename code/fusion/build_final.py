"""Final file: India/US from a model output minus the teammate's India/US removal list (pairs the 0.9822 team file removed from
the 0.977 file), France rows copied from another finished file. Usage: python build_final.py <iu_output_dir> <france_file_dir> <out_dir>"""
import sys, pathlib, shutil, polars as pl
iu_dir, fr_dir, out = (pathlib.Path(p) for p in sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0)
rm = pl.read_parquet("teammate_iu_removed.parquet")
bad = set(zip(rm["s1"], rm["cand"]))
fr_ids = set(pl.read_parquet("../../hv_repo/work/exp2/test_s1_norm.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"])
frm = dict(rd(fr_dir / "matching_results.tsv").filter(pl.col("source1_entity_id").is_in(list(fr_ids))).iter_rows())
n_rm = n_fr = 0
with open(out / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1, ms in rd(iu_dir / "matching_results.tsv").iter_rows():
        if s1 in fr_ids:
            ms = frm.get(s1) or ""
            n_fr += len([m for m in ms.split(",") if m])
        else:
            keep = [m for m in (ms or "").split(",") if m and (s1, m) not in bad]
            n_rm += len([m for m in (ms or "").split(",") if m]) - len(keep)
            ms = ",".join(keep)
        f.write(f"{s1}\t{ms}\n")
shutil.copyfile(iu_dir / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
print(f"India/US from {iu_dir.name}: removed {n_rm:,} teammate-rule pairs | France from {fr_dir.name}: {n_fr:,} pairs")
