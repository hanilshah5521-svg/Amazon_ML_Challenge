"""France from EXP-5 with French decoy removal (teammate's plan).
  set : EXP-5 France minus the fixed S9 list (s10_removed_split.parquet, s9_rm)
  both: set + the S9 RULE on every remaining French pair: both first house numbers exist and differ AND the candidate name is the
        S1 name plus extra word(s) or an acronym of it (legal forms ignored; cats.cat categories plus_word / acronym).
India/US are taken from --iu (default: the --src output). Also prints the rule's recall / precision against S9 on the EXP-1 file.
Usage: python build_fr_exp5.py --src <exp5 output> --iu <india/us output> --out <dir> --mode set|both --e1 <exp1 output>"""
import argparse, pathlib, shutil, sys
import polars as pl
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
from rebuild_s9 import first_num, raw           # same first_num() as the S9 reconstruction
from cats import add_cat, names

ap = argparse.ArgumentParser()
for a in ("--src", "--iu", "--out", "--mode", "--e1"):
    ap.add_argument(a)
A = ap.parse_args(ARGS)
fn = dict(zip(raw["entity_id"], map(first_num, raw["business_address"])))
rd = lambda p: (pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0).rename({"source1_entity_id": "s1", "matched_entity_ids": "cand"})
                  .with_columns(pl.col("cand").fill_null("").str.split(",")).explode("cand").filter(pl.col("cand") != ""))
fr = pl.read_parquet("../../hv_repo/work/exp2/test_s1_norm.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
s9 = pl.read_parquet("s10_removed_split.parquet").filter("s9_rm").select("s1", pl.col("m").alias("cand"))


def rule(d):
    d = add_cat(names("test")(d.select("s1", "cand")))
    nd = pl.struct("s1", "cand").map_elements(lambda r: fn.get(r["s1"]) is not None and fn.get(r["cand"]) is not None and fn[r["s1"]] != fn[r["cand"]], return_dtype=pl.Boolean)
    return d.with_columns((nd & pl.col("cat").str.split("|").list.first().is_in(["plus_word", "acronym"])).alias("hit"))


if A.e1:
    e1 = rule(rd(A.e1 + "/matching_results.tsv").join(fr, on="s1"))
    hits = e1.filter("hit").select("s1", "cand")
    tp = hits.join(s9, on=["s1", "cand"]).height
    print(f"rule recall of S9 {tp / s9.height:.3f} ({tp:,}/{s9.height:,}) | rule precision vs S9 {tp / max(1, hits.height):.3f} ({tp:,}/{hits.height:,})", flush=True)
f5 = rd(A.src + "/matching_results.tsv").join(fr, on="s1").join(s9, on=["s1", "cand"], how="anti")
if A.mode == "both":
    r = rule(f5)
    r.filter("hit").select("s1", "cand", "n1", "n2", "a1", "a2").write_csv(A.out.rstrip("/") + "_fr_rule_hits.tsv", separator="\t")
    f5 = r.filter(~pl.col("hit")).select("s1", "cand")
    print(f"rule removes {r['hit'].sum():,} more French pairs", flush=True)
frm = dict(f5.group_by("s1").agg(pl.col("cand").sort().str.join(",")).iter_rows())
frs = set(fr["s1"].to_list())
iu = pl.read_csv((A.iu or A.src) + "/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
out = pathlib.Path(A.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1, ms in iu.iter_rows():
        f.write(f"{s1}\t{frm.get(s1, '') if s1 in frs else (ms or '')}\n")
shutil.copyfile((A.iu or A.src) + "/candidate_pairs.tsv", out / "candidate_pairs.tsv")
print(f"{A.mode}: France pairs {f5.height:,}; India/US from {A.iu or A.src}", flush=True)
