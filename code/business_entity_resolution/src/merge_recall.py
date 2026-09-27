"""v7 = our v6 matches + the patel-union matches whose candidate our blocking never produced.

The two leaderboard gains over 0.973 come from different places: patel's union (0.977) from a second candidate
source, ours (v6, 0.976) from re-scoring the same 8 candidates. So: keep every v6 decision on our own candidates,
add the union's accepted pairs from outside our candidate set, one S1 owner per S2/S3 record (v6 wins conflicts).
Candidates written = our K=8 set + the added pairs (the matched pairs must be candidates).

  python merge_recall.py <v6 dir> <union matching_results.tsv> <kaggle out dir with test_scores.parquet>
"""
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from common import DATA, ROOT, read_source
from evaluation import write_tsv


def pairs(path):
    d = pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None).fill_null("")
    return (d.rename({"source1_entity_id": "s1"}).with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"matched_entity_ids": "cand"}))


def main():
    v6_dir, union_path, kaggle_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    ours_cand = pl.read_parquet(kaggle_dir / "test_scores.parquet", columns=["s1", "cand"])
    v6, union = pairs(v6_dir / "matching_results.tsv"), pairs(union_path)
    new = union.join(ours_cand, on=["s1", "cand"], how="anti")             # candidate pairs our blocking never produced
    clash = new.join(v6.select("cand"), on="cand", how="semi")               # S2/S3 record already owned in v6
    add = new.join(v6.select("cand"), on="cand", how="anti")
    print(f"union pairs outside our candidates: {new.height:,}; owned by another S1 in v6: {clash.height:,}; added: {add.height:,}")
    final = pl.concat([v6, add])
    assert final["cand"].n_unique() == final.height, "an S2/S3 record has two owners"
    s1 = read_source("test", 1, ["entity_id", "country"])
    per = final.join(s1.rename({"entity_id": "s1"}), on="s1").group_by("country").agg(pl.len().alias("m"))
    n = dict(s1.group_by("country").len().rows())
    print("matches per S1:", {c: round(m / n[c], 3) for c, m in per.rows()}, f"| total {final.height:,}")
    out = Path(os.environ["FINAL_OUT"])
    out.mkdir(parents=True, exist_ok=False)                                  # never overwrite a submission
    ids = s1["entity_id"]
    write_tsv(pl.concat([ours_cand, add]).unique(), "candidate_entity_ids", out / "candidate_pairs.tsv", ids)
    write_tsv(final, "matched_entity_ids", out / "matching_results.tsv", ids)
    print(f"wrote {out}; S1 with no match {1 - final['s1'].n_unique() / len(ids):.1%}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test"), "--check-ids"])


if __name__ == "__main__":
    main()
