"""v9 = a leaderboard file (patel v10, 0.983) + French pairs that only our region-filled v8 model accepts.

Only v8-only French pairs whose first house numbers agree (or one is missing) are added: 35.7% of all v8-only
French pairs have differing house numbers, the decoy pattern behind patel's S9 rule. Pairs whose S2/S3 record is
already owned in the base file are skipped (one S1 owner per record). India/US and every base pair stay unchanged.
Candidates written = our K=8 set + every matched pair (the matched pairs must be candidates).

  python add_fr_pairs.py <base matching_results.tsv> <v8 dir with matching_results.tsv> <kaggle out dir> [filled_only]
FINAL_OUT names the new output folder. filled_only: add only pairs whose S2/S3 address got a region from fr_region_fill.
"""
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from common import DATA, ROOT, WORK, read_source
from evaluation import write_tsv
from merge_recall import pairs


def first_num(a):
    """patel's rebuild_s9.py rule: first number of the raw address, skipping 5-6 digit postal codes after token 1."""
    import re
    a = re.sub(r"n°|no\.", " ", (a or "").lower()).replace("(", " ").replace(")", " ")
    for i, tok in enumerate(re.split(r"[\s,]+", a.strip())):
        m = re.search(r"\d+", tok)
        if m:
            if i > 0 and len(m.group()) in (5, 6):
                continue
            return str(int(m.group()))
    return None


def filled_ids():
    ids = set()
    for i in (2, 3):
        a = pl.read_parquet(WORK / f"test_s{i}_norm.parquet", columns=["entity_id", "addr"])
        b = pl.read_parquet(WORK / f"test_s{i}_norm.pre_frfill.parquet", columns=["entity_id", "addr"])
        ids |= set(a.join(b, on="entity_id").filter(pl.col("addr") != pl.col("addr_right"))["entity_id"].to_list())
    return ids


def main():
    base, v8 = pairs(sys.argv[1]), pairs(Path(sys.argv[2]) / "matching_results.tsv")
    ours_cand = pl.read_parquet(Path(sys.argv[3]) / "test_scores.parquet", columns=["s1", "cand"])
    s1 = read_source("test", 1, ["entity_id", "country"])
    fr = s1.filter(pl.col("country") == "France")["entity_id"]
    new = v8.filter(pl.col("s1").is_in(fr.implode())).join(base, on=["s1", "cand"], how="anti")
    raw = pl.concat([read_source("test", i, ["entity_id", "business_address"]) for i in (1, 2, 3)])
    fn = dict(zip(raw["entity_id"].to_list(), map(first_num, raw["business_address"].to_list())))
    h1, h2 = [fn.get(x) for x in new["s1"].to_list()], [fn.get(x) for x in new["cand"].to_list()]
    new = new.with_columns(pl.Series("hdiff", [a is not None and b is not None and a != b for a, b in zip(h1, h2)]))
    add = new.filter(~pl.col("hdiff")).join(base.select("cand"), on="cand", how="anti").drop("hdiff")
    if len(sys.argv) > 4 and sys.argv[4] == "filled_only":
        add = add.filter(pl.col("cand").is_in(list(filled_ids())))
    print(f"v8-only French pairs {new.height:,}; house number differs {new['hdiff'].sum():,}; added {add.height:,}")
    final = pl.concat([base, add])
    assert final["cand"].n_unique() == final.height, "an S2/S3 record has two owners"
    out = Path(os.environ["FINAL_OUT"])
    out.mkdir(parents=True, exist_ok=False)                                  # never overwrite a submission
    ids = s1["entity_id"]
    write_tsv(pl.concat([ours_cand, final]).unique(), "candidate_entity_ids", out / "candidate_pairs.tsv", ids)
    write_tsv(final, "matched_entity_ids", out / "matching_results.tsv", ids)
    print(f"wrote {out}: {final.height:,} matches (base {base.height:,})", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test"), "--check-ids"])


if __name__ == "__main__":
    main()
