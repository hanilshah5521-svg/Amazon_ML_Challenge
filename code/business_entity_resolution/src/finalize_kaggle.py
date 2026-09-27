"""Turn the Kaggle notebook outputs (kaggle/amlc26_train.py) into a submission.

  python finalize_kaggle.py <dir with test_scores.parquet + config.json> [variant]

Applies the per-country decision chosen on held-out half A (config.json; countries without labels use
"_default"), writes candidate_pairs.tsv (the K=8 cut) and matching_results.tsv to a NEW output/v<N>_kaggle_<variant>/
folder (FINAL_OUT overrides the folder, for tests) and runs the official validator.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from common import DATA, ROOT, read_source
from evaluation import decide, new_output_dir, write_tsv
from model_v2 import decide_expected_f


def main():
    src = Path(sys.argv[1])
    cfg = json.loads((src / "config.json").read_text())
    variant = sys.argv[2] if len(sys.argv) > 2 else cfg["variant"]
    decision = cfg["decision"] if variant == cfg["variant"] else \
        {**{c: cfg["report"][variant][c] for c in cfg["report"][variant] if not c.startswith("_")}}
    if "_default" not in decision:
        decision["_default"] = max(decision.values(), key=lambda r: r["thr"])
    sc = pl.read_parquet(src / "test_scores.parquet").select("s1", "cand", "country", "rank", pl.col(variant).alias("score"))
    s1 = read_source("test", 1, ["entity_id", "country"])
    matches = []
    for country in sorted(s1["country"].unique().to_list()):
        c = sc.filter(pl.col("country") == country)
        d = decision.get(country, decision["_default"])
        m = decide_expected_f(c, d["miss"]) if d["rule"] == "expected_f" else decide(c, cfg["k"], d["thr"])
        matches.append(m)
        n = s1.filter(pl.col("country") == country).height
        print(f"{country}: {d['rule']} (thr {d['thr']:.3f}, miss {d['miss']}), {m.height:,} matches, "
              f"{m.height / n:.2f}/S1, no match {1 - m['s1'].n_unique() / n:.1%}", flush=True)
    matches = pl.concat(matches).unique("cand", keep="first")          # one S1 owner per S2/S3 record
    out = Path(os.environ["FINAL_OUT"]) if os.environ.get("FINAL_OUT") else new_output_dir(f"kaggle_{variant}")
    out.mkdir(parents=True, exist_ok=True)
    ids = s1["entity_id"]
    write_tsv(sc.select("s1", "cand"), "candidate_entity_ids", out / "candidate_pairs.tsv", ids)
    write_tsv(matches, "matched_entity_ids", out / "matching_results.tsv", ids)
    (out / "kaggle_config.json").write_text(json.dumps({**cfg, "variant_used": variant}, indent=1))
    print(f"wrote {out}: {matches.height:,} matches, S1 with no match {1 - matches['s1'].n_unique() / len(ids):.1%}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test"), "--check-ids"])


if __name__ == "__main__":
    main()
