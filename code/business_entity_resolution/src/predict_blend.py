"""Same as model_v2.predict(), but blends the v4 pipeline's stage-2 score with the cross-encoder's p_ce
(work/ce/ce_test_{country}.parquet) before the decision step: score = W*p2 + (1-W)*p_ce.
W was measured on held-out half B (work/blend_eval.py): India +0.0018, US +0.0012 F0.5 at W=0.7. France has no
labels; it uses the same W as the labelled countries (consistent with France already inheriting their decision
settings). Falls back to p2 alone for any pair missing a p_ce score (should be ~0%, same candidate universe).

Usage: MODEL_K=8 TEST_CAND=test_candidates_cut_{country}.parquet python predict_blend.py
"""
import json
import subprocess
import sys

import numpy as np
import polars as pl
import lightgbm as lgb

from common import DATA, ROOT, WORK, read_source
from evaluation import decide, new_output_dir, write_tsv
from model_v2 import V2, MODEL_K, TEST_CAND, P1, P2, features_chunked, stage2_features, decide_expected_f

W_BLEND = 0.7  # measured: weight on the v4 pipeline's own score; (1 - W_BLEND) on the cross-encoder


def main():
    cfg = json.loads((V2 / "config.json").read_text())
    m1 = lgb.Booster(model_file=str(V2 / "stage1.txt"))
    m2 = lgb.Booster(model_file=str(V2 / "stage2.txt")) if cfg["use_stage2"] else None
    s1 = read_source("test", 1, ["entity_id", "country"])
    kept, matches = [], []
    for country in sorted(s1["country"].unique().to_list()):
        cand = (pl.read_parquet(WORK / TEST_CAND.format(country=country)).filter(pl.col("rank") <= MODEL_K)
                  .with_columns(pl.lit(country).alias("country")))

        def score_chunk(f):
            f = f.with_columns(pl.Series("p1", m1.predict(f.select(P1).to_numpy()).astype(np.float32)))
            if m2 is not None:
                f = stage2_features(f)
                f = f.with_columns(pl.Series("p", m2.predict(f.select(P2).to_numpy()).astype(np.float32)))
            else:
                f = f.with_columns(pl.col("p1").alias("p"))
            return f.select("s1", "cand", pl.col("p").alias("p2"), "rank")

        c = features_chunked(cand, "test", on_chunk=score_chunk)
        del cand
        ce = pl.read_parquet(WORK / "ce" / f"ce_test_{country}.parquet")
        n0 = c.height
        c = c.join(ce, on=["s1", "cand"], how="left")
        miss_ce = c["p_ce"].null_count()
        c = c.with_columns(
            (W_BLEND * pl.col("p2") + (1 - W_BLEND) * pl.col("p_ce").fill_null(pl.col("p2"))).alias("score"))
        d = cfg["decision"].get(country, cfg["decision"]["_default"])
        matches.append(decide_expected_f(c, d["miss"]) if d["rule"] == "expected_f" else decide(c, MODEL_K, d["thr"]))
        kept.append(c.select("s1", "cand"))
        print(f"{country}: {n0:,} pairs scored ({miss_ce:,} without a CE score, fell back to p2), "
              f"{matches[-1].height:,} matches", flush=True)
        del c
    kept, matches = pl.concat(kept), pl.concat(matches).unique("cand", keep="first")
    ids = s1["entity_id"]
    OUT = new_output_dir("lgbm_two_stage_ce_blend")
    write_tsv(kept, "candidate_entity_ids", OUT / "candidate_pairs.tsv", ids)
    write_tsv(matches, "matched_entity_ids", OUT / "matching_results.tsv", ids)
    print(f"matches {matches.height:,}, S1 with no match {1 - matches['s1'].n_unique() / len(ids):.1%}", flush=True)
    print(f"output dir: {OUT}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(OUT / "matching_results.tsv"), "--candidate", str(OUT / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test"), "--check-ids"])


if __name__ == "__main__":
    main()
