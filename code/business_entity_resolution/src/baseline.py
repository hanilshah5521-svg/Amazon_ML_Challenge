"""Heuristic baseline matcher on top of the blocking scores (no learned model yet).

score = name_sim + addr_sim + w_num * (num_rel - 0.5) - w_legal * [legal forms conflict]
A candidate matches if it is within the adaptive cut (score >= S1's best - delta, rank <= k),
clears `thr`, and its S1's best candidate clears the stricter `gate`; then each S2/S3 record
is kept only for the S1 record that scores it highest.

Usage:
  python baseline.py tune      # per-country weights/threshold/gate on held-out train -> work/baseline_params.json
  python baseline.py submit    # apply to work/test_candidates.parquet, write output/, run the validator
"""
import itertools
import json
import subprocess
import sys

import polars as pl

from common import DATA, ROOT, WORK, read_pairs, read_source
from pair_features import add_pair_features

OUT = ROOT / "output"
PARAMS = WORK / "baseline_params.json"
K, DELTA = 10, 0.5


def f05_macro(pred, truth, s1_ids):
    """Macro F0.5 over s1_ids. pred/truth: DataFrames (s1, cand)."""
    p = pred.group_by("s1").agg(pl.col("cand").alias("p"))
    t = truth.group_by("s1").agg(pl.col("cand").alias("t"))
    df = (pl.DataFrame({"s1": s1_ids}).join(p, on="s1", how="left").join(t, on="s1", how="left")
            .with_columns(pl.col("p").fill_null([]), pl.col("t").fill_null([])))
    df = df.with_columns(
        pl.col("p").list.len().alias("np"), pl.col("t").list.len().alias("nt"),
        pl.col("p").list.set_intersection(pl.col("t")).list.len().alias("tp"))
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
           .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    return df.select(f.mean()).item()


def rescore(cand, w_num, w_legal):
    # Legal form is penalty-only: agreeing forms are common among decoys too, so no bonus.
    return cand.with_columns((pl.col("name_sim") + pl.col("addr_sim") + w_num * (pl.col("num_rel") - 0.5)
                              - w_legal * (pl.col("legal_rel") == 0).cast(pl.Float32)).alias("score"))


def decide(cand, k, thr, gate=None, delta=None):
    """Keep top-k candidates above `thr`, then one owner per S2/S3 record.

    gate:  an S1 record gets any match only if its best candidate scores >= gate. Singletons
           score 1.0 only with an empty prediction, so this bar is stricter than `thr`.
    delta: adaptive cut, keep only candidates within `delta` of the S1's best score.
    """
    c = cand.filter(pl.col("rank") <= k)
    if gate is not None or delta is not None:
        c = c.with_columns(pl.col("score").max().over("s1").alias("top"))
        if gate is not None:
            c = c.filter(pl.col("top") >= gate)
        if delta is not None:
            c = c.filter(pl.col("score") >= pl.col("top") - delta)
    m = c.filter(pl.col("score") >= thr)
    return (m.sort("score", descending=True).unique("cand", keep="first").select("s1", "cand"))


def tune():
    feat = WORK / "val_candidates_feat.parquet"
    if not feat.exists():
        add_pair_features(pl.read_parquet(WORK / "val_candidates.parquet"), "train").write_parquet(feat)
    cand = pl.read_parquet(feat).filter(pl.col("rank") <= K)
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"}).filter(pl.col("s1").is_in(val["entity_id"]))
    params = {}
    for country in val["country"].unique().sort():
        ids = val.filter(pl.col("country") == country)["entity_id"].to_list()
        t = truth.filter(pl.col("s1").is_in(ids))
        singles = set(ids) - set(t["s1"])
        c0 = cand.filter(pl.col("country") == country)
        best = (0, None)
        for w_num, w_legal in itertools.product((0.4, 0.8, 1.2), (0.0, 0.3, 0.6, 1.0)):
            c = rescore(c0, w_num, w_legal)
            for thr in [x / 20 for x in range(24, 40)]:           # 1.20 .. 1.95
                for gate in (thr, thr + 0.1, thr + 0.2):
                    f = f05_macro(decide(c, K, thr, gate, DELTA), t, ids)
                    if f > best[0]:
                        best = (f, dict(w_num=w_num, w_legal=w_legal, thr=thr, gate=gate))
        p = best[1]
        pred = decide(rescore(c0, p["w_num"], p["w_legal"]), K, p["thr"], p["gate"], DELTA)
        empty_ok = len(singles - set(pred["s1"])) / max(len(singles), 1)
        print(f"{country}: F0.5={best[0]:.4f} with {p}; singletons kept empty {empty_ok:.1%}; "
              f"{pred.height / len(ids):.2f} matches/S1", flush=True)
        params[country] = p
    # France has no labels: use the stricter of the two tuned settings.
    params["France"] = max(params.values(), key=lambda q: (q["gate"], q["thr"]))
    PARAMS.write_text(json.dumps(params, indent=1))
    print("France uses", params["France"], "->", PARAMS)


def _write(df, col, path, s1_ids):
    agg = df.group_by("s1").agg(pl.col("cand").unique().sort().str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")


def submit():
    OUT.mkdir(exist_ok=True)
    params = json.loads(PARAMS.read_text())
    feat = WORK / "test_candidates_feat.parquet"
    if not feat.exists():
        add_pair_features(pl.read_parquet(WORK / "test_candidates.parquet").filter(pl.col("rank") <= K),
                          "test").write_parquet(feat)
    cand = pl.read_parquet(feat)
    s1 = read_source("test", 1, ["entity_id", "country"])
    cand = cand.join(s1.rename({"entity_id": "s1"}), on="s1")
    kept, matches = [], []
    for country, p in params.items():
        c = rescore(cand.filter(pl.col("country") == country), p["w_num"], p["w_legal"])
        # candidate set = what the matcher actually scores: the adaptive cut of the top-K
        c = c.with_columns(pl.col("score").max().over("s1").alias("top")).filter(
            pl.col("score") >= pl.col("top") - DELTA)
        kept.append(c.select("s1", "cand"))
        matches.append(decide(c, K, p["thr"], p["gate"], DELTA))
    kept, matches = pl.concat(kept), pl.concat(matches)
    ids = s1["entity_id"]
    _write(kept, "candidate_entity_ids", OUT / "candidate_pairs.tsv", ids)
    _write(matches, "matched_entity_ids", OUT / "matching_results.tsv", ids)
    print(f"candidates {kept.height:,} ({kept.height / len(ids):.2f}/S1), matches {matches.height:,} "
          f"({matches.height / len(ids):.2f}/S1), S1 with no match: "
          f"{1 - matches['s1'].n_unique() / len(ids):.1%}")
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(OUT / "matching_results.tsv"),
                    "--candidate", str(OUT / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test")])


if __name__ == "__main__":
    tune() if sys.argv[1] == "tune" else submit()
