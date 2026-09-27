"""Export the 0.977 submission's (EXP-1) scores and decision rule. Read-only on existing work/output; writes only to export_0977/.
No retraining: test pairs are re-scored with the saved EXP-1 models (stage1.txt, stage2.txt).
Usage: WORK_DIR=<hv_repo>/work/exp1 MODEL_K=8 python export_0977.py {oof|test|check}"""
import json, os, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

TAG = os.environ.get("EXPORT_TAG", "")      # "" = EXP-1 (the 0.977 file); e.g. "exp2_" / "exp3_" for other experiments' scores
assert TAG or not (os.environ.get("PATTERN_FEATS") or os.environ.get("OWNER_FEATS")), "EXP-1 export uses the EXP-1 feature set"
import model_v2 as M
from common import read_pairs, read_source

EXP = Path(os.environ["WORK_DIR"]); OUT = M.ROOT.parent / "export_0977"; OUT.mkdir(exist_ok=True)
CFG = json.loads((EXP / "v2" / "config.json").read_text())
SUB = M.ROOT / "output" / os.environ.get("EXPORT_SUB", "v2_lgbm_two_stage")


def oof():
    o = pl.read_parquet(EXP / "v2" / "oof.parquet")           # same row order as train_features.parquet -> folds reproducible
    fold = np.zeros(o.height, dtype=np.int8)
    for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(o.height), o["y"].to_numpy(), o["s1"].to_numpy())):
        fold[te] = k
    cut = pl.concat([pl.read_parquet(EXP / f, columns=["s1", "cand", "score"]) for f in ("val_candidates_cut.parquet", "extra_candidates_cut.parquet")])
    halves = pl.read_parquet(EXP / "v2" / "halves.parquet")
    val = pl.read_parquet(EXP / "val_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"}).with_columns(pl.lit(True).alias("held_out"))
    d = (o.with_columns(pl.Series("fold", fold)).join(cut.rename({"score": "ranker_score"}), on=["s1", "cand"], how="left")
          .join(halves, on="s1", how="left").join(val, on="s1", how="left").with_columns(pl.col("held_out").fill_null(False)))
    d = d.select(pl.col("s1").alias("source1_entity_id"), pl.col("cand").alias("candidate_entity_id"), pl.col("p2").alias("score"),
                 pl.col("y").alias("label"), "fold", pl.col("p1").alias("stage1_score"), pl.col("p2").alias("stage2_score"),
                 "ranker_score", pl.col("rank").alias("cut_rank"), "country", "held_out", pl.col("half").alias("heldout_half"),
                 "name_sim", "addr_sim", "via")
    d.write_parquet(OUT / f"{TAG}train_oof_scores.parquet", compression="zstd")
    print("train_oof_scores", d.height, d.select(pl.col("score").min().alias("min"), pl.col("score").mean().alias("mean"), pl.col("score").max().alias("max"),
          pl.col("score").is_between(0.3, 0.7).sum().alias("n_0.3_0.7"), pl.col("label").sum()).to_dicts())


def test():
    m1, m2 = M.lgb.Booster(model_file=str(EXP / "v2" / "stage1.txt")), M.lgb.Booster(model_file=str(EXP / "v2" / "stage2.txt"))
    parts = []
    for country in sorted(read_source("test", 1, ["entity_id", "country"])["country"].unique().to_list()):
        cand = (pl.read_parquet(EXP / f"test_candidates_cut_{country}.parquet").filter(pl.col("rank") <= M.MODEL_K)
                  .with_columns(pl.lit(country).alias("country")))
        def score_chunk(f):                       # identical to model_v2.predict, but keeps every intermediate score
            if M.OWNER:
                f = M.add_oc(f, "test")
            f = f.with_columns(pl.Series("p1", m1.predict(f.select(M.P1).to_numpy()).astype(np.float32)))
            f = M.stage2_features(f)
            f = f.with_columns(pl.Series("p2", m2.predict(f.select(M.P2).to_numpy()).astype(np.float32)))
            return f.select("s1", "cand", "country", "p1", "p2", "rank", pl.col("score").alias("ranker_score"))
        parts.append(M.features_chunked(cand, "test", on_chunk=score_chunk))
        print(country, parts[-1].height, flush=True)
    d = pl.concat(parts).select(pl.col("s1").alias("source1_entity_id"), pl.col("cand").alias("candidate_entity_id"),
                                pl.col("p2").alias("score"), pl.col("p1").alias("stage1_score"), pl.col("p2").alias("stage2_score"),
                                "ranker_score", pl.col("rank").alias("cut_rank"), "country")
    d.write_parquet(OUT / f"{TAG}test_scores.parquet", compression="zstd")


def apply_decision(scores):
    """decision.json applied to test_scores -> (s1, cand) matches, exactly as model_v2.predict."""
    out = []
    for country in sorted(scores["country"].unique().to_list()):
        d = CFG["decision"].get(country, CFG["decision"]["_default"])
        c = scores.filter(pl.col("country") == country).select(pl.col("source1_entity_id").alias("s1"), pl.col("candidate_entity_id").alias("cand"),
                                                                "score", pl.col("cut_rank").alias("rank"))
        out.append(M.decide_expected_f(c, d["miss"]) if d["rule"] == "expected_f" else M.decide(c, M.MODEL_K, d["thr"]))
    return pl.concat(out).unique("cand", keep="first")


def check():
    s = pl.read_parquet(OUT / f"{TAG}test_scores.parquet")
    mine = apply_decision(s)
    sub = (pl.read_csv(SUB / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "cand"}).with_columns(pl.col("cand").fill_null("").str.split(","))
             .explode("cand").filter(pl.col("cand") != ""))
    cp = (pl.read_csv(SUB / "candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema_length=0)
            .rename({"source1_entity_id": "s1", "candidate_entity_ids": "cand"}).with_columns(pl.col("cand").str.split(",")).explode("cand"))
    sp = s.select(pl.col("source1_entity_id").alias("s1"), pl.col("candidate_entity_id").alias("cand"))
    r = dict(test_pairs=s.height, reproduced_matches=mine.height, submitted_matches=sub.height,
             only_reproduced=mine.join(sub, on=["s1", "cand"], how="anti").height, only_submitted=sub.join(mine, on=["s1", "cand"], how="anti").height,
             candidate_pairs_not_in_scores=cp.join(sp, on=["s1", "cand"], how="anti").height,
             scores_not_in_candidate_pairs=sp.join(cp, on=["s1", "cand"], how="anti").height,
             score_min=s["score"].min(), score_mean=s["score"].mean(), score_max=s["score"].max(),
             n_score_0p3_0p7=s.filter(pl.col("score").is_between(0.3, 0.7)).height)
    print(json.dumps(r, indent=1)); (OUT / f"{TAG}sanity_checks.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    {"oof": oof, "test": test, "check": check}[sys.argv[1]]()
