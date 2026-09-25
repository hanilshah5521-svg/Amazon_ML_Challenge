"""LightGBM pair matcher (prototype on the held-out train candidates).

Features: blocking similarities, decoy checks (legal form, house numbers), string ratios on
name_core / name_key / no-space names, empty-field flags, and relative features against the
other candidates of the same S1. Grouped K-fold by S1 so a record's candidates never span folds.
Out-of-fold probabilities are turned into matches with a per-country threshold (+ adaptive cut
and one owner per S2/S3 record) and scored with macro F0.5.
Usage: python lgbm_matcher.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.model_selection import GroupKFold

from evaluation import decide, f05_macro
from common import WORK, read_pairs
from pair_features import legal_relation, number_relation

K = 10


def load_text(split, ids, side):
    """name/name_core/name_key/addr for the given entity ids only (lazy scan keeps RAM low)."""
    files = [WORK / f"{split}_s1_norm.parquet"] if side == "s1" else \
            [WORK / f"{split}_s{i}_norm.parquet" for i in (2, 3)]
    df = pl.concat([pl.scan_parquet(f).filter(pl.col("entity_id").is_in(ids)).collect() for f in files])
    suffix = "1" if side == "s1" else "2"
    return df.select(pl.col("entity_id").alias(side if side == "s1" else "cand"),
                     *[pl.col(c).alias(c + suffix) for c in ("name", "name_core", "name_key", "addr")])


def build_features(cand, split):
    c = (cand.join(load_text(split, cand["s1"].unique(), "s1"), on="s1", how="left")
             .join(load_text(split, cand["cand"].unique(), "cand"), on="cand", how="left"))
    col = lambda n: c[n].fill_null("").to_list()
    n1, n2, c1, c2, k1, k2, a1, a2 = (col(x) for x in ("name1", "name2", "name_core1", "name_core2",
                                                       "name_key1", "name_key2", "addr1", "addr2"))
    ns1, ns2 = [x.replace(" ", "") for x in c1], [x.replace(" ", "") for x in c2]
    pair = lambda scorer, x, y: cpdist(x, y, scorer=scorer, workers=-1).astype(np.float32)
    feats = {
        "legal_rel": [legal_relation(x, y) for x, y in zip(n1, n2)],
        "num_rel": [number_relation(x, y) for x, y in zip(a1, a2)],
        "core_ratio": pair(fuzz.ratio, c1, c2), "core_tsort": pair(fuzz.token_sort_ratio, c1, c2),
        "core_tset": pair(fuzz.token_set_ratio, c1, c2), "core_partial": pair(fuzz.partial_ratio, c1, c2),
        "key_equal": [float(x == y) for x, y in zip(k1, k2)], "key_ratio": pair(fuzz.ratio, k1, k2),
        "nospace_ratio": pair(fuzz.ratio, ns1, ns2),
        "addr_tset": np.where([bool(y) for y in a2], pair(fuzz.token_set_ratio, a1, a2), -1),
        "addr_empty": [float(not y) for y in a2],
        "len_diff": [abs(len(x) - len(y)) for x, y in zip(c1, c2)],
    }
    c = c.with_columns([pl.Series(k, v, dtype=pl.Float32) for k, v in feats.items()])
    # relative features: how this candidate compares with the other candidates of its S1
    return c.with_columns(
        (pl.col("score").max().over("s1") - pl.col("score")).alias("gap_to_best"),
        (pl.col("score") / pl.col("score").max().over("s1")).alias("score_over_best"),
        (pl.col("score") >= pl.col("score").max().over("s1") - 0.3).sum().over("s1").alias("n_close"),
        pl.col("name_sim").rank("dense", descending=True).over("s1").alias("name_rank"),
        pl.col("addr_sim").rank("dense", descending=True).over("s1").alias("addr_rank"),
        (pl.col("key_ratio") == pl.col("key_ratio").max().over("s1")).cast(pl.Float32).alias("best_key"),
    ).drop("name1", "name2", "name_core1", "name_core2", "name_key1", "name_key2", "addr1", "addr2")


FEATURES = ["name_sim", "addr_sim", "score", "rank", "via", "legal_rel", "num_rel", "core_ratio",
            "core_tsort", "core_tset", "core_partial", "key_equal", "key_ratio", "nospace_ratio",
            "addr_tset", "addr_empty", "len_diff", "gap_to_best", "score_over_best", "n_close",
            "name_rank", "addr_rank", "best_key", "is_india"]


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
THR = {"India": 0.65, "US": 0.70, "France": 0.70}   # from out-of-fold tuning; France = stricter


def training_frame():
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"}).filter(pl.col("s1").is_in(val["entity_id"]))
    cand = pl.read_parquet(WORK / "val_candidates.parquet").filter(pl.col("rank") <= K)
    df = build_features(cand, "train").join(truth.with_columns(pl.lit(1).alias("y")),
                                            on=["s1", "cand"], how="left")
    return df.with_columns(pl.col("y").fill_null(0),
                           (pl.col("country") == "India").cast(pl.Float32).alias("is_india"))


def submit():
    import subprocess, sys
    from evaluation import OUT, write_tsv
    from common import DATA, ROOT, read_source
    tr = training_frame()
    model = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy()), 400)
    s1 = read_source("test", 1, ["entity_id", "country"])
    cand = (pl.read_parquet(WORK / "test_candidates.parquet").filter(pl.col("rank") <= K)
              # no per-record cut: measured -2.5 F0.5 pts with it (India .913->.886, US .942->.919)
              .join(s1.rename({"entity_id": "s1"}), on="s1"))
    del tr
    kept, parts = [], []
    for country, thr in THR.items():                     # one country at a time: ~1/3 peak RAM
        feat = build_features(cand.filter(pl.col("country") == country), "test").with_columns(
            (pl.col("country") == "India").cast(pl.Float32).alias("is_india"))
        feat = feat.with_columns(pl.Series("score", model.predict(feat.select(FEATURES).to_numpy())))
        kept.append(feat.select("s1", "cand"))
        parts.append(decide(feat, K, thr))
        print(f"   {country}: {feat.height:,} pairs scored", flush=True)
        del feat
    feat, matches = pl.concat(kept), pl.concat(parts)
    ids = s1["entity_id"]
    OUT.mkdir(exist_ok=True)
    write_tsv(feat.select("s1", "cand"), "candidate_entity_ids", OUT / "candidate_pairs.tsv", ids)
    write_tsv(matches, "matched_entity_ids", OUT / "matching_results.tsv", ids)
    print(f"candidates {feat.height:,} ({feat.height / len(ids):.2f}/S1), matches {matches.height:,}, "
          f"S1 with no match {1 - matches['s1'].n_unique() / len(ids):.1%}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                    "--matching", str(OUT / "matching_results.tsv"),
                    "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")])


def main():
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"}).filter(pl.col("s1").is_in(val["entity_id"]))
    cand = pl.read_parquet(WORK / "val_candidates.parquet").filter(pl.col("rank") <= K)
    df = build_features(cand, "train").join(truth.with_columns(pl.lit(1).alias("y")),
                                            on=["s1", "cand"], how="left")
    df = df.with_columns(pl.col("y").fill_null(0), (pl.col("country") == "India").cast(pl.Float32).alias("is_india"))
    X, y = df.select(FEATURES).to_numpy(), df["y"].to_numpy()
    groups = df["s1"].to_numpy()
    oof = np.zeros(len(df))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
        oof[te] = m.predict(X[te])
    df = df.with_columns(pl.Series("score", oof))   # decide() works on `score`
    imp = sorted(zip(m.feature_importance("gain"), FEATURES), reverse=True)[:12]
    print("top features by gain:", [f for _, f in imp])
    for country in ("India", "US"):
        ids = val.filter(pl.col("country") == country)["entity_id"].to_list()
        t = truth.filter(pl.col("s1").is_in(ids))
        c = df.filter(pl.col("country") == country)
        best = max((f05_macro(decide(c, K, thr, None, None), t, ids), thr)
                   for thr in np.arange(0.2, 0.95, 0.05))
        print(f"{country}: LightGBM out-of-fold F0.5 = {best[0]:.4f} at p >= {best[1]:.2f}")


if __name__ == "__main__":
    import sys
    submit() if len(sys.argv) > 1 and sys.argv[1] == "submit" else main()
