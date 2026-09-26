"""24 base pair features (the feature set of the 0.915 submission), shared by the final pipeline.

Blocking similarities, decoy checks (legal form, house numbers), string ratios on name_core / name_key / no-space names,
empty-field flags, and relative features against the other candidates of the same S1. Country-agnostic: no country flag.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import WORK
from pair_features import legal_relation, number_relation


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
            "name_rank", "addr_rank", "best_key"]


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
