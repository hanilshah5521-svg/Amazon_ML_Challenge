"""Is the candidate set useful, and where should K be?

Reads work/val_candidates.parquet (held-out train S1 vs the full train pool, top-100 per S1)
and reports per country:
  1. K curve: recall, entity coverage, candidates/S1, marginal gain, oracle F0.5, achieved F0.5
  2. adaptive cut: keep candidates within delta of each S1's best score (capped at K_max)
  3. where true matches rank, how well the score separates true from false candidates
  4. singleton exposure and causes of missed matches
Usage: python blocking_analysis.py
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score

from evaluation import decide, f05_macro
from common import WORK, read_pairs

KS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100)
THRESHOLDS = [x / 20 for x in range(10, 41)]      # 0.50 .. 2.00


def best_f05(cand, truth, ids, k):
    best = (0.0, None)
    for thr in THRESHOLDS:
        f = f05_macro(decide(cand, k, thr), truth, ids)
        if f > best[0]:
            best = (f, thr)
    return best


def oracle_f05(cand_k, truth, ids):
    """F0.5 if the matcher were perfect on these candidates (keeps exactly the true ones)."""
    hits = cand_k.join(truth, on=["s1", "cand"], how="inner").select("s1", "cand")
    return f05_macro(hits, truth, ids)


def adaptive(cand, delta, k_max):
    top = cand.group_by("s1").agg(pl.col("score").max().alias("top"))
    return (cand.join(top, on="s1")
                .filter((pl.col("rank") <= k_max) & (pl.col("score") >= pl.col("top") - delta))
                .drop("top"))


def analyse(country, cand, truth, ids):
    n_s1, n_true = len(ids), truth.height
    lab = cand.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "cand"], how="left") \
              .with_columns(pl.col("y").fill_null(0))
    has_true = truth["s1"].unique()
    print(f"\n==================== {country}: {n_s1:,} S1, {n_true:,} true pairs, "
          f"{n_s1 - len(has_true):,} singletons ====================")

    # 1. K curve --------------------------------------------------------------------------
    rows, prev = [], (0, 0)
    for k in KS:
        ck = lab.filter(pl.col("rank") <= k)
        tp = int(ck["y"].sum())
        per_s1 = (truth.join(ck.filter(pl.col("y") == 1).group_by("s1").len(), on="s1", how="left")
                       .group_by("s1").agg(pl.len().alias("t"), pl.col("len").first().fill_null(0)))
        full_cov = (per_s1["len"] >= per_s1["t"]).mean()
        marg = (tp - prev[0]) / max(ck.height - prev[1], 1)
        f_best, thr = best_f05(cand, truth, ids, k)
        rows.append((k, tp / n_true, full_cov, ck.height / n_s1, tp / max(ck.height, 1), marg,
                     oracle_f05(ck, truth, ids), f_best, thr))
        prev = (tp, ck.height)
    curve = pl.DataFrame(rows, orient="row", schema=[
        "K", "recall", "s1_fully_covered", "cand_per_s1", "precision", "marginal_gain",
        "oracle_F05", "achieved_F05", "best_thr"])
    with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=200):
        print("-- 1. fixed-K curve (marginal_gain = new true matches per extra candidate)")
        print(curve)
    best_f = curve["achieved_F05"].max()
    pick = curve.filter(pl.col("achieved_F05") >= best_f - 0.001).sort("K").row(0, named=True)
    print(f"   -> smallest K within 0.001 of best achieved F0.5 ({best_f:.4f}): "
          f"K={pick['K']} ({pick['cand_per_s1']:.1f} cand/S1, recall {pick['recall']:.2%})")

    # 2. adaptive cut ---------------------------------------------------------------------
    arows = []
    for k_max in (5, 10, 20):
        for delta in (0.1, 0.2, 0.3, 0.5, 0.8):
            a = adaptive(cand, delta, k_max)
            tp = a.join(truth, on=["s1", "cand"], how="inner").height
            f_best, thr = best_f05(a, truth, ids, k_max)
            arows.append((k_max, delta, tp / n_true, a.height / n_s1, oracle_f05(a, truth, ids),
                          f_best, thr))
    adf = pl.DataFrame(arows, orient="row", schema=[
        "K_max", "delta", "recall", "cand_per_s1", "oracle_F05", "achieved_F05", "best_thr"])
    with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=200):
        print("-- 2. adaptive cut: keep candidates with score >= (S1's top score - delta), rank <= K_max")
        print(adf)

    # 3. usefulness -----------------------------------------------------------------------
    ranks = truth.join(cand.select("s1", "cand", "rank"), on=["s1", "cand"], how="left")["rank"]
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 100)]
    dist = {f"{a}-{b}" if a != b else str(a): ((ranks >= a) & (ranks <= b)).sum() / n_true for a, b in bins}
    dist["not retrieved"] = ranks.is_null().sum() / n_true
    print("-- 3a. where true matches rank:", {k: f"{v:.1%}" for k, v in dist.items()})
    top10 = lab.filter(pl.col("rank") <= 10)
    aucs = {c: roc_auc_score(top10["y"], top10[c]) for c in ("score", "name_sim", "addr_sim", "num_agree")}
    print("-- 3b. separability inside top-10 (ROC AUC, 1.0 = perfect):", {k: round(v, 4) for k, v in aucs.items()})
    q = top10.group_by("y").agg(pl.col("score").quantile(0.1).alias("p10"), pl.col("score").median().alias("median"),
                               pl.col("score").quantile(0.9).alias("p90"))
    print("   score quantiles (y=1 true, y=0 false):", q.sort("y").rows())
    if "via" in lab.columns:
        v = lab.filter((pl.col("y") == 1) & (pl.col("rank") <= 10)).group_by("via").len().sort("via")
        print("   true matches in top-10 found via (1=name only, 2=address only, 3=both):", v.rows())

    # 4. singletons and misses ------------------------------------------------------------
    single = [s for s in ids if s not in set(has_true)]
    sc = cand.filter(pl.col("s1").is_in(single) & (pl.col("rank") == 1))
    print(f"-- 4a. singletons: {len(single):,}; their top candidate's score p50/p90/max = "
          f"{sc['score'].median():.2f}/{sc['score'].quantile(0.9):.2f}/{sc['score'].max():.2f}")
    missed = truth.join(lab.filter(pl.col("rank") <= 10), on=["s1", "cand"], how="anti")
    print(f"-- 4b. true pairs missed at K=10: {missed.height:,} ({missed.height / n_true:.1%})")
    return curve, adf, missed


def miss_causes(missed, cand):
    """Classify misses at K=10 using the normalized records."""
    s1 = pl.read_parquet(WORK / "train_s1_norm.parquet").select(pl.col("entity_id").alias("s1"),
                                                               pl.col("name_core").alias("n1"), pl.col("addr").alias("a1"))
    oth = pl.concat([pl.read_parquet(WORK / f"train_s{i}_norm.parquet") for i in (2, 3)]).select(
        pl.col("entity_id").alias("cand"), pl.col("name_core").alias("n2"), pl.col("addr").alias("a2"))
    m = missed.join(s1, on="s1").join(oth, on="cand").join(
        cand.select("s1", "cand", "rank"), on=["s1", "cand"], how="left")
    causes = []
    for n1, n2, a1, a2, rank in m.select("n1", "n2", "a1", "a2", "rank").iter_rows():
        ns = fuzz.ratio(n1.replace(" ", ""), n2.replace(" ", ""))
        c = []
        if not a2.strip():
            c.append("match has no address")
        if ns < 60:
            c.append("name very different (<60)")
        if rank is None:
            c.append("not retrieved (outside both top-50s)")
        else:
            c.append("retrieved but ranked 11-100")
        causes.append(" + ".join(c))
    s = pl.Series(causes).value_counts().sort("count", descending=True)
    with pl.Config(tbl_rows=20, fmt_str_lengths=80):
        print(s)


def main():
    cand = pl.read_parquet(WORK / "val_candidates.parquet")
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth_all = read_pairs().rename({"m": "cand"})
    all_curves, missed_all = [], []
    for country in val["country"].unique().sort():
        ids = val.filter(pl.col("country") == country)["entity_id"].to_list()
        t = truth_all.filter(pl.col("s1").is_in(ids))
        c = cand.filter(pl.col("country") == country)
        curve, adf, missed = analyse(country, c, t, ids)
        all_curves.append(curve.with_columns(pl.lit(country).alias("country")))
        missed_all.append((country, missed, c))
    for country, missed, c in missed_all:
        print(f"\n-- 4c. {country}: causes of misses at K=10")
        miss_causes(missed, c)
    pl.concat(all_curves).write_csv(WORK / "blocking_curve.tsv", separator="\t")


if __name__ == "__main__":
    main()
