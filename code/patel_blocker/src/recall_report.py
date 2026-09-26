"""Segmented candidate recall of a train candidate file (full or diagnostic).

Usage: python src/recall_report.py [--cands cands.parquet | cands_diag.parquet]
"""
import polars as pl

import config as C


def main():
    p = C.parser(__doc__)
    p.add_argument("--cands", default="cands.parquet")
    args = p.parse_args()
    log = C.setup(args)
    d = args.work_dir / "train"
    cands = pl.read_parquet(d / args.cands, columns=["q", "s1", "rank", "score"])
    qfile = d / args.cands.replace(".parquet", "_queries.parquet")
    rec = pl.read_parquet(d / "records.parquet", columns=["rid", "src", "country", "addr", "addr_n", "has_indic"])
    queries = pl.read_parquet(qfile) if qfile.exists() else rec.filter(pl.col("src") != 1).select(q="rid")
    gt = pl.read_parquet(d / "gt_pairs.parquet", columns=["s1", "q"]).join(queries, on="q", how="semi")
    n_true = pl.read_parquet(d / "gt_pairs.parquet", columns=["s1"]).group_by("s1").agg(s1_n_true=pl.len())
    qbest = cands.group_by("q").agg(q_best=pl.col("score").max(), q_n=pl.len())
    terc = qbest["q_best"].quantile(1 / 3), qbest["q_best"].quantile(2 / 3)
    g = (gt.join(cands.select("q", "s1", "rank"), on=["q", "s1"], how="left")
         .join(rec.rename({"rid": "q"}), on="q").join(n_true, on="s1").join(qbest, on="q", how="left")
         .with_columns(
             hit=pl.col("rank").is_not_null(),
             addr_missing=pl.col("addr_n") == "",
             addr_noisy=pl.col("addr").str.contains(r"(?i)#|n/a|\bnull\b"),
             single=pl.col("s1_n_true") == 1,
             conf=pl.when(pl.col("q_best").is_null()).then(pl.lit("0 no-cand"))
             .when(pl.col("q_best") < terc[0]).then(pl.lit("1 low")).when(pl.col("q_best") < terc[1])
             .then(pl.lit("2 mid")).otherwise(pl.lit("3 high"))))
    seg = lambda col: g.group_by(col).agg(recall=pl.col("hit").mean().round(4), true_pairs=pl.len(),
                                          missed=(~pl.col("hit")).sum()).sort(col).rows()
    log.info("OVERALL candidate recall %.4f | true pairs %d | missed %d | candidates per query %.2f",
             g["hit"].mean(), g.height, (~g["hit"]).sum(), cands.height / queries.height)
    for col in ("country", "src", "addr_missing", "addr_noisy", "has_indic", "single", "conf"):
        log.info("  by %-12s %s", col, seg(col))
    miss = g.filter(~pl.col("hit"))
    log.info("missed: query had zero candidates %.1f%% | query had candidates but not the true S1 %.1f%%",
             100 * miss["q_n"].is_null().mean(), 100 * miss["q_n"].is_not_null().mean())


if __name__ == "__main__":
    main()
