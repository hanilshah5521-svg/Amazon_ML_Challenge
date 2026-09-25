"""Evaluation and output helpers shared by the matcher and the analysis scripts:
macro F0.5 scorer (README formula), match decision rule (threshold + one owner per S2/S3
record), and the submission TSV writer."""
import polars as pl

from common import ROOT

OUT = ROOT / "output"


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


def write_tsv(df, col, path, s1_ids):
    agg = df.group_by("s1").agg(pl.col("cand").unique().sort().str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
