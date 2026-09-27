"""Build text pairs for the cross-encoder from the team's own K=8 candidate cuts (local machine has no Kaggle
access to hanilshah/amlc26-feat-{train,test}, so this uses the equivalent local artifacts: work/val_candidates_cut.parquet
(200k held-out S1), work/extra_candidates_cut.parquet + extra2_candidates_cut.parquet (600k train-only S1), and
work/test_candidates_cut_{France,India,US}.parquet). Entity IDs are the same raw dataset IDs everywhere.

Text per record: f"{business_name} | {business_address}" (NFKC-normalised, original script kept, "" for missing).

Usage: python prep_data.py
Writes work/ce/{train_pairs,val_pairs}.parquet (s1, cand, text_s1, text_cand, y) and
       work/ce/test_pairs_{France,India,US}.parquet (s1, cand, text_s1, text_cand).
"""
import unicodedata

import polars as pl

from common import CE, DATA, WORK


def raw_text(split, i, ids):
    df = (pl.scan_csv(DATA / split / f"{split}_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False)
            .select("entity_id", "business_name", "business_address").filter(pl.col("entity_id").is_in(ids)).collect()
            .fill_null(""))
    nfkc = lambda s: unicodedata.normalize("NFKC", s)
    return df.with_columns(
        (pl.col("business_name").map_elements(nfkc, return_dtype=pl.Utf8) + " | " +
         pl.col("business_address").map_elements(nfkc, return_dtype=pl.Utf8)).alias("text"))


def texts_for(split, cand_df):
    """cand_df: (s1, cand) pairs of one split. Returns cand_df with text_s1, text_cand added."""
    s1_ids = cand_df["s1"].unique()
    cand_ids = cand_df["cand"].unique()
    s1_text = raw_text(split, 1, s1_ids).select(pl.col("entity_id").alias("s1"), pl.col("text").alias("text_s1"))
    c2 = cand_ids.filter(cand_ids.str.starts_with("S2"))
    c3 = cand_ids.filter(cand_ids.str.starts_with("S3"))
    cand_text = pl.concat([raw_text(split, 2, c2), raw_text(split, 3, c3)]).select(
        pl.col("entity_id").alias("cand"), pl.col("text").alias("text_cand"))
    return cand_df.join(s1_text, on="s1", how="left").join(cand_text, on="cand", how="left")


def labels(cand_df):
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "cand"}).with_columns(pl.lit(1).alias("y")))
    return cand_df.join(ex, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0))


def main():
    train = pl.concat([pl.read_parquet(WORK / f"{f}_candidates_cut.parquet").select("s1", "cand")
                       for f in ("extra", "extra2")])
    val = pl.read_parquet(WORK / "val_candidates_cut.parquet").select("s1", "cand")
    print(f"train pairs {train.height:,} ({train['s1'].n_unique():,} S1), val pairs {val.height:,}", flush=True)
    train = labels(texts_for("train", train))
    val = labels(texts_for("train", val))
    train.write_parquet(CE / "train_pairs.parquet")
    val.write_parquet(CE / "val_pairs.parquet")
    print(f"train saved: {train.height:,} rows, {train['y'].mean():.1%} positive", flush=True)
    print(f"val saved: {val.height:,} rows, {val['y'].mean():.1%} positive", flush=True)
    for country in ("France", "India", "US"):
        t = pl.read_parquet(WORK / f"test_candidates_cut_{country}.parquet").select("s1", "cand")
        t = texts_for("test", t)
        t.write_parquet(CE / f"test_pairs_{country}.parquet")
        print(f"test {country} saved: {t.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
