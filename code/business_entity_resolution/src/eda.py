"""Exploratory profiling of the training data. Prints summaries only."""
import sys
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[3] / "student_resource" / "dataset"


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    srcs = {i: read_tsv(DATA / split / f"{split}_source{i}.tsv") for i in (1, 2, 3)}
    for i, df in srcs.items():
        print(f"\n== {split} source{i}: {df.height:,} rows")
        print(df["country"].value_counts().sort("count", descending=True))
        for c in ("business_name", "business_address"):
            s = df[c]
            non_ascii = s.str.contains(r"[^\x00-\x7F]").mean()
            has_null = s.str.contains(r"(?i)\bnull\b").mean()
            print(f"  {c}: empty={(s.str.len_chars() == 0).mean():.3%} "
                  f"non-ascii={non_ascii:.2%} 'null'={has_null:.2%} "
                  f"avg_len={s.str.len_chars().mean():.1f}")
        print(f"  dup entity_id: {df.height - df['entity_id'].n_unique()}")
    if split != "train":
        return

    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "")
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}))
    per_s1 = gt.select(
        pl.col("source1_entity_id").alias("s1"),
        pl.col("matched_entity_ids").str.count_matches("S2-").alias("n2"),
        pl.col("matched_entity_ids").str.count_matches("S3-").alias("n3"))
    per_s1 = per_s1.join(srcs[1].select(pl.col("entity_id").alias("s1"), "country"), on="s1", how="left")
    per_s1 = per_s1.with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
    print(f"\n== ground truth: {gt.height:,} S1 rows, {pairs.height:,} positive pairs")
    print("S1 in GT == S1 file:", set(gt["source1_entity_id"]) == set(srcs[1]["entity_id"]))
    print(per_s1.group_by("country").agg(
        pl.len().alias("s1"), (pl.col("n") == 0).mean().alias("singleton_rate"),
        pl.col("n").mean().alias("avg_matches"), pl.col("n2").mean().alias("avg_s2"),
        pl.col("n3").mean().alias("avg_s3"), pl.col("n").max().alias("max")))
    print("match-count distribution:")
    print(per_s1["n"].value_counts().sort("n").head(15))

    # how many S2/S3 records are linked, and are any linked to >1 S1?
    for i in (2, 3):
        p = pairs.filter(pl.col("m").str.starts_with(f"S{i}-"))
        multi = p.group_by("m").len().filter(pl.col("len") > 1).height
        print(f"S{i}: linked={p['m'].n_unique():,}/{srcs[i].height:,} "
              f"({p['m'].n_unique() / srcs[i].height:.1%}), linked to >1 S1: {multi}")

    # country agreement between S1 and its matches
    allc = pl.concat([srcs[2], srcs[3]]).select(pl.col("entity_id").alias("m"), pl.col("country").alias("mc"))
    pc = pairs.join(srcs[1].select(pl.col("entity_id").alias("s1"), "country"), on="s1").join(allc, on="m")
    print(f"country mismatch in positive pairs: {(pc['country'] != pc['mc']).mean():.3%}")

    # eyeball some matched groups
    pl.Config.set_fmt_str_lengths(90)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    rec = pl.concat([srcs[1], srcs[2], srcs[3]]).rename({"entity_id": "m"})
    for country in per_s1["country"].unique().sort():
        ids = per_s1.filter((pl.col("country") == country) & (pl.col("n") >= 2)).sample(3, seed=1)["s1"]
        for s1 in ids:
            members = [s1] + pairs.filter(pl.col("s1") == s1)["m"].to_list()
            print(f"\n-- {country} group {s1}")
            print(rec.filter(pl.col("m").is_in(members)).select("m", "business_name", "business_address"))


if __name__ == "__main__":
    main()
