"""Raw TSV loading, ground-truth parsing, sampling, submission writing."""
from pathlib import Path

import polars as pl


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: a few hundred rows contain literal '"' characters.
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df


def load_sources(data_dir: Path, split: str) -> pl.DataFrame:
    """All three sources stacked, with src (1/2/3). Column names follow the official schema."""
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(data_dir / split / f"{split}_source{s}.tsv")
        expected = ["entity_id", "business_name", "business_address", "country"]
        if df.columns != expected:
            raise ValueError(f"{split}_source{s}.tsv: columns {df.columns} != {expected}")
        bad = df.filter(~pl.col("entity_id").str.starts_with(f"S{s}-")).height
        if bad:
            raise ValueError(f"{split}_source{s}.tsv: {bad} ids without S{s}- prefix")
        frames.append(df.with_columns(src=pl.lit(s, pl.UInt8)))
    return pl.concat(frames).rename({"business_name": "name", "business_address": "addr"}).with_columns(
        pl.col("name").fill_null(""), pl.col("addr").fill_null(""), pl.col("country").fill_null(""))


def load_gt_pairs(data_dir: Path) -> pl.DataFrame:
    """(s1_eid, q_eid) true pairs from train_ground_truth.tsv."""
    gt = read_tsv(data_dir / "train" / "train_ground_truth.tsv")
    return (gt.rename({"source1_entity_id": "s1_eid", "matched_entity_ids": "q_eid"})
            .with_columns(pl.col("q_eid").str.split(",")).explode("q_eid")
            .filter(pl.col("q_eid").is_not_null() & (pl.col("q_eid") != "")))


def bucket(col: str = "entity_id", shift: int = 0) -> pl.Expr:
    """Stable 0..9999 bucket from the numeric part of an id (ids are random integers).
    Different `shift` values use different digits, giving independent buckets."""
    return pl.col(col).str.slice(3).cast(pl.Int64, strict=False).fill_null(0) // (10 ** shift) % 10_000


def write_outputs(rec: pl.DataFrame, matches: pl.DataFrame, cands: pl.DataFrame, out_dir: Path, log):
    """matches/cands: (s1, q) rid pairs -> matching_results.tsv + candidate_pairs.tsv with every S1 row."""
    eid = rec.select("rid", "entity_id")
    to_eid = lambda df: (df.select("s1", "q").join(eid.rename({"rid": "s1", "entity_id": "s1_eid"}), on="s1")
                         .join(eid.rename({"rid": "q", "entity_id": "q_eid"}), on="q"))
    s1_ids = rec.filter(pl.col("src") == 1)["entity_id"]
    m = write_id_lists(to_eid(matches), s1_ids, out_dir / "matching_results.tsv", "matched_entity_ids")
    c = write_id_lists(to_eid(cands), s1_ids, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    log.info("wrote %s: %d S1 rows, %d with matches, %d matched ids; candidates: %d ids (%.2f per S1)",
             out_dir, m.height, (m["matched_entity_ids"] != "").sum(), matches.height, cands.height,
             cands.height / max(1, m.height))


def write_id_lists(pairs: pl.DataFrame, s1_ids: pl.Series, path: Path, col: str):
    """One row per S1 id (empty list allowed), comma-joined S2/S3 ids, tab-separated, no quoting."""
    lists = pairs.unique(["s1_eid", "q_eid"]).sort(["s1_eid", "q_eid"]).group_by("s1_eid").agg(
        pl.col("q_eid").str.join(",").alias(col))
    out = (pl.DataFrame({"s1_eid": s1_ids}).join(lists, on="s1_eid", how="left")
           .with_columns(pl.col(col).fill_null("")).rename({"s1_eid": "source1_entity_id"}))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out
