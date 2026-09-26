"""Stage 1: load raw TSVs -> normalised records.parquet (+ gt_pairs.parquet for train).

Usage: python src/prep.py --split train [--sample-frac 0.05 --work-dir work_smoke]
"""
import polars as pl

import config as C
from data import bucket, load_gt_pairs, load_sources
from normalize import normalize_frame


def sample_world(rec: pl.DataFrame, gt: pl.DataFrame | None, frac: float) -> pl.DataFrame:
    """train: keep S1 ids in the first `frac` of buckets, all their true matches, and the same share of other
    records. test: keep ALL S1 (so outputs stay validator-complete) and a `frac` share of S2/S3."""
    cut = int(frac * 10_000)
    if gt is None:
        return rec.with_columns(b=bucket()).filter((pl.col("src") == 1) | (pl.col("b") < cut)).drop("b")
    keep = rec.with_columns(b=bucket()).filter((pl.col("src") != 1) | (pl.col("b") < cut))
    s1 = keep.filter(pl.col("src") == 1).select(s1_eid="entity_id")
    matched_kept = gt.join(s1, on="s1_eid", how="semi").select(entity_id="q_eid")
    matched_all = gt.select(entity_id="q_eid")
    q = keep.filter(pl.col("src") != 1)
    q = pl.concat([q.join(matched_kept, on="entity_id", how="semi"),
                   q.join(matched_all, on="entity_id", how="anti").filter(pl.col("b") < cut)])
    return pl.concat([keep.filter(pl.col("src") == 1), q]).drop("b")


def _abbrev(a: str, b: str) -> bool:
    """Same first letter and the shorter token is a subsequence of the longer one."""
    short, long_ = sorted((a, b), key=len)
    if short[0] != long_[0]:
        return False
    it = iter(long_)
    return all(ch in it for ch in short)


def learn_subs(rec: pl.DataFrame, pairs: pl.DataFrame, min_count: int, min_share: float, log) -> pl.DataFrame:
    """Learn S2/S3-token -> S1-token substitutions (transliterations, abbreviations) from true pairs.
    q_tok -> s_tok is kept when, among true pairs whose S2/S3 side contains q_tok, at least `min_share`
    have s_tok on the S1 side while lacking q_tok. Learned per country and field; digits never mapped."""
    out = []
    for field in ("name_n", "addr_n"):
        toks = rec.select("rid", "country", t=pl.col(field).str.split(" ").list.unique())
        df = (pairs.join(toks.rename({"rid": "q", "t": "qt"}), on="q")
              .join(toks.select(s1="rid", st="t"), on="s1")
              .with_columns(qo=pl.col("qt").list.set_difference("st"), so=pl.col("st").list.set_difference("qt")))
        total = df.select("country", tok=pl.col("qt")).explode("tok").group_by("country", "tok").agg(total=pl.len())
        cross = (df.select("country", "qo", "so").explode("qo").explode("so")
                 .filter(pl.col("qo").str.len_chars() >= 2, pl.col("so").str.len_chars() >= 2,
                         ~pl.col("qo").str.contains(r"\d"), ~pl.col("so").str.contains(r"\d"))
                 .group_by("country", "qo", "so").agg(n=pl.len()))
        best = (cross.sort(["country", "qo", "n", "so"], descending=[False, False, True, False])
                .group_by(["country", "qo"], maintain_order=True).first()  # deterministic tie-break on the token
                .join(total.rename({"tok": "qo"}), on=["country", "qo"])
                .filter((pl.col("n") >= min_count) & (pl.col("n") >= min_share * pl.col("total"))))
        if field == "addr_n":  # addresses: only abbreviation-shaped pairs (st->street, texas->tx), not north->nc
            best = best.filter([_abbrev(a, b) for a, b in best.select("qo", "so").rows()])
        out.append(best.select("country", pl.lit(field).alias("field"), src="qo", dst="so", n="n", total="total"))
    subs = pl.concat(out)
    log.info("learned %d substitutions: %s", subs.height, subs.group_by("country", "field").len().rows())
    log.info("examples: %s", subs.sort("n", descending=True).head(25).select("field", "src", "dst", "n").rows())
    return subs


def mine_subs_unsupervised(rec: pl.DataFrame, countries, min_rate=2e-4, min_excess_ratio=10.0, min_cover=0.3,
                           max_pos_diff=0.25, log=None) -> pl.DataFrame:
    """Label-free address abbreviations: a token over-represented in S2/S3 vs S1 (same country) is mapped to an
    abbreviation-shaped S1 token whose shortfall in S2/S3 best explains that excess, and which occupies the same
    position relative to numbers (share of occurrences preceded / followed by a number). Corpus statistics only."""
    seq = (rec.filter(pl.col("country").is_in(list(countries)))
           .select("country", side=pl.when(pl.col("src") == 1).then(pl.lit("s1")).otherwise(pl.lit("q")),
                   t=pl.col("addr_n").str.split(" ")).with_row_index("i").explode("t")
           .with_columns(prev_num=pl.col("t").shift(1).over("i").str.contains(r"^\d").fill_null(False),
                         next_num=pl.col("t").shift(-1).over("i").str.contains(r"^\d").fill_null(False)))
    toks = seq.filter((pl.col("t").str.len_chars() >= 1) & ~pl.col("t").str.contains(r"\d"))
    pos = toks.group_by("country", "t").agg(pn=pl.col("prev_num").mean(), nn=pl.col("next_num").mean())
    cnt = toks.group_by("country", "side", "t").len()
    tot = cnt.group_by("country", "side").agg(n=pl.col("len").sum())
    rate = cnt.join(tot, on=["country", "side"]).with_columns(r=pl.col("len") / pl.col("n")).pivot(
        on="side", index=["country", "t"], values="r").fill_null(0.0)
    rate = rate.join(pos, on=["country", "t"])
    short = rate.filter((pl.col("q") >= min_rate) & (pl.col("q") >= min_excess_ratio * pl.col("s1"))).with_columns(
        excess=pl.col("q") - pl.col("s1"))
    long_ = rate.filter(pl.col("s1") > pl.col("q")).with_columns(deficit=pl.col("s1") - pl.col("q"))
    out = []
    for country in countries:
        L = long_.filter(pl.col("country") == country).sort("t").select("t", "deficit", "pn", "nn").rows()
        for a, ex, apn, ann in short.filter(pl.col("country") == country).sort("t").select("t", "excess", "pn", "nn").rows():
            cands = [(b, df) for b, df, bpn, bnn in L if len(b) > len(a) and _abbrev(a, b)
                     and abs(apn - bpn) <= max_pos_diff and abs(ann - bnn) <= max_pos_diff]
            if not cands:
                continue
            b, df = max(cands, key=lambda x: min(ex, x[1]))
            if min(ex, df) >= min_cover * ex:
                out.append((country, "addr_n", a, b, ex, df))
    subs = pl.DataFrame(out, schema=["country", "field", "src", "dst", "excess", "deficit"], orient="row")
    if log:
        log.info("unsupervised substitutions: %d %s", subs.height, subs.sort("excess", descending=True).head(30).select("country", "src", "dst").rows())
    return subs


def apply_subs(rec: pl.DataFrame, subs: pl.DataFrame) -> pl.DataFrame:
    """Rewrite S2/S3 tokens with the learned map (only countries/fields it was learned for)."""
    parts, rest = [], rec
    for (country, field), g in subs.group_by(["country", "field"]):
        mask = (pl.col("src") != 1) & (pl.col("country") == country)
        rec = rec.with_columns(pl.when(mask).then(
            pl.col(field).str.split(" ").list.eval(pl.element().replace(g["src"], g["dst"])).list.join(" "))
            .otherwise(pl.col(field)).alias(field))
    return rec.with_columns(name_c=pl.col("name_n").str.replace_all(" ", ""))


def main():
    p = C.parser(__doc__)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--sample-frac", type=float, default=1.0, help="<1 builds a small consistent world for smoke tests")
    p.add_argument("--drop-s1-frac", type=float, default=0.0,
                   help="train only: delete this share of S1 entities (their S2/S3 matches stay as orphan "
                        "distractors). 0.18 reproduces the test S2+S3-per-S1 ratio (5.7 vs 4.67)")
    p.add_argument("--subs", choices=["none", "learn", "learn-half", "use"], default="none",
                   help="learned token substitutions: learn (train, all S1) | learn-half (train, even-bucket S1 only, "
                        "for honest validation) | use (apply <work>/subs.parquet, e.g. for test)")
    p.add_argument("--mine-unseen", type=int, default=0,
                   help="1: for countries absent from the learned map (e.g. France), mine label-free address "
                        "abbreviations from this split's own S1-vs-S2/S3 token statistics")
    p.add_argument("--subs-min-count", type=int, default=20)
    p.add_argument("--subs-min-share", type=float, default=0.5)
    args = p.parse_args()
    log = C.setup(args)
    out = C.split_dir(args.work_dir, args.split)

    with C.Timer(log, "load"):
        rec = load_sources(args.data_dir, args.split)
        gt = load_gt_pairs(args.data_dir) if args.split == "train" else None
    if args.sample_frac < 1:
        rec = sample_world(rec, gt, args.sample_frac)
    if gt is not None and args.drop_s1_frac > 0:
        drop = rec.filter((pl.col("src") == 1) & (bucket(shift=4) < int(args.drop_s1_frac * 10_000)))["entity_id"]
        rec = rec.filter(~pl.col("entity_id").is_in(drop.implode()))
        gt = gt.filter(~pl.col("s1_eid").is_in(drop.implode()))
        log.info("dropped %d S1 entities; their matches are now orphan distractors", drop.len())
    if gt is not None:
        n_s1 = rec.filter(pl.col("src") == 1).height
        log.info("S2+S3 per S1 in this world: %.2f (test: ~5.7)", (rec.height - n_s1) / n_s1)
    rec = rec.sort(["src", "entity_id"]).with_row_index("rid")  # deterministic row ids
    log.info("records per source: %s", rec.group_by("src").len().sort("src").rows())
    log.info("records per country: %s", rec.group_by("country").len().sort("len").rows())

    with C.Timer(log, "normalise"):
        rec = normalize_frame(rec)

    if gt is not None:
        ids = rec.select("rid", "entity_id")
        pairs = (gt.join(ids.rename({"entity_id": "s1_eid", "rid": "s1"}), on="s1_eid")
                 .join(ids.rename({"entity_id": "q_eid", "rid": "q"}), on="q_eid"))
        dup_q = pairs.height - pairs["q"].n_unique()
        if dup_q:
            raise ValueError(f"{dup_q} S2/S3 records appear in >1 ground-truth list (exclusivity assumption broken)")
        pairs.write_parquet(out / "gt_pairs.parquet")
        log.info("true pairs in world: %d", pairs.height)
    if args.subs in ("learn", "learn-half"):
        lp = pairs.select("s1", "q")
        if args.subs == "learn-half":
            lp = lp.join(rec.filter(bucket(shift=4) % 2 == 0).select(s1="rid"), on="s1", how="semi")
        with C.Timer(log, "learn substitutions"):
            subs = learn_subs(rec, lp, args.subs_min_count, args.subs_min_share, log)
        subs.write_parquet(args.work_dir / "subs.parquet")
    if args.subs != "none":
        subs = pl.read_parquet(args.work_dir / "subs.parquet").select("country", "field", "src", "dst")
        unseen = sorted(set(rec["country"].unique()) - set(subs["country"].unique()))
        if args.mine_unseen and unseen:
            with C.Timer(log, f"mine label-free address substitutions for countries without a learned map: {unseen}"):
                extra = mine_subs_unsupervised(rec, unseen, min_excess_ratio=2.5, log=log)
            extra.write_parquet(out / "subs_unseen.parquet")
            subs = pl.concat([subs, extra.select("country", "field", "src", "dst")])
        with C.Timer(log, "apply substitutions"):
            rec = apply_subs(rec, subs)
    rec.write_parquet(out / "records.parquet")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
