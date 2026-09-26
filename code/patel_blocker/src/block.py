"""Stage 2: candidate generation (blocking).

For every S2/S3 record, retrieve the S1 records of the *same country* sharing rare tokens
(IDF-weighted inverted index over name tokens, address tokens and compact-name keys), keep the
top-K by score, and prune candidates far below the record's best score. Direction is
S2/S3 -> S1 because every S2/S3 record matches at most one S1 entity.

Usage: python src/block.py --split train [--top-k 5 --rel-min 0.35]
"""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config as C

T_NAME, T_ADDR, T_COMPACT, T_CORE, T_NAME2, T_ADDR2, T_NUMW = 0, 1, 2, 3, 4, 5, 6
ADDR_TYPES = (T_ADDR, T_ADDR2, T_NUMW)
NUMW = 0  # set by main() from --numw


def _num_word(rec: pl.DataFrame) -> pl.DataFrame:
    """First address number paired with every other address word (len>=3): order- and typo-tolerant."""
    toks = rec.select("rid", "country", t=pl.col("addr_n").str.split(" "))
    num = toks.select("rid", n=pl.col("t").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first())
    words = toks.explode("t").filter((pl.col("t").str.len_chars() >= 3) & ~pl.col("t").str.contains(r"\d"))
    return words.join(num.drop_nulls("n"), on="rid").select("rid", "country", tok=pl.concat_str(["n", "t"], separator="|"))


def _uni_bi(rec: pl.DataFrame, col: str):
    """Unigrams (len>=2 or a digit) and adjacent-token bigrams of a normalised text column."""
    ex = (rec.select("rid", "country", tok=pl.col(col).str.split(" ")).with_row_index("i").explode("tok")
          .filter(pl.col("tok") != ""))
    uni = ex.filter((pl.col("tok").str.len_chars() >= 2) | pl.col("tok").str.contains(r"^\d$"))
    bi = ex.with_columns(nxt=pl.col("tok").shift(-1).over("i")).filter(pl.col("nxt").is_not_null()).select(
        "rid", "country", tok=pl.concat_str(["tok", "nxt"], separator=" "))
    return uni.select("rid", "country", "tok"), bi


def tokens(rec: pl.DataFrame, stop: pl.DataFrame) -> pl.DataFrame:
    """Long table (rid, key, t): unique hashed tokens per record, namespaced by country and field."""
    name, name2 = _uni_bi(rec, "name_n")
    addr, addr2 = _uni_bi(rec, "addr_n")
    # core key: name tokens minus data-driven frequent tokens (llc, pvt, group, ...), concatenated
    core = (rec.select("rid", "country", tok=pl.col("name_n").str.split(" ")).with_row_index("i")
            .explode("tok").join(stop, on=["country", "tok"], how="anti")
            .group_by("rid", "country").agg(pl.col("tok").sort_by("i").str.join("")))
    compact = rec.select("rid", "country", tok="name_c")
    parts = [name.with_columns(t=pl.lit(T_NAME, pl.UInt8)), addr.with_columns(t=pl.lit(T_ADDR, pl.UInt8)),
             compact.with_columns(t=pl.lit(T_COMPACT, pl.UInt8)), core.with_columns(t=pl.lit(T_CORE, pl.UInt8)),
             name2.with_columns(t=pl.lit(T_NAME2, pl.UInt8)), addr2.with_columns(t=pl.lit(T_ADDR2, pl.UInt8))]
    if NUMW:
        parts.append(_num_word(rec).with_columns(t=pl.lit(T_NUMW, pl.UInt8)))
    out = pl.concat([p.select("rid", "country", "tok", "t") for p in parts])
    out = out.filter(pl.col("tok").str.len_chars() >= 1)
    key = pl.concat_str([pl.col("country"), pl.col("t").cast(pl.String), pl.col("tok")], separator="\x1f").hash(C.SEED)
    return out.select("rid", key=key, t="t").unique(["rid", "key"])


def build_index(s1: pl.DataFrame, max_df_frac: float, min_max_df: int, stop_frac: float, log):
    n_c = s1.group_by("country").agg(n=pl.len())
    name_df = (s1.select("rid", "country", tok=pl.col("name_n").str.split(" ")).explode("tok").unique(["rid", "tok"])
               .group_by("country", "tok").agg(df=pl.len()).join(n_c, on="country"))
    stop = name_df.filter(pl.col("df") > stop_frac * pl.col("n")).select("country", "tok")
    log.info("core-key stop tokens per country: %s", stop.group_by("country").len().rows())
    tok = tokens(s1, stop)
    tok = tok.join(s1.select("rid", "country"), on="rid")
    dfs = tok.group_by("key").agg(df=pl.len())
    tok = tok.join(dfs, on="key").join(n_c, on="country").with_columns(
        cap=pl.max_horizontal(pl.lit(min_max_df), (pl.col("n") * max_df_frac).cast(pl.Int64)))
    kept = tok.filter(pl.col("df") <= pl.col("cap"))
    log.info("S1 index: %d token rows kept of %d", kept.height, tok.height)
    idx = kept.select(pl.col("rid").alias("s1"), "key", "t",
                      idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32))
    return idx, stop


def retrieve(q_tok: pl.DataFrame, idx: pl.DataFrame, strings: pl.DataFrame, a) -> pl.DataFrame:
    """IDF retrieval; low-confidence records (best IDF score < deep_below) get a deep pool that is
    re-ranked by string similarity (rr = name token_set + address token_set + idf/5)."""
    hit = q_tok.join(idx, on="key").with_columns(pl.col("idf").cast(pl.Float64))
    # float sums inside a multithreaded group_by depend on addition order: round so ranks are reproducible
    r = lambda e: e.round(4).cast(pl.Float32)
    agg = hit.group_by("q", "s1").agg(
        score=r(pl.col("idf").sum()),
        s_name=r(pl.col("idf").filter(pl.col("t").is_in([T_NAME, T_NAME2])).sum()),
        s_addr=r(pl.col("idf").filter(pl.col("t").is_in(ADDR_TYPES)).sum()),
        s_key=r(pl.col("idf").filter(pl.col("t").is_in([T_COMPACT, T_CORE])).sum()),
        n_shared=pl.len().cast(pl.UInt16))
    # deterministic ranks: ties broken by S1 row id (group_by output order is not deterministic)
    agg = agg.sort(["q", "score", "s1"], descending=[False, True, False]).with_columns(
        idf_rank=pl.int_range(1, pl.len() + 1).over("q").cast(pl.UInt16),
        deep=(pl.col("score").max().over("q") < a.deep_below))
    if a.deep_k <= 0:  # legacy: IDF order only
        return agg.filter((pl.col("idf_rank") <= a.top_k) & (pl.col("score") >= a.rel_min * pl.col("score").max().over("q"))
                          ).with_columns(rank=pl.col("idf_rank"), rr=pl.col("score"))
    pool = agg.filter(pl.col("idf_rank") <= pl.when(pl.col("deep")).then(a.deep_k).otherwise(a.top_k))
    x = pool.join(strings.rename({"rid": "q", "name_n": "qn", "addr_n": "qa", "name_c": "qc"}), on="q").join(
        strings.rename({"rid": "s1", "name_n": "sn", "addr_n": "sa", "name_c": "sc"}), on="s1")
    sim = lambda u, v, f=fuzz.token_set_ratio: pl.Series(cpdist(x[u].to_list(), x[v].to_list(), scorer=f, workers=-1))
    name_sim = pl.max_horizontal(sim("qn", "sn"), sim("qc", "sc", fuzz.partial_ratio) if a.rr_compact else 0)
    x = x.with_columns(rr=(name_sim + sim("qa", "sa") + pl.col("score") / 5).round(3).cast(pl.Float32)).drop(
        "qn", "qa", "qc", "sn", "sa", "sc")
    x = x.sort(["q", "rr", "s1"], descending=[False, True, False]).with_columns(
        rank=pl.int_range(1, pl.len() + 1).over("q").cast(pl.UInt16), best=pl.col("rr").max().over("q"))
    keep = (pl.col("rank") <= pl.when(pl.col("deep")).then(a.top_k_deep).otherwise(a.top_k)) & (pl.col("rr") >= a.rel_min * pl.col("best"))
    return x.filter(keep).drop("best")


def main():
    p = C.parser(__doc__)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--top-k", type=int, default=C.TOP_K, help="candidates kept for confident records")
    p.add_argument("--rel-min", type=float, default=C.REL_MIN, help="keep if rr >= rel_min * record's best rr")
    p.add_argument("--deep-below", type=float, default=C.DEEP_BELOW, help="best IDF score below this = low confidence")
    p.add_argument("--deep-k", type=int, default=C.DEEP_K, help="IDF pool re-ranked for low-confidence records (0 = legacy)")
    p.add_argument("--top-k-deep", type=int, default=C.TOP_K_DEEP, help="candidates kept for low-confidence records")
    p.add_argument("--rr-compact", type=int, default=0, help="1: rerank name similarity = max(token_set, compact partial)")
    p.add_argument("--numw", type=int, default=C.NUMW, help="1: add (first address number x address word) keys")
    p.add_argument("--max-df-frac", type=float, default=C.MAX_DF_FRAC)
    p.add_argument("--min-max-df", type=int, default=C.MIN_MAX_DF)
    p.add_argument("--stop-frac", type=float, default=C.STOP_FRAC)
    p.add_argument("--query-chunk", type=int, default=C.QUERY_CHUNK)
    p.add_argument("--query-frac", type=float, default=1.0,
                   help="diagnostic only: query a sample of S2/S3 against the FULL S1 index "
                        "(realistic density, cheap). Writes cands_diag.parquet, never cands.parquet")
    args = p.parse_args()
    global NUMW
    NUMW = args.numw
    log = C.setup(args)
    d = C.split_dir(args.work_dir, args.split)
    rec = pl.read_parquet(d / "records.parquet", columns=["rid", "src", "country", "name_n", "addr_n", "name_c"])
    strings = rec.select("rid", "name_n", "addr_n", "name_c")
    s1, qs = rec.filter(pl.col("src") == 1), rec.filter(pl.col("src") != 1)
    if args.query_frac < 1:
        qs = qs.sample(fraction=args.query_frac, seed=C.SEED)

    with C.Timer(log, "index S1"):
        idx, stop = build_index(s1, args.max_df_frac, args.min_max_df, args.stop_frac, log)
    parts = []
    with C.Timer(log, f"retrieve for {qs.height} S2/S3 records"):
        for start in range(0, qs.height, args.query_chunk):
            chunk = qs.slice(start, args.query_chunk)
            q_tok = tokens(chunk, stop).rename({"rid": "q"}).drop("t")
            parts.append(retrieve(q_tok, idx, strings, args))
            log.info("  %d/%d queries, %d candidate pairs so far", min(start + args.query_chunk, qs.height),
                     qs.height, sum(x.height for x in parts))
    cands = pl.concat(parts).sort(["s1", "rr", "q"], descending=[False, True, False])
    cands = cands.with_columns(
        s1_rank=pl.int_range(1, pl.len() + 1).over("s1").cast(pl.UInt32)).sort(["q", "s1"])
    cands = cands.with_columns(
        q_n_cand=pl.len().over("q").cast(pl.UInt16),
        q_gap=(pl.col("score").max().over("q") - pl.col("score")).cast(pl.Float32),
        q_rr_gap=(pl.col("rr").max().over("q") - pl.col("rr")).cast(pl.Float32),
        s1_n_q=pl.len().over("s1").cast(pl.UInt32))
    diag = args.query_frac < 1
    cands.write_parquet(d / ("cands_diag.parquet" if diag else "cands.parquet"))
    if diag:
        qs.select(q="rid").write_parquet(d / "cands_diag_queries.parquet")

    n_s1 = s1.height * args.query_frac
    log.info("candidate pairs: %d | per S1 (mean over all S1%s): %.2f | S2/S3 records with >=1 candidate: %.2f%%",
             cands.height, ", extrapolated" if diag else "", cands.height / n_s1,
             100 * cands["q"].n_unique() / qs.height)
    gt_path = d / "gt_pairs.parquet"
    if gt_path.exists():
        gt = pl.read_parquet(gt_path, columns=["s1", "q"]).join(qs.select(q="rid"), on="q", how="semi")
        hit = gt.join(cands.select("s1", "q", "rank"), on=["s1", "q"], how="left")["rank"].fill_null(10_000)
        rec_k = {k: float((hit <= k).mean()) for k in range(1, max(args.top_k, args.top_k_deep) + 1)}
        log.info("blocking recall of true pairs @rank: %s", {k: round(v, 4) for k, v in rec_k.items()})
        C.save_json({"pairs": cands.height, "pairs_per_s1": cands.height / n_s1, "recall_at_rank": rec_k,
                     "args": vars(args)}, d / ("block_report_diag.json" if diag else "block_report.json"))


if __name__ == "__main__":
    main()
