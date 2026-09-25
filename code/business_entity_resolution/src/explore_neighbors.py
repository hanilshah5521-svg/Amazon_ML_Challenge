"""Pattern explorer: for sampled S1 records, show the most similar S2/S3 addresses
from the full same-country pool (entire split), with ground truth marks on train.

Usage: python explore_neighbors.py <train|test> [n_queries=50] [k=10]
Writes work/explore/<split>_<country>.md and .tsv
"""
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from common import WORK, read_pairs, read_source

OUT = WORK / "explore"
OUT.mkdir(exist_ok=True)


def load_norm(split, i, country):
    return pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet").filter(pl.col("country") == country)


def raw_lookup(split, ids):
    """Raw name/address for the given entity ids (only the rows we display)."""
    ids = pl.Series(list(ids))
    frames = [read_source(split, i).filter(pl.col("entity_id").is_in(ids)) for i in (1, 2, 3)]
    return {r[0]: (r[1], r[2]) for r in pl.concat(frames).rows()}


def explore(split, country, n, k, truth):
    q = load_norm(split, 1, country).sample(n, seed=7)
    pool = pl.concat([load_norm(split, 2, country), load_norm(split, 3, country)])
    vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
    P = vec.fit_transform(pool["addr"].to_list())
    Q = vec.transform(q["addr"].to_list())
    S = (Q @ P.T).tocsr()
    pid = pool["entity_id"].to_numpy()
    pname = pool["name_core"].to_list()
    rows, show = [], set(q["entity_id"])
    for qi, (s1, s1_core) in enumerate(q.select("entity_id", "name_core").iter_rows()):
        row = S.getrow(qi)
        order = np.argsort(-row.data)
        cols, vals = row.indices[order], row.data[order]
        rank = {pid[c]: r + 1 for r, c in enumerate(cols)}
        true_ids = truth.get(s1, set())
        for r, (c, v) in enumerate(zip(cols[:k], vals[:k]), 1):
            rows.append((s1, r, pid[c], float(v), fuzz.token_set_ratio(s1_core, pname[c]),
                         "TRUE" if pid[c] in true_ids else ""))
        for t in true_ids:  # true matches that did not make the top-k
            if rank.get(t, 10**9) > k:
                rows.append((s1, rank.get(t, None), t, None, None, "MISSED"))
    show.update(r[2] for r in rows)
    raw = raw_lookup(split, show)
    norm = {r[0]: (r[1], r[3]) for r in pl.concat([q, pool]).filter(
        pl.col("entity_id").is_in(pl.Series(list(show)))).select("entity_id", "name", "name_core", "addr").rows()}
    df = pl.DataFrame(rows, schema=["s1", "rank", "cand", "addr_sim", "name_sim", "label"], orient="row")
    df = df.with_columns(
        pl.col("cand").map_elements(lambda x: raw.get(x, ("", ""))[0], return_dtype=pl.Utf8).alias("raw_name"),
        pl.col("cand").map_elements(lambda x: raw.get(x, ("", ""))[1], return_dtype=pl.Utf8).alias("raw_addr"),
        pl.col("cand").map_elements(lambda x: norm.get(x, ("", ""))[1], return_dtype=pl.Utf8).alias("norm_addr"))
    df.write_csv(OUT / f"{split}_{country}.tsv", separator="\t")

    lines = [f"# {split} / {country}: {n} S1 records vs full S2+S3 pool ({pool.height:,} records)",
             f"Ranked by address word TF-IDF cosine. name_sim = token_set_ratio of name cores.", ""]
    if truth:
        tot = sum(len(truth.get(s, ())) for s in q["entity_id"])
        hit = df.filter(pl.col("label") == "TRUE").height
        lines += [f"**True matches in top-{k}: {hit}/{tot} ({hit / max(tot, 1):.1%})**", ""]
    for s1 in q["entity_id"]:
        nm, ad = raw[s1]
        lines += [f"## {s1}  |  {nm}  |  {ad}", f"`norm: {norm[s1][1]}`", "",
                  "| rank | cand | addr_sim | name_sim | label | name | address |",
                  "|---|---|---|---|---|---|---|"]
        for r in df.filter(pl.col("s1") == s1).iter_rows(named=True):
            sim = "" if r["addr_sim"] is None else f"{r['addr_sim']:.2f}"
            ns = "" if r["name_sim"] is None else f"{r['name_sim']:.0f}"
            lab = {"TRUE": "✅", "MISSED": "❌ missed"}.get(r["label"], "")
            rk = r["rank"] if r["rank"] is not None else "no overlap"
            lines.append(f"| {rk} | {r['cand']} | {sim} | {ns} | {lab} | {r['raw_name']} | {r['raw_addr']} |")
        lines.append("")
    (OUT / f"{split}_{country}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{split}/{country}: pool {pool.height:,} -> {OUT / f'{split}_{country}.md'}", flush=True)


def main():
    split = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    truth = {}
    if split == "train":
        for s1, m in read_pairs().iter_rows():
            truth.setdefault(s1, set()).add(m)
    countries = pl.read_parquet(WORK / f"{split}_s1_norm.parquet")["country"].unique().sort().to_list()
    for country in countries:
        explore(split, country, n, k, truth)


if __name__ == "__main__":
    main()
