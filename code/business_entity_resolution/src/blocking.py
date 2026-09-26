"""Candidate generation (blocking).

Per country, two retrievers over the S2+S3 pool:
  * name:    char 3-gram TF-IDF on name_core with spaces removed (typos, reordering, websites)
  * address: word TF-IDF on the normalized address
Their top-K lists are merged and re-ranked by a cheap combined score; the final top-K per S1
record is the candidate set.

Usage:
  python blocking.py eval [n_val=20000]   # recall vs K on held-out train S1, full train pool
  python blocking.py test K               # write work/test_candidates.parquet
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import WORK, read_pairs

K_NAME, K_ADDR, K_NOSTATE = 30, 30, 10
MAX_DF = 1.0            # v2: per-state search is fast enough to keep common tokens (full recall)


def _state_of(addrs, country):
    """Canonical state token of each normalized address ("" if none). True matches always
    share the state when both sides have one (0 exceptions in 300k train pairs)."""
    from normalize import STATE_COMPONENT
    tokens = set(STATE_COMPONENT.get(country, {}).values())
    out = []
    for a in addrs:
        out.append(next((t for t in a.split() if t in tokens), ""))
    return np.array(out)


def _topk_by_state(Q, P, q_state, p_state, k):
    """Top-k within each state, plus every query against pool records that have no state."""
    qi_all, pj_all = [], []
    nostate = np.where(p_state == "")[0]
    for st in np.unique(q_state):
        qi = np.where(q_state == st)[0]
        pi = np.arange(len(p_state)) if st == "" else np.where(p_state == st)[0]
        if len(pi) == 0:
            continue
        a, b, _ = _topk(Q[qi], P[pi], k)
        qi_all.append(qi[a]); pj_all.append(pi[b])
    if len(nostate):
        a, b, _ = _topk(Q, P[nostate], K_NOSTATE)
        qi_all.append(a); pj_all.append(nostate[b])
    return np.concatenate(qi_all), np.concatenate(pj_all)


def load(split, i, country):
    return pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet").filter(pl.col("country") == country)


def _vectorize(pool_text, query_text, **kw):
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw)
    return vec.fit_transform(pool_text), vec.transform(query_text)


def _topk(Q, P, k, chunk=100_000):
    """Top-k cosine neighbours of each query row -> (query_idx, pool_idx, score)."""
    PT = P.T.tocsr()
    qi, pj, sc = [], [], []
    for s in range(0, Q.shape[0], chunk):
        M = sp_matmul_topn(Q[s:s + chunk], PT, top_n=k, threshold=1e-6, n_threads=16).tocoo()
        qi.append(M.row + s); pj.append(M.col); sc.append(M.data)
    return np.concatenate(qi), np.concatenate(pj), np.concatenate(sc)


def _rowdot(A, B, ai, bj, chunk=2_000_000):
    """Cosine for explicit pairs (rows of L2-normalized TF-IDF matrices)."""
    out = np.empty(len(ai), dtype=np.float32)
    for s in range(0, len(ai), chunk):
        out[s:s + chunk] = np.asarray(A[ai[s:s + chunk]].multiply(B[bj[s:s + chunk]]).sum(1)).ravel()
    return out



def block_country(q, pool, k_final):
    t = [time.time()]
    q_name = [s.replace(" ", "") for s in q["name_core"].to_list()]
    p_name = [s.replace(" ", "") for s in pool["name_core"].to_list()]
    # max_df drops tokens present in >2% of the pool: they carry no identity and are what
    # makes the sparse product slow (every record shares them).
    Pn, Qn = _vectorize(p_name, q_name, analyzer="char", ngram_range=(3, 3), min_df=2, max_df=MAX_DF)
    Pa, Qa = _vectorize(pool["addr"].to_list(), q["addr"].to_list(), token_pattern=r"\S+",
                        min_df=2, max_df=MAX_DF)
    t.append(time.time())
    country = q["country"][0]
    q_state, p_state = _state_of(q["addr"].to_list(), country), _state_of(pool["addr"].to_list(), country)
    n_qi, n_pj = _topk_by_state(Qn, Pn, q_state, p_state, K_NAME)
    a_qi, a_pj = _topk_by_state(Qa, Pa, q_state, p_state, K_ADDR)
    t.append(time.time())
    pairs = np.unique(np.concatenate([
        np.stack([n_qi, n_pj], 1), np.stack([a_qi, a_pj], 1)]), axis=0)
    qi, pj = pairs[:, 0], pairs[:, 1]
    # Which retriever(s) produced each pair: 1 = name, 2 = address, 3 = both.
    width = np.int64(pool.height)
    key = qi.astype(np.int64) * width + pj
    via = (np.isin(key, n_qi.astype(np.int64) * width + n_pj).astype(np.int8)
           + 2 * np.isin(key, a_qi.astype(np.int64) * width + a_pj).astype(np.int8))
    name_sim = _rowdot(Qn, Pn, qi, pj)
    addr_sim = _rowdot(Qa, Pa, qi, pj)
    t.append(time.time())
    # House-number / legal-form checks run later (pair_features) on the much smaller kept set.
    cand = pl.DataFrame({
        "s1": q["entity_id"].to_numpy()[qi], "cand": pool["entity_id"].to_numpy()[pj],
        "name_sim": name_sim, "addr_sim": addr_sim, "score": name_sim + addr_sim, "via": via})
    cand = (cand.with_columns(pl.col("score").rank("ordinal", descending=True).over("s1").alias("rank"))
                .filter(pl.col("rank") <= k_final))
    t.append(time.time())
    print(f"   {q.height:,} queries x {pool.height:,} pool -> union {len(qi):,} pairs, kept {cand.height:,} "
          f"| vectorize {t[1] - t[0]:.0f}s, top-k {t[2] - t[1]:.0f}s, pair sims {t[3] - t[2]:.0f}s, "
          f"rank {t[4] - t[3]:.0f}s", flush=True)
    return cand


def evaluate(n_val):
    truth = read_pairs().rename({"m": "cand"})
    s1_all = pl.read_parquet(WORK / "train_s1_norm.parquet")
    val = s1_all.sample(n_val, seed=42)
    rows, saved = [], []
    for country in val["country"].unique().sort():
        q = val.filter(pl.col("country") == country)
        pool = pl.concat([load("train", 2, country), load("train", 3, country)])
        cand = block_country(q, pool, k_final=40)
        saved.append(cand.with_columns(pl.lit(country).alias("country")))
        t = truth.filter(pl.col("s1").is_in(q["entity_id"]))
        hit = t.join(cand, on=["s1", "cand"], how="left")
        n_true = t.height
        for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40):
            rec = hit.filter(pl.col("rank") <= k).height / n_true
            per = cand.filter(pl.col("rank") <= k).height / q.height
            rows.append((country, k, rec, per))
    df = pl.DataFrame(rows, schema=["country", "K", "recall", "cand_per_s1"], orient="row")
    with pl.Config(tbl_rows=40):
        print(df)
    df.write_csv(WORK / "blocking_eval.tsv", separator="\t")
    pl.concat(saved).write_parquet(WORK / "val_candidates.parquet")
    val.select("entity_id", "country").write_parquet(WORK / "val_s1.parquet")


def run_test(k_final):
    s1 = pl.read_parquet(WORK / "test_s1_norm.parquet")
    parts = []
    for country in s1["country"].unique().sort():
        part = WORK / f"test_candidates_{country}.parquet"
        parts.append(part)
        if part.exists():                      # resume: each country is saved as it finishes
            print(country, "already done", flush=True)
            continue
        print(country, flush=True)
        pool = pl.concat([load("test", 2, country), load("test", 3, country)])
        block_country(s1.filter(pl.col("country") == country), pool, k_final).write_parquet(part)
    pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(WORK / "test_candidates.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    else:
        run_test(int(sys.argv[2]))
