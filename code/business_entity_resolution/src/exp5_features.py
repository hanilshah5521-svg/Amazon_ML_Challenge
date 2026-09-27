"""EXP-5 add-on pair features, from the RAW strings:
  G1 edits  : one flag per atomic edit type (pattern_miner decomposition, 0.976-Hanil branch) + naive-Bayes score
              nb_llr = sum over the pair's edits of log P(edit | true) / P(edit | false)
  G2 words  : supervised scores of added / dropped name words, P(true | word added) (smoothed), min / max per pair
  G3 numbers: premise number inserted, closest-number edit type, "number changed AND name identical / legal changed"
  G4 names  : rarest candidate name token (log document frequency in the split's corpus), handle/web-name overlap with S1

Label hygiene (nb_llr, word scores are learned from labels):
  held-out S1 (val_s1, where F0.5 is reported) -> statistics from the 200k extra S1 only (never their own labels)
  extra S1                                     -> 5-fold out-of-fold within the extra S1 (GroupKFold by S1)
  test                                         -> statistics from all 400k training S1
Output: <WORK>/x5_{train,test}.parquet with (s1, cand, features). Usage: python exp5_features.py {train|test}
"""
import math
import re
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

from common import DATA, WORK, read_pairs
from pattern_miner import LEGALW, addr_edits, name_edits, toks

PREMISE = re.compile(r"^\s*(h\.?\s*no\.?|house\s*no\.?|plot(\s*no\.?)?|flat(\s*no\.?)?|door\s*no\.?|shop\s*no\.?|no\.?|#)\s*(\d+)", re.I)
NUM = re.compile(r"\d+")
WEB = re.compile(r"\.(com|in|net|org|fr|co)\b|^[@#]")
M_SMOOTH = 20


def edit_set(n1, n2, a1, a2):
    e = set(name_edits(n1, n2) + addr_edits(a1, a2))
    return {("N:legal_changed" if x.startswith("N:legal ") else x) for x in e}


def num_feats(n1, n2, a1, a2):
    h1, h2 = NUM.findall(a1), NUM.findall(a2)
    m = PREMISE.match(a2)
    premise = int(bool(m) and m.group(len(m.groups())) not in h1)
    if premise:                                        # strip the inserted premise number, then compare
        h2 = h2[1:]
    s1n, s2n = {str(int(x)) for x in h1}, {str(int(x)) for x in h2}
    changed = int(bool(s1n) and bool(s2n) and s1n != s2n)
    t1, t2 = set(toks(n1)), set(toks(n2))
    same_name = int(t1 == t2)
    legal_changed = int((t1 & LEGALW) != (t2 & LEGALW))
    kind = -1
    if h1 and h2:
        a = str(int(h1[0]))
        b = min((str(int(x)) for x in h2), key=lambda y: (fuzz.ratio(a, y) * -1, abs(len(a) - len(y))))
        if a == b:
            kind = 0 if h1[0] in h2 else 3                                                     # 3: equal only after zero padding
        elif len(a) == len(b) and sum(u != v for u, v in zip(a, b)) == 1:
            kind = 1                                                                           # one digit changed
        elif a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
            kind = 2                                                                           # truncated / extended
        elif sorted(a) == sorted(b):
            kind = 4                                                                           # digits transposed
        elif abs(int(a) - int(b)) <= 3:
            kind = 5
        else:
            kind = 6
    return [premise, kind, changed & same_name, changed & legal_changed]


def one(row):
    n1, n2, a1, a2 = row
    return edit_set(n1, n2, a1, a2), num_feats(n1, n2, a1, a2), set(toks(n1)), set(toks(n2))


def run(rows):
    return [one(r) for r in rows]


def compute_raw(pairs, split):
    """Per-pair edit sets, number features, token sets (raw strings)."""
    need = pl.concat([pairs["s1"], pairs["cand"]]).unique()
    raw = pl.concat([pl.scan_csv(DATA / split / f"{split}_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False)
                     .select("entity_id", "business_name", "business_address").filter(pl.col("entity_id").is_in(need)).collect()
                     for i in (1, 2, 3)]).fill_null("")
    d = (pairs.select("s1", "cand").join(raw.rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1"}), on="s1", how="left")
              .join(raw.rename({"entity_id": "cand", "business_name": "n2", "business_address": "a2"}), on="cand", how="left").fill_null(""))
    rows = d.select("n1", "n2", "a1", "a2").rows()
    with Pool(8) as p:
        out = [x for part in p.imap(run, [rows[i:i + 20_000] for i in range(0, len(rows), 20_000)]) for x in part]
    return d.select("s1", "cand"), out


def doc_freq(split):
    """log document frequency of every name token over all S1/S2/S3 names of the split, normalised by corpus size."""
    cnt, n = Counter(), 0
    for i in (1, 2, 3):
        names = pl.read_csv(DATA / split / f"{split}_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False,
                            columns=["business_name"])["business_name"].fill_null("").to_list()
        n += len(names)
        for s in names:
            cnt.update(set(toks(s)))
    return {k: math.log(v / n) for k, v in cnt.items()}, math.log(1 / n)


class Stats:
    """Label statistics for edits and added/dropped words from a set of labelled pairs."""
    def __init__(self, edits, adds, drops, y):
        self.n1, self.n0 = y.sum(), len(y) - y.sum()
        self.prior = self.n1 / len(y)
        self.e1, self.e0, self.w = Counter(), Counter(), {"add": [Counter(), Counter()], "drop": [Counter(), Counter()]}
        for es, ad, dr, t in zip(edits, adds, drops, y):
            (self.e1 if t else self.e0).update(es)
            self.w["add"][0].update(ad); self.w["drop"][0].update(dr)
            if t:
                self.w["add"][1].update(ad); self.w["drop"][1].update(dr)

    def llr(self, es):
        return sum(math.log((self.e1[e] + 1) / (self.n1 + 2)) - math.log((self.e0[e] + 1) / (self.n0 + 2)) for e in es)

    def word(self, kind, words):
        tot, pos = self.w[kind]
        sc = [(pos[w] + M_SMOOTH * self.prior) / (tot[w] + M_SMOOTH) for w in words]
        return [min(sc), max(sc), sum(tot[w] < 5 for w in words)] if sc else [-1.0, -1.0, 0]


def features(keys, out, stats_for_row, df, df_min, vocab):
    E = {e: i for i, e in enumerate(vocab)}
    flags = np.zeros((len(out), len(vocab)), dtype=np.int8)
    rows = []
    for i, ((es, nf, t1, t2), st) in enumerate(zip(out, stats_for_row)):
        for e in es:
            j = E.get(e)
            if j is not None:
                flags[i, j] = 1
        add, drop = t2 - t1, t1 - t2
        cand_df = min((df.get(t, df_min) for t in t2 if t not in LEGALW), default=0.0)
        rows.append([st.llr(es)] + st.word("add", add) + st.word("drop", drop) + nf + [cand_df])
    cols = ["nb_llr", "w_add_min", "w_add_max", "w_add_rare", "w_drop_min", "w_drop_max", "w_drop_rare",
            "num_premise", "num_kind", "num_chg_same_name", "num_chg_legal_chg", "cand_min_df"]
    d = pl.DataFrame(np.array(rows, dtype=np.float32), schema=cols)
    d = d.with_columns([pl.Series("ed_" + re.sub(r"\W+", "_", e), flags[:, j]) for j, e in enumerate(vocab)])
    return pl.concat([keys, d], how="horizontal")


def handle_overlap(keys, split):
    """Web names / handles: character overlap of the de-TLD'd compact candidate name with the compact S1 name."""
    names = pl.concat([pl.read_csv(DATA / split / f"{split}_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False,
                                   columns=["entity_id", "business_name"]) for i in (1, 2, 3)]).fill_null("")
    nm = dict(zip(names["entity_id"], names["business_name"]))
    out = []
    for s1, c in keys.select("s1", "cand").iter_rows():
        n2 = nm.get(c, "")
        if WEB.search(n2.lower()):
            a = "".join(toks(nm.get(s1, "")))
            b = "".join(toks(re.sub(WEB, "", n2.lower())))
            out.append(fuzz.partial_ratio(a, b) / 100 if a and b else 0.0)
        else:
            out.append(-1.0)
    return keys.with_columns(pl.Series("handle_overlap", out, dtype=pl.Float32))


def main(split):
    df, df_min = doc_freq(split)
    if split == "train":
        tr = (pl.concat([pl.read_parquet(WORK / f, columns=["s1", "cand", "rank"]) for f in ("val_candidates_cut.parquet", "extra_candidates_cut.parquet")])
                .filter(pl.col("rank") <= 8).join(read_pairs().rename({"m": "cand"}).with_columns(pl.lit(1).alias("y")), on=["s1", "cand"], how="left")
                .with_columns(pl.col("y").fill_null(0)))
        keys, out = compute_raw(tr, "train")
        y = keys.join(tr, on=["s1", "cand"], how="left")["y"].to_numpy()
        held = set(pl.read_parquet(WORK / "val_s1.parquet")["entity_id"].to_list())
        s1 = keys["s1"].to_list()
        is_held = np.array([s in held for s in s1])
        edits, adds, drops = [o[0] for o in out], [o[3] - o[2] for o in out], [o[2] - o[3] for o in out]
        sub = lambda idx: Stats([edits[i] for i in idx], [adds[i] for i in idx], [drops[i] for i in idx], y[idx])
        ex = np.where(~is_held)[0]
        st_row = [None] * len(out)
        st_extra = sub(ex)
        for i in np.where(is_held)[0]:
            st_row[i] = st_extra
        for tr_i, te_i in GroupKFold(5).split(ex, groups=np.array(s1)[ex]):
            st = sub(ex[tr_i])
            for i in ex[te_i]:
                st_row[i] = st
        vocab = sorted(e for e, c in Counter(e for es in edits for e in es).items() if c >= 0.001 * len(edits))
        (WORK / "x5_vocab.txt").write_text("\n".join(vocab))
        stats_all = sub(np.arange(len(out)))
        import pickle
        (WORK / "x5_stats.pkl").write_bytes(pickle.dumps(stats_all))
    else:
        import pickle
        vocab = (WORK / "x5_vocab.txt").read_text().split("\n")
        stats_all = pickle.loads((WORK / "x5_stats.pkl").read_bytes())
        allp = pl.concat([pl.read_parquet(WORK / f"test_candidates_cut_{c}.parquet", columns=["s1", "cand", "rank"])
                          .filter(pl.col("rank") <= 8) for c in ("France", "India", "US")])
        parts = []
        for i in range(0, allp.height, 2_000_000):          # chunks: per-pair Python sets are ~3 GB per million pairs
            keys, out = compute_raw(allp.slice(i, 2_000_000), "test")
            parts.append(handle_overlap(features(keys, out, [stats_all] * len(out), df, df_min, vocab), "test"))
            del keys, out
            print(f"   test pairs {min(i + 2_000_000, allp.height):,}/{allp.height:,}", flush=True)
        f = pl.concat(parts)
        f.write_parquet(WORK / f"x5_{split}.parquet")
        print(f"{split}: {f.height:,} pairs, {f.width - 2} EXP-5 features ({len(vocab)} edit flags)", flush=True)
        return
    f = features(keys, out, st_row, df, df_min, vocab)
    f = handle_overlap(f, split)
    f.write_parquet(WORK / f"x5_{split}.parquet")
    print(f"{split}: {f.height:,} pairs, {f.width - 2} EXP-5 features ({len(vocab)} edit flags)", flush=True)


def add_x5(df, split):
    x = _table(split)
    x = x.filter(pl.col("s1").is_in(df["s1"].unique().implode()))
    return df.drop([c for c in x.columns[2:] if c in df.columns]).join(x, on=["s1", "cand"], how="left")


_T = {}
def _table(split):
    if split not in _T:
        _T[split] = pl.read_parquet(WORK / f"x5_{split}.parquet")
    return _T[split]


def x5_features():
    return pl.read_parquet_schema(WORK / "x5_train.parquet").keys() - {"s1", "cand"}


if __name__ == "__main__":
    assert num_feats("a", "a", "H.No 299, 12 MG Road", "12 MG Road")[0] == 0
    assert num_feats("a", "a", "12 MG Road", "Plot 368, 12 MG Road")[0] == 1
    assert num_feats("x sarl", "x sarl", "146 rue a", "148 rue a")[1:3] == [1, 1]
    main(sys.argv[1])
