"""Feature set v2: everything the 0.915 model could not see (see documentation/experiments.md).

  legal / entity type   type added / dropped / changed, Private added/dropped, identical legal words
  compound numbers      "10-2-10" kept whole; exact / subset / one part changed / truncation /
                        within 3 / range contains / first-number distance
  name word categories  IDF-weighted shared and unmatched distinctive words, generic words,
                        first word equal, invented trade name, acronym, website name, place word
  address parts         state agree/conflict/missing, IDF-weighted street/locality words
  namesakes / crowding  how many S1 and pool records share the name key or the address
  group (no model yet)  same-name candidates of this S1 and which has the best address
The existing 24 features (lgbm_matcher.build_features) are kept.
"""
import json
import math
import os
import re
from functools import partial
from multiprocessing import Pool

import numpy as np
import polars as pl

from common import DATA, WORK
from lgbm_matcher import build_features
from normalize import ADDR_COMMON, GENERIC, STATE_COMPONENT

ENT = {"pvt": "private", "ltd": "limited", "plc": "limited", "opc": "opc", "llp": "llp", "llc": "llc",
       "inc": "inc", "corp": "corp", "co": "co", "pc": "pc", "pllc": "pllc", "lp": "lp",
       "trust": "trust", "society": "society", "partnership": "partnership", "association": "association",
       "foundation": "foundation", "public": "public", "proprietorship": "prop", "sarl": "sarl",
       "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "ei": "ei",
       "ets": "ets", "ste": "ste"}
NUM = re.compile(r"\d+(?:[-/]\d+)*")
SEP = re.compile(r"\s*([-/])\s*")
STREET_WORDS = set(ADDR_COMMON.values()) | {"rue", "no", "fl", "col", "ngr"}
PY_FEATURES = ["ent_added", "ent_dropped", "ent_changed", "private_diff", "legal_equal",
               "cnum_exact", "cnum_subset", "cnum_shared", "cnum_n1", "cnum_n2", "cnum_part_changed",
               "cnum_trunc", "cnum_close", "cnum_range_in", "first_num_logdiff", "cnum_off_4_20", "cnum_unrelated",
               "first_tok_equal", "invented_name", "acronym", "web_name", "web_contains", "place_in_name",
               "decoy_added", "noise_added", "decoy_dropped", "n_decoy_added",
               # generator patterns (pattern_miner.py): legal swaps never seen in true pairs, true-only evidence
               "legal_trans_logcount", "legal_trans_unseen", "alias_marker", "tokens_identical", "reordered_only",
               "addr_parts_reordered"]
ALIAS_RE = re.compile(r"(?<![a-z])(?:f */? *k */? *a|formerly|d */? *b */? *a|doing business as|a */ *k */ *a)(?![a-z])", re.I)
LEGAL_TRANS_PATH = WORK / os.environ.get("V2_NAME", "v2") / "legal_transitions.json"
_LT = {}


# France has no labels, so its transitions are looked up in the US table after mapping each French form to a US form
# that behaves the same way there: drop/add/same are common in true pairs, swaps between the four targets never occur.
# Without this, every true French pair that drops "sarl" (~20% of France rank-1/2 candidates) looked like a legal swap.
# ets/ste ("etablissements"/"societe") are name prefixes, not legal types: ignored in the lookup key.
FR_LEGAL = {"sarl": "llc", "eurl": "llc", "sas": "inc", "sasu": "inc", "sa": "corp", "sci": "limited",
            "ei": "lp", "snc": "lp", "ets": None, "ste": None}


def _to_us_key(key):
    sides = []
    for side in key.split(">"):
        forms = {FR_LEGAL.get(f, f) for f in side.split("|") if f}
        sides.append("|".join(sorted(f for f in forms if f)))
    return ">".join(sides)


class _MappedTable(dict):
    """US transition counts looked up with French legal forms mapped to their US analogues."""
    def get(self, key, default=None):
        return dict.get(self, _to_us_key(key), default)


def _legal_table(country):
    """Counts of legal-type transitions in all training TRUE pairs (legal_transitions.py)."""
    if not _LT and LEGAL_TRANS_PATH.exists():
        _LT.update(json.loads(LEGAL_TRANS_PATH.read_text(encoding="utf-8")))
    if country in _LT:
        return _LT[country]
    return _MappedTable(_LT.get("US", {}))
# Per-country generic words that the noise ADDS to a name: "decoy" words (partners/holdings/group...) mark a different
# business (house numbers disagree), "noise" words (center/services...) leave the same business. Learned without labels
# by word_lists.py from candidate pairs; missing file = no lists (features are 0).
WORD_LIST_PATH = WORK / os.environ.get("V2_NAME", "v2") / "word_lists.json"
_WL = {}


def _word_lists(country):
    if not _WL and WORD_LIST_PATH.exists():
        _WL.update(json.loads(WORD_LIST_PATH.read_text(encoding="utf-8")))
    d = _WL.get(country, {})
    return frozenset(d.get("decoy", ())), frozenset(d.get("noise", ()))


def _nums(raw):
    out = set()
    for m in NUM.findall(SEP.sub(r"\1", raw.lower())):
        out.add("-".join(str(int(p)) for p in re.split(r"[-/]", m) if p))
    return out


def _num_feats(a1, a2):
    x, y = _nums(a1), _nums(a2)
    if not x or not y:
        return [0, 0, 0, len(x), len(y), 0, 0, 0, 0, -1, 0, 0]
    exact, subset = float(x == y), float(x <= y or y <= x)
    shared = len(x & y)
    part = trunc = close = rng = off420 = unrel = 0
    for a in x - y:
        pa = a.split("-")
        for b in y - x:
            pb = b.split("-")
            if len(pa) == len(pb) > 1 and sum(u != v for u, v in zip(pa, pb)) == 1:
                part = 1
            if len(pa) == len(pb) == 1:
                if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
                    trunc = 1
                d = abs(int(a) - int(b))
                if d <= 3:
                    close = 1
                elif d <= 20:
                    off420 = 1
                elif not (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)):
                    unrel = 1
            if len(pa) == 2 and len(pb) == 1 and int(pa[0]) <= int(b) <= int(pa[1]):
                rng = 1
            if len(pb) == 2 and len(pa) == 1 and int(pb[0]) <= int(a) <= int(pb[1]):
                rng = 1
    f1 = next((int(n) for n in (v.split("-")[0] for v in sorted(x, key=len)) if n), None)
    f2 = next((int(n) for n in (v.split("-")[0] for v in sorted(y, key=len)) if n), None)
    fd = math.log1p(abs(f1 - f2)) if f1 is not None and f2 is not None else -1
    return [exact, subset, shared, len(x), len(y), part, trunc, close, rng, fd, off420, unrel]


def _rows(chunk, decoy=frozenset(), noise=frozenset(), legal_table=None):
    legal_table = legal_table or {}
    out = []
    for n1, n2, c1, c2, a1, a2, r2, addr1 in chunk:
        e1 = {ENT[t] for t in n1.split() if t in ENT}
        e2 = {ENT[t] for t in n2.split() if t in ENT}
        legal = [len(e2 - e1), len(e1 - e2), float(bool(e1) and bool(e2) and e1 != e2),
                 float(("private" in e1) != ("private" in e2)), float(e1 == e2)]
        t1, t2 = c1.split(), c2.split()
        t1s, t2s = set(t1), set(t2)
        add, drop = t2s - t1s, t1s - t2s
        extra = [float(bool(add & decoy)), float(bool(add & noise)), float(bool(drop & decoy)), float(len(add & decoy))]
        d1 = [t for t in t1 if t not in GENERIC]
        d2 = [t for t in t2 if t not in GENERIC]
        first = float(bool(d1) and bool(d2) and d1[0] == d2[0])
        invented = float(len(t2) == 1 and len(t2[0]) >= 6 and t2[0].isalpha() and t2[0] not in t1
                         and not any(t in t2[0] for t in d1 if len(t) >= 4))
        ini = "".join(t[0] for t in t1 if t)
        acro = float((len(t2) == 1 and 2 <= len(t2[0]) <= 6 and t2[0] == ini) or
                     (len(t1) == 1 and 2 <= len(t1[0]) <= 6 and t1[0] == "".join(t[0] for t in t2 if t)))
        r2l = r2.lower()
        web = float(any(m in r2l for m in (".com", ".in", ".fr", ".net", ".org", "@", "#")))
        ns2 = c2.replace(" ", "")
        web_contains = float(len(t2) == 1 and len(d1) >= 2 and all(t in ns2 for t in d1))
        at = set(addr1.split())
        place = float(sum(t in at for t in d1)) / max(len(d1), 1)
        key = f"{'|'.join(sorted(e1))}>{'|'.join(sorted(e2))}"
        cnt = legal_table.get(key, 0)
        trans = [math.log1p(cnt), float(e1 != e2 and cnt == 0)]
        p1 = [q.strip().lower() for q in a1.split(",") if q.strip()]
        p2 = [q.strip().lower() for q in a2.split(",") if q.strip()]
        f1, f2 = n1.split(), n2.split()          # full normalized names (legal words included)
        true_only = [float(bool(ALIAS_RE.search(r2))), float(f1 == f2), float(f1 != f2 and sorted(f1) == sorted(f2)),
                     float(p1 != p2 and sorted(p1) == sorted(p2))]
        out.append(legal + _num_feats(a1, a2) + [first, invented, acro, web, web_contains, place] + extra + trans + true_only)
    return out


def _idf(tokens_series, n_docs):
    return (tokens_series.explode().drop_nulls().to_frame("tok").group_by("tok").len()
            .with_columns((np.log(n_docs) - pl.col("len").log()).cast(pl.Float32).alias("idf")).drop("len"))


def _idf_sums(df, left, right, idf, prefix, keep):
    """IDF-weighted shared / unmatched tokens between two list columns (pair id = pid)."""
    parts = {"shared": pl.col(left).list.set_intersection(pl.col(right)),
             "un1": pl.col(left).list.set_difference(pl.col(right)),
             "un2": pl.col(right).list.set_difference(pl.col(left))}
    out = df.select("pid")
    for name, expr in parts.items():
        long = (df.select("pid", expr.alias("tok")).explode("tok").drop_nulls()
                  .filter(keep).join(idf, on="tok", how="left")
                  .group_by("pid").agg(pl.col("idf").fill_null(12.0).sum().alias(f"{prefix}_{name}_idf"),
                                       pl.col("idf").fill_null(12.0).max().alias(f"{prefix}_{name}_max")))
        out = out.join(long, on="pid", how="left")
    return out.with_columns(pl.exclude("pid").fill_null(0.0))


_TABLES = {}


def _country_tables(split, country):
    """Normalized S1 / pool records of one country, cached across chunks of the same country."""
    key = (split, country)
    if key not in _TABLES:
        _TABLES.clear()
        cols = ["entity_id", "name", "name_core", "name_key", "addr", "country"]
        s1n = pl.scan_parquet(WORK / f"{split}_s1_norm.parquet").select(cols).filter(pl.col("country") == country).collect()
        pool = pl.concat([pl.scan_parquet(WORK / f"{split}_s{i}_norm.parquet").select(cols)
                          .filter(pl.col("country") == country).collect() for i in (2, 3)])
        _TABLES[key] = (s1n, pool)
    return _TABLES[key]


def country_features(cand, split, country, pool_workers=int(os.environ.get("FEAT_WORKERS", 8))):
    """cand: blocking output for ONE country (s1, cand, name_sim, addr_sim, score, rank, via, country)."""
    base = build_features(cand, split)                            # the 24 existing features
    s1n, pool = _country_tables(split, country)
    ids = pl.concat([cand["s1"], cand["cand"]]).unique()
    raw = pl.concat([pl.scan_csv(DATA / split / f"{split}_source{i}.tsv", separator="	", quote_char=None,
                                 infer_schema=False).select("entity_id", "business_name", "business_address")
                     .filter(pl.col("entity_id").is_in(ids)).collect() for i in (1, 2, 3)]).fill_null("")
    txt = lambda d, sfx: d.select(pl.col("entity_id"), *[pl.col(c).alias(c + sfx) for c in ("name", "name_core", "name_key", "addr")])
    df = (base.join(txt(s1n, "1").rename({"entity_id": "s1"}), on="s1", how="left")
              .join(txt(pool.filter(pl.col("entity_id").is_in(cand["cand"].unique())), "2").rename({"entity_id": "cand"}), on="cand", how="left")
              .join(raw.select(pl.col("entity_id").alias("s1"), pl.col("business_address").alias("raw_a1")), on="s1", how="left")
              .join(raw.select(pl.col("entity_id").alias("cand"), pl.col("business_address").alias("raw_a2"),
                               pl.col("business_name").alias("raw_n2")), on="cand", how="left")
              .with_columns(pl.col(pl.Utf8).fill_null("")).with_row_index("pid"))
    del raw
    # python-loop features in parallel
    cols = ["name1", "name2", "name_core1", "name_core2", "raw_a1", "raw_a2", "raw_n2", "addr1"]
    rows = df.select(cols).rows()
    chunks = [rows[i:i + 50_000] for i in range(0, len(rows), 50_000)]
    with Pool(pool_workers) as p:
        decoy, noise = _word_lists(country)
        res = [r for part in p.map(partial(_rows, decoy=decoy, noise=noise, legal_table=_legal_table(country)), chunks) for r in part]
    del rows, chunks
    df = df.with_columns([pl.Series(n, [r[i] for r in res], dtype=pl.Float32) for i, n in enumerate(PY_FEATURES)])
    del res
    # IDF-weighted name and address words
    allrec = pl.concat([s1n.select("name_core", "addr"), pool.select("name_core", "addr")])
    states = set(STATE_COMPONENT.get(country, {}).values())
    name_idf = _idf(allrec["name_core"].str.split(" "), allrec.height)
    addr_idf = _idf(allrec["addr"].str.split(" "), allrec.height)
    toks = df.select("pid", pl.col("name_core1").str.split(" ").alias("nt1"), pl.col("name_core2").str.split(" ").alias("nt2"),
                     pl.col("addr1").str.split(" ").alias("at1"), pl.col("addr2").str.split(" ").alias("at2"))
    word = ~pl.col("tok").str.contains(r"\d") & ~pl.col("tok").is_in(list(states | STREET_WORDS)) & (pl.col("tok") != "")
    nm = _idf_sums(toks, "nt1", "nt2", name_idf, "nm", ~pl.col("tok").is_in(list(GENERIC)) & (pl.col("tok") != ""))
    ad = _idf_sums(toks, "at1", "at2", addr_idf, "ad", word)
    gen = toks.select("pid", pl.col("nt1").list.set_difference(pl.col("nt2")).list.eval(pl.element().is_in(list(GENERIC))).list.sum().alias("gen_un1"),
                      pl.col("nt2").list.set_difference(pl.col("nt1")).list.eval(pl.element().is_in(list(GENERIC))).list.sum().alias("gen_un2"))
    df = df.join(nm, on="pid").join(ad, on="pid").join(gen, on="pid")
    del toks, nm, ad, gen
    # state agree / conflict / missing (+1 / -1 / 0)
    st = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().is_in(list(states)))).list.first()
    df = df.with_columns(st("addr1").alias("st1"), st("addr2").alias("st2")).with_columns(
        pl.when(pl.col("st1").is_null() | pl.col("st2").is_null()).then(0.0)
          .when(pl.col("st1") == pl.col("st2")).then(1.0).otherwise(-1.0).alias("state_rel"))
    # namesakes and crowding
    key_s1 = s1n.group_by("name_key").len().rename({"name_key": "name_key1", "len": "s1_key_n"})
    key_pool = pool.group_by("name_key").len().rename({"len": "pool_key_n"})
    addr_s1 = s1n.filter(pl.col("addr") != "").group_by("addr").len().rename({"addr": "addr1", "len": "s1_addr_n"})
    addr_pool = pool.filter(pl.col("addr") != "").group_by("addr").len().rename({"addr": "addr2", "len": "pool_addr_n"})
    df = (df.join(key_s1, on="name_key1", how="left")
            .join(key_pool.rename({"name_key": "name_key1", "pool_key_n": "pool_key_n1"}), on="name_key1", how="left")
            .join(key_pool.rename({"name_key": "name_key2", "pool_key_n": "pool_key_n2"}), on="name_key2", how="left")
            .join(addr_s1, on="addr1", how="left").join(addr_pool, on="addr2", how="left")
            .with_columns([pl.col(c).fill_null(0).cast(pl.Float32).log1p() for c in
                           ("s1_key_n", "pool_key_n1", "pool_key_n2", "s1_addr_n", "pool_addr_n")]))
    # group features without a model: same-name candidates and which one has the best address
    df = df.with_columns(
        (pl.col("key_equal") == 1).sum().over("s1").cast(pl.Float32).alias("grp_same_key"),
        pl.col("cnum_exact").max().over("s1").alias("grp_any_num_exact"),
        ((pl.col("cnum_exact").max().over("s1") == 1) & (pl.col("cnum_exact") == 0) &
         ((pl.col("cnum_close") + pl.col("cnum_off_4_20") + pl.col("cnum_part_changed")) > 0)).cast(pl.Float32).alias("num_shift_vs_sibling"),
        pl.col("legal_equal").max().over("s1").alias("grp_any_legal_equal"),
        pl.col("addr_sim").max().over("s1").alias("grp_max_addr"),
        pl.len().over("s1").cast(pl.Float32).alias("grp_n"),
        (pl.col("cand").str.slice(0, 2) == "S2").cast(pl.Float32).alias("is_s2"))
    df = df.with_columns(
        ((pl.col("key_equal") == 1) & (pl.col("addr_sim") >= pl.when(pl.col("key_equal") == 1).then(pl.col("addr_sim")).max().over("s1")))
        .cast(pl.Float32).alias("best_addr_among_namesakes"),
        (pl.col("grp_max_addr") - pl.col("addr_sim")).alias("addr_gap_to_best"))
    return df.drop("name1", "name2", "name_core1", "name_core2", "name_key1", "name_key2", "addr1", "addr2",
                   "raw_a1", "raw_a2", "raw_n2", "st1", "st2", "pid")


NEW_FEATURES = PY_FEATURES + [f"{p}_{k}_{m}" for p in ("nm", "ad") for k in ("shared", "un1", "un2") for m in ("idf", "max")] + \
    ["gen_un1", "gen_un2", "state_rel", "s1_key_n", "pool_key_n1", "pool_key_n2", "s1_addr_n", "pool_addr_n",
     "grp_same_key", "grp_max_addr", "grp_n", "is_s2", "best_addr_among_namesakes", "addr_gap_to_best",
     "grp_any_num_exact", "num_shift_vs_sibling", "grp_any_legal_equal"]
