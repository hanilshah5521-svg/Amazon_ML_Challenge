"""Reverse-engineer the noise generator: atomic edit scripts of TRUE pairs vs NEAR-DECOYS.

Both groups come from the same held-out candidate lists (identical retrieval conditions):
  true       = candidate is a ground-truth match of the S1
  near-decoy = candidate belongs to NO S1 at all and scores high (rank <= 3, name_sim+addr_sim >= 1.3)
  other-owned= candidate belongs to a different S1 and scores high (namesakes / neighbours)
Each pair's raw name and address differences are split into atomic edits; the report gives each
edit's rate per group and the likelihood ratio decoy/true. Edits (near) absent from true pairs
are candidate hard rules. Usage: python pattern_miner.py [n_per_group=120000]
"""
import re
import sys
from collections import Counter
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

from common import DATA, WORK, read_pairs
from normalize import GENERIC, LEGAL, NAME_STOP

OUT = WORK / "patterns"
OUT.mkdir(exist_ok=True)
TOK = re.compile(r"[a-z0-9]+")
NUMS = re.compile(r"\d+(?:[-/]\d+)*")
LOOK = {("0", "o"), ("1", "l"), ("1", "i"), ("5", "s"), ("8", "b"), ("6", "g"), ("rn", "m"), ("vv", "w")}
HONOR = {"sri", "shri", "shree", "sree", "smt", "mr", "mrs", "ms", "dr", "m", "s", "messrs"}
LEGALW = set(LEGAL) | {"private", "limited", "public", "trust", "society", "partnership", "l", "c", "p"}


def toks(s):
    return TOK.findall(anyascii(s).lower())


def tclass(t):
    if t.isdigit():
        return "number"
    if t in LEGALW:
        return "legal"
    if t in GENERIC:
        return "generic"
    if t in HONOR:
        return "honorific"
    if t in NAME_STOP:
        return "stopword"
    if t in ("france", "india", "usa", "us"):
        return "country"
    return "distinctive"


def typo_kind(a, b):
    if sorted(a) == sorted(b):
        return "transposition"
    if len(a) == len(b):
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diff) == 1 and (diff[0] in LOOK or diff[0][::-1] in LOOK):
            return "lookalike"
        return "substitution"
    return "insertion" if len(b) > len(a) else "deletion"


def name_edits(n1, n2):
    e = []
    raw2 = n2
    if re.search(r"[ऀ-෿]", raw2):
        e.append("N:indic_script")
    if re.search(r"[À-ÿ]", raw2) and not re.search(r"[À-ÿ]", n1):
        e.append("N:accent_injected")
    if raw2.isupper() and not n1.isupper():
        e.append("N:all_caps")
    if raw2.islower() and not n1.islower():
        e.append("N:all_lower")
    if re.search(r"\.(com|in|net|org|fr)\b|^[@#]", raw2.strip().lower()):
        e.append("N:website_or_handle")
    if re.search(r"\b(f/?k/?a|formerly|dba|doing business as|a/?k/?a)\b", raw2.lower()):
        e.append("N:alias_marker")
    if re.search(r"^\s*(--|<<|>>|\*)", raw2):
        e.append("N:junk_prefix")
    if re.search(r"[\[\(][^\]\)]*[\]\)]", raw2) and not re.search(r"[\[\(]", n1):
        e.append("N:brackets_added")
    a, b = toks(n1), toks(raw2)
    if a == b:
        e.append("N:tokens_identical")
    elif sorted(a) == sorted(b):
        e.append("N:reordered_only")
    sa, sb = set(a), set(b)
    added, dropped = sb - sa, sa - sb
    for t in added:
        e.append(f"N:add_{tclass(t)}")
    for t in dropped:
        e.append(f"N:drop_{tclass(t)}")
    la, lb = tuple(sorted(t for t in a if t in LEGALW)), tuple(sorted(t for t in b if t in LEGALW))
    if la != lb:
        e.append(f"N:legal {' '.join(la) or '-'} -> {' '.join(lb) or '-'}")
    # typos between unmatched distinctive tokens
    ua = [t for t in dropped if tclass(t) == "distinctive"]
    ub = [t for t in added if tclass(t) == "distinctive"]
    for x in ua:
        best = max(ub, key=lambda y: fuzz.ratio(x, y), default=None)
        if best and fuzz.ratio(x, best) >= 70:
            e.append(f"N:typo_{typo_kind(x, best)}")
        elif best:
            e.append("N:distinctive_replaced")
    if len(b) == 1 and len(b[0]) >= 6 and b[0] not in sa and not any(t in b[0] for t in a if len(t) >= 4):
        e.append("N:invented_single_token")
    d_a = [t for t in a if tclass(t) == "distinctive"]
    d_b = [t for t in b if tclass(t) == "distinctive"]
    if d_a and d_b and d_a[0] != d_b[0] and fuzz.ratio(d_a[0], d_b[0]) < 70:
        e.append("N:first_distinctive_differs")
    return e


def addr_edits(a1, a2):
    e = []
    if not a2.strip():
        return ["A:empty"]
    if re.search(r"\bnull\b|<null>|\bn/a\b", a2.lower()):
        e.append("A:null_token")
    if a2.isupper() and not a1.isupper():
        e.append("A:all_caps")
    p1 = [p.strip().lower() for p in a1.split(",") if p.strip()]
    p2 = [p.strip().lower() for p in a2.split(",") if p.strip()]
    if sorted(p1) == sorted(p2) and p1 != p2:
        e.append("A:components_reordered")
    if len(p2) < len(p1):
        e.append("A:fewer_components")
    if len(p2) > len(p1):
        e.append("A:more_components")
    n1 = {"-".join(str(int(x)) for x in re.split(r"[-/]", m) if x) for m in NUMS.findall(re.sub(r"\s*([-/])\s*", r"\1", a1))}
    n2 = {"-".join(str(int(x)) for x in re.split(r"[-/]", m) if x) for m in NUMS.findall(re.sub(r"\s*([-/])\s*", r"\1", a2))}
    if n1 and not n2:
        e.append("A:numbers_all_dropped")
    elif n2 and not n1:
        e.append("A:numbers_all_added")
    elif n1 and n2:
        if n1 == n2:
            e.append("A:numbers_identical")
        elif n1 < n2:
            e.append("A:number_added")
        elif n2 < n1:
            e.append("A:number_dropped")
        else:
            for x in n1 - n2:
                px = x.split("-")
                for y in n2 - n1:
                    py = y.split("-")
                    if len(px) == len(py) > 1 and sum(u != v for u, v in zip(px, py)) == 1:
                        e.append("A:compound_one_part_changed")
                    elif len(px) == len(py) == 1:
                        dx = abs(int(x) - int(y))
                        if x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x):
                            e.append("A:number_truncated_or_extended")
                        elif dx <= 3:
                            e.append("A:number_off_by_1to3")
                        elif dx <= 20:
                            e.append("A:number_off_by_4to20")
                        else:
                            e.append("A:number_unrelated")
    w1 = {t for t in toks(a1) if not t.isdigit()}
    w2 = {t for t in toks(a2) if not t.isdigit()}
    if w1 and w2:
        j = len(w1 & w2) / len(w1 | w2)
        e.append("A:words_jac>=0.8" if j >= 0.8 else "A:words_jac0.5-0.8" if j >= 0.5 else "A:words_jac<0.5")
    return e


def edits(row):
    n1, n2, a1, a2 = row
    e = set(name_edits(n1, n2) + addr_edits(a1, a2))
    return e, len(e)


def run(rows):
    return [edits(r) for r in rows]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
    val = pl.read_parquet(WORK / "val_s1.parquet")
    truth = read_pairs().rename({"m": "cand"})
    owned = truth.select("cand").unique()
    c = pl.scan_parquet(WORK / "val_candidates.parquet").select("s1", "cand", "rank", "score").collect()
    c = c.join(truth.with_columns(pl.lit("true").alias("grp")), on=["s1", "cand"], how="left")
    c = c.with_columns(pl.when(pl.col("grp") == "true").then(pl.lit("true"))
                         .when(pl.col("cand").is_in(owned["cand"])).then(pl.lit("other_owned"))
                         .otherwise(pl.lit("near_decoy")).alias("grp"))
    hi = (pl.col("rank") <= 3) & (pl.col("score") >= 1.3)
    groups = {"true": c.filter(pl.col("grp") == "true"),
              "near_decoy": c.filter((pl.col("grp") == "near_decoy") & hi),
              "other_owned": c.filter((pl.col("grp") == "other_owned") & hi)}
    groups = {g: d.sample(min(n, d.height), seed=1).select("s1", "cand") for g, d in groups.items()}
    del c
    need = pl.concat([pl.concat([d["s1"], d["cand"]]) for d in groups.values()]).unique()
    raw = pl.concat([pl.scan_csv(DATA / "train" / f"train_source{i}.tsv", separator="	", quote_char=None, infer_schema=False)
                     .select("entity_id", "business_name", "business_address").filter(pl.col("entity_id").is_in(need))
                     .collect() for i in (1, 2, 3)]).fill_null("")
    res, sizes = {}, {}
    for g, d in groups.items():
        d = d.join(raw.rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1"}), on="s1") \
             .join(raw.rename({"entity_id": "cand", "business_name": "n2", "business_address": "a2"}), on="cand")
        rows = d.select("n1", "n2", "a1", "a2").rows()
        with Pool(4) as p:
            out = [x for part in p.map(run, [rows[i:i + 10_000] for i in range(0, len(rows), 10_000)]) for x in part]
        cnt = Counter(t for s, _ in out for t in s)
        res[g] = {k: v / len(out) for k, v in cnt.items()}
        res[g]["_n_edits_mean"] = sum(k for _, k in out) / len(out)
        sizes[g] = len(out)
        print(f"{g}: {len(out):,} pairs mined", flush=True)
    keys = sorted(set().union(*[set(r) for r in res.values()]))
    rows = [(k, res["true"].get(k, 0), res["near_decoy"].get(k, 0), res["other_owned"].get(k, 0)) for k in keys]
    df = pl.DataFrame(rows, schema=["edit", "true", "near_decoy", "other_owned"], orient="row").with_columns(
        ((pl.col("near_decoy") + 1e-4) / (pl.col("true") + 1e-4)).round(2).alias("LR_decoy_vs_true"))
    df.write_csv(OUT / "edit_rates.tsv", separator="\t")
    with pl.Config(tbl_rows=200, fmt_str_lengths=60, tbl_width_chars=160, float_precision=4):
        print("group sizes:", sizes)
        print("\n=== edits much more common in NEAR-DECOYS (LR >= 3, decoy rate >= 0.5%)")
        print(df.filter((pl.col("LR_decoy_vs_true") >= 3) & (pl.col("near_decoy") >= 0.005)).sort("LR_decoy_vs_true", descending=True))
        print("\n=== edits much more common in TRUE pairs (LR <= 0.33, true rate >= 0.5%)")
        print(df.filter((pl.col("LR_decoy_vs_true") <= 0.33) & (pl.col("true") >= 0.005)).sort("LR_decoy_vs_true"))
        print("\n=== frequent edits in both (true rate >= 5%)")
        print(df.filter(pl.col("true") >= 0.05).sort("true", descending=True))


if __name__ == "__main__":
    main()
