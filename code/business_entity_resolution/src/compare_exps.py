"""Compare EXP-0 and EXP-1 on held-out half B with the decision each experiment chose on half A.

Also reports the leak-free subset: half-B S1 whose ids are in the half NOT used to learn the 'patel' substitution map
(bucket = numeric id // 10**4 % 10000, learned on even buckets), so no half-B label reached the union candidates.

Usage: WORK_DIR=<any> python compare_exps.py <exp0_dir> <exp1_dir>
"""
import json
import sys
from pathlib import Path

import polars as pl

from common import read_pairs
from evaluation import decide, f05_macro
from model_v2 import MODEL_K, decide_expected_f


def score(exp, subset=None):
    cfg = json.loads((exp / "v2" / "config.json").read_text())
    oof = pl.read_parquet(exp / "v2" / "oof.parquet")
    halves = pl.read_parquet(exp / "v2" / "halves.parquet")
    col = "p2" if cfg["use_stage2"] else "p1"
    truth = read_pairs().rename({"m": "cand"})
    out = {}
    for country, d in cfg["decision"].items():
        if country.startswith("_"):
            continue
        b = halves.filter(pl.col("half") == 1).join(oof.select("s1", "country").unique(), on="s1")
        b = b.filter(pl.col("country") == country)
        if subset is not None:
            b = b.filter(subset)
        ids = b["s1"].to_list()
        c = oof.filter(pl.col("s1").is_in(ids)).with_columns(pl.col(col).alias("score"))
        pred = decide_expected_f(c, d["miss"]) if d["rule"] == "expected_f" else decide(c, MODEL_K, d["thr"])
        t = truth.filter(pl.col("s1").is_in(ids))
        tp = pred.join(t, on=["s1", "cand"]).height
        out[country] = dict(f05=f05_macro(pred, t, ids), n_s1=len(ids), precision=tp / max(1, pred.height), recall=tp / max(1, t.height),
                            fp=pred.height - tp, fn=t.height - tp)
    w = {"India": 0.47, "US": 0.38}
    out["weighted"] = sum(w[c] * out[c]["f05"] for c in w) / sum(w.values())
    return out


def main():
    e0, e1 = Path(sys.argv[1]), Path(sys.argv[2])
    clean = (pl.col("s1").str.slice(3).cast(pl.Int64) // 10_000 % 10_000 % 2 == 1)
    for label, sub in (("ALL half-B S1", None), ("LEAK-FREE half-B S1 (outside the substitution map's learning half)", clean)):
        r0, r1 = score(e0, sub), score(e1, sub)
        print(f"\n== {label}")
        for c in ("India", "US"):
            a, b = r0[c], r1[c]
            print(f"  {c:6s} n={a['n_s1']:,}  EXP-0 F0.5 {a['f05']:.5f} (P {a['precision']:.4f} R {a['recall']:.4f} FP {a['fp']:,} FN {a['fn']:,})"
                  f"  |  EXP-1 F0.5 {b['f05']:.5f} (P {b['precision']:.4f} R {b['recall']:.4f} FP {b['fp']:,} FN {b['fn']:,})  delta {b['f05'] - a['f05']:+.5f}")
        print(f"  weighted: EXP-0 {r0['weighted']:.5f}  EXP-1 {r1['weighted']:.5f}  delta {r1['weighted'] - r0['weighted']:+.5f}")


if __name__ == "__main__":
    main()
