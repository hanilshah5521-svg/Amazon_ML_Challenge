"""EXP-7: 400k MORE training S1 (disjoint from the held-out val_s1 and the existing extra_s1), through exactly the EXP-1
candidate pipeline: union blocking (Harsh V per-state TF-IDF top-40 + patel blocker) -> EXP-1 learned cut K=8 (the ranker never
saw these S1, so they are cut like test).
Usage (WORK_DIR = work/exp1, RANKER_EXTRA=1, OUR_CANDS_DIR=...): python extra2.py 400000 <out_dir>
Writes <out_dir>/extra2_s1.parquet and extra2_candidates_cut.parquet."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

import blocking as B
import ranker as RK


def main(n, out):
    out = Path(out)
    s1_all = pl.read_parquet(B.WORK / "train_s1_norm.parquet")
    used = pl.concat([pl.read_parquet(B.WORK / f)["entity_id"] for f in ("val_s1.parquet", "extra_s1.parquet")])
    ex = s1_all.filter(~pl.col("entity_id").is_in(used.implode())).sample(n, seed=44)
    m = RK.lgb.Booster(model_file=str(RK.V2 / "ranker.txt"))
    cuts = []
    for country in ex["country"].unique().sort():
        q = ex.filter(pl.col("country") == country)
        pool = pl.concat([B.load("train", 2, country), B.load("train", 3, country)])
        c = B.block_country(q, pool, k_final=40, ours=B.load_ours("train", q["entity_id"])).with_columns(pl.lit(country).alias("country"))
        c = c.drop("rank_hv")                                   # = derive_candidates.py exp1 view
        df = RK.features_all(c, "train")
        df = df.with_columns(pl.Series("p_rank", m.predict(df.select(RK.FEATS).to_numpy()).astype(np.float32)))
        df = df.with_columns(pl.col("p_rank").rank("ordinal", descending=True).over("s1").alias("new_rank")).filter(pl.col("new_rank") <= 8)
        cuts.append(c.join(df.select("s1", "cand", "new_rank"), on=["s1", "cand"]).with_columns(pl.col("new_rank").alias("rank")).drop("new_rank"))
        print(f"{country}: {q.height:,} S1 -> {cuts[-1].height:,} cut pairs", flush=True)
    pl.concat(cuts).write_parquet(out / "extra2_candidates_cut.parquet")
    ex.select("entity_id", "country").write_parquet(out / "extra2_s1.parquet")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
