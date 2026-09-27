"""What separates the 30,364 French pairs removed in H9796 from the 835,151 kept? Usage: python profile_fr.py > 02_france_removed.txt"""
import polars as pl
from compare_files import pairs, R
W = R + "hv_repo/work/exp1/"
E1 = pairs(R + "hv_repo/output/v2_lgbm_two_stage/matching_results.tsv")
H = pairs("/Users/patel/Downloads/matching_results (1).tsv").with_columns(pl.lit(1).alias("kept"))
c = pl.read_parquet(W + "test_candidates_cut_France.parquet").rename({"cand": "m"})
fr = (E1.join(c, on=["s1", "m"]).join(H, on=["s1", "m"], how="left").with_columns(pl.col("kept").fill_null(0))
        .with_columns(pl.col("m").str.slice(0, 2).alias("src"), pl.len().over("s1").alias("n_e1"),
                      pl.col("kept").sum().over("s1").alias("n_h")))
rec = pl.concat([pl.read_parquet(W + f"test_s{i}_norm.parquet") for i in (1, 2, 3)]).select("entity_id", "name", "addr")
fr = fr.join(rec.rename({"entity_id": "s1", "name": "n1", "addr": "a1"}), on="s1").join(rec.rename({"entity_id": "m", "name": "n2", "addr": "a2"}), on="m")
fr.write_parquet("fr_pairs_e1.parquet")
print("France E1 pairs", fr.height, "removed", (fr["kept"] == 0).sum())
for col in ["src", "via", "rank", "n_e1"]:
    print(f"\nby {col}:", fr.group_by(col).agg(pl.len(), (1 - pl.col("kept").mean()).alias("removed_rate")).sort(col).to_dicts())
print("\nS1-level: removed pairs per S1 pattern (n_e1, n_h):")
s = fr.group_by("s1").agg(pl.first("n_e1"), pl.first("n_h")).group_by("n_e1", "n_h").len().sort("n_e1", "n_h")
print(s.filter(pl.col("n_h") < pl.col("n_e1")).to_dicts())
print("S1 with any removal:", fr.filter(pl.col("kept") == 0)["s1"].n_unique(), " S1 fully emptied:", fr.filter(pl.col("n_h") == 0)["s1"].n_unique())
for col in ["name_sim", "addr_sim", "score", "our_rr"]:
    print(f"\n{col} quantiles kept / removed:",
          [round(fr.filter(pl.col("kept") == k)[col].quantile(q) or -1, 3) for k in (1, 0) for q in (.1, .5, .9)])
print("\nsample removed:")
with pl.Config(fmt_str_lengths=45, tbl_rows=40, tbl_width_chars=250):
    print(fr.filter(pl.col("kept") == 0).sample(30, seed=1).select("n1", "n2", "a1", "a2", "rank", "n_e1", "n_h"))
    print("\nsample kept, same S1 as a removal:")
    print(fr.filter((pl.col("kept") == 1) & (pl.col("n_h") < pl.col("n_e1"))).sample(15, seed=1).select("n1", "n2", "a1", "a2", "rank", "n_e1", "n_h"))
