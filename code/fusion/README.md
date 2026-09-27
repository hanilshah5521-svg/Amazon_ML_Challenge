# Fusion scripts: France of v10 (S9 rule) + India/US from a model output

- `rebuild_s9.py`: reconstructs the S9 France removal set (LB 0.9796). A French pair of the EXP-1 file is removed when it is in S10's removal set and both first house numbers exist and differ.
  - The first house number is the first number of the raw address, skipping 5–6 digit postal codes after the first token, with leading zeros stripped.
  - Gives 16,641 pairs (teammate: 16,642).
  - Needs `fr_pairs_e1.parquet` from `profile_fr.py`, which needs the S10 file (`matching_results (1).tsv`).
- `build_v6.py <out_dir> <india_us_output_dir>`: India/US from a model output + S9 France.
  - v10: `python build_v6.py output/v10_e5_s9france output/v9_lgbm_two_stage` (EXP-5).
- The paths (`R = ...`) are absolute to the machine that produced v10; adjust them before running elsewhere.
- `build_v12.py`, `build_final.py`: remove the team file's India/US removal list (`teammate_iu_removed.parquet` = pairs in the 0.977 file but not in the team's 0.9822 file) from a model output, and take France from another file.
  - v17 = `build_final.py output/v15_lgbm_two_stage output/v10_e5_s9france output/v17_e7_teamIU_s9fr`.
- `build_fr_exp5.py`: France from EXP-5 + S9 list (+ rule in `both` mode). Not used for v17; see submission 07.
