# Submission v17: EXP-7 India/US − team India/US rule + S9 France (27 Sep 2026)

**Files:** `output/v17_e7_teamIU_s9fr/{matching_results,candidate_pairs}.tsv`
- matching_results SHA-256 prefix d414527971d56b00; validator PASS (--check-ids).
- 1,732,544 rows, 8 candidates per S1.
- The outputs are not in git (git-ignored; `candidate_pairs.tsv` is 201 MB).

| Part | Source |
|---|---|
| India / US | EXP-7 (`output/v15_lgbm_two_stage`) minus 22,771 pairs on the team's India/US removal list |
| France | S9 (identical to v10, LB 0.983): the 0.977 file minus 16,641 reconstructed S9 pairs; 848,874 pairs |

**EXP-7** = EXP-5 trained on **800k** training S1: the 400k of EXP-5 + 400k new S1 (`extra2.py`, seed 44, disjoint from val_s1).
- The new S1 go through union blocking and the EXP-1 learned cut K=8. The ranker never saw them, so they are cut like test.
- Their EXP-5 label statistics come from the original 400k (`exp5_features.py extra2`), as for test.
- Half B: India 0.98879, US 0.98840; vs EXP-5 +0.00037 weighted, leak-free +0.00029.

**Team India/US rule:** the team's LB-0.9822 file = the 0.977 file − S9 France − 15,397 India / 30,213 US pairs, mostly "same name, different house number".
- France is identical to S9, so its +0.0026 over S9 comes only from India/US.
- EXP-7 still accepted 22,771 of those pairs; they are removed (`code/fusion/build_final.py`).
- **Package:** the team rule / file must be included for reproducibility.

**Why S9 France and not EXP-5 France (v14 / v16).** Branch `Hanil`, submission 6: v10 + 10,531 French pairs accepted only by a newer model (house-number consistent) → LB 0.983 → **0.982**. v14 / v16 add 24,130 such pairs, so they would most likely lose.

**Expected LB ≈ 0.984–0.985** (not measured yet).

**Reproduce:** submission v10 steps, then
```bash
OUR_CANDS_DIR=... bash run_exp7.sh                                                      # EXP-7 -> output/v15_lgbm_two_stage
python code/fusion/build_final.py output/v15_lgbm_two_stage output/v10_e5_s9france output/v17_e7_teamIU_s9fr   # needs teammate_iu_removed.parquet
```
