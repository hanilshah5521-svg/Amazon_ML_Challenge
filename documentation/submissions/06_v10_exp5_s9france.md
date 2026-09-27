# Submission v10: EXP-5 India/US + S9 France. **LB 0.983** (27 Sep 2026)

**Files:** `output/v10_e5_s9france/{matching_results,candidate_pairs}.tsv`
- matching_results SHA-256 prefix 8fc3d0ce211e5f84; validator PASS (--check-ids).
- 1,732,544 rows, 8 candidates per S1.

| Part | Source | Pairs |
|---|---|---:|
| India | EXP-5 (`output/v9_lgbm_two_stage`) | 2,724,839 |
| US | EXP-5 | 2,260,896 |
| France | S9 rule on the EXP-1 file (`code/fusion/`) | 848,874 |

**EXP-5 = EXP-1 candidates (v4 blocking ∪ patel blocker → learned cut K=8) + two-stage LightGBM with 156 stage-1 features:**
1. v4 base + `features_v2` (74)
2. + 11 generator-pattern features and FR→US legal map (EXP-2, `features_v2_pattern.py`)
3. + 6 owner-competition features from the patel blocker's owner lists over all S1 (EXP-3, `owner_competition.py`)
4. + 65 EXP-5 features (`exp5_features.py`): 52 edit flags (`pattern_miner.py`), OOF naive-Bayes edit score, supervised added/dropped word scores, number alignment (premise prefix, edit type, "number changed + same name / legal changed"), rarest candidate word, handle overlap
5. Leak fixes: deterministic first house number; legal-transition table without the 400k training S1; label statistics for held-out S1 learned only from the other 200k S1

**Held-out half B (200k val S1, decision chosen on half A):**

| | EXP-1 (LB 0.977) | EXP-2 | EXP-3 | **EXP-5** |
|---|---:|---:|---:|---:|
| India F0.5 | 0.98358 | 0.98423 | 0.98750 | **0.98845** |
| US F0.5 | 0.98484 | 0.98541 | 0.98709 | **0.98800** |

- Leak-free subset (outside the substitution map's learning half): EXP-3 → EXP-5 +0.00101.
- **Orphan stress** (hide h of train S1 from the owner lists), EXP-5 − EXP-2 at h = 0 / 10 / 20 / 30%:
  - India +0.0040 / +0.0035 / +0.0029 / +0.0020.
  - US +0.0025 / +0.0021 / +0.0018 / +0.0010.
- **Whole-state holdout** (20% of states unseen), stage 1. EXP-2 / EXP-3 / EXP-5:
  - India 0.98216 / 0.98622 / 0.98765.
  - US 0.98418 / 0.98631 / 0.98756.

**Reproduce:**
```bash
run_patel.sh            # EXP-1
run_exp2.sh             # EXP-2
run_exp3.sh             # EXP-3
python exp5_features.py train && python exp5_features.py test && run_exp5.sh   # EXP-5, WORK_DIR=work/exp5
python code/fusion/build_v6.py output/v10_e5_s9france output/v9_lgbm_two_stage
```
- Reruns differ by about 0.01–0.04% of pairs unless `EXP5=1` (hash-order fix) is set.
- Details: `documentation/experiments.md`, sections 11–14.
