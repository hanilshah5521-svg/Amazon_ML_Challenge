# Submission 5: all 2.2M training records, generator-pattern features, owner competition, XGBoost + CatBoost (LB 0.976)

- **Built:** 27 Sep 2026, Kaggle run `amlc26-train` v1; chosen variant `s2_xgb`
- **Output:** `output/v6_kaggle_s2_xgb/`, candidate pairs = the same learned K=8 cut as submission 4
- **Validator:** PASS (--check-ids)
- **Reproduce:**
  1. Local CPU: `extra_all.py 8 200000` (candidate search + cut for every training S1), `word_lists.py`, `legal_transitions.py`,
     `export_features.py train`, `export_features.py test` (with `WORK_DIR`, `MODEL_K=8`, `FEAT_WORKERS=8`).
  2. Upload `feat_train` / `feat_test_<country>` as private Kaggle datasets (`work_hv/kaggle_upload.py`).
  3. Kaggle GPU: `kaggle/amlc26_train.ipynb` → `test_scores.parquet`, `config.json`, `val_oof.parquet`.
  4. Local: `finalize_kaggle.py <kaggle output folder>` → TSVs + validator.

## What changed since submission 4 (harshv, leaderboard 0.973)
| Change | Why | Measured effect (held-out half B, India / US) |
|---|---|---|
| **All 2.2M training S1** (was 400k): candidate search + learned cut for the 2.0M remaining records, 17.65M training pairs, 7.46M positives | more data for the matcher | half B stage 1 0.98149, stage 2 0.98485 (India 0.98421, US 0.98565) |
| **Generator-pattern features** (`pattern_miner.py`, 120k true pairs vs 80k near-decoys): house number shifted by 4–20 or unrelated; legal-form transition never seen in true pairs (+ its log count); alias marker; identical / reordered name words; reordered address parts; group: another candidate keeps the number exactly / keeps the legal form | decoys = legal swaps (0.00% of true pairs, ~13% of decoys) and house-number shifts (~70% of decoys) | (inside the stage-1 number) |
| **France legal-form fix**: French forms mapped to US analogues before the transition lookup (sarl/eurl→llc, sas/sasu→inc, sa→corp, sci→limited, ei/snc→lp) | without it 17.7% of France rank-1 candidates were flagged "unseen legal swap" (India 1.0%); now 1.7% | France has no labels; flag rate check only |
| **Owner competition** (stage 2): how strongly *other* S1 records claim the same S2/S3 candidate (claim count, strongest competing claim, gap, rank among claimants, mutual best) | one S2/S3 record has at most one owner; the previous group features only looked within the S1 | ablation `s2nc_xgb` vs `s2_xgb`: 0.98225 without vs 0.98485 with (+0.0026) |
| **XGBoost + CatBoost (GPU) ensemble** instead of LightGBM | Kaggle GPU; LightGBM there is CPU-only | stage 2: XGB 0.98485, CatBoost 0.98459, ensemble 0.98482; XGB alone chosen |

## Suggestions evaluated and not done (27 Sep, 11:00)
- Extra retrieval channels (exact key, house number): blocking misses are 1.8% of true pairs at top-40 (India 2.4%, US 1.5%). 31% of the misses have an identical name key,
  but a generic one (median 289 pool records share it, p90 6,246); most misses are invented names with partial or empty addresses. Held-out gain would be ≤ ~0.001 and
  would likely not transfer (test has ~2× decoys; larger candidate sets are ranked lower).
- Separator-free numbers ("2 48" vs "248"): 0.43% of true pairs vs 0.06% of decoys, mostly accepted already. ≤ +0.0005, needs a feature re-export.
- Hard-negative mining, typo/reorder alignment and a per-record no-match decision already exist (top-8 learned-cut negatives; `core_tsort`/`reordered_only`/…; expected-F0.5 rule with n = 0).

## Test output
Test matches: France 855,802, India 2,689,325, US 2,250,357.

## Expected leaderboard score
**Leaderboard: 0.976.**

## Known limitations
- France: no labels. It uses the stricter labelled-country decision; features fitted on India/US (legal lookup mapped as above).
- Out-of-fold stage-1 scores on train vs fold-mean stage-1 scores on test feed stage 2 (standard stacking).
- The legal-transition table is built from all training true pairs (positives are never "unseen" by construction). The effect is negligible:
  only 0.004% of true pairs have a transition seen once.
