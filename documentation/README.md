# Documentation: Amazon ML Challenge 2026 (Business Entity Resolution)

This folder is a running log of what we tried, what we measured and what we submitted. It's
updated after every experiment and submission.

| File | Contents |
|---|---|
| [experiments.md](experiments.md) | Everything tried so far, in order, with measured results, including what failed or was dropped |
| [submissions/01_handweighted_0.725.md](submissions/01_handweighted_0.725.md) | Submission 1: hand-weighted score (leaderboard 0.725) |
| [submissions/02_lightgbm_0.915.md](submissions/02_lightgbm_0.915.md) | Submission 2: LightGBM matcher (leaderboard 0.915) |
| [submissions/03_lgbm_two_stage_0.967.md](submissions/03_lgbm_two_stage_0.967.md) | Submission 3: two-stage LightGBM with v2 features (leaderboard 0.967) |

## Submissions at a glance

| # | Date / time (IST) | Method | Held-out F0.5 (India / US) | Leaderboard | Status |
|---|---|---|---|---|---|
| 1 | 25 Sep, ~23:40 | Hand-weighted name + address score | 0.757 / 0.800 | 0.725 | superseded |
| 2 | 25 Sep, ~23:55 | LightGBM + decoy checks | 0.913 / 0.942 | 0.915 | superseded |
| 3 | 26 Sep, ~10:00 | Two-stage LightGBM, v2 features, per-state search, 200k training records | 0.975 / 0.980 | **0.967** | **current best** |

## How we evaluate
- **Held-out set:** 20,000 random train Source 1 records (seed 42), searched against the full train
  Source 2+3 pool, as on test.
- **Metric:** macro F0.5 per Source 1 record, exactly as in the challenge README (singletons score
  1.0 only when the prediction is empty).
- **Validation:** LightGBM uses 5-fold cross-validation grouped by Source 1 record, so a record's
  candidates never appear in both training and scoring.
- **France** has no labels. It is judged only through the leaderboard and unsupervised checks.
