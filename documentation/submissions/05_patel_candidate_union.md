# Submission 5 (branch `patel`): Harsh V v4 + a second candidate source before the learned cut

- **Built:** 27 Sep 2026, ~03:10 IST, on a 10-core / 16 GB Mac (Python 3.12.14; pins as `requirements.txt` except **scipy 1.16.3**, because 1.15.3's wheel does not load on this macOS).
- **Output:** `output/v2_lgbm_two_stage/`. The EXP-0 reproduction of v4 is `output/v1_lgbm_two_stage/`.
- **Reproduce:**
  1. `code/patel_blocker/README.md` builds `OUR_CANDS_DIR`.
  2. `OUR_CANDS_DIR=... bash code/business_entity_resolution/src/run_patel.sh` runs everything else.
- **Only change vs the 0.973 pipeline:** the learned cut's input.
  - Harsh V's per-state TF-IDF top-40 **∪** the candidates of the `patel` S2/S3→S1 blocker (IDF index over name/address unigrams and bigrams, compact keys, S2/S3→S1 substitutions learned on half of the train S1, deep re-rank for low-confidence records).
  - The ranker gets two extra features, `our_rank` and `our_rr`, and `via` gains bit 4.
  - **K stays 8** candidates per S1.
  - Normalisation, word lists (copied from EXP-0), the ~70 matcher features, the two-stage LightGBM and the decision rules are unchanged.
- **Leakage fix applied to both experiments:** held-out S1 are cut with **out-of-fold** ranker scores (`OOF_CUT=1`). v4 cut them with a ranker trained on them. The fix cost EXP-0 about 0.0002, and 519 of 1.35M positives.

## Candidate recall (out-of-fold, 200k held-out S1)
| | India raw → learned @8 | US raw → learned @8 | Ceiling (all candidates) |
|---|---|---|---|
| EXP-0 (v4 reproduced) | 0.9491 → **0.9709** | 0.9657 → **0.9808** | 0.9760 / 0.9855 |
| **EXP-1 (+ union)** | 0.9556 → **0.9885** | 0.9685 → **0.9908** | 0.9926 / 0.9945 |

Positives reaching the matcher (400k S1 × 8): 1,351,424 → **1,369,270**.

## Held-out half B (decision chosen on half A, same `default_rng(0)` halves)
| | India | US | Weighted (0.47 / 0.38) |
|---|---:|---:|---:|
| EXP-0 (v4 reproduced; documented v4: 0.9788 / 0.9818) | 0.97862 | 0.98188 | 0.98008 |
| **EXP-1** | **0.98358** | **0.98484** | **0.98415** |
| Change | +0.00497 | +0.00296 | **+0.00407** |
| **Leak-free subset**: half-B S1 outside the substitution map's learning half (50k S1) | +0.00504 | +0.00304 | **+0.00415** |

**Loss breakdown (half B):**

| Cause | India EXP-0 → EXP-1 | US EXP-0 → EXP-1 |
|---|---|---|
| True match not in top-K | 0.0088 → **0.0031** | 0.0060 → **0.0023** |
| True candidates rejected | 0.0083 → 0.0090 | 0.0088 → 0.0091 |
| False accepts | 0.0030 → 0.0031 | 0.0022 → 0.0027 |
| No-match record given a match | 0.0013 → 0.0012 | 0.0011 → 0.0011 |
| **Total** | **0.0214 → 0.0164** | **0.0181 → 0.0152** |

**Precision, recall and error counts:**

| | Precision | Recall | FP | FN |
|---|---|---|---|---|
| India EXP-0 | 0.9955 | 0.9442 | 589 | 7,727 |
| India EXP-1 | 0.9955 | 0.9583 | 594 | 5,769 |
| US EXP-0 | 0.9967 | 0.9538 | 652 | 9,576 |
| US EXP-1 | 0.9962 | 0.9625 | 756 | 7,762 |

Keep rule, same as §9 of `experiments.md`: weighted ≥ +0.0005 and no country worse than −0.0005. **Passed** (+0.0041).

The one trade-off is **US false accepts, +16%**. Test has about 2× the decoy density of train, so watch the leaderboard.

## Test output
- `output/v2_lgbm_two_stage/matching_results.tsv`, SHA-256 prefix `12d12ec4e49fbf94`. Validator **PASS** with `--check-ids`.
- `candidate_pairs.tsv`: exactly 8 candidates per S1, the same size as v4.

**Matches by country:**

| Country | EXP-0 (v4 reproduced, prefix `371a3ba3c233c44c`) | EXP-1 | Change |
|---|---:|---:|---:|
| France | 842,805 | 865,515 | +2.7% |
| India | 2,660,380 | 2,699,835 | +1.5% |
| US | 2,234,754 | 2,253,736 | +0.8% |
| **Total** | 5,737,939 | **5,819,086** | |

- Share of S1 left empty: 5.8% for EXP-0, **5.6%** for EXP-1, which equals the training singleton rate.
- EXP-0 reproduces v4: 5,737,939 vs 5,736,957 matches, within 0.02% per country.
- Against EXP-0, EXP-1 drops 0.55% of pairs and adds 1.97% new ones.

**Expected leaderboard (not measured):**
- Held-out weighted gain: +0.0041.
- v4 turned +0.0028 held-out into +0.006 on the leaderboard (0.967 → 0.973).
- If the gain transfers at 1:1, expect **about 0.977**, range about 0.974–0.979.
- Risk: US false accepts rose 16% on held-out, and test has about 2× the decoy density. France is unmeasured.

## Why (see `documentation/patel_forensics_and_plan.md`)
- The largest *fixable* v4 loss was true matches never reaching the top 8.
- The two blockers miss different pairs:
  - TF-IDF on names and addresses finds typo-heavy matches.
  - The IDF key index with learned substitutions and deep re-rank finds transliterated names, abbreviation variants and low-evidence records.
- The learned cut uses both sources' ranks to pick the 8.
- The v4 decoy features, and the per-S1 expected-F decision that makes v4 robust to test's higher decoy density, are unchanged.
