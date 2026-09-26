# Progress: improving the 0.967 solution (repo `Amazon_ML_Challenge-0.967`)

Plan approved by the user (option b: also test a light pre-ranker to shrink candidate sets). Instructions come from
`../AGENT_PROMPT_improve_F05.md`. Every change is measured on held-out half B and kept only if half-B F0.5 does not drop in
India or US and the average rises by >= 0.0005 (France-only changes: unsupervised checks + eyeballing changed decisions).
Older, separate pipeline (my first attempt, held-out India 0.934 / US 0.968) lives in `../6ab10eb3b23ba_student_resource/student_resource/`.

## Setup (done)
- venv `.venv` (Python 3.12.10; pinned requirements from `requirements.txt` install as wheels; 3.10 not available).
- `student_resource` = directory junction to the existing data folder (no copy).
- Code edits so far (no effect on results): `NORM_WORKERS` / `FEAT_WORKERS` env vars for worker counts; `loss_breakdown()` + OOF save
  (`work/v2/oof.parquet`, `halves.parquet`) in `model_v2.py`.

## Runs
| Run | What | Status |
|---|---|---|
| phase0 | baseline steps 0-2 + loss breakdown + OOF | DONE 18:0x (normalize 9 min, blocking 8 min, train ~20 min) |
| baseline_test | steps 3-4 with pristine `src/` -> `output/v1_lgbm_two_stage/` | DONE 19:26, validator PASS (--check-ids). Test candidates 43 min, predict 37 min |
| t3 | decoy/noise word features (`src_dev/`, `work/v2_t3/`) | running/queued, `work/t3_chain.log` |
| t8a | early-stopped rounds (`src_t8/`, `work/v2_t8a/`) | queued after T3 |
| t4t5 | normalization rules + S1 spell correction (`src_t45/`, `work_t45/`) | queued after T8a, `work/t45.log` |

## RK: learned candidate cut (`src_rk/ranker.py`, `work/v2_rk/`)
Cheap LightGBM (28 pair features: name/addr sim, ranks, gaps, state relation, number overlap, Jaccards; no per-pair Python) re-ranks the stored top-40 and keeps K.
OOF recall of true matches (200k held-out S1), raw `name_sim+addr_sim` cut vs learned cut:
| K | India raw | India learned | US raw | US learned |
|---|---|---|---|---|
| 8 | 0.9491 | **0.9710** | 0.9657 | **0.9807** |
| 10 | 0.9548 | 0.9742 | 0.9714 | 0.9836 |
| 12 | - | 0.9752 | - | 0.9846 |
| 15 | 0.9601 | 0.9757 | 0.9765 | 0.9853 |
(ceiling = top-40: India 0.976, US 0.985.) Rule picked K=8: learned@8 already exceeds the baseline's raw top-15 in both countries with ~47% fewer candidates.
Matcher retrain on the cut set: 1.6M pairs (baseline 3.0M) with 676,577 positives (baseline 671,356). Stage 1: India 0.9767, US 0.9803; stage 2 (final): India 0.9772, US 0.9813 (see results table).

## Error studies on the M2 model (held-out OOF)
- Rejected true pairs (stage-2 p<0.5): only 1.8% of true pairs (24,332 of 1,351,943). Typical cases: candidate has NO address (name identical/near-identical),
  or address matches but name is an invented trade name. `work/empty_addr_study.py`: for empty-address candidates with the same name key the model is essentially
  perfectly calibrated (mean p2 vs true rate: 1 S1 with that name 0.986/0.985; 2 -> 0.487/0.499; 3 -> 0.349/0.348; 4+ -> 0.043/0.043). The 360k candidates with 4+ same-name
  S1 (chains) have a 4.3% true rate: ~1 true owner in 23, not identifiable without an address -> an information limit, not a model flaw.
- Threshold sensitivity (`work/threshold_sensitivity.py`): half-B F0.5 is flat around the chosen decision. India threshold rule +0.05 stricter: -0.0004, +0.10: -0.0009;
  US +0.05: -0.0001, +0.10: -0.0005. Expected-F with miss 0..0.5: within 0.0002. No hedge applied (test predictions: 3.31 matches/S1 and 5.8% empty vs 5.6% truth in train,
  no sign of decoy over-acceptance).

## T6.1 diagnostic: unowned S2/S3 records, train truth vs baseline test predictions (`work/analyze_unowned.py`)
| | S1 | pool per S1 | unowned per S1 | near-copy of an S1 | orphan cluster (S2+S3, no S1 key) | other |
|---|---|---|---|---|---|---|
| train India | 883k | 4.68 | 1.22 | 9.6% | 1.2% | 89.2% |
| test India | 810k | 5.82 | 2.56 | 9.7% | 4.7% | 85.7% |
| train US | 1.32M | 4.67 | 1.22 | 0.6% | 0.7% | 98.7% |
| test US | 663k | 5.76 | 2.39 | 0.5% | 9.5% | 90.0% |
| test France | 259k | 5.53 | 2.25 | 13.7% | 16.7% | 69.7% |
Reading: test has ~2x the unowned records per S1; most extra ones are lone records (not clusters). Orphan clusters (businesses without an S1 record) are 8-27x more
frequent per S1 than in train (0.12 India, 0.23 US, 0.38 France records/S1) but still a minority. So "drop 19% of S1 from the holdout" would over-create orphan clusters;
use a smaller, per-country drop if T6 is done. (Note: predicted, not true, ownership on test; key-based detection misses noisy clusters.)

## Incident log
- 19:30: T3 died with MaybeEncodingError while the unowned-records analysis (4.5 GB) ran at the same time; the old chain then launched T8a and T4T5 out of order.
  Fix: killed T4T5, everything now runs strictly sequentially via `work/run_seq.sh` (T3 -> T8a -> T4T5, failures logged, no overlap). Never run two heavy jobs at once here.

## Results table (half-B macro F0.5, India / US)
| Change | India | US | Runtime | Kept? |
|---|---|---|---|---|
| baseline stage A (24 base features) | 0.9364 | 0.9469 | | (ablation) |
| baseline stage 1 (+v2 features) | 0.9745 | 0.9791 | | (ablation) |
| **baseline stage 2 (final)** | **0.9751** | **0.9799** | ~1 h (steps 0-2) | reference; weighted (0.47/0.38 normalised) 0.9773 |

Decision rules picked on half A: India threshold 0.75; US expected-F (miss 0).

| Experiment | India | US | weighted | change vs baseline | kept? |
|---|---|---|---|---|---|
| **T3 decoy/noise word features** (per-country lists from candidate pairs) | **0.9762** | **0.9804** | 0.9781 | India +0.0011, US +0.0005, weighted +0.0008 | **KEPT** (>= 0.0005, no country drops) |
| **RK learned cut, K=8** (baseline features; 8 candidates/S1 instead of 15) | **0.9772** | **0.9813** | 0.9790 | India +0.0020, US +0.0014, weighted +0.0018 | **KEPT** (+ 47% smaller candidate set) |

| **M1 = RK (K=8) + T3** (merged code `src_m1/`, `work/v2_m1/`) | **0.9781** | **0.9814** | 0.9796 | India +0.0030, US +0.0016, weighted +0.0023 vs baseline | **KEPT** (best so far; 8 candidates/S1) |

| **M2 = M1 + 200k extra training S1 (400k total, never evaluated)** (`src_m2/`, `work/v2_m2/`) | **0.9790** | **0.9818** | 0.9803 | vs M1: India +0.0009, US +0.0004, weighted +0.0007; vs baseline India +0.0039, US +0.0020, weighted ~+0.0030 | **KEPT** |

| M2ES = M2 + early-stopped rounds (stage 1: 1749 rounds, stage 2: 310; lr 0.05) | 0.9790 | 0.9820 | 0.9805 | vs M2: India +0.00004, US +0.00019, weighted +0.0001 | **DROPPED** (< 0.0005, ~3x slower training) |

| T4+T5 normalization rules + S1-vocabulary spell correction (baseline pipeline, 15 candidates; `src_t45/`, `work_t45/`) | 0.9736 | 0.9799 | 0.9766 | vs baseline: India -0.0015, US +0.0000, weighted -0.0007 | **DROPPED** (India worse) |

T4+T5 detail: candidate recall rose (US recall@15 0.9765->0.9792, India 0.9601->0.9606) and true pairs in the top-15 set +0.19%, but F0.5 did not follow: stage 1 India 0.9728 (baseline 0.9745), US 0.9790 (0.9791).
Loss breakdown (half B): India not-in-top-K 0.0122, rejected 0.0079, false accepts 0.0045 (baseline 0.0030), no-match 0.0017 (0.0006), total 0.0264 (0.0249);
US rejected 0.0086, not-in-top-K 0.0068, false accepts 0.0033 (0.0027), no-match 0.0014, total 0.0201 (0.0201). Reading: the extra cleaning made decoys look more like matches (more false accepts), esp. India.
Not isolated which of T4 / T5 hurts; both dropped. Final pipeline keeps the baseline normalization (so test normalization/blocking need no re-run).

| K=10 variant of M2 (`work/v2_m2k10/`; 10 candidates/S1, 4.0M training pairs) | 0.9788 | 0.9822 | 0.9804 | vs M2 (K=8): India -0.0002, US +0.0003, weighted +0.0001 | **DROPPED** (no measurable gain, 25% more candidates) |

K=10 loss breakdown: India rejected 0.0087, not-in-top-K 0.0081 (K=8: 0.0086), false accepts 0.0030, no-match 0.0014, total 0.0212 (K=8 0.0210);
US rejected 0.0091, not-in-top-K 0.0054 (0.0058), false accepts 0.0023, no-match 0.0010, total 0.0178 (0.0182). More recall did not convert into F0.5. K=8 (submission v3) stays the final size.

M2 loss breakdown (half B): India rejected 0.0090, not-in-top-K 0.0086, false accepts 0.0028, no-match 0.0005, total 0.0210 (M1 0.0219);
US rejected 0.0090, not-in-top-K 0.0058, false accepts 0.0022, no-match 0.0011, total 0.0182 (M1 0.0186). Rules: India threshold 0.725, US expected-F (miss 0).
Training set 3.2M pairs, 1,351,943 positives; ~18 min for the whole `train`.

M1 loss breakdown (half B): India rejected 0.0087, not-in-top-K 0.0086, false accepts 0.0032, no-match 0.0014, total 0.0219 (baseline 0.0249);
US rejected 0.0090, not-in-top-K 0.0058, false accepts 0.0026, no-match 0.0011, total 0.0186 (baseline 0.0201). Rules: expected-F both (miss 0).
Marginal: T3 on top of RK adds India +0.0009, US +0.0001.

RK loss breakdown (half B): India not-in-top-K 0.0123->0.0086, rejected 0.0089->0.0090, false accepts 0.0030->0.0038, no-match 0.0006->0.0013, total 0.0249->0.0228;
US not-in-top-K 0.0079->0.0058, rejected 0.0084->0.0092, false accepts 0.0027->0.0027, no-match 0.0012->0.0010, total 0.0201->0.0187. Rules: India expected-F (miss .1), US expected-F (miss 0).

T3 loss breakdown (half B): India rejected 0.0089->0.0073, false accepts 0.0030->0.0036, not-in-top-K 0.0123 (same), total 0.0249->0.0238; US rejected 0.0084->0.0081, false accepts 0.0027->0.0025, total 0.0201->0.0196.
Word lists (`work/v2_t3/word_lists.json`): India decoy {agencies, enterprises, group, holdings, industries, solutions, technologies, traders, trading, ventures}, noise {center, partners, service, services};
US decoy {group, holdings, partners}, noise {association, center, council, service, services, society, trust}; France (from unlabelled test candidates) decoy {college, developpement, distribution, groupe, holding, international, participations, service}, noise {association, associes, council, fils, foundation, services, society, trust}.

### Baseline test output `output/v1_lgbm_two_stage/` (validator PASS, `--check-ids`)
France 259,452 S1 -> 850,412 matches (3.28/S1); India 809,986 -> 2,646,083 (3.27); US 663,106 -> 2,234,785 (3.37); total 5,731,280 matches,
S1 with no match 5.9% (train truth 5.6%). candidate_pairs 357 MB (<=15 per S1). SHA-256 prefix 74558bd0224b2616 (reference file b5edd37b8508fe00: differs,
expected with Python 3.12 vs 3.10 / thread order; the reference file is not available here to diff).

### Submission v2 = M1 (learned cut K=8 + T3): `output/v2_lgbm_two_stage/` (validator PASS, --check-ids) 20:52
France 846,583 matches (3.26/S1), India 2,657,676 (3.28), US 2,231,084 (3.36); total 5,735,343; S1 with no match 5.8%; 100,251 empty rows.
candidate_pairs.tsv 192 MB (exactly 8 per S1; v1 was 341 MB, 15 per S1). vs v1: 5,660,688 pairs common, 70,592 only in v1, 74,655 only in v2; 132,867 S1 (7.7%) changed.
SHA-256 prefix 79ce841f3352013c. Expected leaderboard ~0.969-0.970 (held-out weighted gain +0.0023 over the 0.967 baseline; assumes the gain transfers). Not uploaded (user uploads).
Test pipeline for M1: `ranker.py cut test 8` (5.5 min) + `model_v2.py predict` with TEST_CAND=test_candidates_cut_{country}.parquet MODEL_K=8 (~25 min).

### Submission v3 = M2 (learned cut K=8 + T3 + 400k training S1): `output/v3_lgbm_two_stage/` (validator PASS, --check-ids) 22:05
France 842,879 matches (3.25/S1), India 2,660,213 (3.28), US 2,233,352 (3.37); total 5,736,444; S1 with no match 5.9%; 102,726 empty rows.
candidate_pairs.tsv 192 MB (8 per S1). vs v2: 5,709,417 pairs common (25,926 only v2, 27,027 only v3); vs v1: 1.25% / 1.33% differ.
SHA-256 prefix 22e36898f9c56fd0. Held-out weighted gain vs the 0.967 baseline ~+0.0030 -> expected leaderboard ~0.970 (range 0.969-0.971). Not uploaded.
Predict ~24 min: `V2_NAME=v2_m2 MODEL_K=8 TEST_CAND=test_candidates_cut_{country}.parquet python src_m2/model_v2.py predict`.

### Loss breakdown, baseline final model, half B (macro F0.5 lost)
| Cause | India | US |
|---|---|---|
| true match not in top-15 | 0.0123 | 0.0079 |
| true candidates rejected | 0.0089 | 0.0084 |
| false accepts | 0.0030 | 0.0027 |
| no-match record given a match | 0.0006 | 0.0012 |
| total | 0.0249 | 0.0201 |

Candidate recall (blocking_eval.tsv, 200k held-out S1): India recall@5/8/10/15/20/30 = 0.879/0.949/0.955/0.960/0.963/0.968; US 0.892/0.966/0.971/0.977/0.979/0.983.

Reading: wrong merges (false accepts + no-match) are only ~0.004; blocking recall and rejected true candidates are ~85% of the loss -> T8/T9 (model strength, calibration, features for uncertain pairs) matter more than T3's decoy words (estimated <= +0.001-0.003).

## FINAL (26 Sep, 23:45): submission v4 + package
- M3 = M2 without the inherited `is_india` feature, countries from the data (open-set rule): India 0.9788 / US 0.9818 (M2 0.9790 / 0.9818), adopted. `src/` is now the country-agnostic final code;
  the original 0.967 code is kept in `code/business_entity_resolution/src_baseline_0967/`; experiment copies `src_dev`, `src_t8`, `src_t45`, `src_rk`, `src_m1`, `src_m2`, `src_m3` stay in the repo for reference (not packaged).
- **Final file: `output/v4_lgbm_two_stage/`**, validator PASS (--check-ids): France 841,511 (3.24/S1), India 2,660,675 (3.28), US 2,234,771 (3.37), total 5,736,957, 5.8% empty, 8 candidates per S1 (192 MB),
  SHA-256 prefix 00ccf1b7586a3332. Expected leaderboard ~0.970 (range 0.966-0.972).
- Package: `submission_package/team_submission.zip` (121 MB; output/, code/business_entity_resolution/{src, README.md, requirements.txt}, Documentation_template.md) + `Approach_summary_2page.md`.
  Team name / members are placeholders in both documents.
- Not done: leaderboard upload (user), end-to-end re-run of `run_final.sh`, T6/T9/T10, France-specific fixes.
