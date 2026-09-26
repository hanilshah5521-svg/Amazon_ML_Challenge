# Submission 4: learned candidate cut, decoy/noise words, 400k training records (expected ~0.970)

- **Built:** 26 Sep 2026, 23:41 IST (not yet uploaded when written)
- **Output:** `output/v4_lgbm_two_stage/matching_results.tsv` (+ `candidate_pairs.tsv`, 192 MB, exactly 8 candidates per Source 1 record), SHA-256 prefix `00ccf1b7586a3332`
  (an earlier build with the `is_india` feature, `output/v3_lgbm_two_stage/`, prefix `22e36898f9c56fd0`, differs in 0.4% of pairs)
- **Validator:** PASS with `--check-ids`
- **Reproduce:** `bash code/business_entity_resolution/src/run_final.sh` (about 2 h 15 min on a 16 GB / 24-thread laptop; versions pinned in `requirements.txt`, Python 3.12)

## What changed since submission 3 (0.967)
| Change | Measured effect (held-out half B, India / US) |
|---|---|
| **Learned candidate cut** (`ranker.py`): blocking keeps the top-40, a cheap LightGBM keeps the best 8 per S1 (was the top-15 by `name_sim + addr_sim`) | 0.9751 / 0.9799 → 0.9772 / 0.9813; recall of true matches 0.960 / 0.977 → 0.971 / 0.981 with 47% fewer candidates |
| **Decoy / noise word features** (`word_lists.py`, per-country lists learned without labels) | +0.0011 / +0.0005 alone; +0.0009 / +0.0001 on top of the cut |
| **400k training records** (200k extra, never scored) | +0.0009 / +0.0004 on top of the two changes above |
| **Country-agnostic** (no `is_india` feature, countries taken from the data; needed for the open-set rule) | −0.0002 / −0.0000 |
| **Total** | **0.9788 / 0.9818** (baseline 0.9751 / 0.9799) |

## Tested and dropped
- Early-stopped boosting rounds (1749 / 310 instead of 600): +0.0000 / +0.0002 vs the final model, 3× slower.
- T4 + T5 normalization rules + S1-vocabulary spell correction: −0.0015 India, +0.0000 US (more false accepts).
- Cut size K=10 instead of 8: −0.0002 / +0.0003, 25% more candidates.

## Test output
- Matches: France 841,511 (3.24 per S1), India 2,660,675 (3.28), US 2,234,771 (3.37); total 5,736,957; 5.8% of S1 left empty (training truth 5.6%); validator PASS (`--check-ids`).
- Against the reproduced 0.967 pipeline (`output/v1_lgbm_two_stage`): 1.27% of its pairs are dropped and 1.37% new ones added; against the cut-only file (`output/v2_lgbm_two_stage`): 0.48% / 0.51%; against v3: 0.39% / 0.40%.

## Expected leaderboard score
About **0.970**, range 0.966–0.972. Basis: held-out weighted gain +0.0028 over the 0.967 pipeline (India +0.0037, US +0.0019), France unmeasured and assumed to gain
about as much. Downside: gains not transferring to test (about twice the unowned records per S1) or a France-specific failure of the ranker / decoy lists (trained on
India and US only). Not measured on the leaderboard at the time of writing.

## Known limitations
- France has no labels; it uses the US decision settings. The cut model and the decoy lists were fitted on India/US (the France decoy list comes from unlabelled test candidates).
- Most remaining loss is an information limit (no-address candidates whose name is shared by many S1 records) or blocking recall.
- `run_final.sh` chains the commands of the stage-by-stage runs that produced this file (each stage was run with its own work folder); it has not been re-run end-to-end.
