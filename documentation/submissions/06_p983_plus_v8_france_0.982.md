# Submission 6: patel v10 (0.983) + 10,531 French pairs from our region-filled v8 model (LB 0.982)

- **Built:** 27 Sep 2026
- **Output:** `output/v10_p983_plus_v8fr_filled/`
- **Validator:** PASS (--check-ids)
- **Result: LB 0.982**, down from 0.983. The added French pairs hurt, so their precision was below the ~77% break-even.

## What went in

- **Base:** patel's v10 file (LB 0.983; branch `patel`, EXP-5 India/US + S9 France). India, US and every base pair are unchanged.
- **v8 model** = Kaggle `hanilshah/amlc26-v8` (`kaggle/amlc26_train.py`: XGB only, fold models saved, T6 variant), run on French test data after `fr_region_fill.py`.
  - `fr_region_fill.py` addresses the 31.9% of French S2/S3 records that have a full address but no region. It learns 22 city tokens → region from French S1, and 99.95% of its region guesses agree on records that do carry a region.
  - Half B:
    - stage 1: 0.98149
    - stage 2 `s2_xgb`: 0.98484 (India 0.98413, US 0.98570)
    - T6 `s2t6_xgb`: 0.98412, so it was dropped
- **Added** (`add_fr_pairs.py ... filled_only`): 10,531 French pairs accepted only by v8, where all three conditions hold:
  - The S2/S3 address was region-filled.
  - The first house numbers agree, or one of them is missing (patel's S9 rule).
  - No S1 owns the S2/S3 record in the base.

## Numbers

- v8-only French pairs: 54,096, of which 35.7% have different house numbers (Patel-only French pairs: 1.7%).
- Added pairs: median v8 score 0.985.
- On half B, v8 scores are well calibrated for India/US, with a true rate of 0.998 above 0.725. For France, the LB says the calibration is not.

## Lesson

- Our France scores are over-confident even after the region fill. Pairs that only our model accepts should not be added to a stronger file.
- Not uploaded: `output/v9_p983_plus_v8fr` (all 34,301 house-consistent pairs). It would most likely lose more.

## Reproduce (all commands in `code/business_entity_resolution/src/`, with `WORK_DIR` set to one work folder)

1. **Upstream:** follow submission 4 (`run_final.sh`: normalize, blocking, learned cut) to get the `test_s*_norm.parquet` files and the cut test candidates.
2. **Training candidates:** `extra_all.py 8 200000`. **Test candidates:** `cut_test_chunked.py`. **Tables:** `word_lists.py`, `legal_transitions.py`.
3. **France region fill:** `fr_region_fill.py`, **before** exporting test features. It rewrites the French rows of `test_s2/s3_norm.parquet` and keeps `*.pre_frfill.parquet`.
4. **Export features:** `export_features.py train` and `export_features.py test` (`MODEL_K=8`, `FEAT_WORKERS=8`).
5. **Upload the features to Kaggle:** run `kaggle_upload.py train` and `kaggle_upload.py test`, which create private datasets `<user>/amlc26-feat-train` and `<user>/amlc26-feat-test`. On a different Kaggle account, change `hanilshah` to that account's name in `kaggle/kernel-metadata.json` (`id`, `dataset_sources`).
6. **Train on Kaggle GPU:** run `kaggle/amlc26_train.ipynb`, about 75 min on a P100, with defaults `MODELS=xgb`, `FOLDS=4`, `T6=1`. It writes `test_scores.parquet`, `val_oof.parquet`, `config.json` and `models/`.
7. **Build v8:** `FINAL_OUT=<out>/v8_kaggle_s2_frfill python finalize_kaggle.py <kaggle out> s2_xgb`.
8. **Build this submission:** `FINAL_OUT=<out>/v10_p983_plus_v8fr_filled python add_fr_pairs.py "<Patel's 0.983 matching_results.tsv>" <out>/v8_kaggle_s2_frfill <kaggle out> filled_only`.
   - The base file comes from branch `patel` (submission v10) or `Main_Submissions/`; it is not in git.

The competition data and work folders are not in git; the steps need `student_resource/dataset/` in the repo root. Re-runs match to 5 decimals on half B (stage 1 and stage 2 matched the earlier Kaggle run).
