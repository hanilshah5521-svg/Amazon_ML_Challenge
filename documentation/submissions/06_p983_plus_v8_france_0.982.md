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
