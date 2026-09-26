#!/usr/bin/env bash
# Reproduces the 0.967 submission from the raw competition data. Runs every step in order and
# stops at the first error. Resume from a step with: bash run_v2.sh <step>
#   0 learn Indic dictionary + normalize train/test   (~30 min)
#   1 candidate search for 200k training S1           (~10 min)
#   2 features + stage 1 + stage 2 + ablation report  (~30 min)
#   3 candidate search for all test S1                (~40 min)
#   4 test features + prediction + validator          (~45 min) -> output/v<N>_lgbm_two_stage/
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
W="../../../work"
START=${1:-0}
step() { echo "=== [$(date +%H:%M)] STEP $1: $2"; }

if [ "$START" -le 0 ]; then
  step 0 "Indic dictionary + normalization"
  python -W ignore normalize.py build-dict
  python -W ignore normalize.py test
  python -W ignore normalize.py train
fi
if [ "$START" -le 1 ]; then
  step 1 "candidate search for 200k training records (per state, full recall)"
  python -W ignore blocking.py eval 200000
fi
if [ "$START" -le 2 ]; then
  step 2 "features + stage 1 + stage 2 + ablation on 200k records"
  rm -f "$W/v2/train_features.parquet"
  python -W ignore model_v2.py train
fi
if [ "$START" -le 3 ]; then
  step 3 "test candidate search (per state, full recall)"
  rm -f "$W"/test_candidates_France.parquet "$W"/test_candidates_India.parquet "$W"/test_candidates_US.parquet "$W"/test_candidates.parquet
  python -W ignore blocking.py test 40
fi
if [ "$START" -le 4 ]; then
  step 4 "test features + prediction + output files + validator"
  timeout 3h python -W ignore model_v2.py predict
fi
echo "=== [$(date +%H:%M)] ALL STEPS DONE"
