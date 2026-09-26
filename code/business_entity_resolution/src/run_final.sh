#!/usr/bin/env bash
# Reproduces the final submission (learned candidate cut K=8 + decoy/noise-word features + 400k training records) from the raw
# competition data. Runs every step in order and stops at the first error. Resume from a step with: bash run_final.sh <step>
#   0 learn Indic dictionary + normalize train/test                           (~10 min)
#   1 candidate search: 200k held-out train S1 + 200k extra training S1          (~16 min)
#   2 candidate search for all test S1 (top-40 per S1, per state)                (~45 min)
#   3 learned cut: fit the ranker, keep the best K=8 per S1 (train, extra)       (~8 min)
#   4 decoy/noise word lists (India/US from train candidates, France from test)  (~2 min)
#   5 train: features, stage 1, stage 2, decision rules on 400k S1               (~20 min)
#   6 test: learned cut, features, prediction, both TSVs, validator              (~35 min) -> output/v<N>_lgbm_two_stage/
# Total about 2 h 15 min on a 16 GB / 24-thread laptop. Lower NORM_WORKERS / FEAT_WORKERS if RAM is tight.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export NORM_WORKERS=${NORM_WORKERS:-6} FEAT_WORKERS=${FEAT_WORKERS:-4}
K=${K:-8}                                             # candidates per S1 kept by the learned cut
START=${1:-0}
step() { echo "=== [$(date +%H:%M)] STEP $1: $2"; }

if [ "$START" -le 0 ]; then
  step 0 "Indic dictionary + normalization"
  python -W ignore normalize.py build-dict
  python -W ignore normalize.py test
  python -W ignore normalize.py train
fi
if [ "$START" -le 1 ]; then
  step 1 "train candidate search (200k held-out S1 + 200k extra training S1)"
  python -W ignore blocking.py eval 200000
  python -W ignore blocking.py extra 200000
fi
if [ "$START" -le 2 ]; then
  step 2 "test candidate search (top-40 per S1)"
  python -W ignore blocking.py test 40
fi
if [ "$START" -le 3 ]; then
  step 3 "learned cut: ranker fit (prints the recall-vs-K table), then cut train/extra to K=$K"
  python -W ignore ranker.py fit
  python -W ignore ranker.py cut train "$K"
fi
if [ "$START" -le 4 ]; then
  step 4 "decoy / noise word lists"
  python -W ignore word_lists.py
fi
if [ "$START" -le 5 ]; then
  step 5 "features + stage 1 + stage 2 + decision rules (400k training S1)"
  rm -f ../../../work/v2/train_features.parquet
  MODEL_K=$K VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 \
    python -W ignore model_v2.py train
fi
if [ "$START" -le 6 ]; then
  step 6 "test: learned cut + features + prediction + output files + validator"
  python -W ignore ranker.py cut test "$K"
  MODEL_K=$K TEST_CAND="test_candidates_cut_{country}.parquet" timeout 3h python -W ignore model_v2.py predict
fi
echo "=== [$(date +%H:%M)] ALL STEPS DONE"
