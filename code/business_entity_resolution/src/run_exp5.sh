#!/usr/bin/env bash
# EXP-5 = EXP-3 + leak fixes (deterministic first house number; legal-transition table without any of the 400k training S1)
#       + exp5_features.py (edit flags + naive Bayes, supervised word scores, number alignment, rare-name / handle features).
# Needs work/exp5 (symlinks to exp3 inputs + oc tables) and x5_train.parquet (python exp5_features.py train).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 FEAT_WORKERS=${FEAT_WORKERS:-4} PATTERN_FEATS=1 OWNER_FEATS=1 X5_FEATS=1 EXP5=1 MODEL_K=8
ROOTW=$(cd ../../..; pwd)/work
export WORK_DIR=$ROOTW/exp5
echo "=== [$(date +%H:%M)] legal transitions (no training S1)"
python -W ignore legal_transitions.py
echo "=== [$(date +%H:%M)] train"
VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 python -W ignore model_v2.py train
echo "=== [$(date +%H:%M)] compare EXP-3 vs EXP-5 on half B"
python -W ignore compare_exps.py "$ROOTW/exp3" "$ROOTW/exp5"
echo "=== [$(date +%H:%M)] waiting for x5_test.parquet"
until [ -f "$WORK_DIR/x5_test.parquet" ]; do sleep 20; done
echo "=== [$(date +%H:%M)] test prediction"
TEST_CAND="test_candidates_cut_{country}.parquet" python -W ignore model_v2.py predict
echo "=== [$(date +%H:%M)] EXP-5 DONE"
