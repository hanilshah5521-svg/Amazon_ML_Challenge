#!/usr/bin/env bash
# EXP-0 (Harsh V v4 reproduced, out-of-fold learned cut) and EXP-1 (+ 'patel' candidate union) on the same held-out split.
#   OUR_CANDS_DIR = work folder of the patel S2/S3->S1 blocker (records.parquet + cands.parquet per split)
#   bash run_patel.sh [start_step]
#   0 normalize (shared) · 1 train blocking, union (shared) · 2 test blocking, union (shared) · 3 derive exp0/exp1 candidates
#   4 exp0: ranker fit+cut, word lists, train, test cut+predict · 5 exp1: same, ranker has our_rank/our_rr, word lists copied from exp0
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 NORM_WORKERS=${NORM_WORKERS:-6} FEAT_WORKERS=${FEAT_WORKERS:-4} OOF_CUT=1
: "${OUR_CANDS_DIR:?set OUR_CANDS_DIR}"
ROOTW=$(cd ../../..; pwd)/work
SH=$ROOTW/shared; E0=$ROOTW/exp0; E1=$ROOTW/exp1
K=8; START=${1:-0}
step() { echo "=== [$(date +%H:%M)] STEP $1: $2"; }
mkdir -p "$SH"

if [ "$START" -le 0 ]; then step 0 "normalize"; export WORK_DIR=$SH
  python -W ignore normalize.py build-dict; python -W ignore normalize.py test; python -W ignore normalize.py train; fi
if [ "$START" -le 1 ]; then step 1 "train blocking (union)"; export WORK_DIR=$SH
  python -W ignore blocking.py eval 200000; python -W ignore blocking.py extra 200000; fi
if [ "$START" -le 2 ]; then step 2 "test blocking (union)"; export WORK_DIR=$SH
  python -W ignore blocking.py test 40; fi
if [ "$START" -le 3 ]; then step 3 "derive exp0 / exp1 candidates"
  python -W ignore derive_candidates.py "$SH" "$E0" "$E1"; fi
run_exp() {  # $1 = work dir, $2 = extra ranker features (0/1)
  export WORK_DIR=$1 RANKER_EXTRA=$2
  python -W ignore ranker.py fit
  python -W ignore ranker.py cut train "$K"
  if [ "$2" = "0" ]; then python -W ignore word_lists.py; else mkdir -p "$1/v2"; cp "$E0/v2/word_lists.json" "$1/v2/word_lists.json"; fi
  rm -f "$1/v2/train_features.parquet"
  MODEL_K=$K VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 python -W ignore model_v2.py train
  python -W ignore ranker.py cut test "$K"
  MODEL_K=$K TEST_CAND="test_candidates_cut_{country}.parquet" python -W ignore model_v2.py predict
}
if [ "$START" -le 4 ]; then step 4 "EXP-0 (Harsh V v4, OOF cut)"; run_exp "$E0" 0; fi
if [ "$START" -le 5 ]; then step 5 "EXP-1 (+ patel candidate union)"; run_exp "$E1" 1; fi
echo "=== [$(date +%H:%M)] ALL STEPS DONE"
