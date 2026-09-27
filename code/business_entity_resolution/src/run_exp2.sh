#!/usr/bin/env bash
# EXP-2 = EXP-1 (0.977) + 0.976-Hanil generator-pattern features + FR legal map. Reuses EXP-1's cut candidates and word lists
# byte-for-byte, so the ONLY change is the matcher's feature set. Needs work/exp1 from run_patel.sh.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 FEAT_WORKERS=${FEAT_WORKERS:-4} PATTERN_FEATS=1 MODEL_K=8
ROOTW=$(cd ../../..; pwd)/work
E1=$ROOTW/exp1; E2=$ROOTW/exp2
mkdir -p "$E2/v2"
for f in indic_dict.json train_s1_norm.parquet train_s2_norm.parquet train_s3_norm.parquet test_s1_norm.parquet \
         test_s2_norm.parquet test_s3_norm.parquet val_s1.parquet extra_s1.parquet val_candidates_cut.parquet \
         extra_candidates_cut.parquet test_candidates_cut_France.parquet test_candidates_cut_India.parquet test_candidates_cut_US.parquet; do
  [ -e "$E2/$f" ] || ln -s "$(readlink -f "$E1/$f")" "$E2/$f"
done
cp "$E1/v2/word_lists.json" "$E2/v2/word_lists.json"
export WORK_DIR=$E2
echo "=== [$(date +%H:%M)] legal transitions (held-out S1 excluded)"
python -W ignore legal_transitions.py
echo "=== [$(date +%H:%M)] train (features + stage 1 + stage 2 + decisions)"
VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 python -W ignore model_v2.py train
echo "=== [$(date +%H:%M)] compare EXP-1 vs EXP-2 on half B"
python -W ignore compare_exps.py "$E1" "$E2"
echo "=== [$(date +%H:%M)] test prediction"
TEST_CAND="test_candidates_cut_{country}.parquet" python -W ignore model_v2.py predict
echo "=== [$(date +%H:%M)] EXP-2 DONE"
