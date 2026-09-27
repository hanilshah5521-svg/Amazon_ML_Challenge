#!/usr/bin/env bash
# EXP-3 = EXP-2 + owner-competition features (owner_competition.py) in both stages. Reuses EXP-2's training features and
# EXP-1's cut candidates, so the ONLY change is the 6 oc_* features. Needs work/exp2 (run_exp2.sh).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 FEAT_WORKERS=${FEAT_WORKERS:-4} PATTERN_FEATS=1 OWNER_FEATS=1 MODEL_K=8
: "${OUR_CANDS_DIR:?set OUR_CANDS_DIR}"
ROOTW=$(cd ../../..; pwd)/work
E2=$ROOTW/exp2; E3=$ROOTW/exp3
mkdir -p "$E3/v2"
for f in "$E2"/*.parquet "$E2"/indic_dict.json; do
  [ -e "$E3/$(basename "$f")" ] || ln -s "$(readlink -f "$f")" "$E3/$(basename "$f")"
done
cp "$E2/v2/word_lists.json" "$E2/v2/legal_transitions.json" "$E3/v2/"
[ -e "$E3/v2/train_features.parquet" ] || ln -s "$E2/v2/train_features.parquet" "$E3/v2/train_features.parquet"
export WORK_DIR=$E3
echo "=== [$(date +%H:%M)] owner-competition tables (train + test, full populations)"
python -W ignore owner_competition.py
echo "=== [$(date +%H:%M)] train (stage 1 + stage 2 + decisions)"
VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 python -W ignore model_v2.py train
echo "=== [$(date +%H:%M)] compare EXP-2 vs EXP-3 on half B"
python -W ignore compare_exps.py "$E2" "$E3"
echo "=== [$(date +%H:%M)] test prediction"
TEST_CAND="test_candidates_cut_{country}.parquet" python -W ignore model_v2.py predict
echo "=== [$(date +%H:%M)] EXP-3 DONE"
