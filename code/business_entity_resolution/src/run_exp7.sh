#!/usr/bin/env bash
# EXP-7 = EXP-5 trained on 800k S1 (the 400k of EXP-5 + 400k new, disjoint from val_s1), same held-out half B.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 FEAT_WORKERS=${FEAT_WORKERS:-4}
: "${OUR_CANDS_DIR:?set OUR_CANDS_DIR}"
ROOTW=$(cd ../../..; pwd)/work; E5=$ROOTW/exp5; E7=$ROOTW/exp7
echo "=== [$(date +%H:%M)] block + cut 400k new training S1"
[ -f "$E7/extra2_candidates_cut.parquet" ] || WORK_DIR=$ROOTW/exp1 RANKER_EXTRA=1 python -W ignore extra2.py 400000 "$E7"
echo "=== [$(date +%H:%M)] EXP-5 features for the new S1 (statistics of the original 400k)"
[ -f "$E7/x5_extra2.parquet" ] || WORK_DIR=$E7 EXP5=1 python -W ignore exp5_features.py extra2
python -W ignore - <<PY
import polars as pl
E5, E7 = "$E5", "$E7"
pl.concat([pl.read_parquet(E5 + "/extra_s1.parquet"), pl.read_parquet(E7 + "/extra2_s1.parquet")]).write_parquet(E7 + "/extra_s1.parquet")
a = pl.read_parquet(E5 + "/extra_candidates_cut.parquet"); b = pl.read_parquet(E7 + "/extra2_candidates_cut.parquet")
pl.concat([a, b.select(a.columns)], how="vertical_relaxed").write_parquet(E7 + "/extra_candidates_cut.parquet")
x = pl.read_parquet(E5 + "/x5_train.parquet"); y = pl.read_parquet(E7 + "/x5_extra2.parquet")
pl.concat([x, y.select(x.columns)], how="vertical_relaxed").write_parquet(E7 + "/x5_train.parquet")
print("extra S1", pl.read_parquet(E7 + "/extra_s1.parquet").height, "| x5_train rows", x.height + y.height, flush=True)
PY
[ -e "$E7/x5_test.parquet" ] || ln -s "$E5/x5_test.parquet" "$E7/x5_test.parquet"
export WORK_DIR=$E7 PATTERN_FEATS=1 OWNER_FEATS=1 X5_FEATS=1 EXP5=1 MODEL_K=8
echo "=== [$(date +%H:%M)] legal transitions (no training S1)"
python -W ignore legal_transitions.py
echo "=== [$(date +%H:%M)] train (800k S1)"
VAL_CAND=val_candidates_cut.parquet EXTRA_CAND=extra_candidates_cut.parquet SKIP_BASE=1 python -W ignore model_v2.py train
echo "=== [$(date +%H:%M)] compare EXP-5 vs EXP-7 on half B"
python -W ignore compare_exps.py "$E5" "$E7"
echo "=== [$(date +%H:%M)] test prediction"
TEST_CAND="test_candidates_cut_{country}.parquet" python -W ignore model_v2.py predict
echo "=== [$(date +%H:%M)] EXP-7 DONE"
