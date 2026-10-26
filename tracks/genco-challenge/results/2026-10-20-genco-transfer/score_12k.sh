#!/bin/bash
# Audit R1-F4 control: the pre-registered 12,000-update data/mask contrast. Scores state_step12000.pt of all 21 runs
# once on the locked 15 into heldout-12k/<run>/. Not used in the report until the queued disposition is approved.
cd <REPO>
R=tracks/genco-challenge/results/2026-10-20-genco-transfer
XROOT=${XROOT:-experiments/genco-transfer}   # SIMBENCH round: XROOT=experiments/genco-transfer-simbench OUTD=heldout-12k-simbench
OUTD=${OUTD:-heldout-12k}
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-0}
for run in "$@"; do
  c=$XROOT/$run/steps/state_step12000.pt
  [ -f $c ] || { echo "missing $c"; continue; }
  nice -n 10 python -m genco_challenge.score_ckpts --out $R/$OUTD/$run --ckpt ${run}_12k=$c > /dev/null 2>&1
  echo "$run done"
done
