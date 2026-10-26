#!/bin/bash
# Audit R2-F1: the pre-registered selection-free read (protocol-rescue amendment 3): every rescue run's FINAL state,
# scored once on the locked 15 into heldout-rescue-final/<run>/. Not a go/no-go input.
cd <REPO>
R=tracks/genco-challenge/results/2026-10-20-genco-transfer
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-0}
for run in "$@"; do
  c=experiments/genco-transfer-rescue/$run/steps/state_final.pt
  [ -f $c ] || { echo "missing $c"; exit 1; }
  nice -n 10 python -m genco_challenge.score_ckpts --out $R/heldout-rescue-final/$run --ckpt ${run}_final=$c > /dev/null 2>&1
  echo "$run done"
done
