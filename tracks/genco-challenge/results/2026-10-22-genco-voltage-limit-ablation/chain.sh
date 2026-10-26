#!/bin/bash
# usage: chain.sh <worker-name>; runs the pre-registered order. Each step exits when no unclaimed grid remains.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
W=$1
for step in "control pieces" "fvs pieces" "oos seen_whole" "oos unseen_whole"; do
  set -- $step
  nice -n 15 $PY run_ablate.py --study $1 --set $2 --worker $W >> log_$W.txt 2>&1
  if [ "$(date +%s)" -ge "$(date -d '2026-10-22 07:30' +%s)" ]; then break; fi
done
echo "CHAIN $W END $(date)" >> log_$W.txt
