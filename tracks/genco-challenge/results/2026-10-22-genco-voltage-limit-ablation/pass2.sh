#!/bin/bash
# protocol_amendment_2: after worker $1's pass-1 chain ends: pass 2a (cap 3100) then 2b (uncapped), seen then unseen, ascending, until the deadline.
cd "$(dirname "$0")"
W=$1
until grep -q "CHAIN $W END" log_$W.txt 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
for capargs in "--cap 3100" "--nocap"; do
 for set in seen_whole unseen_whole; do
  if [ "$(date +%s)" -ge "$(date -d '2026-10-22 07:30' +%s)" ]; then break 2; fi
  nice -n 15 $PY run_ablate.py --study oos --set $set --worker ${W}p2 $capargs >> log_${W}p2.txt 2>&1
 done
done
echo "PASS2 $W END $(date)" >> log_${W}p2.txt
