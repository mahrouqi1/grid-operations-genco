#!/bin/bash
# dedicated worker with its own queue file (amendment 3c). usage: launch_v2q.sh <worker-id> <queue-file> ; waits for the original shared-queue worker <id> to exit first (cap: <=2 processes).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
W=$1; Q=$2
while pgrep -u $USER -f "launch_v2.sh $W\$" > /dev/null; do sleep 15; done
mkdir -p locks_v2q
i=0
while read arm study grid lim cfgs; do
  i=$((i+1))
  [ "$(date +%s)" -ge "$(date -d '2026-10-22 07:15' +%s)" ] && break
  mkdir "locks_v2q/${Q}_$i" 2>/dev/null || continue
  nice -n 15 $PY run_wls_v2.py --arm $arm --study $study --grids $grid --limit $lim --configs $cfgs --deadline "2026-10-22 07:15" >> logs_v2/${arm}_${study}_q$W.log 2>&1
done < $Q
echo "queue worker $W ($Q) done $(date)" >> logs_v2/workers_done.txt
