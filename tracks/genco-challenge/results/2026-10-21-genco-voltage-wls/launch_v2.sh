#!/bin/bash
# Round-1 audit arms. Two workers share jobs_v2.txt (in order) via mkdir-lock claims. CPU only, 1 thread each, nice 15. Hard stop 07:15 EDT (run_wls_v2.py --deadline).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
mkdir -p locks_v2 logs_v2
W=$1; i=0
while read arm study grid lim cfgs; do
  i=$((i+1))
  mkdir "locks_v2/$i" 2>/dev/null || continue
  nice -n 15 $PY run_wls_v2.py --arm $arm --study $study --grids $grid --limit $lim --configs $cfgs --deadline "2026-10-22 07:15" >> logs_v2/${arm}_${study}_w$W.log 2>&1
  [ "$(date +%s)" -ge "$(date -d '2026-10-22 07:15' +%s)" ] && break
done < jobs_v2.txt
echo "worker $W done $(date)" >> logs_v2/workers_done.txt
