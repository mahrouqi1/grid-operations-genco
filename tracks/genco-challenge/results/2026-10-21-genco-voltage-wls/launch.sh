#!/bin/bash
# 2 workers max, 2 threads each... (coordinator cap: <=2 cores total) -> 2 workers x 1 thread
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
cat jobs.txt | xargs -P2 -L1 bash -c 'lim=""; [ "$2" != "0" ] && lim="--limit $2"; nice -n 10 '$PY' run_wls.py --study $0 --grids $1 $lim >> logs/$0.log 2>&1'
echo ALLDONE > logs/ALLDONE
