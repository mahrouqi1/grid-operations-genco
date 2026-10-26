#!/bin/bash
# resumed with --limit 30 scenarios per piece/grid (compute); at most 2 concurrent run_wls.py (gate), 1 thread each, nice 15
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
run() { while [ "$(pgrep -u $USER -fc 'python run_wls.py')" -ge 2 ]; do sleep $((20 + RANDOM % 20)); done
  nice -n 15 $PY run_wls.py --study $1 --grids $2 --limit $3 >> logs/$1.log 2>&1; }
export -f run; export PY
while read s g l; do run $s $g $l &
 sleep 45
 while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 20; done
done < jobs2.txt
wait; echo ALLDONE > logs/ALLDONE
