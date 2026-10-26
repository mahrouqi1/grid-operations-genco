#!/bin/bash
cd <REPO>
export PYTHONNOUSERSITE=1 PYTHONPATH=core:tracks/genco-challenge/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-21-genco-robust-start
GRIDS=$(python -c "import json;print(' '.join(g for f in json.load(open('$R/protocol.json'))['grids'].values() for g in f))")
for g in $GRIDS; do
  # skip a finished grid, or one the other worker (run_rev.sh) has claimed (audit genco-make-it-work R1-F8)
  { [ -f $R/per_scenario/$g.csv ] || [ -f $R/running_$g ]; } && continue
  touch $R/running_$g
  echo "load $(cat /proc/loadavg)" >> $R/run_all.log
  taskset -c 16-19 python $R/run_starts.py $g >> $R/run_all.log 2>&1
done
# RUN_DONE only when every per_scenario CSV exists (a grid claimed by the other worker may still be running)
ALL=1; for g in $GRIDS; do [ -f $R/per_scenario/$g.csv ] || ALL=0; done
[ $ALL = 1 ] && touch $R/RUN_DONE
