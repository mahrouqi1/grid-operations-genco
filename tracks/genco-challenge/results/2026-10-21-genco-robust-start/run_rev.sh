#!/bin/bash
# second worker, reverse grid order, different cores (added to shorten wall clock; timing is per-scenario on a shared loaded box either way)
cd <REPO>
export PYTHONNOUSERSITE=1 PYTHONPATH=core:tracks/genco-challenge/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-21-genco-robust-start
for g in $(python -c "import json;print(' '.join(reversed([g for f in json.load(open('$R/protocol.json'))['grids'].values() for g in f])))"); do
  [ -f $R/per_scenario/$g.csv ] || [ -f $R/running_$g ] && continue
  touch $R/running_$g; echo "w2 $g load $(cat /proc/loadavg)" >> $R/run_rev.log
  taskset -c 12-15 python $R/run_starts.py $g >> $R/run_rev.log 2>&1
done
touch $R/RUN_REV_DONE
