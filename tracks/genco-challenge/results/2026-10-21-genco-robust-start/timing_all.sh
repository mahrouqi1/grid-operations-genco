#!/bin/bash
cd <REPO>
export PYTHONNOUSERSITE=1 PYTHONPATH=tracks/genco-challenge/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-21-genco-robust-start
for g in $(python -c "import json;print(' '.join(g for f in json.load(open('$R/protocol.json'))['grids'].values() for g in f))"); do
  [ -f $R/timing/$g.json ] || taskset -c 16-19 python $R/timing_genco.py $g > $R/timing_$g.log 2>&1
done
touch $R/TIMING_DONE
