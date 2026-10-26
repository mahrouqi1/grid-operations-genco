#!/bin/bash
# run verdict_v2.py while keeping the <=2-process cap: SIGSTOP worker 2's python while the verdict runs
cd "$(dirname "$0")"
PY=<CONDA_ENVS>/grid-operations-genco/bin/python
P=$(pgrep -u $USER -f "python run_wls_v2.py" | tail -n 1)
[ -n "$P" ] && kill -STOP $P
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" nice -n 15 $PY verdict_v2.py "$@"
[ -n "$P" ] && kill -CONT $P
