#!/bin/bash
# timing, GPU 1 only, CPU pinned to cores 16-19 as the timing table; waits for the 9241 evaluation (no other GPU job of ours running)
source "$(dirname "$0")/env.sh"
R=tracks/genco-challenge/results/2026-10-21-corrector-at-scale
true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TIMING_JSONL=$PWD/$R/timing.jsonl
for c in case2869pegase case6470rte case9241pegase; do
  { echo "== $c $(date +%T)"; uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; } >> $R/logs/timing_env.log
  taskset -c 16-19 python $R/scripts/timing.py $c --reps 5 > $R/logs/timing_$c.log 2>&1
  { uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; } >> $R/logs/timing_env.log
done
echo done > $R/logs/timing.done
