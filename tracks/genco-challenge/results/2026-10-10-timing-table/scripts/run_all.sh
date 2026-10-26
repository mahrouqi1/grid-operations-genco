#!/bin/bash
# Driver for the H3 timing table. Blocks run strictly one after another on cores 16-19 (CPU) and GPU 0; before each block wait (up to 20 min)
# for the 1-minute load average to be <= 20, then run regardless (the Block row records the load, and marks it noisy above 25).
# usage: run_all.sh <block-list file>   (lines: "<kind> <grid> [extra args]")
D=$(cd "$(dirname "$0")" && pwd)
source $D/env.sh
export TIMING_JSONL=$D/../timing.jsonl TT_SCRATCH=${TT_SCRATCH:?} CUDA_VISIBLE_DEVICES=0
LOG=$D/../logs
mkdir -p $LOG
while read -r kind grid extra; do
  [ -z "$kind" ] && continue
  for w in $(seq 1 40); do
    L=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l=$L 'BEGIN{exit !(l<=20)}' && break
    echo "$(date +%T) load $L > 20, waiting before $kind $grid" >> $LOG/driver.log
    sleep 30
  done
  echo "$(date +%T) START $kind $grid $extra load $(cut -d' ' -f1-3 /proc/loadavg)" >> $LOG/driver.log
  case $kind in
    classical) OMP_NUM_THREADS=1 taskset -c 16-19 python $D/t_classical.py $grid $extra > $LOG/classical_$grid.log 2>&1 ;;
    pm)        OMP_NUM_THREADS=1 taskset -c 16-19 python $D/t_pm.py $grid $extra > $LOG/pm_$grid.log 2>&1 ;;
    genco)     OMP_NUM_THREADS=1 taskset -c 16-19 python $D/t_genco.py $grid $extra > $LOG/genco_${grid}_$(echo $extra | tr -d ' -').log 2>&1 ;;
  esac
  echo "$(date +%T) END $kind $grid rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" >> $LOG/driver.log
done < "$1"
