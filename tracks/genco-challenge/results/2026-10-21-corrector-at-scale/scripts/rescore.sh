#!/bin/bash
source "$(dirname "$0")/env.sh"
R=tracks/genco-challenge/results/2026-10-21-corrector-at-scale
for c in case2869pegase case6470rte; do nice -n 10 python $R/scripts/scale.py eval --case $c --n 100 > $R/logs/rescore_$c.log 2>&1; done
echo done > $R/logs/rescore.done
