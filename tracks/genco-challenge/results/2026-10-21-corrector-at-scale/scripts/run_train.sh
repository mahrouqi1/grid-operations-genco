#!/bin/bash
# usage: run_train.sh <case> <n> [variant]   (GPU 1 only)
source "$(dirname "$0")/env.sh"
R=tracks/genco-challenge/results/2026-10-21-corrector-at-scale
V=${3:-p0.1_lr0.0005}
nice -n 10 python $R/scripts/scale.py train --case $1 --n $2 --variant $V > $R/logs/train_$1_n$2_$V.log 2>&1
echo done > $R/logs/train_$1_n$2_$V.done
