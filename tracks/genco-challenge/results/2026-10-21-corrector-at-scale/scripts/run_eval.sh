#!/bin/bash
# usage: run_eval.sh <case> <n> [variant]   waits for the trained run, then evaluates it on the test split (GPU 1)
source "$(dirname "$0")/env.sh"
R=tracks/genco-challenge/results/2026-10-21-corrector-at-scale
V=${3:-p0.1_lr0.0005}
K=data/external/genco-corrector-scale/fs/$1/runs/c3_${V}_d0__finetune__$1__n$2.json
until [ -f $K ]; do sleep 30; done
nice -n 10 python $R/scripts/scale.py eval --case $1 --n $2 --variant $V > $R/logs/eval_$1_n$2_$V.log 2>&1
echo done > $R/logs/eval_$1_n$2_$V.done
