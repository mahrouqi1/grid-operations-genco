#!/bin/bash
# usage: launch.sh w1|w2   (ONE GPU = 0, at most two streams)
cd <REPO>
R=tracks/genco-challenge/results/2026-10-21-fewshot-locked15
source $R/scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
exec python $R/scripts/run.py worker --name $1 >> $R/logs/$1.log 2>&1
