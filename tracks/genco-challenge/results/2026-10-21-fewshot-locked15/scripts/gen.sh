#!/bin/bash
cd <REPO>
R=tracks/genco-challenge/results/2026-10-21-fewshot-locked15
source $R/scripts/env.sh
exec python $R/scripts/gen_missing.py "$@" >> $R/logs/gen_$1.log 2>&1
