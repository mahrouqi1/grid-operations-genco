#!/bin/bash
# m1 on the locked 15 (score_ckpts, the post's pipeline, zero-shot, no normalizer file) at the given step tags for ONE arm.
# usage: ARM=original GPU=0 bash score_l15.sh step12000 step24000 ... final   -> l15/<arm>/results.jsonl (resumable)
set -eu
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-0}
R=tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair
E=experiments/genco-fromscratch-pair
arm=$ARM; specs=""; for tag in "$@"; do specs="$specs ${arm}_$tag=$E/$arm/steps/state_$tag.pt"; done
nice -n 10 python -m genco_challenge.score_ckpts --out $R/l15/$arm --ckpt $specs
