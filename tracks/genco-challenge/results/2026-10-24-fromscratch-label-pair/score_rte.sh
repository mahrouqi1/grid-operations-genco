#!/bin/bash
# m1 on the 2 RTE validation grids (data/external/genco-transfer-val, as the transfer study) at the given step tags, both arms.
# usage: GPU=0 bash score_rte.sh step0 step2000 ... final      -> rte/<arm>/results.jsonl (resumable)
set -eu
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-0}
R=tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair
E=experiments/genco-fromscratch-pair
for arm in original corrected; do
  specs=""; for tag in "$@"; do specs="$specs ${arm}_$tag=$E/$arm/steps/state_$tag.pt"; done
  nice -n 10 python -m genco_challenge.score_ckpts --out $R/rte/$arm --ckpt $specs --grids case1888rte case2848rte --data-root data/external/genco-transfer-val > $R/rte_$arm.log 2>&1
done
