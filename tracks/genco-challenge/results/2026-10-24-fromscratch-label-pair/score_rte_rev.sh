#!/bin/bash
# Second scoring stream for case2848rte, tags in REVERSE order, into rte_rev/<arm> (merged with rte/<arm> by make_tables.py).
set -eu
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-0}
R=tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair
E=experiments/genco-fromscratch-pair
arm=$ARM; specs=""; for tag in $(tac -s' ' <<< "$(cat $R/rte_tags.txt)" | tr ' ' '\n' | tac | tr '\n' ' ' | xargs -n1 | tac | xargs); do specs="$specs ${arm}_$tag=$E/$arm/steps/state_$tag.pt"; done
nice -n 10 python -m genco_challenge.score_ckpts --out $R/rte_rev/$arm --ckpt $specs --grids case2848rte --data-root data/external/genco-transfer-val
