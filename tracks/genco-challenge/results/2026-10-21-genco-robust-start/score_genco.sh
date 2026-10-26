#!/bin/bash
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src CUDA_VISIBLE_DEVICES=0
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
EVAL_BATCH=${EVAL_BATCH:-0} python tracks/genco-challenge/results/2026-10-21-genco-robust-start/score_genco.py > tracks/genco-challenge/results/2026-10-21-genco-robust-start/score_genco.log 2>&1
