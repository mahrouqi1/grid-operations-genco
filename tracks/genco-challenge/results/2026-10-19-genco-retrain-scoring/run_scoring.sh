#!/bin/bash
# Scoring of the 10-19 retrain (pitzer 56391920), run 2026-10-19 from the repo root. Env grid-operations-genco.
set -u
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src JULIA_DEPOT_PATH=<JULIA_DEPOT>
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-19-genco-retrain-scoring
RR=experiments/genco-retrain-pitzer
case "$1" in
  zeroshot) CUDA_VISIBLE_DEVICES=1 nice -n 10 python -m genco_challenge.score_retrain --out $R/zeroshot --run-root $RR ;;
  pmask) for ARM in fixed published; do CUDA_VISIBLE_DEVICES=0 nice -n 10 python tracks/genco-challenge/results/2026-10-09-collapsed-label-pretrain-mask/run_pretrain_mask.py --sets pretrain --ckpt $RR/$ARM/final/best_model_state_dict.pt --out $R/pretrain_mask_$ARM.jsonl; done ;;
  fewshot0) CUDA_VISIBLE_DEVICES=0 nice -n 10 python -m genco_challenge.score_retrain_fewshot --out $R/fewshot --run-root $RR --arms fixed ;;
  fewshot1) CUDA_VISIBLE_DEVICES=1 nice -n 10 python -m genco_challenge.score_retrain_fewshot --out $R/fewshot --run-root $RR --arms published released ;;
esac
