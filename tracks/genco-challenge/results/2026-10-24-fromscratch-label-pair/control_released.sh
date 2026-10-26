#!/bin/bash
# Identity control: the released checkpoint through the SAME run_pretrain_mask.py invocation; must reproduce
# ../2026-10-20-genco-continued-pretraining/pmask/published_step0.jsonl (compared by control_compare.py).
set -eu
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src JULIA_DEPOT_PATH=<JULIA_DEPOT>
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair
CUDA_VISIBLE_DEVICES=${GPU:-0} nice -n 10 python tracks/genco-challenge/results/2026-10-09-collapsed-label-pretrain-mask/run_pretrain_mask.py \
  --sets pretrain --ckpt data/external/genco-pf-transfer/pretraining/best_model_state_dict.pt --out $R/control/released.jsonl.tmp && mv $R/control/released.jsonl.tmp $R/control/released.jsonl
