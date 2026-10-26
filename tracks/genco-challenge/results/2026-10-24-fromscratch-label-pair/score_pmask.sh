#!/bin/bash
# m2/m3 source: collapse reproduction under the pretraining mask (run_pretrain_mask.py, unchanged) for the given step tags.
# usage: GPU=0 bash score_pmask.sh step12000 step24000 ... final   (arm = original|corrected; files pmask/<arm>_<tag>.jsonl)
set -eu
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src JULIA_DEPOT_PATH=<JULIA_DEPOT>
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair
E=experiments/genco-fromscratch-pair
mkdir -p $R/pmask
for tag in "$@"; do for arm in original corrected; do
  F=$R/pmask/${arm}_$tag.jsonl
  [ -s $F ] && [ "$(grep -c . $F)" -ge 9 ] && continue
  CUDA_VISIBLE_DEVICES=${GPU:-0} nice -n 10 python tracks/genco-challenge/results/2026-10-09-collapsed-label-pretrain-mask/run_pretrain_mask.py \
    --sets pretrain --ckpt $E/$arm/steps/state_$tag.pt --out $F.tmp && mv $F.tmp $F
done; done
