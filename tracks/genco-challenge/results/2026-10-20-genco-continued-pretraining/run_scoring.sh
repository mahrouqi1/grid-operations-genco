#!/bin/bash
# Scoring of B (continued pretraining from the released checkpoint, local 3090s, 2026-10-19), per protocol.json.
# Zero-shot on the 15 held-out grids at the final state and at step 500 (score_retrain via a run-root shim whose
# final/best_model_state_dict.pt links to steps/state_<tag>.pt), then collapse reproduction under the pretraining
# mask (mask seeds 1000-1002) for every saved step of both arms. Run from anywhere; env grid-operations-genco.
set -eu   # audit R1-F7: a failed step stops the script; SCORE_DONE only on success
cd <REPO>
export PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=tracks/genco-challenge/src JULIA_DEPOT_PATH=<JULIA_DEPOT>
export PATH=<CONDA_ENVS>/grid-operations-genco/bin:$PATH
R=tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining
S=experiments/b-continue/experiments/genco-retrain-cont
GPU=${GPU:-1}
for tag in final step500; do [ -f $R/zeroshot-$tag/tally.md ] && continue
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python -m genco_challenge.score_retrain --out $R/zeroshot-$tag --run-root experiments/b-continue/score-$tag
done
mkdir -p $R/pmask
for st in step0 final step500 step1000 step100 step250 step50 step25; do for arm in fixed published; do
  F=$R/pmask/${arm}_$st.jsonl
  [ -s $F ] && [ "$(grep -c . $F)" -ge 9 ] && continue     # audit R1-F7: a partial file (< 9 subgrids) is redone
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python tracks/genco-challenge/results/2026-10-09-collapsed-label-pretrain-mask/run_pretrain_mask.py \
    --sets pretrain --ckpt $S/$arm/steps/state_$st.pt --out $F.tmp && mv $F.tmp $F
done; done
touch $R/SCORE_DONE
