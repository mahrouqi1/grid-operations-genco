# GENCO retrained with the fixes vs the published recipe: zero-shot and collapsed-label scoring (2026-10-19)

> **DO NOT USE THESE NUMBERS (2026-10-19 15:35, audit genco-retrain round 1, R1-F1/F2).** Both arms are underfit:
> on normal scenarios of their own training subgrids their hidden-|V| error (0.03 to 0.11 pu) is no better than a
> flat 1.0 pu guess (0.02 to 0.07); the released checkpoint gets 0.001 to 0.004. Cause: GENCO's schedule (LR
> patience 5 epochs, early-stopping patience 100 checks) was tuned for epochs over about 100 subgrids; with
> 124-step epochs the LR fell below 1e-6 at about epoch 130 to 147 in both arms. A rerun with a step-scaled
> schedule is queued. Everything below describes this first, underfit run.

UNAUDITED. Retrain: OSC Pitzer job 56391920 (2026-10-19 10:07 to 14:32), billed to <OSC_PROJECT>, 2 x V100-32G per arm,
`slurm/genco_retrain_body.sh` with the an owner decision variant (`DROP_SUBS=subnet_case9591_goc_idx_12`, `FILTER90=1`,
`RUNTAG=pitzer`). Checkpoints: `experiments/genco-retrain-pitzer/{fixed,published}/final/` (gitignored). Scoring:
`run_scoring.sh` (`score_retrain` zero-shot, `run_pretrain_mask.py --ckpt`), `compare_arms.py`, `pmask_pool.py`.
The case118 few-shot column is running (`fewshot/`); it is not in this report yet.

## What was trained

Both arms are pretrained FROM SCRATCH with GENCO's published pretraining config (`HGNS_PreTrain_subnets_100.yaml`) on
the 9 pretraining subgrids that contain collapsed scenarios (7 for training, validation on
`subnet_case19402_goc_idx_47`; the 46-percent >90-degree subgrid `9591_goc_idx_12` dropped from both arms, an owner decision).
- **fixed**: corrected labels (relabeled physical roots), the 4 remaining >90-degree training scenarios dropped, and a
  1,000-step linear warm-up. Ran all 300 epochs (4 h 25 min); best validation loss at epoch 278.
- **published**: the original published labels and recipe. Early-stopped by the recipe's own patience (100) at epoch
  222 (3 h 17 min); best validation loss at epoch 122.

**Scope, read this first:** the released GENCO checkpoint was pretrained on GENCO's full subgrid set; these two arms
saw 7 subgrids. Absolute numbers of the arms are therefore NOT comparable to the released checkpoint. The
comparison this experiment supports is fixed against published at equal data. One seed per arm.

## Bottom line

1. **At equal data, the fixed pipeline is better than the published one on most held-out grids.** Fixed beats
   published on branch angle on 13 of 15 grids (median ratio 0.62), on voltage magnitude on 14 of 15 (0.51) and on
   residual on 11 of 15 (0.87) (`compare_arms.md`). The two exceptions on branch angle are the Polish grids
   (case2736sp_k, case3120sp_k). The arms differ in labels, warm-up and the 4 filtered scenarios together, and they
   trained for different lengths (300 against 222 epochs, by the recipe's early stopping), so this does not isolate
   one fix. One seed.
2. **Neither retrained arm approaches the released checkpoint zero-shot, and neither beats DC.** Against DC: residual
   0 of 15 for both arms (released 0 of 15), branch angle 0 of 15 for both (released 1 of 15), voltage magnitude 8 of
   15 (fixed) and 5 of 15 (published) against the released 13 of 15 (`zeroshot/tally.md`). This is the expected cost
   of pretraining on 7 subgrids instead of the full set; it says nothing about the fixes.
3. **Collapsed-label reproduction: the released checkpoint's 96 percent is not reproduced by retraining the published
   recipe on these subgrids.** Under the pretraining mask on the 9 affected subgrids (scored against the published
   labels), the share of hidden collapsed buses predicted below 0.5 pu is 96 percent for the released checkpoint, 0.0
   percent for the fixed arm and 2 to 3 percent for the published arm (3 mask seeds; `pretrain_mask_pooled.md`). The
   published arm trained on the same collapsed labels yet predicts close to the physical root (MAE to the root 0.16
   pu, fixed 0.15, released 0.99). So the 0 percent of the fixed arm cannot be credited to the label fix alone: what
   makes the released model reproduce collapsed labels (training on the full corpus, its length, or capacity) is
   not present in a 7-subgrid retrain. These subgrids are training data for both arms.

## What this means for the post

The retrain answers "do the fixes help at equal data": yes on most grids, one seed, with the fixes not separated. It
does not answer "does a retrained GENCO beat DC": at this data scale it does not, and the released checkpoint remains
the stronger model. The post's retrain table should carry this scope line verbatim.

## Limits

One seed per arm; 7 training subgrids; the fixes are confounded with each other and with training length; validation
on one subgrid; the collapsed-label check is on training subgrids. The few-shot case118 column is pending.
