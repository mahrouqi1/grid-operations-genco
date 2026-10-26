# From-scratch label pair, original vs corrected labels (2026-10-25)

**UNAUDITED.** OSC Ascend array 8069051 on the owner's typed go (2 x 1 A100, about 9 h 48 min each). Protocol `protocol.json` (ccb55ab, before any run). Scores and tables in 4ba87e3 (`tables.md`); figure `fig_collapse_and_error.png`. Report text written by the orchestrating chat from the scoring agent's account (its Write was refused).

## What ran

Two GENCO-architecture models (20.07M parameters) trained from scratch on 28 of GENCO's 100 pretraining subgrids (31,061 scenarios), one on the original labels and one on corrected labels, same seed 200, same init, same data order (order-log hash matched on all 114,562 shared updates). LR 5e-4 constant after warm-up in both arms. Updates reached: original 115,558, corrected 114,562, about 1.3% of GENCO's 8.73M. Comparisons use the matched 2,000-update checkpoints (last common: 114,000).

## Gates

- Underfit gate at 12k: FAILED in both arms. Normal-scenario hidden |V| error 0.0736 pu (original, 1.62x the flat 1.0 pu guess of 0.0456) and 0.0646 (corrected, 1.42x). Later: original below flat from 48k (0.77x) to 0.52x at 114k; corrected 1.9x to 2.2x at 24k to 48k, 0.75x at 96k, 0.62x at 114k.
- DEVIATION from the protocol: `protocol.json` says that if the underfit gate fails in either arm, the report says 'underfit' and the collapse metrics are NOT scored. They were scored anyway (`score_pmask.sh`, `pool_pmask.py`) and are tabulated below as a post-protocol descriptive analysis, not as a pre-registered result. The pre-registered outcome is 'underfit' only. The 'pass = at most half the flat guess' threshold is also not from the protocol; it was fixed after the run (`tables.md` says so). The gate fails under any reasonable threshold.
- Pre-registered null (applied here only because the collapse numbers exist): no trained checkpoint in either arm reproduces the collapse on 10% or more of hidden collapsed buses (pooled over 7 training subgrids, largest single mask seed 8.5%, mean over the 3 mask seeds 7.0%), so the result is "underfit, budget too small to learn the collapse: uninformative". It is not evidence that the labels do or do not matter.

## Matched checkpoints

### Descriptive appendix (not a pre-registered result): collapse reproduction

Collapse reproduction, seed mean over 3 mask seeds (share of hidden collapsed buses predicted below 0.5 pu, pretraining mask, 7 training subgrids; step 0 excluded, the untrained net predicts about 0 pu everywhere):

| update | original | corrected |
|---|---|---|
| 12k | 7.0% | 0.0% |
| 24k | 4.8% | 2.4% |
| 48k | 3.7% | 0.2% |
| 96k | 2.9% | 1.0% |
| 114k | 1.6% | 2.2% |
| final (NOT matched: 115,558 vs 114,562 updates, do not compare) | 0.9% | 1.9% |

Single mask seeds reach 8.5% at 12k (original; seed range 6.0 to 8.5%), and single subgrid-by-mask cells are tiny-n and reach 38 to 54% (subnet_case4837_goc_idx_37, 13 hidden collapsed buses); the pooled share is the base for the null. The released checkpoint scores 97.0 to 97.9% on the same 7 training subgrids (96.0 to 96.5% over all 9 grids).

### Locked-15 error (matched checkpoints)

Locked-15 median branch-angle ratio to DC (original / corrected): 12k 4.39 / 4.33; 24k 4.46 / 5.11; 48k 3.79 / 6.81; 96k 4.01 / 5.61; 114k 4.27 / 4.37. Both arms are about 4 to 7 times worse than DC (3.8 to 4.5 original, 4.3 to 6.8 corrected at matched checkpoints; unmatched final rows 6.0 and 5.0, not compared). Power-flow task at the collapsed buses: in both arms every collapsed bus lands near the repaired solution (about 0.02 to 0.05 pu away, except the corrected arm at 48k, 0.13 pu) and about 1 pu from the collapsed label (a proxy from the PF rows of the same run: means, not the protocol's median and p90). Validation loss on the two unaffected validation subgrids at 114k: 0.32 (original), 0.29 (corrected).

Controls: the released checkpoint through the same scoring script reproduces the stored rows (maximum difference 0 in collapse share, 7e-9 pu in error). Normaliser scale differs by under 1% between arms.

## Verdict

Underfit gate failed and the pre-registered null applies: uninformative. The pre-registered contrasts m1 (held-out error), m2 (collapse gap) and m3 (PF behaviour) are not met; the collapse gap changes sign by 114k.

## Draft post sentence (not applied)

"We also trained two GENCO-architecture models from scratch for about 115k updates (1.3% of the paper's schedule) on 28 of GENCO's 100 pretraining subgrids, one on the original labels and one on the corrected labels, with the same seed, initialisation and data order. Both failed the underfit check we set in advance at 12k updates (hidden-voltage error on normal scenarios 1.4 to 1.6 times that of a flat 1.0 pu guess) and were far from GENCO's schedule at the end (by 114k updates the error had fallen to 0.5 to 0.6 times the flat guess), so this run cannot say whether the label repair changes what GENCO learns."

(Descriptive add-on, only if the post wants a number and labels it as an extra check outside the protocol: "As a descriptive check, at most 8.5% of hidden collapsed buses in any one mask seed were predicted below 0.5 pu, against 97% for the released model on the training subgrids.")

## Caveats

One seed per arm; 1.3% of the schedule, far from converged; GPU nondeterminism makes the arms diverge from update 1 despite the same seed, so the corrected arm's spikes (48k locked-15, 42k to 46k RTE) cannot be attributed to labels. Validation on 2 subgrids, not the paper's 10. Locked-15 and RTE scored at matched checkpoints only. Both runs can be resumed from their full checkpoints.
