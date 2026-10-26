# Continued pretraining, corrected vs original labels: three seeds (2026-10-24)

**UNAUDITED.** Seeds 201 and 202 added to seed 200 (protocol_amendment_seeds.json; protocol_amendment_seeds_2.json replaced the step-25 weight gate, which failed at 5.86e-4 against 1e-4, by a config-identity plus steps 1-6 loss check within 1e-2; that change was approved by the owner via the hub relay, 14:18). Caveats on that amendment, stated plainly: the replacement gate was chosen after the data were seen; the weight-level identity gate FAILED and is not passed; the config-identity half cannot fail (the seed-200 and seed-201/202 configs differ only in the `seed:` line); and the steps 1-6 window with 1e-2 tolerance stops just before the loss difference between seed 200 and its rerun first exceeds 1e-2 (steps 1-8 are all below 1e-2, step 9 is 1.5e-2, step 24 is 1.8e-1; single-batch losses, noisy), so a window of 1-8 passes and 1-9 fails and 1-6 is not a natural cut. The four trainings are unaffected by the gate; what it does not support is a claim that seeds 201/202 'passed the original identity check'. The s201 corrected collapse scoring was restarted once (the first run stopped after 6 of 9 grids; the complete file's first 6 lines are byte-identical to the stale partial output). Pooled by `pool_seeds.py` (`pooled_seeds.json`), which reproduces the seed-200 numbers in `../pooled.json` and `../preservation.json` to 1e-12. Each seed is divided by its own released-model row (rows differ by at most 2.4e-5 relative across score runs). Report written by the orchestrating chat from the pooling agent's account (its Write was refused). Commit 76f90ee.

## Per seed

| arm | seed | collapse share < 0.5 pu (mask 1000 / 1001 / 1002) | branch-angle ratio vs released | \|V\| ratio vs released | angle better than released (of 15) |
|---|---|---|---|---|---|
| corrected | 200 | 0.834 / 0.835 / 0.846 | 1.162 | 1.347 | 3 |
| corrected | 201 | 0.698 / 0.685 / 0.692 | 1.171 | 1.380 | 4 |
| corrected | 202 | 0.801 / 0.799 / 0.829 | 1.089 | 2.209 | 2 |
| original | 200 | 0.937 / 0.941 / 0.935 | 1.441 | 2.307 | 0 |
| original | 201 | 0.967 / 0.973 / 0.980 | 1.326 | 1.927 | 0 |
| original | 202 | 0.923 / 0.941 / 0.935 | 1.298 | 2.559 | 2 |

Released checkpoint collapse share: 0.970 / 0.973 / 0.979.

## Median over 3 seeds (min to max)

| arm | collapse, mask 1000 | angle ratio | \|V\| ratio |
|---|---|---|---|
| corrected | 0.801 (0.698 to 0.834) | 1.162 (1.089 to 1.171) | 1.380 (1.347 to 2.209) |
| original | 0.937 (0.923 to 0.967) | 1.326 (1.298 to 1.441) | 2.307 (1.927 to 2.559) |

## Reading

1. Corrected labels lower the collapse reproduction more than original labels in every seed; the ranges do not overlap (corrected max 0.846, original min 0.923). The size varies with seed (corrected 0.69 to 0.83).
2. Continued pretraining costs held-out accuracy in both arms: all 12 ratios (median over the 15 held-out grids of the ratio to released) exceed 1 (lowest 1.089); a median statement, since it still beats released on 2 to 4 (corrected) and 0 to 2 (original) of 15 grids.
3. Original labels do more damage than corrected on branch angle (ranges do not overlap) but NOT separably on |V| (1.35 to 2.21 vs 1.93 to 2.56 overlap), so by the amendment the |V| comparison is dropped.

## Draft post sentence (not applied)

"Over three training seeds per label set, continued pretraining made held-out error worse than the released model's in both arms (median over the 15 held-out grids of the ratio to released; the corrected-label models still beat released on 2 to 4 of 15 grids on branch angle): median branch-angle error 1.16 times released with corrected labels (range 1.09 to 1.17) and 1.33 times with original labels (1.30 to 1.44); median |V| error 1.38 times (1.35 to 2.21) and 2.31 times (1.93 to 2.56), so only the branch-angle gap is separated by seed. The share of hidden collapsed buses still predicted below 0.5 pu was 0.80 (0.70 to 0.83) with corrected labels and 0.94 (0.92 to 0.97) with original labels (mask seed 1000), lower with corrected labels in all three seeds."

## Caveats

Three seeds, one starting checkpoint, one learning rate: ranges are the min and max of 3 values, not intervals. Mask-seed spread within a run (at most 0.03) is much smaller than training-seed spread (up to 0.15).
