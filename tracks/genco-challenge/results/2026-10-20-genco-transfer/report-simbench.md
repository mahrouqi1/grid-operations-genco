# SimBench-HV-only continued pretraining (arm H), 2026-10-24

**UNAUDITED.** OSC Ascend array 8059012 on the owner's typed go (3 x 1 A100, about 1 h 03 each, about 3.1 GPU-hours). Protocol `protocol-simbench.json` committed before any H score was read (09f06e1); scores, selection and scripts in 2b8997d. Tables: `simbench_summary.md` / `.json` (`report_simbench.py`). Report text written by the orchestrating chat from the scoring agent's account (its Write was refused).

## Key numbers (median branch-angle ratio to DC, locked 15; lower is better)

| basis | H (SimBench HV only) | D (ladder data, random mask) | C00 (original data only) | C01 | released |
|---|---|---|---|---|---|
| at 12,000 updates | 4.402 | 5.703 | 4.477 | 4.451 | 3.692 |
| validation-selected | 4.145 | 3.899 | 3.985 | 3.692 | 3.692 |

H per seed: 4.79 / 4.48 / 3.92 at 12k; 3.98 / 4.42 / 4.09 selected. H beats released on 1 of 15 grids at 12k and 2 of 15 selected. |V|: H median error 0.0070 pu at 12k (D 0.0208, released 0.0065); beats DC on |V| on 13 of 15. Go/no-go: criteria 1 to 3 fail (median 4.145 vs 2.77; 2 of 15 vs 11; every seed worse than released), criterion 4 passes: NO-GO, as every main-round arm.

## Bottom line

SimBench novelty did not rescue transfer. H is no better than the released model on the locked 15, and at 12,000 updates it sits within 1.1 to 1.7% of the controls trained only on GENCO's own data (H 4.40, C00 4.48, C01 4.45), which never saw a new family. What separates D from H is D's extra damage (|V| error 0.0208 pu vs 0.0070 for H and 0.0065 for released), which plausibly comes from D's large-grid generated data rather than from family novelty (a hypothesis, not tested). So "H beats D" mostly means D was hurt by its data, not that H learned to transfer.

**Pre-registered outcome: partial/unresolved.** `protocol-simbench.json` says an in-between result "is reported as partial/unresolved, not forced into either reading", and that is where arm H lands. Family-familiarity reading: needs H clearly beating D on both the 12k and the selected comparison. The 12k half is met (H 4.402 vs D 5.703, a 22.8% gap, threshold 15%), but the selected half fails (H 4.145 is worse than D 3.899, and both beat released on 2 of 15 grids). Recipe reading: needs H at or above 4.85 at 12k and no more grids better than released than D (D: 0). H is 4.402, below 4.85, and beats released on 1 grid at 12k against D's 0. Neither is met. Seeds are unpaired, and by value H's best seed (3.92) vs D's best (4.60) shows no clean seed-wise gap.

**Evidence on the family/scale side that must not be left out: H is the best arm on the RTE validation grids.** Seed-mean validation metric (the quantity used for selection; lower is better): H 6.52, against 7.47 to 8.88 for the other ladder arms (a 7.85, b 8.56, c 7.47, m 7.49, D 8.88), C00 7.79, and 9.24 for C01 and the released model (which always picks step 0). At matched 12k, not selected, the RTE-validation ratio is H 6.71 / 7.32 / 7.13 (mean 7.05) against C00 12.4 / 10.1 / 9.1, D 31.8 / 23.5 / 14.6 and C01 23.3 / 34.3 / 34.5. At matched 12k updates H is clearly the best arm on RTE validation; at the selected checkpoints, arms c, m, a and C00 also improve on released by 15 to 19% (audit R2-F3), and RTE is the set used to select checkpoints, so this is not an independent held-out result. Caveat: that is one validation family, and the locked-15 benchmark does not show the gain.

**Post-hoc, not a pre-registered criterion:** on the locked 15, H is about as bad as the original-data-only controls (H 4.40, C00 4.48, C01 4.45). That comparison leans toward "the recipe limits transfer", but the protocol lists C00/C01 as a contrast, not as a decision rule, and the RTE-validation result above leans the other way. We therefore do not lean either way. D's extra damage (|V| error 0.0208 pu vs 0.0070 for H and 0.0065 for released) plausibly comes from its large-grid generated data (a hypothesis, not tested). With 3 seeds, per-seed spread (H 3.92 to 4.79) is comparable to the H vs C gaps.

## Draft post sentence (not applied)

"We also continued pretraining on the SimBench high-voltage grids, the only grids from families GENCO never saw, together with GENCO's own corrected training subgrids (3 seeds). Judged by the rules we fixed in advance, the result is partial: at 12,000 updates it beat the run on our mostly-familiar generated data, but at the validation-selected checkpoint it did not, and on the 15 locked grids it was no better than the released model (median branch-angle error 4.4 times DC at 12,000 updates, against 3.7). So this control cannot tell whether the training recipe or unfamiliar grid families limits transfer."

(Audit R2-F1 to F3 applied: the arm also trains on the 7 corrected replay subgrids; the post-hoc comparisons with the original-data-only controls and with RTE validation, which is the selection set, are kept in the report but out of the sentence.)

**Post status note:** `post-draft-v3.4-DRAFT.md` lines 122, 124 and 213 and `claim-register-v3.4.md` line 80 (claim L3) still say "We did not run a SimBench-HV-only run"; that is false now that arm H is reported. They need updating together with this report. Not edited here.

## Caveats

Seeds are not paired across arms. H differs from D in grid size (64 to about 100 buses vs up to 9,591), batch composition (32 graphs vs about 16) and passes over the generated data (about 19 vs 4 at 12k). One control arm, one validation family. The locked 15 shaped earlier work, so it is a benchmark, not a blind test.
