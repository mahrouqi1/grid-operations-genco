# Mask-fraction sweep, zero-shot voltage inference (UNAUDITED, descriptive only)

2026-10-26. Protocol `protocol.json` (sha256 starts `bc21a0ee`), committed in f5cce84 before any run; results in a61a878. Report text written by the orchestrating chat from the run agent's account (its Write was refused).

The released GENCO checkpoint and the five interpolators I1 to I5 ran at measured fractions 10% to 90% in steps of 10%, mask seed 0, on the locked 15 and on the 36 carved pieces (19 control plus 17 family-vs-size). Methods, scenarios, metric and win rule are those of `2026-10-21-genco-voltage-inference`. The recorded 20% primary verdict (8 of 15, three mask seeds pooled, FAIL) stands as recorded; this sweep makes no new pass/fail.

## Wins per fraction

| measured | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|
| locked 15 (of 15) | 6 | 7 | 11 | 12 | 12 | 12 | 12 | 12 | 12 |
| locked 15, point ratio below 1 only | 6 | 9 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| 36 carved pieces (of 36) | 16 | 27 | 31 | 32 | 33 | 33 | 32 | 30 | 27 |
| 36 pieces, point ratio below 1 only | 21 | 29 | 32 | 33 | 33 | 33 | 33 | 32 | 30 |

- The locked-15 value at 20% (7) is mask seed 0 only (`f20_seed0` in the 2026-10-21 verdict); the recorded primary pools three seeds (8).
- GENCO wins on none of case240, case300 and case3120 at any fraction; case2736 becomes a win at 40%.
- The 24 ladder-regenerated pieces were not used.

## Control reproduction (f10_s0, f20_s0, f40_s0 against the stored 2026-10-21 per-scenario files)

| set | max abs diff, GENCO (limit 5e-5) | max abs diff, I1 to I5 (limit 1e-9) | cells over limit |
|---|---|---|---|
| locked 15 | 3.2e-5 | 0 | 0 of 270 |
| 19 control pieces | 7.9e-7 | 0 | 0 of 342 |
| 17 family-vs-size pieces | 4.4e-6 | 0 | 0 of 306 |

Locked-15 win flags at 10%, 20% (seed 0) and 40% match the recorded flags on 15 of 15 grids each; largest ratio difference 1.3e-4.

## Runtimes

00:33 to 00:59 EDT wall clock. GENCO on one GPU: 230 s for the locked 15 at nine fractions, 1,125 s for the 36 pieces. Interpolators on 5 CPU workers: 7,735 CPU-seconds; the longest task was case3120 at 90% (1,030 s).

## Deviations from protocol.json

- The bootstrap uses one `default_rng(7)` stream consumed in sequence, as the protocol states. Point ratios match the recorded ones exactly; the confidence intervals at 10, 20 and 40% are not the recorded draws.
- Raw per-scenario files (51 MB) are gitignored; the sha256 of all 112 files is in `raw_sha256.txt`.
- 5 interpolator workers plus 1 GENCO process (6 in total).

Figure: `fig_sweep.png` (locked 15 on the left axis, of 15; pieces on the right axis, of 36), checked by `check_sweep.py`.

## Audit (fresh agent, 2026-10-26 01:05; no invalidating findings)

All 459 cells recompute exactly from the raw files; the protocol commit (f5cce84, 00:32:11) precedes every run output (first at 00:37:04); the wrappers change only CONFIGS and output paths, and the masks at each fraction are cuts of one seeded permutation, so they are nested and equal the earlier 10/20/40% masks. Fixes from the audit:

- The 36-piece counts are mask seed 0 only. The earlier carved-piece verdicts pool three mask seeds at 40% (33/36 in the post); here 40% gives 32/36. The one cell that differs is `carved_case2848_rte_c4` (pooled: ratio 0.970, CI upper 0.984, win; seed 0: ratio 0.958, CI upper 1.0001, not a win), a boundary cell under any bootstrap stream. "Point ratios match the recorded ones exactly" above holds for the locked 15 (to 1.3e-4); for pieces the recorded ratios are pooled over three seeds and differ in the second decimal.
- Several cells sit within about 0.03 of the CI boundary (10%: case20758_epigrids_c0, case4837_goc_c1, case2383wp_k_c41; 20%: case2869_pegase_c1, locked case5_pjm, case24_ieee_rts; 70 to 90%: case2869_pegase_c2, case5658_epigrids_c3, case2848_rte_c20, case9241_pegase_c1, case7336_epigrids_c2), so single-count changes between fractions are within bootstrap noise.
- The dual-axis fig_sweep.png drew the 10/15 bar across both series; the post uses the single-axis ../2026-10-26-figures-v5/fig_sweep_v5.png, where the bar applies to the locked 15 only.
- The "about 20% visible" label is supported by the released pretraining config (mask_p_high 0.8 for Vm; gridfm-graphkit genco-paper-repro-pretraining HGNS_PreTrain_subnets_100.yaml, datasets/masking.py AddPretrainMask).
- protocol.json says "written ~00:45"; the commit time 00:32:11 is authoritative.
