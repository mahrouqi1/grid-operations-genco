# Generator-limit ablation of the GENCO voltage-inference result (2026-10-22)

**UNAUDITED.** Pre-registered control (`protocol.json`, pinned by `protocol.sha256`; amendments 1 and 2 timestamped before any whole grid ran). Run 00:58 to 07:13 EDT on CPU, 2 workers. Tables: `report_tables.md`; counts: `verdict.json`. Report text written by the orchestrating chat from the run agent's write-up (that agent's Write call for this file was refused).

## Question

The leak check (`../2026-10-21-genco-voltage-inference/leak-check-2026-10-22.md`) found no hidden-label leak, but GENCO sees generator P limits and costs that the interpolators do not, and on ladder grids Pg sits at Pmax on 71-95% of generators. Does the zero-shot |V| win over the best of five classical interpolators depend on that information?

## Bottom line

**No sign of it, on what ran.** The pre-registered threshold (a loss of 3 or more wins on the pieces or on the pooled whole grids, in any ablated arm) was not crossed: the largest loss was 1 win, in every set and arm.

| set | grids run | released | limits constant | costs zeroed | both |
|---|---|---|---|---|---|
| pieces | 36 of 36 | 33 | 33 | 32 | 33 |
| seen whole | 6 of 15 | 6 | 6 | 6 | 6 |
| unseen whole | 10 of 17 | 6 | 5 | 6 | 5 |
| whole grids pooled | 16 of 32 | 12 | 11 | 12 | 11 |

40% measured, 3 seeds pooled, the original studies' win rule. The released counts recompute to the original studies' counts on every grid that ran. A CPU rerun of the released model on one piece matched the stored predictions to 1.9e-7 pu.

- Median GENCO/best-interpolator error ratio moves 1-4% worse in the ablated arms.
- Win-to-loss flips: `1-HV-mixed--0` (64 buses, 0.92 to 1.03 with limits constant) and `carved_case2848_rte_c4` (costs zeroed).

## Not run (listed in `verdict.json` under `skipped`)

- Seen whole (9): case3970, case4020, case4601, case4619, case4837, case4917, case9591, case5658_epigrids, case7336_epigrids. These are the ladder grids the leak check flagged most.
- Unseen whole (7): case6470rte, case6495rte, case6515rte, case8387_pegase, case9241pegase, case3012wp_k, case3375wp_k.
- Reason: measured CPU cost 1.1-2.0 ms per bus-scenario-pass; grids above the size cap (2,500, then 3,100 buses) did not fit before 07:30.

## Caveats

- The seen-whole result covers 6 of 15 grids, below the "at least half" mark, so it is inconclusive on its own; the pooled whole-grid count sits exactly at half.
- The 4,000-9,600-bus ladder grids, where limit reliance is most plausible, remain unchecked. The largest-concern seen grid that did run, case2000_goc, barely moved (ratio 0.515 to 0.513).
- Ablating limits also moves GENCO's inputs off its pretraining distribution, which if anything would understate reliance.
- This tests GENCO's reliance on limits against the five interpolators only. It does not address the WLS comparator, which itself leans on a limits-based dispatch prior (WLS audit R1-F1).
