# Few-shot fine-tuning of the released GENCO on the 15 locked-benchmark grids (plan candidate 4)

**UNAUDITED.** 2026-10-21. Protocol `protocol.json` (sha256 in `protocol.sha256`, committed before any new fine-tune) and
`protocol_amendment1.json` (written before any AC-labelled run; explains the variant split below). Every number is recomputed from
`results.jsonl` (+ the reused B4 rows) by `scripts/verdict.py`, output in `table_wu100L.md`, `table_wu100.md`, `verdict_*.json`.

## Bottom line

1. **The pre-registered success test FAILED, in both variants.** Test: draw-mean branch dVa below DC on >= 10 of 15 grids at n = 100.
   Result: **7 of 15** (7 of the 12 grids that could be run) for the AC-labelled recipe, and **7 of 15** for the as-run B4 recipe. With 3 grids
   not runnable (below) the test needed 10 of the remaining 12; it got 7.
2. **What is true and usable:** with 100 labelled scenarios of a new grid, GENCO (released checkpoint, 100-step warm-up, full fine-tune) beats DC on
   branch angle differences on **7 of the 12 evaluable locked grids, all of them small (5 to 89 buses)**, 3 of 3 draws each, ratio to DC 0.07 to 0.85.
   It does **not** beat DC on case39 (1.33), case118 (1.05), case162 (1.01), case240 (1.21), case300 (1.56). Mean-ratio crossover sits near 100 to 150 buses, with
   case39 (39 buses) an unexplained small-grid loss in both variants.
3. **Against guarded fd2 (falls back to DC on a larger mismatch) it loses everywhere: 0 of 12.** Against raw fd2 (the second fast-decoupled iterate, as in robust-start and corrector-at-scale) it wins on case240 only (ratio 0.40 with the pool-fixed splits: draws 1.244 / 1.235 / 1.276 deg over raw fd2's 3.156 deg; 0.37 with the original splits; `../2026-10-21-fewshot-locked15-rawfd2/`, audit R1-F1, claims audit R2-F5). That win is not a useful one: raw fd2 there is 3.3x worse than DC (0.97 deg) and GENCO (1.25 deg) is worse than DC too. The two definitions coincide on the other 11 grids. GENCO's branch dVa is 1.8x (case5) to 32x (case300) fd2's, 2.6x on case240 (where fd2 itself is poor,
   0.45 deg vs DC 0.97). fd2 needs no labels and no training. So "N labelled scenarios makes GENCO beat DC" is true on small grids, and "beats fd2" is false;
   this does not rescue GENCO as an angle estimator.
4. **Do not write "with N labelled scenarios" for the B4 numbers.** B4 (`2026-10-10-warmup-adaptation`) and the first 15 rows here trained on **DC-prior targets, not AC labels**
   (details below). That variant is a label-free few-shot result (n unlabelled load scenarios), and it is nearly as good: 7 of 12 as well. Labels help on the small grids
   (case14 0.32 vs 0.56 of DC, case30 0.41 vs 0.62) and are not decisive on the large ones.

## Per-grid table: branch dVa, draw-mean ratio to DC (3 draws, n = 100)

AC-labelled recipe (primary, `wu100L`) beside the as-run B4 recipe (DC-prior targets, label-free, `wu100`). Ratio to fd2 is for the labelled recipe.
`k/3` = draws below DC. DC / fd2 / GENCO are absolute branch dVa in degrees, labelled recipe.

| grid | buses | DC | fd2 | GENCO (L) | L: ratio to DC | L: draws < DC | L: ratio to fd2 | as-run (DC-prior): ratio to DC | verdict (L) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case5_pjm | 5 | 0.386 | 0.0146 | 0.027 | 0.07 | 3/3 | 1.8 | 0.09 | beats DC |
| case14_ieee* | 14 | 0.411 | 0.0189 | 0.133 | 0.32 | 3/3 | 7.1 | 0.56 | beats DC |
| case24_ieee_rts | 24 | 0.272 | 0.0101 | 0.110 | 0.40 | 3/3 | 10.8 | 0.45 | beats DC |
| case30_ieee* | 30 | 0.288 | 0.0085 | 0.117 | 0.41 | 3/3 | 13.7 | 0.62 | beats DC |
| case39_epri | 39 | 0.254 | 0.0234 | 0.337 | 1.33 | 0/3 | 14.4 | 1.53 | loses to DC |
| case57_ieee | 57 | 0.389 | 0.0308 | 0.277 | 0.71 | 3/3 | 9.0 | 0.74 | beats DC |
| case73_ieee_rts | 73 | 0.311 | 0.0101 | 0.256 | 0.82 | 3/3 | 25.4 | 0.76 | beats DC |
| case89_pegase | 89 | 0.360 | 0.0168 | 0.307 | 0.85 | 3/3 | 18.3 | 0.88 | beats DC |
| case118_ieee | 118 | 0.354 | 0.0209 | 0.370 | 1.05 | 1/3 | 17.7 | 1.22 | loses to DC |
| case162_ieee_dtc | 162 | 0.240 | 0.0204 | 0.242 | 1.01 | 1/3 | 11.9 | 1.16 | loses to DC |
| case240_pserc | 240 | 0.968 | 0.4503 | 1.167 | 1.21 | 0/3 | 2.6 | 1.38 | loses to DC |
| case300_ieee | 300 | 0.567 | 0.0273 | 0.886 | 1.56 | 0/3 | 32.4 | 1.71 | loses to DC |
| case1354_pegase | 1354 | | | | | | | | NOT RUN |
| case2736sp_k | 2736 | | | | | | | | NOT RUN |
| case3120sp_k | 3120 | | | | | | | | NOT RUN |

`*` development grids of the earlier frozen-trunk recipe (B4 rule R4). Per-draw ratios, |V| and our AC-residual ratios are in `table_wu100L.md` / `verdict_wu100L.json`.
|V| ratio to DC is below 1 on every grid (0.03 to 0.28 for the labelled recipe on all 12); our AC active residual ratio to DC is below 1 except
case39 (1.09) and about 1 on case240 (0.98).

## What was and was not run

- **Run:** 12 grids x 3 draws x 2 variants. AC-labelled: all 36 new runs. As-run: case14/30/57/118 taken from B4 `results.jsonl` (same recipe, splits, eval2, scoring; not rerun),
  the other 8 grids x 3 draws run here (case300 rerun; B4's OOM'd). 60 successful runs here.
- **Not run (3 grids):** case1354_pegase (datakit AC labels collapse: 69 of 97 eval and ~76% of training scenarios; the few-shot study excluded it by design), case2736sp_k and case3120sp_k
  (training-pool plus eval generation projected well over the 1 h budget; ladder runs took 2,735 s and 1,070 s for 200 scenarios with only 74 and 99 surviving). They count as not beating DC in the 15 denominator.
- **n = 1000 not run** (about 6,000 s per run by the B4 estimate).
- **Data generated for this study** (case5, case24, case240; datakit, same configs/seeds as the existing procedures, `scripts/gen_missing.py`): training pool requested 700 scenarios, not 1200 (pools of 500, 500, 469 after the profile-index rule);
  eval2 = 200 scenarios each. case240's eval2 had 5 collapsed-label scenarios, which `fewshot._score` refuses to score; they were dropped and the rest renumbered 0..185 (`scripts/filter_eval_case240.py`, graphkit needs contiguous ids),
  so case240 eval2 has 191 generated, 5 collapsed dropped, 186 written and 184 scored (audit R1-F9).
- Pool fix (audit R1-F2, `../2026-10-21-fewshot-locked15-pool-fix/`): the wu100L splits came from the label-free pool; with the collapse-filtering pool, case240's ratio to DC is 1.293 (was 1.206; 0/3 draws below DC either way). The 7/15 verdict is unchanged, and no other grid's splits contained collapsed labels.

## Amendment: the B4 recipe did not use AC labels (read before citing B4)

B4's `run.py` (copied here) imports `genco_on_fd2/scripts/common.py`, which imports `genco_challenge.labelfree`; that module patches `fewshot` at import time to `TRAIN_DATA = train_dcprior`
and adds `FEWSHOT_INPUT_NORM=1`. `train_dcprior` replaces the AC voltage labels by the DC power-flow prior (`Va := Va_dc`, `Vm := 1.0` at PQ buses). So every B4 fine-tune, and the first 15 rows of this study,
trained with a DC-prior MSE term plus the physics-residual loss and an input-only normalizer; the AC solution was never a training target. B4's protocol text says the published recipe "unchanged except data paths", and its report says "labelled";
neither is accurate as run (train log line "input-only normalizer ON" confirms the normalizer). Found after 15 rows; the AC-labelled variant (`fs.TRAIN_DATA = train`, no `FEWSHOT_INPUT_NORM`, separate split dir `splits_L`) was added in
`protocol_amendment1.json` before any labelled run, and it is the primary variant because the stated goal is a claim about labelled scenarios. The same success test is applied to both. Both give 7 of 15. B4's report should be corrected
(the case14/30/57 "beats DC at n = 100" is a label-free result).

## Caveats

- UNAUDITED; single codebase, no independent re-run. The 15 locked grids were the target; the fine-tune runs reuse draws 0 to 2 of the same pool-split code as B4.
- Draw-mean ratios over 3 draws; draw spread is small except case118 (1.16, 1.18, 0.81), where one draw beats DC. The success test uses the draw mean, as pre-registered.
- The DC and fd2 comparators need no labels or training; the comparison is not equal-cost. fd2 comparison is reported, not part of the success test.
- case14/case30 are development grids of an earlier recipe; leaving them out leaves 5 of 10 evaluable.
- All 5 non-development wins are on grids under 90 buses; I make no claim about 100+ bus grids other than the losses above.
- Failed attempts (CUDA OOM from another process on GPU 0; case240 eval failures before the scenario-renumbering fix) were retried; the 3 persistent error rows are in `results.jsonl` (they later succeeded on retry) and are ignored by `verdict.py`; no rejected checkpoint was scored.

## Compute

Local, GPU 0 only, 2 to 3 concurrent streams, shared and CPU-contended box (load average 40 to 48 on 24 cores). 60 successful runs: 7.8 training-hours summed (labelled 5.0 h, as-run 2.8 h; about 5 to 10 minutes per run
plus about 1 minute of evaluation), about 5 hours elapsed, GPU occupancy not measured (small graphs, evidently CPU-bound on this contended box). Data generation for case5/24/240 took under 30 minutes. No OSC was used or needed.
Reproduce: `scripts/plan.py`, `scripts/plan_amendment1.py`, `scripts/gen.sh`, `scripts/filter_eval_case240.py`, `scripts/launch.sh w1..` (runs `scripts/run.py worker`), then `python scripts/verdict.py wu100L` and `wu100`.
