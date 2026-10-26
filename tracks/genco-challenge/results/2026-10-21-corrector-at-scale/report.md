# Does a trained fd2 corrector beat fd_k at equal cost on large grids? (candidate 2)

> **UNAUDITED.** 2026-10-21. Pre-registered in `protocol.json` (commit f2247ba, before any test scoring) with four addenda
> (below). Scripts in `scripts/`, per-scenario scores in `per_scenario/`, verdict from `scripts/verdict.py`
> (`verdict_n100.json`, `summary_table_n100.md`).

## Bottom line

1. **The pre-registered success test fails: the corrector wins on 0 of 3 grids.** The corrector (n = 100 labelled points of the target
   grid, the case1354-selected recipe) is **worse than fd2 on branch dVa on every grid**: ratio to fd2 of 1.50
   (95% CI 1.44 to 1.58) on case2869pegase, 3.71 (3.53 to 3.90) on case9241pegase and 1.20 (1.16 to 1.24) on
   case6470rte. So it also loses to every fd_k with k >= 2, at any time basis. The timing cannot change this.
2. **There is no cost crossover either.** Per scenario, the full corrector pipeline (fd2 on the CPU plus GENCO
   end to end at GPU batch 8 to 64) cost 134 / 249 / 148 ms on 2869 / 9241 / 6470 buses, against 18 / 64 / 42 ms
   for fd2, and 48 / 175 / 111 ms for fd30. Even the forward-only lower bound (graph conversion excluded) costs as much as
   17 to 19 fd iterations. The hypothesis "fd iterations get more expensive with size while GENCO stays flat" is
   half right: an fd iteration grows from 1.1 ms (2,869 buses) to 2.5 ms (6,470) to 4.0 ms (9,241), but one batched
   GENCO forward per scenario grows with size too (18 to 58 ms), so the ratio did not move in the corrector's favour.
3. **Why the corrector fails here is the same failure as case300 in `2026-10-10-corrector-above-162`, not a size
   effect.** On 2869 and 9241 the validation dVa is worse than fd2 from epoch 1 and keeps rising (the checkpoint
   monitor keeps epoch 1, which is already worse than the zeroed-head start: validation ratio to fd2 1.71 on 2869 and
   3.66 on 9241). lr 5e-4 is too large for these grids; lowering it to 5e-5 (post hoc, 2869 only) gives 1.05 (1.04 to 1.06),
   still not below 1.
4. **On case6470rte the corrector never received fd2.** The guard in `lp2.priors_for` replaces any fd iterate whose
   max power mismatch exceeds DC's by DC, and it did so in **100% of the scored scenarios** (`fd2_input_is_dc_frac` = 1.0). So
   the "fd2 prior" was the DC state. There the corrector is genuinely useful against what it was given: dVa 0.129 vs DC 0.201
   (ratio 0.64, CI 0.639 to 0.643) and active residual 4.7 vs 93 MW/bus. But the RAW fd2 iterate (no guard) has dVa 0.107, better than
   the corrector, and raw fd5 reaches 0.019. Whether a corrector trained on top of the raw fd2 state would do better
   was **not run** (see "not done").

## Table (n = 100 corrector, 217 scored test scenarios per grid; ratio columns are to fd2)

Branch dVa in degrees, |V| MAE in pu, active residual in MW per bus (from our own Ybus/Sbus), time in ms per scenario
(median of 5 repeats; contended GPU, see Timing).

| grid | method | dVa | ratio to fd2 | |V| MAE | active res | ms/scenario |
|---|---|---:|---:|---:|---:|---:|
| case2869pegase | DC | 0.2302 | 13.5 | 2.74e-02 | 136 | 9.2 |
| | fd2 | 0.01708 | 1 | 2.17e-04 | 6.74 | 18.1 |
| | fd3 | 0.00652 | 0.38 | 8.02e-05 | 0.449 | 19.1 |
| | fd4 | 0.00394 | 0.23 | 4.83e-05 | 0.083 | 20.2 |
| | fd5 | 0.00269 | 0.16 | 3.33e-05 | 0.041 | 21.2 |
| | fd30 (time-matched comparator) | 2e-05 | 0.001 | 3.1e-07 | 1.5e-4 | 48.0 |

Note (audit R1-F6): classical fd_k states are scored against the float32 Va_target of the prediction parquet, so fd_k branch-dVa entries below about 1e-3 deg are float32 floors, not exact errors. No comparison changes: the corrector is far above them.

| | **corrector** | **0.02569** | **1.50** | 1.02e-03 | 9.93 | 133.7 (99 to 153) |
| case9241pegase | DC | 0.4139 | 16.5 | 2.60e-02 | 94.6 | 28.0 |
| | fd2 | 0.02502 | 1 | 2.78e-04 | 5.60 | 64.3 |
| | fd3 | 0.00543 | 0.22 | 6.82e-05 | 0.465 | 68.2 |
| | fd4 | 0.00179 | 0.07 | 2.31e-05 | 0.076 | 72.4 |
| | fd5 | 0.00080 | 0.03 | 1.05e-05 | 0.023 | 76.2 |
| | fd30 | <1e-5 | | 2.9e-08 | 1.5e-6 | 175.4 |
| | **corrector** | **0.09284** | **3.71** | 7.10e-04 | 18.9 | 248.6 (212 to 288) |
| case6470rte | DC (= the guarded fd2 the corrector received) | 0.2011 | 1.87 | 3.74e-02 | 93.1 | 17.1 |
| | fd2 (raw) | 0.10742 | 1 | 5.82e-03 | 5.54 | 41.9 |
| | fd3 | 0.04826 | 0.45 | 3.40e-03 | 0.734 | 44.3 |
| | fd4 | 0.02816 | 0.26 | 2.29e-03 | 0.364 | 46.7 |
| | fd5 | 0.01902 | 0.18 | 1.73e-03 | 0.351 | 49.2 |
| | fd30 | 2e-05 | 0.0002 | 3.6e-06 | 3.7e-4 | 110.9 |
| | **corrector** | **0.12886** | **1.20** | 1.29e-02 | 4.66 | 148.3 (132 to 153) |

Success test (`verdict_n100.json`): comparator = the cheapest fd_k whose time is >= the corrector's. No fd_k up to 30 is as
expensive as the corrector on any grid, so the comparator is fd30 (the protocol's conservative fallback), and the
corrector/fd30 dVa ratio is 5.5e3 (4.4e3 to 6.5e3) on 6470, 1.0e3 on 2869 and 2.5e4 on 9241. With the generous time
basis (forward-only lower bound: 36 / 123 / 79 ms) the matched comparators are fd19 / fd17 / fd18 and the corrector loses
by factors of 221 / 10,078 / 313, CI upper bounds all far above 1. `SUCCESS_primary = false`, `SUCCESS_generous_time = false`,
`strict_wins = []`. The corrector beats DC on every grid (ratios 0.11, 0.22, 0.64) and no fd_k for any k >= 2.

## What was run

- **Grids and data.** case2869pegase, case9241pegase, case6470rte from `experiments/2026-10-18-genladder/<case>.npz`
  (verified ladder labels), converted by `ladder_to_datakit` (1,320 points per grid, seed 0, all kept, max residual
  < 1e-9 pu), split with a fixed permutation into 1,000 train-pool / 100 validation / 220 test. Test scored 217 per grid
  (graphkit's test_ratio 0.99 over unique load profiles). The 1,000-point pool was not used beyond drawing the n = 100 subset.
- **Corrector.** Exactly the case1354 pipeline (`corr3.py` -> `scripts/scale.py`, same plugins, scoring and
  bootstrap code), variant p0.1_lr5e-4 as selected on case1354 validation, 250 epochs, frozen trunk, zeroed head, checkpoint =
  best validation branch dVa. Training seconds: 3,753 (2869), 6,480 (6470), 11,198 (9241) on a shared RTX 3090.
- **Comparators.** fd_k = k-th raw XB iterate from the same DC start, k = 2..30, same scorer.
- **Timing.** `scripts/timing.py`, 64 test scenarios, 5 repeats after a warm-up, CPU pinned to cores 16-19 and
  one BLAS thread, GPU 1. Corrector = assembly + DC + two fd iterations (CPU) + GENCO end to end at the largest batch that
  fits (<= 64; 32 on 2869, 64 on 6470, 8 on 9241) + add.

## Timing caveat (important)

The box was heavily loaded (1-minute load average 50 to 56 at timing; other jobs on both GPUs). GPU 1 carried another
process (about 18 to 23 GB, 100% utilisation) during all three timed blocks (`gpu_before` / `gpu_after` of every repeat are in
`timing.jsonl`). The first timing pass (`timing_run1_contended.jsonl`) was run under the same kind of contention and one
block went out of memory; the second pass is the recorded one and is also contended. The GPU-side numbers are therefore
inflated, to an unknown degree (the corrector's repeat-to-repeat spread is 1.5x to 1.6x). This does not decide the verdict: the
corrector's accuracy is worse than fd2's, which costs 18 to 64 ms, so no time basis rescues it. It does mean the "forward-only
17 to 19 fd iterations" figures are upper bounds on GPU cost, not clean measurements. The conversion (graph build) is also unfused Python
on one CPU core, which is the dominant corrector cost (about 70 to 120 ms per scenario in `gpu_e2e - gpu_fwd_only`).

## Deviations, all disclosed

- **Addendum 1, 2, 3 (batch sizes, before any scoring).** Training memory is about 0.32 MB per bus per graph, 2.5x what
  `protocol.json` assumed; batch 24 on 2869 ran out of memory. Final batches: 8 (2869), 5 (6470), 2 (9241), against 32 on case1354.
  Steps per epoch therefore rise (13, 20, 50 vs 4). This may be part of why lr 5e-4 is too aggressive here; it was not separated.
- **Addendum 4 (after seeing the first test scoring of 2869 and 6470).** The first scoring defined fd_k as steps from the *guarded* fd2
  state (inherited from `corr3.classical`), which on 6470 is DC. fd_k was redefined as the raw k-th iterate from DC, which is what
  `protocol.json` says and what the timing path measures; 2869 and 6470 were rescored (superseded rows kept in
  `results_superseded_v1.jsonl`). 9241 was scored with the corrected code. The corrector and its predictions are unchanged
  by this.
- **One draw per grid** (the case1354 study had three at n = 100); the CI covers test scenarios only.
- **Post hoc, labelled:** lr 5e-5 variant on case2869pegase (`per_scenario/case2869pegase_n100_p0.1_lr5e-05.npz`): dVa 0.01797,
  1.05 (1.04 to 1.06) of fd2, best epoch 1, validation ratio 1.08. Training of this run was restarted once after I
  mis-killed a first copy (an orphaned process kept running; both copies used the same output key); the recorded run is the
  restart, whose log was re-checked, but this is the least clean run here.

## Not done

- **n = 1,000** was not run (cost: about 9 h on case9241 alone at the measured rate on a shared GPU; and n = 100 already fails
  before the data quantity matters).
- **Corrector on the raw fd2 state for case6470rte** (priors with `make_priors.py --raw` exist; a launch was attempted but
  GPU 1 was held by another job and the run was cancelled). This is the one experiment the 6470 result invites.
- **A different recipe for 2869 / 9241** (lower lr, label-only loss, the metric-in-loss runs of case300): only lr 5e-5 on 2869 was tried.
  The earlier study already found that the recipe's loss makes training dVa worse from epoch 1 on case300; 2869 and 9241 look the same.
- case2848rte (optional), a draw-to-draw spread, and an uncontended timing pass.

## Surprises

1. On case6470rte the guarded fd2 prior equals DC in every scored scenario, so every previous "corrector on top of fd2" claim
   should be read per grid: where the guard fires, it is a corrector on top of DC. The raw fd2 there is better than the guarded state.
2. The corrector's validation dVa is worse than its start from epoch 1 on the two PEGASE grids, and gets worse with grid size (1.5x, 3.7x
   fd2), consistent with the case300 finding that the training objective, not capacity, is the obstacle.
3. fd on case6470rte starts slower (raw fd2 is not better than DC on |V| or reactive residual by much) but converges: fd5 is already 5.6x
   better than raw fd2 on dVa.
