# Can GENCO add anything on top of a cheap classical solve (fd2)?

2026-10-09. UNAUDITED. Protocol and selection rule: `protocol.json` (sha256 `1c1e288c...e0272d8dd0c5e445`, `protocol.sha256`), written at 14:28 EDT before any run. Every number below is recomputed from `results.jsonl` (rows) and `timing.jsonl` by `scripts/make_tables.py` into `tables.md`; full tables are there.

## Bottom line

**No. GENCO does not improve on fd2 on branch dVa, slack-referenced Va, or |V| on any of the 9 grids, in any of the combinations tested, and not on a single scenario.** The pre-registered gate for stage 2 failed on validation, so stage 2 (new label-free training) was not run. The only place any GENCO-derived state came out ahead of fd2 is the AC active-power residual, on two or three grids, by a small absolute amount, and a classical iteration beats it there for less than a millisecond.

1. Stage 1 (zero-shot GENCO, fresh eval2 sets, 9 grids): GENCO raw has branch dVa 25x to 129x that of fd2 (0 of 9 grids beat fd2; GENCO raw and e50 beat fd2 on 0% of scenarios on every grid), |V| MAE 12x to 178x, and AC residual 11x to 55x. Variant (b) (GENCO |V| with fd2 angles) leaves dVa exactly at fd2 by construction but makes |V| 12x to 177x worse and the residual 1.2x to 11.6x worse. Variant (c) (fd2 |V| with GENCO angles) is as bad as GENCO on dVa (25x to 129x). Variant (d), the per-bus-type blend, chose all weights 0 on validation (the next-best blend was 6.7x worse on |V| and 25.9x worse on dVa), so (d) equals fd2 exactly.
2. Gate (validation, case14 and case30, zero-shot): no variant reached the required ratio of at most 0.98 to fd2 on dVa, |V| or active residual. d is exactly 1.00 everywhere; b is 1.00 on dVa but 26x and 31x on |V| and 4.5x and 11x on residual; c is 93x and 116x on dVa. Failed, so stage 2 was not run.
3. Supplementary, outside the gate, no new training: the 27 existing label-free adapted checkpoints (DC prior, 9 grids x 3 draws, n = 1,000) scored the same way on eval2. Adapted GENCO raw is 2.4x (case14) to 26x (case300) worse than fd2 on dVa, and 2.5x to 24x worse on |V|; the combinations b, c, e50 never beat fd2 on dVa or |V| on any grid or draw. Two existing fd2-prior adapted checkpoints (case14, case30) are 2.7x and 12x worse than fd2 on dVa.
4. The one metric with any GENCO-side advantage: AC active residual (own Ybus/Sbus). Adapted GENCO raw beats fd2 on all 3 draws on case14 (0.76x, 0.81x, 0.75x of fd2's 0.261 MW) and case89 (0.72x to 0.74x of 8.59 MW); the 50/50 blend e50 also on case118 (0.90x to 0.97x of 1.14 MW). On the same grids dVa is 1.3x to 15x worse, and fd3 (one more 0.1 ms iteration) gets 0.020, 0.447 and 0.062 MW, which beats every GENCO-derived residual. So this is not a GENCO advantage over a cheap classical solve.
5. Cost (shared box, load 45 to 60): fd2 from raw tables 2.7 to 4.5 ms per scenario, of which 2.0 to 2.7 ms is building the problem and only 0.3 to 1.3 ms is the two iterations with their factorizations; nr1 5.6 to 8.9 ms. GENCO forward 1.5 to 3.8 ms per scenario on a shared GPU batched at 64 (10 to 192 ms on 4 CPU threads), plus 14 to 28 ms per scenario to turn tables into graphs through graphkit's dataloader and 0.25 to 2.8 s once to load the checkpoint. So fd2 end to end costs about the same as a batched GPU forward pass alone (fd2 is 1.2x to 2.0x the GPU forward time, so it is the slower of the two) and is cheaper than the CPU forward on every grid and than forward plus data-to-graph; cost is not a reason to prefer GENCO.

Plain statement for the entry: on these nine grids, with a DC start available, two fast-decoupled iterations dominate GENCO (zero-shot or label-free adapted) on every accuracy metric except a low-order residual that more classical iterations improve further.

## Table (stage 1, eval2, zero-shot GENCO; ratio to fd2, lower is better; 1.00 = fd2)

Branch dVa, ratio to fd2 with 95% paired-bootstrap interval over scenarios (`results.jsonl`, state `genco_raw`, `e50`); b and d equal fd2 on dVa, c equals GENCO raw.

| grid | fd2 dVa (deg) | GENCO raw | c (= raw) | e50 (not selected) | scenarios where GENCO raw beats fd2 |
|---|---:|---:|---:|---:|---:|
| case14 | 0.0189 | 92.8 [79.3, 107.3] | 92.8 | 46.4 [39.7, 53.6] | 0% |
| case30 | 0.0085 | 121.5 [115.4, 127.2] | 121.5 | 60.7 [57.7, 63.6] | 0% |
| case39 | 0.0234 | 28.8 [24.7, 33.1] | 28.8 | 14.4 [12.4, 16.4] | 0% |
| case57 | 0.0308 | 25.1 [23.3, 26.8] | 25.1 | 12.6 [11.7, 13.4] | 0% |
| case73 | 0.0101 | 83.5 [81.0, 86.4] | 83.5 | 41.7 [40.4, 43.1] | 0% |
| case89 | 0.0168 | 128.6 [124.3, 132.9] | 128.6 | 64.5 [62.4, 66.7] | 0% |
| case118 | 0.0209 | 52.7 [50.1, 55.7] | 52.7 | 26.2 [24.9, 27.7] | 0% |
| case162 | 0.0204 | 72.5 [68.7, 76.5] | 72.5 | 36.2 [34.2, 38.2] | 0% |
| case300 | 0.0273 | 50.9 [45.3, 56.2] | 50.9 | 25.5 [22.8, 28.2] | 0% |

Absolute values for DC, fd2, nr1, GENCO raw, b, c, d, e50 on branch dVa, slack-referenced Va, |V|, active and reactive residual are in `tables.md` (first five tables). Summary of how often each state beats fd2 (upper end of the 95% interval below 1): GENCO raw, pinned, b, c, d, e50 and DC: 0 of 9 on every metric. nr1: 0 of 9 on dVa, 6 of 9 on active residual (nr1 beats fd2 on residual, and loses on dVa on every grid, 1.6x to 13.9x).

Selected on validation (case14 and case30, 200 held-out training-pool scenarios each, labels used only there): `stage1_blend_selection.json`: w_vm 0, w_pv 0, w_pq 0.

## Cost per scenario (median of 3 repeats over up to 60 scenarios per grid; `timing.jsonl`)

| grid | fd2 total (ms) | nr1 total (ms) | of which build problem (ms) | of which DC solve (ms) | GENCO forward, GPU, batch 64 (ms) | GENCO forward, CPU 4 threads (ms) | ckpt load (s, once) |
|---|---:|---:|---:|---:|---:|---:|---:|
| case14 | 2.70 | 5.76 | 2.28 | 0.13 | 1.83 | 35.5 | 2.79 |
| case30 | 2.67 | 5.55 | 2.15 | 0.15 | 2.18 | 28.0 | 0.25 |
| case39 | 3.04 | 6.65 | 2.47 | 0.17 | 1.49 | 10.5 | 0.65 |
| case57 | 3.11 | 6.40 | 2.44 | 0.20 | 1.63 | 15.3 | 0.88 |
| case73 | 2.87 | 5.99 | 2.15 | 0.23 | 1.88 | 25.6 | 0.41 |
| case89 | 2.93 | 5.78 | 2.05 | 0.27 | 2.17 | 29.6 | 1.45 |
| case118 | 3.45 | 7.13 | 2.53 | 0.31 | 2.26 | 53.8 | 1.37 |
| case162 | 4.35 | 8.43 | 2.68 | 0.48 | 2.19 | 191.6 | 0.50 |
| case300 | 4.52 | 8.86 | 2.59 | 0.58 | 3.78 | 163.2 | 0.62 |

fd2 and nr1 are single-threaded numpy/scipy including the sparse assembly from the datakit tables (the Ybus, B' and B'' slicing dominates); the DC solve is a separate sparse solve (datakit already stores Va_dc, which the priors use). The GENCO forward is `task.forward` only (model, no loss, no data loading), warm, 5 repeats; data-to-graph through graphkit's dataloader (`ms_data_to_graph_per_scenario_cached_dataloader` in `timing.jsonl`) is 14 to 28 ms per scenario and is not in these columns. The GPU is shared with other jobs (load 45 to 60 on 24 cores), so absolute values are noisy; the order of magnitude is the claim. Not measured: a GENCO forward on the CPU alone without batching, or a compiled/bf16 forward.

## What was run

- Stage 1 (protocol.json "stage1"): zero-shot GENCO checkpoint, graphkit `evaluate` unchanged, input-only normalizer fit (`FEWSHOT_INPUT_NORM=1` through `gk_freeze`, `input_norm_plugin_seen: true` in `score_zeroshot_*.jsonl`), on the 9 eval2 sets (196 to 198 scored scenarios per grid; case162 113, case300 180). States: dc, fd2 (`labelfree_prior2.priors_for`), nr1, GENCO raw, GENCO pinned (ladder_eval definition), b, c, d, and an extra e50 (all weights 0.5, no selection; it was pre-registered for stage 2 only and is reported here for completeness). All states scored on the scenarios graphkit scored.
- Residual definition: computed from each state's complex voltage with our own per-scenario Ybus and Sbus (`labelfree_prior2.input_problem`, `hybrid.Scenario.f`), as sum over PV+PQ buses of |P mismatch| times 100 MVA divided by the bus count (slack mismatch counted as 0), and sum over PQ buses of |Q mismatch| times 100 divided by the bus count. This is NOT graphkit's residual (which is what `score_zeroshot_*.jsonl` carries as `genco_active_res_mw`). The two agreed to 4 digits for GENCO raw in the earlier label-free prior2 study; I did not re-check that here.
- Validation sets: 200 scenarios per dev grid from `labelfree.pool_lf` of the training set (profile index 200 to 1199; no collapsed labels; disjoint from the eval2 window 1600 to 1799), renumbered 0..199. Stage 1 trains nothing, so overlap with the later training subsets is irrelevant.
- Supplementary (not pre-registered as a stage-2 run; no new training): `scripts/s2_existing.py` scores existing checkpoints on eval2. DC-prior adapted: the 27 `lfc_w1_lr3e-05_all` runs (predictions from `fewshot_fresh`, produced by the earlier fresh-eval study). fd2-prior adapted: `lfp2_fd2_w1_lr3e-05_all` on case14 and case30, draw 0 (evaluated here, input-only normalizer and the checkpoint's own stats).
- Extra, not requested: `eval2_classical` rows (fd1 to fd4, nr1 to nr3, DC) to put GENCO's residual next to cost-matched classical iterations. fd3 and fd4 are extra XB iterations from the fd2 state (kept only if finite; no mismatch guard).

## Deviations and gaps (read before citing)

1. Stage 2 was not run because the validation gate failed (as the protocol says). Not implemented either way: the "Newton-style correction inputs" (extra input channels on the GENCO model); a corrector on top of fd2 was only tested as post-hoc combinations of an adapted GENCO's output with fd2. A GENCO trained to predict a delta to fd2 with fd2 as an input channel is not tested and nothing here says it would fail; the evidence is that the plain target prior route (supplementary, fd2-prior runs on case14 and case30) lands at 2.7x and 12x fd2's dVa.
2. The supplementary fd2-prior evidence is two dev grids, one draw, one config (w = 1, lr = 3e-5, all parameters), not the validation-selected stage-2 grid. The DC-prior evidence is 27 checkpoints but a different prior from the one stage 2 would use. Neither replaces a stage-2 run.
3. The first zero-shot run used graphkit's default normalizer fit by mistake: `fewshot.evaluate` only routes through the `gk_freeze` plugin module when slack_vm is on, so `FEWSHOT_INPUT_NORM` was ignored. It was detected before any report. That run is kept as a sensitivity set (`results_defaultnorm_zeroshot.jsonl`, `tables_old_defaultnorm.md`, `stage1_blend_selection_defaultnorm.json`); the gate and selection were repeated on the input-only run and are identical in outcome (all blend weights 0). GENCO raw dVa differs between the two fits by at most 9% (case57), see the sensitivity table in `tables.md`.
4. The first validation build broke graphkit's AC/DC metric step (`scenario_partition` not renumbered); fixed (`scenario // 200`) and re-run; the failed attempts left no score rows.
5. Protocol text said "mean over PQ buses" for the Q residual and "mean over buses" for P; the implemented definition is the one above (divided by total bus count). It does not change any ranking here because every state is scored identically.
6. Timing: single machine, shared GPU, high CPU load; fd2/nr1 timings are for the Python reference implementation, a compiled solver would be faster; GENCO timings exclude data loading and are not a tuned deployment.
7. Adapted-GENCO rows in `tables.md` are means over 3 draws; the bootstrap intervals in `results.jsonl` are per draw and cover scenario variation only. The 'beats on all 3 draws' counts use each draw's point ratio.

## Files

`protocol.json`, `protocol.sha256`, `commands.sh`, `scripts/` (write_protocol, s1_predict, s1_score, s1b_classical, s2_existing, s0_timing, common, make_tables), `results.jsonl` (401 rows: val 16, eval2 81, eval2_classical 72, supp_dcprior_adapted 216, supp_fd2prior_adapted 16), `results_defaultnorm_zeroshot.jsonl`, `score_zeroshot_val.jsonl`, `score_zeroshot_eval2.jsonl`, `stage1_blend_selection.json`, `timing.jsonl`, `tables.md`. Per-scenario metric arrays are in `data/external/genco-ladder/genco_on_fd2/scen_*.npz` (gitignored data tree).
