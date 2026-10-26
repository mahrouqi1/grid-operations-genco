# Zero-shot released GENCO as a Newton start on large unseen grids (plan candidate #1)

> **UNAUDITED.** Produced by an agent run on 2026-10-21 (internal branch). Protocol frozen in `protocol.json` (commit 17d35d7) before any GENCO-start Newton run. Verdict computed by `verdict.py` from `per_scenario/*.csv`.

## Bottom line

**NO-GO. Both pre-registered conditions fail, by a wide margin.** The released GENCO, used zero-shot as a Newton start, does not rescue large unseen grids and costs more than the classical start that works.

- Condition (a) fails: GENCO's verified-root rate is *below* the best classical start on every family (RTE 0% vs 40% for fd2; PEGASE 66.7% vs 100%; Polish 66.7% vs 100%). No family passes the 10-point test.
- Condition (b) fails: on the five grids with >= 5,000 buses GENCO is never at least 20% faster at equal convergence. On case6470/6495/6515rte nothing converges (all five starts, GENCO included, are 0/100). On case8387 and case9241 pegase classical starts already converge 100% (fd2 total 282 and 331 ms); GENCO takes 1.85x and 7.7x as long (case9241: 0/99 converge from either GENCO start).
- The plan's premise (flat/DC/fd2 fail on large grids, GENCO rescues) is only half visible: flat/DC do fail on RTE, but **fd2 does not** on case1888rte and case2848rte (100% verified root), and on the PEGASE and Polish grids flat, DC or fd2 already reach 100%. Where all classical starts fail (case6470/6495/6515rte) GENCO fails too. (Classical starts here are capped at fd2 by protocol; a classical fd10 start reaches the root on these grids, see the caveat in section 7, audit R1-F3.)
- Where GENCO converged (genco_full: case1888rte, case2869pegase, case8387, case2383wp) it needed more iterations than fd2 (6 vs 3 on case1888rte), and it never converged anywhere fd2 did not. The only grid where a GENCO arm beat every classical arm on a per-grid basis: none.

## Setup (see protocol.json for the frozen version)

11 grids, first 100 scenarios per grid by sorted id in graphkit's test split (99 for the six converted non-RTE grids and 100 for RTE: graphkit's `test_ratio` 0.99 split drops one). Labels: verified ladder points in datakit format. Five starts: flat, dc (datakit `Va_dc`, setpoint |V|), fd2 (dc + two XB fast-decoupled iterations), genco_vmdc (released-GENCO |V| with REF/PV |V| put back, DC angles), genco_full (released-GENCO |V| and angle with REF angle and REF/PV |V| put back). Solver: `grid_ops.reference_pf.newton_raphson_pf` (sparse), tol 1e-8 pu max mismatch, 25 iterations, no damping, identical for all starts. Verified root = converged and max |dVm| < 1e-4 pu and max wrapped |dVa| < 1e-2 deg after aligning the slack angle.

## Summary table

(verified / other root / diverged are percentages of that grid's scenarios; times are medians per scenario; the GENCO start cost is the constant per-grid batch-64 end-to-end figure, see caveats.)

| grid | buses | start | verified % | other root % | diverged % | median iters | median start ms | median Newton ms | median total ms |
|---|---|---|---|---|---|---|---|---|---|
| case1888rte | 1888 | flat | 0 | 0 | 100 | 25 | 0.0 | 389.1 | 389.1 |
| case1888rte | 1888 | dc | 0 | 0 | 100 | 25 | 2.7 | 394.9 | 397.2 |
| case1888rte | 1888 | fd2 | 100 | 0 | 0 | 3 | 8.9 | 52.0 | 61.5 |
| case1888rte | 1888 | genco_vmdc | 0 | 0 | 100 | 25 | 28.6 | 387.1 | 415.3 |
| case1888rte | 1888 | genco_full | 100 | 0 | 0 | 6 | 25.8 | 92.8 | 118.6 |
| case2848rte | 2848 | flat | 0 | 76 | 24 | 9 | 0.0 | 181.6 | 181.6 |
| case2848rte | 2848 | dc | 0 | 73 | 27 | 9 | 3.5 | 184.4 | 187.9 |
| case2848rte | 2848 | fd2 | 100 | 0 | 0 | 7 | 11.4 | 127.0 | 138.3 |
| case2848rte | 2848 | genco_vmdc | 0 | 76 | 24 | 9 | 41.8 | 179.9 | 221.9 |
| case2848rte | 2848 | genco_full | 0 | 0 | 100 | 25 | 38.3 | 473.6 | 511.9 |
| case6470rte | 6470 | flat | 0 | 0 | 100 | 25 | 0.0 | 1115.8 | 1115.8 |
| case6470rte | 6470 | dc | 0 | 0 | 100 | 25 | 8.6 | 1106.8 | 1115.8 |
| case6470rte | 6470 | fd2 | 0 | 0 | 100 | 25 | 27.6 | 1055.8 | 1083.6 |
| case6470rte | 6470 | genco_vmdc | 0 | 0 | 100 | 25 | 94.0 | 1122.9 | 1217.1 |
| case6470rte | 6470 | genco_full | 0 | 0 | 100 | 25 | 85.4 | 1120.3 | 1205.7 |
| case6495rte | 6495 | flat | 0 | 0 | 100 | 25 | 0.0 | 997.5 | 997.5 |
| case6495rte | 6495 | dc | 0 | 0 | 100 | 25 | 8.2 | 999.4 | 1007.2 |
| case6495rte | 6495 | fd2 | 0 | 0 | 100 | 25 | 26.0 | 952.6 | 976.9 |
| case6495rte | 6495 | genco_vmdc | 0 | 0 | 100 | 25 | 67.6 | 1009.5 | 1077.5 |
| case6495rte | 6495 | genco_full | 0 | 0 | 100 | 25 | 59.4 | 1004.4 | 1063.9 |
| case6515rte | 6515 | flat | 0 | 0 | 100 | 25 | 0.0 | 1015.5 | 1015.5 |
| case6515rte | 6515 | dc | 0 | 0 | 100 | 25 | 8.2 | 1028.4 | 1036.6 |
| case6515rte | 6515 | fd2 | 0 | 0 | 100 | 25 | 26.1 | 962.9 | 989.0 |
| case6515rte | 6515 | genco_vmdc | 0 | 0 | 100 | 25 | 77.7 | 1011.8 | 1090.0 |
| case6515rte | 6515 | genco_full | 0 | 0 | 100 | 25 | 69.5 | 990.1 | 1059.6 |
| case2869pegase | 2869 | flat | 100 | 0 | 0 | 5 | 0.0 | 86.2 | 86.2 |
| case2869pegase | 2869 | dc | 100 | 0 | 0 | 5 | 3.8 | 83.7 | 87.7 |
| case2869pegase | 2869 | fd2 | 100 | 0 | 0 | 3 | 11.0 | 49.2 | 60.4 |
| case2869pegase | 2869 | genco_vmdc | 100 | 0 | 0 | 4 | 37.8 | 65.8 | 103.6 |
| case2869pegase | 2869 | genco_full | 100 | 0 | 0 | 5 | 33.9 | 79.7 | 113.7 |
| pglib_opf_case8387_pegase | 8387 | flat | 69 | 0 | 31 | 7 | 0.0 | 561.7 | 561.7 |
| pglib_opf_case8387_pegase | 8387 | dc | 100 | 0 | 0 | 7 | 18.1 | 546.2 | 564.3 |
| pglib_opf_case8387_pegase | 8387 | fd2 | 100 | 0 | 0 | 3 | 50.8 | 231.6 | 282.3 |
| pglib_opf_case8387_pegase | 8387 | genco_vmdc | 100 | 0 | 0 | 5 | 135.9 | 385.8 | 521.6 |
| pglib_opf_case8387_pegase | 8387 | genco_full | 100 | 0 | 0 | 7 | 117.8 | 538.7 | 656.5 |
| case9241pegase | 9241 | flat | 89 | 0 | 11 | 6 | 0.0 | 546.4 | 546.4 |
| case9241pegase | 9241 | dc | 100 | 0 | 0 | 6 | 20.4 | 533.9 | 555.7 |
| case9241pegase | 9241 | fd2 | 100 | 0 | 0 | 3 | 59.5 | 271.0 | 330.7 |
| case9241pegase | 9241 | genco_vmdc | 0 | 0 | 100 | 25 | 96.3 | 2435.4 | 2530.9 |
| case9241pegase | 9241 | genco_full | 0 | 0 | 100 | 25 | 75.8 | 2406.2 | 2482.1 |
| pglib_opf_case2383wp_k | 2383 | flat | 100 | 0 | 0 | 5 | 0.0 | 107.8 | 107.8 |
| pglib_opf_case2383wp_k | 2383 | dc | 100 | 0 | 0 | 5 | 4.4 | 105.6 | 109.9 |
| pglib_opf_case2383wp_k | 2383 | fd2 | 100 | 0 | 0 | 3 | 12.6 | 63.6 | 76.1 |
| pglib_opf_case2383wp_k | 2383 | genco_vmdc | 100 | 0 | 0 | 4 | 41.1 | 84.5 | 125.5 |
| pglib_opf_case2383wp_k | 2383 | genco_full | 100 | 0 | 0 | 5 | 36.7 | 88.9 | 125.6 |
| pglib_opf_case3012wp_k | 3012 | flat | 100 | 0 | 0 | 5 | 0.0 | 136.7 | 136.7 |
| pglib_opf_case3012wp_k | 3012 | dc | 100 | 0 | 0 | 5 | 5.7 | 134.8 | 140.3 |
| pglib_opf_case3012wp_k | 3012 | fd2 | 100 | 0 | 0 | 3 | 16.4 | 80.6 | 96.9 |
| pglib_opf_case3012wp_k | 3012 | genco_vmdc | 100 | 0 | 0 | 5 | 37.2 | 134.2 | 171.1 |
| pglib_opf_case3012wp_k | 3012 | genco_full | 0 | 0 | 100 | 25 | 31.5 | 717.2 | 748.7 |
| pglib_opf_case3375wp_k | 3374 | flat | 100 | 0 | 0 | 5 | 0.0 | 151.9 | 151.9 |
| pglib_opf_case3375wp_k | 3374 | dc | 94 | 2 | 4 | 5 | 6.2 | 149.6 | 156.0 |
| pglib_opf_case3375wp_k | 3374 | fd2 | 100 | 0 | 0 | 3 | 18.1 | 88.8 | 106.8 |
| pglib_opf_case3375wp_k | 3374 | genco_vmdc | 0 | 0 | 100 | 25 | 62.1 | 739.3 | 801.4 |
| pglib_opf_case3375wp_k | 3374 | genco_full | 0 | 0 | 100 | 25 | 55.9 | 788.4 | 844.3 |

## Success-test verdict (verdict.py)

# Verdict (generated by verdict.py): **NO-GO**

GENCO arm used: `genco_vmdc` (pooled verified-root rate over all grids: genco_vmdc 36.4%, genco_full 36.4%)

- Condition (a): **False** (families passing 10-pt test: none; strict variant: False)
- Condition (b): **False**

## Condition (a): family rates (verified-root %, mean over the family's grids)

| family | GENCO | best classical | its rate | diff (pts) | passes |
|---|---|---|---|---|---|
| RTE | 0.0 | fd2 | 40.0 | -40.0 | False |
| PEGASE | 66.7 | dc | 100.0 | -33.3 | False |
| Polish | 66.7 | flat | 100.0 | -33.3 | False |

## Condition (b): grids >= 5,000 buses

| grid | GENCO % | best classical % | equal conv. | ref classical | GENCO ms | ref ms | ratio | >=20% faster | satisfied |
|---|---|---|---|---|---|---|---|---|---|
| case6470rte | 0 | 0 | True | fd2 | 1217 | 1084 | 1.12 | False | False |
| case6495rte | 0 | 0 | True | fd2 | 1078 | 977 | 1.10 | False | False |
| case6515rte | 0 | 0 | True | fd2 | 1090 | 989 | 1.10 | False | False |
| pglib_opf_case8387_pegase | 100 | 100 | True | fd2 | 522 | 282 | 1.85 | False | False |
| case9241pegase | 0 | 100 | False | fd2 | 2531 | 331 | 7.65 | False | False |

Pooled rates tie exactly (genco_vmdc 36.4%, genco_full 36.4%), so the tie rule picked genco_vmdc; genco_full would also fail (RTE 20% [case1888 only], PEGASE 66.7%, Polish 33.3%; see summary.csv).

## Controls

Identity: Newton started at the label converges in 0 iterations with the label unchanged on all 11 grids (rows below). Released-GENCO errors against the labels, per grid (median over scenario-bus pairs; branch angle difference over in-service branches):

| grid | buses | scen | median abs dVm, GENCO raw (pu) | GENCO REF/PV put back | setpoint/1.0 (flat, DC) | median branch dVa GENCO (deg) | DC | flat | mean branch dVa GENCO | DC | flat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| case1888rte | 1888 | 100 | 0.0206 | 0.0206 | 0.0324 | 0.442 | 0.055 | 0.542 | 1.333 | 0.179 | 1.506 |
| case2848rte | 2848 | 100 | 0.0119 | 0.0118 | 0.0321 | 0.341 | 0.047 | 0.432 | 1.361 | 0.123 | 1.087 |
| case6470rte | 6470 | 100 | 0.0116 | 0.0116 | 0.0310 | 0.383 | 0.093 | 0.634 | 1.267 | 0.198 | 1.362 |
| case6495rte | 6495 | 100 | 0.0130 | 0.0130 | 0.0409 | 0.361 | 0.106 | 0.639 | 1.105 | 0.222 | 1.381 |
| case6515rte | 6515 | 100 | 0.0130 | 0.0130 | 0.0275 | 0.451 | 0.098 | 0.738 | 1.139 | 0.215 | 1.578 |
| case2869pegase | 2869 | 99 | 0.0037 | 0.0037 | 0.0267 | 0.415 | 0.077 | 0.653 | 1.157 | 0.228 | 1.569 |
| pglib_opf_case8387_pegase | 8387 | 99 | 0.0028 | 0.0028 | 0.0347 | 0.542 | 0.139 | 0.697 | 1.798 | 0.455 | 1.805 |
| case9241pegase | 9241 | 99 | 0.0065 | 0.0065 | 0.0222 | 0.598 | 0.120 | 0.749 | 2.653 | 0.416 | 1.909 |
| pglib_opf_case2383wp_k | 2383 | 99 | 0.0051 | 0.0051 | 0.0235 | 0.218 | 0.053 | 0.295 | 0.757 | 0.134 | 0.953 |
| pglib_opf_case3012wp_k | 3012 | 99 | 0.0071 | 0.0071 | 0.0234 | 0.161 | 0.036 | 0.168 | 1.141 | 0.111 | 0.751 |
| pglib_opf_case3375wp_k | 3374 | 99 | 0.0095 | 0.0095 | 0.0224 | 0.257 | 0.053 | 0.267 | 1.178 | 0.193 | 1.138 |

Identity control (Newton started AT the label):

| grid | scenarios | max iterations | all converged | max change (pu) |
|---|---|---|---|---|
| case1888rte | 100 | 0 | True | 0.0e+00 |
| case2848rte | 100 | 0 | True | 0.0e+00 |
| case2869pegase | 99 | 0 | True | 0.0e+00 |
| case6470rte | 100 | 0 | True | 0.0e+00 |
| case6495rte | 100 | 0 | True | 0.0e+00 |
| case6515rte | 100 | 0 | True | 0.0e+00 |
| case9241pegase | 99 | 0 | True | 0.0e+00 |
| pglib_opf_case2383wp_k | 99 | 0 | True | 0.0e+00 |
| pglib_opf_case3012wp_k | 99 | 0 | True | 0.0e+00 |
| pglib_opf_case3375wp_k | 99 | 0 | True | 0.0e+00 |
| pglib_opf_case8387_pegase | 99 | 0 | True | 0.0e+00 |

GENCO's median branch angle-difference error is 3-9x the DC error on every grid (e.g. 0.38 vs 0.093 deg on case6470rte), which is why genco_full is a poor angle start. Its |V| is 2-10x better than the setpoint/1.0 magnitude in the median, which the genco_vmdc arm exploits, but that does not matter because DC angles are what fail (below).

## Start error (median over scenarios of max |V_start - V_label|)

Median over scenarios of max |V_start - V_label| (complex, pu), per start:

| grid | flat | dc | fd2 | genco_vmdc | genco_full |
|---|---|---|---|---|---|
| case1888rte | 0.782 | 0.495 | 0.132 | 0.505 | 0.773 |
| case2848rte | 0.833 | 0.219 | 0.402 | 0.398 | 1.144 |
| case2869pegase | 2.084 | 0.627 | 0.017 | 0.627 | 2.072 |
| case6470rte | 1.729 | 0.792 | 1.244 | 0.818 | 1.992 |
| case6495rte | 1.095 | 0.476 | 1.204 | 0.649 | 1.141 |
| case6515rte | 1.659 | 0.677 | 1.240 | 0.692 | 1.534 |
| case9241pegase | 2.175 | 1.935 | 0.062 | 1.935 | 2.182 |
| pglib_opf_case2383wp_k | 1.217 | 0.167 | 0.009 | 0.129 | 1.118 |
| pglib_opf_case3012wp_k | 1.302 | 0.286 | 0.047 | 0.250 | 2.111 |
| pglib_opf_case3375wp_k | 1.614 | 0.522 | 0.043 | 0.780 | 1.856 |
| pglib_opf_case8387_pegase | 2.038 | 1.257 | 0.076 | 1.257 | 2.055 |

## Surprising or noteworthy

- genco_vmdc and dc have almost identical start errors on the PEGASE grids (0.627, 1.935, 1.257): the max error is set by angles, so improving |V| alone does not move the start. fd2 cuts that max error 10-100x on PEGASE and Polish; no GENCO arm does.
- On case2848rte, flat, dc and genco_vmdc land on a wrong root in 73-76% of scenarios and diverge in the rest; only fd2 reaches the verified root (100%). genco_full diverges 100%.
- genco_full converges 100% on case1888rte while genco_vmdc diverges 100% there: its predicted angles, not its magnitudes, are what help on that grid. It is erratic across grids (0% on case3012wp, case3375wp, 99/99 on case2383wp), so no robustness claim is supported.
- case6470/6495/6515rte (min label Vm 0.53-0.54 pu, the low-voltage RTE family): every start diverges in every scenario, including fd2, which sets the failure floor for this experiment. These are the grids the plan most hoped to rescue. A classical fd10 start (about +20 ms) reaches the verified root on these grids (auditor's spot check); the 0% floor is the protocol's choice of k = 2, not a classical failure. (audit genco-make-it-work R1-F3)
- The pooled-rate tie between the two GENCO arms (36.4% each) is a coincidence: the arms succeed on different grids (vmdc: case2869, 8387, 2383wp, 3012wp; full: case1888, 2869, 8387, 2383wp).

## Caveats and deviations (candid)

1. **Timing under shared load.** The box was loaded (1-minute load average 11-46 during the runs); CPU work was pinned (taskset) but shared. Two worker processes ran Newton on different core sets (cores 16-19 and 12-15), so cross-grid timing is not comparable. Within a grid all five starts were run interleaved per scenario by the same process, so within-grid comparisons are fair. Timing ratios of ~1.1x (RTE 6470/6495/6515) are noise-level and irrelevant because nothing converges there.
2. **GENCO start cost** is a per-grid constant: the median per-scenario time of the in-memory end-to-end pipeline at batch 64 (graph build, normalisation, collate, H2D, forward, D2H; `timing/<grid>.json`), measured with load 35-45, dominated by CPU graph construction (about 12 ms of ~26 ms per scenario on case1888rte). Forward-only per scenario is 9-48 ms (batched): below fd2's total on every grid and below fd2's start cost on 5 of 11. The realistic end-to-end GENCO start (graph build, normalisation, collate, transfer, forward; genco_vmdc arm) costs 29-136 ms per scenario against fd2's 9-60 ms, 1.6x to 3.7x on all 11 grids (genco_full arm: 26-118 ms, 1.3x to 3.4x), split between graph construction and the forward pass; the verdict rests on convergence failures, not on cost (corrected per audit genco-make-it-work R1-F4 and claims audit R2-F6). Batch 64 did not fit in 24 GB for the 20M-parameter model at 8,387 and 9,241 buses, so those two use batch 32 (also allowed by protocol). Even a 3x faster GPU/CPU path would not flip the verdict because the failures are convergence failures.
3. **GENCO predictions on case8387 and case9241 were generated at graphkit batch 16** (batch 64 ran out of memory in `graphkit evaluate`; a monkeypatch of the config batch size in `score_genco.py`, env `EVAL_BATCH`). The in-memory forward reproduces the saved predictions to 3e-5 (|V|) and 3e-5 rad (control in `timing/*.json`), so batch size does not change the result.
4. **DC cost proxy.** The protocol stated that the timed own DC solve agrees with datakit `Va_dc` to ~2e-10. That is wrong: the own solve uses B' = 1/x (the timing-table recipe) while `Va_dc` uses x/(r^2+x^2); they differ by a median 0.9-4.7 deg max per scenario (`dc_vs_va_dc_max_deg` column). The start used is `Va_dc` as pre-registered; the charged DC time is the own solve, a same-order-of-cost proxy.
5. **case3375wp_k** is converted with `--pandapower-gens` (the default PGLib semantics raised "PGLib .m bus ids are not sorted"); sgens are folded into Pd as negative demand, a known deviation from datakit generator semantics. Treat that grid's GENCO inputs as slightly off-convention. Its labels remain exact roots.
6. The six non-RTE grids have 99 scenarios (graphkit's split drops one), RTE 100. 100 scenarios per grid means a rate step is 1 point; CIs are not computed.
7. Not done: no seeds or checkpoint variants (only the released checkpoint); no damping/line-search arms (excluded by protocol); no GENCO+fd2 hybrid. Audit: genco-make-it-work round 1 (verdict reproduced; wording fixes queued as an owner decision; round 2 after an owner decision).
8. Prediction runs wrote GraphKit config/log files into `data/external/genco-ladder/{configs,logs}/` under new names `*_genco-robust-start*`; the RTE trees are symlinked into `data/external/genco-robust-start/`.

## Files

`protocol.json`, `verdict.py` -> `verdict.json`, `verdict.md`; `summary.csv`, `summary.md`; `per_scenario/<grid>.csv` (grid, scenario, start, status, iterations, final_mismatch, max_dVm, max_dVa_deg, start_s, newton_s, total_s, start_max_abs_dV, ...); `identity/`, `controls.py` -> `controls.csv`, `controls.md`; `timing/`, `timing_genco.py`; `run_starts.py`, `score_genco.py`, shell launchers and logs. `mlruns/` (GENCO predictions, 910 MB) is gitignored.
