# Label validation: collapsed power-flow labels dropped at the data level (2026-10-07)

UNAUDITED. Code: `src/genco_challenge/ladder_data.py` (`validate_grid`, `--validate-only`);
`genco_data_check.py` and `summarize.py` in this dir. Built by a subagent; report saved by the lead
chat; the 13-grid summary counts below were recomputed independently from `results.jsonl`.

## Rule

A scenario is invalid if any bus has Vm < 0.5 pu or any Vm < 0. For a bus with |Vm| < 0.5 this is a
collapsed root that satisfies the power-flow equations but is not an operating point (see
`../2026-10-07-genco-hybrid-warmstart/report.md` section 7). A negative Vm with |Vm| >= 0.5 is a different
case: datakit can write a normal operating point as -|V| with the angle shifted by 180 degrees (audit
genco-eval R2-F1). Such a label is still an unusable Vm/Va regression target, so the scenario is still
dropped, but `label_diagnostics` reports it as `negative_vm_sign_flip`, not as a collapsed root. Invalid scenarios are dropped; bus, gen, branch, y_bus and runtime tables are rewritten with
scenario ids renumbered 0..N-1 (`raw/scenario_map.csv` maps back; `raw/dropped_scenarios.csv` lists the
drops); `load_scenario_idx` and `scenarios_agg_load_profile.parquet` are unchanged; the original raw/ is
kept as `raw.unvalidated-<time>` and processed/ is moved aside. Grids with fewer than 20 valid scenarios
get status `insufficient_valid_scenarios` and are not scored. PV/REF reactive-limit violations are a
diagnostic, not a rejection rule: datakit's PF mode does not enforce Q limits, and GENCO's own case118
eval set violates them in 7,324 of 9,952 scenarios.

## Ladder grids

| grid | generated | dropped (negative Vm / other Vm < 0.5) | kept |
|---|---|---|---|
| case1354_pegase | 97 | 69 (60 / 9) | 28 |
| case240_pserc | 179 | 3 (2 / 1) | 176 |
| case1888_rte | 6 | 6 (6 / 0) | 0, insufficient valid scenarios |
| case1951_rte | 6 | 6 (6 / 0) | 0, insufficient valid scenarios |
| case2736sp_k | 82 | 8 (7 / 1) | 74 (not yet scored) |
| 11 grids, case5 to case300 | | 0 | unchanged |

Oracle cross-check (this repo's Newton-Raphson from flat and DC, tol 1e-8, on each scenario's own
Ybus/Sbus; agree = max |V - V_label| < 1e-4 pu), up to 10 kept labels per grid: flat agrees 140 of
140, DC 138 of 140 (the 2 misses on case240, where DC reaches a different root or diverges). Dropped
labels: flat and DC both reach a physical root (every Vm >= 0.5) on 10 of 10 sampled case1354 drops and
8 of 8 case2736 drops; case240 1 of 3 (DC only); case1888 and case1951 0 of 6 each.

## Effect on the held-out evaluation (GENCO / DC)

| grid, checkpoint | test scen. | residual MW | branch dVa deg | |V| pu (GENCO pinned / DC) |
|---|---|---|---|---|
| case1354 zero-shot | 96 -> 27 | 63.35/5.10 -> 60.08/5.04 | 0.694/0.289 -> 0.667/0.203 | 0.0072/0.0552 -> 0.0065/0.0525 |
| case1354 ft118_1000 | 96 -> 27 | 883.7 -> 876.1 | 3.175 -> 3.178 | 0.1491 -> 0.1529 |
| case1354 ft118_250000 | 96 -> 27 | 535.3 -> 508.7 | 2.154 -> 2.090 | 0.0570 -> 0.0554 |
| case240 zero-shot | 177 -> 174 | 1078.2/8.56 -> 1007.0/8.37 | 4.635/0.763 -> 4.529/0.747 | 0.0325/0.0374 -> 0.0341/0.0371 |
| case240 ft118_1000 | 177 -> 174 | 1108.8 -> 1109.2 | 5.492 -> 5.477 | 0.0326 -> 0.0310 |
| case240 ft118_250000 | 177 -> 174 | 1494.5 -> 1576.4 | 5.447 -> 5.487 | 0.0227 -> 0.0229 |
| case1888, case1951 | | insufficient valid scenarios | | |

Held-out summary now over 13 grids (zero-shot): DC has the lower power residual on 13 of 13 (GENCO/DC
2.1x to 120x); DC has the lower branch dVa on 12 of 13 (ratio 0.84 to 6.07; GENCO lower only on
case5_pjm); GENCO beats the flat start on dVa on 12 of 13; GENCO-pinned |V| beats DC's on 11 of 13 (not
case57, case162); ft118_250000 is worse than zero-shot on 10 of 12 grids for both residual and dVa (case118 is in-distribution for it and is not counted).

## GENCO's own data (measured, same rule)

(a) **case118 PF-transfer eval set**, all 9,952 scenarios: 0 with min Vm < 0.5, 0 with negative Vm;
lowest Vm anywhere 0.864 pu. Clean.

(b) **Pretraining subgrids**, HF `gridfm/genco-pf-pretraining` as of 2026-10-07: 31 subgrids are on the
hub (all from train_networks.txt, which lists 100; none of the 10 validation or 10 test subgrids).
Sample: 6 evenly spaced bus_data partitions per subgrid (34,274 scenarios, 1.2 GB with gen_data).
**1,085 of 34,274 sampled scenarios have min Vm < 0.5 or a negative Vm (the literal flag count; 684
have a negative Vm), in 9 of 31 subgrids. Of the 1,085, 1,069 (3.1%) have a bus with |Vm| < 0.5, a collapsed
magnitude; the other 16 are negative only, with every |Vm| >= 0.5 (case4837_goc_idx_35: 12,
case19402_goc_idx_47: 2, case9591_goc_idx_12: 2). Of those 16, 11 (case4837_idx_35: 9, case19402_idx_47: 2)
are the normal operating point written with a negative magnitude: |V| 1.04 to 1.08 pu, within 30 degrees of
every in-service neighbour in complex voltage (checked from the sampled partitions).** The table below uses
the literal flag (Vm < 0.5 or negative):

| subgrid | buses | sampled | Vm < 0.5 | negative |
|---|---|---|---|---|
| case4837_goc_idx_35 | 801 | 1198 | 666 (55.6%) | 538 |
| case20758_epigrids_idx_49 | 822 | 1083 | 275 (25.4%) | 93 |
| case9591_goc_idx_12 | 546 | 1132 | 72 (6.4%) | 3 |
| case5658_epigrids_idx_8 | 803 | 1159 | 28 (2.4%) | 19 |
| case4837_goc_idx_28 | 814 | 1136 | 16 (1.4%) | 11 |
| case19402_goc_idx_47 | 555 | 1067 | 10 (0.9%) | 9 |
| case10480_goc_idx_40 | 355 | 1060 | 9 (0.8%) | 5 |
| case4837_goc_idx_37 | 274 | 1131 | 6 (0.5%) | 5 |
| case10480_goc_idx_26 | 515 | 1162 | 3 (0.3%) | 1 |
| other 22 subgrids | | | 0 | 0 |

All affected buses are PQ buses (1,156 of 1,156; PV and REF: 0), 1 to 7 per affected scenario. Unlike the
ladder grids (zero-injection buses), most carry load: the median Pd of the collapsed buses is 8.8 to 74.4 MW
in eight of the nine subgrids; in case5658_idx_8 it is 0.0 MW (78% zero-load) and in case4837_idx_37, 32%
are zero-load. (Source: `low_bus_*` fields of `genco_data_check.json`, added 2026-10-08, audit R1-F11; the
earlier text had no result file behind it and the rerun confirmed it.) **This measures the published raw files only.** Whether GENCO's training pipeline
filters these scenarios was not examined, and the 69 train subgrids not on the hub (and the 10 validation and 10 test subgrids) were not checked.
Public wording was reviewed before release.

## Open decisions (owner decisions)

- Recover dropped scenarios by replacing a collapsed label with the physical root our Newton reaches
  from flat and DC (case1354 would go from 28 back to ~97 scenarios). Changes labels: an owner decision (taken 2026-10-08, see 2026-10-08-relabel).
- Train on a filtered copy of GENCO's pretraining data (~8.7 GB bus_data for the 31 subgrids on the hub).

## Caveats

- case1354 now has only 27 test scenarios; case2736sp_k is validated but not scored.
- Grids generated by the still-running ladder_data process (old code) need `--validate-only`.
