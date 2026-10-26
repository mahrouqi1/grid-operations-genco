# Leak check: does the GENCO zero-shot |V| result see hidden quantities? (2026-10-22)

> **UNAUDITED.** Independent read-only check by a fresh agent; no `/audit` round has cleared it. Scratch scripts and raw
> outputs are in the session scratchpad (`audit1.py`, `norm_test.py`, `abl2.py`, `limits*.py`, `oosbase.py`, `refva.py`);
> nothing in the repo besides this file was written. Studies covered: `2026-10-21-genco-voltage-inference` (locked-15),
> `-oos`, `-control`, `-family-vs-size`, `-generator-control`, and `2026-10-21-genco-voltage-wls`. CPU only, 2 threads, nice 15.

Tags: **LEAK** = a hidden quantity (true Pg, Qg, Va, Va_dc, unmeasured |V|, a setpoint) reaches a model or comparator input.
**POSSIBLE LEAK** = a channel that carries information about the unknown dispatch or that favours one side, with the measured size.
**CLEAN** = traced and confirmed absent.

## Verdict

| channel | verdict |
|---|---|
| GENCO input features at an unmeasured bus and at a generator bus | **CLEAN** for Pg, Qg, Va, Va_dc, Vm and any |V| setpoint. **POSSIBLE LEAK (small, asymmetric)** for generator limits and costs (see F2). |
| GENCO normaliser (baseMVA fit on label columns) | **POSSIBLE LEAK, small and not in GENCO's favour**: a per-grid constant fitted on Pg/Qg labels of 1-2 other scenarios; swapping it for an inputs-only constant moves the median error by -3.6% to +5.5% in both directions. |
| Interpolators I1-I5 | **CLEAN** (read only Vm at measured buses, Pd, Qd, branch graph, Ybus). They use less than GENCO, not more. |
| WLS | **CLEAN** (loads, flags, in-service gen list, limits, Ybus, REF angle, measured |V|). |

## Part 1. What GENCO is actually fed

Path: the datamodule normalises (`normalizers.py:186-232`), then the task transform runs `RemoveInactiveBranches`,
`RemoveInactiveGenerators`, the (patched) mask, then `ApplyMasking` (`transforms.py:80-95`), which overwrites every masked
entry with `mask_value = 0.0`. The model input is therefore the masked tensor only; `mask_dict` is used afterwards to pick
outputs (`gnn_heterogeneous_gns.py:192-199, 241`). The graphkit actually imported (`conda_envs/grid-operations-genco/.../site-packages/gridfm_graphkit`) is byte-identical to the
`data/external/gridfm-graphkit-repro` tree I read for the 9 relevant files (checked with `cmp`).

**Feature list is explicit, so extra parquet columns never reach the model.** `hetero_preprocess.py:14-32` builds bus x from exactly
`[Pd, Qd, Qg, Vm, Va, PQ, PV, REF, min_vm_pu, max_vm_pu, min_q_mvar, max_q_mvar, GS, BS, vn_kv]` and gen x from
`[p_mw, min_p_mw, max_p_mw, cp0, cp1, cp2, in_service]` (`:203`). The DC columns (`Va_dc`, `Pf_dc`) and the bus-table `Pg` column are in
the parquet but are never read by graphkit. So the `ladder_to_datakit.py:129-130` pattern (DC angles solved from the true `pg_gen`)
**does exist in the whole-grid / generator-control data, but only as unread columns** (CLEAN for these studies; it would become a LEAK only if a
`Va_dc` feature is added to BUS_FEATURES, which is what plan-rev3 proposes).

**DetMask** (`2026-10-21-genco-state-inference/run_genco.py:31-49`): Vm masked at `~vis` (`:42`), Va masked at `~ref` (`:43`), Qg masked at
every bus (`:44`), Pg masked at every gen (`:45`), branch P and Q masked (`:47`). `vis` comes from `C.measured_set`, a seeded permutation of bus
indices that touches no label (`common.py:32-36`).

Empirical dump of the model input, case14, first test batch (64 scenarios), DetMask 20%, fraction of rows with a nonzero entry:

| feature | unmeasured PQ bus | unmeasured PV (generator) bus | unmeasured REF bus |
|---|---|---|---|
| Pd, Qd | 0.88 (loads) | 0.74 (loads) | 0 (no load) |
| **Qg, Vm, Va** | **0, 0, 0** | **0, 0, 0** | **0, 0, 0** (REF Va is nonzero-masked only because it is visible; its value is 0.0 in every scenario of all 15+32+19+17+36 grids checked) |
| PQ / PV / REF flags | 1 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 1 |
| min_vm, max_vm | 0.94, 1.06 (case limits) | same | same |
| min_qg, max_qg (sum over the bus's gens) | 0.02 (only bus-level aggregates of gens at the bus) | 0.76 / 1.0 | 0 / 1.0 |
| GS, BS, vn_kv | static shunts, 1.0 | static | static |

Generator nodes: `p_mw` nonzero fraction 0.0 (masked); `min_p_mw`, `max_p_mw`, `c1` nonzero on 40% (the case14 condensers have zero Pmax and cost);
`c0`, `c2` zero; in-service status is applied by removing off gens, so the surviving topology is visible. Edge features: `pf`, `qf` 0.0 (masked);
admittances, tap, angle limits and `rate_a` visible.

So at an unmeasured bus GENCO sees: loads, bus type flag, voltage limits, generator Q limits (generator buses), static shunts, line parameters,
generator P limits and cost coefficients (generator nodes), the REF angle, and the |V| at the other measured buses through message passing.
It does **not** see Pg, Qg, any Va except the case-constant REF angle, any `Va_dc`, any |V| setpoint, or the slack's dispatched power.

**F1. Slack/REF marker and REF angle visible. CLEAN.** REF flag is known grid data (WLS gets it too, `run_wls.py:33,42`). REF Va is a case constant
(0.0 in all 119 grid/piece trees checked, `refva.py`), not a per-scenario state quantity.

**F2. Generator P limits and cost coefficients visible to GENCO only. POSSIBLE LEAK (information asymmetry, small).**
Evidence: `run_genco.py:5-8` and `protocol.json` ("limits and costs ... visible (p_low=0)"); `hetero_preprocess.py:36-43`.
Why it matters: the locked-15 labels are AC-OPF solutions: datakit PF mode runs OPF on the cost-permuted network first
(`gridfm_datakit/process/process_network.py:819-875`, `args.log`: `mode: pf`, `generation_perturbation: cost_permutation`), so dispatch is a function of loads,
limits, costs and admittance, all of which GENCO sees and no comparator does. In those data 57-98% of in-service gens sit exactly at Pmin or Pmax
(`limits2.py`; e.g. case118 90%, case1354 95%, case3120sp_k 96%). In the ladder-generated whole grids the base dispatch is constant across scenarios and Pg equals Pmax on 71-95% of in-service gens of case2000_goc, case4020_goc, case4837_goc, case9591_goc (`limits.py`), so there the limit alone nearly states the dispatch.
Measured (case14, 30, 57, 118; 20%, seed 0, same checkpoint, same DetMask, CPU; default reproduces the stored medians exactly,
max sorted per-scenario difference 1.5e-6 pu):

| grid | default median | costs masked | limits masked | both masked |
|---|---|---|---|---|
| case14 | 0.01670 | 0.01508 (-9.7%) | 0.01744 (+4.4%) | 0.01738 (+4.1%) |
| case30 | 0.00811 | 0.00788 (-2.8%) | 0.00814 (+0.4%) | 0.00814 (+0.4%) |
| case57 | 0.00876 | n/a | 0.00915 (+4.4%) | 0.00898 (+2.4%) |
| case118 | 0.01266 | 0.01257 (-0.8%) | 0.01261 (-0.4%) | 0.01247 (-1.5%) |

Reading: GENCO does not rely on costs (error falls when they are removed); it gains at most about 4% from Pmin/Pmax on 2 of 4 grids. That is the same size as the
margin on the borderline grids (case14 pooled ratio 0.95, case118 0.98), so on those grids the win/loss call is within this effect. Caveats: the ablation
shifts the input distribution away from what the checkpoint was pretrained on, so it understates reliance if anything; four small grids only; the ladder-generated
large grids (where limits ~ dispatch) were NOT measured.

**F3. Visible measured |V| passes through unchanged and noise is added only to inputs. CLEAN.** `diag.vm_visible_dev` is 0.0 in every logged run; the
noisy mask (`oos/run_genco_o.py` NoisyDetMask) perturbs only the visible entries, with the same z as the interpolators (`ocommon.py:60`).

**F4. Carved pieces fold parent-base-point tie flows into boundary-bus loads. POSSIBLE LEAK (shared by every method, not quantified).**
`control/carve.py:123`: `Pd2 = Pgb - Re(V conj(Y_sub V))*100` sets boundary Pd/Qd from the parent's solved base point, so the "known load" at boundary buses carries a
dispatch-dependent term (parent base dispatch, not the per-scenario one). GENCO, I4, WLS and the zero-injection rows of WLS all read the same Pd/Qd, so this is
not an asymmetry between GENCO and the comparators; it only means "loads known" is not purely load at those buses. Tier-b pieces are tie-dominated (`control/report.md`).

## Part 2. The normaliser

`normalizers.py:130-170`: `fit` reads `bus_data[Pd, Qd, Qg, vn_kv]` and `gen_data[p_mw]` (`:141-142`) and sets
`baseMVA = percentile95(nonzero Pd, Qd, Pg, Qg)` (`:155-169`). It is fitted on the **train split** (`hetero_powergrid_datamodule.py:362-371`). With `test_ratio: 0.99` (case118_eval.yaml:14)
the train split is **1-2 scenarios per grid** (15 locked grids: 2, 2, ..., 1 for the big ones; OOS staged trees: 1-2), disjoint from the scored test scenarios. So
the constant carries (a) the Pg/Qg labels of those 1-2 other scenarios and (b) nothing about the test scenario's own dispatch. It is a per-grid scale factor on Pd, Qd, Pmin,
Pmax, and the admittance scaling `baseMVA_orig/baseMVA`, i.e. it rescales every input power by one number. This is the native graphkit/GENCO recipe (the checkpoint was pretrained with the same rule).

Size of the effect on the scale: the label-fitted baseMVA is 1.01-2.86x (median 1.62x) an inputs-only (Pd, Qd) 95th percentile on the 32 OOS trees (`oosbase.py`),
and 0.94-2.26x on the locked-15 (`audit1.py`). So the scale is not a rounding detail.

Effect on the result, same four-grid CPU test, 20%, seed 0, same checkpoint, only `data.normalizer_preset.baseMVA` changed (`norm_test.py`):

| grid | default (label-fitted) | Pd,Qd-only | Pd,Qd,Pmax | nominal 100 |
|---|---|---|---|---|
| case14 | 88.96 -> 0.01670 | 39.41 -> 0.01620 (-3.0%) | 90.88 -> 0.01671 (+0.1%) | 100 -> 0.01697 (+1.6%) |
| case30 | 42.76 -> 0.00811 | 23.58 -> 0.00827 (+2.0%) | 79.42 -> 0.00847 (+4.5%) | 100 -> 0.00856 (+5.5%) |
| case118 | 123.2 -> 0.01266 | 82.66 -> 0.01220 (-3.6%) | 223.0 -> 0.01323 (+4.5%) | 100 -> 0.01241 (-2.0%) |

**F5. baseMVA from label columns. POSSIBLE LEAK, small, sign mixed.** The inputs-only constant is better on 2 of 3 grids (case14, case118) and worse on case30, so the
label-fitted constant does not systematically help GENCO; the spread is about +/-5%, which is again the order of the borderline margins (case118: GENCO 0.01266 vs I2 0.01237 loses at default, 0.01220 would win with the Pd,Qd-only constant).
It is a scale-sensitivity of the pretrained checkpoint more than a leak. Not measured on any grid above 118 buses.
An inputs-only recipe would be the cleaner protocol for a deployed claim; the dispatch-scale information in it is only "typical Pg/Qg magnitude of 1-2 other scenarios".

## Part 3. The interpolators

`vcommon.py:114-125` `predict_all`: `y = np.where(m, vt, 0.0)` (`:116`) so unmeasured Vm is zeroed before any method sees it; the other arguments are `Pd`, `Qd` (`:118`),
the in-service branch graph (`graph`, `br.br_status == 1`) and Ybus off-diagonals. `harmonic` reads `y[k]` only; `i3_nearest` copies `y[src]` for a measured `src`;
`i4_loadaware` fits two coefficients on out-of-fold residuals at measured buses (`:66-85`, `X` at `:77` uses Pd,Qd of measured buses); `i5_gp` uses measured `y[k]` and the graph (`:84-112`).
`ocommon.py:32-66` repeats this for OOS/carved arms (identical logic; `y` zeroed at `:61`; noise added only to measured values, `:60`). `grep` over `vcommon.py`, `ocommon.py`,
`wlslib.py`, `run_wls.py`, `run_interp.py` finds no use of `Pg`, `Qg`, `Va`, `p_mw`, `pf`, `qf`, case files, or `base_case` (the only label reads are `bus.Vm` at measured buses and `bus.Va[ref]` in WLS).
Note the module docstring claims REF/PV/PQ flags are used; the code does not use them, so the interpolators are, if anything, handicapped relative to GENCO (CLEAN, asymmetric against the comparators).
The state-inference baselines B1-B3 (`common.py:152-217`) use case-file Pg and Vg and an "oracle" with the label Pg, but they are not the comparators in these voltage studies (candidate 8+ use I1-I5 only).

## Part 4. WLS

`run_wls.py:28-49`: measured |V| (noisy variants noised, `:30`), `Pd`, `Qd` (`:31`), generator buses from the in-service gen list and PV/REF flags (`:32-34`), limits-based priors
(`:35-41`: a single scalar `s` from total load, `pmin + s (pmax - pmin)`; no case-file dispatch; the sigma on the prior is half the range, so limits shape it), REF angle (`:42`), Ybus.
Zero-injection rows at non-generator buses use the known load only. Nothing is read from an unmeasured bus's label. CLEAN. (Its priors use the same Pmin/Pmax that GENCO sees, so F2 is symmetric for WLS in kind
though not in strength.)

## Not checked / limits of this check

- The ablations (F2, F5) ran on 3-4 grids of 118 buses or fewer, one measured fraction (20%) and one mask seed (seed 0). No ablation on the OOS or carved studies, whose ladder-generated tables have constant dispatch and
  limits ~ dispatch on up to 95% of gens; for those studies F2 is the open question.
- I did not re-run the OOS/carved GENCO numbers; claims there rest on code identity (`run_genco_o.py` calls candidate 7's `run_variant` unchanged, md5-identical `run_interp.py` across studies).
- The generated data's OPF-versus-PF semantics (datakit PF mode = OPF first) were read from datakit source and `args.log`, not re-derived.
- Planned plan-rev3 `Va_dc` column is not present in any of these inputs (above); this file says nothing about it beyond that.
