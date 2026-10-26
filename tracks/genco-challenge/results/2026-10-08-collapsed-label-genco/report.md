# Does GENCO learn or reproduce collapsed power-flow labels? (2026-10-08)

> **UNDER REVIEW, 2026-10-08 12:10 (audit genco-diagnostics R1-F1, invalidates-result).** The
> bottom line below, "GENCO does not reproduce the collapsed point", holds only for the
> power-flow task the report scored (PF mask, PF decoder). Under the PRETRAINING mask, the task
> the collapsed labels were trained under (about 80% of voltages hidden), the auditor found the
> pretrained checkpoint predicts the collapsed Vm at 97% of hidden collapsed buses on case4837_idx_35
> (median -0.050 vs label -0.061) and case20758_idx_49 (median 0.012 vs label 0.022); the other 7
> subgrids were not run. So GENCO DID learn the bad labels under its own training task. The
> 'nothing filters them' finding (Question 2) stands. Full re-analysis in
> `../2026-10-09-collapsed-label-pretrain-mask/`. Do not quote the bottom line until that lands.

UNAUDITED. Scripts in this directory; data under `data/external/` (gitignored).

## Bottom line

**Question 1. GENCO does not reproduce the collapsed point.** On every collapsed scenario we scored (5 held-out ladder
grids, 1,083 collapsed scenarios in 9 pretraining subgrids, 0 exceptions) the pretrained model predicts a normal
voltage at the collapsed buses: median predicted Vm 0.90 to 1.10 pu, against collapsed labels of about 0 or
negative. Not one scenario has a predicted Vm below 0.5 at a collapsed bus. Where our Newton finds the physical
root, GENCO is closer to it than to the label in 100% of scenarios (case4837_idx_35: 0.009 pu from the root vs
1.15 pu from the label). In-distribution, on subgrids whose training labels are up to 55.6% collapsed
(case4837_idx_35), the pretrained model has therefore not fit the bad labels. The effect on the score is
a large error, not a learned collapse: the bad label contributes about 1 pu of error at each collapsed bus.

**Question 2. The training pipeline does not filter these scenarios (not filtered, as far as the code shows).** The only
data-generation filter is solver termination status; the voltage-magnitude validation exists only for OPF mode and
is a report-only CLI check; the pretraining loader reads every scenario of each network and applies no value
filter. The paper describes no such filter either. One caveat: the pretraining run's exact script is not in the
repo we have (see caveats), so "cannot tell" applies to anything outside the files below.

Note: the collapsed Vm labels are supervised training targets (Question 2 below), so this shows the checkpoint's outputs
are not collapsed, not that the labels cost nothing; we did not train.

## Question 1 evidence

Method (`analyze.py`): collapsed scenario = label has a bus with Vm < 0.5 pu or negative. GENCO output =
graphkit's own `evaluate --save_output` on the pretrained checkpoint (`gridfm/genco-pf-transfer`
`pretraining/best_model_state_dict.pt`, zero-shot, same command and config as `ladder_eval.run_graphkit`).
Physical root = our sparse Newton (`core/grid_ops/reference_pf.newton_raphson_pf`) on the scenario's own
`y_bus_data` Ybus and Sbus, from flat then DC start, tol 1e-8, max 30 iterations, accepted only if every Vm >= 0.5
and an independent power-mismatch check (P at PV+PQ, Q at PQ) is < 1e-8 pu. Each flagged label itself satisfies
the power-flow equations (max mismatch 1.0e-08 pu). For a bus with |Vm| < 0.5 it is a genuine but non-physical root;
16 of the 1,085 flagged scenarios have no bus with |Vm| < 0.5 (negative Vm only), and 11 of those are normal operating
points written with a negative magnitude (label-validation report, audit genco-eval R2-F1), so "collapsed" here is the
flag definition, not a claim that every flagged label is a non-physical root;
the model's output is therefore not a solution of the same root.

### (b) In-distribution: pretraining subgrids, pretrained checkpoint

Sample: the same 6 bus_data partitions per subgrid as `../2026-10-07-label-validation/genco_data_check.py`
(branch_data and y_bus_data for the same partitions fetched with `fetch_extra.py`); scenarios renumbered
(`build_pretrain_eval_data.py`, map kept) and scored with `run_genco_inference.py`. The graphkit splitter rejects
test_ratio 1.0, so it used 0.9999 and left out about one load scenario per subgrid (1,083 of the 1,085 collapsed
scenarios were scored). Only the 9 affected subgrids were run; "normal" scenarios are the non-collapsed scenarios
of the same sampled partitions. Predictions are from the pretrained model with its normalizer fit on the sampled
data, as in the ladder evaluation.

| subgrid | collapsed scen. | label Vm at collapsed buses (median) | GENCO pred min Vm at collapsed buses (median, min..max) | pred < 0.5 | physical root found | MAE vs label at collapsed buses | MAE vs root at collapsed buses | normal scen.: MAE vs label (PQ) |
|---|---|---|---|---|---|---|---|---|
| case4837_goc_idx_35 | 665 | -0.061 | 1.093 (1.001..1.102) | 0% | 69/665 | 1.152 | 0.009 | 0.0027 |
| case20758_epigrids_idx_49 | 275 | 0.022 | 1.007 (0.944..1.075) | 0% | 142/275 | 0.998 | 0.002 | 0.0038 |
| case9591_goc_idx_12 | 71 | 0.477 | 1.020 (0.952..1.091) | 0% | 3/71 | 0.549 | 0.362 | 0.0077 |
| case5658_epigrids_idx_8 | 28 | 0.000 | 1.065 (0.872..1.076) | 0% | 28/28 | 1.065 | 0.005 | 0.0047 |
| case4837_goc_idx_28 | 16 | -0.123 | 0.902 (0.898..0.910) | 0% | 16/16 | 1.026 | 0.001 | 0.0037 |
| case19402_goc_idx_47 | 10 | -0.043 | 1.097 (1.084..1.100) | 0% | 10/10 | 1.142 | 0.002 | 0.0030 |
| case10480_goc_idx_40 | 9 | -0.004 | 1.077 (1.018..1.089) | 0% | 9/9 | 1.081 | 0.019 | 0.0058 |
| case4837_goc_idx_37 | 6 | 0.041 | 1.091 (1.083..1.093) | 0% | 6/6 | 1.042 | 0.001 | 0.0040 |
| case10480_goc_idx_26 | 3 | 0.006 | 1.086 (1.074..1.096) | 0% | 0/3 | 1.091 | - | 0.0015 |

Reading: the pretrained model's min Vm at collapsed buses is at or above 0.87 pu in all nine subgrids (medians 0.90 to 1.10; never below 0.5), and across all 9 subgrids every collapsed scenario is predicted normal. Normal scenarios: predicted min Vm
never below 0.5 either, and the Vm MAE on normal scenarios is 0.001 to 0.008 pu (same order as the collapsed
scenarios' error at their non-collapsed buses, 0.001 to 0.03 pu). Physical roots exist from flat/DC for only part
of the collapsed scenarios (4837_idx_35: 69 of 665; 20758_idx_49: 142 of 275; 9591_idx_12: 3 of 71; counts are in the table); for the others we cannot say whether a physical operating point exists, only that
Newton from flat/DC did not reach one, so those rows are compared to the label only. In case9591_idx_12 the label
median at the collapsed buses is 0.48 (partial collapse), the three roots found sit at 0.54 to 0.63 pu and GENCO
predicts about 1.0: error vs root 0.36 pu, so there GENCO also differs from the root, which has a low-voltage
solution that is not the normal-looking one. Per-scenario rows: `per_scenario_pretrain_collapsed.csv`;
aggregates: `results_pretrain.json`.

### (a) Held-out ladder, predictions made before validation

Predictions: the pre-validation zero-shot runs already in `data/external/genco-ladder/mlruns` (selected as the
latest run before the `raw.unvalidated-<time>` stamp; recorded per grid in `results_ladder.json`; each is checked
against the unvalidated raw by requiring `Vm_target` to equal `Vm` within 1e-4). case2736sp_k had no
pre-validation predictions (its only saved ones are post-validation, 74 scenarios), so zero-shot inference was
re-run on `raw.unvalidated-20261007-075603` (81 scenarios scored). Scenario counts below exclude one scenario the
splitter held out (96 of 97 for case1354, and so on).

| grid | scen. | collapsed | label Vm at collapsed (median) | GENCO pred min Vm at collapsed (median, min..max) | pred < 0.5 | root found | MAE vs label | MAE vs root |
|---|---|---|---|---|---|---|---|---|
| case1354_pegase | 96 | 69 | -0.000 | 1.098 (1.085..1.100) | 0% | 69/69 | 1.098 | 0.001 |
| case240_pserc | 177 | 3 | -0.000 | 1.075 (1.022..1.089) | 0% | 1/3 | 1.075 | 0.048 |
| case1888_rte | 5 | 5 | -0.176 | 1.063 (1.052..1.068) | 0% | 0/5 | 1.136 | - |
| case1951_rte | 5 | 5 | 0.151 | 1.035 (1.032..1.049) | 0% | 0/5 | 0.986 | - |
| case2736sp_k | 81 | 8 | -0.000 | 1.027 (1.013..1.051) | 0% | 8/8 | 1.027 | 0.013 |

Reading: on all five grids GENCO predicts 1.0 to 1.1 pu at the buses whose label is about 0 or negative. On
case1354 (69 of 96 scenarios collapsed) it is 0.0009 pu from our physical root at those buses. For case1888 and
case1951 no scenario has a physical root from our Newton (as the label-validation report found), so only the
label comparison exists: GENCO still predicts about 1.03 to 1.07 pu.

Side finding, not about labels: on case240 GENCO's zero-shot output has a very low Vm somewhere in almost every
NORMAL scenario (median min Vm 0.14, 99.4% of normal scenarios below 0.5, one negative, against label minimum
0.90). That is GENCO failing on that grid, independent of the collapsed labels.

## Question 2 evidence

**Data generation (gridfm-datakit 1.0.4, installed under
`conda_envs/grid-operations-genco/lib/python3.11/site-packages/gridfm_datakit/`):**
- `process/solvers.py:83-90` `run_pf`: raises (scenario skipped) only if `termination_status` is not converged
  ("PF did not converge"). That is the single filter on PF scenarios.
- `process/process_network.py:903-910`: `try: res = run_pf(...) except Exception: ... continue` writes the error
  to the error log and drops the scenario. Nothing is checked on the solution's Vm.
- `process/process_network.py:263-289` (`_run_pf_core`, Ipopt tol 1e-6) and `:225-240` (`run_pf_fast`): only the
  termination status is tested. A PF solved as a feasibility problem can converge to the collapsed root.
- `validation.py:821-836`: the Vm limit check runs only `if generated_data["mode"] != "opf": ... skipped`; in PF
  mode there is no Vm check at all. `cli.py:115` calls `validate_generated_data` as an optional post-hoc command
  that asserts and reports; it removes nothing.

**Training loader (gridfm-graphkit branch `genco-paper-repro-pretraining`**, cloned to
`data/external/gridfm-graphkit-repro-pretraining`; the `genco-paper-repro` clone lacks the PreTraining task and
carries the evaluation path):
- `gridfm_graphkit/datasets/hetero_powergrid_datamodule.py:119-164`: networks are chosen by name only (all
  directories minus validation/test, shuffled by seed, first `training_networks`); `:199-207` builds each network's
  full `HeteroGridDatasetDisk` and appends it; no scenario subsetting or value filter.
- `gridfm_graphkit/datasets/powergrid_hetero_dataset.py:71-155`: reads all of bus/gen/branch parquet, fits the
  normalizer, writes one graph per scenario; the only check is `scenario not in gen_groups.groups ... raise`
  (missing tables). No Vm test, no NaN/outlier removal.
- `gridfm_graphkit/datasets/masking.py:214-259` (`AddPretrainMask`): Vm and Va are masked with probability
  `mask_p_high` (0.8 in `HGNS_PreTrain_subnets_100.yaml`), so the label Vm of PQ buses, collapsed or not, is a
  training target.
- `gridfm_graphkit/training/loss.py:141-171` (`MaskedBusMSE`): MSE of predicted vs label Vm, Va at the masked
  entries, no clipping or weighting by label. (The config also uses `LayeredWeightedPhysics` and `MaskedGenMSE`.)
- Greps for `isnan`, `isfinite`, `clip`, `outlier`, `converge`, `< 0.5` and `filter` over both graphkit clones
  hit only the SE measurement-outlier noise (`masking.py`, unrelated) and `RemoveInactiveBranches/Generators`.

**Paper (arXiv 2608.09921, local copy `data/external/genco-paper/2608.09921.pdf`):** Table 2 reports "successfully
converged samples" and a convergence rate; Sec. 3.1.4 (PF mode) says AC-PF is solved without re-optimizing
dispatch; Sec. 5.5.3 says 100 subgrids x 9,000 PF samples = 900,000 scenarios, with no mention of a voltage or
plausibility filter. The paper therefore claims convergence, not physicality.

**Verdict: not filtered** in datakit (convergence only), not filtered in the pretraining loader, and not described
in the paper. What we cannot tell: the exact data copy used for the paper's run (published as
`gridfm/genco-pf-pretraining`, which does contain the collapsed scenarios we measured) and any private preprocessing
before upload; the training script `train.sh` calls `gridfm_graphkit train` directly on that directory.

## Caveats

- 9 of the 31 hub subgrids were scored; the other 22 had no collapsed scenarios in the sample. The 69 train
  subgrids not on the hub (and the 10 validation and 10 test subgrids) were not seen.
- The sample is 6 of about 50 partitions per subgrid; shares are sample shares.
- We measured only the released checkpoint's outputs. Whether GENCO would score better with filtered training data is
  untested (that needs a training run).
- `mae_vs_label_at_low` is dominated by the label being near 0; the root comparison is the meaningful one and
  exists for only part of the scenarios.
- The pretrained checkpoint has no saved normalizer; stats are fit on the evaluated sample, the same as in the
  existing ladder evaluation.

## Files

`analyze.py` (metrics, both parts), `build_pretrain_eval_data.py`, `run_genco_inference.py`, `fetch_extra.py`,
`results_pretrain.json`, `results_ladder.json`, `per_scenario_*_collapsed.csv`, logs.

Update 2026-10-09: the pretraining-mask re-analysis on all 9 subgrids is in `../2026-10-09-collapsed-label-pretrain-mask/report.md` (GENCO does reproduce the collapsed labels under its pretraining task, 9 of 9 subgrids); read that before quoting the bottom line above.
