# Did GENCO learn the collapsed labels? Redo under the pretraining mask (2026-10-09)

**UNAUDITED.** Answers audit genco-diagnostics R1-F1. Scripts `run_pretrain_mask.py` (model runs) and `summarize.py` (tables); raw rows `results.jsonl`, aggregates `summary.json` / `summary.csv`, tables `tables.md`, logs `run_*.log`. Data trees under `data/external/genco-collapsed-pretrain-mask/` (symlinks to the eval copies and to `raw.unvalidated-*`, gitignored). No training, no crashed run was scored (0 tracebacks, 0 retries).

## Bottom line

**Yes: under its own pretraining task the released checkpoint reproduces the collapsed Vm labels, on 9 of 9 of the affected pretraining subgrids, at least 75 percent on each subgrid at every mask seed (wording tightened 2026-10-18, audit genco-collapsed-repair R1-F8; the pooled figure below is dominated by two grids).** Pooled over the 9 subgrids, 96.1%, 96.0% and 96.5% of the hidden collapsed PQ buses (892, 921, 904 buses for mask seeds 1000, 1001, 1002) are predicted below 0.5 pu, and the mean absolute error against the collapsed label is 0.102, 0.096, 0.091 pu. Under the power-flow task the same buses give 0.0% below 0.5 and an error of 1.065 pu (10 to 12 times larger). Per subgrid the lowest per-seed fraction below 0.5 is 0.97 (case4837_idx_35), 0.97 (case20758_idx_49), 0.75 (case9591_idx_12, label median 0.48 so the 0.5 line cuts through it), 0.96 (case5658_idx_8), 0.80 (case4837_idx_28), 0.75 (case19402_idx_47), 1.00 (case10480_idx_40), 0.78 (case4837_idx_37), 1.00 (case10480_idx_26). The median predicted Vm tracks the median label (case4837_idx_35: -0.050 vs -0.061; case20758_idx_49: 0.015 vs 0.022). Where Newton finds a physical root, the model is as far from it as the label is on four subgrids (case4837_idx_35: 1.129 vs 1.128 pu; also case20758_idx_49, case5658_idx_8 and case10480_idx_40), i.e. it reproduces the bad label, not the physical solution. On the other four subgrids with roots it sits between the label and the root (MAE vs root against the label's distance: case9591_idx_12 0.226 vs 0.896, case19402_idx_47 0.959 vs 1.240, case4837_idx_28 0.822 vs 0.969, case4837_idx_37 0.841 vs 1.008). Those are small subsets (qualified 2026-10-19, audit genco-collapsed-repair R3-F5). Control: on hidden PQ buses of normal scenarios the same runs have a Vm error of 0.0008 to 0.0037 pu and essentially nothing below 0.5, so the collapse predictions are specific, not a general low-Vm bias. This confirms the auditor's two-subgrid result (97.1% on case4837_idx_35 at seed 1000, reproduced here) and replaces the old report's reading "the model has not fit the bad labels".

**What this does and does not show.**
- The earlier statement still holds for the task it measured: under the PF mask and decoder the model outputs a normal voltage (0 of 1,143 hidden collapsed buses below 0.5). Which task counts depends on the question; the labels were trained under the pretraining task, and that is where they are reproduced.
- It is a learned input-to-collapse mapping, not necessarily memorization of these exact scenarios. Almost every collapsed scenario has a single collapsed bus and that bus is hidden (e.g. 512 of 518 hidden collapsed buses on case4837_idx_35, seed 1000, sit in a scenario with no visible collapsed bus), so the model infers the collapse from other inputs. On the held-out PEGASE grid case1354_pegase, which GENCO never saw, it also predicts the collapse at 86 to 91% of 56 hidden buses (error 0.134 pu). It does not on the held-out Polish grid case2736sp_k (0 of 4 to 7 hidden buses, error 0.99 pu), partly on case240_pserc (0.5, 0.5, 0.0 of only 1 to 2 buses) and partly on the two RTE grids (0.14 to 0.52, those scenarios have no physical root and some "collapsed" labels there are negative-sign normal points, audit genco-eval R2-F1). So the behaviour transfers within some grid families and not others; this run did not isolate what input feature drives it. For the pretraining subgrids the v3 mechanism control (`../2026-10-10-collapsed-repair-v3/mech_table.md`, added 2026-10-18, audit genco-collapsed-repair R1-F5) does: with P and Q also masked on the branches incident to a collapsed bus, the collapsed fraction on the pretrained checkpoint falls from 0.966 to 0.171 (hs) and from 0.986 to 0.180 (hb). The wrong-root branch flows are the cue, and neighbour Vm and Va alone leave about 17 percent. It was not tested on the held-out grids.
- Sample sizes are small for most subgrids (1 to 62 hidden collapsed buses per run on 7 of the 9; about 520 and 230 on the two large ones), and the fractions for those are coarse. The pooled numbers are dominated by case4837_idx_35 and case20758_idx_49; the other 7 subgrids alone pool to 0.905, 0.858, 0.887 below 0.5 (147, 155, 142 buses) with an error of 0.083, 0.126, 0.078 pu, the same picture.
- Whether the collapsed labels hurt GENCO (the question that motivated the dir) is not answered here: that needs a training run on filtered data.


Column notes. `coll. scen. (roots)` = collapsed scenarios in the scored test split (graphkit holds out about 1% of scenarios, so 659 not 665 on case4837_idx_35) and how many have a physical Newton root (`analyze.physical_root`, unchanged). `hidden collapsed buses` = mean over the 3 PRE seeds of collapsed PQ buses (label Vm < 0.5, scenario min Vm < 0.5) that the pretraining mask hid, out of all such buses; visible entries pass through unchanged and are not scored. PRE = pretraining mask + OPF decoder, 3 mask-draw seeds (1000, 1001, 1002); PF = power-flow mask + PF decoder (the task of the old report), one draw (it is deterministic). `PRE: MAE vs root (label vs root)` uses only hidden collapsed buses of scenarios with a root (3 to 113 buses per run; none on case10480_idx_26, case1888_rte, case1951_rte), and the number in brackets is the label's own distance to the root on those buses, so a model that copies the label has the two nearly equal. `normal hidden PQ MAE`: Vm error at hidden PQ buses of non-collapsed scenarios of the same grid (the control); `-` where every scenario of the grid is collapsed.

The PF columns reproduce the old report's table (case4837_idx_35 median 1.093; the old report's median is of per-scenario minima over all collapsed buses, ours is over buses, so small differences are expected on the multi-bus grids).

## Extra checks

- Visible-collapse leak: for every run I counted hidden collapsed buses whose scenario has no VISIBLE collapsed PQ bus (`pre_n_no_visible_low` in `summary.json`). Nearly all hidden collapsed buses are of that kind, and the fraction below 0.5 among them is the same as overall (case4837_idx_35 0.97/0.98/0.98 on 512/519/529 buses), so the prediction is not a copy of a visible collapsed neighbour.
- Collapsed scenarios, other hidden PQ buses: Vm error 0.0012 to 0.0042 pu under PRE on the nine subgrids (`pre_coll_other_mae`), against 0.0017 to 0.035 under PF.
- Per-seed spread of the fraction below 0.5 is at most 0.25 on any subgrid (case4837_idx_37, 0.78 to 1.00 on 1 to 5 buses); pooled spread 0.960 to 0.965.

## Results (generated by `summarize.py`, from `results.jsonl`)

### Pretraining subgrids

| grid | coll. scen. (roots) | hidden collapsed buses | PRE: frac pred < 0.5 (3 seeds) | PRE: median pred / median label | PRE: MAE vs label | PRE: MAE vs root (label vs root) | PF: frac < 0.5 | PF: median pred | PF: MAE vs label | normal hidden PQ MAE vs label: PRE / PF |
|---|---|---|---|---|---|---|---|---|---|---|
| case4837_goc_idx_35 | 659 (69) | 525 of 665 | 0.97/0.98/0.98 | -0.050 / -0.061 | 0.097 | 1.129 (1.128) | 0.00 | 1.093 | 1.145 | 0.0017 / 0.0029 |
| case20758_epigrids_idx_49 | 272 (140) | 232 of 293 | 0.97/0.98/0.98 | 0.015 / 0.022 | 0.095 | 0.979 (0.995) | 0.00 | 1.010 | 1.016 | 0.0018 / 0.0039 |
| case9591_goc_idx_12 | 70 (3) | 59 of 72 | 0.85/0.81/0.75 | 0.474 / 0.477 | 0.069 | 0.226 (0.896) | 0.00 | 1.020 | 0.597 | 0.0037 / 0.0061 |
| case5658_epigrids_idx_8 | 28 (28) | 44 of 55 | 0.97/0.96/0.98 | -0.000 / 0.001 | 0.031 | 0.980 (0.997) | 0.00 | 1.065 | 0.967 | 0.0008 / 0.0044 |
| case4837_goc_idx_28 | 16 (16) | 11 of 16 | 0.80/0.82/1.00 | -0.037 / -0.128 | 0.218 | 0.822 (0.969) | 0.00 | 0.902 | 0.957 | 0.0014 / 0.0041 |
| case19402_goc_idx_47 | 10 (10) | 8 of 11 | 0.89/0.75/0.89 | -0.047 / -0.043 | 0.408 | 0.959 (1.240) | 0.00 | 1.097 | 1.281 | 0.0015 / 0.0029 |
| case10480_goc_idx_40 | 9 (9) | 8 of 9 | 1.00/1.00/1.00 | 0.004 / -0.004 | 0.013 | 1.077 (1.084) | 0.00 | 1.077 | 1.066 | 0.0016 / 0.0059 |
| case4837_goc_idx_37 | 6 (6) | 15 of 19 | 0.93/0.78/0.92 | 0.228 / 0.073 | 0.176 | 0.841 (1.008) | 0.00 | 1.092 | 1.021 | 0.0018 / 0.0043 |
| case10480_goc_idx_26 | 3 (0) | 2 of 3 | 1.00/1.00/1.00 | 0.014 / 0.009 | 0.013 | - | 0.00 | 1.086 | 1.082 | 0.0013 / 0.0015 |

### Held-out ladder grids

| grid | coll. scen. (roots) | hidden collapsed buses | PRE: frac pred < 0.5 (3 seeds) | PRE: median pred / median label | PRE: MAE vs label | PRE: MAE vs root (label vs root) | PF: frac < 0.5 | PF: median pred | PF: MAE vs label | normal hidden PQ MAE vs label: PRE / PF |
|---|---|---|---|---|---|---|---|---|---|---|
| case1354_pegase | 69 (69) | 56 of 73 | 0.88/0.91/0.86 | 0.009 / -0.000 | 0.134 | 0.964 (1.096) | 0.00 | 1.098 | 1.095 | 0.0034 / 0.0081 |
| case2736sp_k | 8 (8) | 6 of 8 | 0.00/0.00/0.00 | 0.991 / -0.000 | 0.992 | 0.047 (1.039) | 0.00 | 1.027 | 1.030 | 0.0036 / 0.0140 |
| case240_pserc | 3 (1) | 1 of 3 | 0.50/0.50/0.00 | 0.583 / -0.000 | 0.583 | - | 0.00 | 1.075 | 1.062 | 0.0652 / 0.0413 |
| case1888_rte | 5 (0) | 29 of 35 | 0.21/0.14/0.14 | 0.850 / -0.030 | 0.751 | - | 0.00 | 1.064 | 1.063 | - / - |
| case1951_rte | 5 (0) | 45 of 56 | 0.52/0.36/0.43 | 0.761 / 0.016 | 0.478 | - | 0.00 | 1.072 | 0.999 | - / - |

### Pooled over the 9 pretraining subgrids (weights = hidden collapsed buses)

- PRE seed 1000: hidden collapsed buses 892; frac pred < 0.5 = 0.961; MAE vs label = 0.102
- PRE seed 1001: hidden collapsed buses 921; frac pred < 0.5 = 0.960; MAE vs label = 0.096
- PRE seed 1002: hidden collapsed buses 904; frac pred < 0.5 = 0.965; MAE vs label = 0.091
- PF task: hidden collapsed buses 1143; frac pred < 0.5 = 0.000; MAE vs label = 1.065
- Grids where every seed has frac < 0.5 >= 0.5 AND mean PRE MAE vs label < half the PF MAE: 9 of 9
- lowest per-seed frac < 0.5 by subgrid: case4837_goc_idx_35 0.97, case20758_epigrids_idx_49 0.97, case9591_goc_idx_12 0.75, case5658_epigrids_idx_8 0.96, case4837_goc_idx_28 0.80, case19402_goc_idx_47 0.75, case10480_goc_idx_40 1.00, case4837_goc_idx_37 0.78, case10480_goc_idx_26 1.00
