# Candidate 10: carved seen-family subgrids, a control for the seen/unseen gap in the GENCO |V| result (2026-10-21)

> **NOTE 2026-10-21 13:05 (audit genco-voltage-inference R1-F1 / R2-F3, approved by the owner 12:49).** The arms in this
> report come from different scenario generators.
> - Whole grids: ladder generator (load-only perturbation, PV |V| fixed).
> - Locked-15 and carved pieces: datakit `default_pf` (admittance, dispatch and outage perturbation).
> - The generator-controlled rerun is in `../2026-10-21-genco-voltage-inference-generator-control/`; the combined
>   statement is in `../2026-10-21-genco-voltage-inference-summary.md`.

> **UNAUDITED.** Produced by the lead chat; no `/audit` round has cleared it. Do not quote the numbers until one has.

## Bottom line

**PASS on the pre-registered primary test: on 19 freshly carved subgrids of the seen-family parents (GOC and EPIGRIDS), grids
GENCO never trained on, zero-shot GENCO beats the best of five learning-free interpolators for unmeasured-bus |V| at 40% measured
on 19 of 19 (needed 13, i.e. 2/3), every bootstrap CI below 1.** Candidate 9's 15/15 on GOC/EPIGRIDS is therefore not
memorisation of the specific training grids: the advantage appears on new grids drawn from the pretraining families. Secondary:
20% noise-free 16/19 (PASS), 40% with 0.5% noise 12/19 (FAIL, needed 13); descriptive: 10% 12/19, 20% + noise 14/19.

What this does and does not establish:
- It supports a **family/distribution-exposure** reading of candidate 9's seen/unseen gap (unseen families: 8/17 at 40%).
- It does **not** show that unseen-family failure is only about family. Size is only partly controlled: all carved subgrids are
  <= 1,000 buses (165 to 903), and the only unseen-family grids <= 1,000 buses are the six SimBench-HV grids (64 to 120 buses),
  where GENCO wins 4/6 at 40% (carved 19/19, Fisher p = 0.050), 1/6 at 20% (carved 16/19, p = 0.006) and 0/6 with noise at 40%
  (carved 12/19, p = 0.015). Above 1,000 buses only unseen grids exist (4/11 win at 40%), so the large-grid failures on RTE and PEGASE
  cannot be assigned to family versus size by this control. A partial counter-signal against a pure size story: the unseen Polish grids
  (2,383 to 3,375 buses) win 3/3 at 40%.
- **Our carving differs from GENCO's** (their procedure is not in the repo), and the carved subgrids were OPF-screened and are smaller
  than GENCO's hosted ones, so "in-distribution" is an assumption, though the result (wins everywhere) is what in-distribution would predict.

## What was done

Pre-registered in `protocol.json` (commit 24dc39d, before any GENCO-vs-interpolator evaluation on the carved set).

1. **Held-out subgrids were not obtainable** (the Hugging Face dataset hosts only the 31 training subgrids; reported earlier), so per the
   hub decision I carved new ones. datakit has no carving procedure (it reads a pre-made `subnets_casefiles/` directory), so `carve.py`
   uses a documented classical alternative: parent AC base point (reference_pf Newton, DC start, dispatch Pmin + alpha (Pmax-Pmin) covering
   load + losses), region = seeded BFS grown to a log-uniform 150-950 buses, tie branches removed with their base-point flow folded into the
   inside boundary bus as extra load (so the region's base point is an exact root), REF = largest in-service region generator, costs and limits copied.
2. Candidates were screened with PowerModels AC OPF (18/30 tier-a and 10/16 tier-b feasible). Selection: first two feasible per parent,
   plus two more for the three parents with many feasible candidates (case4837, 30000, 7336). Tier a = tie flow <= 1.0 x max(region load, generation);
   tier b = relaxed to 4.0 for parents with no tier-a candidate (case4619, 4917, 9591, 10480, 19402). Tier-b subgrids are tie-dominated.
3. Scenarios generated with datakit through the repo's `ladder_data` (default_pf.yaml with GENCO's evaluation-set edits, 120 requested), collapsed
   labels dropped by its validation, then **every label verified** (`verify.py`): power-balance residual < 1e-6 pu (all passed, worst 1.2e-8),
   Vm in [0.5, 1.5]. No scenario failed the residual check.
4. Leakage: bus-count and (r,x) fingerprint (Jaccard < 0.3) against the 31 hosted training subgrids (max seen 0.215, case30000 where line types repeat).
   The other 69 training subgrids are not available to check; GENCO's subgrids (idx 0 to 50 per parent) may overlap our regions physically.
5. Evaluation exactly as candidate 9 (same code): 10/20/40% x 3 seeds, 0.5% noise at 20/40%; GENCO zero-shot vs best of I1-I5; scenario-cluster bootstrap.

**Ran: 19 subgrids from 7 parents** (case4837 x4, case30000 x4, case7336 x4, case5658 x2, case20758 x2, case10480 x2, case19402 x1), 23 to 118 verified test
scenarios each (2,096 in all). **Not run (3):** case4619 c10 (datakit produced 0 valid scenarios), case4619 c11 (3), case19402 c11 (17), all tier b, below the
20-scenario minimum. No candidate survived for case4917 (all OPF-infeasible) or case9591 (rejected by the tie rule at both tiers).

## Per-subgrid table (generated by `verdict.py`; also `table.md`)

Cells: GENCO median / best-interpolator median (which) ratio [95% CI], W = GENCO wins. Medians in pu, 3 seeds pooled.

| parent | subgrid | tier | buses | n scen | 10% | 20% | 40% | 20% + noise | 40% + noise |
|---|---|---|---|---|---|---|---|---|---|
| case20758_epigrids | case20758_epigrids_c0 | a | 165 | 116 | 0.0123/0.0127 (I2) 0.97 [0.92,1.04] | 0.0061/0.0090 (I2) 0.68 [0.65,0.71] W | 0.0031/0.0065 (I2) 0.48 [0.46,0.49] W | 0.0077/0.0097 (I2) 0.79 [0.76,0.81] W | 0.0054/0.0076 (I5) 0.71 [0.68,0.72] W |
| case20758_epigrids | case20758_epigrids_c2 | a | 405 | 111 | 0.0059/0.0081 (I2) 0.72 [0.70,0.75] W | 0.0030/0.0058 (I4) 0.52 [0.50,0.54] W | 0.0015/0.0038 (I4) 0.40 [0.39,0.42] W | 0.0048/0.0067 (I5) 0.72 [0.70,0.74] W | 0.0041/0.0052 (I5) 0.78 [0.77,0.80] W |
| case30000_goc | case30000_goc_c0 | a | 247 | 118 | 0.0131/0.0038 (I2) 3.47 [3.30,3.60] | 0.0046/0.0028 (I2) 1.65 [1.59,1.71] | 0.0017/0.0019 (I2) 0.86 [0.82,0.89] W | 0.0068/0.0048 (I1) 1.42 [1.38,1.48] | 0.0050/0.0045 (I1) 1.12 [1.10,1.15] |
| case30000_goc | case30000_goc_c1 | a | 299 | 118 | 0.0057/0.0043 (I2) 1.33 [1.25,1.40] | 0.0028/0.0027 (I2) 1.04 [1.02,1.07] | 0.0014/0.0019 (I2) 0.77 [0.74,0.80] W | 0.0050/0.0045 (I2) 1.10 [1.08,1.13] | 0.0046/0.0041 (I5) 1.13 [1.11,1.14] |
| case4837_goc | case4837_goc_c0 | a | 279 | 116 | 0.0057/0.0051 (I2) 1.12 [1.07,1.17] | 0.0023/0.0036 (I2) 0.66 [0.63,0.70] W | 0.0010/0.0023 (I4) 0.45 [0.43,0.47] W | 0.0047/0.0049 (I2) 0.96 [0.94,0.99] W | 0.0040/0.0041 (I2) 0.98 [0.96,0.99] W |
| case4837_goc | case4837_goc_c1 | a | 235 | 118 | 0.0035/0.0038 (I2) 0.93 [0.89,0.99] W | 0.0017/0.0027 (I2) 0.62 [0.61,0.65] W | 0.0008/0.0018 (I5) 0.44 [0.42,0.46] W | 0.0040/0.0043 (I2) 0.94 [0.92,0.97] W | 0.0038/0.0038 (I2) 1.01 [1.00,1.04] |
| case5658_epigrids | case5658_epigrids_c0 | a | 360 | 113 | 0.0051/0.0081 (I2) 0.63 [0.61,0.66] W | 0.0026/0.0052 (I2) 0.50 [0.48,0.52] W | 0.0011/0.0031 (I5) 0.35 [0.34,0.37] W | 0.0046/0.0062 (I2) 0.74 [0.72,0.76] W | 0.0039/0.0045 (I5) 0.86 [0.84,0.88] W |
| case5658_epigrids | case5658_epigrids_c3 | a | 494 | 115 | 0.0032/0.0049 (I1) 0.64 [0.60,0.68] W | 0.0017/0.0034 (I4) 0.50 [0.48,0.52] W | 0.0011/0.0021 (I4) 0.51 [0.50,0.53] W | 0.0038/0.0046 (I1) 0.84 [0.82,0.86] W | 0.0039/0.0038 (I4) 1.02 [0.99,1.04] |
| case7336_epigrids | case7336_epigrids_c0 | a | 632 | 118 | 0.0045/0.0085 (I4) 0.54 [0.52,0.56] W | 0.0025/0.0056 (I4) 0.45 [0.43,0.46] W | 0.0012/0.0033 (I4) 0.37 [0.36,0.39] W | 0.0043/0.0064 (I4) 0.67 [0.66,0.68] W | 0.0039/0.0048 (I4) 0.81 [0.80,0.82] W |
| case7336_epigrids | case7336_epigrids_c1 | a | 501 | 90 | 0.0042/0.0070 (I1) 0.60 [0.58,0.63] W | 0.0024/0.0045 (I4) 0.53 [0.50,0.56] W | 0.0012/0.0025 (I4) 0.49 [0.46,0.50] W | 0.0043/0.0054 (I4) 0.80 [0.78,0.81] W | 0.0040/0.0042 (I4) 0.96 [0.94,0.98] W |
| case19402_goc | case19402_goc_c10 | b | 259 | 23 | 0.0025/0.0027 (I5) 0.91 [0.76,1.05] | 0.0012/0.0024 (I2) 0.50 [0.44,0.58] W | 0.0007/0.0019 (I2) 0.37 [0.35,0.40] W | 0.0030/0.0032 (I5) 0.95 [0.90,1.01] | 0.0031/0.0031 (I5) 1.00 [0.96,1.06] |
| case10480_goc | case10480_goc_c12 | b | 185 | 69 | 0.0090/0.0139 (I2) 0.65 [0.62,0.68] W | 0.0052/0.0097 (I2) 0.54 [0.51,0.57] W | 0.0032/0.0070 (I2) 0.46 [0.44,0.49] W | 0.0066/0.0104 (I2) 0.63 [0.61,0.65] W | 0.0052/0.0078 (I2) 0.67 [0.65,0.69] W |
| case10480_goc | case10480_goc_c13 | b | 268 | 38 | 0.0015/0.0040 (I1) 0.38 [0.34,0.40] W | 0.0008/0.0029 (I4) 0.28 [0.27,0.30] W | 0.0005/0.0019 (I4) 0.25 [0.23,0.26] W | 0.0031/0.0040 (I1) 0.77 [0.73,0.81] W | 0.0035/0.0036 (I1) 0.95 [0.92,0.99] W |
| case4837_goc | case4837_goc_c2 | a | 903 | 105 | 0.0051/0.0059 (I2) 0.86 [0.83,0.88] W | 0.0026/0.0040 (I4) 0.65 [0.63,0.66] W | 0.0012/0.0025 (I4) 0.50 [0.49,0.51] W | 0.0046/0.0052 (I5) 0.87 [0.86,0.89] W | 0.0040/0.0042 (I5) 0.96 [0.95,0.97] W |
| case4837_goc | case4837_goc_c3 | a | 493 | 118 | 0.0055/0.0072 (I2) 0.76 [0.74,0.78] W | 0.0028/0.0048 (I2) 0.59 [0.57,0.60] W | 0.0014/0.0031 (I4) 0.47 [0.45,0.48] W | 0.0049/0.0060 (I2) 0.81 [0.79,0.82] W | 0.0042/0.0047 (I5) 0.90 [0.89,0.92] W |
| case30000_goc | case30000_goc_c2 | a | 453 | 118 | 0.0083/0.0030 (I2) 2.79 [2.66,2.94] | 0.0027/0.0021 (I2) 1.29 [1.24,1.34] | 0.0010/0.0015 (I2) 0.70 [0.68,0.72] W | 0.0050/0.0040 (I1) 1.24 [1.22,1.27] | 0.0043/0.0038 (I1) 1.11 [1.09,1.13] |
| case30000_goc | case30000_goc_c3 | a | 792 | 118 | 0.0069/0.0040 (I2) 1.72 [1.67,1.79] | 0.0029/0.0032 (I2) 0.90 [0.87,0.92] W | 0.0017/0.0026 (I4) 0.67 [0.66,0.69] W | 0.0051/0.0048 (I2) 1.07 [1.06,1.09] | 0.0051/0.0046 (I1) 1.12 [1.10,1.14] |
| case7336_epigrids | case7336_epigrids_c2 | a | 610 | 114 | 0.0073/0.0081 (I2) 0.90 [0.87,0.92] W | 0.0039/0.0055 (I2) 0.70 [0.69,0.72] W | 0.0020/0.0034 (I4) 0.60 [0.58,0.61] W | 0.0055/0.0065 (I2) 0.85 [0.84,0.87] W | 0.0044/0.0048 (I5) 0.92 [0.90,0.94] W |
| case7336_epigrids | case7336_epigrids_c4 | a | 673 | 117 | 0.0055/0.0094 (I5) 0.59 [0.57,0.61] W | 0.0030/0.0069 (I4) 0.43 [0.42,0.44] W | 0.0015/0.0043 (I4) 0.34 [0.33,0.35] W | 0.0048/0.0075 (I5) 0.64 [0.62,0.66] W | 0.0042/0.0056 (I4) 0.76 [0.74,0.77] W |

Size-banded win counts (GENCO beats best interpolator, CI rule): carved (all <= 1000 buses) vs unseen families

| condition | band | unseen wins/grids | carved wins/grids | Fisher p |
|---|---|---|---|---|
| f40 | <=1000 | 4/6 | 19/19 | 0.050 |
| f40 | >1000 | 4/11 | 0/0 | nan |
| f20 | <=1000 | 1/6 | 16/19 | 0.006 |
| f20 | >1000 | 2/11 | 0/0 | nan |
| n40 | <=1000 | 0/6 | 12/19 | 0.015 |
| n40 | >1000 | 2/11 | 0/0 | nan |

## Observations and surprises

- **Everything wins at 40%** (19/19, ratios 0.25 to 0.86), by wider margins than on the pglib or unseen families (median ratio 0.47 vs 0.87 for the SimBench band). At 20% the three
  losses are all case30000 subgrids (c0 1.65, c1 1.04, c2 1.29); case30000 also loses at 10% (ratios 1.3 to 3.5). I did not investigate why that parent is harder.
- **Noise is the weak point again**: at 40% with 0.5% noise only 12/19 win (tier a 10/16, tier b 2/3); GENCO passes noisy measurements through unchanged while the interpolators
  smooth them. The same effect appeared in candidate 9. A modest sigma erases much of the advantage even on in-distribution subgrids.
- **Absolute errors are small** (GENCO 0.0005 to 0.013 pu), smaller than on the unseen real grids (0.002 to 0.015), probably because OPF-generated voltage profiles on these subgrids
  vary little; ratios, not absolute errors, are the comparable quantity.
- Per parent at 40%: all parents 100% (4837 4/4, 30000 4/4, 7336 4/4, 5658 2/2, 20758 2/2, 10480 2/2, 19402 1/1). Tier a 16/16, tier b 3/3 at 40%.
- The best classical method on carved subgrids is mostly I2 or I4; I4 (load-aware) is best on the epigrids parents at 40%.

## Caveats

- Our carving differs from GENCO's; boundary injections are folded into loads (and scaled by datakit's load perturbation); subgrids are OPF-screened (easier) and smaller
  (median ~300 buses vs 555 for the hosted ones; largest 903); tier-b subgrids are tie-dominated and produce few valid scenarios (23 to 69 test scenarios).
- Parent-level exposure is by construction: GENCO trained on other subgrids of these parents, and some of those may overlap our regions (69 training names unchecked).
  This control therefore tests "new grids from the pretraining families", not "grids sharing no content with training".
- The unseen-family comparison reuses candidate 9's results (not re-run). Seven parents, 19 subgrids from a few BFS draws: not independent draws of the family.
- All results unaudited; the verification is the residual check plus ladder_data's collapsed-label rule (the repo's two-start oracle was not run on every scenario).

## Proposed claim wording (pre-registered for PASS, applied)

Freshly carved subgrids of the seen-family parents, which GENCO never trained on, behave like the seen family, not like the unseen families: GENCO beats the best classical
interpolator on 19 of 19 at 40% (unseen families: 8 of 17). Candidate 9's seen/unseen gap is therefore not grid memorisation but an effect of exposure to the pretraining
families, with the caveats that our carving may differ from GENCO's, that size is only partly controlled (all carved subgrids <= 1,000 buses; above 1,000 buses only unseen grids exist), and that the result is unaudited.

## What was not done

- The 20 true held-out validation/test subgrids (not hosted), and a reference group of the 31 hosted training subgrids (T): not run.
- No unseen-family grids in the 150 to 1,000 bus band beyond SimBench-HV; no carving of unseen-family parents (a carve of RTE/PEGASE/Polish would separate family from size, and would be the next control).
- The oracle (two-start Newton) on every scenario; no audit.

## Files

`protocol.json` (+ `.sha256`), `carve.py`, `screen.jl`, `gen.py`, `verify.py`, `make_ids.py`, `ocommon.py`, `run_interp.py`, `run_genco_o.py`, `verdict.py`, `grids.py`, `write_protocol.py`,
`selected.json`, `manifest_a/b.json`, `verified.json`, `test_ids.json`, `per_grid/*.jsonl`, `verdict.json`, `table.md`, logs. Carved cases and generated data: `data/external/genco-carved-subgrids/` (gitignored).
