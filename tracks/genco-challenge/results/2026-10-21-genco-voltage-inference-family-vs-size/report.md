# Candidate 11: family vs size, carved pieces of UNSEEN families vs carved pieces of SEEN families (2026-10-21)

> **CORRECTION 2026-10-21 12:35 (audit genco-voltage-inference R1-F1/F2, approved by the owner 11:40).** The reading
> "size (or whole-grid structure) explains most of candidate 9's unseen-family failure" is superseded.
> - Under ONE generator (ladder), seen-family whole grids of 500-9,591 buses win 15/15 at 40%, while unseen-family
>   whole grids win 8/17 (R1-F2).
> - The generator-controlled rerun (`../2026-10-21-genco-voltage-inference-generator-control/`) puts all 36 pieces
>   through the ladder generator. At piece size the family contrast is then significant: seen 19/19 vs unseen 11/17,
>   two-sided p = 0.006. Unseen pieces vs unseen whole grids at 40% is 11/17 vs 8/17 (p = 0.245, not shown).
> - Neither size nor family alone accounts for the gap, and family clearly matters. See
>   `../2026-10-21-genco-voltage-inference-summary.md` for the combined, audited-wording statement.

> **UNAUDITED.** Produced by the lead chat; no `/audit` round has cleared it. Do not quote the numbers until one has.

## Bottom line

**The pre-registered primary test is NOT SHOWN: carved 150-950-bus pieces of the UNSEEN families (RTE, PEGASE, Polish) are won by
GENCO on 14 of 17 at 40% (CI rule), against 19 of 19 for the carved seen-family pieces; one-sided Fisher p = 0.095 (needed < 0.05).**
At 20% it is 12/17 vs 16/19 (p = 0.28); size-matched (16 pairs) 13/16 vs 16/16 at 40% (p = 0.11).

Reading: **size (or whole-grid structure) explains most of candidate 9's unseen-family failure, with a residual family effect that is
suggestive but not significant on win counts.** The same unseen families whose whole grids were won 8/17 at 40% (candidate 9) are
won 14/17 once cut into seen-sized pieces. But the unseen pieces are not as good as the seen pieces: the median GENCO/best-interpolator
error ratio is 0.78 on unseen pieces vs 0.47 on seen pieces at 40% (0.88 vs 0.59 at 20%), and at 10% (descriptive, not in any
test) GENCO wins 5/17 on unseen vs 12/19 on seen pieces (one-sided p = 0.045). Per-family win counts are RTE 4/5, PEGASE 4/6, Polish 6/6.

Pre-registered post sentence that applies (NOT SHOWN): *Carved pieces of the unseen families win at 40% about as often as the
seen-family pieces (14 of 17 vs 19 of 19, one-sided Fisher p = 0.095). So the unseen-family failures of candidate 9 are better explained by SIZE
(or whole-grid structure) than by family: pieces of 150-950 buses of unseen families are mostly handled as well as seen ones.* It is
a failure to show a difference, not evidence of equivalence: with 17 pieces from about 6 parents the test has little power, and the
ratio gap and the 10% result point toward a smaller residual family effect.

## What was done

Pre-registered in `protocol.json` (commit 7191838, before any GENCO-vs-interpolator evaluation on these pieces). Same procedure as
candidate 10 (`carve.py`: seeded BFS, tie-flow folding, REF = largest generator, AC-OPF screen, datakit generation through `ladder_data`,
every label verified to residual < 1e-6 pu; worst 1.4e-8), same evaluation code (`ocommon.py`, 10/20/40% x 3 seeds, 0.5% noise), seen-family
results reused from `../2026-10-21-genco-voltage-inference-control/verdict.json` without rerunning. PGLib versions of the parents were used.

**Deviations forced by the unseen parents, pre-registered:**
- **RTE is represented by case2848_rte only.** The base point (an AC power-flow solution of the parent) that the carving needs could not be obtained
  for case1888_rte, case6470_rte, case6495_rte, case6515_rte by any fallback tried (plain Newton, damped Newton, load scaled 0.9 to 0.5, the case file's own dispatch; pandapower
  finds no reference bus). So the RTE result rests on one parent (5 pieces).
- **Polish is effectively one parent (case2383wp_k, 5 pieces) plus one piece of case3375wp_k (tier b)**; case3012wp_k and almost all case3375wp_k candidates were OPF-infeasible even after extra candidate batches.
- Fallbacks added to the parent solve (damped Newton, load scale, shipped dispatch; type-1-bus generators' Q) were not needed by any parent that solved (all parent load scales 1.0).
- Extra candidate batches (offsets 20, 40, 50) were drawn for parents with few feasible candidates; selection rule: per parent cap 5, per family cap 7, round-robin by candidate index; two selected case8387_pegase pieces
  (c0, c1) produced < 20 valid datakit scenarios and were replaced by the next feasible ones (c3, c4), of which c4 also failed.

**Ran: 17 pieces** (RTE 5, PEGASE 6, Polish 6), 30 to 118 verified test scenarios each. Not run: case8387_pegase c0, c1, c4 (datakit gave 0, 3 and 10 valid scenarios).

**Size distributions** (buses): unseen pieces min 159, median 327, max 735; seen pieces min 165, median 405, max 903; KS p = 0.80, median ratio 0.81. By the pre-registered rule
(KS p < 0.10 or medians differ > 25%) they do **not** differ; the size-matched subset (greedy nearest neighbour on log bus count, caliper 0.25, 16 pairs) is reported anyway.

## Per-piece table (generated by `verdict.py`; also `table.md`)

Cells: GENCO median / best-interpolator median (which) ratio [95% CI], W = GENCO wins. Medians in pu, 3 seeds pooled.

| family | subgrid | tier | buses | n scen | 10% | 20% | 40% | 20% + noise | 40% + noise |
|---|---|---|---|---|---|---|---|---|---|
| RTE | case2848_rte_c1 | a | 242 | 118 | 0.0165/0.0179 (I2) 0.92 [0.90,0.93] W | 0.0135/0.0163 (I2) 0.83 [0.82,0.84] W | 0.0113/0.0149 (I2) 0.75 [0.74,0.78] W | 0.0152/0.0174 (I2) 0.87 [0.86,0.88] W | 0.0141/0.0164 (I2) 0.85 [0.84,0.87] W |
| RTE | case2848_rte_c4 | a | 554 | 117 | 0.0170/0.0133 (I2) 1.28 [1.26,1.30] | 0.0129/0.0119 (I2) 1.08 [1.07,1.09] | 0.0106/0.0109 (I3) 0.97 [0.96,0.98] W | 0.0142/0.0133 (I2) 1.07 [1.06,1.08] | 0.0126/0.0125 (I2) 1.00 [0.99,1.02] |
| RTE | case2848_rte_c20 | a | 167 | 118 | 0.0166/0.0148 (I2) 1.12 [1.09,1.16] | 0.0127/0.0137 (I2) 0.93 [0.91,0.95] W | 0.0106/0.0123 (I2) 0.86 [0.83,0.87] W | 0.0142/0.0147 (I2) 0.96 [0.95,0.99] W | 0.0130/0.0138 (I2) 0.95 [0.92,0.96] W |
| RTE | case2848_rte_c21 | a | 735 | 117 | 0.0211/0.0148 (I2) 1.42 [1.40,1.43] | 0.0164/0.0137 (I2) 1.20 [1.19,1.21] | 0.0143/0.0128 (I2) 1.12 [1.10,1.14] | 0.0176/0.0149 (I2) 1.18 [1.17,1.19] | 0.0165/0.0145 (I2) 1.14 [1.12,1.16] |
| RTE | case2848_rte_c22 | a | 323 | 118 | 0.0151/0.0160 (I2) 0.94 [0.93,0.96] W | 0.0118/0.0142 (I2) 0.83 [0.82,0.85] W | 0.0097/0.0129 (I2) 0.75 [0.74,0.77] W | 0.0136/0.0153 (I2) 0.89 [0.87,0.90] W | 0.0123/0.0145 (I5) 0.85 [0.84,0.86] W |
| PEGASE | case2869_pegase_c1 | a | 513 | 117 | 0.0138/0.0129 (I2) 1.07 [1.05,1.09] | 0.0090/0.0093 (I2) 0.97 [0.95,0.99] W | 0.0053/0.0060 (I2) 0.88 [0.86,0.91] W | 0.0102/0.0101 (I5) 1.02 [1.00,1.04] | 0.0074/0.0070 (I2) 1.06 [1.03,1.08] |
| PEGASE | case9241_pegase_c1 | a | 439 | 109 | 0.0232/0.0248 (I2) 0.93 [0.91,0.95] W | 0.0156/0.0179 (I2) 0.88 [0.86,0.90] W | 0.0095/0.0116 (I2) 0.82 [0.79,0.84] W | 0.0166/0.0184 (I2) 0.90 [0.89,0.92] W | 0.0112/0.0126 (I2) 0.89 [0.87,0.91] W |
| PEGASE | case2869_pegase_c2 | a | 327 | 118 | 0.0113/0.0084 (I2) 1.34 [1.30,1.38] | 0.0062/0.0061 (I2) 1.02 [1.00,1.06] | 0.0035/0.0041 (I2) 0.86 [0.83,0.88] W | 0.0079/0.0073 (I2) 1.08 [1.05,1.10] | 0.0064/0.0058 (I2) 1.09 [1.06,1.11] |
| PEGASE | case9241_pegase_c2 | a | 488 | 114 | 0.0228/0.0111 (I1) 2.04 [2.00,2.08] | 0.0142/0.0091 (I2) 1.56 [1.54,1.59] | 0.0091/0.0070 (I2) 1.30 [1.29,1.33] | 0.0152/0.0098 (I1) 1.55 [1.53,1.58] | 0.0109/0.0080 (I2) 1.37 [1.35,1.39] |
| PEGASE | case8387_pegase_c2 | a | 327 | 30 | 0.0165/0.0242 (I1) 0.68 [0.63,0.73] W | 0.0099/0.0183 (I4) 0.54 [0.52,0.60] W | 0.0051/0.0117 (I4) 0.44 [0.42,0.47] W | 0.0111/0.0187 (I4) 0.60 [0.56,0.64] W | 0.0070/0.0125 (I4) 0.56 [0.54,0.60] W |
| Polish | case2383wp_k_c2 | a | 353 | 117 | 0.0155/0.0162 (I2) 0.96 [0.94,0.98] W | 0.0095/0.0128 (I2) 0.74 [0.72,0.76] W | 0.0052/0.0090 (I2) 0.58 [0.56,0.59] W | 0.0109/0.0136 (I2) 0.80 [0.79,0.82] W | 0.0075/0.0102 (I2) 0.73 [0.71,0.74] W |
| Polish | case3375wp_k_c53 | b | 165 | 64 | 0.0242/0.0242 (I2) 1.00 [0.97,1.05] | 0.0148/0.0189 (I2) 0.78 [0.76,0.82] W | 0.0093/0.0130 (I2) 0.72 [0.70,0.74] W | 0.0161/0.0199 (I2) 0.81 [0.79,0.84] W | 0.0117/0.0145 (I2) 0.81 [0.78,0.83] W |
| Polish | case2383wp_k_c4 | a | 202 | 118 | 0.0176/0.0161 (I2) 1.09 [1.07,1.12] | 0.0110/0.0129 (I2) 0.85 [0.84,0.88] W | 0.0066/0.0097 (I5) 0.68 [0.66,0.70] W | 0.0124/0.0138 (I2) 0.90 [0.88,0.92] W | 0.0088/0.0111 (I2) 0.79 [0.77,0.81] W |
| Polish | case2383wp_k_c40 | a | 241 | 118 | 0.0214/0.0197 (I2) 1.08 [1.05,1.11] | 0.0131/0.0154 (I2) 0.85 [0.83,0.87] W | 0.0071/0.0103 (I3) 0.69 [0.66,0.71] W | 0.0144/0.0163 (I2) 0.88 [0.86,0.90] W | 0.0092/0.0120 (I2) 0.77 [0.75,0.79] W |
| Polish | case2383wp_k_c41 | a | 660 | 117 | 0.0166/0.0168 (I2) 0.98 [0.97,1.00] | 0.0101/0.0132 (I2) 0.77 [0.76,0.79] W | 0.0055/0.0088 (I5) 0.63 [0.62,0.63] W | 0.0116/0.0141 (I2) 0.83 [0.81,0.84] W | 0.0079/0.0102 (I2) 0.78 [0.77,0.79] W |
| Polish | case2383wp_k_c42 | a | 159 | 117 | 0.0172/0.0143 (I2) 1.20 [1.16,1.24] | 0.0100/0.0107 (I2) 0.93 [0.91,0.95] W | 0.0051/0.0066 (I2) 0.78 [0.76,0.82] W | 0.0115/0.0117 (I2) 0.98 [0.96,1.01] | 0.0076/0.0082 (I2) 0.92 [0.90,0.95] W |
| PEGASE | case8387_pegase_c3 | a | 405 | 95 | 0.0214/0.0092 (I2) 2.32 [2.20,2.40] | 0.0124/0.0069 (I2) 1.79 [1.71,1.86] | 0.0073/0.0048 (I2) 1.51 [1.46,1.56] | 0.0137/0.0079 (I2) 1.73 [1.67,1.82] | 0.0099/0.0062 (I5) 1.58 [1.55,1.62] |

Unseen vs seen carved subgrids (seen results reused from the control run):

| comparison | unseen wins | seen wins | one-sided Fisher p (unseen lower) | median ratio unseen / seen |
|---|---|---|---|---|
| 40% (primary) | 14/17 | 19/19 | 0.0952 | 0.78 / 0.47 |
| 20% | 12/17 | 16/19 | 0.2811 | 0.88 / 0.59 |
| 40% + noise | 11/17 | 12/19 | 0.6704 | 0.89 / 0.96 |
| 10% | 5/17 | 12/19 | 0.0448 | 1.08 / 0.86 |
| PEGASE 40% | 4/6 | 19/19 | 0.0500 | 0.87 / 0.47 |
| PEGASE 20% | 3/6 | 16/19 | 0.1246 | 1.00 / 0.59 |
| Polish 40% | 6/6 | 19/19 | 1.0000 | 0.68 / 0.47 |
| Polish 20% | 6/6 | 16/19 | 1.0000 | 0.82 / 0.59 |
| RTE 40% | 4/5 | 19/19 | 0.2083 | 0.86 / 0.47 |
| RTE 20% | 3/5 | 16/19 | 0.2705 | 0.93 / 0.59 |
| size-matched (16 pairs) 40% | 13/16 | 16/16 | 0.1129 | 0.77 / 0.47 |
| size-matched (16 pairs) 20% | 11/16 | 13/16 | 0.3425 | 0.87 / 0.61 |

## Observations and surprises

- **Whole grids lose, pieces win.** Candidate 9: case2848rte, case2869pegase, case9241pegase and the other whole unseen grids lose to the best interpolator at 40% by 1.1x to 2.1x; the same parents cut into pieces win
  (e.g. case2848_rte pieces 4/5, case2869 pieces 2/2). The three losing pieces at 40% are case2848_rte c21 (735 buses, 1.12), case9241_pegase c2 (488, 1.30) and case8387_pegase c3 (405, 1.51);
  within the pieces there is no clear size trend (the two largest Polish and RTE pieces still win).
- **The seen advantage is larger, not different in kind**: seen median ratio 0.47, unseen 0.78 at 40%. The unseen pieces' losses at 10% (12/17 lose) and 20% (5/17 lose) show the margin is thinner.
- **Noise does not separate them**: with 0.5% noise at 40% unseen pieces win 11/17 vs 12/19 on seen (p = 0.67); the noise sensitivity found earlier is common to both.
- Caveat on the whole-grid vs piece contrast: pieces are folded-boundary equivalents with OPF screening, so they may be easier than the real grids' internal structure for any method, seen or not. Candidate 10 already shows seen pieces are very easy for GENCO (median ratio 0.47).

## Caveats

- Family coverage is thin: RTE = one parent, Polish ~ one parent, PEGASE three parents; 17 pieces are not independent draws of a family; power to detect a family effect is low (Fisher with 17 vs 19).
- Our carving differs from GENCO's; pieces are OPF-screened; pieces of the unseen parents come from the PGLib files, not the pandapower built-ins used for the whole-grid runs of candidate 9.
- The seen-family pieces were made with the earlier carve.py version (before the Q-injection correction for type-1 generators); that change did not affect which seen parents solved but was not re-verified against their carved cases.
- All results unaudited.

## What was not done

- RTE parents case1888, 6470, 6495, 6515 (base point unobtainable); Polish case3012 and most of case3375. A different base-point method (e.g. the repo's prepare_base_case with pandapower's built-in RTE networks and OPF dispatch) could recover them and would be the next step.
- No re-run of the seen-family pieces; no re-carving with a common code version; no audit.

## Files

`protocol.json` (+ `.sha256`), `carve.py` (with fallbacks), `screen.jl`, `gen.py`, `verify.py`, `make_ids.py`, `ocommon.py`, `run_interp.py`, `run_genco_o.py`, `verdict.py`, `grids.py`, `write_protocol.py`,
`selected.json`, `manifest_*.json`, `verified.json`, `test_ids.json`, `per_grid/*.jsonl`, `verdict.json`, `table.md`, logs. Carved cases and generated data: `data/external/genco-carved-unseen/` (gitignored).
