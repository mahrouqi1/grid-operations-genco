# Audit: repaired roots for the 1,083 scored collapsed scenarios

Auditor: fresh pass, read-only apart from this file and two check scripts in this folder
(`audit_repair_roots.py` -> `audit_repair_roots.json`, `_resid.csv`, `_dist.csv`; `audit_repair_roots_b.py`). Written from scratch;
`q_a_roots.py` was read, not reused.

## Verdict: CONFIRMED WITH CAVEATS

Every number reproduces. The caveats concern how strong the words "verified" and "closer to the root" are, not the counts.

## 1. Same scenario set?
- Scored set (`2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv`, 1,083 rows) mapped to original ids through each
  `scenario_map.csv`: 1,083 of 1,083 are in the repair `status.csv` (1,085 rows); the 2 not scored are case4837_idx_35 #198 and case9591_idx_12 #14.
  Of the 1,083: 1,070 `recovered`, 13 `recovered_signflip`.
- The label in the eval data equals the original label (max |diff| 6e-8). Same predictions file set (`collapsed-label/mlruns`, one per subgrid).
- NOT the same as the mask-task set: `2026-10-09-collapsed-label-pretrain-mask/results.jsonl` has only per-grid aggregates (no scenario ids) and
  its pretrain test split holds 1,073 collapsed scenarios (659+272+70+28+16+10+9+6+3), not 1,083. The 1,083 is the PF-task set (10-08). Do not
  call it the mask set.

## 2. Acceptance rule (`relabel.accept_relabel` / `judge_candidate`, constants in `relabel.py:49-51`)
- Residual < 1e-9 pu (RES_TOL), every |V| >= 0.5 (VM_MIN), max branch angle <= 90 deg, two starts from different initial points
  (`start_group`; plain and damped from one point count as one) agreeing to max |dV| < 1e-6 (AGREE_TOL).
- My independent recomputation, using the ORIGINAL Pd/Qd/generator setpoints/Ybus and the corrected voltages, over all 1,083:
  max residual 8.5e-10 pu (0 above 1e-9); min |V| 0.509 (0 below 0.5, 0 negative); max |V| 1.175; PV/REF magnitudes equal the original
  setpoints (dev 2e-15); Pd/Qd unchanged. A control with the same formula on 45 healthy original scenarios gives <= 9.9e-9, so the formula is sound.
- Two-start agreement: `max_dv_between_starts` max 1.2e-10 across the 1,070 (none >= 1e-6, none missing). Max branch angle 89.85 deg.
- Start pairs of the 1,070: 84 are two neighbour-root starts, 393 have one neighbour start, 132 name a homotopy start, 466 are
  flat/DC-type only; phase 1 accepted 427, phase 2 (neighbour/homotopy) 643. So 643 of the 1,070 came from the neighbour/homotopy phase, where agreement is not independent of the subgrid topology (the repair report says so itself, report.md:30).
- "13 by the PF identity" (`repair.py:128-153`): the label has |Vm| >= 0.5 everywhere and only some Vm negative; the label's own complex voltage
  (negative magnitude = same point 180 deg rotated) is tested with the same `judge_candidate` (residual < 1e-9, |V| >= 0.5, angle <= 90). It is a real
  root check (my residual recomputation gives <= 8.5e-10 for them) but NOT a second independent start; a flat-start Newton agreed with only
  3 of the 13. In effect: "the stored label is already a valid solution, with a sign defect".

## 3. Closer to the root than to the label
Measured at the flagged buses (label Vm < 0.5 or negative; 1,154 buses, all PQ buses, none PV/REF), prediction = `Vm_pred` of the
released checkpoint's PF-task output (`run_genco_inference.py`: HF `gridfm/genco-pf-transfer` `pretraining/best_model_state_dict.pt`, PowerFlow task):

| distance | closer to root (of 1,083) | median to root | p90 | max | median to label |
|---|---|---|---|---|---|
| mean |Vm diff| over flagged buses | 1,083 | 0.0083 | 0.0169 | 0.558 | 1.135 |
| worst flagged bus | 1,083 | 0.0083 | 0.0178 | 0.558 | 1.135 |
| every flagged bus individually | 1,154 of 1,154 buses | | | | |
| all buses, mean (not the claim) | 1,065 | 0.0027 | | | label 0.0044 |

So 1,083/1,083 and median 0.008 reproduce (0.0083). It holds per bus as well as per scenario. Over ALL buses the root is not always closer
(18 scenarios are not closer; in the printed ones the two distances are 0.004 to 0.006 pu) because off-flag buses are identical in label and root; the claim must stay "at the flagged buses".
Magnitude only (angles not compared).

Caveats on the strength of this result:
- The label side is a very low bar (label is 0 or negative; the model never predicts below 0.87 anywhere). "Closer than the label" is nearly guaranteed.
  The informative number is the distance to the root.
- Six scenarios are farther than 0.2 pu from their root (0.20 to 0.56): 4837_idx_35 #402 and #1183, 9591_idx_12 #316 and #1076, 20758_idx_49 #62 and #100.
  All have a stressed low-voltage root (min Vm 0.51 to 0.79, branch angle 69 to 90 deg); 3 are sign-flip rows (label -0.514 vs root +0.514).
  The 40 scored scenarios with min Vm < 0.70 have median distance 0.047 (all still closer to root than label).
- Roots are not unique and the rule does not choose among valid ones. Example: in case9591_idx_12 the OLD Newton search found 0.54/0.63 pu roots for 2 scenarios
  (orig ids 1741, 5029). The repair accepted near-1 roots for the same scenarios (median at the flagged buses 1.03 and 0.98, via neighbour/homotopy
  starts). Both pass the rule. 8 more scenarios have old Newton roots differing from the repaired root by 0.03 to 0.05 pu. The statement is "GENCO is close
  to a valid root we found", not "to the" root.

## 4. case9591_idx_12
71 scored scenarios. Median root magnitude at the flagged buses >= 0.95 for 68 of 71 (the other three: 0.514, 0.847, 0.911). GENCO distance to root:
median 0.0087 over all 71 (0.0081 over the 68). Confirms "68 of 71" and "0.0087". Distance to the label median 0.549. The post's "0.54 to 0.63 roots"
are real roots of 2 scenarios (plus one more low root) but not the only ones; the repaired near-1 roots exist for the same scenarios.

## 5. Which predictions
Same ones as the post's PF-task statements: `data/external/genco-ladder/collapsed-label/mlruns/*/*/artifacts/test/<subgrid>_predictions.parquet`,
produced by `2026-10-08-collapsed-label-genco/run_genco_inference.py` with the released pretraining checkpoint and graphkit `evaluate`, PowerFlow task
(not the pretraining mask). Predictions at the flagged PQ buses were hidden inputs (PF mask), so this is a true prediction. The normalizer was fit on the
sampled data (report says so), not on GENCO's own. The post's "283" came from `analyze.physical_root` (flat then DC start only); the repair work is a wider search, so the two figures can be reconciled but the 1,083 comes from a different, later script run (unaudited by the repair author; this audit covers it).
Not audited: whether the post's other PF numbers (fig/tables) were produced from this same mlruns folder; only the script and checkpoint path match.

## Safe sentence for the post
"A wider search (flat and DC starts, then starts taken from the roots of neighbouring scenarios of the same subgrid, and gradual morphing from healthy
scenarios) found a power-flow solution with every voltage at least 0.5 pu for all 1,083 scored collapsed scenarios: for 1,070 two different starts reached
the same solution to 1e-6, and for the other 13 the stored label already satisfied the power-flow equations once its sign defect was removed. At the collapsed
buses GENCO's output is nearer to this solution than to the stored label in all 1,083 scenarios, with a median gap of 0.008 pu (6 scenarios are over 0.2 pu away;
those have strained low-voltage solutions). A scenario can have more than one such solution, so this shows GENCO lands near a physical solution, not that it
finds a unique one."

Also disclose (one clause): 643 of the 1,070 came from neighbour or homotopy starts, and 40 solutions have a voltage below 0.70 pu.
