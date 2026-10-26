# Voltage-limit violation counts, locked 15 (2026-10-24)

**AUDITED R1 (the independent audit, F4 to F6 fixed in this revision).** Overnight option 3 (D1), on the owner's typed go 2026-10-24 14:05. Protocol `protocol.json` committed before any prediction or count (6b0179e). Scripts, `counts.csv`, `pooled.csv`, `repro_diff.json` in 4e86853. Report text written by the orchestrating chat from the run agent's account (that agent's Write call was refused).

## Reproduction gate

- Interpolators: max |diff| vs stored per-scenario MAE 1e-15 pu over 23,980 rows (gate 1e-9: pass).
- GENCO: max |diff| 1.9e-5 pu over 4,796 rows (gate 1e-5: MISS). 3 of 30 grid-configs exceed it (case240_pserc at both fractions, 1.4e-5; case3120sp_k at 20%, 1.9e-5), at most 0.2% of the MAE in those groups. Cause (audit F4): float32 device noise, not a code difference. The 1e-5 gate was set too tight: the published report (`2026-10-21-genco-voltage-inference/report.md` line 32) had already recorded 1.7e-5 for the same re-run at 20%. Evidence: `run_variant_pb` is line-for-line `run_variant` plus the `vm[mb_]` copy (same seed, batch size, checkpoint, mask), and `repro_diff.json` shows the logged and saved values agree, so saving adds nothing. Running the same code on CPU (float32, 1 thread) gives max per-scenario MAE diff 1.3e-5 vs the published run and 3.9e-6 vs this rerun for case240_pserc f40_s0, and 1.7e-5 vs published and 9.3e-6 vs rerun for case3120sp_k f20_s0; both GPU runs sit about 1e-5 from a CPU run and from each other. All 30 GENCO groups differ from the published values (min 1.5e-7, median 6e-7), the signature of noise. Effect on counts: CPU vs saved GPU predictions flip 0 of 4,928 violations (case240 f40, tau 0 and 0.005) and 1 of 3,605 (case3120 f20, tau 0; net 0 false alarms, 2 flips at tau 0.005, net 0); only 754 of 1,033,116 unmeasured predictions lie within 1e-4 of a limit, so any pooled count can move by at most about ten (under 0.1%). Not done: the other 28 groups on CPU, and a same-GPU repeat (no GPU use). A gate of about 5e-5, or one relative to the MAE, would have been the right pre-registration.

## Setup

Scored buses: all unmeasured buses (PQ, PV, REF), as the published MAE; PQ-only is a secondary row in `pooled.csv`. Scenarios: all 2,398 locked-15 test scenarios (none had a collapsed label). Limits: each bus's own min_vm_pu / max_vm_pu (constant across scenarios: the protocol said 'asserted', but no assert was in the code; checked after the fact in audit round 1, F6: for all 15 grids min_vm_pu and max_vm_pu take one value per bus over all test scenarios, and the saved lo/hi equal the last scenario's, so no count changes); tolerance tau 0 and 0.005 pu applied to label and prediction. Best interpolator: per grid and config by median published MAE (I2 on most grids).

## Counts (micro-pooled over the 15 grids, all unmeasured buses)

| config | tau | true violations | method | caught | missed | false alarms |
|---|---|---|---|---|---|---|
| 40% | 0 | 5,930 | GENCO | 1,634 (27.6%) | 4,296 | 11,853 |
| 40% | 0 | 5,930 | best interpolator | 226 (3.8%) | 5,704 | 1,574 |
| 40% | 0.005 | 388 | GENCO | 278 (71.6%) | 110 | 10,087 |
| 40% | 0.005 | 388 | best interpolator | 135 (34.8%) | 253 | 959 |
| 20% | 0 | 7,852 | GENCO | 1,816 (23.1%) | 6,036 | 18,015 |
| 20% | 0 | 7,852 | best interpolator | 234 (3.0%) | 7,618 | 3,741 |
| 20% | 0.005 | 523 | GENCO | 310 (59.3%) | 213 | 15,955 |
| 20% | 0.005 | 523 | best interpolator | 143 (27.3%) | 380 | 2,771 |

## Reading

- GENCO catches more violations but raises about 7.5 times more alarms. Precision at 40%, tau 0: GENCO 12.1%, best interpolator 12.6%; at tau 0.005: GENCO 2.7%, best interpolator 12.3%.
- Very sensitive to the tolerance: tau 0 to 0.005 cuts true violations about 15-fold (5,930 to 388 at 40%); GENCO's false alarms barely fall (11,853 to 10,087), the interpolators' fall by about 40%.
- GENCO's false alarms concentrate in case240 and case300 (4,510 and 2,892 of 11,853 at 40%, tau 0).
- Caveat on tau 0 (audit F5): the tau-0 rates mostly measure labels sitting on a limit to solver precision. Of the 5,930 true violations at tau 0 (40%), 4,068 (69%) exceed the limit by less than 1e-7 pu (median exceedance 9.4e-9 pu). Cumulative counts by exceedance: <1e-7 4,068; <1e-6 4,268; <1e-5 4,548; <1e-4 4,913; <1e-3 5,260; <5e-3 5,542 (f40, all unmeasured buses, float64 labels). So the 15-fold drop from tau 0 to 0.005 reflects labels on a limit, not 15 times fewer real violations; treat tau 0.005 as the operational headline. The counts are correct as defined.
- What it does not mean: labels are load-perturbed PGLib cases whose limits are OPF limits, so this is not operating-state screening. Unmeasured PV buses are included. One checkpoint, seed-0 masks, no confidence intervals.

The post's limitation sentence ("We did not save per-bus predictions, so we cannot say how many violations they miss or falsely flag") can be replaced only after an audit, and the GENCO reproduction miss should be explained first.

Regenerate the 68 MB per-bus arrays (`perbus/`, gitignored) with `run_interp_pb.py` and `run_genco_pb.py`.
