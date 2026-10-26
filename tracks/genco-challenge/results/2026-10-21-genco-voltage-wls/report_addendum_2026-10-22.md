# WLS baseline: addendum after audit round 1 (2026-10-22)

**UNAUDITED.** Controls for audit `2026-10-22-genco-voltage-wls` R1-F1 (dispatch prior) and R1-F2 (capped solves), plus F3-F6 fixes. Arms pre-registered in `protocol_amendment_3.json` (with 3b, 3c) before they ran; hashes in `files_v2.sha256`. Compute stopped 07:15 EDT. `verdict_v2.py` reproduces all 306 published cells (0 mismatches). Cells are low-powered: 30 scenarios each (10 on whole grids). Text written by the orchestrating chat from the run agent's account (that agent's Write call for this file was refused).

## Released GENCO wins at 40% measured

A win = released GENCO beats the best of the comparators named in the column header (error ratio below 1, upper end of its 95% bootstrap interval below 1). The last column is GENCO-A1 (limits-ablated GENCO) against I1-I5 + WLS-flat. "Published" WLS = informed WLS with the true generator limits; "WLS-flat" = limit-blind WLS, iteration-capped.

| set | cells | I1-I5 only | + WLS (published) | published, fallback | + WLS-flat | WLS-flat, fallback | GENCO-flat vs I1-I5 + WLS-flat |
|---|---|---|---|---|---|---|---|
| carved pieces | 36 | 31 | 6 | 16 | 8 | 29 | 7 (all 36) |
| locked-15, f40 s0 | 15 | 11 | 2 | 5 | 3 | 5 | no GENCO-flat data |
| locked-15, f20 pooled | 15 | 7 | 1 | 3 | 2 | 2 | no GENCO-flat data |
| seen whole grids | 15 | 14 | 1 | 13 | 8 | 14 | 2 (6 of 15 cells) |
| unseen whole grids | 17 | 5 | 0 | 3 | 3 | 3 | 2 (10 of 17 cells) |

- WLS-flat: every in-service generator's Pmax set to the scenario median and Pmin to 0 (the generator-limit ablation's A1 arm, `../2026-10-22-genco-voltage-limit-ablation/`); Q limits unchanged.
- GENCO-flat: GENCO-A1 from that ablation, so it exists only where the ablation ran.
- "fallback": WLS scenarios that hit the iteration cap are replaced by I2's error (a fallback policy, not a bound on WLS: the direction varies by grid, see report.md item 2).

## Reading

- **Carved pieces and locked-15:** the dispatch prior explains little. Of the 25 pieces wins that informed (published) WLS removed, 4 return with WLS-flat; 2 of the 6 that survived informed WLS are lost; net 6 -> 8. Locked-15: 1 of 9 returns, none lost (2 -> 3). With limits uninformative on BOTH sides (GENCO-A1 against I1-I5 + WLS-flat), GENCO-A1 wins 7 of 36 pieces (32 of 36 against I1-I5 alone). Removing the iteration cap from WLS-flat on 29 pieces cut non-convergence from 87% to 34% on those 29 and moves released GENCO's wins from 8 to 5 of the same 29 pieces (11 if unconverged solves are replaced by I2), so the direction holds and 8 is the GENCO-favourable end.
- **Seen (ladder-generated) whole grids:** the prior explains most of it. With WLS-flat, released GENCO wins 8 of 15 and has the lower median error in 12 of 15. "WLS beats GENCO" is NOT established there; only 6 of 15 cells have a both-sides-flat comparison (GENCO 2 wins).
- **Unseen whole grids:** informed (published) WLS is clearly ahead (it beats released GENCO alone in 13 of 17 cells, GENCO beats it in 1). WLS-flat is not: released GENCO has the lower median in 9 of 17, is significantly better in 8 cells and WLS-flat in 7. Against I1-I5 + WLS-flat released GENCO wins 3 of 17 (GENCO-A1: 2 of 10 comparable cells).
- **Uncapped informed WLS on 8 seen grids (≤ 4,837 buses):** GENCO wins 1 of 8 either way, so the direction survives the cap; uncapped WLS still fails to converge on 93% of scenarios on case2312 and case3022.

## Not run

WLS-flatuncapped on 7 pieces, locked-15 and all whole grids; informed uncapped on 7 seen grids (4020, 4601, 4619, 4917, 5658, 7336, 9591) and all pieces and unseen grids; GENCO-flat on 9 seen and 7 unseen whole grids and all of locked-15; flat prior at 10% / 20% / noisy 40%; the 36 ladder-regenerated pieces.

## Audit fixes F3-F6

- F3: the tuning-run files do not exist (`tuning_evidence_2026-10-22.md`); the "2x" sentence in `report.md` has no file behind it.
- F4: `protocol_amendment_3.json` pins sha256 of protocol and code and documents the drift (superseded solver text, "seed 7" vs the crc32 seed used, amendment-2 timestamp, post-tuning code hash) without rewriting `protocol.json`.
- F5: `table_v2.md` corrects the inverted column. At 40%, informed (published) WLS is significantly better than released GENCO in 29 of 36 pieces cells (GENCO in 6), 7 vs 2 seen whole, 13 vs 1 unseen whole, 10 vs 2 locked-15.
- F6: `tests/test_wls_partial.py` (case14, 8 of 14 |V| measured): identity control matches `reference_pf` to 1.2e-9 pu; 3 tests in `test_wls_partial.py` pass (4 with `test_wls_recovery.py`).

## Caveats

One fixed flat-prior construction, no retuning. GENCO-A1 moves GENCO off its pretraining distribution (a lower bound on a limits-blind GENCO). Flat arms inherit the published solver settings, tuned on two scored grids.
