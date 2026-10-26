# Transfer experiment: continued pretraining of GENCO for unseen grid families, go/no-go (2026-10-20)

**UNAUDITED; audit `2026-10-20-genco-transfer` round 1 findings F1-F5 applied (the owner 2026-10-20 23:09), round 2 pending.**
- Protocol `protocol.json`, frozen at 52d0b5d, with amendments 3e07e3d, 4fc4062 and 771c142, all recorded before
  any result.
- Training: OSC Ascend array 7939787 (15 runs) plus 6 local replay-only runs. 73.4 OSC GPU-hours on <OSC_PROJECT>,
  smoke tests included.
- Selection: on the RTE validation family (`selection.md`).
- Evaluation: one scoring of the 15 locked-benchmark grids per run (`heldout/`). Verdict computed by `go_no_go.py`
  -> `go_no_go.md` / `.json`.

## Verdict: NO-GO

The validation-selected arm c (curriculum of power-flow masking, ladder data plus corrected replay) fails pre-registered
criteria 1 to 3 on the locked 15:

| criterion | arm c | required |
|---|---|---|
| 1: median of the seed-mean branch-angle ratio to DC | 5.52 (released 3.69: 50% worse) | ≤ 2.77 |
| 2: grids where branch angle is better than released | 1 of 15 | ≥ 11 |
| 3: per-seed median drop | -57%, -58%, -37% | ≥ +15% each |
| 4: \|V\| beats DC | 15 of 15 (passes) | ≥ 12 |

Criterion 5 (preservation) was not scored, because criteria 1 to 3 already decide the verdict.

## What the numbers say (rewritten 2026-10-20 23:20 after audit round 1, approved by the owner)

1. **At matched training length, every arm is worse than released on branch angle** (`contrast_12k.md`, every run's
   12,000-update checkpoint, audit R1-F4):

   | arm | change in median branch-angle ratio vs released |
   |---|---|
   | C00 (replay only, random mask) | +21% |
   | C01 (replay only, 50/50 mask) | +21% |
   | D (ladder, random mask) | +54.5% |
   | b at 12k (still on the random mask) | +59% |
   | a (ladder, PF mask) | +87% |
   | c (ladder, curriculum) | +93% |
   | m (ladder, 50/50 mask) | +106% |

   The ladder data roughly doubles the damage of continued pretraining (D vs C00). Adding power-flow masking on top
   of the ladder data makes it worse again (a, c, m vs D). Without the ladder data the mask is neutral (C01 = C00).
2. **With validation selection, step 0 (the released weights) is a candidate** (audit R1-F1). 6 of 21 runs select
   it. The selected-checkpoint table (`go_no_go.md`) then reads:

   | arm | change vs released |
   |---|---|
   | D | -5.6% |
   | b | -36% |
   | C01 | 0% (all three seeds select the released weights) |
   | C00 | -8% |
   | c | -50% |
   | m | -55% |
   | a | -80% |

   The earlier "D -42%" and "b -64%" came from leaving the released weights out of the candidates.
3. **|V|: criterion 4 passes, but |V| error got worse** (audit R1-F2). GENCO stays below DC on |V| on 12-15 of 15
   grids in every arm. Its |V| ERROR is nevertheless worse than released on 12-14 of 15 grids in every arm that moved off the released
   weights; C01 selects the released weights on every seed (audit R2-F3) (median
   1.6x to 3.7x released; arm c 2.3x). Only the beats-DC count moved, and arm b (12/15) beats DC less often than
   released (13/15).
4. **RTE validation is noisier than the differences between arms** (audit R1-F3).
   - D and b are the same configuration through 14k updates (b's PF fraction is 0 before epoch 5), yet their paired
     validation ratios differ by a median 1.62x and up to 2.77x.
   - Only 44 of 135 ladder checkpoints beat released on validation (median 10.47 vs released 9.24).
   - So the validation "gain of up to 19%" is a minimum over noisy checkpoints, not a family effect, and the arm
     ranking on validation is within noise.
   - 9 of 21 selected checkpoints are at 2k-4k updates, all 9 in ladder arms; 6 of 21 select step 0 (audit R2-F3).
     The final state beats step 2000
     on validation in 10 of 15 ladder runs (R1-F5).
5. **Rescue round: NO-GO** (`protocol-rescue.json`, OSC array 7959186, 15.0 GPU-hours; `selection-rescue.md`,
   `go_no_go-rescue.md`; UNAUDITED). Three anti-forgetting variants, each one knob on top of arm m.
   - **Validation selection** (5 RTE grids, step 0 offered):

     | arm | validation metric (seed mean) | note |
     |---|---|---|
     | released | 7.00 | baseline |
     | Rrep (replay x9) | 6.05 | selected |
     | Rlr (LR 1e-5) | 6.72 | 1 seed selects step 0 |
     | Rfrz (frozen trunk) | 7.00 | all seeds select step 0 |

   - **Locked 15:** Rrep's median branch-angle ratio is 4.83 vs released 3.69 (31% worse). It is better than released
     on 2 of 15 grids, and the per-seed drops are -32% / -38% / -22%. Criteria 1-3 fail; |V| beats DC 15/15, but Rrep's |V| ERROR is worse than released on 12 of 15 grids
     (median 2.1x), and Rlr's on 10 of 15 (1.4x) (audit R2-F4).
   - Rlr is 12% worse; Rfrz equals released (no checkpoint beat it on validation).
   - **Selection-free read** (pre-registered amendment 3, `rescue_final_read.md`, audit R2-F1; not a go/no-go
     input): every rescue arm at its FINAL state on the locked 15.

     | arm | branch angle vs released | grids better | grids with \|V\| worse |
     |---|---|---|---|
     | Rlr | +48.1% | 2/15 | 10/15 |
     | Rfrz | +31.4% | 2/15 | 14/15 |
     | Rrep | +47.6% | 2/15 | 12/15 |

   - As in the main round, a validation gain on RTE did not transfer to the locked 15.
   - Record correction (audit R2-F5, wording per claims audit R2-F3): before the 02:45 amendment, 39 Rlr_s0
     validation evaluations at the 1,000-update cadence existed on disk (values never shown). The cadence change
     dropped the 17 scored odd-step rows (steps 1000, 3000, 5000, 7000); Rlr_s0 then selected the released weights.
     Like for like on the 4 grids scored at step3000, step3000 averages 6.845 against released 7.420 (step5000
     7.096, step7000 6.325), so it would probably have selected a non-zero step. Rrep stays selected: Rlr_s0 would
     have to score below about 5.0 on all 5 grids to bring Rlr's seed mean under Rrep's 6.053.

## Checks

- **Pipeline identity:** the released weights scored through the run path (`c_s0` `state_step0.pt`) reproduce the
  released rows exactly on case14, case118 and case300 (`identity-check/`).
- **Leakage check, normaliser audit, converter verification:** see `leakage.md`, `normalizer_audit.md` and
  `../2026-10-20-ladder-to-datakit/verify_tree.json`.
