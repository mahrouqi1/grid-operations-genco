# Corrected labels for the collapsed pretraining scenarios (2026-10-10)

**UNAUDITED.** Scripts `repair.py` (recovery and write-back), `summarize.py` (counts), `verify.py` (read-back of the written copies); per-scenario table `status.csv` (also per grid in `per_grid/`, and as `relabeled.csv` next to each corrected copy); logs in `logs/`. Every count below is recomputed from `status.csv` by `summarize.py` (`tables.md`).

## Bottom line

**All 1,085 flagged scenarios of the 9 affected subgrids in the local sample were recovered: 1,072 with a physical root (two independent starts agreeing), 13 sign-flip-only labels fixed by the identity check, 0 with no root.** The repair fine-tune (A2) therefore has **1,085 corrected scenarios** (of 10,128 across the 9 subgrids' local samples, 10.7%), 1,069 of them former collapsed roots (|Vm| < 0.5) and 16 former sign-flip-only labels. The earlier low root counts (69 of 665 on case4837_idx_35, 3 of 71 on case9591_idx_12) came from the starts of `analyze.physical_root` only; adding neighbour-root starts and continuation from healthy sibling scenarios (already in `relabel.py`) recovers the rest. Corrected copies: `data/external/genco-pf-pretraining-corrected/subnet_<grid>/raw/` (gitignored, 2.1 GB).

| subgrid | flagged | collapsed (min abs Vm < 0.5) | sign-flip only | recovered | recovered_signflip | no_root | rate |
|---|---|---|---|---|---|---|---|
| case4837_goc_idx_35 | 666 | 654 | 12 | 657 | 9 | 0 | 1.000 |
| case20758_epigrids_idx_49 | 275 | 275 | 0 | 275 | 0 | 0 | 1.000 |
| case9591_goc_idx_12 | 72 | 70 | 2 | 70 | 2 | 0 | 1.000 |
| case5658_epigrids_idx_8 | 28 | 28 | 0 | 28 | 0 | 0 | 1.000 |
| case4837_goc_idx_28 | 16 | 16 | 0 | 16 | 0 | 0 | 1.000 |
| case19402_goc_idx_47 | 10 | 8 | 2 | 8 | 2 | 0 | 1.000 |
| case10480_goc_idx_40 | 9 | 9 | 0 | 9 | 0 | 0 | 1.000 |
| case4837_goc_idx_37 | 6 | 6 | 0 | 6 | 0 | 0 | 1.000 |
| case10480_goc_idx_26 | 3 | 3 | 0 | 3 | 0 | 0 | 1.000 |
| **total** | 1085 | 1069 | 16 | 1072 | 13 | 0 | 1.000 |

("flagged" = label_diagnostics invalid: any Vm < 0.5 or < 0. The nine collapsed counts here are over the full local sample, not the test split of the mask report, hence 666 not 659 on case4837_idx_35. 3 of the 16 sign-flip-only scenarios, 3 on case4837_idx_35, did not pass the identity check and went through the normal recovery path, counted under `recovered`.)

## What was done

- Acceptance rule unchanged (`relabel.accept_relabel`: residual < 1e-9, all |V| >= 0.5, two starts from different initial points agree to 1e-6, max branch angle <= 90 deg). Functions reused from `relabel.py` as is (`solve_scenario`, `neighbor_pass`, `new_label_tables`, `check_consistency`); no edit to it. The pretraining raw format equals datakit raw minus `runtime_data`, so no adaptation was needed beyond making the PV/REF setpoints use |Vm| (a sign defect on a PV bus would otherwise corrupt the setpoint).
- Phase 0 sign-flip check (label |Vm| >= 0.5, some Vm < 0): accept the label's own complex voltage if it satisfies the power-flow identity; 13 of 16 did (all as written, `flip_as_written+identity`). Not confirmed by a second start: a flat-start Newton agreed on only 3 of those 13 (`flip_newton_agrees`; flat-start Newton is a weak check on stressed points). Treat these 13 as identity-verified, not two-start-verified.
- Phase 1 `solve_scenario` (flat, DC, damped flat, 4 shift-aware DC starts, repaired label): 1,085 - 13 = 1,072 scenarios entered; 428 accepted. Phase 2 `neighbor_pass` (roots of other scenarios of the same subgrid as starts, homotopy from 3 nearest healthy ids and 2 solved ids): 644 accepted. Phase 3 (my additions: homotopy from the 8 healthy scenarios nearest in Ybus/Sbus, 10 neighbour starts, Levenberg-Marquardt from flat, shift-DC and repaired label, load-scaling continuation from zero load recording mu_max) is implemented but was **never needed** (0 scenarios reached it), so the 3-hour budget was not used (compute: about 80 minutes wall in total, 6 workers, nice 10). Because nothing is unrecoverable, there is no evidence on scenarios without an operating point; the mu_max and homotopy-lambda columns exist but are empty.
- Start pairs: the accepted pair most often contains a neighbour-root or homotopy start (521 of the 1,072 pairs contain a neighbour start). A neighbour start is another scenario's root of the same subgrid; agreement with it is independent of this scenario's own DC solution but not of the subgrid's shared topology. Do not read it as DC-quality evidence.
- Write-back: collapsed scenario's bus Vm/Va/Pg/Qg, gen p_mw/q_mvar, branch pf/qf/pt/qt replaced consistently (relabel's `new_label_tables`, `check_consistency` ok on all 1,085); all other rows, the Ybus table and all other columns identical; scenario ids unchanged; tables keep the partitioned parquet layout and column order. No flag column was added (schema unchanged); the sidecar is `raw/relabeled.csv` (scenario, status, phase, start pair, residual, min Vm, max angle, ...).

## Read-back check (`verify.py`, `verify_written.json`)

On the written copies for all 9 subgrids: untouched rows of bus, gen, branch and y_bus equal the originals (all True); for the 1,085 repaired scenarios the written Vm/Va satisfy the power-flow equations with the written Pg/Qg/Pd/Qd to a max residual of 2.9e-10 pu (worst 8.5e-10 on case19402_idx_47, from the sign-flip labels, below the 1e-9 bar); min Vm 0.509; bus Pg/Qg equal the in-service generator sums at PV/REF buses (max gap 1.8e-12 MW); branch pf matches the Ybus branch formula (max gap 9.4e-10 MW).

## Caveats the fine-tune should know

- **Stressed roots.** 40 of the 1,085 repaired scenarios have min Vm < 0.70 and 42 have a branch angle > 64 deg (37 > 80 deg; max 89.85 deg, just under the 90 deg guard; lowest Vm 0.509, just above 0.5). Valid datakit labels reach 64 deg at most on the grids relabel measured. These are near-collapse roots that satisfy the rule; they are the least certain labels. Rows are identifiable in `status.csv` (`min_vm_new`, `max_branch_angle_deg`); A2 can report results with and without them.
- **Q limits.** Reactive-limit violations of the new roots reach 780 Mvar (case4837_idx_35), 321 (case20758_idx_49), 285 (case4837_idx_37); datakit does not enforce Q limits in PF mode (see the label-validation report), but A2 should compare against the violation level of healthy scenarios before treating it as normal.
- **Slack Pg moved** (e.g. 577.6 MW to 529.4 MW on the first case10480_idx_26 scenario): the physical root implies a different slack dispatch than the collapsed label; this is intended.
- Local-sample scope only: the 9 subgrids' local data (about 1,060 to 1,200 scenarios each), not GENCO's full published sets; counts of collapsed scenarios in the full sets will be larger.
- Unaudited; the sign-flip subset rests on the identity check alone.
