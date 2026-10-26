# Do the published collapsed labels come from datakit's fast path, and does `method = :newton` fix them? (2026-10-18)

UNAUDITED. Protocol written before the run (`protocol.json`); scripts `run.py`, `probe2.jl`, `build_mpc.py` (copied
from `2026-10-10-datakit-collapse-cause`), `summarize.py`; raw rows `out.jsonl`; control arm `textbook_newton.py` -> `textbook_newton.out` (protocol_addendum1.json; its first planned form, PowerModels with a finite-difference Jacobian, took about 15 minutes per case and was stopped after 1); tables `table.md`, `summary.json`;
rebuilt inputs in `data/external/genco-published-newton-work/` (gitignored). Asked by audit post-claims R1-F1 and
an owner decision: the 10-10 cause and fix were shown only on grids we generated ourselves.

## Bottom line

1. **The published collapsed labels are exactly what datakit's fast path produces.** Rebuilding each scenario's PF
   input from GENCO's published raw tables and calling PowerModels `compute_ac_pf` as datakit's fast path does
   (default NLsolve trust-region, flat start) reproduces the published label to below 1e-6 pu on **51 of 51**
   sampled collapsed scenarios (all 9 affected subgrids, up to 6 each) and on **9 of 9** healthy controls (one per
   subgrid). So the defect is in the label-generation step, not in GENCO's training or model. This does not say
   which property of these grids drives the solver to the collapsed root (the zero-injection explanation from
   10-10 was traced on our own grids only).
2. **Switching that call to `method = :newton` is not a fix for the published grids; a textbook Newton does better but
   is not a drop-in fix either.**
   PowerModels `compute_ac_pf(method = :newton)` from the same flat start reaches our physical root on **15 of 51**
   collapsed scenarios and does not converge on **35** (also not on 5 of 9 healthy ones); its failures blow up to
   NaN (audit genco-published-newton R1-F1). An independent textbook polar Newton (MATPOWER form, no damping,
   `textbook_newton.py`, copied from that audit) on the same rebuilt inputs and the same flat start reaches the
   physical root on **31 of 51**: all 6 on 5658_epigrids_idx_8, 4837_goc_idx_28, 19402_goc_idx_47 and
   4837_goc_idx_37, 5 of 6 on 10480_goc_idx_40, 2 of 6 on 20758_epigrids_idx_49, and none on 4837_goc_idx_35,
   9591_goc_idx_12 and 10480_goc_idx_26. It lands on the published collapsed label on 3, on another root on 1, and
   does not converge on 16. So PowerModels' Newton path fails where Newton itself does not. But from a flat start the
   textbook Newton also fails on 3 of 9 healthy controls and reaches a different low-voltage root on a fourth
   (9591_goc_idx_12 s0, min Vm 0.586), all of which the fast path labels correctly, and it is not enough on four
   subgrids (4837_35, 9591_12 and 10480_26 at 0 of 6 or 0 of 3, 20758_49 at 2 of 6).
3. **What does fix them.** Our relabeler (two independent starts including DC-type and damped starts, accepted only
   when they agree) gives a physical label to 1,072 of the 1,085 flagged scenarios (`2026-10-10-collapsed-repair-data`;
   1,069 below 0.5 pu plus 16 negative-only, 13 of which are sign flips checked by identity only). A data-generator
   fix therefore needs either a Newton that works on these grids plus a better start on the hard subgrids, or the
   relabeler's starts. Not tested here.

## Table (totals; per subgrid in `table.md`)

| kind | method | n | equals published label (< 1e-6 pu) | our physical root (< 1e-4 pu) | collapsed | not converged |
|---|---|---|---|---|---|---|
| collapsed | trust_region (datakit default) | 51 | 51 | 0 | 51 | 0 |
| collapsed | PowerModels newton | 51 | 0 | 15 | 1 | 35 |
| collapsed | textbook Newton (control) | 51 | 3 | 31 | 3 | 16 |
| healthy | trust_region (datakit default) | 9 | 9 | n/a | 0 | 0 |
| healthy | PowerModels newton | 9 | 4 | n/a | 0 | 5 |
| healthy | textbook Newton (control) | 9 | 5 | n/a | 0 | 3 (1 other root) |

## Provenance

`out.jsonl` and `table.md` come from the scripts as committed in 2f9b1df (probe2.jl, run.py) and 0a83faf
(summarize.py); the later robustness edits (fb72913: exceptions recorded as such, resume needs both rows; this
commit: one row per tag and method kept, asserted) do not change them (no exception occurred). The stopped
finite-difference arm (`out_fd.jsonl`, `run_fd.log`, 1 case, about 11 minutes of solver time) was produced by a
`run.py`/`probe2.jl` variant that was not kept (it only swapped the method keyword). `textbook_newton.out` was
produced by `python textbook_newton.py` with the 60 `.m` files of `cases.csv` in that order (not a glob).

## Limits

51 of the 1,069 sample scenarios with a bus below 0.5 pu (the 1,085 flagged also include 16 negative-only rows), evenly spaced per subgrid; one healthy control per subgrid. The input
rebuild assumes the published raw tables hold datakit's exact PF input, which the 51 + 9 exact reproductions
support. The datakit version that produced the published data is not known; the result shows its output is
reproduced by the fast-path call as in the `genco-paper-repro` branch.
