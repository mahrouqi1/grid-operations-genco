# GENCO RepairKit

A checker and whole-state relabeler for power-flow datasets in the gridfm-datakit raw format
(`bus_data`, `gen_data`, `branch_data`, `y_bus_data` parquet tables, hive-partitioned `scenario_partition=N` or plain files).
It generalizes the tested code of `results/2026-10-10-collapsed-repair-data/` (`repair.py`, `verify.py`), `relabel.py`,
`ladder_data.label_diagnostics` and `results/2026-10-18-angle90-filter/scan_all31.py`. The numerics are those modules; nothing is rewritten.

Environment: `grid-operations-genco`, CPU only. From the repo root:
`export PYTHONNOUSERSITE=1 PYTHONPATH=tracks/genco-challenge/src:core`.

## What it detects (`check`)

Per scenario the report holds: `min_abs_vm` (all buses), `min_pq_abs_vm`, `min_vm` (signed), `n_neg_vm`, `n_collapsed`,
`max_branch_angle_deg` (in-service branches, from the complex label voltage `Vm*exp(j Va)`, so a sign-shifted label is not misread),
`residual` (power mismatch in pu of the label as written, from the scenario's own `y_bus`), `angle_gt90`, `residual_gt_tol` and a `category`:

| category | rule | action |
|---|---|---|
| `collapsed` | some `abs(Vm) < 0.5`, or non-finite: an unintended low-voltage root | repaired |
| `negative_only` | no `abs(Vm) < 0.5`, some `Vm < 0`: a sign defect (`-abs(V)` written with the angle shifted by 180 degrees) | repaired |
| `angle_gt90` | otherwise valid, but an in-service branch angle is above 90 degrees | flagged for review, never repaired |
| `healthy` | none of the above | untouched |

Precedence is collapsed, negative_only, angle_gt90, healthy; `angle_gt90` is also a boolean column on every row.
`residual_gt_tol` is informational: datakit converges to about 1e-8, so healthy labels often exceed the 1e-9 bar that only the
repaired labels have to meet. Without `y_bus_data.parquet` the residual column is NaN and the summary says so; `repair` needs it.

## What it repairs (`repair`) and what it does not

Flagged scenarios (`--categories collapsed,negative`, default both) go through the phases of the 2026-10-10 repair:
0 sign-flip identity check (the label's own complex voltage, accepted if it passes the rule below),
1 `relabel.solve_scenario` (flat, DC, damped flat, shift-aware DC starts, repaired label),
2 `relabel.neighbor_pass` (neighbour roots, homotopy from the 3 nearest healthy ids and 2 solved ids),
3 extra effort (homotopy from the 8 healthy scenarios nearest in (Ybus, Sbus), neighbour starts, Levenberg-Marquardt, load continuation).

**Acceptance (unchanged, `relabel.accept_relabel`)**: power-mismatch residual < 1e-9, every `abs(V) >= 0.5`, two starts from
different initial points reach roots that agree to 1e-6, and the largest in-service branch angle is <= 90 degrees.

An accepted root is written to every dependent field: bus `Vm, Va, Pg, Qg`; gen `p_mw, q_mvar`; branch `pf, qf, pt, qt`
(`relabel.new_label_tables`, with `relabel.check_consistency` as a gate). All other cells, the y_bus table, scenario ids and row order are unchanged.

Does not do: repair `angle_gt90` scenarios (a branch angle above 90 degrees is a question for a human, not a defect with a known fix);
change any input (PV/REF setpoints, loads, topology); enforce reactive limits (datakit does not either; the log carries `q_limit_violation_mvar`);
promise a root exists. A scenario with no accepted root is **unresolved**: it stays byte-identical in the output and is listed in `repair_log.csv`.
Two-start agreement is stronger than the identity check used in phase 0; where phase 0 applies the log says `phase 0` and `start_pair` ends in `identity`.

Differences from the 2026-10-10 script: it works on any raw dir (hive or flat, any subgrid) and streams partitions when writing; the pool of healthy
donor scenarios is the 16 healthy ids nearest to each flagged id instead of the whole subgrid (phase 2 uses the 3 nearest, so it is unchanged;
phase 3 picks its 8 nearest from this pool). Memory for `repair` is the flagged scenarios plus donors, not the dataset. A flat (non-partitioned) table is read whole when rewritten.

## Commands

```
python -m genco_challenge.repairkit check  <raw_dir> [--out report.csv] [--workers N]
python -m genco_challenge.repairkit repair <raw_dir> <out_dir> [--workers N] [--categories collapsed,negative]
python -m genco_challenge.repairkit verify <orig_dir> <repaired_dir> [--log repair_log.csv] [--out result.json]
```

`<raw_dir>` is the directory holding the `*.parquet` tables (`.../subnet_<grid>/raw`). `check` never writes to it.
`repair` refuses a non-empty `out_dir` and a missing `y_bus_data`.

## Outputs

`check`: `report.csv` (one row per scenario) and `report.csv.summary.json`; counts per category are printed.
`repair` writes a full copy of the dataset to `out_dir` (changed partitions rewritten with the original schema, every other file copied) plus:
- `repair_log.csv`: scenario, category, status (`recovered`/`unresolved`), phase, start_pair, residual, max_dv_between_starts, max_branch_angle_deg,
  min_vm_before (signed, as written), min_vm_after, then extras (q_limit_violation_mvar, unresolved_reason, ...);
- `originals.parquet/`: the original rows of every changed scenario, one parquet per table (`bus_data`, `gen_data`, `branch_data`, `y_bus_data`);
- `check_report_before.csv`, `repair_summary.json`.

`verify` exits 0 or 1. It checks (a) every recovered scenario's written label solves its own power-flow equations (residual < 1e-9), all
`Vm >= 0.5` and positive, angles <= 90, bus Pg/Qg equal the in-service generator sums, branch flows match the Ybus formula, the label really changed and no column outside
the label set changed; (b) every scenario that is not `recovered` (unresolved ones included) is byte-identical or equal to 1e-12 in all tables;
(c) tables, columns, partitions, row counts and key columns (scenario, bus, idx, index1, index2) match in order.

## Loading with graphkit

`examples/load_with_graphkit.py` copies a repaired dir, renumbers the copy to scenario ids 0..N-1 (`genco_challenge.renumber_raw`; graphkit needs this,
the repaired dir keeps the original ids) and iterates samples with `HeteroGridDatasetDisk` (pretraining branch under `data/external/gridfm-graphkit-repro-pretraining`).

## Tests

`tests/test_repairkit.py` (CPU, about 2 minutes): categories, read-only check, hive and flat layouts, missing y_bus, repair round trip on three collapsed scenarios and a sign defect,
verify pass, and verify failing on a damaged healthy scenario, a bad repaired label and a dropped row.

## Limits

- Tested on the pretraining subgrid format; datakit `runtime_data` and other files are copied unchanged.
- A root that satisfies the rule can still be a near-collapse operating point (low Vm, large angles); filter on `min_vm_after` and `max_branch_angle_deg` in the log if that matters.
- Unresolved scenarios are left as they were, collapsed label included; downstream code should drop or mask them using `repair_log.csv`.
- Local numbers and the demo comparison against the 2026-10-10 corrected tree are in `results/2026-10-19-repairkit-check31/` and the task report.

## Licence of the data this was built for

The Hugging Face dataset card of `gridfm/genco-pf-pretraining` (README front matter, checked 2026-10-19 with `HfApi().dataset_info`, revision `4e036b57578d0aa235680eef1f72aea07166bac4`)
declares `license: apache-2.0`. There is no separate LICENSE or NOTICE file in the dataset repo; the card is the only statement. Apache-2.0 allows redistributing modified
copies and derived works (corrected tables included), provided the recipient gets a copy of the licence, modified files carry a notice that they were changed, and the
original copyright and attribution notices are kept. So a corrected derived copy may be shared if it says it is a modified version of `gridfm/genco-pf-pretraining` (cite the dataset and the
GENCO paper), ships the Apache-2.0 text, and describes the change (`repair_log.csv` does). The card says nothing about the upstream case files (GOC, EPIGRIDS) that the subgrids
come from, so check their terms before any public redistribution. This is a reading of the card, not legal advice. Nothing has been redistributed.
