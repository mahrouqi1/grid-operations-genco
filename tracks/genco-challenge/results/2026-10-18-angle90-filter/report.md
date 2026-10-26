# Branch angles above 90 degrees in the published pretraining sample (2026-10-18)

UNAUDITED as a result dir. The scan and the root search were first run by audit genco-collapsed-repair round 2
(findings R2-F1, R2-F2) and copied here unchanged apart from paths, so the post draft can cite them (post-claims
R1-F6, R2-F7). The scan was re-run from this dir on 2026-10-18 and gives the same totals.

## What it shows

- **Scan of all 31 publicly hosted subgrids** (`scan_all31.py` -> `scan_all31.out`, `scan_all31.csv`; 34,274
  sampled scenarios, the same 6-partition sample as the collapsed-label check). Per subgrid it uses the corrected
  copy for the 9 subgrids with collapsed scenarios and the original for the other 22, and reports: scenarios, how
  many have an in-service branch angle above 90 degrees (`relabel.ANG_MAX`) or above 64 degrees, how many of the
  >90-degree ones also have every PQ |V| at least 0.85, and the largest angle. Total: **535 scenarios above 90
  degrees (1.6 percent) on 6 of 31 subgrids, 517 of them on `case9591_goc_idx_12`** (12 on
  `case5658_epigrids_idx_30`, 3 on `case4837_goc_idx_35`, 1 each on three others); 783 above 64 degrees. 15 of the 535 (all those
  outside `case9591_goc_idx_12` and `case4837_goc_idx_35`) have every PQ |V| at or above 0.85; none of the 517 on
  `case9591_goc_idx_12` does. The Vm < 0.5 collapse flag catches none of the 535.
- **Root search on 12 low-voltage scenarios of `case9591_goc_idx_12`** (`root_search_9591_12.py` ->
  `root_search_9591_12.jsonl`; `relabel.solve_scenario` with flat, DC, shift-aware DC and damped starts). 11 of the
  12 have a label above 90 degrees, and none of those 11 found a root that passes the 90-degree guard
  (`implausible_angle_roots_only`). The 12th (sid 26) is not above 90 degrees; the root accepted for it is its
  published label (corrected in audit genco-published-newton R1-F2). So these labels may be genuine solutions
  beyond the steady-state stability limit rather than wrong roots; this was not established either way.
- **Filtered data for OSC job B** (`build_filter90.py` -> `scan.csv`, `dropped.csv`, data in
  `data/external/genco-pf-pretraining-filtered90/`, gitignored): drops the >90-degree scenarios from job B's 9
  subgrids (521 dropped: 517, 3, 1). Used only with the opt-in `FILTER90=1` (an owner decision).

## Limits

A sample of 6 partitions per subgrid, not the corpus. The 90-degree threshold is this repo's own plausibility
guard, not a physical law. The root search covers 12 scenarios on one subgrid.
