# Tuning-run evidence (audit round 1, F3) - UNAUDITED

**Status: the tuning runs left no files. Nothing is reconstructed here.**

protocol.json amendment 1 says the solver's iteration cap and damping were chosen by comparing MAE on 4 scenarios each of
`pglib_opf_case500_goc` and `case1888rte` (both are scored sets). Checked on 2026-10-22:

- The results directory holds no tuning log, no tuning script, and no per-setting MAE table. `logs/` starts at 22:15 (the first scored run, after the final solver).
- `protocol_code.sha` hashes the post-tuning `wlslib.py` / `run_wls.py` (last written 22:15), so it does not pin the settings that were tried.
- The numbers quoted in protocol.json (cost 5e7 vs 88 at the label start; MAE 0.021 vs best interpolator 0.0145 on case500_goc for the first flat-start attempt) are the writing chat's recollection of console output; no file backs them.
- report.md caveat 1 ("a different solver setting moved individual MAEs by up to 2x") has no file behind it either. Read it as "not recorded".
- The audit measured the selection effect instead of reproducing the tuning: dropping the two tuned grids (6 of 306 cells, all in the whole-grid sets) leaves every carved-piece count unchanged and moves the seen-whole 40% re-derived pair from 13 -> 1 of 15 to 13 -> 1 of 14 (audit F3, recomputed).

What is on disk that bears on solver sensitivity, all produced after the fact:
- the auditor's probe results (the auditor's start-point and prior probes).
- This follow-up's own arms: `per_case_v2/uncapped/` (iteration budget (40,60) instead of (8,12), same informed prior) is a direct, recorded sensitivity of the published numbers to the iteration cap.
