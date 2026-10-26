"""Write protocol.json (+ sha256) once, before any run. Refuses to overwrite."""
import hashlib, json, time
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
path = OUT / "protocol.json"
if path.exists():
    raise SystemExit("protocol.json exists: written once, before any run")
proto = {
 "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
 "question": "Can GENCO add anything on top of fd2 (DC + two fast-decoupled XB iterations, computed from inputs only)?",
 "grids": ["case14_ieee", "case30_ieee", "case39_epri", "case57_ieee", "case73_ieee_rts", "case89_pegase", "case118_ieee", "case162_ieee_dtc", "case300_ieee"],
 "grid_prefix": "pglib_opf_",
 "eval_sets": "data/external/genco-ladder/eval2/<grid> (fresh eval2 sets). Used for REPORTING only; no choice below reads them.",
 "validation_sets": {
   "grids": ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee"],
   "what": "200 scenarios per dev grid drawn without replacement (numpy default_rng(20261009)) from labelfree.pool_lf(grid) of the training set data/external/genco-ladder/train (profile index 200..1199, disjoint from eval2 window 1600..1799), scenario ids renumbered 0..199 into data/external/genco-ladder/genco_on_fd2/val/<grid>. Their AC labels are used as validation labels. Stage 1 trains nothing, so the overlap of this pool with the stage 2 training subsets does not matter for stage 1."},
 "stage1": {
  "model": "zero-shot GENCO pretrained checkpoint (genco-pf-transfer/pretraining/best_model_state_dict.pt), graphkit evaluate unchanged, no slack-Vm plugin, input-only normalizer fit (FEWSHOT_INPUT_NORM=1: baseMVA from inputs only). Sensitivity row set: graphkit default normalizer fit (reads AC Qg and slack Pg of 2 scenarios), eval2 only, reported but never used to choose.",
  "states": {
   "dc": "datakit Va_dc, |V| = setpoint at PV/REF, 1.0 at PQ (as ladder_eval 'dc')",
   "fd2": "labelfree_prior2.priors_for(input_problem(...))['fd2'] (DC, 2 XB iterations, guarded fallback to DC), angles unwrapped as labelfree_prior2._unwrapped",
   "nr1": "same, key 'nr1' (1 damped Newton step from DC)",
   "genco_raw": "GENCO prediction as is (reference only)",
   "genco_pinned": "ladder_eval 'genco_pinned' (reference only)"},
  "variants": {
   "a_fd2": "fd2 |V| and fd2 angles",
   "b_genco_vm_fd2_va": "angles = fd2; |V| = setpoint at PV/REF, GENCO at PQ",
   "c_fd2_vm_genco_va": "|V| = fd2; angles = GENCO angles shifted so the REF bus angle equals the REF angle input (mean over REF buses); REF angle fixed",
   "d_blend": "per bus type blend. |V|: PV/REF = setpoint, PQ = fd2 + w_vm*(GENCO - fd2). Angle: REF fixed; PV and PQ buses = fd2 + w*wrap(GENCO_shifted - fd2) with w_pv, w_pq. Weights chosen on validation only (below)."},
  "d_selection_rule": [
   "Grid for w_vm: {0, 0.25, 0.5, 0.75, 1}. Score = geometric mean over the 2 dev grids of validation |V| MAE (all buses) divided by that of w_vm=0. Pick argmin; ties to the smaller w.",
   "Grid for (w_pv, w_pq): {0,0.25,0.5,0.75,1}^2. Score = geometric mean over the 2 dev grids of validation branch dVa MAE divided by that at (0,0). Pick argmin; ties to the smaller w_pv + w_pq.",
   "A chosen weight set is adopted only if its score is <= 0.98 (at least 2% better than fd2); otherwise the weights are 0 for that component (d equals fd2 there).",
   "The same weights are used on all 9 grids."],
  "metrics": {
   "branch_dva": "mean over in-service branches and scenarios of |wrap((Va_f - Va_t)_pred - (Va_f - Va_t)_AC)| in degrees (lead)",
   "va_slack_ref": "mean |wrap(Va_pred - Va_AC)| over buses, degrees (angles referenced to the REF angle input)",
   "vm_mae": "mean ||V|pred - |V|AC| over buses, pu",
   "ac_residual": "mean over buses of |P mismatch| in MW and over PQ buses of |Q mismatch| in MVAr, computed from the predicted complex voltage with OUR OWN per-scenario Ybus/Sbus (labelfree_prior2.input_problem; hybrid.Scenario.f; slack P and PV/REF Q unconstrained). NOT graphkit's residual. Reported for dc, fd2, nr1, genco_raw and every variant on the same scenarios.",
   "scenarios": "exactly the scenarios graphkit scored for the GENCO prediction; all other states are scored on that same set",
   "uncertainty": "paired bootstrap over scenarios (2000 resamples, rng seed 1) 95% interval of the ratio of mean branch dVa (and |V| MAE, residual) variant/fd2; fraction of scenarios where the variant has lower branch dVa than fd2"},
  "improves_fd2_means": "on a grid and metric: ratio to fd2 < 1 with the bootstrap upper bound < 1."},
 "gate": {
  "rule": "Stage 2 is run iff, on the VALIDATION sets (case14 and case30, zero-shot), some variant among b, c, d has geometric-mean ratio to fd2 <= 0.98 on branch dVa, or on |V| MAE, or on AC active residual, AND that holds on both dev grids individually (ratio < 1). The gate reads validation only. Eval-set numbers are reported either way.",
  "if_gate_fails": "Stage 2 is not run as pre-registered. Any later stage 2 run is labelled outside the gate."},
 "stage2": {
  "what": "Label-free adaptation of GENCO with the fd2 state as the training target prior (labelfree_prior2 machinery, train_prior2_fd2 built from training inputs only, loss = physics + w * MSE to the fd2 state). NOT implemented: extra Newton-style correction input channels on the GENCO model (would change the architecture inputs); the 'correction on top of fd2' is evaluated through variants raw, b, c and e50 applied to the adapted prediction.",
  "grid": {"prior_weight": [1, 3], "lr": [3e-5, 1e-4], "trainable": ["all", "frozen trunk (labelfree.FROZEN)"]},
  "fixed": "slack |V| known, input-only normalizer, pool_lf/splits_lf, n=1000, draw 0 for selection, GENCO case118_1000 fine-tune recipe, pretrained init. The existing run lfp2_fd2_w1_lr3e-05_all (case14, case30, draw 0, n=1000) is reused as the grid cell (w=1, lr=3e-5, all) because it is the identical configuration.",
  "selection_rule": "labelfree.select rule applied to this grid: validation layer_11_residual R (best epoch), r = R / min over configs per dev grid, S = geometric mean over the 2 dev grids; ineligible if prior-distance > 10x median on a dev grid, or the run kept the pretrained checkpoint, or a dev run failed; argmin S; ties within 2% to the smaller prior distance. No eval number enters.",
  "final": "chosen config, 9 grids, draws 0-2, n=1000, scored on eval2 (draws that cannot finish inside the time box are reported as missing, never filled in)",
  "variants_scored": ["adapted raw", "b", "c", "e50 (all blend weights 0.5, no selection)"],
  "improves_fd2_means": "mean over draws of ratio to fd2 < 1 AND < 1 on every draw run"},
 "timing": {
  "what": "per-scenario wall time, same machine, reported in a plain table: DC (sparse b' solve, from input_problem factors), fd2 total (DC + 2 XB iterations incl. LU), nr1 total (DC + 1 damped Newton step), GENCO forward pass (batched 64 on GPU shared, and on CPU), plus checkpoint state load and data-to-graph time reported separately. Median over scenarios of 3 repeats; box load recorded.",
  "note": "box load about 45 on 24 cores and shared GPUs: absolute numbers are noisy; the order of magnitude is the claim."},
 "rules": "nice 10; retry OOM; a crashed run is never scored; eval sets never used for any choice."
}
text = json.dumps(proto, indent=1)
path.write_text(text)
(OUT / "protocol.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  protocol.json\n")
print(hashlib.sha256(text.encode()).hexdigest())
