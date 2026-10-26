"""why only 283 of 1,083 scored collapsed scenarios have a verified root, and how GENCO's PF-task output
compares to the repair-recovered roots (2026-10-10-collapsed-repair-data) on ALL 1,083.
Reads only existing files. Run (env grid-operations-genco, repo root):
    OMP_NUM_THREADS=1 nice -n 15 python tracks/genco-challenge/results/2026-10-24-repair-root-audit/q_a_roots.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
RES = REPO / "tracks/genco-challenge/results"
PS = pd.read_csv(RES / "2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv")
ST = pd.read_csv(RES / "2026-10-10-collapsed-repair-data/status.csv")
ML = REPO / "data/external/genco-ladder/collapsed-label/mlruns"
EVAL = REPO / "data/external/genco-pf-pretraining-eval"
COR = REPO / "data/external/genco-pf-pretraining-corrected"
out = {}

# ---- 1. outcome of the original root search (analyze.physical_root: flat then DC start, all |V|>=0.5, resid<1e-8)
PS["outcome"] = np.where(PS.root_found, "root_found_" + PS.root_start.fillna(""),
                         np.where(np.isfinite(PS.root_best_resid), "converged_but_rejected", "no_finite_convergence"))
out["n_scored"] = int(len(PS))
out["n_low_buses"] = int(PS.n_low.sum())
out["outcome_counts"] = PS.outcome.value_counts().to_dict()
rej = PS[PS.outcome == "converged_but_rejected"]
out["rejected_best_resid_quantiles"] = rej.root_best_resid.quantile([0, .5, 1]).tolist() if len(rej) else None
out["rejected_resid_lt_1e-8"] = int((rej.root_best_resid < 1e-8).sum())
out["by_grid"] = PS.groupby(["grid", "outcome"]).size().unstack(fill_value=0).to_dict("index")

# ---- 2. join to the repair status (scenario ids: per_scenario uses renumbered ids; map back to original)
rows = []
for g in sorted(PS.grid.unique()):
    sm = pd.read_csv(EVAL / g / "raw/scenario_map.csv").set_index("new")["orig"]
    p = PS[PS.grid == g].copy()
    p["orig"] = p.scenario.map(sm)
    rows.append(p)
P = pd.concat(rows)
P["g2"] = P.grid.str.replace("subnet_", "", regex=False)
J = P.merge(ST.rename(columns={"grid": "g2", "scenario": "orig"}), on=["g2", "orig"], how="left", suffixes=("", "_st"))
out["scored_in_repair_status"] = int(J.status.notna().sum())
out["repair_status_counts_for_scored"] = J.status.value_counts(dropna=False).to_dict()
out["repair_phase_counts_for_scored"] = J.phase.value_counts(dropna=False).to_dict()
nf = J[~J.root_found]
out["not_found_n"] = int(len(nf))
out["not_found_repair_status"] = nf.status.value_counts(dropna=False).to_dict()
out["not_found_repair_phase"] = nf.phase.value_counts(dropna=False).to_dict()
sp = nf.start_pair.fillna("").astype(str)
out["not_found_start_family"] = {
    "contains_neighbor": int(sp.str.contains("neighbor").sum()),
    "contains_homotopy": int(sp.str.contains("homotopy|continuation|lambda").sum()),
    "other": int((~sp.str.contains("neighbor|homotopy|continuation|lambda")).sum())}
out["not_found_start_pair_top"] = sp.value_counts().head(8).to_dict()

# ---- 3. GENCO PF-task output vs the repaired root, all scored scenarios
res = []
for g in sorted(PS.grid.unique()):
    f = glob.glob(f"{ML}/*/*/artifacts/test/{g}_predictions.parquet")
    assert len(f) == 1, (g, f)
    pr = pd.read_parquet(f[0], columns=["scenario", "bus", "Vm_pred", "Vm_target"])
    sm = pd.read_csv(EVAL / g / "raw/scenario_map.csv").set_index("new")["orig"]
    pr["orig"] = pr.scenario.map(sm)
    cor = pd.read_parquet(COR / g / "raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]).rename(columns={"Vm": "Vm_root"})
    cor["Vm_root"] = cor.Vm_root.abs()
    pr = pr.merge(cor, left_on=["orig", "bus"], right_on=["scenario", "bus"], how="left", suffixes=("", "_c"))
    flagged = set(PS[PS.grid == g].scenario)
    for s, d in pr[pr.scenario.isin(flagged)].groupby("scenario"):
        low = (d.Vm_target < 0.5).to_numpy()
        if not low.any():
            low = (d.Vm_target < 0).to_numpy()  # sign-flip-only scenarios
        if not low.any():
            continue
        vp, vl, vr = d.Vm_pred.to_numpy(), d.Vm_target.to_numpy(), d.Vm_root.to_numpy()
        res.append(dict(grid=g, scenario=int(s), n_low=int(low.sum()),
                        label_med=float(np.median(vl[low])), root_med=float(np.median(vr[low])), pred_med=float(np.median(vp[low])),
                        mae_label=float(np.abs(vp[low] - vl[low]).mean()), mae_root=float(np.abs(vp[low] - vr[low]).mean()),
                        root_min_vm_all=float(vr.min()), pred_min_low=float(vp[low].min()),
                        pred_lt_half=float((vp[low] < 0.5).mean())))
R = pd.DataFrame(res).merge(P[["grid", "scenario", "root_found"]], on=["grid", "scenario"])
R["closer_to_root"] = R.mae_root < R.mae_label
R.to_csv(Path(__file__).with_name("q_a_all_scored_vs_repaired_root.csv"), index=False)
out["n_compared"] = int(len(R))
out["closer_to_repaired_root_all"] = int(R.closer_to_root.sum())
out["closer_to_repaired_root_not_found"] = [int(R[~R.root_found].closer_to_root.sum()), int((~R.root_found).sum())]
out["closer_to_repaired_root_found"] = [int(R[R.root_found].closer_to_root.sum()), int(R.root_found.sum())]
out["median_mae_root_found"] = float(R[R.root_found].mae_root.median())
out["median_mae_root_not_found"] = float(R[~R.root_found].mae_root.median())
out["median_mae_label_all"] = float(R.mae_label.median())
out["pred_lt_half_buses_frac_all"] = float((R.pred_lt_half * R.n_low).sum() / R.n_low.sum())
per = R.groupby("grid").agg(n=("scenario", "size"), n_nf=("root_found", lambda x: int((~x).sum())),
                            label_med=("label_med", "median"), root_med=("root_med", "median"), pred_med=("pred_med", "median"),
                            mae_label=("mae_label", "median"), mae_root=("mae_root", "median"),
                            closer=("closer_to_root", "sum"), root_min_vm=("root_min_vm_all", "median"))
out["per_grid_vs_repaired_root"] = per.round(4).to_dict("index")
# the scenarios where GENCO is NOT closer to the repaired root
bad = R[~R.closer_to_root]
out["not_closer_list"] = bad[["grid", "scenario", "mae_label", "mae_root", "label_med", "root_med", "pred_med"]].round(3).to_dict("records")[:20]
# 9591 specifics
r9 = R[R.grid == "subnet_case9591_goc_idx_12"]
out["case9591_idx_12"] = dict(n=int(len(r9)), root_med_median=float(r9.root_med.median()), root_med_min=float(r9.root_med.min()),
                              root_med_max=float(r9.root_med.max()), pred_med_median=float(r9.pred_med.median()),
                              mae_root_median=float(r9.mae_root.median()), mae_label_median=float(r9.mae_label.median()),
                              closer=int(r9.closer_to_root.sum()))
(Path(__file__).with_name("q_a_results.json")).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
