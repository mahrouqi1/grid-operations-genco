"""Item 5: how the 0.5 pu threshold behaves. CPU only, read-only on data. UNAUDITED.
Run: OMP_NUM_THREADS=1 nice -n 15 python -I item5_threshold.py   (grid-operations env: numpy, pandas, pyarrow)
Writes item5_threshold.json and item5_bus_hist.csv next to this file.

Definitions (all from existing files):
  label  = original GENCO label table  data/external/genco-pf-pretraining/<g>/raw/bus_data.parquet  (Vm is SIGNED; negative Vm exists)
  root   = repaired table              data/external/genco-pf-pretraining-corrected/<g>/raw/bus_data.parquet
  flagged = the 1,085 rows of 2026-10-10-collapsed-repair-data/status.csv (any Vm < 0.5 or < 0 in the label)
  scored  = the 1,083 scenarios of 2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv, mapped to original ids
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RES = REPO / "tracks/genco-challenge/results"
EXT = REPO / "data/external"
ORIG, COR, EVAL = EXT / "genco-pf-pretraining", EXT / "genco-pf-pretraining-corrected", EXT / "genco-pf-pretraining-eval"
ST = pd.read_csv(RES / "2026-10-10-collapsed-repair-data/status.csv")
PS = pd.read_csv(RES / "2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv")
out = {}

# scored set in original ids
rows = []
for g in sorted(PS.grid.unique()):
    sm = pd.read_csv(EVAL / g / "raw/scenario_map.csv").set_index("new")["orig"]
    p = PS[PS.grid == g][["grid", "scenario"]].copy(); p["orig"] = p.scenario.map(sm); rows.append(p)
P = pd.concat(rows); P["g2"] = P.grid.str.replace("subnet_", "")
scored = set(zip(P.g2, P.orig.astype(int)))
ST["key"] = list(zip(ST.grid, ST.scenario))
ST["scored"] = ST.key.isin(scored)
out["n_flagged"] = len(ST); out["n_scored"] = len(scored); out["n_scored_in_flagged"] = int(ST.scored.sum())

# ---- A. whole local sample, label min |V| and min V per scenario (31 subgrids)
allrows = []
for d in sorted(ORIG.glob("subnet_*")):
    t = pq.read_table(d / "raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]).to_pandas()
    g = t.groupby("scenario").Vm
    a = pd.DataFrame({"min_v": g.min(), "min_abs": t.assign(a=t.Vm.abs()).groupby("scenario").a.min()})
    a["grid"] = d.name.replace("subnet_", ""); a = a.reset_index(); allrows.append(a)
A = pd.concat(allrows, ignore_index=True)
A["key"] = list(zip(A.grid, A.scenario)); A["scored"] = A.key.isin(scored)
out["n_scenarios_local_sample"] = len(A)
bins = [-np.inf, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, np.inf]
lab = ["<0 (signed)", "0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-0.95", ">=0.95"]
# scenario-level distribution of the label's signed minimum Vm over the whole local sample
out["scenario_signed_min_hist_all"] = dict(zip(lab, pd.cut(A.min_v, bins, right=False).value_counts(sort=False).tolist()))
# fine view of min |V| near the gaps, whole sample
fine = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 5]
out["scenario_min_abs_hist_all_fine"] = {f"{fine[i]}-{fine[i+1]}": int(((A.min_abs >= fine[i]) & (A.min_abs < fine[i + 1])).sum()) for i in range(len(fine) - 1)}
out["smallest_min_abs_above_0.5_all_labels"] = float(A.min_abs[A.min_abs >= 0.5].min())
out["n_label_min_abs_0.5_to_0.7_all"] = int(((A.min_abs >= 0.5) & (A.min_abs < 0.7)).sum())

# ---- B. flagged scenarios: label min |V| and signed min (status.csv), scenario level
S = ST.copy()
def sc_hist(x, edges, names):
    return dict(zip(names, np.histogram(x, edges)[0].tolist()))
out["flagged_scen_label_min_abs_hist"] = sc_hist(S.label_min_abs_vm, [0, 0.1, 0.3, 0.5, 0.7, 5], ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", ">=0.7"])
out["flagged_scen_label_signed_min_neg"] = int((S.label_min_vm < 0).sum())
out["flagged_scen_label_reason"] = S.label_reason.value_counts().to_dict()
out["flagged_scen_negative_and_abs_lt_0.5"] = int(((S.label_min_vm < 0) & (S.label_min_abs_vm < 0.5)).sum())
out["flagged_scen_negative_only_abs_ge_0.5"] = int(((S.label_min_vm < 0) & (S.label_min_abs_vm >= 0.5)).sum())
out["flagged_scen_nonneg_collapsed_lt_0.5"] = int(((S.label_min_vm >= 0) & (S.label_min_abs_vm < 0.5)).sum())
out["flagged_scen_label_min_abs_quantiles"] = {str(q): float(S.label_min_abs_vm.quantile(q)) for q in [0, .05, .25, .5, .75, .95, 1]}

# ---- C. bus level at flagged buses (label Vm < 0.5 signed, i.e. includes negatives), and the root at those same buses
BL, BR = [], []
for g in sorted(ST.grid.unique()):
    ids = set(ST[ST.grid == g].scenario)
    sub = "subnet_" + g
    lo = pq.read_table(ORIG / sub / "raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]).to_pandas()
    lo = lo[lo.scenario.isin(ids)]
    co = pq.read_table(COR / sub / "raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]).to_pandas()
    co = co[co.scenario.isin(ids)]
    m = lo.merge(co, on=["scenario", "bus"], suffixes=("_lab", "_root"), validate="1:1")
    m["grid"] = g
    BL.append(m)
B = pd.concat(BL, ignore_index=True)
B["scored"] = [(g, s) in scored for g, s in zip(B.grid, B.scenario)]
fl = B[B.Vm_lab < 0.5].copy()      # flagged buses
for name, sel in [("flagged_1085", fl), ("scored_1083", fl[fl.scored])]:
    x = sel.Vm_lab
    out[f"bus_{name}"] = dict(
        n_flagged_buses=int(len(sel)), n_neg=int((x < 0).sum()), frac_neg=float((x < 0).mean()),
        n_abs_lt_0p1=int((x.abs() < 0.1).sum()), n_abs_0p1_0p3=int(((x.abs() >= 0.1) & (x.abs() < 0.3)).sum()),
        n_abs_0p3_0p5=int(((x.abs() >= 0.3) & (x.abs() < 0.5)).sum()),
        n_neg_and_abs_lt_0p5=int(((x < 0) & (x.abs() < 0.5)).sum()), n_neg_abs_ge_0p5=int(((x < 0) & (x.abs() >= 0.5)).sum()),
        n_strict_between_0_and_0p1=int(((x >= 0) & (x < 0.1)).sum()), n_exactly_zero=int((x == 0).sum()),
        root_min_abs_at_flagged=float(sel.Vm_root.abs().min()), root_median_at_flagged=float(sel.Vm_root.median()),
        root_n_abs_lt_0p7_at_flagged=int((sel.Vm_root.abs() < 0.7).sum()), root_n_abs_lt_0p9_at_flagged=int((sel.Vm_root.abs() < 0.9).sum()),
        root_n_neg_at_flagged=int((sel.Vm_root < 0).sum()))
bh_edges = [-5, -1.0, -0.5, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
bh = np.histogram(fl.Vm_lab, bh_edges)[0]
pd.DataFrame({"lo": bh_edges[:-1], "hi": bh_edges[1:], "n_flagged_buses_label_Vm_signed": bh}).to_csv(HERE / "item5_bus_hist.csv", index=False)
out["bus_flagged_hist_signed_label_Vm"] = {f"[{bh_edges[i]},{bh_edges[i+1]})": int(bh[i]) for i in range(len(bh))}
rh_edges = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3]
rh = np.histogram(fl.Vm_root.abs(), rh_edges)[0]
out["bus_flagged_hist_root_abs_Vm"] = {f"[{rh_edges[i]},{rh_edges[i+1]})": int(rh[i]) for i in range(len(rh))}

# ---- D. repaired roots, scenario level min |V| (whole scenario) from status.csv
for name, sel in [("flagged_1085", S), ("scored_1083", S[S.scored])]:
    x = sel.min_vm_new
    out[f"root_scen_{name}"] = dict(
        n=int(len(sel)), min=float(x.min()),
        hist={k: int(v) for k, v in zip(["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", ">=1.0"], np.histogram(x, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5])[0])},
        n_lt_0p3=int((x < 0.3).sum()), n_lt_0p5=int((x < 0.5).sum()), n_lt_0p7=int((x < 0.7).sum()))

# ---- E. sensitivity: classification and acceptance at 0.3 / 0.5 / 0.7
sens = {}
for t in (0.3, 0.5, 0.7):
    row = {}
    # (i) label-side, whole local sample: scenarios whose label min |V| < t, or min signed V < 0 (negative flagged separately)
    row["all34274_label_min_abs_lt_t"] = int((A.min_abs < t).sum())
    row["all34274_label_min_abs_lt_t_or_neg"] = int(((A.min_abs < t) | (A.min_v < 0)).sum())
    row["scored1083_label_min_abs_lt_t"] = int((A[A.scored].min_abs < t).sum())
    row["scored1083_label_min_abs_lt_t_or_neg"] = int(((A[A.scored].min_abs < t) | (A[A.scored].min_v < 0)).sum())
    row["flagged1085_label_min_abs_lt_t"] = int((S.label_min_abs_vm < t).sum())
    row["flagged1085_label_min_abs_ge_t_i.e._not_collapsed_at_t"] = int((S.label_min_abs_vm >= t).sum())
    row["flagged1085_negative_only_label_abs_ge_t"] = int(((S.label_min_vm < 0) & (S.label_min_abs_vm >= t)).sum())
    row["scenarios_collapsed_at_t_not_in_the_1085_flagged"] = int((((A.min_abs < t) | (A.min_v < 0)) & ~A.key.isin(set(S.key))).sum())
    # (ii) root-side: repaired roots a rule with floor t would accept
    row["roots1085_min_abs_ge_t_accepted"] = int((S.min_vm_new >= t).sum())
    row["roots1085_rejected_at_t"] = int((S.min_vm_new < t).sum())
    row["roots1083_rejected_at_t"] = int((S[S.scored].min_vm_new < t).sum())
    # buses (flagged) counts at t
    row["flagged_buses_label_abs_lt_t_in_1085_scen"] = int((B.Vm_lab.abs() < t).sum())
    row["flagged_buses_label_abs_lt_t_or_neg_in_1085_scen"] = int(((B.Vm_lab.abs() < t) | (B.Vm_lab < 0)).sum())
    sens[str(t)] = row
out["sensitivity"] = sens
(HERE / "item5_threshold.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
