"""Follow-up checks for the repair-roots audit: sign-flip subset, the 3 old 9591 roots vs repaired roots, 1070 vs 13 medians."""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).parent
REPO = H.parents[3]
RES = REPO / "tracks/genco-challenge/results"
EXT = REPO / "data/external"
D = pd.read_csv(H / "audit_repair_roots_dist.csv")
PS = pd.read_csv(RES / "2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv")
ST = pd.read_csv(RES / "2026-10-10-collapsed-repair-data/status.csv")
rows = []
for g in sorted(PS.grid.unique()):
    sm = pd.read_csv(EXT / "genco-pf-pretraining-eval" / g / "raw/scenario_map.csv").set_index("new")["orig"]
    p = PS[PS.grid == g].copy(); p["orig"] = p.scenario.map(sm); p["g"] = g.replace("subnet_", ""); rows.append(p)
P = pd.concat(rows)
M = D.merge(P[["g", "scenario", "orig", "root_found", "root_med_vm_at_low", "mae_vs_root_at_low"]], left_on=["g", "s"], right_on=["g", "scenario"])
M = M.merge(ST[["grid", "scenario", "status", "min_vm_new", "max_branch_angle_deg"]].rename(columns={"scenario": "orig"}), left_on=["g", "orig"], right_on=["grid", "orig"])
sf = M[M.status == "recovered_signflip"]
print("signflip n", len(sf), "closer", int((sf.A_root < sf.A_lab).sum()), "median A_root", sf.A_root.median(), "max", sf.A_root.max())
print("signflip rows with A_root>0.1:", sf[sf.A_root > 0.1][["g", "s", "A_root", "root_med_low", "lab_med_low"]].to_dict("records"))
t = M[M.status == "recovered"]
print("two-start n", len(t), "median A_root", t.A_root.median(), "p90", t.A_root.quantile(.9))
print("far rows status:", M[M.A_root > 0.1][["g", "s", "status", "min_vm_new", "max_branch_angle_deg"]].to_dict("records"))
st = M[(M.min_vm_new < 0.70)]
print("stressed (min Vm<0.7) in scored:", len(st), "median A_root", st.A_root.median(), "closer", int((st.A_root < st.A_lab).sum()))
ex = M[M.min_vm_new >= 0.70]
print("non-stressed:", len(ex), "median A_root", ex.A_root.median())
r9 = M[(M.g == "case9591_goc_idx_12") & M.root_found]
print("9591 old-found roots vs repaired roots:")
print(r9[["s", "root_med_vm_at_low", "root_med_low", "mae_vs_root_at_low", "A_root", "status"]])
# old-search roots across all grids that were found: compare old-root median at low vs repaired root
f = M[M.root_found]
dd = (f.root_med_vm_at_low - f.root_med_low).abs()
print("old Newton roots vs repaired: n", len(f), "median |diff of median-at-low|", dd.median(), "n >0.01:", int((dd > 0.01).sum()))
print(f[dd > 0.01][["g", "s", "root_med_vm_at_low", "root_med_low", "status"]].to_string())
