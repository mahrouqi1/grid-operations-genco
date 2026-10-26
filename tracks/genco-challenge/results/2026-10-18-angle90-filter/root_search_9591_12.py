"""Round-2 audit: are the unflagged low-voltage scenarios of the F4 test set the physical (high-voltage) root?
Runs relabel.solve_scenario (flat, DC, ...) on a sample of lowv_scenarios.csv rows and compares to the label."""
import sys, json, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[4]
from genco_challenge import relabel as R
from genco_challenge.hybrid import Scenario
g = sys.argv[1]; n = int(sys.argv[2])
lv = pd.read_csv(REPO / "tracks/genco-challenge/results/2026-10-10-collapsed-repair-v3/lowv_scenarios.csv")
lv = lv[(lv.g == g) & (~lv.repaired)].sort_values("min_pq_vm")
pick = list(lv.sid.iloc[np.linspace(0, len(lv) - 1, min(n, len(lv))).astype(int)])
raw = REPO / f"data/external/genco-pf-pretraining/subnet_{g}/raw"
filt = [("scenario", "in", [int(x) for x in pick])]
bus = pd.read_parquet(raw / "bus_data.parquet", filters=filt); br = pd.read_parquet(raw / "branch_data.parquet", filters=filt)
yb = pd.read_parquet(raw / "y_bus_data.parquet", filters=filt)
B, BR, Y = dict(tuple(bus.groupby("scenario"))), dict(tuple(br.groupby("scenario"))), dict(tuple(yb.groupby("scenario")))
for s in pick:
    s = int(s)
    sc = Scenario(B[s], Y[s], BR[s]); sc.vm_set = np.where(sc.is_t["PV"] | sc.is_t["REF"], np.abs(sc.vm_label), 1.0)
    b = BR[s].sort_values("idx").reset_index(drop=True)
    res = R.solve_scenario(sc, b)
    vl = sc.vm_label * np.exp(1j * np.radians(B[s].sort_values("bus")["Va"].to_numpy())) if hasattr(sc, "vm_label") else None
    pq = sc.is_t["PQ"]
    out = {"sid": s, "label_min_pq_vm": float(np.abs(sc.vm_label)[pq].min()), "accepted": bool(res["accepted"]), "reason": res.get("reason"), "pair": res.get("pair")}
    if res["v"] is not None:
        v = res["v"]
        out["root_min_pq_vm"] = float(np.abs(v)[pq].min())
        out["max_dvm_vs_label"] = float(np.abs(np.abs(v) - np.abs(sc.vm_label)).max())
    # every candidate's last iterate (relabel._newton returns it whether or not it converged; audit published-newton R2-F4): min PQ Vm
    out["cands"] = [(nm, None if v is None else round(float(np.abs(v)[pq].min()), 4), None if v is None else round(float(np.abs(np.abs(v) - np.abs(sc.vm_label)).max()), 5)) for nm, v in res["cands"]]
    print(json.dumps(out), flush=True)
