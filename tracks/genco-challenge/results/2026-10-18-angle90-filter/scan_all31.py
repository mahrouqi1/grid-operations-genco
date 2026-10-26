"""Copied 2026-10-18 from audit genco-collapsed-repair round 2 (scratch/round-2/angle_scan_all.py) so the post can cite it (post-claims R1-F6). Across all 31 hosted subgrids (physical-root copy: corr for affected, orig otherwise), count scenarios whose LABEL
has an in-service branch angle > 90 deg (relabel.ANG_MAX) or > 64 deg, and how many of those the Vm<0.5 flag catches."""
import json, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # results/<dir>/file -> parents[4] is the repo root
W = REPO / "data/external/genco-repair-finetune"
sp = json.loads((REPO / "tracks/genco-challenge/results/2026-10-10-collapsed-repair-finetune/splits.json").read_text())
AFF = set(sp["affected"]); tot = []
for g in sorted(sp["grids"]):
    kind = "corr" if g in AFF else "orig"
    b = pd.read_parquet(W / kind / g / "raw/bus_data.parquet", columns=["scenario", "bus", "Vm", "Va", "PQ"])
    br = pd.read_parquet(W / kind / g / "raw/branch_data.parquet", columns=["scenario", "from_bus", "to_bus", "br_status"])
    br = br[br.br_status == 1]
    va = b.set_index(["scenario", "bus"]).Va
    a = va.reindex(pd.MultiIndex.from_arrays([br.scenario, br.from_bus])).to_numpy() - va.reindex(pd.MultiIndex.from_arrays([br.scenario, br.to_bus])).to_numpy()
    a = np.abs((a + 180) % 360 - 180)
    mx = pd.Series(a, index=br.scenario.to_numpy()).groupby(level=0).max()
    mn = b[b.PQ == 1].groupby("scenario").Vm.min()
    n = len(mx); g90 = int((mx > 90).sum()); g64 = int((mx > 64).sum())
    tot.append((g, kind, n, g90, g64, int(((mx > 90) & (mn.reindex(mx.index) >= 0.85)).sum()), round(float(mx.max()), 1)))
    print(*tot[-1], flush=True)
d = pd.DataFrame(tot, columns=["g", "kind", "n_scen", "gt90", "gt64", "gt90_and_minVm>=0.85", "max_angle"])
print("TOTAL scen", d.n_scen.sum(), "gt90", d.gt90.sum(), "gt64", d.gt64.sum(), "grids with any gt90", int((d.gt90 > 0).sum()))
d.to_csv(Path(__file__).resolve().parent / "scan_all31.csv", index=False)
