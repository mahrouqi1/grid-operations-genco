"""Verify EVERY generated label of each carved subgrid: power-balance residual |V conj(Y V) - S| < 1e-6 pu at every bus (V from the label's
Vm/Va, Y = the scenario's own y_bus_data, S = (Pg-Pd) + j(Qg-Qd) per bus / 100), all Vm in [0.5, 1.5], all finite. Scenarios failing are
excluded from every test (collapsed ones were already dropped by ladder_data.validate_grid). Writes verified.json. python verify.py [grid...] UNAUDITED."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grids as G
TOL = 1e-6
out_p = HERE / "verified.json"; out = json.loads(out_p.read_text()) if out_p.exists() else {}
for g in (sys.argv[1:] or G.SEL):
    raw = G.ROOT / g / "raw"
    if not (raw / "bus_data.parquet").exists(): print(g, "no data"); continue
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va"]).sort_values(["scenario", "bus"])
    yb = pd.read_parquet(raw / "y_bus_data.parquet", columns=["scenario", "index1", "index2", "G", "B"])
    by = dict(tuple(yb.groupby("scenario"))); res = {}
    for s, b in bus.groupby("scenario"):
        n = len(b); y = by[s]; Y = sp.csc_matrix((y.G.to_numpy() + 1j * y.B.to_numpy(), (y.index1.to_numpy(), y.index2.to_numpy())), shape=(n, n))
        V = b.Vm.to_numpy() * np.exp(1j * np.radians(b.Va.to_numpy())); S = ((b.Pg - b.Pd).to_numpy() + 1j * (b.Qg - b.Qd).to_numpy()) / 100.0
        r = float(np.abs(V * np.conj(Y @ V) - S).max()); fin = bool(np.isfinite(r))
        res[int(s)] = (r, fin and 0.5 <= b.Vm.min() and b.Vm.max() <= 1.5)
    ok = sorted(s for s, (r, f) in res.items() if f and r < TOL)
    out[g] = {"n_scenarios": len(res), "n_pass": len(ok), "pass_ids": ok, "fail": {s: v[0] for s, v in res.items() if s not in set(ok)},
              "max_residual_pass": max((res[s][0] for s in ok), default=None), "median_residual": float(np.median([v[0] for v in res.values()]))}
    print(g, len(res), "pass", len(ok), "max pass residual", out[g]["max_residual_pass"], "median", out[g]["median_residual"], flush=True)
    out_p.write_text(json.dumps(out))
