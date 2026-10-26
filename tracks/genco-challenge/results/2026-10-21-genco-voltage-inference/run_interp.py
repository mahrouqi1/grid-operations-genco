"""Run I1-I5 on the locked-15 test scenarios for every config in vcommon.CONFIGS. UNAUDITED. python run_interp.py [--grids ..]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import vcommon as V
C = V.C
ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="*", default=C.HELDOUT15); ap.add_argument("--configs", nargs="*", default=[c[0] for c in V.CONFIGS])
ap.add_argument("--out", type=Path, default=HERE / "interp_per_scenario.jsonl"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
ids = json.loads((SI := V.SI / "test_ids.json").read_text())
for g in a.grids:
    sc = ids[g]["test"][: a.limit or None]; tab = C.tables(g, sc)
    for name, frac, si in V.CONFIGS:
        if name not in a.configs: continue
        ms = V.measured_seeded(si); t0 = time.time(); rows = []
        for s in sc:
            bus, br, yb = tab["bus"][s], tab["br"][s], tab["y"][s]; n = len(bus)
            out, m, hp = V.predict_all(bus, br, yb, s, ms(s, n, frac)); vt = bus.Vm.to_numpy()
            for meth, x in out.items():
                rows.append({"grid": g, "config": name, "method": meth, "scenario": int(s), "vm_mae_unmeas": float(np.abs(x - vt)[~m].mean()) if (~m).any() else float("nan")})
            rows.append({"grid": g, "config": name, "method": "hyper", "scenario": int(s), "hp": {k: list(v) for k, v in hp.items()}})
        with a.out.open("a") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        print(g, name, len(sc), f"{time.time()-t0:.0f}s", flush=True)
