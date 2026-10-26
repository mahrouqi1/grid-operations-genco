"""I1-I5 for the OOS grids, every config. python run_interp.py --grids g1 g2 --out file.jsonl  (UNAUDITED)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ocommon as O
C, V = O.C, O.V
ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="+", required=True); ap.add_argument("--configs", nargs="*"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
ids = json.loads((HERE / "test_ids.json").read_text())
for g in a.grids:
    out = a.out or HERE / "per_grid" / f"interp_{g}.jsonl"; out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["config"] for l in out.read_text().splitlines()} if out.exists() else set()
    sc = ids[g]["test"][: a.limit or None]; tab = O.tables(g, sc)
    for name, frac, si, noisy in O.CONFIGS:
        if (a.configs and name not in a.configs) or name in done: continue
        ms = V.measured_seeded(si); t0 = time.time(); rows = []
        for s in sc:
            bus, br, yb = tab["bus"][s], tab["br"][s], tab["y"][s]; n = len(bus)
            pred, m, hp = O.predict_all(bus, br, yb, s, ms(s, n, frac), noisy, si); vt = bus.Vm.to_numpy()
            for meth, x in pred.items():
                rows.append({"grid": g, "config": name, "method": meth, "scenario": int(s), "vm_mae_unmeas": float(np.abs(x - vt)[~m].mean()) if (~m).any() else float("nan")})
        with out.open("a") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        print(g, name, len(sc), f"{time.time()-t0:.0f}s", flush=True)
