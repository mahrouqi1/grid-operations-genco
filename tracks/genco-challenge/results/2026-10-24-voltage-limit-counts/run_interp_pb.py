"""D1: copy of ../2026-10-21-genco-voltage-inference/run_interp.py that ALSO saves per-bus |V| predictions. UNAUDITED.
vcommon.py is an unmodified copy. Only configs f40_s0, f20_s0 (protocol.json).
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 python -I? (needs the genco env; run normally) python run_interp_pb.py"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import vcommon as V
C = V.C
CFGS = ["f40_s0", "f20_s0"]
ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="*", default=C.HELDOUT15)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--outdir", type=Path, default=HERE / "perbus")
a = ap.parse_args(); a.outdir.mkdir(exist_ok=True)
ids = json.loads((V.SI / "test_ids.json").read_text())
log = HERE / "interp_pb_mae.jsonl"
for g in a.grids:
    sc = ids[g]["test"][: a.limit or None]; tab = C.tables(g, sc)
    for name, frac, si in V.CONFIGS:
        if name not in CFGS: continue
        fn = a.outdir / f"interp__{g}__{name}.npz"
        if fn.exists() and not a.limit: continue
        ms = V.measured_seeded(si); t0 = time.time()
        P = {k: [] for k in ["I1", "I2", "I3", "I4", "I5"]}; lab, meas, pq, rows = [], [], [], []
        for s in sc:
            bus, br, yb = tab["bus"][s], tab["br"][s], tab["y"][s]; n = len(bus)
            out, m, hp = V.predict_all(bus, br, yb, s, ms(s, n, frac)); vt = bus.Vm.to_numpy()
            for meth, x in out.items():
                P[meth].append(x)
                rows.append({"grid": g, "config": name, "method": meth, "scenario": int(s), "vm_mae_unmeas": float(np.abs(x - vt)[~m].mean()) if (~m).any() else float("nan")})
            lab.append(vt); meas.append(m); pq.append(bus.PQ.to_numpy().astype(bool))
            lo, hi = bus.min_vm_pu.to_numpy(), bus.max_vm_pu.to_numpy()
        if not a.limit:
            np.savez_compressed(fn, scen=np.array(sc, int), label=np.array(lab), meas=np.array(meas), pq=np.array(pq), lo=lo, hi=hi,
                                **{k: np.array(v) for k, v in P.items()})
            with log.open("a") as fh:
                for r in rows: fh.write(json.dumps(r) + "\n")
        print(g, name, len(sc), f"{time.time()-t0:.0f}s", flush=True)
