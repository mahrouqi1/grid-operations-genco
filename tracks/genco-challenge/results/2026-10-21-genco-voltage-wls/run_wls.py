"""Run the WLS estimator on one study's scenarios/masks/noise. UNAUDITED. Spec: protocol.json.
python run_wls.py --study oos|control|fvs|genctl|locked15 [--grids ...] [--limit k]   -> per_case/<study>/<grid>.csv (resumable)"""
import argparse, json, sys, time, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parent
DIRS = {"locked15": "2026-10-21-genco-voltage-inference", "oos": "2026-10-21-genco-voltage-inference-oos", "control": "2026-10-21-genco-voltage-inference-control",
        "fvs": "2026-10-21-genco-voltage-inference-family-vs-size", "genctl": "2026-10-21-genco-voltage-inference-generator-control"}
ap = argparse.ArgumentParser(); ap.add_argument("--study", required=True); ap.add_argument("--grids", nargs="*"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
SD = R / DIRS[a.study]; sys.path.insert(0, str(SD)); sys.path.insert(0, str(HERE))
import wlslib as W
if a.study == "locked15":
    import vcommon as V; C = V.C; tables = C.tables; noise_z = None
    ids = json.loads((V.SI / "test_ids.json").read_text()); grids = a.grids or C.HELDOUT15
    CONF = [("f20_s0", .2, 0, False), ("f20_s1", .2, 1, False), ("f20_s2", .2, 2, False), ("f40_s0", .4, 0, False)]
    getids = lambda g: ids[g]["test"]
else:
    import ocommon as O; V = O.V; C = O.C; tables = O.tables; noise_z = O.noise_z
    ids = json.loads((SD / "test_ids.json").read_text())
    grids = a.grids or sorted(p.name[len("interp_"):-len(".jsonl")] for p in (SD / "per_grid").glob("interp_*.jsonl"))
    want = {"f20_s0", "f20_s1", "f20_s2", "f40_s0", "f40_s1", "f40_s2"} | (set() if a.study == "genctl" else {"n40_s0", "n40_s1", "n40_s2"})
    CONF = [c for c in O.CONFIGS if c[0] in want]
    getids = lambda g: ids[g]["test"]
SIGMA_REL = 0.005


def scenario(bus, gen, yb, sid, name, frac, si, noisy):
    n = len(bus); ms = V.measured_seeded(si)(sid, n, frac); m = np.zeros(n, bool); m[ms] = True
    vt = bus.Vm.to_numpy(); ymeas = vt * (1 + SIGMA_REL * noise_z(sid, n, si)) if noisy else vt
    Pd, Qd = bus.Pd.to_numpy(float), bus.Qd.to_numpy(float)
    ins = gen[gen.in_service == 1]
    isg = np.zeros(n, bool); isg[ins.bus.to_numpy()] = True; isg |= (bus.PV.to_numpy() == 1) | (bus.REF.to_numpy() == 1)
    gen_idx = np.where(isg)[0]; zero_idx = np.where(~isg)[0]
    pmin = np.zeros(n); pmax = np.zeros(n); qmin = np.zeros(n); qmax = np.zeros(n)
    np.add.at(pmin, ins.bus.to_numpy(), ins.min_p_mw.to_numpy(float)); np.add.at(pmax, ins.bus.to_numpy(), ins.max_p_mw.to_numpy(float))
    np.add.at(qmin, ins.bus.to_numpy(), ins.min_q_mvar.to_numpy(float)); np.add.at(qmax, ins.bus.to_numpy(), ins.max_q_mvar.to_numpy(float))
    den = pmax.sum() - pmin.sum(); s = float(np.clip((1.02 * Pd.sum() - pmin.sum()) / den, 0, 1)) if den > 1e-9 else 0.0
    pg = pmin + s * (pmax - pmin); qg = 0.5 * (qmin + qmax)
    p_pri = (pg[gen_idx] - Pd[gen_idx]) / 100.0; q_pri = (qg[gen_idx] - Qd[gen_idx]) / 100.0
    sp_ = np.maximum((pmax - pmin)[gen_idx] / 2, 10.0) / 100.0; sq_ = np.maximum((qmax - qmin)[gen_idx], 100.0) / 100.0
    ref = int(np.where(bus.REF.to_numpy() == 1)[0][0]); th_ref = float(np.deg2rad(bus.Va.to_numpy()[ref]))
    Y = C.ybus(yb, n)
    sig = (SIGMA_REL * np.abs(ymeas[ms])) if noisy else 1e-3
    est = W.estimate(Y, n, ref, th_ref, ms, ymeas[ms], sig, zero_idx, 1e-4, gen_idx, p_pri, q_pri, sp_, sq_, Pd=Pd, Qd=Qd)
    x = np.clip(est["vm"], V.VM_LO, V.VM_HI)
    mae = float(np.abs(x - vt)[~m].mean()) if (~m).any() else float("nan")
    g = len(gen_idx)
    return {"wls_mae": mae, "status": est["status"], "iters": est["iters"], "n": n, "m_V": int(m.sum()), "g_buses": g, "count_observable": int(m.sum() >= 2 * g - 1)}


for g in grids:
    out = a.out or (HERE / "per_case" / a.study / f"{g}.csv"); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh: done = {r["config"] for r in csv.DictReader(fh)}
    sc = getids(g)[: a.limit or None]; tab = tables(g, sc)
    for name, frac, si, noisy in CONF:
        if name in done: continue
        t0 = time.time(); rows = []
        for s in sc:
            r = scenario(tab["bus"][s], tab["gen"][s], tab["y"][s], s, name, frac, si, noisy)
            rows.append({"study": a.study, "grid": g, "config": name, "scenario": int(s), **r})
        new = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); 
            if new: w.writeheader()
            w.writerows(rows)
        st = [r["status"] for r in rows]
        print(a.study, g, name, len(sc), f"{time.time()-t0:.0f}s", "conv", sum(x == "converged" for x in st), "med", f"{np.median([r['wls_mae'] for r in rows]):.4f}", flush=True)
