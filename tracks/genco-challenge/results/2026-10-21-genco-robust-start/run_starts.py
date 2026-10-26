"""Newton from five starts on the verified ladder labels (UNAUDITED). See protocol.json (frozen before the first GENCO-start run).
    python run_starts.py <grid>      -> per_scenario/<grid>.csv, identity/<grid>.json
Needs score_genco.py's predictions (mlruns/) and timing/<grid>.json (GENCO per-scenario cost, a constant per grid)."""
import glob, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse.linalg as spla
HERE = Path(__file__).resolve().parent
from genco_challenge import hybrid as hy
from grid_ops.reference_pf import newton_raphson_pf

ROOT = hy.REPO / "data/external/genco-robust-start"
TOL, MAX_IT = 1e-8, 25
VM_TOL, VA_TOL_DEG = 1e-4, 1e-2
grid = sys.argv[1]


def wrap_deg(d):
    return (np.degrees(d) + 180.0) % 360.0 - 180.0


def own_dc(sc):
    """Own DC solve (t_classical.dc_state recipe): angles = REF angle + B'^-1 P. Returns (Va rad, seconds)."""
    t0 = time.perf_counter()
    th = np.full(sc.n, sc.va_ref)
    th[sc.pvpq] = sc.va_ref + spla.spsolve(sc._bp, sc.sbus.real[sc.pvpq])
    return th, time.perf_counter() - t0


def newton(sc, v0):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        try:
            r = newton_raphson_pf(sc.ybus, sc.sbus, v0, sc.pv, sc.pq, tol=TOL, max_it=MAX_IT)
            v, it, ok = r.v, int(r.iterations), bool(r.converged)
        except Exception:
            v, it, ok = np.full(sc.n, np.nan, complex), MAX_IT, False
        sec = time.perf_counter() - t0
    fin = bool(np.all(np.isfinite(v))) and float(np.abs(v).max()) < 1e3
    mis = hy.maxmis(sc, v) if fin else float("nan")
    ok = ok and fin and mis <= TOL
    return v, it, ok, mis, sec


def classify(sc, v, ok):
    if not ok:
        return "diverged", np.nan, np.nan
    dvm = float(np.abs(np.abs(v) - sc.vm_label).max())
    d = np.angle(v) - sc.va_label
    d = d - d[sc.ref].mean()                       # align the slack angle
    dva = float(np.abs(wrap_deg(d)).max())
    return ("verified_root" if dvm < VM_TOL and dva < VA_TOL_DEG else "other_root"), dvm, dva


run = max((Path(p).parent.parent for p in glob.glob(str(HERE / "mlruns/*/*/tags/mlflow.runName")) if Path(p).read_text().strip() == f"zeroshot__{grid}"), key=lambda p: p.stat().st_mtime)
pred_all = pd.read_parquet(run / "artifacts/test" / f"{grid}_predictions.parquet")
scen = sorted(int(s) for s in pred_all["scenario"].unique())[:100]
pred_by = {s: p.sort_values("bus") for s, p in pred_all[pred_all["scenario"].isin(scen)].groupby("scenario")}
tg = json.loads((HERE / "timing" / f"{grid}.json").read_text())["per_scenario_s"]["total"]
scs = hy.load_grid(grid, scen, data_root=ROOT)
rows, ident = [], []
for s in scen:
    sc = scs[s]
    low = float(sc.vm_label.min())
    sc.v_ref = sc.v_label                           # the label IS the verified root (protocol)
    st = hy.starts_for(sc, pred_by[s])
    dc_th, t_dc = own_dc(sc)
    dc_check = float(np.abs(wrap_deg(dc_th - sc.va_dc)).max())
    sc._bp_lu = sc._bpp_lu = None
    t0 = time.perf_counter(); v = st["dc"].copy()
    for _ in range(2):
        v = sc.fdxb_step(v)
    t_fd = time.perf_counter() - t0
    starts = {"flat": (st["flat"], 0.0), "dc": (st["dc"], t_dc), "fd2": (v, t_dc + t_fd),
              "genco_vmdc": (st["vmdc"], t_dc + tg), "genco_full": (st["pinned"], tg)}
    for name, (v0, tstart) in starts.items():
        vf, it, ok, mis, sec = newton(sc, v0)
        status, dvm, dva = classify(sc, vf, ok)
        e0 = np.abs(v0 - sc.v_label)
        rows.append(dict(grid=grid, scenario=s, start=name, status=status, iterations=it, final_mismatch=mis, max_dVm=dvm, max_dVa_deg=dva,
                         start_s=tstart, newton_s=sec, total_s=tstart + sec, start_max_abs_dV=float(e0.max()), n_bus=sc.n,
                         label_min_vm=low, dc_vs_va_dc_max_deg=dc_check))
    vf, it, ok, mis, sec = newton(sc, sc.v_label.copy())      # identity control
    ident.append(dict(scenario=s, iterations=it, converged=ok, max_change=float(np.abs(vf - sc.v_label).max()) if ok else None))
(HERE / "per_scenario").mkdir(exist_ok=True); (HERE / "identity").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(HERE / "per_scenario" / f"{grid}.csv", index=False)
(HERE / "identity" / f"{grid}.json").write_text(json.dumps({"grid": grid, "n": len(ident), "max_iterations": max(i["iterations"] for i in ident),
    "all_converged": all(i["converged"] for i in ident), "max_change_pu": max((i["max_change"] or 0) for i in ident), "rows": ident}))
print(grid, "done", len(rows))
