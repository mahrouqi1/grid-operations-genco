"""Question 1: what does the pretrained GENCO predict on collapsed power-flow scenarios? UNAUDITED.

For a data dir (bus_data, y_bus_data) and a graphkit predictions parquet:
  * collapsed scenario = label has some bus with Vm < 0.5 pu (or negative); L = those buses;
  * physical root = our Newton (core/grid_ops/reference_pf.newton_raphson_pf, sparse) on the scenario's OWN Ybus
    (y_bus_data) and Sbus = (Pg - Pd + j(Qg - Qd))/100, from flat then DC start, tol 1e-8, max_it 30; accepted only if
    converged, finite, every Vm >= 0.5 and an independent power-mismatch check (P at PV+PQ, Q at PQ) < 1e-8 pu;
  * compare GENCO's predicted Vm at L to the collapsed label and to the physical root.
Run (env grid-operations-genco, repo root): python tracks/genco-challenge/results/2026-10-08-collapsed-label-genco/analyze.py
"""
from __future__ import annotations
import glob, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "core"))
from grid_ops.reference_pf import injected_power, newton_raphson_pf  # noqa: E402

HERE = Path(__file__).resolve().parent
VM_MIN, TOL, MAXIT = 0.5, 1e-8, 30
LAB = REPO / "data/external/genco-ladder/collapsed-label"
EVAL = REPO / "data/external/genco-pf-pretraining-eval"
LADDER = REPO / "data/external/genco-ladder"


def physical_root(b: pd.DataFrame, y: pd.DataFrame):
    """(v, start, residual) of a physical Newton root, or (None, None, residual-of-best-try)."""
    b = b.sort_values("bus")
    n = len(b)
    ybus = sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(), (y["index1"].to_numpy(), y["index2"].to_numpy())), shape=(n, n))
    sbus = (b["Pg"].to_numpy() - b["Pd"].to_numpy() + 1j * (b["Qg"].to_numpy() - b["Qd"].to_numpy())) / 100.0
    pv, pq, ref = (np.where(b[t].to_numpy() == 1)[0] for t in ("PV", "PQ", "REF"))
    vm0 = np.where((b["PV"].to_numpy() == 1) | (b["REF"].to_numpy() == 1), b["Vm"].to_numpy(), 1.0)
    va_ref = np.radians(b["Va"].to_numpy()[ref].mean())
    starts = {"flat": np.full(n, va_ref), "dc": np.radians(b["Va_dc"].to_numpy())}
    pvpq = np.r_[pv, pq]
    best = np.inf
    for name, va0 in starts.items():
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                r = newton_raphson_pf(ybus, sbus, vm0 * np.exp(1j * va0), pv, pq, tol=TOL, max_it=MAXIT)
            except Exception:
                continue
        if not (r.converged and np.all(np.isfinite(r.v))):
            continue
        ds = injected_power(ybus, r.v) - sbus
        res = max(np.abs(ds[pvpq].real).max(), np.abs(ds[pq].imag).max() if len(pq) else 0.0)
        best = min(best, res)
        if res < TOL and np.abs(r.v).min() >= VM_MIN:
            return r.v, name, float(res)
    return None, None, float(best)


def label_residual(b, y):
    """Power mismatch of the (collapsed) label under the scenario's own Ybus/Sbus: it IS a root of the equations."""
    b = b.sort_values("bus")
    n = len(b)
    ybus = sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(), (y["index1"].to_numpy(), y["index2"].to_numpy())), shape=(n, n))
    sbus = (b["Pg"].to_numpy() - b["Pd"].to_numpy() + 1j * (b["Qg"].to_numpy() - b["Qd"].to_numpy())) / 100.0
    v = b["Vm"].to_numpy() * np.exp(1j * np.radians(b["Va"].to_numpy()))
    pv, pq = (np.where(b[t].to_numpy() == 1)[0] for t in ("PV", "PQ"))
    ds = injected_power(ybus, v) - sbus
    return float(max(np.abs(ds[np.r_[pv, pq]].real).max(), np.abs(ds[pq].imag).max()))


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else None


def analyze(name: str, raw: Path, pred_path: Path, max_collapsed: int | None = None) -> tuple[dict, pd.DataFrame]:
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF", "Va_dc"])
    ybus = pd.read_parquet(raw / "y_bus_data.parquet", columns=["scenario", "index1", "index2", "G", "B"])
    pred = pd.read_parquet(pred_path, columns=["scenario", "bus", "Vm_target", "Vm_pred"])
    j = pred.merge(bus[["scenario", "bus", "Vm", "PQ", "Pd"]], on=["scenario", "bus"], how="left")
    gap = float((j["Vm_target"] - j["Vm"]).abs().max())
    if not np.isfinite(gap) or gap > 1e-4:
        raise RuntimeError(f"{name}: predictions' Vm_target differ from raw Vm by {gap}: wrong data for these predictions")
    scen = np.sort(pred["scenario"].unique())
    j = j.sort_values(["scenario", "bus"])
    g = j.groupby("scenario")
    lab_min = g["Vm"].min()
    collapsed = lab_min[lab_min < VM_MIN].index.to_numpy()
    normal = lab_min[lab_min >= VM_MIN].index.to_numpy()
    if max_collapsed and len(collapsed) > max_collapsed:
        collapsed = collapsed[np.linspace(0, len(collapsed) - 1, max_collapsed).astype(int)]
    bg, yg = dict(tuple(bus.groupby("scenario"))), dict(tuple(ybus.groupby("scenario")))
    jg = dict(tuple(j.groupby("scenario")))
    rows = []
    for s in collapsed:
        d = jg[s].sort_values("bus")
        v, start, res = physical_root(bg[s], yg[s])
        low = (d["Vm"] < VM_MIN).to_numpy()
        vl, vp = d["Vm"].to_numpy(), d["Vm_pred"].to_numpy()
        r = {"scenario": int(s), "n_low": int(low.sum()), "label_min_vm": float(vl.min()), "label_resid_pu": label_residual(bg[s], yg[s]),
             "pred_min_vm_all": float(vp.min()), "pred_min_vm_at_low": float(vp[low].min()), "pred_med_vm_at_low": float(np.median(vp[low])),
             "label_med_vm_at_low": float(np.median(vl[low])), "frac_low_pred_lt_0.5": float((vp[low] < VM_MIN).mean()),
             "frac_low_pred_lt_0": float((vp[low] < 0).mean()),
             "mae_vs_label_at_low": float(np.abs(vp[low] - vl[low]).mean()), "mae_vs_label_other": float(np.abs(vp[~low] - vl[~low]).mean()),
             "root_found": v is not None, "root_start": start, "root_best_resid": res}
        if v is not None:
            vr = np.abs(v)
            r |= {"root_min_vm": float(vr.min()), "root_med_vm_at_low": float(np.median(vr[low])),
                  "mae_vs_root_at_low": float(np.abs(vp[low] - vr[low]).mean()), "mae_vs_root_other": float(np.abs(vp[~low] - vr[~low]).mean()),
                  "label_vs_root_mae_at_low": float(np.abs(vl[low] - vr[low]).mean()),
                  "pred_closer_to_root_than_label_at_low": bool(np.abs(vp[low] - vr[low]).mean() < np.abs(vp[low] - vl[low]).mean())}
        rows.append(r)
    cs = pd.DataFrame(rows)
    # matched normal scenarios: same data, label min Vm >= 0.5
    nm = []
    for s in normal:
        d = jg[s]
        pq = d["PQ"].to_numpy() == 1
        vp, vl = d["Vm_pred"].to_numpy(), d["Vm"].to_numpy()
        nm.append((vp.min(), vl.min(), np.abs(vp[pq] - vl[pq]).mean(), np.abs(vp - vl).mean()))
    nm = np.array(nm) if nm else np.zeros((0, 4))
    out = {"grid": name, "n_pred_scenarios": int(len(scen)), "n_collapsed": int(len(lab_min[lab_min < VM_MIN])), "n_collapsed_analyzed": int(len(cs)),
           "n_normal": int(len(normal))}
    if len(cs):
        ok = cs[cs["root_found"]]
        out |= {"collapsed": {
            "root_found": int(cs["root_found"].sum()),
            "label_resid_pu_max": float(cs["label_resid_pu"].max()),
            "label_min_vm_median": q(cs["label_min_vm"], .5), "label_min_vm_min": float(cs["label_min_vm"].min()),
            "label_med_vm_at_low_median": q(cs["label_med_vm_at_low"], .5),
            "pred_min_vm_at_low": {"min": float(cs["pred_min_vm_at_low"].min()), "median": q(cs["pred_min_vm_at_low"], .5), "max": float(cs["pred_min_vm_at_low"].max())},
            "pred_med_vm_at_low_median": q(cs["pred_med_vm_at_low"], .5),
            "pred_min_vm_all_median": q(cs["pred_min_vm_all"], .5),
            "frac_scen_pred_min_at_low_lt_0.5": float((cs["pred_min_vm_at_low"] < VM_MIN).mean()),
            "frac_scen_pred_min_at_low_lt_0": float((cs["pred_min_vm_at_low"] < 0).mean()),
            "frac_low_buses_pred_lt_0.5": float((cs["frac_low_pred_lt_0.5"] * cs["n_low"]).sum() / cs["n_low"].sum()),
            "mae_vs_label_at_low_median": q(cs["mae_vs_label_at_low"], .5),
            "mae_vs_label_other_median": q(cs["mae_vs_label_other"], .5),
            "root": {"n": int(len(ok)), "root_min_vm_median": q(ok["root_min_vm"], .5) if len(ok) else None,
                     "root_med_vm_at_low_median": q(ok["root_med_vm_at_low"], .5) if len(ok) else None,
                     "mae_vs_root_at_low_median": q(ok["mae_vs_root_at_low"], .5) if len(ok) else None,
                     "mae_vs_root_other_median": q(ok["mae_vs_root_other"], .5) if len(ok) else None,
                     "label_vs_root_mae_at_low_median": q(ok["label_vs_root_mae_at_low"], .5) if len(ok) else None,
                     "frac_pred_closer_to_root_than_label": float(ok["pred_closer_to_root_than_label_at_low"].mean()) if len(ok) else None}}}
    if len(nm):
        out["normal"] = {"pred_min_vm_median": float(np.median(nm[:, 0])), "pred_min_vm_min": float(nm[:, 0].min()),
                         "label_min_vm_median": float(np.median(nm[:, 1])),
                         "frac_scen_pred_min_lt_0.5": float((nm[:, 0] < VM_MIN).mean()), "frac_scen_pred_min_lt_0": float((nm[:, 0] < 0).mean()),
                         "mae_vs_label_PQ_median": float(np.median(nm[:, 2])), "mae_vs_label_all_median": float(np.median(nm[:, 3]))}
    return out, cs


def latest_pred(mlruns: Path, model: str, grid: str, before: float | None = None) -> Path:
    c = []
    for tag in glob.glob(str(mlruns / "*/*/tags/mlflow.runName")):
        if Path(tag).read_text().strip() == f"{model}__{grid}":
            f = Path(tag).parent.parent / "artifacts/test" / f"{grid}_predictions.parquet"
            if f.exists() and (before is None or f.stat().st_mtime < before):
                c.append((f.stat().st_mtime, f))
    return max(c)[1]


def main():
    which = sys.argv[1]
    res, per = {}, {}
    if which == "pretrain":
        grids = sorted(p.name for p in EVAL.glob("subnet_*"))
        for g in grids:
            try:
                pp = latest_pred(LAB / "mlruns", "zeroshot", g)
            except ValueError:
                print("no predictions yet", g); continue
            o, cs = analyze(g, EVAL / g / "raw", pp)
            res[g] = o; per[g] = cs
            cs.insert(0, "grid", g)
            print(json.dumps({k: o[k] for k in ("grid", "n_pred_scenarios", "n_collapsed")}), flush=True)
        (HERE / "results_pretrain.json").write_text(json.dumps(res, indent=1))
        pd.concat(per.values()).to_csv(HERE / "per_scenario_pretrain_collapsed.csv", index=False)
    else:  # ladder, pre-validation predictions on raw.unvalidated-*
        import time, datetime
        for g in ("pglib_opf_case1354_pegase", "pglib_opf_case240_pserc", "pglib_opf_case1888_rte", "pglib_opf_case1951_rte", "pglib_opf_case2736sp_k"):
            un = sorted((LADDER / "data" / g).glob("raw.unvalidated-*"))[0]
            stamp = datetime.datetime.strptime(un.name.split("-", 1)[1], "%Y%m%d-%H%M%S").timestamp()
            if g == "pglib_opf_case2736sp_k":
                pp, src = latest_pred(LAB / "mlruns_ladder", "zeroshot", g), "re-run 2026-10-08 on raw.unvalidated"
            else:
                pp, src = latest_pred(LADDER / "mlruns", "zeroshot", g, before=stamp), "saved before validation"
            o, cs = analyze(g, un, pp)
            o |= {"unvalidated_raw": un.name, "predictions": str(pp.relative_to(REPO)), "predictions_source": src,
                  "predictions_mtime": datetime.datetime.fromtimestamp(pp.stat().st_mtime).isoformat(timespec="seconds")}
            res[g] = o
            cs.insert(0, "grid", g); per[g] = cs
            print(g, src, flush=True)
        (HERE / "results_ladder.json").write_text(json.dumps(res, indent=1))
        pd.concat(per.values()).to_csv(HERE / "per_scenario_ladder_collapsed.csv", index=False)


if __name__ == "__main__":
    main()
