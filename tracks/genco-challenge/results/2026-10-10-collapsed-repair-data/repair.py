"""Recover physical roots for the collapsed scenarios of GENCO's pretraining subgrids (UNAUDITED, 2026-10-10).

Reuses genco_challenge.relabel (acceptance logic untouched: residual < 1e-9, all |V| >= 0.5, two independent starts
agree to 1e-6, branch angle <= 90 deg). Pretraining raw format = datakit raw without runtime_data, scenario ids kept.

Phases per grid (multiprocessing, fork, <= 6 workers):
  0 sign-flip check (label with sign-only defect: |Vm| >= 0.5 everywhere, some Vm < 0)
  1 relabel.solve_scenario  (flat, DC, damped flat, shift-aware DC x4, repaired label)
  2 relabel.neighbor_pass   (neighbour roots, homotopy from 3 nearest healthy ids + 2 pool roots), as in relabel_grid
  3 extra effort (this script): homotopy from the 8 healthy scenarios nearest in (Ybus, Sbus), neighbour starts k=10,
    Levenberg-Marquardt from flat / shift-DC / repaired, and a load-scaling continuation from zero load
    (also records mu_max, the largest load fraction reached: evidence of no operating point when < 1).
Run: python repair.py --grids a b ... [--workers 6] [--limit N]   (env grid-operations-genco, PYTHONPATH=tracks/genco-challenge/src:core)
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SRC = REPO / "data/external/genco-pf-pretraining"
DST = REPO / "data/external/genco-pf-pretraining-corrected"
from genco_challenge import relabel as R
from genco_challenge.hybrid import Scenario, newton_raphson_pf
from genco_challenge.ladder_data import label_diagnostics, VM_MIN, N_PER_PARTITION

GRIDS = ["case4837_goc_idx_35", "case20758_epigrids_idx_49", "case9591_goc_idx_12", "case5658_epigrids_idx_8",
         "case4837_goc_idx_28", "case19402_goc_idx_47", "case10480_goc_idx_40", "case4837_goc_idx_37", "case10480_goc_idx_26"]
G: dict = {}   # fork-shared per-grid state


def mkscen(s):
    sc = Scenario(G["bus"][s], G["ybus"][s], G["br"][s])
    # PV/REF setpoints and the reference angle must not inherit a sign defect of the label
    sc.vm_set = np.where(sc.is_t["PV"] | sc.is_t["REF"], np.abs(sc.vm_label), 1.0)
    return sc


def brt(s):
    return G["br"][s].sort_values("idx").reset_index(drop=True)


# ----------------------------------------------------------------------------- extra tools
def lm_root(sc, v0, max_nfev=200):
    from scipy.optimize import least_squares
    from grid_ops.reference_pf import _sparse_jacobian
    pvpq, pq, k = sc.pvpq, sc.pq, len(sc.pvpq)
    va0, vm0 = np.angle(v0), np.abs(v0)

    def vof(x):
        va, vm = va0.copy(), vm0.copy()
        va[pvpq] = x[:k]; vm[pq] = x[k:]
        return vm * np.exp(1j * va)

    def fun(x):
        a, b = sc.f(vof(x)); return np.r_[a, b]

    def jac(x):
        return _sparse_jacobian(sc.ybus, vof(x), pvpq, pq).toarray()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            r = least_squares(fun, np.r_[va0[pvpq], vm0[pq]], jac=jac, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                              max_nfev=max_nfev)
        except Exception:
            return None
    v = vof(r.x)
    return v if np.all(np.isfinite(v)) else None


def load_continuation(sc, h0=0.1, hmin=1e-4, max_steps=500):
    """Scale the whole injection Sbus by mu from 0 (flat voltages are the exact root) to 1 at the scenario's own Ybus,
    PV/REF setpoints fixed. Returns (root or None, mu_max reached)."""
    fixed = sc.is_t["PV"] | sc.is_t["REF"]
    v = np.where(fixed, sc.vm_set, 1.0) * np.exp(1j * sc.va_ref)
    mu, h = 0.0, h0
    for _ in range(max_steps):
        nm = min(1.0, mu + h)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                r = newton_raphson_pf(sc.ybus, nm * sc.sbus, v, sc.pv, sc.pq, tol=1e-10, max_it=12)
            except Exception:
                r = None
        if r is not None and r.converged and np.all(np.isfinite(r.v)) and np.abs(r.v).min() > 0.3 and np.abs(r.v - v).max() < 0.5:
            v, mu = r.v, nm
            if mu >= 1.0:
                return v, 1.0
            h = min(h * 1.5, 0.25)
        else:
            h /= 2
            if h < hmin:
                return None, mu
    return None, mu


def hom_lam(sc, src, v_src, h0=0.1, hmin=1e-4, max_steps=400):
    """relabel.homotopy_root, but also returns the largest lambda reached (nose evidence)."""
    fixed = sc.is_t["PV"] | sc.is_t["REF"]
    vm0 = np.where(fixed, np.abs(v_src), 1.0)
    lam, h, v = 0.0, h0, np.where(fixed, vm0, np.abs(v_src)) * np.exp(1j * np.angle(v_src))
    for _ in range(max_steps):
        ln = min(1.0, lam + h)
        y = (1 - ln) * src.ybus + ln * sc.ybus
        sb = (1 - ln) * src.sbus + ln * sc.sbus
        vm_set = (1 - ln) * vm0 + ln * sc.vm_set
        v0 = np.where(fixed, vm_set, np.abs(v)) * np.exp(1j * np.angle(v))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                r = newton_raphson_pf(y, sb, v0, sc.pv, sc.pq, tol=1e-10, max_it=12)
            except Exception:
                r = None
        if r is not None and r.converged and np.all(np.isfinite(r.v)) and np.abs(r.v).min() > 0.3 and np.abs(r.v - v0).max() < 0.5:
            v, lam = r.v, ln
            if lam >= 1.0:
                return v, 1.0
            h = min(h * 1.5, 0.25)
        else:
            h /= 2
            if h < hmin:
                return None, lam
    return None, lam


def signflip_try(sc, ang_fn):
    """Label has |Vm| >= 0.5 everywhere and some Vm < 0. Candidates: complex label as written (-|V| at angle Va equals
    |V| at Va+180), and |Vm| with the written angle. Keep the first that satisfies the power-flow identity and plausibility."""
    out = []
    for name, v in (("flip_as_written", sc.v_label), ("flip_abs_magnitude", np.abs(sc.vm_label) * np.exp(1j * sc.va_label))):
        c = R.judge_candidate(name, v, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=ang_fn)
        out.append((name, v, c))
        if c["physical"]:
            return name, v, out
    return None, None, out


# ----------------------------------------------------------------------------- workers
def w_phase1(s):
    try:
        sc = mkscen(s); br = brt(s)
        info = {"scenario": s, "ang_fn_ok": True}
        res = None
        if G["reason"][s] == "negative_vm_sign_flip":
            name, v, tries = signflip_try(sc, R.branch_angle_fn(br))
            info["flip_tries"] = [(n, c.get("residual"), c.get("min_vm"), c["why"]) for n, _, c in tries]
            if name is not None:
                # independent confirmation: does Newton from flat reach the same root?
                vf = R._newton(sc, np.full(sc.n, sc.va_ref) * 0 + sc.vm_set * np.exp(1j * sc.va_ref))
                info["flip_newton_agrees"] = bool(vf is not None and np.abs(vf - v).max() < 1e-6)
                return {**info, "status": "recovered_signflip", "v": v, "pair": [name, "identity"], "cands": []}
        res = R.solve_scenario(sc, br)
        return {**info, "accepted": res["accepted"], "v": res["v"], "pair": res["pair"], "reason": res["reason"],
                "cands": [(n, v) for n, v in res["cands"]], "max_dv": res["max_dv"]}
    except Exception as e:  # noqa: BLE001
        return {"scenario": s, "error": repr(e), "accepted": False, "cands": [], "reason": "error"}


def w_phase2(args):
    s, cands, pool = args
    try:
        sc = mkscen(s); ang = R.branch_angle_fn(brt(s))
        srcs = []
        for j in sorted(G["healthy"], key=lambda j: abs(j - s))[:3]:
            sj = mkscen(j); srcs.append((j, sj, sj.v_label))
        for j in sorted((j for j in pool if j != s), key=lambda j: abs(j - s))[:2]:
            srcs.append((j, mkscen(j), pool[j]))
        res = R.neighbor_pass(sc, {"cands": list(cands), "ang_fn": ang}, pool, s, sources=srcs)
        return {"scenario": s, "accepted": res["accepted"], "v": res["v"], "pair": res["pair"], "reason": res["reason"],
                "cands": [(n, v) for n, v in res["cands"]], "max_dv": res["max_dv"]}
    except Exception as e:  # noqa: BLE001
        return {"scenario": s, "error": repr(e), "accepted": False, "cands": cands, "reason": "error"}


def nearest_healthy(sc, s, k=8):
    d = []
    for j in G["healthy"]:
        sj = G["scen_cache"].get(j)
        if sj is None:
            sj = G["scen_cache"][j] = mkscen(j)
        d.append((abs(sc.ybus - sj.ybus).sum() + 50 * np.abs(sc.sbus - sj.sbus).sum(), j))
    return [j for _, j in sorted(d)[:k]]


def w_phase3(args):
    s, cands, pool = args
    t0 = time.time()
    try:
        G["scen_cache"] = {}
        sc = mkscen(s); br = brt(s); ang = R.branch_angle_fn(br)
        cands = list(cands)
        info = {"scenario": s}
        lams = []
        for j in nearest_healthy(sc, s, 8):
            sj = G["scen_cache"][j]
            v, lam = hom_lam(sc, sj, sj.v_label)
            cands.append((f"homotopy_near{j}", v)); lams.append(lam)
        info["homotopy_lambda_max"] = max(lams) if lams else None
        fixed = sc.is_t["PV"] | sc.is_t["REF"]
        for j in sorted((j for j in pool if j != s), key=lambda j: abs(j - s))[:10]:
            v0 = np.where(fixed, sc.vm_set, np.abs(pool[j])) * np.exp(1j * np.angle(pool[j]))
            cands.append((f"neighborK{j}", R._newton(sc, v0)))
        v_lc, mu = load_continuation(sc)
        cands.append(("loadcont", v_lc)); info["mu_max"] = mu
        flat = sc.vm_set * np.exp(1j * sc.va_ref)
        th = R.shift_dc_angles(sc, br)
        sdc = sc.vm_set * np.exp(1j * th)
        rep = R._repaired_start(sc)
        cands.append(("lm_flat", lm_root(sc, flat)))
        cands.append(("lm_dc_shift", lm_root(sc, sdc)))
        if rep is not None:
            cands.append(("lm_repaired", lm_root(sc, rep)))
        res = R.accept_relabel(cands, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=ang)
        n_phys = sum(c["physical"] for c in res["candidates"])
        # distinct physical roots found
        phys = [v for (n, v), c in zip(cands, res["candidates"]) if c["physical"]]
        return {**info, "accepted": res["accepted"], "v": res["v"], "pair": res["pair"], "reason": res["reason"],
                "max_dv": res["max_dv"], "n_phys_cands": n_phys, "n_distinct_phys": _ndistinct(phys),
                "cand_summary": [{k: c.get(k) for k in ("start", "converged", "physical", "why", "residual", "min_vm")}
                                 for c in res["candidates"]], "secs": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"scenario": s, "error": repr(e), "accepted": False, "reason": "error"}


def _ndistinct(vs, tol=1e-6):
    reps = []
    for v in vs:
        if not any(np.abs(v - r).max() < tol for r in reps):
            reps.append(v)
    return len(reps)


def w_writeback(args):
    s, v = args
    sc = mkscen(s)
    b2, g2, br2, dg = R.new_label_tables(sc, v, G["bus"][s], G["gen"][s], G["br"][s])
    cc = R.check_consistency(sc, G["bus"][s], b2, g2)
    return s, b2, g2, br2, dg, cc


# ----------------------------------------------------------------------------- per grid
def run_grid(grid, workers, limit=None, skip_phase3=False):
    t0 = time.time()
    raw = SRC / f"subnet_{grid}" / "raw"
    bus_all = pd.read_parquet(raw / "bus_data.parquet"); gen_all = pd.read_parquet(raw / "gen_data.parquet")
    br_all = pd.read_parquet(raw / "branch_data.parquet"); yb_all = pd.read_parquet(raw / "y_bus_data.parquet")
    diag = label_diagnostics(bus_all[["scenario", "bus", "Vm", "PV", "REF"]])
    ids = diag.index.to_numpy()
    bad = [int(s) for s in ids[diag["invalid"].to_numpy()]]
    if limit:
        bad = bad[:limit]
    G.clear()
    G.update(bus=dict(tuple(bus_all.groupby("scenario"))), gen=dict(tuple(gen_all.groupby("scenario"))),
             br=dict(tuple(br_all.groupby("scenario"))), ybus=dict(tuple(yb_all.groupby("scenario"))),
             reason={int(s): diag.at[s, "reason"] for s in ids},
             healthy=[int(x) for x in ids if not diag.at[x, "invalid"]], scen_cache={})
    print(grid, "scenarios", len(ids), "flagged", len(bad), flush=True)
    rows = {}
    with Pool(workers) as p:
        p1 = {r["scenario"]: r for r in p.imap_unordered(w_phase1, bad, chunksize=1)}
        pool = {}
        for s, r in p1.items():
            if r.get("status") != "recovered_signflip" and r.get("cands"):
                v = _first_phys(s, r)
                if v is not None:
                    pool[s] = v
        print(grid, f"phase1 done {time.time()-t0:.0f}s accepted",
              sum(1 for r in p1.values() if r.get("accepted") or r.get("status") == "recovered_signflip"), flush=True)
        rows = {s: dict(r, phase=("0" if r.get("status") == "recovered_signflip" else "1")) for s, r in p1.items()
                if r.get("accepted") or r.get("status") == "recovered_signflip"}
        todo = [s for s in bad if s not in rows]
        p2 = {r["scenario"]: r for r in p.imap_unordered(w_phase2, [(s, p1[s]["cands"], pool) for s in todo], chunksize=1)}
        for s, r in p2.items():
            if r.get("accepted"):
                rows[s] = dict(r, phase="2")
        print(grid, f"phase2 done {time.time()-t0:.0f}s accepted {len(rows)} of {len(bad)}", flush=True)
        todo = [s for s in bad if s not in rows]
        p3 = {}
        if not skip_phase3:
            p3 = {r["scenario"]: r for r in p.imap_unordered(w_phase3, [(s, p2[s]["cands"], pool) for s in todo], chunksize=1)}
            for s, r in p3.items():
                if r.get("accepted"):
                    rows[s] = dict(r, phase="3")
        print(grid, f"phase3 done {time.time()-t0:.0f}s accepted {len(rows)} of {len(bad)}", flush=True)
        wb = {r[0]: r for r in p.imap_unordered(w_writeback, [(s, r["v"]) for s, r in rows.items()], chunksize=4)}
    # ---- status rows + patched tables
    recs, ok = [], {}
    for s in bad:
        base = {"grid": grid, "scenario": s, "load_scenario_idx": int(G["bus"][s]["load_scenario_idx"].iloc[0]),
                "label_reason": G["reason"][s], "label_min_vm": float(G["bus"][s]["Vm"].min()),
                "label_min_abs_vm": float(G["bus"][s]["Vm"].abs().min())}
        if s in rows:
            _, b2, g2, br2, dg, cc = wb[s]
            if cc["ok"]:
                ok[s] = (b2, g2, br2)
                r = rows[s]
                status = "recovered_signflip" if r.get("status") == "recovered_signflip" else "recovered"
                recs.append(base | {"status": status, "phase": r["phase"], "start_pair": "+".join(r["pair"]),
                                    "residual": float(R.mismatch_max(mkscen(s).ybus, mkscen(s).sbus, r["v"], mkscen(s).pv, mkscen(s).pq)),
                                    "min_vm_new": dg["min_vm"], "max_vm_new": dg["max_vm"],
                                    "max_branch_angle_deg": dg["max_branch_angle_diff_deg"],
                                    "max_dv_between_starts": r.get("max_dv"), "q_limit_violation_mvar": dg["max_q_limit_violation_mvar"],
                                    "slack_pg_mw_old": dg["slack_pg_mw_old"], "slack_pg_mw_new": dg["slack_pg_mw_new"],
                                    "flip_newton_agrees": r.get("flip_newton_agrees")})
                continue
            rr = {"reason": "writeback_inconsistent"}
        else:
            rr = p3.get(s) or p2.get(s) or {}
        recs.append(base | {"status": "no_root", "no_root_reason": rr.get("reason"), "mu_max": rr.get("mu_max"),
                            "homotopy_lambda_max": rr.get("homotopy_lambda_max"), "n_phys_cands": rr.get("n_phys_cands"),
                            "n_distinct_phys_roots": rr.get("n_distinct_phys"), "max_dv_between_starts": rr.get("max_dv"),
                            "error": rr.get("error"),
                            "cand_summary": json.dumps(rr.get("cand_summary"), default=float) if rr.get("cand_summary") else None})
    out = DST / f"subnet_{grid}" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    patch_write(raw, out, bus_all, gen_all, br_all, ok)
    df = pd.DataFrame(recs)
    df.to_csv(out / "relabeled.csv", index=False)
    df.to_csv(HERE / "per_grid" / f"{grid}.csv", index=False)
    print(grid, "DONE", df.status.value_counts().to_dict(), f"{time.time()-t0:.0f}s", flush=True)
    return df


_cache = {}
def mkscen_cached(s):
    return _cache.setdefault(s, mkscen(s))


def _first_phys(s, r):
    sc = mkscen(s)
    for n, v in r["cands"]:
        if R.judge_candidate(n, v, sc.ybus, sc.sbus, sc.pv, sc.pq)["physical"]:
            return v
    return None


def patch_write(raw, out, bus_all, gen_all, br_all, ok):
    def patch(df, key, cols, which):
        df = df.copy()
        grp = df.groupby("scenario").indices
        for s, t in ok.items():
            new = t[which].set_index(key)
            pos = grp[s]
            cur = df.iloc[pos][key].to_numpy()
            for c in cols:
                df.iloc[pos, df.columns.get_loc(c)] = new.loc[cur, c].to_numpy()
        return df
    tabs = {"bus_data": patch(bus_all, "bus", ["Vm", "Va", "Pg", "Qg"], 0),
            "gen_data": patch(gen_all, "idx", ["p_mw", "q_mvar"], 1),
            "branch_data": patch(br_all, "idx", ["pf", "qf", "pt", "qt"], 2)}
    for t, df in tabs.items():
        d = out / f"{t}.parquet"
        if d.exists():
            shutil.rmtree(d)
        df = df.drop(columns=["scenario_partition"], errors="ignore").copy()
        df["scenario_partition"] = (df["scenario"] // N_PER_PARTITION).astype("int64")
        df.to_parquet(d, partition_cols=["scenario_partition"], engine="pyarrow", index=False)
    d = out / "y_bus_data.parquet"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(raw / "y_bus_data.parquet", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=GRIDS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-phase3", action="store_true")
    a = ap.parse_args()
    (HERE / "per_grid").mkdir(exist_ok=True)
    for g in a.grids:
        run_grid(g, min(a.workers, 6), a.limit, a.no_phase3)


if __name__ == "__main__":
    main()
