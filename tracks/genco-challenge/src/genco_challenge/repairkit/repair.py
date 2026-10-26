"""``repairkit repair``: recover the physical root of flagged scenarios and write a full repaired copy of the raw dir.

Numerics are NOT new. The phase workers (0 sign-flip identity check, 1 ``relabel.solve_scenario``, 2 ``relabel.neighbor_pass``,
3 extra effort: homotopy from the nearest healthy scenarios, neighbour starts, Levenberg-Marquardt, load continuation) are
the ones of ``results/2026-10-10-collapsed-repair-data/repair.py``, copied with only these changes: they read one
dataset-wide state built by ``_load_state`` instead of a whole-subgrid load, and the donor pool of healthy scenarios is the 16
healthy ids nearest to each flagged id (phase 2 uses the 3 nearest, so it is unchanged; phase 3 picks its 8 nearest in
(Ybus, Sbus) from that pool instead of from every healthy scenario). Acceptance is ``relabel.accept_relabel`` untouched:
residual < 1e-9, every |V| >= 0.5, two starts from different initial points agree to 1e-6, max branch angle <= 90 degrees.
"""
from __future__ import annotations

import json
import shutil
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from genco_challenge import relabel as R
from genco_challenge.hybrid import Scenario, newton_raphson_pf

from . import _io
from .check import DEFECTS, check

G: dict = {}   # fork-shared state: bus/gen/br/ybus by scenario id, reason, healthy donor ids
DONORS_PER_FLAGGED = 16
PATCH = {"bus_data": ("bus", ["Vm", "Va", "Pg", "Qg"], 0), "gen_data": ("idx", ["p_mw", "q_mvar"], 1),
         "branch_data": ("idx", ["pf", "qf", "pt", "qt"], 2)}
ALIASES = {"collapsed": "collapsed", "negative": "negative_only", "negative_only": "negative_only", "sign_flip": "negative_only"}


# ------------------------------------------------------------------ copied from collapsed-repair-data/repair.py
def mkscen(s):
    sc = Scenario(G["bus"][s], G["ybus"][s], G["br"][s])
    # PV/REF setpoints and the reference angle must not inherit a sign defect of the label
    sc.vm_set = np.where(sc.is_t["PV"] | sc.is_t["REF"], np.abs(sc.vm_label), 1.0)
    return sc


def brt(s):
    return G["br"][s].sort_values("idx").reset_index(drop=True)


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
    out = []
    for name, v in (("flip_as_written", sc.v_label), ("flip_abs_magnitude", np.abs(sc.vm_label) * np.exp(1j * sc.va_label))):
        c = R.judge_candidate(name, v, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=ang_fn)
        out.append((name, v, c))
        if c["physical"]:
            return name, v, out
    return None, None, out


def w_phase1(s):
    try:
        sc = mkscen(s); br = brt(s)
        info = {"scenario": s, "ang_fn_ok": True}
        if G["reason"][s] == "negative_only":
            name, v, tries = signflip_try(sc, R.branch_angle_fn(br))
            info["flip_tries"] = [(n, c.get("residual"), c.get("min_vm"), c["why"]) for n, _, c in tries]
            if name is not None:
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


def _ndistinct(vs, tol=1e-6):
    reps = []
    for v in vs:
        if not any(np.abs(v - r).max() < tol for r in reps):
            reps.append(v)
    return len(reps)


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
        phys = [v for (n, v), c in zip(cands, res["candidates"]) if c["physical"]]
        return {**info, "accepted": res["accepted"], "v": res["v"], "pair": res["pair"], "reason": res["reason"],
                "max_dv": res["max_dv"], "n_phys_cands": n_phys, "n_distinct_phys": _ndistinct(phys), "secs": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"scenario": s, "error": repr(e), "accepted": False, "reason": "error"}


def w_writeback(args):
    s, v = args
    sc = mkscen(s)
    b2, g2, br2, dg = R.new_label_tables(sc, v, G["bus"][s], G["gen"][s], G["br"][s])
    cc = R.check_consistency(sc, G["bus"][s], b2, g2)
    return s, b2, g2, br2, dg, cc


def _first_phys(s, r):
    sc = mkscen(s)
    for n, v in r["cands"]:
        if R.judge_candidate(n, v, sc.ybus, sc.sbus, sc.pv, sc.pq)["physical"]:
            return v
    return None


# ------------------------------------------------------------------ orchestration
def parse_categories(text: str) -> list[str]:
    out = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok not in ALIASES:
            raise SystemExit(f"unknown category {tok!r}; use collapsed and/or negative")
        out.append(ALIASES[tok])
    return sorted(set(out))


def _load_state(raw: Path, rep: pd.DataFrame, flagged: list[int]) -> None:
    ok = rep.loc[~rep["category"].isin(DEFECTS), "scenario"].astype(int).to_numpy()
    donors: set[int] = set()
    for s in flagged:
        donors.update(int(x) for x in ok[np.argsort(np.abs(ok - s), kind="stable")[:DONORS_PER_FLAGGED]])
    ids = sorted(set(flagged) | donors)
    G.clear()
    t = {name: _io.read_scenarios(raw, name, ids) for name in ("bus_data", "gen_data", "branch_data", "y_bus_data")}
    G.update(bus=dict(tuple(t["bus_data"].groupby("scenario"))), gen=dict(tuple(t["gen_data"].groupby("scenario"))),
             br=dict(tuple(t["branch_data"].groupby("scenario"))), ybus=dict(tuple(t["y_bus_data"].groupby("scenario"))),
             reason=dict(zip(rep["scenario"].astype(int), rep["category"])), healthy=sorted(donors), scen_cache={})


def solve_flagged(flagged: list[int], workers: int, log=print) -> tuple[dict, dict, dict]:
    """Phases 0 to 3 on G. Returns (accepted rows by scenario, last result of each unresolved scenario, written tables)."""
    t0 = time.time()
    rows: dict = {}
    with Pool(max(1, workers)) as p:
        p1 = {r["scenario"]: r for r in p.imap_unordered(w_phase1, flagged, chunksize=1)}
        pool = {}
        for s, r in p1.items():
            if r.get("status") != "recovered_signflip" and r.get("cands"):
                v = _first_phys(s, r)
                if v is not None:
                    pool[s] = v
        rows = {s: dict(r, phase=("0" if r.get("status") == "recovered_signflip" else "1")) for s, r in p1.items()
                if r.get("accepted") or r.get("status") == "recovered_signflip"}
        log(f"phase 0/1 done {time.time()-t0:.0f}s, accepted {len(rows)} of {len(flagged)}")
        todo = [s for s in flagged if s not in rows]
        p2 = {r["scenario"]: r for r in p.imap_unordered(w_phase2, [(s, p1[s]["cands"], pool) for s in todo], chunksize=1)}
        for s, r in p2.items():
            if r.get("accepted"):
                rows[s] = dict(r, phase="2")
        log(f"phase 2 done {time.time()-t0:.0f}s, accepted {len(rows)} of {len(flagged)}")
        todo = [s for s in flagged if s not in rows]
        p3 = {r["scenario"]: r for r in p.imap_unordered(w_phase3, [(s, p2[s]["cands"], pool) for s in todo], chunksize=1)}
        for s, r in p3.items():
            if r.get("accepted"):
                rows[s] = dict(r, phase="3")
        log(f"phase 3 done {time.time()-t0:.0f}s, accepted {len(rows)} of {len(flagged)}")
        wb = {r[0]: r for r in p.imap_unordered(w_writeback, [(s, r["v"]) for s, r in rows.items()], chunksize=4)}
    last = {s: (p3.get(s) or p2.get(s) or p1.get(s) or {}) for s in flagged if s not in rows}
    return rows, last, wb


def patch_frame(df: pd.DataFrame, table: str, ok: dict) -> pd.DataFrame:
    """Overwrite the label columns of the scenarios in ``ok`` (scenario -> (bus, gen, branch) tables); other cells untouched."""
    key, cols, which = PATCH[table]
    df = df.copy()
    grp = df.groupby("scenario").indices
    arrs = {c: df[c].to_numpy(copy=True) for c in cols}
    keys = df[key].to_numpy()
    for s, t in ok.items():
        if s not in grp:
            continue
        pos = grp[s]
        new = t[which].set_index(key)
        for c in cols:
            arrs[c][pos] = new.loc[keys[pos], c].to_numpy()
    for c in cols:
        df[c] = arrs[c]
    return df


def write_repaired(raw: Path, out: Path, ok: dict) -> None:
    """Full copy of ``raw`` into ``out``; partitions (or tables) that contain a repaired scenario are rewritten, the rest copied."""
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(raw.iterdir()):
        name = entry.name.removesuffix(".parquet")
        if name not in _io.CORE:
            _io.copy_entry(entry, out / entry.name)
    for t in _io.CORE:
        src, dst = _io.tpath(raw, t), _io.tpath(out, t)
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        if _io.is_hive(raw, t):
            for leaf in sorted(c for c in src.iterdir() if c.is_dir()):
                files = _io.leaf_files(leaf)
                df = pd.read_parquet(leaf) if ok else None
                if df is not None and "scenario" in df and set(map(int, df["scenario"].unique())) & set(ok):
                    new = patch_frame(df, t, ok)
                    _io.write_like(new, dst / leaf.name / files[0].name, files[0])
                else:
                    _io.copy_entry(leaf, dst / leaf.name)
        else:
            df = pd.read_parquet(src)
            if set(map(int, df["scenario"].unique())) & set(ok):
                _io.write_like(patch_frame(df, t, ok), dst, src if src.is_file() else None)
            else:
                _io.copy_entry(src, dst)


def run(raw: Path, out: Path, workers: int = 4, categories: str = "collapsed,negative", log=print) -> pd.DataFrame:
    raw, out = Path(raw), Path(out)
    if out.resolve() == raw.resolve():
        raise SystemExit("out_dir must differ from raw_dir")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    if not _io.has_table(raw, "y_bus_data"):
        raise SystemExit(f"{raw}: y_bus_data.parquet is required to repair (the solver needs each scenario's Ybus)")
    cats = parse_categories(categories)
    t0 = time.time()
    rep = check(raw, workers)
    flagged = [int(s) for s in rep.loc[rep["category"].isin(cats), "scenario"]]
    log(f"{raw}: {len(rep)} scenarios, flagged for repair ({'+'.join(cats)}): {len(flagged)}")
    ok: dict = {}
    recs = []
    if flagged:
        _load_state(raw, rep, flagged)
        rows, last, wb = solve_flagged(flagged, workers, log)
        by = rep.set_index("scenario")
        for s in flagged:
            base = {"scenario": s, "category": by.at[s, "category"], "min_vm_before": float(G["bus"][s]["Vm"].min())}
            if s in rows:
                _, b2, g2, br2, dg, cc = wb[s]
                if cc["ok"]:
                    ok[s] = (b2, g2, br2)
                    r, sc = rows[s], mkscen(s)
                    recs.append(base | {"status": "recovered", "phase": r["phase"], "start_pair": "+".join(r["pair"]),
                                        "residual": float(R.mismatch_max(sc.ybus, sc.sbus, r["v"], sc.pv, sc.pq)),
                                        "max_dv_between_starts": r.get("max_dv"), "max_branch_angle_deg": dg["max_branch_angle_diff_deg"],
                                        "min_vm_after": dg["min_vm"], "q_limit_violation_mvar": dg["max_q_limit_violation_mvar"],
                                        "flip_newton_agrees": r.get("flip_newton_agrees"), "unresolved_reason": None})
                    continue
                rr = {"reason": "writeback_inconsistent"}
            else:
                rr = last.get(s, {})
            recs.append(base | {"status": "unresolved", "unresolved_reason": rr.get("error") or rr.get("reason"),
                                "max_dv_between_starts": rr.get("max_dv"), "n_phys_cands": rr.get("n_phys_cands"),
                                "n_distinct_phys_roots": rr.get("n_distinct_phys"), "mu_max": rr.get("mu_max"),
                                "homotopy_lambda_max": rr.get("homotopy_lambda_max")})
        G.clear()
    log(f"recovered {len(ok)}, unresolved {len(flagged) - len(ok)}; writing {out}")
    if ok:   # originals of every changed scenario, all tables, before anything is overwritten
        od = out / "originals.parquet"
        od.mkdir(parents=True, exist_ok=True)
        for t in _io.TABLES:
            if _io.has_table(raw, t):
                _io.read_scenarios(raw, t, sorted(ok)).to_parquet(od / f"{t}.parquet", index=False)
    write_repaired(raw, out, ok)
    cols = ["scenario", "category", "status", "phase", "start_pair", "residual", "max_dv_between_starts", "max_branch_angle_deg",
            "min_vm_before", "min_vm_after"]
    df = pd.DataFrame(recs)
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(out / "repair_log.csv", index=False)
    rep.to_csv(out / "check_report_before.csv", index=False)
    (out / "repair_summary.json").write_text(json.dumps({"flagged": len(flagged), "recovered": len(ok),
                                                          "unresolved": len(flagged) - len(ok), "categories": cats,
                                                          "seconds": round(time.time() - t0, 1)}, indent=1))
    log(f"done in {time.time()-t0:.0f}s")
    return df
