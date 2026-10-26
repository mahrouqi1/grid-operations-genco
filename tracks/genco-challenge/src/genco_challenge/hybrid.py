"""GENCO as a power-flow initializer: k-step hybrid (gate G1-2c) and warm start (gate G1-3).

For every held-out grid with GENCO predictions and every test scenario, this script:

1. **Identity check.** Builds the scenario's own Ybus (topology and admittances are perturbed per
   scenario, so one Ybus per scenario from ``y_bus_data``) and the specified injections (P, Q at PQ;
   P at PV; Vm at PV/REF; REF angle), and measures the power mismatch of the AC label under them.
   Rows checked are exactly the equations a power flow solves (P at PV+PQ, Q at PQ). ``y_bus_data``
   already contains the bus shunts (adding ``GS``/``BS`` again breaks the identity), and bus
   indices are 0-based and contiguous.
2. **Starts.** ``flat`` (Vm setpoints at PV/REF, 1.0 at PQ, Va = REF angle), ``dc`` (datakit's
   ``Va_dc``, Vm as flat), and per GENCO checkpoint: ``raw`` (prediction as emitted), ``pinned``
   (REF Vm/Va and PV Vm put back from the specification; uses only inputs) and ``vmdc``
   (GENCO-pinned Vm with DC angles) and ``vmflat`` (GENCO-pinned Vm, all angles = REF angle): both
   added because GENCO's angles, not its magnitudes, are what break Newton from a GENCO start;
   ``clip`` / ``clipdc`` (``pinned`` / ``vmdc`` with PQ Vm clipped into the case's own
   ``min_vm_pu``..``max_vm_pu`` limits), added because on case89 GENCO's PQ magnitudes go
   out of band and break Newton even with DC angles. A checkpoint whose prediction targets do not
   match ``bus_data`` (max |Vm_target - Vm|, |Va_target - Va| > 1e-5) is EXCLUDED and recorded:
   its predictions were made on a different dataset than the labels here.
3. **Hybrid.** k = 0, 1, 2 iterations of Newton-Raphson (``nr``, the repo's
   ``core/grid_ops/reference_pf.newton_raphson_pf``, one call with ``max_it=k``), of a damped Newton
   (``nr_ls``: same Newton direction, backtracking on the 2-norm of the mismatch) and of
   fast-decoupled XB (``fdxb``: B' from 1/x only, B'' = -Im(Ybus) on PQ rows, so B'' keeps the
   phase-shift effect of the exact Ybus; PYPOWER's XB zeroes shifts in B'', which matters only on
   grids with phase shifters, here case89). Scored: max/mean power mismatch (MW/MVAr at baseMVA
   100) and Vm/Va errors vs the label, by bus type.
4. **Warm start.** Each solver (plus ``fd2nr``: two XB iterations, then Newton; its iteration
   count includes the two XB iterations) to convergence (tol 1e-8 pu max-norm, max_it 30): converged,
   iterations, wall time, and right root (max |V - V_ref| < 1e-4 pu) vs wrong root vs diverged.
5. **Reference root (v4, 2026-10-07).** ``V_ref`` is the label, EXCEPT when the label is a
   degenerate root: some bus has Vm < 0.5 pu (datakit's PF labels on case1354_pegase, case240_pserc,
   case1888_rte, case1951_rte carry Vm ~ 0 or NEGATIVE at zero-injection buses; such a point
   satisfies the power-flow equations, so the identity check passes, but it is a collapsed,
   non-physical root). Then ``V_ref`` is the root Newton reaches from the label with the collapsed
   buses' voltages replaced by their neighbours' mean ("repaired"), accepted only if it converges
   with every Vm >= 0.5 pu; otherwise the scenario has no reference ("none") and is left out of
   the right-root fraction and the Vm/Va error columns (mismatch metrics need no reference).

Outputs per grid (resumable: a grid is redone only when new checkpoints land or ``--force``):
``per_grid/<grid>.json`` (aggregates), ``per_grid/<grid>_hybrid.parquet`` and
``per_grid/<grid>_warm.parquet`` (per-scenario rows), plus ``summary.md`` over all grids.

Run (env ``grid-operations-genco``, from the repo root, PYTHONPATH = tracks/genco-challenge/src):

    python -m genco_challenge.hybrid --out tracks/genco-challenge/results/2026-10-07-genco-hybrid-warmstart
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "core"))
from grid_ops.reference_pf import injected_power, newton_raphson_pf  # noqa: E402

WORK = REPO / "data/external/genco-ladder"
BASE_MVA = 100.0
TYPES = ("PQ", "PV", "REF")
TOL = 1e-8
MAX_IT = 30
ROOT_TOL = 1e-4
LABEL_VM_MIN = 0.5  # a label with any bus below this (or negative) is a collapsed root, not an operating point
KS = (0, 1, 2)
VERSION = 4  # bump when starts/methods change; older per-grid results are recomputed


# ---------------------------------------------------------------------------- data

def prediction_files() -> dict[str, dict[str, Path]]:
    """{grid: {model: predictions.parquet}} from the mlruns store (latest run per name)."""
    out: dict[str, dict[str, tuple[float, Path]]] = {}
    for tag in glob.glob(str(WORK / "mlruns/*/*/tags/mlflow.runName")):
        name = Path(tag).read_text().strip()
        if "__" not in name:
            continue
        model, grid = name.split("__", 1)
        run = Path(tag).parent.parent
        f = run / "artifacts/test" / f"{grid}_predictions.parquet"
        if not f.exists():
            continue
        mt = f.stat().st_mtime
        if mt > out.setdefault(grid, {}).get(model, (-1.0, f))[0]:
            out[grid][model] = (mt, f)
    return {g: {m: p for m, (_, p) in d.items()} for g, d in out.items()}


def run_stamp(grid: str, preds: dict[str, Path]) -> dict:
    """What a grid's per-grid result was computed from: the label version (``ladder_eval.data_stamp``) and
    each prediction file's mtime. Stored in ``per_grid/<grid>.json``; a mismatch with the current state
    means the result is stale and the grid is re-run (audit genco-eval R2-F7; ladder_eval got the same
    check for its results rows in R1-F5)."""
    from genco_challenge.ladder_eval import data_stamp
    return {"data_stamp": data_stamp(grid), "pred_mtimes": {m: p.stat().st_mtime for m, p in sorted(preds.items())}}


def is_current(prev: dict, grid: str, preds: dict[str, Path]) -> bool:
    """True when ``prev`` (a per_grid json) carries a stamp equal to the grid's current one. A result with
    no stamp (written before R2-F7) is not current."""
    now = run_stamp(grid, preds)
    return prev.get("data_stamp") == now["data_stamp"] and prev.get("pred_mtimes") == now["pred_mtimes"]


class Scenario:
    """One scenario's power-flow problem in per unit, 0-based bus order."""

    def __init__(self, b: pd.DataFrame, y: pd.DataFrame, br: pd.DataFrame):
        b = b.sort_values("bus")
        n = len(b)
        if not (b["bus"].to_numpy() == np.arange(n)).all():
            raise ValueError("bus indices are not 0..n-1")
        self.n = n
        self.ybus = sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(),
                                   (y["index1"].to_numpy(), y["index2"].to_numpy())), shape=(n, n))
        self.sbus = (b["Pg"].to_numpy() - b["Pd"].to_numpy()
                     + 1j * (b["Qg"].to_numpy() - b["Qd"].to_numpy())) / BASE_MVA
        self.is_t = {t: b[t].to_numpy() == 1 for t in TYPES}
        self.pv = np.where(self.is_t["PV"])[0]
        self.pq = np.where(self.is_t["PQ"])[0]
        self.ref = np.where(self.is_t["REF"])[0]
        self.vm_label = b["Vm"].to_numpy()
        self.va_label = np.radians(b["Va"].to_numpy())
        self.v_label = self.vm_label * np.exp(1j * self.va_label)
        self.va_dc = np.radians(b["Va_dc"].to_numpy())
        # specified quantities: Vm at PV/REF, angle at REF (these are inputs of the problem)
        self.vm_set = np.where(self.is_t["PV"] | self.is_t["REF"], self.vm_label, 1.0)
        self.va_ref = float(self.va_label[self.ref].mean())
        self.vm_lo, self.vm_hi = b["min_vm_pu"].to_numpy(), b["max_vm_pu"].to_numpy()
        # B' for XB fast-decoupled: series 1/x only, no charging, no shunts, tap 1, no shift
        on = br[br["br_status"] == 1]
        f, t = on["from_bus"].to_numpy(), on["to_bus"].to_numpy()
        bx = 1.0 / on["x"].to_numpy()
        bp = sp.coo_matrix((np.r_[bx, bx, -bx, -bx], (np.r_[f, t, f, t], np.r_[f, t, t, f])),
                           shape=(n, n)).tocsr()
        pvpq = np.r_[self.pv, self.pq]
        self.pvpq = pvpq
        self._bp = bp[pvpq, :][:, pvpq].tocsc()
        self._bpp = (-self.ybus.imag)[self.pq, :][:, self.pq].tocsc()
        self._bp_lu = self._bpp_lu = None
        self.v_ref, self.ref_kind = reference_root(self)

    def f(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(P mismatch at PV+PQ, Q mismatch at PQ), per unit."""
        ds = injected_power(self.ybus, v) - self.sbus
        return ds[self.pvpq].real, ds[self.pq].imag

    def fdxb_step(self, v: np.ndarray) -> np.ndarray:
        """One XB fast-decoupled iteration (P-theta half then Q-V half), PYPOWER fdpf order."""
        if self._bp_lu is None:
            self._bp_lu = spla.splu(self._bp)
            self._bpp_lu = spla.splu(self._bpp) if len(self.pq) else None
        va, vm = np.angle(v), np.abs(v)
        p, _ = self.f(v)
        va[self.pvpq] -= self._bp_lu.solve(p / vm[self.pvpq])
        v = vm * np.exp(1j * va)
        if self._bpp_lu is not None:
            _, q = self.f(v)
            vm[self.pq] -= self._bpp_lu.solve(q / vm[self.pq])
            v = vm * np.exp(1j * va)
        return v


def reference_root(sc: Scenario) -> tuple[np.ndarray | None, str]:
    """(V_ref, kind): the label if it is an operating point, else the repaired-label Newton root.

    kind is ``"label"``, ``"repaired"`` or ``"none"`` (no physical root found from the label).
    """
    low = sc.vm_label < LABEL_VM_MIN
    if not low.any():
        return sc.v_label, "label"
    adj = sc.ybus.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    v = sc.v_label.copy()
    for _ in range(sc.n):  # fill collapsed buses from live neighbours, outward
        todo = np.where(low)[0]
        if not len(todo):
            break
        filled = False
        for k in todo:
            nb = [j for j in adj.indices[adj.indptr[k]:adj.indptr[k + 1]] if not low[j]]
            if nb:
                v[k], low[k], filled = np.mean(v[nb]), False, True
        if not filled:
            return None, "none"
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            r = newton_raphson_pf(sc.ybus, sc.sbus, v, sc.pv, sc.pq, tol=TOL, max_it=MAX_IT)
        except Exception:
            return None, "none"
    if r.converged and _finite(r.v) and float(np.abs(r.v).min()) >= LABEL_VM_MIN:
        return r.v, "repaired"
    return None, "none"


def load_grid(grid: str, scen: list[int], data_root: Path | None = None, shift: bool = False) -> dict[int, Scenario]:
    """Defaults unchanged. ``data_root``: another label variant (e.g. ``WORK/data-relabeled``). ``shift``: attach
    ``sc.va_dc_shift`` (radians), the shift-aware DC angles of ``relabel.shift_dc_angles`` from the scenario's own
    Ybus/Sbus and full branch table; ``starts_for`` then also emits ``dc_shift`` / ``vmdc_shift``."""
    raw = (data_root or WORK / "data") / grid / "raw"
    flt = [("scenario", "in", list(map(int, scen)))]
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    yb = pd.read_parquet(raw / "y_bus_data.parquet", filters=flt)
    br = pd.read_parquet(raw / "branch_data.parquet", filters=flt,
                         columns=None if shift else ["scenario", "from_bus", "to_bus", "x", "br_status"])
    by, bb = dict(tuple(yb.groupby("scenario"))), dict(tuple(br.groupby("scenario")))
    out = {int(s): Scenario(b, by[s], bb[s]) for s, b in bus.groupby("scenario")}
    if shift:
        from genco_challenge.relabel import shift_dc_angles
        for s, sc in out.items():
            sc.va_dc_shift = shift_dc_angles(sc, bb[s])
    return out


# ---------------------------------------------------------------------------- solvers

def _finite(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v))) and float(np.abs(v).max()) < 1e3


def nr_step(sc: Scenario, v: np.ndarray) -> np.ndarray:
    """One full Newton step via the repo solver (tol=0 forces exactly one iteration)."""
    return newton_raphson_pf(sc.ybus, sc.sbus, v, sc.pv, sc.pq, tol=0.0, max_it=1).v


def nr_ls_step(sc: Scenario, v: np.ndarray) -> np.ndarray:
    """Newton direction with backtracking (Armijo on 0.5*||F||_2^2, halving down to 1/64)."""
    vn = nr_step(sc, v)
    if not np.all(np.isfinite(vn)):
        return vn
    va, vm = np.angle(v), np.abs(v)
    dva = np.angle(vn / v)
    dvm = np.abs(vn) - vm
    p, q = sc.f(v)
    m0 = 0.5 * (p @ p + q @ q)
    alpha = 1.0
    while True:
        vt = (vm + alpha * dvm) * np.exp(1j * (va + alpha * dva))
        pt, qt = sc.f(vt)
        if 0.5 * (pt @ pt + qt @ qt) <= (1 - 1e-4 * alpha) * m0 or alpha <= 1 / 64:
            return vt
        alpha *= 0.5


STEPS = {"nr": nr_step, "nr_ls": nr_ls_step, "fdxb": lambda sc, v: sc.fdxb_step(v)}
WARM = (*STEPS, "fd2nr")  # fd2nr: 2 XB iterations then Newton (iterations count both)


def maxmis(sc: Scenario, v: np.ndarray) -> float:
    p, q = sc.f(v)
    return float(max(np.abs(p).max(initial=0), np.abs(q).max(initial=0)))


def solve(sc: Scenario, v0: np.ndarray, method: str) -> dict:
    """Iterate to tol/max_it. ``nr`` is one uninterrupted call of the repo solver (timed as such)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        if method in ("nr", "fd2nr"):
            try:
                v, it = v0, 0
                if method == "fd2nr":  # two XB iterations to repair the angles, then Newton
                    while it < 2 and maxmis(sc, v) > TOL and _finite(v):
                        v = sc.fdxb_step(v)
                        it += 1
                r = newton_raphson_pf(sc.ybus, sc.sbus, v, sc.pv, sc.pq, tol=TOL, max_it=MAX_IT - it)
                v, it, ok = r.v, it + r.iterations, r.converged
            except Exception:  # singular Jacobian etc.
                v, it, ok = v0 * np.nan, MAX_IT, False
        else:
            v, it, step = v0.copy(), 0, STEPS[method]
            try:
                while maxmis(sc, v) > TOL and it < MAX_IT and _finite(v):
                    v = step(sc, v)
                    it += 1
            except Exception:
                v = v0 * np.nan
            ok = _finite(v) and maxmis(sc, v) <= TOL
        sec = time.perf_counter() - t0
    ok = ok and _finite(v)
    dv_lab = float(np.abs(v - sc.v_label).max()) if ok else float("nan")
    dv = float(np.abs(v - sc.v_ref).max()) if ok and sc.v_ref is not None else float("nan")
    if not ok:
        outcome = "diverged"
    elif sc.v_ref is None:
        outcome = "no_reference"
    else:
        outcome = "right_root" if dv < ROOT_TOL else "wrong_root"
    return {"converged": bool(ok), "iterations": int(it), "seconds": sec, "max_dv": dv,
            "max_dv_label": dv_lab, "min_vm": float(np.abs(v).min()) if ok else float("nan"),
            "ref_kind": sc.ref_kind, "outcome": outcome}


def trajectory(sc: Scenario, v0: np.ndarray, method: str, kmax: int) -> list[np.ndarray]:
    out = [v0]
    v = v0
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for _ in range(kmax):
            try:
                v = STEPS[method](sc, v) if _finite(v) else v * np.nan
            except Exception:
                v = v * np.nan
            out.append(v)
    return out


def score(sc: Scenario, v: np.ndarray) -> dict:
    """Per-scenario state score: mismatch (MW/MVAr) and error sums/counts (vs V_ref) by bus type."""
    row: dict = {}
    if not _finite(v):
        row["finite"] = False
        v = np.full(sc.n, np.nan, complex)
    else:
        row["finite"] = True
    p, q = sc.f(v)
    a = np.abs(np.r_[p, q]) * BASE_MVA
    row["mis_max"] = float(a.max())
    row["mis_mean"] = float(a.mean())
    row["p_max"] = float(np.abs(p).max(initial=0) * BASE_MVA)
    row["q_max"] = float(np.abs(q).max(initial=0) * BASE_MVA)
    vref = sc.v_ref if sc.v_ref is not None else np.full(sc.n, np.nan, complex)
    e_vm = np.abs(np.abs(v) - np.abs(vref))
    e_va = np.degrees(np.abs(np.angle(v * np.conj(vref))))
    for t in TYPES:
        m = sc.is_t[t] & (sc.v_ref is not None)
        row[f"n_{t}"] = int(m.sum())
        row[f"vm_sum_{t}"] = float(e_vm[m].sum())
        row[f"va_sum_{t}"] = float(e_va[m].sum())
    return row


# ---------------------------------------------------------------------------- per grid

def starts_for(sc: Scenario, pred: pd.DataFrame | None) -> dict[str, np.ndarray]:
    st = {"flat": sc.vm_set * np.exp(1j * np.full(sc.n, sc.va_ref)),
          "dc": sc.vm_set * np.exp(1j * sc.va_dc)}
    th_s = getattr(sc, "va_dc_shift", None)  # set by load_grid(shift=True); absent = original starts only
    if th_s is not None:
        st["dc_shift"] = sc.vm_set * np.exp(1j * th_s)
    if pred is not None:
        vm_p, va_p = pred["Vm_pred"].to_numpy(float), pred["Va_pred"].to_numpy(float)
        st["raw"] = vm_p * np.exp(1j * va_p)
        pvref = sc.is_t["PV"] | sc.is_t["REF"]
        vm_pin = np.where(pvref, sc.vm_set, vm_p)
        va_pin = np.where(sc.is_t["REF"], sc.va_ref, va_p)
        st["pinned"] = vm_pin * np.exp(1j * va_pin)
        st["vmdc"] = vm_pin * np.exp(1j * sc.va_dc)
        st["vmflat"] = vm_pin * np.exp(1j * np.full(sc.n, sc.va_ref))
        # PQ Vm clipped into the grid's own voltage limits (case data, not labels)
        vm_clip = np.where(sc.is_t["PQ"], np.clip(vm_pin, sc.vm_lo, sc.vm_hi), vm_pin)
        st["clip"] = vm_clip * np.exp(1j * va_pin)
        st["clipdc"] = vm_clip * np.exp(1j * sc.va_dc)
        if th_s is not None:
            st["vmdc_shift"] = vm_pin * np.exp(1j * th_s)
    return st


def run_grid(grid: str, preds: dict[str, Path], out: Path) -> dict:
    stamp = run_stamp(grid, preds)  # taken before reading anything, so a change mid-run reads as stale
    pr = {m: pd.read_parquet(p) for m, p in preds.items()}
    raw = WORK / "data" / grid / "raw"
    lab = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Vm", "Va"])
    join_err, excluded = {}, {}
    for m in list(pr):
        j = pr[m].merge(lab, on=["scenario", "bus"], how="left")
        e = float(np.nanmax(np.r_[np.abs(j["Vm_target"] - j["Vm"]),
                                  np.abs(j["Va_target"] - np.radians(j["Va"]))]))
        join_err[m] = e if j["Vm"].notna().all() else float("inf")
        if not join_err[m] <= 1e-5:
            excluded[m] = join_err[m]
            del pr[m]
    if not pr:  # still run the identity check on the labels themselves
        scs = load_grid(grid, sorted(lab["scenario"].unique()))
        ident = [maxmis(sc, sc.v_label) * BASE_MVA for sc in scs.values()]
        res = {"version": VERSION, "grid": grid, "n_bus": int(next(iter(scs.values())).n),
               "models": sorted(preds), "excluded_models": excluded,
               "n_scenarios": {"labels": len(scs)},
               "identity": {"label_mismatch_max_mw": float(np.max(ident)),
                            "label_mismatch_median_mw": float(np.median(ident)),
                            "pred_target_vs_bus_data_max_abs": join_err},
               "error": "no checkpoint's prediction targets match bus_data (stale processed data?)",
               "hybrid": {}, "warm": {}}
        (out / "per_grid").mkdir(parents=True, exist_ok=True)
        res.update(stamp)
        (out / "per_grid" / f"{grid}.json").write_text(json.dumps(res, indent=1))
        return res
    scen = sorted(set().union(*(set(d["scenario"].unique()) for d in pr.values())))
    t0 = time.time()
    scs = load_grid(grid, scen)
    pr_by = {m: {int(s): g.sort_values("bus") for s, g in d.groupby("scenario")} for m, d in pr.items()}
    ident = []
    hyb, warm = [], []
    kinds = pd.Series({s: sc.ref_kind for s, sc in scs.items()})
    for s, sc in scs.items():
        ident.append(maxmis(sc, sc.v_label) * BASE_MVA)
        jobs = [("-", k, v) for k, v in starts_for(sc, None).items()]
        for m, d in pr_by.items():
            if s not in d:
                continue
            pdf = d[s]
            jobs += [(m, k, v) for k, v in starts_for(sc, pdf).items() if k not in ("flat", "dc", "dc_shift")]
        for model, start, v0 in jobs:
            for method in STEPS:
                for k, v in enumerate(trajectory(sc, v0, method, max(KS))):
                    if k == 0 and method != "nr":
                        continue  # k=0 is the start itself, scored once
                    hyb.append({"scenario": s, "model": model, "start": start,
                                "method": "start" if k == 0 else method, "k": k} | score(sc, v))
            for method in WARM:
                warm.append({"scenario": s, "model": model, "start": start, "method": method}
                            | solve(sc, v0, method))
    hyb_df, warm_df = pd.DataFrame(hyb), pd.DataFrame(warm)
    pg = out / "per_grid"
    pg.mkdir(parents=True, exist_ok=True)
    hyb_df.to_parquet(pg / f"{grid}_hybrid.parquet")
    warm_df.to_parquet(pg / f"{grid}_warm.parquet")
    res = {"version": VERSION, "grid": grid, "n_bus": int(next(iter(scs.values())).n), "models": sorted(preds),
           "pred_files": {m: str(p.relative_to(REPO)) for m, p in preds.items()},
           "n_scenarios": {m: int(d["scenario"].nunique()) for m, d in pr.items()},
           "identity": {"label_mismatch_max_mw": float(np.max(ident)),
                        "label_mismatch_median_mw": float(np.median(ident)),
                        "pred_target_vs_bus_data_max_abs": join_err},
           "excluded_models": excluded,
           "reference": {k: int((kinds == k).sum()) for k in ("label", "repaired", "none")},
           "seconds": round(time.time() - t0, 1)}
    res["hybrid"], res["warm"] = {}, {}
    for m in pr:
        ss = set(pr[m]["scenario"].unique())
        h = hyb_df[hyb_df["scenario"].isin(ss) & hyb_df["model"].isin(["-", m])]
        w = warm_df[warm_df["scenario"].isin(ss) & warm_df["model"].isin(["-", m])]
        res["hybrid"][m] = agg_hybrid(h)
        res["warm"][m] = agg_warm(w)
    res.update(stamp)
    (pg / f"{grid}.json").write_text(json.dumps(res, indent=1))
    return res


def agg_hybrid(h: pd.DataFrame) -> list[dict]:
    rows = []
    for (start, method, k), g in h.groupby(["start", "method", "k"], sort=False):
        r = {"start": start, "method": method, "k": int(k), "n": len(g),
             "nonfinite": int((~g["finite"]).sum()),
             "mis_max_mw": float(g["mis_max"].max()), "mis_max_median_mw": float(g["mis_max"].median()),
             "mis_mean_mw": float(g["mis_mean"].mean()),
             "p_max_mw": float(g["p_max"].max()), "q_max_mvar": float(g["q_max"].max())}
        # Diverged (non-finite) states have NaN error sums: drop their bus counts from the denominator too,
        # so numerator and denominator cover the same rows (audit R1-F4). mis_mean is likewise over the
        # finite rows only; r["nonfinite"] / r["n"] is the diverged k/n shown in the headline table.
        gf = g[g["finite"].astype(bool)]
        tot = sum(gf[f"n_{t}"].sum() for t in TYPES)  # 0 when no finite scenario has a reference root
        r["vm_mae_pu"] = float(sum(gf[f"vm_sum_{t}"].sum() for t in TYPES) / tot) if tot else float("nan")
        r["va_mae_deg"] = float(sum(gf[f"va_sum_{t}"].sum() for t in TYPES) / tot) if tot else float("nan")
        for t in TYPES:
            nt = gf[f"n_{t}"].sum()
            if nt:
                r[f"vm_mae_pu_{t}"] = float(gf[f"vm_sum_{t}"].sum() / nt)
                r[f"va_mae_deg_{t}"] = float(gf[f"va_sum_{t}"].sum() / nt)
        rows.append(r)
    return rows


def agg_warm(w: pd.DataFrame) -> list[dict]:
    rows = []
    for (start, method), g in w.groupby(["start", "method"], sort=False):
        c = g[g["converged"]]
        hr = g[g["ref_kind"] != "none"]  # right/wrong root only where a reference root exists
        rows.append({"start": start, "method": method, "n": len(g), "n_ref": len(hr),
                     "converged_frac": float(g["converged"].mean()),
                     "right_root_frac": float((hr["outcome"] == "right_root").mean()) if len(hr) else float("nan"),
                     "wrong_root_frac": float((hr["outcome"] == "wrong_root").mean()) if len(hr) else float("nan"),
                     "diverged_frac": float((g["outcome"] == "diverged").mean()),
                     "diverged_frac_ref": float((hr["outcome"] == "diverged").mean()) if len(hr) else float("nan"),
                     "iters_mean_conv": float(c["iterations"].mean()) if len(c) else float("nan"),
                     "iters_mean_right": float(g.loc[g["outcome"] == "right_root", "iterations"].mean())
                     if (g["outcome"] == "right_root").any() else float("nan"),
                     "iters_max_conv": int(c["iterations"].max()) if len(c) else -1,
                     "ms_median": float(g["seconds"].median() * 1e3),
                     "ms_median_conv": float(c["seconds"].median() * 1e3) if len(c) else float("nan"),
                     "max_dv_median_conv": float(c["max_dv"].median()) if len(c) else float("nan")})
    return rows


# ---------------------------------------------------------------------------- summary

def headline(res: list[dict]) -> list[str]:
    """Compact G1-2c / G1-3 tables: Newton from DC vs from GENCO starts, per grid and checkpoint."""
    hs = ("dc", "pinned", "vmdc", "clipdc")
    L = ["", "## Headline G1-2c: mean |mismatch| (MW/MVAr) after k Newton iterations "
         "[median over scenarios of the per-scenario max, k=1]", "",
         "| grid | ckpt | " + " | ".join(f"{s} k=1" for s in hs) + " | " + " | ".join(f"{s} k=2" for s in hs)
         + " | ratio k=1 pinned/dc, vmdc/dc, clipdc/dc | Va MAE deg k=1 " + " / ".join(hs)
         + " | diverged k/n at k=1, k=2 (" + " / ".join(hs) + ") |",
         "|---|---|" + "---|" * (2 * len(hs) + 3)]
    for r in res:
        for m, rows in r["hybrid"].items():
            x = {(a["start"], a["method"], a["k"]): a for a in rows}
            g = {(s, k): x[(s, "nr", k)] for s in hs for k in (1, 2)}
            k1 = " | ".join(f"{g[s, 1]['mis_mean_mw']:.3g} [{g[s, 1]['mis_max_median_mw']:.3g}]" for s in hs)
            k2 = " | ".join(f"{g[s, 2]['mis_mean_mw']:.2g}" for s in hs)
            d1 = g["dc", 1]["mis_mean_mw"]
            L.append(f"| {r['grid'].replace('pglib_opf_', '')} | {m} | {k1} | {k2} | "
                     f"{g['pinned', 1]['mis_mean_mw'] / d1:.2f}, {g['vmdc', 1]['mis_mean_mw'] / d1:.2f}, "
                     f"{g['clipdc', 1]['mis_mean_mw'] / d1:.2f} | "
                     + " / ".join(f"{g[s, 1]['va_mae_deg']:.3g}" for s in hs) + " | "
                     + "; ".join(" / ".join(f"{g[s, k]['nonfinite']}/{g[s, k]['n']}" for s in hs)
                                 for k in (1, 2)) + " |")
    ws = [("flat", "nr"), ("dc", "nr"), ("raw", "nr"), ("pinned", "nr"), ("pinned", "nr_ls"),
          ("pinned", "fd2nr"), ("vmdc", "nr"), ("vmflat", "nr"), ("clip", "nr"), ("clipdc", "nr"),
          ("clipdc", "fd2nr")]
    L += ["", "## Headline G1-3: warm start, right-root fraction (mean iterations when right)", "",
          "Fraction over the scenarios that have a reference root (the label, or the repaired-label root "
          "where the label is collapsed; see the identity table). `nan` = no scenario has one.", "",
          "| grid | ckpt | " + " | ".join(f"{s} {m}" for s, m in ws) + " |", "|---|---|" + "---|" * len(ws)]
    for r in res:
        for m, rows in r["warm"].items():
            x = {(a["start"], a["method"]): a for a in rows}

            def c(s, meth, x=x):
                a = x.get((s, meth))
                if a is None:
                    return "-"
                return f"{a['right_root_frac']:.3f} ({a.get('iters_mean_right', a['iters_mean_conv']):.2f})"
            L.append(f"| {r['grid'].replace('pglib_opf_', '')} | {m} | " + " | ".join(c(s, k) for s, k in ws) + " |")
    return L


def write_summary(out: Path) -> None:
    res = [json.loads(p.read_text()) for p in sorted((out / "per_grid").glob("*.json"))]
    res.sort(key=lambda r: r["n_bus"])
    L = ["# Summary (generated by genco_challenge.hybrid; do not edit)", "",
         "## Identity check (label mismatch under the scenario's own Ybus/Sbus, MW/MVAr)", "",
         "Reference root per scenario: `label` (an operating point), `repaired` (label collapsed, i.e. some "
         f"Vm < {LABEL_VM_MIN} pu; reference = Newton from the label with the collapsed buses repaired), "
         "`none` (label collapsed and no physical root found from it).", "",
         "| grid | n_bus | scen | max | median | pred target vs bus_data | ref label / repaired / none |",
         "|---|---|---|---|---|---|---|"]
    for r in res:
        i = r["identity"]
        rk = r.get("reference")
        rks = "-" if rk is None else f"{rk['label']} / {rk['repaired']} / {rk['none']}"
        L.append(f"| {r['grid']} | {r['n_bus']} | {max(r['n_scenarios'].values())} | "
                 f"{i['label_mismatch_max_mw']:.2e} | {i['label_mismatch_median_mw']:.2e} | "
                 f"{max(i['pred_target_vs_bus_data_max_abs'].values()):.1e} | {rks} |")
    for r in res:
        if r.get("excluded_models"):
            L.append(f"\nEXCLUDED on {r['grid']}: {r['excluded_models']} (prediction targets do not match "
                     "bus_data labels; the checkpoint was evaluated on a different dataset)")
    L += headline(res)
    for m in ("zeroshot", "ft118_1000", "ft118_250000"):
        L += ["", f"## Hybrid, checkpoint `{m}` (mean of per-scenario mean |mismatch| MW; "
              "pooled Va MAE deg; Vm MAE pu at PQ)", "",
              "| grid | start | method | k=0 mis | k=1 mis | k=2 mis | k=1 Va | k=2 Va | k=2 Vm PQ |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            if m not in r["hybrid"]:
                continue
            rows = {(x["start"], x["method"], x["k"]): x for x in r["hybrid"][m]}
            for st in ("flat", "dc", "raw", "pinned", "vmdc", "vmflat", "clip", "clipdc"):
                k0 = rows.get((st, "start", 0))
                for meth in STEPS:
                    k1, k2 = rows.get((st, meth, 1)), rows.get((st, meth, 2))
                    if k0 is None or k1 is None:
                        continue
                    L.append(f"| {r['grid']} | {st} | {meth} | {k0['mis_mean_mw']:.3g} | "
                             f"{k1['mis_mean_mw']:.3g} | {k2['mis_mean_mw']:.3g} | "
                             f"{k1['va_mae_deg']:.3g} | {k2['va_mae_deg']:.3g} | "
                             f"{k2.get('vm_mae_pu_PQ', float('nan')):.2e} |")
        L += ["", f"## Warm start, checkpoint `{m}` (tol 1e-8 pu, max_it 30)", "",
              "| grid | start | method | conv | right | wrong | div | iters mean | iters max | ms median |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            if m not in r["warm"]:
                continue
            for x in r["warm"][m]:
                L.append(f"| {r['grid']} | {x['start']} | {x['method']} | {x['converged_frac']:.3f} | "
                         f"{x['right_root_frac']:.3f} | {x['wrong_root_frac']:.3f} | "
                         f"{x['diverged_frac']:.3f} | {x['iters_mean_conv']:.2f} | "
                         f"{x['iters_max_conv']} | {x['ms_median']:.2f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def upgrade_v3(grid: str, out: Path, prev: dict, preds: dict[str, Path] | None = None) -> dict | None:
    """Re-aggregate a VERSION-3 grid from its parquet rows when no label is collapsed.

    v4 changed only the reference root, and on such a grid the reference is the label everywhere, so
    the per-scenario rows are unchanged and re-solving would only re-time them. Returns None when the
    grid has a collapsed label (it must be recomputed). v3 results carry no stamp, so when ``preds`` is given
    the parquet rows must also be newer than raw/ and every prediction file (R2-F7); otherwise None.
    """
    if prev.get("version") != 3 or "error" in prev:
        return None
    raw = WORK / "data" / grid / "raw"
    vm = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "Vm"])
    wf = out / "per_grid" / f"{grid}_warm.parquet"
    if not wf.exists() or float(vm["Vm"].min()) < LABEL_VM_MIN:
        return None
    if preds is not None and (wf.stat().st_mtime < raw.stat().st_mtime
                              or any(wf.stat().st_mtime < p.stat().st_mtime for p in preds.values())):
        return None
    w = pd.read_parquet(wf)
    w["ref_kind"], w["max_dv_label"] = "label", w["max_dv"]
    for m in prev["warm"]:
        ss = set(w.loc[w["model"] == m, "scenario"])
        prev["warm"][m] = agg_warm(w[w["scenario"].isin(ss) & w["model"].isin(["-", m])])
    prev["version"] = VERSION
    prev["reference"] = {"label": int(w["scenario"].nunique()), "repaired": 0, "none": 0}
    prev["upgraded_from_v3"] = True
    if preds is not None:  # freshness was checked above, so the upgraded result is stamped current
        prev.update(run_stamp(grid, preds))
    (out / "per_grid" / f"{grid}.json").write_text(json.dumps(prev, indent=1))
    return prev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    files = prediction_files()
    for g in sorted(args.grids or files):
        if g not in files:
            print(f"{g}: no predictions yet", flush=True)
            continue
        meta_p = WORK / "data" / g / "meta.json"
        status = json.loads(meta_p.read_text()).get("status") if meta_p.exists() else None
        if status != "ok":
            print(f"{g}: skipped, meta.json status is {status!r} (not 'ok'; labels removed or missing)", flush=True)
            continue
        done = args.out / "per_grid" / f"{g}.json"
        if done.exists() and not args.force:
            prev = json.loads(done.read_text())
            if set(prev["models"]) >= set(files[g]):
                if prev.get("version") == VERSION and is_current(prev, g, files[g]):
                    print(f"{g}: done ({prev['models']})", flush=True)
                    continue
                if prev.get("version") != VERSION and upgrade_v3(g, args.out, prev, files[g]) is not None:
                    print(f"{g}: v3 -> v{VERSION} re-aggregated (no collapsed label)", flush=True)
                    continue
        r = run_grid(g, files[g], args.out)
        print(f"{g}: n_bus={r['n_bus']} models={r['models']} identity_max_mw="
              f"{r['identity']['label_mismatch_max_mw']:.2e} ({r.get('seconds')}s) {r.get('error', '')}", flush=True)
    write_summary(args.out)


if __name__ == "__main__":
    main()
