"""Shared scoring code for stages 1 and 2: per-scenario states (dc, fd2, nr1, GENCO and combinations) and metrics.
All states are scored on the scenarios graphkit scored for the GENCO prediction, against the AC label in the
prediction file (graphkit targets, join-checked against raw by fewshot._score at prediction time)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from genco_challenge import fewshot as fs
from genco_challenge import hybrid as hy
from genco_challenge import labelfree_prior2 as lp2

OUT = Path(__file__).resolve().parents[1]
W = fs.WORK / "genco_on_fd2"
BASE = hy.BASE_MVA


def wrap(d):
    """degrees, to (-180, 180]"""
    return -((-d + 180.0) % 360.0 - 180.0)


def wrap_rad(d):
    return np.angle(np.exp(1j * d))


def pred_file(fsdir: Path, key: str, grid: str) -> Path:
    return fs._find_run(fsdir / "mlruns_eval", f"eval__{key}") / "artifacts/test" / f"{grid}_predictions.parquet"


class Bundle:
    """Everything about one scenario needed to build and score states."""


def bundles(root: Path, grid: str, pred: pd.DataFrame, with_nr=True):
    scen = sorted(int(s) for s in pred["scenario"].unique())
    bus, ys, brs = lp2._grid_tables(root, grid, scen)
    keyed = pred.set_index(["scenario", "bus"]).sort_index()
    for s in scen:
        b = bus[bus["scenario"] == s].sort_values("bus")
        sc = lp2.input_problem(b, ys[s], brs[s])
        pr = lp2.priors_for(sc)
        g = keyed.loc[s].sort_index()
        o = Bundle()
        o.s, o.sc, o.pr = s, sc, pr
        o.n = sc.n
        on = brs[s][brs[s]["br_status"] == 1]
        o.f, o.t = on["from_bus"].to_numpy(), on["to_bus"].to_numpy()
        o.vm_t, o.va_t = g["Vm_target"].to_numpy(), g["Va_target"].to_numpy()           # rad
        o.vm_g, o.va_g = g["Vm_pred"].to_numpy(), g["Va_pred"].to_numpy()               # rad
        o.is_t = sc.is_t
        o.va_dc = sc.va_dc
        yield o


def genco_aligned(o, va_g=None):
    va_g = o.va_g if va_g is None else va_g
    return va_g - va_g[o.sc.ref].mean() + o.sc.va_ref


def fd_state(o, key):
    v = o.pr[key]["v"]
    return np.abs(v), np.radians(lp2._unwrapped(o.sc, v))


def build_states(o, blend=None, e50=False, vm_g=None, va_g=None):
    """dict name -> (vm, va_rad). blend = (w_vm, w_pv, w_pq) or None."""
    sc = o.sc
    vm_g = o.vm_g if vm_g is None else vm_g
    va_g_in = o.va_g if va_g is None else va_g
    pvref = o.is_t["PV"] | o.is_t["REF"]
    pq = o.is_t["PQ"]
    vm_set = sc.vm_set
    vm_fd, va_fd = fd_state(o, "fd2")
    vm_nr, va_nr = fd_state(o, "nr1")
    gal = genco_aligned(o, va_g_in)
    st = {"dc": (vm_set, o.va_dc), "fd2": (vm_fd, va_fd), "nr1": (vm_nr, va_nr),
          "genco_raw": (vm_g, va_g_in),
          "genco_pinned": (np.where(pvref, o.vm_t, vm_g), np.where(o.is_t["REF"], o.va_t, va_g_in))}
    vm_b = np.where(pvref, vm_set, vm_g)
    st["b_genco_vm_fd2_va"] = (vm_b, va_fd)
    st["c_fd2_vm_genco_va"] = (vm_fd, gal)
    def mix(w_vm, w_pv, w_pq):
        vm = np.where(pq, vm_fd + w_vm * (vm_g - vm_fd), vm_set)
        # REF and PV |V| are setpoints in every state
        dv = wrap_rad(gal - va_fd)
        w = np.where(o.is_t["PV"], w_pv, np.where(pq, w_pq, 0.0))
        va = va_fd + w * dv
        va = np.where(o.is_t["REF"], sc.va_ref, va)
        return vm, va
    if blend is not None:
        st["d_blend"] = mix(*blend)
    if e50:
        st["e50"] = mix(0.5, 0.5, 0.5)
    return st


def residual(sc, vm, va):
    v = vm * np.exp(1j * va)
    p, q = sc.f(v)
    return float(np.abs(p).sum() * BASE / sc.n), float(np.abs(q).sum() * BASE / sc.n)


def score(o, vm, va):
    """per-scenario sums: (dva_sum, dva_cnt, va_sum, vm_sum, n, p_res, q_res)"""
    d_p = np.degrees(va[o.f] - va[o.t])
    d_t = np.degrees(o.va_t[o.f] - o.va_t[o.t])
    e_br = np.abs(wrap(d_p - d_t))
    e_va = np.abs(wrap(np.degrees(va - o.va_t)))
    e_vm = np.abs(vm - o.vm_t)
    p, q = residual(o.sc, vm, va)
    return [float(e_br.sum()), len(e_br), float(e_va.sum()), float(e_vm.sum()), o.n, p, q]


COLS = ["dva_sum", "dva_cnt", "va_sum", "vm_sum", "n", "p_res", "q_res"]


def agg(arr):
    """arr: (S,7) -> dict of aggregate metrics (ratio of sums for per-element means)"""
    a = np.asarray(arr, float)
    return {"branch_dva_deg": a[:, 0].sum() / a[:, 1].sum(), "va_deg": a[:, 2].sum() / a[:, 4].sum(),
            "vm_pu": a[:, 3].sum() / a[:, 4].sum(), "p_res_mw": a[:, 5].mean(), "q_res_mvar": a[:, 6].mean()}


def boot_ratio(x, ref, kind, n=2000, seed=1):
    """paired bootstrap over scenarios of metric(x)/metric(ref); kind in dva, va, vm, p, q"""
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    rng = np.random.default_rng(seed)
    S = len(x)
    def m(a, idx):
        a = a[idx]
        if kind == "dva": return a[:, 0].sum() / a[:, 1].sum()
        if kind == "va": return a[:, 2].sum() / a[:, 4].sum()
        if kind == "vm": return a[:, 3].sum() / a[:, 4].sum()
        if kind == "p": return a[:, 5].mean()
        return a[:, 6].mean()
    full = np.arange(S)
    pt = m(x, full) / m(ref, full)
    rs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, S, S)
        rs[i] = m(x, idx) / m(ref, idx)
    return float(pt), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))
