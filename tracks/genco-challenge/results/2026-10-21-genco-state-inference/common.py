"""Shared pieces for candidate #7 (zero-shot state inference, unknown dispatch, |V| at 20% of buses). UNAUDITED.

Task per scenario: loads (Pd, Qd), bus-type flags, limits, costs, topology/admittances known; Pg at EVERY generator
unknown; Qg unknown; |V| known at a seeded 20% of buses (exactly k = max(1, round(0.2 n))); all angles unknown EXCEPT the
REF bus angle, which is visible at its label value (the case's reference angle, input data of any power-flow problem).
Branch flows unknown. Same information goes to GENCO (as mask) and to the classical baselines (as arguments).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXT = REPO / "data/external"
LAD = EXT / "genco-ladder"
SEED = 20261021
FRAC = 0.2
BASE_MVA = 100.0
HELDOUT15 = [
    "pglib_opf_case5_pjm", "pglib_opf_case14_ieee", "pglib_opf_case24_ieee_rts", "pglib_opf_case30_ieee",
    "pglib_opf_case39_epri", "pglib_opf_case57_ieee", "pglib_opf_case73_ieee_rts", "pglib_opf_case89_pegase",
    "pglib_opf_case118_ieee", "pglib_opf_case162_ieee_dtc", "pglib_opf_case240_pserc", "pglib_opf_case300_ieee",
    "pglib_opf_case1354_pegase", "pglib_opf_case2736sp_k", "pglib_opf_case3120sp_k",
]


def measured_set(sid: int, n: int, frac: float = FRAC) -> np.ndarray:
    """Seeded measured-|V| bus set for one scenario (sorted local bus indices). Shared by the mask and the baselines."""
    k = n if frac >= 1.0 else max(1, int(round(frac * n)))
    rng = np.random.default_rng([SEED, int(n), int(sid)])
    return np.sort(rng.permutation(n)[:k])


def wrap(d):
    return -((-d + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------- data
def tables(grid: str, ids) -> dict:
    raw = LAD / "data" / grid / "raw"
    f = [("scenario", "in", [int(i) for i in ids])]
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=f)
    gen = pd.read_parquet(raw / "gen_data.parquet", filters=f)
    br = pd.read_parquet(raw / "branch_data.parquet", filters=f)
    yb = pd.read_parquet(raw / "y_bus_data.parquet", columns=["scenario", "index1", "index2", "G", "B"], filters=f)
    return {"bus": dict(tuple(bus.sort_values(["scenario", "bus"]).groupby("scenario"))),
            "gen": dict(tuple(gen.sort_values(["scenario", "idx"]).groupby("scenario"))),
            "br": dict(tuple(br.groupby("scenario"))), "y": dict(tuple(yb.groupby("scenario")))}


def base_case(grid: str) -> dict:
    """Case-file information the classical baselines may use: per-gen base Pg and Vg (gen row order = datakit gen idx)
    and the base case's loss fraction (pandapower AC run of the case as given)."""
    from matpowercaseframes import CaseFrames
    cf = CaseFrames(str(LAD / "grids" / f"{grid}.m"))
    ids = np.sort(cf.bus.BUS_I.to_numpy().astype(int))
    pos = {b: i for i, b in enumerate(ids)}
    g = cf.gen
    out = {"gen_bus": np.array([pos[int(b)] for b in g.GEN_BUS]), "pg": g.PG.to_numpy(float), "vg": g.VG.to_numpy(float),
           "status": g.GEN_STATUS.to_numpy(float)}
    import pandapower as pp
    from pandapower.converter.matpower import from_mpc
    net = from_mpc(str(LAD / "grids" / f"{grid}.m"))
    # amendment 1: the case files' dispatch is not balanced against their load (case39: 3,360 of 6,254 MW), so an AC run
    # of the file as given can fail. Loss fraction = losses of the AC solution with every generator's Pg scaled by
    # sum(Pd)/sum(Pg_file) (slack picks up the rest); applied to ALL grids.
    net.gen["p_mw"] = net.gen.p_mw * (float(net.load.p_mw.sum()) / float(g.PG.sum()))
    solved = False
    for kw in ({}, {"init": "dc", "max_iteration": 100}, {"algorithm": "iwamoto_nr", "init": "dc", "max_iteration": 100}):
        try:
            pp.runpp(net, numba=False, **kw)
            out["pp_runpp_kw"] = str(kw)
            solved = True
            break
        except Exception:
            continue
    if solved:
        loss = -float(net.res_bus.p_mw.sum())
    else:       # AC run of the (scaled) case file fails (stressed files, e.g. case162): DC branch flows with I^2 R losses
        pp.rundcpp(net)
        br = net._ppc["branch"]
        loss = float((br[:, 2] * (br[:, 13] / float(net._ppc["baseMVA"])) ** 2).sum() * float(net._ppc["baseMVA"]))
        out["pp_runpp_kw"] = "dc_i2r_fallback"
    out["loss_frac"] = loss / float(net.load.p_mw.sum())
    return out


def ybus(ys: pd.DataFrame, n: int) -> sp.csc_matrix:
    return sp.csc_matrix((ys.G.to_numpy() + 1j * ys.B.to_numpy(), (ys.index1.to_numpy(), ys.index2.to_numpy())), shape=(n, n))


# ---------------------------------------------------------------- solvers
def dc_angles(br: pd.DataFrame, bus: pd.DataFrame, pinj_mw: np.ndarray, ref: np.ndarray, va_ref_deg: np.ndarray) -> np.ndarray:
    """MATPOWER-style DC power flow (taps, shifters, branch status honoured), degrees. Losses and shunts ignored."""
    n = len(bus)
    b = br[br.br_status == 1]
    f, t = b.from_bus.to_numpy(), b.to_bus.to_numpy()
    tap = np.where(b.tap.to_numpy() == 0, 1.0, b.tap.to_numpy())
    bb = 1.0 / (b.x.to_numpy() * tap)
    m = len(b)
    C = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[f, t])), shape=(m, n))
    Bf = sp.diags(bb) @ C
    Bbus = (C.T @ Bf).tocsc()
    Pinj = C.T @ (-bb * np.radians(b["shift"].to_numpy()))
    rhs = pinj_mw / BASE_MVA - Pinj
    nr = np.setdiff1d(np.arange(n), ref)
    th = np.zeros(n)
    th[ref] = np.radians(va_ref_deg[ref])
    r = rhs[nr] - Bbus[nr][:, ref] @ th[ref]
    th[nr] = sp.linalg.spsolve(Bbus[nr][:, nr].tocsc(), r)
    return np.degrees(th)


def newton(Y, sbus, v0, pv, pq, tol=1e-8, max_it=30):
    import sys
    sys.path.insert(0, str(REPO / "core"))
    from grid_ops.reference_pf import newton_raphson_pf
    try:
        return newton_raphson_pf(Y, sbus, v0, pv, pq, tol=tol, max_it=max_it)
    except Exception:
        return None


def pg_from_solution(res_v, Y, bus, gen, pg_bus_mw):
    """Per-gen Pg (MW) for in-service gens: scheduled dispatch, with REF-bus generation recomputed from the solved injection."""
    n = len(bus)
    S = res_v * np.conj(Y @ res_v)
    ref = bus.REF.to_numpy() == 1
    pg_bus = pg_bus_mw.copy()
    pg_bus[ref] = S.real[ref] * BASE_MVA + bus.Pd.to_numpy()[ref]
    return pg_bus


def split_to_gens(pg_bus, gen, w_sched):
    """Distribute bus-level Pg over the in-service gens at each bus proportional to w_sched (equal if all zero)."""
    ins = gen.in_service.to_numpy() == 1
    gb = gen.bus.to_numpy()
    out = np.zeros(len(gen))
    for b in np.unique(gb[ins]):
        sel = ins & (gb == b)
        w = np.abs(w_sched[sel])
        w = w / w.sum() if w.sum() > 0 else np.ones(sel.sum()) / sel.sum()
        out[sel] = pg_bus[b] * w
    return out


def baselines(tab: dict, sid: int, base: dict, frac: float = FRAC) -> dict:
    """All classical predictors for one scenario. Returns {name: dict(vm, va, pg, converged)}.
    Inputs used (and nothing else): bus tables Pd, Qd, type flags, GS/BS via Ybus, branch table, gen table (bus,
    in_service), the measured |V| at the seeded set, the REF angle, and the CASE FILE (base Pg, Vg, loss fraction)."""
    bus, gen, br, yb = tab["bus"][sid], tab["gen"][sid], tab["br"][sid], tab["y"][sid]
    n = len(bus)
    Y = ybus(yb, n)
    pq = np.where(bus.PQ.to_numpy() == 1)[0]
    pv = np.where(bus.PV.to_numpy() == 1)[0]
    ref = np.where(bus.REF.to_numpy() == 1)[0]
    Pd, Qd = bus.Pd.to_numpy(), bus.Qd.to_numpy()
    vm_t, va_t = bus.Vm.to_numpy(), bus.Va.to_numpy()
    meas = np.zeros(n, bool)
    meas[measured_set(sid, n, frac)] = True
    ins = gen.in_service.to_numpy() == 1
    gidx = gen.idx.to_numpy().astype(int)
    gbus = gen.bus.to_numpy().astype(int)
    out = {}

    def run(name, pg_bus, vset, vstart_va=None, conv_fallback=None):
        sbus = (pg_bus - Pd) / BASE_MVA + 1j * (-Qd) / BASE_MVA
        va0 = np.radians(vstart_va if vstart_va is not None else np.zeros(n))
        vm0 = np.ones(n)
        fixed = np.r_[pv, ref]
        vm0[fixed] = vset[fixed]
        va0[ref] = np.radians(va_t[ref])
        r = newton(Y, sbus, vm0 * np.exp(1j * va0), pv, pq)
        ok = r is not None and r.converged and np.isfinite(r.vm).all()
        if not ok:
            return None
        pgb = pg_from_solution(r.v, Y, bus, gen, pg_bus)
        return {"vm": r.vm, "va": r.va_deg, "pgb": pgb, "converged": True, "iters": r.iterations}

    # ---- dispatch of B1: base dispatch scaled to total load * (1 + base loss fraction)
    bpg = np.where(ins, base["pg"][gidx] * (base["status"][gidx] > 0), 0.0)
    k = (Pd.sum() * (1.0 + base["loss_frac"])) / bpg.sum() if bpg.sum() > 0 else 0.0
    pg_gen = bpg * k
    pg_bus1 = np.zeros(n)
    np.add.at(pg_bus1, gbus[ins], pg_gen[ins])
    # |V| setpoints: case-file Vg of the in-service gens at the bus (first one); REF same
    vset = np.ones(n)
    for j in np.where(ins)[0][::-1]:
        vset[gbus[j]] = base["vg"][gidx[j]]
    # ---- B3 DC with B1's dispatch
    va_dc = dc_angles(br, bus, pg_bus1 - Pd, ref, va_t)
    vm_dc = np.where(np.isin(np.arange(n), np.r_[pv, ref]), vset, 1.0)
    out["B3_dc"] = {"vm": vm_dc, "va": va_dc, "pgb": pg_bus1, "converged": True}
    # ---- B1, B2
    r1 = run("B1", pg_bus1, vset, va_dc)
    vset2 = vset.copy()
    m_fix = meas & np.isin(np.arange(n), np.r_[pv, ref])
    vset2[m_fix] = vm_t[m_fix]
    r2 = run("B2", pg_bus1, vset2, va_dc)
    for name, r in (("B1_ac", r1), ("B2_ac_meas", r2)):
        if r is None:
            out[name] = dict(out["B3_dc"], converged=False)       # fallback to DC, flagged
        else:
            out[name] = r
    # ---- oracle dispatch + setpoints (control): label bus Pg, label Vm at PV/REF
    pg_lab = bus.Pg.to_numpy()
    ro = run("oracle", pg_lab, vm_t, va_dc)
    out["oracle"] = ro if ro is not None else dict(out["B3_dc"], converged=False)
    out["flat"] = {"vm": np.ones(n), "va": np.full(n, va_t[ref].mean()), "pgb": pg_bus1, "converged": True}
    for v in out.values():
        v["pg"] = split_to_gens(v["pgb"], gen, bpg)
    return out


def score(tab: dict, sid: int, vm_p, va_p, pg_gen_p, frac: float = FRAC) -> dict:
    """pg_gen_p: (n_gen,) in gen_data row order (out-of-service entries ignored)."""
    bus, gen, br = tab["bus"][sid], tab["gen"][sid], tab["br"][sid]
    n = len(bus)
    meas = np.zeros(n, bool)
    meas[measured_set(sid, n, frac)] = True
    vm_t, va_t = bus.Vm.to_numpy(), bus.Va.to_numpy()
    b = br[br.br_status == 1]
    f, t = b.from_bus.to_numpy(), b.to_bus.to_numpy()
    e_br = np.abs(wrap((va_p[f] - va_p[t]) - (va_t[f] - va_t[t])))
    ins = gen.in_service.to_numpy() == 1
    out = {"dva_mae": float(e_br.mean()), "pg_mae": float(np.abs(pg_gen_p[ins] - gen.p_mw.to_numpy()[ins]).mean()),
           "pd_total": float(bus.Pd.sum())}
    out["vm_mae_unmeas"] = float(np.abs(vm_p - vm_t)[~meas].mean()) if (~meas).any() else float("nan")
    return out
