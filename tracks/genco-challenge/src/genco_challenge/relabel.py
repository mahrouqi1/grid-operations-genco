"""Relabel collapsed datakit labels with the physical root our oracle finds (2026-10-08).

``ladder_data.validate_grid`` DROPS a scenario whose AC label is a collapsed root (some Vm < 0.5 pu or
negative). This module builds a SEPARATE data variant, ``WORK/data-relabeled/<grid>/``, in which such a
scenario instead gets a new label: the physical root of its own Ybus/Sbus, if one exists. The validated
datakit-only data under ``WORK/data/`` is never touched and stays the primary set.

Source: the earliest ``raw.unvalidated-<time>`` of the grid (the pre-validation raw, all scenarios).
Scenarios whose label is already an operating point keep their datakit label unchanged.

**Acceptance rule** (``accept_relabel``): a candidate root is physical if it converged, its power-mismatch
max-norm is < 1e-9 pu, and every |V| >= 0.5 (the sign of Vm is carried by the representation, see below).
A new label is accepted only if TWO independent starts reach physical roots that agree to max |dV| < 1e-6
pu. Starts, in order: flat, DC (datakit's ``Va_dc``), then only if needed damped-Newton from flat, the
neighbour-repaired collapsed label, and three starts from a SHIFT-AWARE DC solution (``shift_dc_angles``: plain
Newton, Newton from a perturbed copy, damped Newton from each). Otherwise the scenario is dropped, with the reason
recorded. The shift-aware DC start is what rescues the RTE grids: datakit's ``Va_dc`` ignores phase shifters.

**Consistent write-back.** Bus types, Pd/Qd, the specified quantities (Vm at PV/REF, Va at REF, P at PV,
P and Q at PQ) are unchanged. What changes: bus Vm/Va; PV/REF reactive generation and REF active generation
implied by the new root (bus ``Qg``/``Pg`` and the matching ``gen_data`` rows: PV Q on the first in-service
generator at the bus, REF P and Q on the last slack-flagged in-service generator, other generators keep
their values, which is how datakit books them); branch ``pf/qf/pt/qt`` recomputed from the branch
admittances. ``Va_dc``, ``Pg_dc``, ``p_mw_dc``, y_bus and runtime are untouched.

Run (env ``grid-operations-genco``, from the repo root, PYTHONPATH = tracks/genco-challenge/src):

    python -m genco_challenge.relabel                       # every grid whose validation dropped scenarios
    python -m genco_challenge.relabel --names pglib_opf_case240_pserc
    python -m genco_challenge.ladder_data --relabel          # same entry point
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from genco_challenge.hybrid import Scenario, newton_raphson_pf
from genco_challenge.ladder_data import (N_PER_PARTITION, SCEN_TABLES, VM_MIN, WORK, label_diagnostics)

RES_TOL = 1e-9     # pu, power-mismatch max-norm of an accepted root
AGREE_TOL = 1e-6   # pu, max |dV| between the two independent starts
ANG_MAX = 90.0     # deg, plausibility guard (beyond the approved rule): max angle across an in-service branch.
#                    Valid datakit labels reach 64 deg at most (measured on 8 grids); the spurious roots found on
#                    case1951_rte (scenarios 3 to 5) sit at 150 to 155 deg.
RELABEL_ROOT = WORK / "data-relabeled"
BASE_MVA = 100.0


# ---------------------------------------------------------------------------- acceptance rule

def mismatch_max(ybus, sbus, v, pv, pq) -> float:
    """Max-norm of (P at PV+PQ, Q at PQ) mismatch, per unit."""
    ds = v * np.conj(ybus @ v) - sbus
    pvpq = np.r_[pv, pq]
    f = np.r_[ds[pvpq].real, ds[pq].imag]
    return float(np.max(np.abs(f))) if f.size else 0.0


def judge_candidate(name, v, ybus, sbus, pv, pq, *, res_tol=RES_TOL, vm_min=VM_MIN, ang_fn=None,
                    ang_max=ANG_MAX) -> dict:
    """Is one start's end point a physical root? ``v`` is None when the solver failed or diverged."""
    if v is None or not np.all(np.isfinite(v)):
        return {"start": name, "converged": False, "physical": False, "why": "no_convergence"}
    res = mismatch_max(ybus, sbus, v, pv, pq)
    vmin = float(np.abs(v).min())
    why = "" if (res < res_tol and vmin >= vm_min) else ("residual" if res >= res_tol else "collapsed")
    out = {"start": name, "converged": res < res_tol, "residual": res, "min_vm": vmin}
    if ang_fn is not None and why == "":
        out["max_branch_angle_deg"] = ang = float(ang_fn(v))
        if ang > ang_max:
            why = "implausible_angle"
    return out | {"physical": why == "", "why": why or "ok"}


def start_group(name: str) -> str:
    """Starts sharing an initial point (plain and damped Newton from it) are one group."""
    return name.removeprefix("damped_")


def accept_relabel(cands, ybus, sbus, pv, pq, *, res_tol=RES_TOL, vm_min=VM_MIN, agree_tol=AGREE_TOL,
                   ang_fn=None, ang_max=ANG_MAX) -> dict:
    """Acceptance rule. ``cands`` = [(start_name, v or None), ...] in preference order.

    Accept iff two starts from DIFFERENT initial points (``start_group``) end at physical roots (residual < res_tol, every |V| >= vm_min) that
    agree to max |dV| < agree_tol. Returns {accepted, v, reason, pair, max_dv, candidates}; the label is
    the first start of the agreeing pair. ``reason`` when rejected is one of no_start_converged,
    collapsed_roots_only, implausible_angle_roots_only, one_physical_root, physical_roots_disagree.
    ``ang_fn(v)`` (optional) returns the largest angle across an in-service branch in degrees; a root above
    ``ang_max`` is not physical (extra guard, see ``ANG_MAX``).
    """
    info = [judge_candidate(n, v, ybus, sbus, pv, pq, res_tol=res_tol, vm_min=vm_min, ang_fn=ang_fn, ang_max=ang_max)
            for n, v in cands]
    phys = [(i, n, v) for i, ((n, v), c) in enumerate(zip(cands, info)) if c["physical"]]
    out = {"accepted": False, "v": None, "pair": None, "max_dv": None, "candidates": info}
    best = None
    for a in range(len(phys)):
        for b in range(a + 1, len(phys)):
            if start_group(phys[a][1]) == start_group(phys[b][1]):
                continue  # same initial point, only a different iteration: not independent
            dv = float(np.abs(phys[a][2] - phys[b][2]).max())
            best = dv if best is None else min(best, dv)
            if dv < agree_tol:
                return out | {"accepted": True, "v": phys[a][2], "pair": [phys[a][1], phys[b][1]],
                              "max_dv": dv, "reason": "relabeled"}
    out["max_dv"] = best
    if len(phys) >= 2:
        out["reason"] = "physical_roots_disagree"
    elif len(phys) == 1:
        out["reason"] = "one_physical_root"
    elif any(c["why"] == "implausible_angle" for c in info):
        out["reason"] = "implausible_angle_roots_only"
    elif any(c["converged"] for c in info):
        out["reason"] = "collapsed_roots_only"
    else:
        out["reason"] = "no_start_converged"
    return out


# ---------------------------------------------------------------------------- per-scenario solve

def _newton(sc: Scenario, v0, max_it=30):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            r = newton_raphson_pf(sc.ybus, sc.sbus, v0, sc.pv, sc.pq, tol=1e-11, max_it=max_it)
        except Exception:
            return None
    return r.v if np.all(np.isfinite(r.v)) else None


def _damped(sc: Scenario, v0, max_it=100, tol=1e-10):
    """Newton with backtracking on ||F||_2 (halving down to 1/1024) and a floor of 0.05 pu on PQ |V|, so the
    iteration cannot cross zero voltage into a collapsed (negative-Vm) root. Own Jacobian step (repo's
    ``_sparse_jacobian``), because the repo solver's one-step call updates |V| without such a floor."""
    import scipy.sparse.linalg as spla
    from grid_ops.reference_pf import _sparse_jacobian
    pvpq, pq = sc.pvpq, sc.pq
    k = len(pvpq)
    v = v0.astype(complex).copy()

    def fvec(x):
        a, b = sc.f(x)
        return np.r_[a, b]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for _ in range(max_it):
            f = fvec(v)
            if not np.all(np.isfinite(f)):
                return None
            if np.max(np.abs(f)) < tol:
                return v
            try:
                dx = spla.spsolve(_sparse_jacobian(sc.ybus, v, pvpq, pq).tocsc(), -f)
            except Exception:
                return None
            if not np.all(np.isfinite(dx)):
                return None
            f0, t, moved = np.linalg.norm(f), 1.0, False
            va, vm = np.angle(v), np.abs(v)
            while t > 1 / 1024:
                va2, vm2 = va.copy(), vm.copy()
                va2[pvpq] += t * dx[:k]
                vm2[pq] += t * dx[k:]
                vt = vm2 * np.exp(1j * va2)
                if np.all(vm2[pq] > 0.05) and np.linalg.norm(fvec(vt)) < f0:
                    v, moved = vt, True
                    break
                t /= 2
            if not moved:
                return None
    return v if np.max(np.abs(fvec(v))) < tol else None


def _repaired_start(sc: Scenario):
    """Collapsed label with collapsed buses replaced by the mean of live neighbours (hybrid.py's idea)."""
    low = sc.vm_label < VM_MIN
    adj = sc.ybus.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    v = sc.v_label.copy()
    for _ in range(sc.n):
        todo = np.where(low)[0]
        if not len(todo):
            break
        moved = False
        for k in todo:
            nb = [j for j in adj.indices[adj.indptr[k]:adj.indptr[k + 1]] if not low[j]]
            if nb:
                v[k], low[k], moved = np.mean(v[nb]), False, True
        if not moved:
            return None
    return v


def branch_angle_fn(br: pd.DataFrame):
    """v -> largest |angle difference| (degrees, wrapped) across in-service branches."""
    on = br["br_status"].to_numpy() == 1
    f, t = br["from_bus"].to_numpy()[on], br["to_bus"].to_numpy()[on]
    return lambda v: float(np.abs(np.degrees(np.angle(v[f] * np.conj(v[t])))).max()) if len(f) else 0.0


def shift_dc_angles(sc: Scenario, br: pd.DataFrame) -> np.ndarray:
    """DC power-flow angles WITH phase-shifter injections (MATPOWER ``makeBdc``: susceptance 1/(x*tap) and
    Pbusinj from the shift angles). datakit's ``Va_dc`` ignores phase shifters (on case1888_rte the angle across
    a -9.95 degree shifter is -0.04 degrees), so a Newton start from it is off by up to the shift on the grids
    that have them (RTE), and Newton diverges to a collapsed root or not at all."""
    f, t = br["from_bus"].to_numpy(), br["to_bus"].to_numpy()
    on = (br["br_status"].to_numpy() == 1).astype(float)
    tap = np.where(br["tap"].to_numpy() == 0, 1.0, br["tap"].to_numpy())
    b = on / (br["x"].to_numpy() * tap)
    i = np.arange(len(f))
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    cft = sp.csr_matrix((np.r_[np.ones(len(f)), -np.ones(len(f))], (np.r_[i, i], np.r_[f, t])), shape=(len(f), sc.n))
    bf = sp.csr_matrix((np.r_[b, -b], (np.r_[i, i], np.r_[f, t])), shape=(len(f), sc.n))
    bbus = (cft.T @ bf).tocsr()
    pbusinj = cft.T @ (b * (-np.radians(br["shift"].to_numpy())))
    pvpq = sc.pvpq
    th = np.full(sc.n, sc.va_ref)
    rhs = sc.sbus.real - pbusinj
    th[pvpq] = spla.spsolve(bbus[pvpq, :][:, pvpq].tocsc(),
                            rhs[pvpq] - bbus[pvpq, :][:, sc.ref] @ np.full(len(sc.ref), sc.va_ref))
    return th


def solve_scenario(sc: Scenario, br: pd.DataFrame | None = None) -> dict:
    """Run the starts and apply the acceptance rule. Extra starts only when flat + DC do not agree.
    With ``br`` (the scenario's full branch table) the shift-aware DC starts are added to the extras."""
    ang_fn = branch_angle_fn(br) if br is not None else None
    flat = np.full(sc.n, sc.va_ref)
    starts = [("flat", sc.vm_set * np.exp(1j * flat)), ("dc", sc.vm_set * np.exp(1j * sc.va_dc))]
    cands = [(n, _newton(sc, v0)) for n, v0 in starts]
    res = accept_relabel(cands, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=ang_fn)
    if not res["accepted"]:
        cands.append(("damped_flat", _damped(sc, starts[0][1])))
        if br is not None:
            th = shift_dc_angles(sc, br)
            v_sdc = sc.vm_set * np.exp(1j * th)
            pert = np.where(sc.is_t["PQ"], 0.97, sc.vm_set) * np.exp(1j * (sc.va_ref + 1.1 * (th - sc.va_ref)))
            cands += [("dc_shift", _newton(sc, v_sdc)), ("dc_shift_pert", _newton(sc, pert)),
                      ("damped_dc_shift", _damped(sc, v_sdc)), ("damped_dc_shift_pert", _damped(sc, pert))]
        rep = _repaired_start(sc)
        cands.append(("repaired_label", _newton(sc, rep) if rep is not None else None))
        res = accept_relabel(cands, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=ang_fn)
    res["cands"], res["ang_fn"] = cands, ang_fn
    return res


def homotopy_root(sc: Scenario, src: Scenario, v_src: np.ndarray, h0: float = 0.1, hmin: float = 1e-4, max_steps=400):
    """Natural-parameter continuation from a solved scenario ``src`` (root ``v_src``) to ``sc`` along the straight
    line in (Ybus, Sbus, PV/REF setpoints), with ``sc``'s bus types. Newton corrector with step halving; returns
    the root at lambda = 1 or None (nose reached or step below ``hmin``)."""
    fixed = sc.is_t["PV"] | sc.is_t["REF"]
    vm0 = np.where(fixed, np.abs(v_src), 1.0)   # source magnitudes where the target fixes them
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
        if r is not None and r.converged and np.all(np.isfinite(r.v)) and np.abs(r.v).min() > 0.3 \
                and np.abs(r.v - v0).max() < 0.5:
            v, lam = r.v, ln
            if lam >= 1.0:
                return v
            h = min(h * 1.5, 0.25)
        else:
            h /= 2
            if h < hmin:
                return None
    return None


def neighbor_pass(sc: Scenario, res: dict, pool: dict[int, np.ndarray], me: int, k: int = 4,
                  sources: list[tuple[int, Scenario, np.ndarray]] = ()) -> dict:
    """Second pass for a scenario still without two agreeing roots: add starts from OTHER scenarios' physical
    roots of the same grid (``pool``: scenario id -> a root of that scenario; the nearest ids first), Newton
    and damped Newton. A neighbour's root is a start unrelated to this scenario's DC solution, so agreement
    with it is independent confirmation; the neighbour root itself is never emitted for scenario ``me``."""
    cands = list(res["cands"])
    fixed = sc.is_t["PV"] | sc.is_t["REF"]
    for j in sorted((j for j in pool if j != me), key=lambda j: abs(j - me))[:k]:
        # the neighbour's angles and PQ magnitudes, THIS scenario's PV/REF setpoints (Newton keeps them fixed)
        v0 = np.where(fixed, sc.vm_set, np.abs(pool[j])) * np.exp(1j * np.angle(pool[j]))
        cands.append((f"neighbor{j}", _newton(sc, v0)))
        cands.append((f"damped_neighbor{j}", _damped(sc, v0)))
    for j, src, v_src in sources:   # continuation from healthy sibling scenarios (their labels are roots)
        cands.append((f"homotopy{j}", homotopy_root(sc, src, v_src)))
    out = accept_relabel(cands, sc.ybus, sc.sbus, sc.pv, sc.pq, ang_fn=res.get("ang_fn"))
    out["cands"], out["ang_fn"] = cands, res.get("ang_fn")
    return out


# ---------------------------------------------------------------------------- write-back

def new_label_tables(sc: Scenario, v: np.ndarray, bus: pd.DataFrame, gen: pd.DataFrame, br: pd.DataFrame):
    """Return updated copies of one scenario's bus/gen/branch tables for root ``v``, plus diagnostics."""
    bus = bus.sort_values("bus").reset_index(drop=True).copy()
    gen, br = gen.copy(), br.copy()
    s = v * np.conj(sc.ybus @ v) * BASE_MVA
    pd_, qd_ = bus["Pd"].to_numpy(), bus["Qd"].to_numpy()
    bus["Vm"], bus["Va"] = np.abs(v), np.degrees(np.angle(v))
    ref, pv = sc.ref, sc.pv
    old_pg_ref = float(bus.loc[ref, "Pg"].sum())
    bus.loc[ref, "Pg"] = (s.real + pd_)[ref]
    bus.loc[np.r_[ref, pv], "Qg"] = (s.imag + qd_)[np.r_[ref, pv]]
    # generator rows
    gen.index = np.arange(len(gen))
    for b in np.r_[pv, ref]:
        on = np.where((gen["bus"].to_numpy() == b) & (gen["in_service"].to_numpy() == 1))[0]
        if not len(on):
            raise ValueError(f"PV/REF bus {b} has no in-service generator")
        isref = b in set(ref.tolist())
        sl = [i for i in on if gen.at[i, "is_slack_gen"] == 1]
        tgt = (sl[-1] if (isref and sl) else on[0])
        others_q = float(gen.loc[[i for i in on if i != tgt], "q_mvar"].sum())
        gen.at[tgt, "q_mvar"] = float(bus.at[b, "Qg"]) - others_q
        if isref:
            others_p = float(gen.loc[[i for i in on if i != tgt], "p_mw"].sum())
            gen.at[tgt, "p_mw"] = float(bus.at[b, "Pg"]) - others_p
    # branch flows
    fb, tb = br["from_bus"].to_numpy(), br["to_bus"].to_numpy()
    c = lambda k: br[f"{k}_r"].to_numpy() + 1j * br[f"{k}_i"].to_numpy()  # noqa: E731
    on_ = (br["br_status"].to_numpy() == 1)
    sf = v[fb] * np.conj(c("Yff") * v[fb] + c("Yft") * v[tb]) * BASE_MVA
    st = v[tb] * np.conj(c("Ytf") * v[fb] + c("Ytt") * v[tb]) * BASE_MVA
    for col, val in (("pf", sf.real), ("qf", sf.imag), ("pt", st.real), ("qt", st.imag)):
        br[col] = np.where(on_, val, br[col].to_numpy())
    qg = gen[gen["in_service"] == 1].groupby("bus")["q_mvar"].sum().reindex(np.r_[pv, ref])
    qmax = gen[gen["in_service"] == 1].groupby("bus")["max_q_mvar"].sum().reindex(np.r_[pv, ref])
    qmin = gen[gen["in_service"] == 1].groupby("bus")["min_q_mvar"].sum().reindex(np.r_[pv, ref])
    viol = np.maximum(0.0, np.maximum(qg - qmax, qmin - qg)).to_numpy()
    dang = np.degrees(np.angle(v[fb] * np.conj(v[tb])))[on_]
    diag = {"slack_pg_mw_old": old_pg_ref, "slack_pg_mw_new": float(bus.loc[ref, "Pg"].sum()),
            "max_q_limit_violation_mvar": float(viol.max()) if len(viol) else 0.0,
            "max_branch_angle_diff_deg": float(np.abs(dang).max()) if len(dang) else 0.0,
            "min_vm": float(np.abs(v).min()), "max_vm": float(np.abs(v).max())}
    return bus, gen, br, diag


def check_consistency(sc: Scenario, bus_old, bus_new, gen_new) -> dict:
    """Specified quantities unchanged; bus Pg/Qg equal the in-service generator sums at PV/REF buses."""
    bo = bus_old.sort_values("bus").reset_index(drop=True)
    bn = bus_new.sort_values("bus").reset_index(drop=True)
    for c in ("Pd", "Qd", "PQ", "PV", "REF", "GS", "BS", "min_vm_pu", "max_vm_pu", "Va_dc", "Pg_dc"):
        assert np.array_equal(bo[c].to_numpy(), bn[c].to_numpy()), f"{c} changed"
    sp_ = np.r_[sc.pv, sc.ref]
    spec_gap = float(np.abs(bn.loc[sp_, "Vm"].to_numpy() - bo.loc[sp_, "Vm"].to_numpy()).max())
    ref_va_gap = float(np.abs(bn.loc[sc.ref, "Va"].to_numpy() - bo.loc[sc.ref, "Va"].to_numpy()).max())
    pvq_gap = float(np.abs(bn.loc[sc.pv, "Pg"].to_numpy() - bo.loc[sc.pv, "Pg"].to_numpy()).max()) if len(sc.pv) else 0.0
    pq_gap = float(np.abs(bn.loc[sc.pq, ["Pg", "Qg"]].to_numpy() - bo.loc[sc.pq, ["Pg", "Qg"]].to_numpy()).max())
    on = gen_new[gen_new["in_service"] == 1].groupby("bus")[["p_mw", "q_mvar"]].sum()
    gs = bn.set_index("bus").join(on).fillna({"p_mw": 0.0, "q_mvar": 0.0})
    sum_gap = float(max(np.abs(gs["Pg"] - gs["p_mw"]).max(), np.abs(gs["Qg"] - gs["q_mvar"]).max()))
    out = {"specified_vm_gap": spec_gap, "ref_va_gap_deg": ref_va_gap, "pv_pg_gap_mw": pvq_gap,
           "pq_pg_qg_gap": pq_gap, "bus_vs_gen_sum_gap_mw": sum_gap}
    out["ok"] = bool(spec_gap < 1e-12 and ref_va_gap < 1e-9 and pvq_gap < 1e-9 and pq_gap < 1e-9 and sum_gap < 1e-6)
    return out


# ---------------------------------------------------------------------------- one grid

def source_raw(name: str, data_root: Path | None = None) -> Path:
    cand = sorted(glob.glob(str((data_root or WORK / "data") / name / "raw.unvalidated-*")))
    if not cand:
        raise FileNotFoundError(f"{name}: no raw.unvalidated-* (nothing was dropped, nothing to relabel)")
    return Path(cand[0])  # earliest = the original, all generated scenarios


def _write_table(df: pd.DataFrame, dst: Path, keep_order: np.ndarray) -> int:
    new_id = pd.Series(np.arange(len(keep_order), dtype="int64"), index=keep_order)
    df = df[df["scenario"].isin(keep_order)].copy()
    df["scenario"] = new_id.reindex(df["scenario"].to_numpy()).to_numpy()
    df = df.sort_values("scenario", kind="stable").reset_index(drop=True)
    df["scenario_partition"] = (df["scenario"] // N_PER_PARTITION).astype("int64")
    df.to_parquet(dst, partition_cols=["scenario_partition"], engine="pyarrow", index=False)
    return len(df)


def code_provenance() -> dict:
    """Git sha of the repo plus a dirty flag over the modules a relabel run imports (relabel, hybrid,
    ladder_data). Never raises: on any git failure the fields say so instead of hiding the gap."""
    import subprocess
    here = Path(__file__).resolve().parent
    mods = [here / f"{m}.py" for m in ("relabel", "hybrid", "ladder_data")]

    def git(*a):
        return subprocess.run(["git", "-C", str(here), *a], capture_output=True, text=True, timeout=60)
    try:
        sha = git("rev-parse", "HEAD").stdout.strip() or "unknown"
        st = git("status", "--porcelain", "--", *map(str, mods)).stdout.strip().splitlines()
        return {"git_sha": sha, "code_dirty": bool(st), "dirty_modules": [ln[3:].strip() for ln in st]}
    except Exception as e:  # noqa: BLE001
        return {"git_sha": "unknown", "code_dirty": None, "dirty_modules": [], "error": repr(e)}


def relabel_grid(name: str, *, root: Path = RELABEL_ROOT, out_json: Path | None = None,
                 data_root: Path | None = None) -> dict:
    """``data_root`` = directory holding ``<name>/{meta.json,raw.unvalidated-*}`` (default ``WORK/data``;
    the extended RTE sets of 2026-10-09 live in ``WORK/data-rte2``)."""
    src = source_raw(name, data_root)
    gdir = root / name
    raw = gdir / "raw"
    meta = json.loads(((data_root or WORK / "data") / name / "meta.json").read_text())
    t0 = time.time()
    bus_all = pd.read_parquet(src / "bus_data.parquet")
    gen_all = pd.read_parquet(src / "gen_data.parquet")
    br_all = pd.read_parquet(src / "branch_data.parquet")
    yb_all = pd.read_parquet(src / "y_bus_data.parquet")
    diag = label_diagnostics(bus_all[["scenario", "bus", "Vm", "PV", "REF"]])
    ids = diag.index.to_numpy()
    bad = [int(s) for s in ids[diag["invalid"].to_numpy()]]
    by_b = dict(tuple(bus_all.groupby("scenario")))
    by_g = dict(tuple(gen_all.groupby("scenario")))
    by_br = dict(tuple(br_all.groupby("scenario")))
    by_y = dict(tuple(yb_all.groupby("scenario")))
    per, upd_b, upd_g, upd_br = [], [], [], []
    healthy = [int(x) for x in ids if int(x) not in set(bad)]
    pass1: dict[int, tuple] = {}   # scenario -> (Scenario, result, branch table), for the neighbour pass
    pool: dict[int, np.ndarray] = {}
    for s in bad:
        sc = Scenario(by_b[s], by_y[s], by_br[s])
        res = solve_scenario(sc, by_br[s].sort_values("idx").reset_index(drop=True))
        pass1[s] = (sc, res)
        phys = [v for (n, v), c in zip(res["cands"], res["candidates"]) if c["physical"]]
        if phys:
            pool[s] = phys[0]
    for s in bad:
        rec = {"orig_scenario": s, "label_reason": diag.at[s, "reason"], "label_min_vm": float(diag.at[s, "min_vm"]),
               "label_argmin_bus": int(diag.at[s, "argmin_bus"])}
        sc, res = pass1[s]
        if not res["accepted"] and (pool or healthy):
            srcs = []
            for j in sorted(healthy, key=lambda j: abs(j - s))[:3]:
                sj = Scenario(by_b[j], by_y[j], by_br[j])
                srcs.append((j, sj, sj.v_label))
            for j in sorted((j for j in pool if j != s), key=lambda j: abs(j - s))[:2]:   # plausible roots found in pass 1
                srcs.append((j, pass1[j][0], pool[j]))
            res = neighbor_pass(sc, res, pool, s, sources=srcs)
        rec |= {"outcome": "relabeled" if res["accepted"] else "no_physical_root", "reason": res["reason"],
                "pair": res["pair"], "max_dv_between_starts": res["max_dv"], "candidates": res["candidates"]}
        if res["accepted"]:
            b2, g2, br2, dg = new_label_tables(sc, res["v"], by_b[s], by_g[s], by_br[s])
            cc = check_consistency(sc, by_b[s], b2, g2)
            rec |= {"new_label": dg, "consistency": cc}
            if not cc["ok"]:  # never emit an inconsistent label
                rec |= {"outcome": "no_physical_root", "reason": "writeback_inconsistent"}
            else:
                upd_b.append((s, b2)), upd_g.append((s, g2)), upd_br.append((s, br2))
        per.append(rec)
    relabeled = {s for s, _ in upd_b}
    keep = np.array([int(s) for s in ids if s not in set(bad) or s in relabeled], dtype="int64")

    def patch(df, upd, cols):
        if not upd:
            return df
        df = df.copy()
        for s, new in upd:
            m = (df["scenario"] == s).to_numpy()
            old_idx = df.index[m]
            key = "bus" if "Vm" in new.columns else "idx"
            new = new.set_index(key)
            cur = df.loc[old_idx, key].to_numpy()
            for c in cols:
                df.loc[old_idx, c] = new.loc[cur, c].to_numpy()
        return df
    bus_all = patch(bus_all, upd_b, ["Vm", "Va", "Pg", "Qg"])
    gen_all = patch(gen_all, upd_g, ["p_mw", "q_mvar"])
    br_all = patch(br_all, upd_br, ["pf", "qf", "pt", "qt"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    gdir.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raw.rename(gdir / f"raw.superseded-{stamp}")
    if (gdir / "processed").exists():   # features/targets built from the raw being replaced: make unselectable
        (gdir / "processed").rename(gdir / f"processed.stale-{stamp}")   # rename only, never delete
    shutil.copytree(src, raw, ignore=shutil.ignore_patterns(*(f"{t}.parquet" for t in SCEN_TABLES)))
    tabs = {"bus_data": bus_all, "gen_data": gen_all, "branch_data": br_all, "y_bus_data": yb_all,
            "runtime_data": pd.read_parquet(src / "runtime_data.parquet")}
    for t, df in tabs.items():
        if not len(keep):  # nothing usable: leave the tables out, as validate_grid does
            break
        _write_table(df.drop(columns=["scenario_partition"], errors="ignore"), raw / f"{t}.parquet", keep)
    ls = bus_all.groupby("scenario")["load_scenario_idx"].first()
    smap = pd.DataFrame({"scenario": np.arange(len(keep)), "orig_scenario": keep,
                         "load_scenario_idx": ls.reindex(keep).to_numpy(),
                         "relabeled": [int(s in relabeled) for s in keep]})
    smap.to_csv(raw / "scenario_map.csv", index=False)
    pd.DataFrame([{k: r[k] for k in ("orig_scenario", "outcome", "reason", "label_reason", "label_min_vm")}
                  for r in per]).to_csv(raw / "relabel_log.csv", index=False)
    (raw / "n_scenarios.txt").write_text(str(len(keep)))
    if len(keep):
        chk = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "Vm"])
        assert chk["scenario"].nunique() == len(keep) and chk["scenario"].max() == len(keep) - 1
        assert chk["Vm"].notna().all() and chk["Vm"].min() >= VM_MIN
    n_ok = sum(r["outcome"] == "relabeled" for r in per)
    reasons: dict[str, int] = {}
    for r in per:
        if r["outcome"] != "relabeled":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summary = {"name": name, "source_raw": src.name, "generated": int(len(ids)),
               "collapsed_labels": len(bad), "relabeled": n_ok, "no_physical_root": len(bad) - n_ok,
               "no_root_reasons": reasons, "usable_before_validated_primary": int(len(ids) - len(bad)),
               "usable_after_relabel": int(len(keep)),
               "status": "ok" if len(keep) >= 20 else "insufficient_valid_scenarios",
               "seconds": round(time.time() - t0, 1), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
               "provenance": code_provenance()}
    (gdir / "provenance.json").write_text(json.dumps(summary["provenance"] | {"when": summary["when"]}, indent=1))
    (gdir / "meta.json").write_text(json.dumps(
        meta | {"scenarios_written": int(len(keep)), "status": summary["status"], "relabel": summary,
                "variant": "relabeled (collapsed labels replaced by the physical root; see genco_challenge.relabel)"},
        indent=1))
    if out_json is not None:
        out_json.write_text(json.dumps({"summary": summary, "scenarios": per}, indent=1, default=float))
    return summary


def grids_with_drops() -> list[str]:
    out = []
    for p in sorted((WORK / "data").glob("*/meta.json")):
        v = json.loads(p.read_text()).get("validation", {})
        if v.get("dropped", 0) > 0:
            out.append(p.parent.name)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--out", type=Path, default=None, help="directory for per-scenario <grid>.json files")
    ap.add_argument("--data-root", type=Path, default=None, help="validated-data root (default WORK/data)")
    a = ap.parse_args(argv)
    for n in a.names or grids_with_drops():
        oj = None
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            oj = a.out / f"{n}.json"
        print(json.dumps(relabel_grid(n, out_json=oj, data_root=a.data_root)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
