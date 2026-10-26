"""Block C (UNAUDITED): classical pipelines, per scenario, in-memory per-scenario tables, one process, single thread.
    taskset -c 16-19 python t_classical.py <grid> [--n 40]
Per timed scenario (5 untimed warm-up scenarios first, then n distinct scenarios, cycled if the grid has fewer):
  ybus_csr      scipy csr build of Ybus from the datakit (index1,index2,G,B) table only
  assembly      lp2.input_problem: Ybus + Sbus + B' + B'' slicing (the 'setup' of every pipeline)
  dc            our own DC solve spsolve(B', P) (datakit's stored Va_dc is NOT used: it comes from a PowerModels run)
  fd2_iters     two XB fast-decoupled iterations from the DC state including the two LU factorisations
  fd3_extra     one more iteration (LU cached)
  nr_flat / nr_dc / fd2nr   hy.solve(...) to tol (hybrid.TOL), seconds as measured inside solve, + iterations, converged
Totals per scenario are sums of components of the same scenario; medians/IQR are over scenarios."""
import argparse, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.linalg as spla
from tcommon import *
from genco_challenge import hybrid as hy
from genco_challenge import labelfree_prior2 as lp2

WARM = 5


def load_tables(root, name):
    bus, ys, brs = lp2._grid_tables(root, name)
    bus_by = dict(tuple(bus.groupby("scenario")))
    return bus_by, ys, brs


def assemble(b, y, br):
    t0 = time.perf_counter()
    sc = lp2.input_problem(b, y, br)
    return sc, time.perf_counter() - t0


def dc_state(sc):
    t0 = time.perf_counter()
    th = np.full(sc.n, sc.va_ref)
    th[sc.pvpq] = sc.va_ref + spla.spsolve(sc._bp, sc.sbus.real[sc.pvpq])
    sc.va_dc = th
    v = sc.vm_set * np.exp(1j * th)
    return v, time.perf_counter() - t0


def fd_iters(sc, v0, k):
    sc._bp_lu = sc._bpp_lu = None
    t0 = time.perf_counter()
    v = v0.copy()
    for _ in range(k):
        v = sc.fdxb_step(v)
    return v, time.perf_counter() - t0


def label_of(b):
    b = b.sort_values("bus")
    return b["Vm"].to_numpy() * np.exp(1j * np.radians(b["Va"].to_numpy()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    root, name, fam = GRIDS[a.grid]
    with Block("classical", grid=a.grid):
        bus_by, ys, brs = load_tables(root, name)
        scen = sorted(bus_by)
        need = WARM + a.n
        order = [scen[i % len(scen)] for i in range(need)]
        R = {k: [] for k in ("ybus_csr", "assembly", "dc", "fd2_iters", "fd3_extra", "nr_flat", "nr_dc", "fd2nr", "nr1")}
        meta = {k: [] for k in ("it_flat", "ok_flat", "it_dc", "ok_dc", "it_fd2nr", "ok_fd2nr", "dv_flat", "dv_dc", "dv_fd2nr")}
        for i, s in enumerate(order):
            b, y, br = bus_by[s], ys[s], brs[s]
            t0 = time.perf_counter()
            sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(), (y["index1"].to_numpy(), y["index2"].to_numpy())),
                          shape=(len(b), len(b)))
            t_y = time.perf_counter() - t0
            sc, t_as = assemble(b, y, br)
            v_dc, t_dc = dc_state(sc)
            v2, t_fd2 = fd_iters(sc, v_dc, 2)
            t0 = time.perf_counter(); v3 = sc.fdxb_step(v2); t_fd3 = time.perf_counter() - t0
            t0 = time.perf_counter(); hy.nr_ls_step(sc, v_dc); t_nr1 = time.perf_counter() - t0
            lab = label_of(b)
            sc.v_label, sc.vm_label, sc.va_label = lab, np.abs(lab), np.angle(lab)
            sc.v_ref, sc.ref_kind = None, "input_only"
            v_flat = sc.vm_set * np.exp(1j * np.full(sc.n, sc.va_ref))
            w_f = hy.solve(sc, v_flat, "nr")
            w_d = hy.solve(sc, v_dc, "nr")
            sc._bp_lu = sc._bpp_lu = None
            w_2 = hy.solve(sc, v_dc, "fd2nr")
            if i < WARM:
                continue
            for k, d in zip(R, (t_y, t_as, t_dc, t_fd2, t_fd3, w_f["seconds"], w_d["seconds"], w_2["seconds"], t_nr1)):
                R[k].append(d)
            for tag, w in (("flat", w_f), ("dc", w_d), ("fd2nr", w_2)):
                meta[f"it_{tag}"].append(w["iterations"]); meta[f"ok_{tag}"].append(bool(w["converged"]))
                meta[f"dv_{tag}"].append(w["max_dv_label"])
        tot = {
            "dc_total": [x + y for x, y in zip(R["assembly"], R["dc"])],
            "fd2_total": [a_ + b_ + c for a_, b_, c in zip(R["assembly"], R["dc"], R["fd2_iters"])],
            "fd3_total": [a_ + b_ + c + d for a_, b_, c, d in zip(R["assembly"], R["dc"], R["fd2_iters"], R["fd3_extra"])],
            "newton_flat_total": [x + y for x, y in zip(R["assembly"], R["nr_flat"])],
            "newton_dc_total": [x + y + z for x, y, z in zip(R["assembly"], R["dc"], R["nr_dc"])],
            "fd2nr_total": [x + y + z for x, y, z in zip(R["assembly"], R["dc"], R["fd2nr"])],
        }
        for k, v in {**R, **tot}.items():
            emit(kind="classical", grid=a.grid, n_bus=N_BUS[a.grid], component=k, **stats(v))
        emit(kind="classical_meta", grid=a.grid, n_scen_distinct=len(scen), warmups=WARM,
             **{k: (float(np.mean(v)) if k.startswith("ok") else float(np.median(v))) for k, v in meta.items()},
             converged_frac_flat=float(np.mean(meta["ok_flat"])), converged_frac_dc=float(np.mean(meta["ok_dc"])),
             converged_frac_fd2nr=float(np.mean(meta["ok_fd2nr"])))


if __name__ == "__main__":
    main()
