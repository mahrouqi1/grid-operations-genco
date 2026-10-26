"""Candidate 9 shared code: I1-I5 from candidate 8 (vcommon, imported unchanged) except I5's implementation, which is made scalable
(exactly the same GP posterior mean; marginal-likelihood fit on at most 300 measured buses). Noise model. UNAUDITED."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp, scipy.sparse.linalg as spl
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-21-genco-voltage-inference"))
import vcommon as V   # also puts candidate 7's common.py on the path as V.C
import grids as G
C = V.C
SIGMA_REL = 0.005
CONFIGS = ([(f"f{p}_s{s}", p / 100, s, False) for p in (10, 20, 40) for s in range(3)]
           + [(f"n{p}_s{s}", p / 100, s, True) for p in (20, 40) for s in range(3)])
ML_CAP = 300


def tables(grid, ids):
    raw = G.ROOT / grid / "raw"; f = [("scenario", "in", [int(i) for i in ids])]
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=f); gen = pd.read_parquet(raw / "gen_data.parquet", filters=f)
    br = pd.read_parquet(raw / "branch_data.parquet", filters=f)
    yb = pd.read_parquet(raw / "y_bus_data.parquet", columns=["scenario", "index1", "index2", "G", "B"], filters=f)
    return {"bus": dict(tuple(bus.sort_values(["scenario", "bus"]).groupby("scenario"))), "gen": dict(tuple(gen.sort_values(["scenario", "idx"]).groupby("scenario"))),
            "br": dict(tuple(br.groupby("scenario"))), "y": dict(tuple(yb.groupby("scenario")))}


def noise_z(sid, n, seed_idx):
    return np.random.default_rng([C.SEED + 5000 + seed_idx, int(n), int(sid)]).standard_normal(n)


def i5_gp(AY, m, y):
    """Same model as vcommon.i5_gp. Hyperparameters (eps, lam): grids as in vcommon, marginal likelihood (s2 profiled) on the measured buses,
    or on a seeded random subset of ML_CAP of them if more were measured. Posterior mean uses ALL measured buses via the exact sparse
    identity (L + eps I + P_m/lam) f = eps*ybar*1 + P_m y/lam."""
    n = AY.shape[0]; L = V.lap(AY); L = L / max(L.diagonal().mean(), 1e-12)
    k = np.where(m)[0]; mu = y[k].mean(); mk = len(k)
    S = k if mk <= ML_CAP else np.sort(np.random.default_rng([C.SEED, n, mk, 55]).choice(k, ML_CAP, replace=False))
    r = y[S] - mu; ms = len(S); best = None; I = sp.eye(n, format="csc")
    for eps in V.EPS_GRID:
        lu = spl.splu((L + eps * I).tocsc()); E = np.zeros((n, ms)); E[S, np.arange(ms)] = 1.0
        Kss = lu.solve(E)[S]
        for lam in V.LAM_GRID:
            A = Kss + lam * np.eye(ms)
            try: c = np.linalg.cholesky(A)
            except np.linalg.LinAlgError: continue
            z = np.linalg.solve(c, r); s2 = max(float(z @ z) / ms, 1e-18)
            nll = 0.5 * ms * np.log(s2) + np.log(np.diag(c)).sum()
            if best is None or nll < best[0]: best = (nll, eps, lam)
    _, eps, lam = best
    P = sp.diags(m.astype(float)); rhs = eps * mu * np.ones(n) + (P @ np.where(m, y, 0.0)) / lam
    f = spl.spsolve((L + eps * I + P / lam).tocsc(), rhs)
    x = f.copy(); x[k] = y[k]
    return x, (eps, lam)


def predict_all(bus, br, yb, sid, mset, noisy, seed_idx):
    n = len(bus); m = np.zeros(n, bool); m[mset] = True
    vt = bus.Vm.to_numpy()
    ymeas = vt * (1.0 + SIGMA_REL * noise_z(sid, n, seed_idx)) if noisy else vt
    y = np.where(m, ymeas, 0.0)
    A1, AY = V.graph(bus, br, yb, n); Pd, Qd = bus.Pd.to_numpy(), bus.Qd.to_numpy()
    out = {"I1": V.harmonic(A1, m, y), "I2": V.harmonic(AY, m, y), "I3": V.i3_nearest(AY, m, y)}
    out["I4"], ab = V.i4_loadaware(AY, m, y, Pd, Qd, [C.SEED, n, int(sid)])
    out["I5"], hp = i5_gp(AY, m, y)
    for k_ in out: out[k_] = np.clip(out[k_], V.VM_LO, V.VM_HI)
    return out, m, {"I4_ab": ab, "I5_eps_lam": hp}
