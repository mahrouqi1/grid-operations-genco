"""Candidate 8: classical |V| interpolators from sparse |V| measurements (no dispatch). UNAUDITED.
All methods see: loads Pd,Qd, in-service branch graph, Ybus off-diagonals, REF/PV/PQ flags, measured |V| at the measured set.
No labels at unmeasured buses are ever used; no learning across grids or scenarios."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
SI = HERE.parent / "2026-10-21-genco-state-inference"
sys.path.insert(0, str(SI))
import common as C  # noqa: E402

CONFIGS = [  # (name, frac, seed_index)  seed_index 0 == candidate 7's mask seed
    ("f10_s0", 0.10, 0), ("f20_s0", 0.20, 0), ("f20_s1", 0.20, 1), ("f20_s2", 0.20, 2), ("f40_s0", 0.40, 0)]
EPS_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]      # I5 kernel regularisation (L_w normalised to mean diagonal 1)
LAM_GRID = [1e-3, 1e-2, 1e-1]                  # I5 noise/scale ratio sigma2/s2
KFOLD = 5                                      # I4 residual folds
VM_LO, VM_HI = 0.5, 1.5


def measured_seeded(seed_idx: int):
    def f(sid, n, frac=C.FRAC):
        k = n if frac >= 1.0 else max(1, int(round(frac * n)))
        rng = np.random.default_rng([C.SEED + seed_idx, int(n), int(sid)])  # seed_idx 0 == C.measured_set exactly
        return np.sort(rng.permutation(n)[:k])
    return f


def graph(bus, br, yb, n):
    """symmetric adjacency with unit weights (A1) and |Y_ij| weights (AY) over in-service branches."""
    b = br[br.br_status == 1]
    A1 = sp.coo_matrix((np.ones(len(b)), (b.from_bus, b.to_bus)), shape=(n, n)); A1 = (A1 + A1.T).tocsr(); A1.data[:] = 1.0
    Y = C.ybus(yb, n)
    Yo = abs(Y - sp.diags(Y.diagonal())).tocsr()
    Yo = Yo.multiply(A1 > 0).tocsr()                      # restrict to in-service branch pairs
    Yo = ((Yo + Yo.T) / 2).tocsr()
    return A1, Yo


def lap(A):
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def harmonic(A, m, y):
    """unmeasured values solve L_uu x = -L_um y (Dirichlet at measured). Components with no measured bus get the measured mean."""
    n = A.shape[0]; L = lap(A); u = np.where(~m)[0]; k = np.where(m)[0]
    ybar = y[k].mean(); x = np.full(n, ybar); x[k] = y[k]
    if len(u):
        Luu = L[u][:, u].tocsc() + 1e-9 * sp.eye(len(u)); x[u] = ybar + spl.spsolve(Luu, -L[u][:, k] @ (y[k] - ybar))
    return x


def i3_nearest(AY, m, y):
    D = AY.copy(); D.data = 1.0 / D.data
    k = np.where(m)[0]
    dist, _, src = dijkstra(D, directed=False, indices=k, min_only=True, return_predecessors=True)
    x = np.full(len(m), y[k].mean()); ok = src >= 0
    x[ok] = y[src[ok]]; x[k] = y[k]
    return x


def i4_loadaware(AY, m, y, Pd, Qd, rng_seed):
    """I2 + one two-coefficient least-squares drop correction x_u += a*Pd_u/100 + b*Qd_u/100 (MW,MVAr per 100 MVA).
    (a,b): ridge(1e-6) LS of out-of-fold harmonic residuals at the MEASURED buses, KFOLD random folds, on measured buses only."""
    k = np.where(m)[0]; h = harmonic(AY, m, y)
    if len(k) < 2 * KFOLD:
        return h, (0.0, 0.0)
    rng = np.random.default_rng(rng_seed); fold = rng.permutation(len(k)) % KFOLD
    res = np.zeros(len(k))
    for f in range(KFOLD):
        mm = m.copy(); mm[k[fold == f]] = False
        res[fold == f] = (y - harmonic(AY, mm, np.where(mm, y, 0.0)))[k[fold == f]]
    X = np.c_[Pd[k], Qd[k]] / 100.0
    ab = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ res)
    x = h.copy(); u = ~m
    x[u] = h[u] + (np.c_[Pd[u], Qd[u]] / 100.0) @ ab
    return x, (float(ab[0]), float(ab[1]))


def i5_gp(AY, m, y):
    """GP posterior mean, prior f ~ N(ybar, s2 K), K = (Lw + eps I)^-1, Lw = |Y|-weighted Laplacian scaled to mean diagonal 1,
    y_m = f_m + noise(sigma2 = lam*s2). (eps, lam) chosen from fixed grids by marginal likelihood on the measured buses (s2 profiled)."""
    n = AY.shape[0]; L = lap(AY); L = L / max(L.diagonal().mean(), 1e-12)
    k = np.where(m)[0]; u = np.where(~m)[0]; mu = y[k].mean(); r = y[k] - mu; mk = len(k)
    best = None
    cache = {}
    for eps in EPS_GRID:
        lu = spl.splu((L + eps * sp.eye(n)).tocsc())
        E = np.zeros((n, mk)); E[k, np.arange(mk)] = 1.0
        Kc = lu.solve(E)                                   # n x mk : K[:, measured]
        cache[eps] = Kc
        Kmm = Kc[k]
        for lam in LAM_GRID:
            A = Kmm + lam * np.eye(mk)
            try:
                c = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                continue
            z = np.linalg.solve(c, r); s2 = max(float(z @ z) / mk, 1e-18)
            nll = 0.5 * mk * np.log(s2) + np.log(np.diag(c)).sum()
            if best is None or nll < best[0]:
                best = (nll, eps, lam, A)
    _, eps, lam, A = best
    x = np.full(n, mu); x[k] = y[k]
    if len(u):
        x[u] = mu + cache[eps][u] @ np.linalg.solve(A, r)
    return x, (eps, lam)


def predict_all(bus, br, yb, sid, mset):
    n = len(bus); m = np.zeros(n, bool); m[mset] = True
    vt = bus.Vm.to_numpy(); y = np.where(m, vt, 0.0)       # unmeasured entries never read
    A1, AY = graph(bus, br, yb, n)
    Pd, Qd = bus.Pd.to_numpy(), bus.Qd.to_numpy()
    out = {"I1": harmonic(A1, m, y), "I2": harmonic(AY, m, y), "I3": i3_nearest(AY, m, y)}
    out["I4"], ab = i4_loadaware(AY, m, y, Pd, Qd, [C.SEED, n, int(sid)])
    out["I5"], hp = i5_gp(AY, m, y)
    for k_ in out:
        out[k_] = np.clip(out[k_], VM_LO, VM_HI)
    return out, m, {"I4_ab": ab, "I5_eps_lam": hp}
