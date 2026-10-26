"""Classical AC WLS state estimator for the GENCO voltage-inference task. UNAUDITED. Spec: protocol.json."""
from __future__ import annotations
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl

BASE = 100.0


def dS_dV(Y, V):
    """dS/dVa, dS/dVm of S = V conj(Y V) (complex sparse), standard polar form."""
    I = Y @ V; Vn = V / np.abs(V)
    dSdVm = sp.diags(V) @ (Y @ sp.diags(Vn)).conj() + sp.diags(np.conj(I)) @ sp.diags(Vn)
    dSdVa = 1j * sp.diags(V) @ (sp.diags(I) - Y @ sp.diags(V)).conj()
    return dSdVa.tocsr(), dSdVm.tocsr()


def _est(Y, n, ref, theta_ref, mset, vmeas, sigma_v, zero_idx, sz_pq, gen_idx, p_pri, q_pri, sp_, sq_, sv_prior=0.1,
             Pd=None, Qd=None, maxit=40, lam0=1e-3, start=None, ldec=3.0):
    """Y complex sparse (n,n). ref: REF bus index. mset: measured-|V| bus indices, vmeas their values (sigma_v scalar or array).
    zero_idx: buses with exactly-known injection -Pd-jQd (pu, pass Pd,Qd in MW/MVAr). gen_idx: generator buses with
    pseudo P,Q injection priors p_pri,q_pri (pu, net of load) with sigmas sp_,sq_ (pu). Returns dict."""
    Sd = -(Pd + 1j * Qd) / BASE
    vbar = float(np.mean(vmeas))
    nr = np.setdiff1d(np.arange(n), [ref])
    # measurement vector and weights
    zV = np.asarray(vmeas, float); wV = 1.0 / np.broadcast_to(np.asarray(sigma_v, float), zV.shape)
    zP0, zQ0 = Sd.real[zero_idx], Sd.imag[zero_idx]; w0 = np.full(len(zero_idx), 1.0 / sz_pq)
    wP, wQ = 1.0 / sp_, 1.0 / sq_
    allb = np.arange(n); wVp = np.full(n, 1.0 / sv_prior)
    z = np.r_[zV, zP0, zQ0, p_pri, q_pri, np.full(n, vbar)]
    w = np.r_[wV, w0, w0, wP, wQ, wVp]
    nth = len(nr)
    def unpack(x):
        th = np.full(n, theta_ref); th[nr] = x[:nth]; return th, x[nth:]
    def h_and_J(x, want_J=True):
        th, vm = unpack(x); V = vm * np.exp(1j * th); S = V * np.conj(Y @ V)
        h = np.r_[vm[mset], S.real[zero_idx], S.imag[zero_idx], S.real[gen_idx], S.imag[gen_idx], vm]
        if not want_J: return h, None
        dSa, dSm = dS_dV(Y, V)
        def blk(rows, M_a, M_m): return sp.hstack([M_a[rows][:, nr], M_m[rows]], format="csr")
        JV = sp.hstack([sp.csr_matrix((len(mset), nth)), sp.csr_matrix((np.ones(len(mset)), (np.arange(len(mset)), mset)), shape=(len(mset), n))], format="csr")
        JP0, JQ0 = blk(zero_idx, dSa.real, dSm.real), blk(zero_idx, dSa.imag, dSm.imag)
        JPg, JQg = blk(gen_idx, dSa.real, dSm.real), blk(gen_idx, dSa.imag, dSm.imag)
        JVp = sp.hstack([sp.csr_matrix((n, nth)), sp.eye(n)], format="csr")
        return h, sp.vstack([JV, JP0, JQ0, JPg, JQg, JVp], format="csr")
    if start is None:
        vm0 = np.full(n, vbar); vm0[mset] = zV; x = np.r_[np.full(nth, theta_ref), vm0]
    else:
        x = start.copy()
    W = sp.diags(w)
    def cost(h): r = z - h; return float((w * r) @ (w * r))
    h, J = h_and_J(x); c = cost(h); lam = lam0; best = (c, x.copy()); status = "maxit"; it = 0
    try:
        for it in range(1, maxit + 1):
            r = z - h; Jw = sp.diags(w) @ J; A = (Jw.T @ Jw).tocsc(); g = Jw.T @ (w * r); dA = A.diagonal()
            accepted = False
            while lam < 1e12:
                try:
                    dx = spl.spsolve((A + lam * sp.diags(np.maximum(dA, 1e-12))).tocsc(), g)
                except Exception:
                    lam *= 5; continue
                if not np.all(np.isfinite(dx)): lam *= 5; continue
                xn = x + dx; xn[nth:] = np.clip(xn[nth:], 0.05, 3.0)
                hn, _ = h_and_J(xn, False); cn = cost(hn)
                if cn < c:
                    accepted = True; break
                lam *= 5
            if not accepted:
                try: d0 = spl.spsolve((A + 1e-12 * sp.diags(np.maximum(dA, 1e-12))).tocsc(), g); small = bool(np.abs(d0).max() < 1e-5)
                except Exception: small = False
                status = "converged" if small else "stalled"; break
            cold = c; x = xn; c = cn; lam = max(lam / ldec, 1e-12); h, J = h_and_J(x)
            if c < best[0]: best = (c, x.copy())
            if np.abs(dx).max() < 1e-4 or (cold - c) / max(cold, 1e-30) < 1e-5: status = "converged"; break
    except Exception as e:  # noqa
        status = "error:" + type(e).__name__
    th, vm = unpack(best[1])
    return {"vm": vm, "theta": th, "status": status, "iters": it, "cost": best[0]}


def estimate(Y, n, ref, theta_ref, mset, vmeas, sigma_v, zero_idx, sz_pq, gen_idx, p_pri, q_pri, sp_, sq_, sv_prior=0.1,
             Pd=None, Qd=None, maxit=(8, 12), stages=(100.0, 1.0), dc_start=True, lam0=1e-5, ldec=10.0):
    """DC-angle start (prior dispatch) then LM with sigma continuation on the zero-injection rows (first stage sigma x100, last stage the registered sigma)."""
    vbar = float(np.mean(vmeas)); start = None
    if dc_start:
        P = np.zeros(n); P[zero_idx] = -Pd[zero_idx] / BASE; P[gen_idx] = p_pri
        B = (-Y.imag).tocsr(); nr = np.setdiff1d(np.arange(n), [ref])
        th = np.full(n, theta_ref)
        try:
            th[nr] = theta_ref + spl.spsolve(B[nr][:, nr].tocsc(), P[nr])
        except Exception:
            pass
        th = np.clip(th, theta_ref - 1.5, theta_ref + 1.5)
        vm0 = np.full(n, vbar); vm0[mset] = vmeas
        start = np.r_[th[nr], vm0]
    out = None
    for sc, mi in zip(stages, maxit):
        out = _est(Y, n, ref, theta_ref, mset, vmeas, sigma_v, zero_idx, sz_pq * sc, gen_idx, p_pri, q_pri, sp_, sq_, sv_prior, Pd, Qd, mi, lam0, start, ldec)
        start = np.r_[out["theta"][np.setdiff1d(np.arange(n), [ref])], out["vm"]]
    return out
