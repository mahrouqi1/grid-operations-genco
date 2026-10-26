"""Copied 2026-10-18 from audit genco-published-newton round 1 (scratch/round-1/np_newton.py), unchanged except this line: an
independent textbook polar Newton (MATPOWER newtonpf form, no damping, flat start) on the rebuilt .m files. Control arm
for R1-F1. Run: python textbook_newton.py <work dir>/*.m (cases.csv lists the 60)."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.linalg as spl
from matpowercaseframes import CaseFrames
def run(m):
    cf = CaseFrames(m); bus = cf.bus.to_numpy(float); gen = cf.gen.to_numpy(float); br = cf.branch.to_numpy(float); base = 100.0
    ids = bus[:,0].astype(int); idx = {b:i for i,b in enumerate(ids)}; n=len(ids)
    on = br[:,10]==1; br=br[on]
    f = np.array([idx[int(x)] for x in br[:,0]]); t = np.array([idx[int(x)] for x in br[:,1]])
    ys = 1/(br[:,2]+1j*br[:,3]); bc = br[:,4]; tap = np.where(br[:,8]==0,1.0,br[:,8])*np.exp(1j*np.radians(br[:,9]))
    Ytt = ys+1j*bc/2; Yff = Ytt/(tap*np.conj(tap)); Yft = -ys/np.conj(tap); Ytf = -ys/tap
    Y = sp.csr_matrix((np.r_[Yff,Yft,Ytf,Ytt],(np.r_[f,f,t,t],np.r_[f,t,f,t])),shape=(n,n)) + sp.diags((bus[:,4]+1j*bus[:,5])/base)
    g = gen[gen[:,7]==1]
    Sg = np.zeros(n,complex); np.add.at(Sg,[idx[int(x)] for x in g[:,0]],(g[:,1]+1j*g[:,2])/base)
    S = Sg - (bus[:,2]+1j*bus[:,3])/base
    typ = bus[:,1].astype(int); pv = np.where(typ==2)[0]; pq = np.where(typ==1)[0]; pvpq=np.r_[pv,pq]
    vm = np.where(typ==1,1.0,bus[:,7]); va=np.zeros(n); V=vm*np.exp(1j*va)
    for it in range(30):
        mis = V*np.conj(Y @ V)-S; F=np.r_[mis[pvpq].real, mis[pq].imag]; nf=np.abs(F).max()
        if not np.isfinite(nf): return f"diverged(nan) it={it}"
        if nf<1e-8:
            lab=pd.read_csv(m.replace(".m","_label.csv")); vl=(lab.vm*np.exp(1j*lab.va)).to_numpy()
            ph=m.replace(".m","_phys.csv"); import os
            dp=np.abs(V-(lambda d:(d.vm*np.exp(1j*d.va)).to_numpy())(pd.read_csv(ph))).max() if os.path.exists(ph) else float("nan")
            return f"conv it={it} minvm={np.abs(V).min():.3f} dVlab={np.abs(V-vl).max():.1e} dVphys={dp:.1e}"
        Ib = Y @ V; dV_m = sp.diags(V) @ np.conj(sp.diags(Ib)) + sp.diags(V) @ np.conj(Y @ sp.diags(V/np.abs(V)))
        dV_a = 1j*sp.diags(V) @ np.conj(sp.diags(Ib)-Y @ sp.diags(V))
        dV_m = sp.diags(V) @ sp.diags(np.conj(Ib)) ; dV_m = sp.csr_matrix(dV_m)
        # MATPOWER dSbus_dV
        Vn = V/np.abs(V); dS_dVm = sp.diags(V) @ np.conj(Y @ sp.diags(Vn)) + sp.diags(np.conj(Ib)) @ sp.diags(Vn)
        dS_dVa = 1j*sp.diags(V) @ np.conj(sp.diags(Ib)-Y @ sp.diags(V))
        J = sp.vstack([sp.hstack([dS_dVa.real[pvpq][:,pvpq], dS_dVm.real[pvpq][:,pq]]), sp.hstack([dS_dVa.imag[pq][:,pvpq], dS_dVm.imag[pq][:,pq]])]).tocsc()
        dx = spl.spsolve(J,-F)
        va[pvpq]+=dx[:len(pvpq)]; vm[pq]+=dx[len(pvpq):]; V=vm*np.exp(1j*va)
    return f"maxit nf={nf:.2e}"
for m in sys.argv[1:]:
    lab=pd.read_csv(m.replace(".m","_label.csv"))
    print(m.split("/")[-1], run(m), flush=True)
