"""Rebuild the exact AC-PF input datakit handed to PowerModels, from its raw (pre-validation) tables.

Datakit's PF-mode input per scenario = the mpc written by Network.to_mpc: loads Pd/Qd, gen Pg/Qg/status, branch
(r,x,b,tap,shift,status) after topology perturbation, bus types, Vm at PV/REF = the OPF set point (unchanged by
the PF, so the label's Vm at PV/REF is that set point). PQ-bus Vm/Va in the file are never read by either solver
path (both start PQ buses at vm=1.0, va=0.0), so they are written as 1.0 / 0.0.

Usage: python build_mpc.py <grid> <rawdir> <scenario> <out.m> [--phys <data-relabeled/<grid>/raw>]
"""
import sys, pandas as pd, numpy as np
BASE = 100.0

def build(raw, sc, out):
    b = pd.read_parquet(f"{raw}/bus_data.parquet", filters=[("scenario", "==", sc)]).sort_values("bus")
    g = pd.read_parquet(f"{raw}/gen_data.parquet", filters=[("scenario", "==", sc)]).sort_values("idx")
    br = pd.read_parquet(f"{raw}/branch_data.parquet", filters=[("scenario", "==", sc)]).sort_values("idx")
    typ = np.where(b.REF == 1, 3, np.where(b.PV == 1, 2, 1))
    vm_set = np.where(typ == 1, 1.0, b.Vm.to_numpy())
    L = ["function mpc = case_rebuilt", "mpc.version = '2';", f"mpc.baseMVA = {BASE};", "mpc.bus = ["]
    for i, r in enumerate(b.itertuples()):
        va = 0.0
        L.append(f"{r.bus+1} {typ[i]} {r.Pd:.15g} {r.Qd:.15g} {r.GS*BASE:.15g} {r.BS*BASE:.15g} 1 {float(vm_set[i]):.15g} {va} {r.vn_kv:.15g} 1 {r.max_vm_pu:.15g} {r.min_vm_pu:.15g};")
    L += ["];", "mpc.gen = ["]
    vmap = dict(zip(b.bus, vm_set))
    for r in g.itertuples():
        L.append(f"{r.bus+1} {r.p_mw:.15g} {r.q_mvar:.15g} {r.max_q_mvar:.15g} {r.min_q_mvar:.15g} {vmap[r.bus]:.15g} {BASE} {int(r.in_service)} {r.max_p_mw:.15g} {r.min_p_mw:.15g};")
    L += ["];", "mpc.branch = ["]
    for r in br.itertuples():
        L.append(f"{r.from_bus+1} {r.to_bus+1} {r.r:.15g} {r.x:.15g} {r.b:.15g} {r.rate_a:.15g} 0 0 {r.tap:.15g} {r.shift:.15g} {int(r.br_status)} {r.ang_min:.15g} {r.ang_max:.15g};")
    L += ["];", "mpc.gencost = ["]
    for r in g.itertuples():
        L.append(f"2 0 0 3 {r.cp2_eur_per_mw2:.15g} {r.cp1_eur_per_mw:.15g} {r.cp0_eur:.15g};")
    L += ["];"]
    open(out, "w").write("\n".join(L) + "\n")
    return b

if __name__ == "__main__":
    build(sys.argv[2], int(sys.argv[3]), sys.argv[4])
