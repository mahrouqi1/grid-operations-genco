"""Carve connected subgrids of the SEEN-family parents (GOC / EPIGRIDS), UNAUDITED. python carve.py
datakit has no subgrid/subnetwork carving procedure (searched gridfm-datakit-repro and the installed gridfm_datakit: the pretraining
launcher just reads a pre-made directory of subnets_casefiles/*.m), so this is a documented classical alternative:
 1. parent base point: AC Newton (core/grid_ops/reference_pf, DC start) on the pglib .m with dispatch Pmin+alpha(Pmax-Pmin) set to cover load + losses
    (slack ~ its share), voltage setpoints = gen VG;
 2. region = seeded BFS from a random root over in-service branches, grown to a target size (log-uniform 150..950 buses);
 3. boundary: the tie branches are removed and the flow they carried at the parent base point is folded into the inside boundary bus as
    extra Pd/Qd (so the region's base point is a root of the carved case: Pd' = Pg - Re(V conj(Y_sub V)) at every region bus);
 4. REF = bus of the largest in-service generator in the region; renumbered 1..n; gen costs/limits copied from the parent.
Rejection: no in-service generator, gen capacity < 1.2 x net load, bus count or branch (r,x) fingerprint matching a hosted GENCO training subgrid."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.linalg as spl
from matpowercaseframes import CaseFrames
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "core"))
from grid_ops.reference_pf import newton_raphson_pf, injected_power, _sparse_jacobian, PFResult


def damped_newton(Y, sbus, v0, pv, pq, tol=1e-8, max_it=100):
    """Fallback only (used when plain Newton fails on a parent): the same Newton step with backtracking (step halved until ||F||_2 drops)."""
    v = v0.astype(complex).copy(); va, vm = np.angle(v), np.abs(v); pvpq = np.r_[pv, pq]; k = len(pvpq)
    def F(v):
        ds = injected_power(Y, v) - sbus; return np.r_[ds[pvpq].real, ds[pq].imag]
    f = F(v); hist = [float(np.abs(f).max())]; it = 0
    while hist[-1] > tol and it < max_it:
        it += 1; dx = spl.spsolve(_sparse_jacobian(Y, v, pvpq, pq), -f); a = 1.0
        for _ in range(14):
            va2, vm2 = va.copy(), vm.copy(); va2[pvpq] += a * dx[:k]; vm2[pq] += a * dx[k:]
            v2 = vm2 * np.exp(1j * va2); f2 = F(v2)
            if np.isfinite(f2).all() and np.linalg.norm(f2) < np.linalg.norm(f) and vm2[pq].min() > 0.2: break
            a *= 0.5
        else: break
        va, vm, v, f = va2, vm2, v2, f2; hist.append(float(np.abs(f).max()))
    return PFResult(v=v, vm=np.abs(v), va_deg=np.degrees(np.angle(v)), converged=hist[-1] <= tol, iterations=it, mismatch_history=hist)
EXT = REPO / "data/external"; PGLIB = EXT / "pglib-opf"; OUT = EXT / "genco-carved-unseen"
PARENTS = ["case1888_rte", "case2848_rte", "case6470_rte", "case6495_rte", "case6515_rte", "case2869_pegase", "case8387_pegase", "case9241_pegase",
           "case2383wp_k", "case3012wp_k", "case3375wp_k"]
PER_PARENT, LO, HI, SEED, JAC_THR = 5, 150, 950, 20261022, 0.3   # POOL of 5 candidates per parent; screen.py keeps the first 2 OPF-feasible


def ybus(bus, br, n, f, t):
    st = br.BR_STATUS.to_numpy() > 0
    ys = 1.0 / (br.BR_R.to_numpy() + 1j * br.BR_X.to_numpy()); bc = br.BR_B.to_numpy()
    tap = np.where(br.TAP.to_numpy() == 0, 1.0, br.TAP.to_numpy()) * np.exp(1j * np.pi / 180 * br.SHIFT.to_numpy())
    ys, bc, tap = ys * st, bc * st, tap
    ytt = ys + 1j * bc / 2; yff = ytt / (tap * np.conj(tap)); yft = -ys / np.conj(tap); ytf = -ys / tap
    m = len(br); i = np.arange(m)
    Yf = sp.csr_matrix((np.r_[yff, yft], (np.r_[i, i], np.r_[f, t])), shape=(m, n))
    Yt = sp.csr_matrix((np.r_[ytf, ytt], (np.r_[i, i], np.r_[f, t])), shape=(m, n))
    Cf = sp.csr_matrix((np.ones(m), (f, i)), shape=(n, m)); Ct = sp.csr_matrix((np.ones(m), (t, i)), shape=(n, m))
    Ysh = (bus.GS.to_numpy() + 1j * bus.BS.to_numpy()) / 100.0
    return (Cf @ Yf + Ct @ Yt + sp.diags(Ysh)).tocsc()


def parent_state(name, ls=1.0, damped=False, shipped=False):
    cf = CaseFrames(str(PGLIB / f"pglib_opf_{name}.m")); bus, gen, br = cf.bus.reset_index(drop=True), cf.gen.reset_index(drop=True), cf.branch.reset_index(drop=True)
    ids = bus.BUS_I.to_numpy().astype(int); pos = {b: i for i, b in enumerate(ids)}; n = len(bus)
    f = np.array([pos[int(b)] for b in br.F_BUS]); t = np.array([pos[int(b)] for b in br.T_BUS]); gb = np.array([pos[int(b)] for b in gen.GEN_BUS])
    Y = ybus(bus, br, n, f, t); Pd, Qd = bus.PD.to_numpy() * ls, bus.QD.to_numpy() * ls; on = gen.GEN_STATUS.to_numpy() > 0
    pmin, pmax = gen.PMIN.to_numpy(), gen.PMAX.to_numpy()
    ref = np.where(bus.BUS_TYPE.to_numpy() == 3)[0]; pvset = set(gb[on]); pv = np.array([i for i in np.where(bus.BUS_TYPE.to_numpy() == 2)[0] if i in pvset])
    pq = np.setdiff1d(np.arange(n), np.r_[ref, pv])
    vset = np.ones(n); vset[gb[on]] = gen.VG.to_numpy()[on]
    b = br[br.BR_STATUS > 0]; bb = 1 / (b.BR_X.to_numpy() * np.where(b.TAP.to_numpy() == 0, 1, b.TAP.to_numpy())); fi = np.array([pos[int(x)] for x in b.F_BUS]); ti = np.array([pos[int(x)] for x in b.T_BUS])
    Bm = sp.coo_matrix((np.r_[bb, bb, -bb, -bb], (np.r_[fi, ti, fi, ti], np.r_[fi, ti, ti, fi])), shape=(n, n)).tocsc()
    nr = np.setdiff1d(np.arange(n), ref)
    target = Pd.sum() * 1.02
    for it in range(6):     # dispatch = Pmin + alpha (Pmax - Pmin), alpha set so total generation = load + losses (slack ~ its share)
        alpha = np.clip((target - pmin[on].sum()) / (pmax[on].sum() - pmin[on].sum()), 0, 1)
        pgg = np.where(on, gen.PG.to_numpy() * ls, 0.0) if shipped else np.where(on, pmin + alpha * (pmax - pmin), 0.0)   # shipped: the case file's own dispatch (scaled with the load), slack absorbs
        pg = np.zeros(n); np.add.at(pg, gb[on], pgg[on])
        qg_fixed = np.zeros(n); npv = on & ~np.isin(gb, pv); np.add.at(qg_fixed, gb[npv], gen.QG.to_numpy()[npv] * ls)   # gens at non-voltage-controlled (type 1) buses inject their file Q
        sb = (pg - Pd) / 100.0 + 1j * (qg_fixed - Qd) / 100.0
        th = np.zeros(n); th[nr] = spl.spsolve(Bm[nr][:, nr].tocsc(), sb.real[nr])
        r = (damped_newton if damped else newton_raphson_pf)(Y, sb, vset * np.exp(1j * th), pv, pq, tol=1e-8, max_it=100 if damped else 40)
        if not r.converged: raise RuntimeError(f"{name}: parent AC PF did not converge (iter {it}, alpha {alpha:.3f})")
        V = r.v; S = V * np.conj(Y @ V); pgb = S.real * 100 + Pd; qgb = S.imag * 100 + Qd
        loss = (pgb.sum() - Pd.sum()); target = Pd.sum() + loss   # generation needed = load + losses of the last solution
        if shipped or abs(pgb[ref].sum() - pg[ref].sum()) < 1.0: break
    return dict(ls=ls, cf=cf, bus=bus, gen=gen, br=br, n=n, ids=ids, f=f, t=t, gb=gb, V=V, pgb=pgb, qgb=qgb, on=on, ref=ref)


def fingerprint(r, x): return Counter((round(float(a), 4), round(float(b), 4)) for a, b in zip(r, x))
def jac(a, b):
    u = sum((a | b).values()); return sum((a & b).values()) / u if u else 0.0


def hosted_fps():
    out = {}
    for d in sorted((EXT / "genco-pretrain-casefiles").glob("subnet_*")):
        b = pd.read_csv(d / "branches.csv"); out[d.name] = (len(set(b.from_bus) | set(b.to_bus)), fingerprint(b.r, b.x))
    return out


def bfs(adj, root, target, rng):
    seen = {root}; order = [root]; q = [root]
    while q and len(order) < target:
        u = q.pop(0); nb = list(adj[u]); rng.shuffle(nb)
        for v in nb:
            if v not in seen and len(order) < target: seen.add(v); order.append(v); q.append(v)
    return order


def write_m(path, name, bus, gen, br, gc):
    L = [f"function mpc = {name}", "mpc.version = '2';", "mpc.baseMVA = 100.0;", "", "mpc.bus = ["]
    for r in bus.itertuples(index=False): L.append("\t" + "\t".join(f"{v:.8g}" if k not in (0, 1, 6, 10) else f"{int(v)}" for k, v in enumerate(r)) + ";")
    L += ["];", "", "mpc.gen = ["]
    for r in gen.itertuples(index=False): L.append("\t" + "\t".join(f"{v:.8g}" if k not in (0, 7) else f"{int(v)}" for k, v in enumerate(r)) + ";")
    L += ["];", "", "mpc.branch = ["]
    for r in br.itertuples(index=False): L.append("\t" + "\t".join(f"{v:.10g}" if k not in (0, 1, 10) else f"{int(v)}" for k, v in enumerate(r)) + ";")
    L += ["];", "", "mpc.gencost = ["]
    for r in gc.itertuples(index=False): L.append("\t" + "\t".join(f"{v:.8g}" if k not in (0, 3) else f"{int(v)}" for k, v in enumerate(r)) + ";")
    L += ["];", ""]
    path.write_text("\n".join(L))


def main(parents=None, tie_max=1.0, offset=0, tag='a', pool=None):
    rng = np.random.default_rng(SEED + offset); hosted = hosted_fps(); (OUT / "grids").mkdir(parents=True, exist_ok=True)
    manifest = []
    PER = pool or PER_PARENT
    for pn in (parents or PARENTS):
        P = None
        for ls, dmp, shp in [(l, d, sh) for l in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5) for sh in (False, True) for d in (False, True)]:   # fallbacks (never needed for the seen-family parents): damped Newton, then lower load
            try: P = parent_state(pn, ls, dmp, shp); P["damped"] = dmp; P["shipped"] = shp; break
            except RuntimeError as e: print("  parent PF failed at load scale", ls, "damped" if dmp else "plain", "shipped-dispatch" if shp else "alpha-dispatch", flush=True)
        if P is None: print("  PARENT UNSOLVABLE", pn, flush=True); continue
        print(pn, "parent load scale", P["ls"], flush=True); n = P["n"]; on = P["on"]; gen, br, bus = P["gen"], P["br"], P["bus"]
        adj = [set() for _ in range(n)]
        for a, b, s in zip(P["f"], P["t"], br.BR_STATUS.to_numpy()):
            if s > 0: adj[a].add(b); adj[b].add(a)
        print(pn, "parent PF ok, n", n, flush=True)
        made, tries = 0, 0
        while made < PER and tries < 3000:
            tries += 1
            target = int(round(np.exp(rng.uniform(np.log(LO), np.log(HI))))); root = int(rng.integers(n))
            reg = bfs(adj, root, target, rng)
            if len(reg) < 0.9 * target: continue
            R = np.sort(np.array(reg)); loc = -np.ones(n, int); loc[R] = np.arange(len(R)); m = len(R)
            bi = br.BR_STATUS.to_numpy() > 0
            ein = np.where((loc[P["f"]] >= 0) & (loc[P["t"]] >= 0))[0]
            brs = br.iloc[ein].reset_index(drop=True); fs, ts = loc[P["f"][ein]], loc[P["t"][ein]]
            bs = bus.iloc[R].reset_index(drop=True); bs["PD"] = bs.PD * P["ls"]; bs["QD"] = bs.QD * P["ls"]
            Ys = ybus(bs, brs, m, fs, ts); Vr = P["V"][R]; Ssub = Vr * np.conj(Ys @ Vr)
            Pd2 = P["pgb"][R] - Ssub.real * 100; Qd2 = P["qgb"][R] - Ssub.imag * 100
            gin = np.where((loc[P["gb"]] >= 0) & on)[0]
            if len(gin) == 0: continue
            pmax = gen.PMAX.to_numpy()[gin]
            if pmax.sum() < 1.2 * max(Pd2.sum(), 1.0) or Pd2.sum() <= 0: continue
            tie_mw = float(np.abs(Pd2 - bs.PD.to_numpy()).sum()); scale = max(float(bs.PD.sum()), float(P["pgb"][R].sum()))
            if bs.PD.sum() < 20 or tie_mw > tie_max * scale: continue          # a corridor that carries more tie flow than its own load/generation is rejected
            # fingerprint leakage check against the 31 hosted training subgrids
            fp = fingerprint(brs.BR_R, brs.BR_X)
            worst = max(((jac(fp, hfp), h, hn) for h, (hn, hfp) in hosted.items()), key=lambda z: z[0])
            same_n = [h for h, (hn, _) in hosted.items() if hn == m]
            if worst[0] >= JAC_THR or same_n: print("  rejected (leak)", m, round(worst[0], 3), worst[1], same_n, flush=True); continue
            gb_loc = loc[P["gb"][gin]]; ref_local = int(gb_loc[np.argmax(pmax)])
            btype = np.ones(m, int); btype[gb_loc] = 2; btype[ref_local] = 3
            nb = bs.copy(); nb["BUS_I"] = np.arange(1, m + 1); nb["BUS_TYPE"] = btype; nb["PD"] = Pd2; nb["QD"] = Qd2
            nb["VM"] = np.abs(Vr); nb["VA"] = np.degrees(np.angle(Vr)) - np.degrees(np.angle(Vr[ref_local]))
            ng = gen.iloc[gin].reset_index(drop=True).copy(); ng["GEN_BUS"] = gb_loc + 1
            # dispatch the solved bus generation over that bus's in-service gens proportional to Pmax / Q range
            for b_ in np.unique(gb_loc):
                s = np.where(ng.GEN_BUS.to_numpy() == b_ + 1)[0]; w = ng.PMAX.to_numpy()[s]; w = w / w.sum() if w.sum() > 0 else np.ones(len(s)) / len(s)
                ng.loc[s, "PG"] = P["pgb"][R][b_] * w; ng.loc[s, "QG"] = P["qgb"][R][b_] * w
            nbr = brs.copy(); nbr["F_BUS"] = fs + 1; nbr["T_BUS"] = ts + 1
            gc = P["cf"].gencost.reset_index(drop=True).iloc[gin].reset_index(drop=True)
            name = f"carved_{pn}_c{offset + made}"
            write_m(OUT / "grids" / f"{name}.m", name, nb[["BUS_I", "BUS_TYPE", "PD", "QD", "GS", "BS", "BUS_AREA", "VM", "VA", "BASE_KV", "ZONE", "VMAX", "VMIN"]],
                    ng[["GEN_BUS", "PG", "QG", "QMAX", "QMIN", "VG", "MBASE", "GEN_STATUS", "PMAX", "PMIN"]],
                    nbr[["F_BUS", "T_BUS", "BR_R", "BR_X", "BR_B", "RATE_A", "RATE_B", "RATE_C", "TAP", "SHIFT", "BR_STATUS", "ANGMIN", "ANGMAX"]], gc)
            ncut = int(((loc[P["f"]] >= 0) ^ (loc[P["t"]] >= 0)).sum())
            manifest.append({"name": name, "parent": pn, "n_bus": m, "n_branch": len(brs), "n_gen": len(gin), "n_cut_branches": ncut, "root_bus_id": int(P["ids"][root]),
                             "target": target, "net_load_mw": float(Pd2.sum()), "gen_pmax_mw": float(pmax.sum()), "neg_pd_buses": int((Pd2 < -1e-6).sum()),
                             "min_pd_mw": float(Pd2.min()), "tie_mw": round(tie_mw, 1), "internal_scale_mw": round(scale, 1), "max_jaccard_vs_hosted_train": round(worst[0], 4), "closest_hosted": worst[1], "tries": tries, "parent_load_scale": P["ls"], "parent_damped_newton": P["damped"], "parent_shipped_dispatch": P["shipped"], "tie_max_rule": tie_max, "tier": tag})
            print("  carved", name, m, "buses", len(brs), "branches", len(gin), "gens", "cut", ncut, "jac", round(worst[0], 3), flush=True); made += 1
        if made < PER: print("  WARNING only", made, "carved for", pn)
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "carved")


if __name__ == "__main__":
    if len(sys.argv) > 1: main(parents=sys.argv[4:], tie_max=float(sys.argv[1]), offset=int(sys.argv[2]), tag=sys.argv[3], pool=4)
    else: main()
