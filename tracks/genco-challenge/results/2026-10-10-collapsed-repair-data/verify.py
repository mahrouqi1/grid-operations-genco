"""Independent read-back of the corrected copies (UNAUDITED). Per grid: (1) untouched scenarios/y_bus byte-equal to original;
(2) every recovered scenario: written Vm,Va satisfy the power-flow equations with the WRITTEN Pg/Qg/Pd/Qd and the Ybus (residual),
Vm>=0.5, bus Pg/Qg == in-service gen sums at gen buses, branch flows match; (3) originals untouched (mtime not checked, we never wrote there)."""
import json, sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
H = Path(__file__).resolve().parent; REPO = H.parents[3]
S = REPO / "data/external/genco-pf-pretraining"; D = REPO / "data/external/genco-pf-pretraining-corrected"
out = {}
for g in sys.argv[1:]:
    o, c = S / f"subnet_{g}/raw", D / f"subnet_{g}/raw"
    st = pd.read_csv(c / "relabeled.csv")
    rec = set(st[st.status.str.startswith("recovered")].scenario)
    res = {}
    bo, bc = pd.read_parquet(o / "bus_data.parquet"), pd.read_parquet(c / "bus_data.parquet")
    res["bus_same_shape_order"] = bool((bo[["scenario", "bus"]].to_numpy() == bc[["scenario", "bus"]].to_numpy()).all()) and list(bo.columns) == list(bc.columns)
    other = ~bo.scenario.isin(rec)
    res["untouched_bus_equal"] = bool(bo[other].drop(columns="scenario_partition").reset_index(drop=True).equals(bc[other].drop(columns="scenario_partition").reset_index(drop=True)))
    for t in ("gen_data", "branch_data", "y_bus_data"):
        a, b = pd.read_parquet(o / f"{t}.parquet"), pd.read_parquet(c / f"{t}.parquet")
        m = ~a.scenario.isin(rec)
        res[f"untouched_{t}_equal"] = bool(a[m].drop(columns="scenario_partition").reset_index(drop=True).equals(b[m].drop(columns="scenario_partition").reset_index(drop=True))) and list(a.columns) == list(b.columns)
        if t == "gen_data": ga, gb = a, b
        if t == "y_bus_data": yb = b
        if t == "branch_data": ba_, bb_ = a, b
    gb_on = gb[gb.in_service == 1].groupby(["scenario", "bus"])[["p_mw", "q_mvar"]].sum()
    worst, minvm, sumgap, brgap, angs = 0.0, 9, 0.0, 0.0, 0.0
    yg, bg, brg = dict(tuple(yb.groupby("scenario"))), dict(tuple(bc.groupby("scenario"))), dict(tuple(bb_.groupby("scenario")))
    for s in sorted(rec):
        b = bg[s].sort_values("bus"); n = len(b); y = yg[s]
        Y = sp.csr_matrix((y.G.to_numpy() + 1j * y.B.to_numpy(), (y.index1.to_numpy(), y.index2.to_numpy())), shape=(n, n))
        v = b.Vm.to_numpy() * np.exp(1j * np.radians(b.Va.to_numpy()))
        sb = (b.Pg - b.Pd + 1j * (b.Qg - b.Qd)).to_numpy() / 100
        ds = v * np.conj(Y @ v) - sb
        pq, pv = b.PQ.to_numpy() == 1, b.PV.to_numpy() == 1
        f = np.r_[ds[pq | pv].real, ds[pq].imag]
        worst = max(worst, float(np.abs(f).max())); minvm = min(minvm, float(b.Vm.min()))
        gs = gb_on.loc[s].reindex(b.bus.to_numpy()).fillna(0.0)
        isg = (b.PV.to_numpy() == 1) | (b.REF.to_numpy() == 1)
        sumgap = max(sumgap, float(np.abs(gs.p_mw.to_numpy()[isg] - b.Pg.to_numpy()[isg]).max()), float(np.abs(gs.q_mvar.to_numpy()[isg] - b.Qg.to_numpy()[isg]).max()))
        br = brg[s]; on = br.br_status.to_numpy() == 1
        cc = lambda k: (br[f"{k}_r"] + 1j * br[f"{k}_i"]).to_numpy()
        vf, vt = v[br.from_bus.to_numpy()], v[br.to_bus.to_numpy()]
        sf = vf * np.conj(cc("Yff") * vf + cc("Yft") * vt) * 100
        brgap = max(brgap, float(np.abs(sf.real - br.pf.to_numpy())[on].max()))
    res |= {"n_recovered": len(rec), "max_pf_residual_pu": worst, "min_vm": minvm, "max_gen_sum_gap_mw": sumgap, "max_branch_pf_gap_mw": brgap}
    out[g] = res; print(g, json.dumps(res), flush=True)
(H / "verify_written.json").write_text(json.dumps(out, indent=1))
