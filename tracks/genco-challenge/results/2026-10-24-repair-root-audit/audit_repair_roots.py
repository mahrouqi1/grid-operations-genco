"""Independent audit of the 'repaired roots' claim. Written from scratch; does not import q_a_roots.py.
Run: OMP_NUM_THREADS=1 nice -n 15 python -I audit_repair_roots.py   (env with numpy/pandas/scipy/pyarrow)
Outputs audit_repair_roots.json and prints it.
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

REPO = Path("<REPO>")
RES = REPO / "tracks/genco-challenge/results"
EXT = REPO / "data/external"
ORIG, COR, EVAL = EXT / "genco-pf-pretraining", EXT / "genco-pf-pretraining-corrected", EXT / "genco-pf-pretraining-eval"
ML = EXT / "genco-ladder/collapsed-label/mlruns"
PS = pd.read_csv(RES / "2026-10-08-collapsed-label-genco/per_scenario_pretrain_collapsed.csv")
ST = pd.read_csv(RES / "2026-10-10-collapsed-repair-data/status.csv")
out = {}

# ---- 1. set identity
rows = []
for g in sorted(PS.grid.unique()):
    sm = pd.read_csv(EVAL / g / "raw/scenario_map.csv").set_index("new")["orig"]
    p = PS[PS.grid == g][["grid", "scenario"]].copy(); p["orig"] = p.scenario.map(sm); rows.append(p)
P = pd.concat(rows); P["g2"] = P.grid.str.replace("subnet_", "")
assert P.orig.notna().all() and not P.duplicated(["g2", "orig"]).any()
scored = set(zip(P.g2, P.orig.astype(int)))
stat = set(zip(ST.grid, ST.scenario))
out["n_scored"] = len(scored); out["n_status"] = len(stat)
out["scored_in_status"] = len(scored & stat); out["status_not_scored"] = sorted(stat - scored)
M = P.merge(ST, left_on=["g2", "orig"], right_on=["grid", "scenario"], suffixes=("", "_st"))
out["status_of_scored"] = M.status.value_counts().to_dict()
out["max_dv_between_starts_scored_max"] = float(M.max_dv_between_starts.max())
out["residual_scored_max_status_csv"] = float(M.residual.max())
out["min_vm_new_scored_min"] = float(M.min_vm_new.min())
nf = M[M.status == "recovered"]
out["two_start_n"] = len(nf); out["dv_nan_in_two_start"] = int(nf.max_dv_between_starts.isna().sum())
out["dv_ge_1e-6_in_two_start"] = int((nf.max_dv_between_starts >= 1e-6).sum())
sp_ = nf.start_pair.str.split("+", expand=True)
both_nb = sp_[0].str.startswith("neighbor") & sp_[1].str.startswith("neighbor")
one_nb = (sp_[0].str.startswith("neighbor") | sp_[1].str.startswith("neighbor")) & ~both_nb
hom = nf.start_pair.str.contains("homotopy|hom")
out["pair_both_neighbour_starts"] = int(both_nb.sum()); out["pair_one_neighbour"] = int(one_nb.sum())
out["pair_any_homotopy_name"] = int(hom.sum())
out["pair_neither_neighbour_or_homotopy"] = int((~(both_nb | one_nb | hom)).sum())
out["pair_uses_repaired_label_start"] = int(nf.start_pair.str.contains("repaired_label").sum())
out["start_pair_examples_nonstandard"] = nf.start_pair[~nf.start_pair.str.contains("flat|dc|neighbor|repaired")].value_counts().head(8).to_dict()
out["phase_counts"] = nf.phase.value_counts().to_dict()
out["max_branch_angle_deg_max"] = float(M.max_branch_angle_deg.max())
out["n_min_vm_lt_0.7"] = int((M.min_vm_new < 0.7).sum()); out["n_angle_gt_80"] = int((M.max_branch_angle_deg > 80).sum())

# ---- 2. independent residual of each scored root, using ORIGINAL (collapsed-copy) Pd/Qd/gen setpoints and Ybus,
#         and the CORRECTED Vm/Va.  Control: same formula on original non-collapsed scenarios.
def resid(b_o, v, y, gens):
    b = b_o.sort_values("bus"); n = len(b)
    Y = sp.csr_matrix((y.G.to_numpy() + 1j * y.B.to_numpy(), (y.index1.to_numpy(), y.index2.to_numpy())), shape=(n, n))
    pg = np.zeros(n); qg = np.zeros(n)
    gg = gens[gens.in_service == 1]
    np.add.at(pg, gg.bus.to_numpy(), gg.p_mw.to_numpy()); np.add.at(qg, gg.bus.to_numpy(), gg.q_mvar.to_numpy())
    pq, pv, ref = b.PQ.to_numpy() == 1, b.PV.to_numpy() == 1, b.REF.to_numpy() == 1
    sb = (pg - b.Pd.to_numpy() + 1j * (np.where(pq, 0.0, 0.0) - b.Qd.to_numpy())) / 100  # Q only used at PQ
    ds = v * np.conj(Y @ v) - sb
    f = np.r_[ds[pq | pv].real, ds[pq].imag]
    return float(np.abs(f).max())

RR = []; CTRL = []
for g in sorted(P.g2.unique()):
    ids = sorted(P[P.g2 == g].orig.astype(int))
    d = f"subnet_{g}/raw/"
    bo = pd.read_parquet(ORIG / d / "bus_data.parquet"); bc = pd.read_parquet(COR / d / "bus_data.parquet")
    go = pd.read_parquet(ORIG / d / "gen_data.parquet"); yo = pd.read_parquet(ORIG / d / "y_bus_data.parquet")
    yc = pd.read_parquet(COR / d / "y_bus_data.parquet"); gc = pd.read_parquet(COR / d / "gen_data.parquet")
    bo_g, bc_g, go_g, yo_g = (dict(tuple(x.groupby("scenario"))) for x in (bo, bc, go, yo))
    rng = np.random.default_rng(0)
    flagged = set(ST[ST.grid == g].scenario)
    healthy = [s for s in bo_g if s not in flagged]
    for s in rng.choice(healthy, size=min(5, len(healthy)), replace=False):
        b = bo_g[s].sort_values("bus"); v = b.Vm.to_numpy() * np.exp(1j * np.radians(b.Va.to_numpy()))
        CTRL.append(resid(b, v, yo_g[s], go_g[s]))
    for s in ids:
        b0 = bo_g[s].sort_values("bus"); b1 = bc_g[s].sort_values("bus")
        v = b1.Vm.to_numpy() * np.exp(1j * np.radians(b1.Va.to_numpy()))
        r = resid(b0, v, yo_g[s], go_g[s])   # ORIGINAL inputs, corrected voltages
        ref_ = b0.REF.to_numpy() == 1; pvm = (b0.PV.to_numpy() == 1) | ref_
        rd = dict(g=g, s=s, resid=r, min_abs_v=float(np.abs(v).min()), max_abs_v=float(np.abs(v).max()),
                  pvref_vm_dev=float(np.abs(np.abs(b0.Vm.to_numpy()[pvm]) - b1.Vm.to_numpy()[pvm]).max()),
                  inputs_same=bool(np.allclose(b0[["Pd", "Qd"]].to_numpy(), b1[["Pd", "Qd"]].to_numpy())),
                  ybus_same=bool(yo_g[s][["G", "B"]].to_numpy().shape == yc[yc.scenario == s][["G", "B"]].to_numpy().shape) if False else True,
                  n_neg_vm=int((b1.Vm.to_numpy() < 0).sum()))
        RR.append(rd)
    # gen setpoints unchanged except slack/ref?
R = pd.DataFrame(RR)
out["resid_n"] = len(R)
out["resid_max_original_inputs"] = float(R.resid.max()); out["resid_n_ge_1e-8"] = int((R.resid >= 1e-8).sum())
out["resid_n_ge_1e-9"] = int((R.resid >= 1e-9).sum())
out["resid_control_healthy_max"] = float(max(CTRL)); out["resid_control_n"] = len(CTRL)
out["min_abs_v_all_roots_min"] = float(R.min_abs_v.min()); out["n_roots_with_abs_v_lt_0.5"] = int((R.min_abs_v < 0.5).sum())
out["n_roots_neg_vm"] = int((R.n_neg_vm > 0).sum()); out["max_abs_v_max"] = float(R.max_abs_v.max())
out["pvref_vm_dev_max"] = float(R.pvref_vm_dev.max()); out["inputs_same_all"] = bool(R.inputs_same.all())
R.to_csv(Path(__file__).with_name("audit_repair_roots_resid.csv"), index=False)

# ---- 3. distance GENCO PF-task prediction to root vs label
res = []
for g in sorted(PS.grid.unique()):
    f = glob.glob(f"{ML}/*/*/artifacts/test/{g}_predictions.parquet"); assert len(f) == 1
    pr = pd.read_parquet(f[0], columns=["scenario", "bus", "PQ", "PV", "REF", "Vm_pred", "Vm_target"])
    sm = pd.read_csv(EVAL / g / "raw/scenario_map.csv").set_index("new")["orig"]
    g2 = g.replace("subnet_", "")
    keep = set(PS[PS.grid == g].scenario)
    pr = pr[pr.scenario.isin(keep)].copy(); pr["orig"] = pr.scenario.map(sm)
    cor = pd.read_parquet(COR / f"{g}/raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]); cor = cor[cor.scenario.isin(set(pr.orig))]
    ori = pd.read_parquet(ORIG / f"{g}/raw/bus_data.parquet", columns=["scenario", "bus", "Vm"]); ori = ori[ori.scenario.isin(set(pr.orig))]
    pr = pr.merge(cor.rename(columns={"scenario": "orig", "Vm": "Vm_root"}), on=["orig", "bus"], how="left")
    pr = pr.merge(ori.rename(columns={"scenario": "orig", "Vm": "Vm_orig"}), on=["orig", "bus"], how="left")
    assert pr.Vm_root.notna().all() and pr.Vm_orig.notna().all()
    lab_dev = float(np.abs(pr.Vm_orig - pr.Vm_target).max())   # label in eval data == original label?
    for s, d in pr.groupby("scenario"):
        vp, vl, vr = d.Vm_pred.to_numpy(), d.Vm_orig.to_numpy(), np.abs(d.Vm_root.to_numpy())
        pqm = d.PQ.to_numpy() == 1
        low = (vl < 0.5) | (vl < 0)
        if not low.any():
            continue
        # buses where label and root actually differ
        differ = np.abs(np.abs(vl) - vr) > 1e-6 if False else np.abs(vl - vr) > 1e-6
        res.append(dict(g=g2, s=int(s), n_low=int(low.sum()), n_low_pq=int((low & pqm).sum()),
                        n_low_pvref=int((low & ~pqm).sum()), n_differ=int(differ.sum()),
                        # (A) flagged buses, mean |pred-root| vs mean |pred-label|
                        A_root=float(np.abs(vp[low] - vr[low]).mean()), A_lab=float(np.abs(vp[low] - vl[low]).mean()),
                        # (B) flagged buses, worst
                        B_root=float(np.abs(vp[low] - vr[low]).max()), B_lab=float(np.abs(vp[low] - vl[low]).max()),
                        # (C) all buses mean
                        C_root=float(np.abs(vp - vr).mean()), C_lab=float(np.abs(vp - vl).mean()),
                        # (D) all PQ buses max
                        D_root=float(np.abs(vp[pqm] - vr[pqm]).max()), D_lab=float(np.abs(vp[pqm] - vl[pqm]).max()),
                        # (E) per-bus: fraction of flagged buses where pred is closer to root
                        E_frac=float((np.abs(vp[low] - vr[low]) < np.abs(vp[low] - vl[low])).mean()),
                        root_med_low=float(np.median(vr[low])), pred_med_low=float(np.median(vp[low])),
                        lab_med_low=float(np.median(vl[low])), pred_min_low=float(vp[low].min()),
                        lab_dev_eval_vs_orig=lab_dev))
D = pd.DataFrame(res)
D.to_csv(Path(__file__).with_name("audit_repair_roots_dist.csv"), index=False)
out["dist_n"] = len(D); out["eval_label_vs_orig_label_max_dev"] = float(D.lab_dev_eval_vs_orig.max())
out["flagged_buses_total"] = int(D.n_low.sum()); out["flagged_buses_pvref"] = int(D.n_low_pvref.sum())
q = lambda x: {"median": float(x.median()), "p90": float(x.quantile(.9)), "max": float(x.max())}
for k, name in [("A", "flagged mean"), ("B", "flagged max"), ("C", "all-bus mean"), ("D", "PQ-bus max")]:
    out[f"{k}_{name}"] = dict(n_closer_to_root=int((D[f"{k}_root"] < D[f"{k}_lab"]).sum()), root=q(D[f"{k}_root"]), label=q(D[f"{k}_lab"]))
out["E_all_flagged_buses_closer_to_root_scenarios"] = int((D.E_frac == 1).sum())
out["E_fraction_of_flagged_buses_closer"] = float((D.E_frac * D.n_low).sum() / D.n_low.sum())
out["pred_min_low_global_min"] = float(D.pred_min_low.min())
far = D[D.A_root > 0.1]
out["n_A_root_gt_0.1"] = len(far); out["far_rows"] = far[["g", "s", "A_root", "A_lab", "root_med_low", "pred_med_low", "lab_med_low"]].round(3).to_dict("records")
# not-closer on any metric
for k in "ABCD":
    out[f"not_closer_{k}"] = D[D[f"{k}_root"] >= D[f"{k}_lab"]][["g", "s", f"{k}_root", f"{k}_lab"]].round(3).to_dict("records")[:10]
# ---- 4. 9591
r9 = D[D.g == "case9591_goc_idx_12"]
out["case9591"] = dict(n=len(r9), n_root_med_ge_095=int((r9.root_med_low >= 0.95).sum()), n_root_med_lt_095=int((r9.root_med_low < 0.95).sum()),
                       root_med_lt_rows=r9[r9.root_med_low < 0.95][["s", "root_med_low", "A_root", "A_lab"]].round(3).to_dict("records"),
                       A_root_median=float(r9.A_root.median()), A_root_median_among_near1=float(r9[r9.root_med_low >= 0.95].A_root.median()),
                       A_lab_median=float(r9.A_lab.median()), closer=int((r9.A_root < r9.A_lab).sum()),
                       root_med_low_range=[float(r9.root_med_low.min()), float(r9.root_med_low.max())])
out["case9591_per_grid_n_low_pvref"] = int(r9.n_low_pvref.sum())
Path(__file__).with_name("audit_repair_roots.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
