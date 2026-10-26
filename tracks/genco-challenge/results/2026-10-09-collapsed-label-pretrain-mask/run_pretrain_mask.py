"""Do GENCO's pretrained weights reproduce collapsed Vm labels under the PRETRAINING task? UNAUDITED.

For each grid: run the released pretrained checkpoint (no training) under
  (pre, opf): the pretraining mask (AddPretrainMask verbatim, ~80% of Vm/Va hidden) + OPF decoder, 3 mask-draw seeds
  (pf,  pf) : the power-flow mask + PF decoder (the task scored in ../2026-10-08-collapsed-label-genco), 1 draw
using shift_eval.run_condition from ../2026-10-08-zeroshot-shift-decomposition. Collapsed scenario = label has a bus
with Vm < 0.5; collapsed bus = PQ bus with label Vm < 0.5 in such a scenario; only HIDDEN (masked) entries are scored
(visible entries pass through). Physical root = analyze.physical_root (our Newton) from the scenario's own Ybus/Sbus.

Run (env grid-operations-genco, repo root, GPU shared: CUDA_VISIBLE_DEVICES=<i> nice -n 10):
    python tracks/genco-challenge/results/2026-10-09-collapsed-label-pretrain-mask/run_pretrain_mask.py
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tracks/genco-challenge/results/2026-10-08-zeroshot-shift-decomposition"))
sys.path.insert(0, str(REPO / "tracks/genco-challenge/results/2026-10-08-collapsed-label-genco"))
import numpy as np, pandas as pd, torch, lightning as L
import shift_eval as S
import analyze as A

EXT = REPO / "data/external"
EVAL = EXT / "genco-pf-pretraining-eval"
LAD = EXT / "genco-ladder/data"
TREE = EXT / "genco-collapsed-pretrain-mask"
S.TREE = TREE
PRETRAIN = ["case4837_goc_idx_35", "case20758_epigrids_idx_49", "case9591_goc_idx_12", "case5658_epigrids_idx_8",
            "case4837_goc_idx_28", "case19402_goc_idx_47", "case10480_goc_idx_40", "case4837_goc_idx_37", "case10480_goc_idx_26"]
LADDER = ["case1354_pegase", "case2736sp_k", "case240_pserc", "case1888_rte", "case1951_rte"]
SEEDS = [1000, 1001, 1002]
VM_MIN = 0.5


def stage(setname, grid, src_raw):
    d = TREE / setname / grid
    d.mkdir(parents=True, exist_ok=True)
    link = d / "raw"
    if not link.exists():
        link.symlink_to(Path(src_raw).resolve())
    return TREE / setname


def roots_for(raw, scen):
    """{scenario: root |V| vector (sorted by bus) or None} for collapsed scenarios."""
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF", "Va_dc"])
    ybus = pd.read_parquet(raw / "y_bus_data.parquet", columns=["scenario", "index1", "index2", "G", "B"])
    sset = set(int(s) for s in scen)
    bg = {s: d for s, d in bus[bus.scenario.isin(sset)].groupby("scenario")}
    yg = {s: d for s, d in ybus[ybus.scenario.isin(sset)].groupby("scenario")}
    out = {}
    for s in scen:
        v, start, res = A.physical_root(bg[s], yg[s])
        out[int(s)] = None if v is None else np.abs(v)
    return out


def metrics(bus: pd.DataFrame, roots: dict) -> dict:
    """bus: per-bus table from run_condition for one draw. Hidden PQ buses only."""
    pq = bus["PQ"].to_numpy().astype(bool)
    hid = bus["mvm"].to_numpy().astype(bool)
    lab_min = bus.groupby("scenario")["vm_t"].transform("min").to_numpy()
    coll_s = lab_min < VM_MIN
    low = pq & (bus["vm_t"].to_numpy() < VM_MIN) & coll_s
    # scenarios that keep at least one VISIBLE low bus (collapse is observable from inputs)
    vis_low = bus.assign(v=low & ~hid).groupby("scenario")["v"].transform("any").to_numpy()
    sel = low & hid
    d = bus[sel]
    vp, vt = d["vm_p"].to_numpy(), d["vm_t"].to_numpy()
    out = {"n_coll_scen": int(bus.loc[coll_s, "scenario"].nunique()), "n_low_buses": int(low.sum()), "n_low_hidden": int(sel.sum()),
           "low_hidden_frac": float(sel.sum() / max(low.sum(), 1))}
    if sel.any():
        out |= {"pred_median": float(np.median(vp)), "pred_min": float(vp.min()), "label_median": float(np.median(vt)),
                "frac_pred_lt_0.5": float((vp < VM_MIN).mean()), "mae_vs_label": float(np.abs(vp - vt).mean()),
                "median_abs_err_vs_label": float(np.median(np.abs(vp - vt))),
                "frac_within_0.1_of_label": float((np.abs(vp - vt) < 0.1).mean())}
        # root comparison on hidden low buses of scenarios where a physical root exists
        e_r, e_l, e_lr, n = [], [], [], 0
        for s, g in d.groupby("scenario"):
            r = roots.get(int(s))
            if r is None:
                continue
            rv = r[g["bus"].to_numpy()]
            e_r.append(np.abs(g["vm_p"].to_numpy() - rv)); e_l.append(np.abs(g["vm_p"].to_numpy() - g["vm_t"].to_numpy()))
            e_lr.append(np.abs(g["vm_t"].to_numpy() - rv)); n += 1
        if n:
            e_r, e_l, e_lr = map(np.concatenate, (e_r, e_l, e_lr))
            out |= {"root_n_scen": n, "root_n_buses": int(len(e_r)), "mae_vs_root": float(e_r.mean()), "mae_vs_label_same_buses": float(e_l.mean()),
                    "label_vs_root_mae": float(e_lr.mean()), "frac_buses_closer_to_root": float((e_r < e_l).mean())}
        else:
            out |= {"root_n_scen": 0}
        # collapse observable from visible inputs?
        nv = sel & ~vis_low
        out["n_low_hidden_no_visible_low_in_scenario"] = int(nv.sum())
        if nv.any():
            dd = bus[nv]
            out["no_visible_low"] = {"frac_pred_lt_0.5": float((dd["vm_p"] < VM_MIN).mean()), "pred_median": float(dd["vm_p"].median()),
                                     "label_median": float(dd["vm_t"].median()), "n_scen": int(dd["scenario"].nunique())}
    # collapsed scenarios, other hidden PQ buses
    oth = pq & hid & coll_s & ~low
    if oth.any():
        out["coll_scen_other_hidden_pq_mae"] = float(np.abs(bus.loc[oth, "vm_p"] - bus.loc[oth, "vm_t"]).mean())
    # control: hidden PQ buses of NORMAL scenarios
    nm = pq & hid & ~coll_s
    if nm.any():
        e = np.abs(bus.loc[nm, "vm_p"].to_numpy() - bus.loc[nm, "vm_t"].to_numpy())
        out["normal"] = {"n_scen": int(bus.loc[~coll_s, "scenario"].nunique()), "n_hidden_pq": int(nm.sum()),
                         "pred_median": float(bus.loc[nm, "vm_p"].median()), "label_median": float(bus.loc[nm, "vm_t"].median()),
                         "mae_vs_label": float(e.mean()), "median_abs_err": float(np.median(e)),
                         "frac_pred_lt_0.5": float((bus.loc[nm, "vm_p"] < VM_MIN).mean())}
    return out


def run_grid(setname, grid, root, dev, roots_cache):
    raw = root / grid / "raw"
    bus0 = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Vm", "PQ"])
    n_scen = int(bus0["scenario"].nunique())
    L.seed_everything(0, workers=True)
    cfg = S.make_cfg(grid, n_scen)
    dm = S.LitGridHeteroDataModule(cfg, str(root)); dm.setup("test")
    norm = dm.data_normalizers[0]; task = S.get_task(cfg, dm.data_normalizers)
    sd = torch.load(S.CKPT, map_location="cpu"); task.load_state_dict(S.adapt_state_dict_for_model(sd, task)); task.to(dev).eval()
    loader = dm.test_dataloader()[0]
    test_ids = sorted(int(i) for i in dm.test_scenario_ids[0])
    va_dc = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Va_dc"])
    lm = bus0[bus0.scenario.isin(test_ids)].groupby("scenario")["Vm"].min()
    coll = sorted(int(s) for s in lm.index[lm < VM_MIN])
    t0 = time.time()
    roots = roots_cache.get(grid) or roots_for(raw, coll)
    print(grid, "collapsed", len(coll), "roots", sum(r is not None for r in roots.values()), f"{time.time()-t0:.0f}s", flush=True)
    res = {"set": setname, "grid": grid, "n_scen_test": len(test_ids), "n_scen_data": n_scen, "n_coll_test": len(coll),
           "n_roots": sum(r is not None for r in roots.values()), "runs": []}
    for mask, dec, seed in [("pre", "opf", s) for s in SEEDS] + [("pf", "pf", SEEDS[0])]:
        t0 = time.time()
        for attempt in range(6):
            try:
                bus, br, rp, rq = S.run_condition(task, loader, dev, norm, mask, dec, seed, va_dc)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError("OOM after 6 retries")
        if not np.isfinite(rp) or not np.isfinite(bus["vm_p"].to_numpy()).all():
            raise RuntimeError("non-finite output, not scored")
        m = metrics(bus, roots)
        res["runs"].append({"mask": mask, "decoder": dec, "seed": seed, "seconds": round(time.time() - t0, 1), "res_mw": rp, **m})
        print(json.dumps({"grid": grid, "mask": mask, "seed": seed, "frac<0.5": m.get("frac_pred_lt_0.5"), "pred_med": m.get("pred_median"),
                          "lab_med": m.get("label_median")}), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["pretrain", "ladder"])
    ap.add_argument("--grids", nargs="*")
    # added 2026-10-18 for the OSC retrain arms (default behaviour unchanged): score another checkpoint, write elsewhere
    ap.add_argument("--ckpt", type=Path, help="state dict to score instead of the released pretrained checkpoint")
    ap.add_argument("--out", type=Path, help="results file (default: results.jsonl next to this script)")
    a = ap.parse_args()
    if a.ckpt:
        S.CKPT = a.ckpt
    dev = torch.device("cuda:0")
    out = a.out or (HERE / "results.jsonl")
    done = {(r["set"], r["grid"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    jobs = []
    if "pretrain" in a.sets:
        jobs += [("pretrain", "subnet_" + g, stage("pretrain", "subnet_" + g, EVAL / ("subnet_" + g) / "raw")) for g in PRETRAIN]
    if "ladder" in a.sets:
        for g in LADDER:
            gg = "pglib_opf_" + g
            un = sorted((LAD / gg).glob("raw.unvalidated-*"))[0]
            jobs.append(("ladder", gg, stage("ladder", gg, un)))
    for setname, grid, root in jobs:
        if (setname, grid) in done or (a.grids and grid not in a.grids):
            continue
        r = run_grid(setname, grid, root, dev, {})
        with out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
