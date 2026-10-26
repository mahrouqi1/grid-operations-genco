"""Pool pmask/<arm>_<tag>.jsonl: collapse reproduction (share of hidden collapsed PQ buses predicted < 0.5 pu, pretraining
mask, seeds 1000-1002) pooled over the 7 training-only subgrids weighted by hidden collapsed buses (same pooling as
../2026-10-20-genco-continued-pretraining/pool.py), plus normal-scenario hidden-PQ |V| MAE (underfit gate), plus the
all-9-subgrid share and the 'pf' mask/PF-decoder row (m3 proxy: distance of |V| prediction to the label/root is in the
'pf' run if present). UNAUDITED."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TRAIN7 = {"subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case5658_epigrids_idx_8",
          "subnet_case4837_goc_idx_28", "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37", "subnet_case10480_goc_idx_26"}

def pooled(path, grids):
    G = [json.loads(l) for l in open(path) if l.strip()]
    G = [g for g in G if g["grid"] in grids]
    o = {"grids": len(G)}
    per = {}
    for seed in (1000, 1001, 1002):
        runs = [r for g in G for r in g["runs"] if r["mask"] == "pre" and r["seed"] == seed]
        w = np.array([r["n_low_hidden"] for r in runs], float); wn = np.array([r["normal"]["n_hidden_pq"] for r in runs], float)
        per[seed] = {"hidden_collapsed": int(w.sum()),
                     "frac_lt_0.5": float(np.average([r["frac_pred_lt_0.5"] for r in runs], weights=w)),
                     "pred_median": float(np.average([r["pred_median"] for r in runs], weights=w)),
                     "mae_vs_label": float(np.average([r["mae_vs_label"] for r in runs], weights=w)),
                     "normal_vm_mae": float(np.average([r["normal"]["mae_vs_label"] for r in runs], weights=wn))}
    o["seeds"] = per
    pf = [r for g in G for r in g["runs"] if r["mask"] == "pf"]
    pf = [r for r in pf if r.get("root_n_buses")]
    if pf:   # m3 proxy: PF mask + PF decoder, one draw (seed 1000); pooled by hidden collapsed buses with a physical root
        w = np.array([r["root_n_buses"] for r in pf], float)
        wl = np.array([r["n_low_hidden"] for r in pf], float)
        o["pf"] = {"n_buses_with_root": int(w.sum()),
                   "frac_lt_0.5": float(np.average([r["frac_pred_lt_0.5"] for r in pf], weights=wl)),
                   "mae_vs_root": float(np.average([r["mae_vs_root"] for r in pf], weights=w)),
                   "mae_vs_label": float(np.average([r["mae_vs_label_same_buses"] for r in pf], weights=w)),
                   "frac_closer_to_root": float(np.average([r["frac_buses_closer_to_root"] for r in pf], weights=w))}
    for k in ("frac_lt_0.5", "pred_median", "mae_vs_label", "normal_vm_mae"):
        o["mean_" + k] = float(np.mean([per[s][k] for s in per]))
    return o

if __name__ == "__main__":
    out = []
    for f in sorted((HERE / "pmask").glob("*.jsonl")):
        arm, tag = f.stem.split("_", 1)
        if sum(1 for l in open(f) if l.strip()) < 9:
            continue
        all9 = {json.loads(l)["grid"] for l in open(f) if l.strip()}
        out.append({"arm": arm, "tag": tag, "train7": pooled(f, TRAIN7), "all9": pooled(f, all9)})
    json.dump(out, open(HERE / "pooled_pmask.json", "w"), indent=1)
    for r in out:
        t = r["train7"]
        print(r["arm"], r["tag"], f"collapse {t['mean_frac_lt_0.5']:.3f}", f"normalMAE {t['mean_normal_vm_mae']:.4f}", f"allgrids {r['all9']['mean_frac_lt_0.5']:.3f}")
