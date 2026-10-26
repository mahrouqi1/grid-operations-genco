"""Tables for report.md from results.jsonl (no model runs). Writes summary.csv, summary.json, tables.md. UNAUDITED."""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
R = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines()]
rows, summ = [], {}
f = lambda x, d=3: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
for g in R:
    pre = [r for r in g["runs"] if r["mask"] == "pre"]; pf = [r for r in g["runs"] if r["mask"] == "pf"][0]
    def col(k, rs=pre):
        v = [r.get(k) for r in rs]
        return v if all(x is not None for x in v) else [np.nan] * len(rs)
    s = {"set": g["set"], "grid": g["grid"], "n_coll_scen": pre[0]["n_coll_scen"], "n_roots": g["n_roots"], "n_low_buses": pre[0]["n_low_buses"],
         "n_low_hidden_per_seed": [r["n_low_hidden"] for r in pre]}
    for k in ("frac_pred_lt_0.5", "pred_median", "label_median", "mae_vs_label", "mae_vs_root", "mae_vs_label_same_buses", "frac_buses_closer_to_root",
              "frac_within_0.1_of_label"):
        v = col(k); s[f"pre_{k}_seeds"] = v; s[f"pre_{k}_mean"] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else None
        s[f"pf_{k}"] = pf.get(k)
    s["pre_root_n_buses"] = [r.get("root_n_buses") for r in pre]
    nm = [r.get("normal") for r in pre]
    s["normal_pre"] = {k: float(np.mean([n[k] for n in nm])) for k in ("mae_vs_label", "frac_pred_lt_0.5", "pred_median", "label_median")} if all(nm) else None
    s["normal_pf"] = pf.get("normal")
    s["pre_no_visible_low"] = [r.get("no_visible_low") for r in pre]
    s["pre_n_no_visible_low"] = [r.get("n_low_hidden_no_visible_low_in_scenario") for r in pre]
    s["pre_coll_other_mae"] = float(np.mean([r.get("coll_scen_other_hidden_pq_mae") for r in pre]))
    s["pf_coll_other_mae"] = pf.get("coll_scen_other_hidden_pq_mae")
    summ[g["grid"]] = s
(HERE / "summary.json").write_text(json.dumps(summ, indent=1))
flat = []
for g, s in summ.items():
    flat.append({k: v for k, v in s.items() if not isinstance(v, (list, dict))} | {"seed_min_frac_lt_0.5": min(s["pre_frac_pred_lt_0.5_seeds"]), "seed_max_frac_lt_0.5": max(s["pre_frac_pred_lt_0.5_seeds"])})
pd.DataFrame(flat).to_csv(HERE / "summary.csv", index=False)
T = []
for setname, title in (("pretrain", "Pretraining subgrids"), ("ladder", "Held-out ladder grids")):
    T.append(f"### {title}\n")
    T.append("| grid | coll. scen. (roots) | hidden collapsed buses | PRE: frac pred < 0.5 (3 seeds) | PRE: median pred / median label | PRE: MAE vs label | PRE: MAE vs root (label vs root) | PF: frac < 0.5 | PF: median pred | PF: MAE vs label | normal hidden PQ MAE vs label: PRE / PF |")
    T.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g, s in summ.items():
        if s["set"] != setname: continue
        seeds = "/".join(f(x, 2) for x in s["pre_frac_pred_lt_0.5_seeds"])
        root = f"{f(s['pre_mae_vs_root_mean'])} ({f(np.nanmean([r.get('label_vs_root_mae', np.nan) for r in [x for x in R if x['grid']==g][0]['runs'] if r['mask']=='pre']))})" if s["pre_mae_vs_root_mean"] is not None else "-"
        nr = s["normal_pre"]; nf = s["normal_pf"]
        T.append(f"| {g.removeprefix('subnet_').removeprefix('pglib_opf_')} | {s['n_coll_scen']} ({s['n_roots']}) | {int(np.mean(s['n_low_hidden_per_seed']))} of {s['n_low_buses']} | {seeds} | {f(s['pre_pred_median_mean'])} / {f(s['pre_label_median_mean'])} | {f(s['pre_mae_vs_label_mean'])} | {root} | {f(s['pf_frac_pred_lt_0.5'],2)} | {f(s['pf_pred_median'])} | {f(s['pf_mae_vs_label'])} | {f(nr['mae_vs_label'],4) if nr else '-'} / {f(nf['mae_vs_label'],4) if nf else '-'} |")
    T.append("")
(HERE / "tables.md").write_text("\n".join(T))
print("\n".join(T))

# pooled over the 9 pretraining subgrids (weighted by hidden collapsed buses per run) and the verdict count
P = [g for g in R if g["set"] == "pretrain"]
lines = ["### Pooled over the 9 pretraining subgrids (weights = hidden collapsed buses)\n"]
for seed_i, seed in enumerate((1000, 1001, 1002)):
    runs = [[r for r in g["runs"] if r["mask"] == "pre" and r["seed"] == seed][0] for g in P]
    w = np.array([r["n_low_hidden"] for r in runs], float)
    lines.append(f"- PRE seed {seed}: hidden collapsed buses {int(w.sum())}; frac pred < 0.5 = {np.average([r['frac_pred_lt_0.5'] for r in runs], weights=w):.3f}; MAE vs label = {np.average([r['mae_vs_label'] for r in runs], weights=w):.3f}")
runs = [[r for r in g["runs"] if r["mask"] == "pf"][0] for g in P]
w = np.array([r["n_low_hidden"] for r in runs], float)
lines.append(f"- PF task: hidden collapsed buses {int(w.sum())}; frac pred < 0.5 = {np.average([r['frac_pred_lt_0.5'] for r in runs], weights=w):.3f}; MAE vs label = {np.average([r['mae_vs_label'] for r in runs], weights=w):.3f}")
hold = [g["grid"] for g in P if all(r["frac_pred_lt_0.5"] >= 0.5 for r in g["runs"] if r["mask"] == "pre") and np.mean([r["mae_vs_label"] for r in g["runs"] if r["mask"] == "pre"]) < 0.5 * [r for r in g["runs"] if r["mask"] == "pf"][0]["mae_vs_label"]]
lines.append(f"- Grids where every seed has frac < 0.5 >= 0.5 AND mean PRE MAE vs label < half the PF MAE: {len(hold)} of {len(P)}")
mins = {g["grid"]: min(r["frac_pred_lt_0.5"] for r in g["runs"] if r["mask"] == "pre") for g in P}
lines.append("- lowest per-seed frac < 0.5 by subgrid: " + ", ".join(f"{k.removeprefix('subnet_')} {v:.2f}" for k, v in mins.items()))
(HERE / "tables.md").write_text((HERE / "tables.md").read_text() + "\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
