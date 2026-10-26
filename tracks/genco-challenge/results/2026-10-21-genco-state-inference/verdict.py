"""Verdict for candidate #7, computed from the saved per-scenario files only (protocol.json is the spec). UNAUDITED.
    python verdict.py  ->  verdict.json, table.md"""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
G = P["grids"]; BASES = P["eligible_baselines"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
b = pd.read_json(HERE / "baselines_per_scenario.jsonl", lines=True)
g = pd.read_json(HERE / "genco_per_scenario.jsonl", lines=True).drop(columns=["diag"])
d = pd.concat([b, g], ignore_index=True)


def wide(grid, metric):
    x = d[(d.grid == grid)].pivot(index="scenario", columns="method", values=metric)
    return x.dropna(subset=["genco_det20"] if metric != "pg_mae" else ["genco_det20"])


def ratio_ci(x, metric_cols=BASES):
    G_ = x["genco_det20"].to_numpy(); B_ = x[metric_cols].to_numpy()
    point = np.median(G_) / np.median(B_, axis=0).min()
    idx = rng.integers(0, len(x), (NB, len(x)))
    bs = np.array([np.median(G_[i]) / np.median(B_[i], axis=0).min() for i in idx])
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), metric_cols[int(np.argmin(np.median(B_, axis=0)))]


rows = []
for gr in G:
    r = {"grid": gr, "n": int(len(wide(gr, "dva_mae")))}
    for m in P["primary_metrics"] + P["secondary_metrics"]:
        x = wide(gr, m)
        pt, lo, hi, best = ratio_ci(x)
        r[m] = {"ratio": pt, "lo": lo, "hi": hi, "best_baseline": best, "genco_median": float(x["genco_det20"].median()),
                "best_median": float(x[best].median()), **{c + "_median": float(x[c].median()) for c in x.columns if c in ("flat", "oracle", "genco_det100", *BASES)}}
        r[m]["win"] = bool(pt < 1 and hi < 1)
        r[m]["win_point_only"] = bool(pt < 1)
    r["grid_win"] = all(r[m]["win"] for m in P["primary_metrics"])
    r["grid_win_point_only"] = all(r[m]["win_point_only"] for m in P["primary_metrics"])
    rows.append(r)
nwin = sum(r["grid_win"] for r in rows)
v = {"verdict": "PASS" if nwin >= P["threshold_grids"] else "FAIL", "grids_won_both_with_ci": nwin, "grids_won_both_point_only": sum(r["grid_win_point_only"] for r in rows),
     "per_metric_wins_with_ci": {m: sum(r[m]["win"] for r in rows) for m in P["primary_metrics"] + P["secondary_metrics"]},
     "threshold_grids": P["threshold_grids"], "rows": rows}
(HERE / "verdict.json").write_text(json.dumps(v, indent=1))
L = ["| grid | n | dVa GENCO | dVa best base (which) | ratio [CI] | Vm GENCO | Vm best base | ratio [CI] | Pg GENCO MW | Pg best base | ratio [CI] | win |", "|" + "---|" * 12]
for r in rows:
    a, c, p = r["dva_mae"], r["vm_mae_unmeas"], r["pg_mae"]
    L.append(f"| {r['grid'].replace('pglib_opf_','')} | {r['n']} | {a['genco_median']:.3f} | {a['best_median']:.3f} ({a['best_baseline'].split('_')[0]}) | {a['ratio']:.2f} [{a['lo']:.2f},{a['hi']:.2f}] | "
             f"{c['genco_median']:.4f} | {c['best_median']:.4f} ({c['best_baseline'].split('_')[0]}) | {c['ratio']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] | "
             f"{p['genco_median']:.1f} | {p['best_median']:.1f} ({p['best_baseline'].split('_')[0]}) | {p['ratio']:.2f} [{p['lo']:.2f},{p['hi']:.2f}] | {'WIN' if r['grid_win'] else '-'} |")
(HERE / "table.md").write_text("\n".join(L) + "\n")
print(v["verdict"], nwin, "of", len(rows), "| point-only", v["grids_won_both_point_only"], "| per metric", v["per_metric_wins_with_ci"])
