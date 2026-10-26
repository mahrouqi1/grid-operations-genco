"""Pooled collapsed-bus reproduction under the pretraining mask, per arm, weighted by hidden collapsed buses per run
(same weighting as results/2026-10-09-collapsed-label-pretrain-mask/summarize.py). Writes pretrain_mask_pooled.md."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "../2026-10-09-collapsed-label-pretrain-mask/results.jsonl")
out = ["| model | mask seed | hidden collapsed buses | frac predicted < 0.5 | MAE vs collapsed label | MAE vs physical root (buses with a root) |", "|---|---|---|---|---|---|"]
for name, f in (("released", REF), ("retrain_fixed", f"{HERE}/pretrain_mask_fixed.jsonl"), ("retrain_published", f"{HERE}/pretrain_mask_published.jsonl")):
    G = [json.loads(l) for l in open(f) if json.loads(l)["set"] == "pretrain"]
    for seed in (1000, 1001, 1002):
        runs = [r for g in G for r in g["runs"] if r["mask"] == "pre" and r["seed"] == seed]
        if not runs: continue
        w = np.array([r["n_low_hidden"] for r in runs], float)
        rr = [r for r in runs if r.get("mae_vs_root") is not None]
        wr = np.array([r["root_n_buses"] for r in rr], float)
        out.append(f"| {name} | {seed} | {int(w.sum())} | {np.average([r['frac_pred_lt_0.5'] for r in runs], weights=w):.3f} | "
                   f"{np.average([r['mae_vs_label'] for r in runs], weights=w):.3f} | "
                   f"{np.average([r['mae_vs_root'] for r in rr], weights=wr):.3f} |" if len(rr) else "n/a |")
open(f"{HERE}/pretrain_mask_pooled.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
