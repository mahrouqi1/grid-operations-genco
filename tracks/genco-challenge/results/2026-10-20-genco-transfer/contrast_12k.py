"""Audit R1-F4 prepared result: the pre-registered 12,000-update contrast (every run's state_step12000 on the locked
15), per arm: median over grids of the seed-mean branch-angle ratio to DC, change vs released, grids better than
released, |V| error worse than released (count). NOT in the report until an owner decision is answered. UNAUDITED."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H = json.load(open(HERE.parents[0] / "2026-10-19-genco-retrain-scoring/zeroshot/run_meta.json"))["grids"]
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
rel = {r["grid"]: r for r in rows(HERE.parents[0] / "2026-10-19-genco-retrain-scoring/zeroshot/results.jsonl") if r["model"] == "zeroshot"}
ang = lambda r: r["errors"]["genco"]["branch_dva_mae_deg"] / r["errors"]["dc"]["branch_dva_mae_deg"]
vm = lambda r: r["errors"]["genco_pinned"]["vm_mae_pu"]
rel_med = float(np.median([ang(rel[g]) for g in H]))
out, L = {}, ["# 12,000-update contrast on the locked 15 (contrast_12k.py; audit R1-F4 prepared result; UNAUDITED)", "",
              f"Released median branch-angle ratio {rel_med:.3f}.", "",
              "| arm | median seed-mean angle ratio | change vs released | grids better than released (angle) | grids with \\|V\\| error worse than released |",
              "|---|---|---|---|---|"]
for arm in ("C00", "C01", "D", "a", "b", "c", "m"):
    A, V = [], []
    for s in (0, 1, 2):
        R = {r["grid"]: r for r in rows(HERE / "heldout-12k" / f"{arm}_s{s}" / "results.jsonl") if not r.get("error")}
        A.append([ang(R[g]) for g in H]); V.append([vm(R[g]) for g in H])
    a, v = np.mean(A, 0), np.mean(V, 0)
    med = float(np.median(a))
    better = int(sum(a[i] < ang(rel[g]) for i, g in enumerate(H)))
    vworse = int(sum(v[i] > vm(rel[g]) for i, g in enumerate(H)))
    out[arm] = {"median_angle_ratio": med, "change_vs_released": med / rel_med - 1, "angle_better": better, "V_worse": vworse}
    L.append(f"| {arm} | {med:.3f} | {100*(med/rel_med-1):+.1f}% | {better} | {vworse} |")
L += ["", "Data effect at 12k: D vs C00 (random mask), m vs C01 (50/50). Mask effect at 12k: C01 vs C00 (no ladder), "
      "a/c/m vs D (with ladder); b at 12k is still pure pretraining mask (equivalent to D)."]
(HERE / "contrast_12k.md").write_text("\n".join(L) + "\n"); json.dump(out, open(HERE / "contrast_12k.json", "w"), indent=1)
print("\n".join(L))
