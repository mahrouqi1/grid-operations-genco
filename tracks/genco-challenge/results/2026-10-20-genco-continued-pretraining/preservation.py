"""B preservation ratios per protocol.json (audit R1-F8: the generating script for preservation.json). UNAUDITED.
Held-out 15: per grid, arm error / released ('zeroshot' row of the same zeroshot-<tag>/results.jsonl) for branch-angle
(genco branch_dva_mae_deg) and |V| (genco_pinned vm_mae_pu); median over grids. Normal-scenario hidden |V| MAE ratio:
pooled.json final / step0 (released weights), mean over the three mask seeds."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
out = {}
for tag in ("final", "step500"):
    rows = [json.loads(l) for l in open(R / f"zeroshot-{tag}/results.jsonl") if l.strip()]
    by = {(r["grid"], r["model"]): r for r in rows if not r.get("error")}
    grids = sorted({g for g, m in by})
    for arm in ("retrain_fixed", "retrain_published"):
        gs = [g for g in grids if (g, arm) in by and (g, "zeroshot") in by]
        a = [by[(g, arm)]["errors"]["genco"]["branch_dva_mae_deg"] / by[(g, "zeroshot")]["errors"]["genco"]["branch_dva_mae_deg"] for g in gs]
        v = [by[(g, arm)]["errors"]["genco_pinned"]["vm_mae_pu"] / by[(g, "zeroshot")]["errors"]["genco_pinned"]["vm_mae_pu"] for g in gs]
        out[f"{tag}/{arm}"] = {"n": len(a), "median_angle_ratio_to_released": float(np.median(a)),
                               "median_V_ratio_to_released": float(np.median(v)), "angle_better_than_released": int(sum(x < 1 for x in a))}
P = json.load(open(R / "pooled.json"))
nm = {(p["arm"], p["state"]): np.mean([p[s]["normal_vm_mae"] for s in ("1000", "1001", "1002")]) for p in P}
out["normal_vm_mae_ratio_fixed_final"] = float(nm[("fixed", "final")] / nm[("fixed", "step0")])
out["normal_vm_mae_ratio_published_final"] = float(nm[("published", "final")] / nm[("published", "step0")])
json.dump(out, open(R / "preservation.json", "w"), indent=1)
print(json.dumps(out, indent=1))
