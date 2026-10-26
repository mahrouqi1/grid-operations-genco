"""Seed-200 step-25 identity check (protocol_amendment_seeds.json gate). Usage: identity_compare.py <new_step25.pt> <stored_step25.pt> <step0.pt>
Prints per-tensor max abs difference new-vs-stored, the drift step0->step25 for scale, and PASS/FAIL (tolerance 1e-4)."""
import json
import sys

import torch

TOL = 1e-4
new, old, s0 = (torch.load(p, map_location="cpu") for p in sys.argv[1:4])
assert new.keys() == old.keys() == s0.keys()
worst, drift, n = 0.0, 0.0, 0
rows = []
for k in new:
    if not torch.is_tensor(new[k]) or not new[k].is_floating_point():
        continue
    d = float((new[k] - old[k]).abs().max())
    g = float((old[k] - s0[k]).abs().max())
    worst, drift, n = max(worst, d), max(drift, g), n + 1
    rows.append((d, g, k))
rows.sort(reverse=True)
print("worst 5 (diff new-vs-stored, drift step0-vs-stored, tensor):")
for r in rows[:5]:
    print(f"  {r[0]:.3e}  {r[1]:.3e}  {r[2]}")
res = {"tensors": n, "max_abs_diff_new_vs_stored": worst, "max_abs_drift_step0_to_step25": drift, "tolerance": TOL,
       "verdict": "PASS" if worst <= TOL else "FAIL"}
print(json.dumps(res))
sys.exit(0 if worst <= TOL else 1)
