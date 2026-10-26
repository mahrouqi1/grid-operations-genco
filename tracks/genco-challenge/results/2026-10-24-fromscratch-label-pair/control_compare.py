"""Identity control: control/released.jsonl (released checkpoint, this folder's invocation) vs the continued-pretraining
study's pmask/published_step0.jsonl (same checkpoint, run 2026-10-20). Compares the per-run frac_pred_lt_0.5 and normal MAE
for every grid x mask x seed. UNAUDITED."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
A = {json.loads(l)["grid"]: json.loads(l) for l in open(H / "control/released.jsonl") if l.strip()}
B = {json.loads(l)["grid"]: json.loads(l) for l in open(H.parent / "2026-10-20-genco-continued-pretraining/pmask/published_step0.jsonl") if l.strip()}
d1 = d2 = 0.0
n = 0
for g in sorted(set(A) & set(B)):
    for ra, rb in zip(A[g]["runs"], B[g]["runs"]):
        assert (ra["mask"], ra["seed"]) == (rb["mask"], rb["seed"])
        d1 = max(d1, abs(ra["frac_pred_lt_0.5"] - rb["frac_pred_lt_0.5"]))
        if "normal" in ra and "normal" in rb:
            d2 = max(d2, abs(ra["normal"]["mae_vs_label"] - rb["normal"]["mae_vs_label"]))
        n += 1
out = {"grids_compared": len(set(A) & set(B)), "runs_compared": n, "max_abs_diff_frac_pred_lt_0.5": d1, "max_abs_diff_normal_mae": d2}
json.dump(out, open(H / "control/compare.json", "w"), indent=1)
print(out)
