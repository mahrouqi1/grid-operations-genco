"""Fixed vs published arm, per grid and metric, from zeroshot/results.jsonl (same rows as tally.md). Writes compare_arms.md."""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {(r["grid"], r["model"]): r for r in map(json.loads, open(f"{HERE}/zeroshot/results.jsonl")) if "error" not in r}
grids = sorted({g for g, m in rows if m == "zeroshot"}, key=lambda g: rows[(g, "zeroshot")]["n_bus"])
M = {"residual": lambda r: r["genco_active_res_mw"], "branch angle": lambda r: r["errors"]["genco"]["branch_dva_mae_deg"],
     "V (pinned)": lambda r: r["errors"]["genco_pinned"]["vm_mae_pu"]}
out = ["| metric | fixed better than published (of 15) | median fixed/published ratio | fixed better than released zero-shot | published better than released zero-shot |", "|---|---|---|---|---|"]
for k, f in M.items():
    fb = sum(f(rows[(g, "retrain_fixed")]) < f(rows[(g, "retrain_published")]) for g in grids)
    ratio = st.median(f(rows[(g, "retrain_fixed")]) / f(rows[(g, "retrain_published")]) for g in grids)
    fr = sum(f(rows[(g, "retrain_fixed")]) < f(rows[(g, "zeroshot")]) for g in grids)
    pr = sum(f(rows[(g, "retrain_published")]) < f(rows[(g, "zeroshot")]) for g in grids)
    out.append(f"| {k} | {fb} | {ratio:.2f} | {fr} | {pr} |")
open(f"{HERE}/compare_arms.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
