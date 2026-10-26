"""Arm H (SimBench-HV-only) numbers on the locked 15, protocol-simbench.json. Same metrics as go_no_go.py / contrast_12k.py.
Writes simbench_summary.json + simbench_summary.md (tables for report-simbench.md). UNAUDITED."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H15 = json.load(open(HERE.parent / "2026-10-19-genco-retrain-scoring/zeroshot/run_meta.json"))["grids"]
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
ang = lambda r: r["errors"]["genco"]["branch_dva_mae_deg"] / r["errors"]["dc"]["branch_dva_mae_deg"]
angerr = lambda r: r["errors"]["genco"]["branch_dva_mae_deg"]
vmr = lambda r: r["errors"]["genco_pinned"]["vm_mae_pu"] / r["errors"]["dc"]["vm_mae_pu"]
vme = lambda r: r["errors"]["genco_pinned"]["vm_mae_pu"]
rel = {r["grid"]: r for r in rows(HERE.parent / "2026-10-19-genco-retrain-scoring/zeroshot/results.jsonl") if r["model"] == "zeroshot" and not r.get("error")}
rel_med = float(np.median([ang(rel[g]) for g in H15]))


def load(d):
    return {r["grid"]: r for r in rows(d / "results.jsonl") if not r.get("error")}


def summarize(seed_rows):
    A = np.array([[ang(R[g]) for g in H15] for R in seed_rows])
    E = np.array([[angerr(R[g]) for g in H15] for R in seed_rows])
    V = np.array([[vmr(R[g]) for g in H15] for R in seed_rows])
    VE = np.array([[vme(R[g]) for g in H15] for R in seed_rows])
    mA, mE, mV, mVE = A.mean(0), E.mean(0), V.mean(0), VE.mean(0)
    per_seed = [float(np.median(a)) for a in A]
    return {"median_seedmean_ratio": float(np.median(mA)), "per_seed_median_ratio": per_seed,
            "per_seed_drop": [1 - m / rel_med for m in per_seed],
            "grids_better_than_released": int(sum(mE[i] < angerr(rel[g]) for i, g in enumerate(H15))),
            "per_seed_grids_better": [int(sum(E[s][i] < angerr(rel[g]) for i, g in enumerate(H15))) for s in range(len(A))],
            "V_beats_dc": int(sum(mV < 1)), "V_error_worse_than_released": int(sum(mVE[i] > vme(rel[g]) for i, g in enumerate(H15))),
            "median_V_error_pu": float(np.median(mVE)), "angle_beats_dc": int(sum(mA < 1))}


out = {"released": {"median_ratio": rel_med, "median_V_error_pu": float(np.median([vme(rel[g]) for g in H15])),
                    "V_beats_dc": int(sum(vmr(rel[g]) < 1 for g in H15))}}
selH = json.load(open(HERE / "selection-simbench.json"))
selM = json.load(open(HERE / "selection.json"))


def selected(sel, arm):
    sd = HERE / ("heldout-simbench" if sel is selH else "heldout")
    R = []
    for run in sorted(r for r in sel["runs"] if r.rsplit("_s", 1)[0] == arm):
        R.append(dict(rel) if sel["runs"][run]["selected"].endswith("_step0") else load(sd / run))
    return R


def at12(arm, d):
    return [load(HERE / d / f"{arm}_s{s}") for s in (0, 1, 2)]


for arm, sel in (("H", selH), ("D", selM), ("C00", selM), ("C01", selM)):
    out[f"{arm}_selected"] = summarize(selected(sel, arm))
    out[f"{arm}_selected"]["selected_ckpts"] = {r: v["selected"].rsplit("_", 1)[1] for r, v in sel["runs"].items() if r.rsplit("_s", 1)[0] == arm}
    out[f"{arm}_selected"]["seed_mean_val_metric"] = sel["arms"][arm]["seed_mean_val_metric"]
    out[f"{arm}_12k"] = summarize(at12(arm, "heldout-12k-simbench" if arm == "H" else "heldout-12k"))
out["released_val_metric"] = selM["released_val_metric"]
c = out["H_selected"]
out["H_criteria_selected"] = {"1_median<=2.77": c["median_seedmean_ratio"] <= 2.77, "2_breadth>=11": c["grids_better_than_released"] >= 11,
                              "3_each_seed_drop>=15%": all(d >= 0.15 for d in c["per_seed_drop"]), "4_V>=12": c["V_beats_dc"] >= 12}
c = out["H_12k"]
out["H_criteria_12k"] = {"1_median<=2.77": c["median_seedmean_ratio"] <= 2.77, "2_breadth>=11": c["grids_better_than_released"] >= 11,
                         "3_each_seed_drop>=15%": all(d >= 0.15 for d in c["per_seed_drop"]), "4_V>=12": c["V_beats_dc"] >= 12}
json.dump(out, open(HERE / "simbench_summary.json", "w"), indent=1)
L = [f"Released median ratio {rel_med:.3f}; released median |V| error {out['released']['median_V_error_pu']:.5f} pu.", "",
     "| arm / basis | median seed-mean ratio | per-seed medians | grids better than released (seed-mean; per seed) | beats DC on \\|V\\| | median \\|V\\| MAE (pu) | selected ckpts |",
     "|---|---|---|---|---|---|---|"]
for k in ("H_12k", "D_12k", "C00_12k", "C01_12k", "H_selected", "D_selected", "C00_selected", "C01_selected"):
    o = out[k]
    L.append(f"| {k} | {o['median_seedmean_ratio']:.3f} | " + " / ".join(f"{x:.2f}" for x in o["per_seed_median_ratio"])
             + f" | {o['grids_better_than_released']}; " + " / ".join(map(str, o["per_seed_grids_better"]))
             + f" | {o['V_beats_dc']} | {o['median_V_error_pu']:.5f} | "
             + (", ".join(v.replace("step", "") for v in o.get("selected_ckpts", {}).values()) if "selected_ckpts" in o else "12000") + " |")
L += ["", f"H criteria 1-4 on the validation-selected checkpoints: {out['H_criteria_selected']}",
      f"H criteria 1-4 at 12,000 updates (informational): {out['H_criteria_12k']}"]
(HERE / "simbench_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
