"""Pool the 3 seeds (200, 201, 202) x 2 arms of the GENCO continued-pretraining runs. UNAUDITED.
Same quantities as ../pool.py (collapse reproduction, final state, 7 training subgrids, weighted by hidden
collapsed buses, mask seeds 1000-1002) and ../preservation.py (median over 15 held-out grids of arm error /
released error, branch angle and |V|). The formulas are re-implemented here (pool.py / preservation.py write
files at import time). Step 1 checks the recomputed seed-200 numbers against the published pooled.json and
preservation.json; the script aborts if they differ. Writes seeds/pooled_seeds.json."""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
R = S.parent
TRAIN7 = {"subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case5658_epigrids_idx_8",
          "subnet_case4837_goc_idx_28", "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37",
          "subnet_case10480_goc_idx_26"}
MASKS = (1000, 1001, 1002)
ARMS = ("fixed", "published")
SEEDS = (200, 201, 202)


def pmask_file(arm, seed):
    return R / "pmask" / f"{arm}_final.jsonl" if seed == 200 else S / "pmask" / f"{arm}_s{seed}_final.jsonl"


def zs_file(seed):
    return R / "zeroshot-final" / "results.jsonl" if seed == 200 else S / f"zeroshot-final-s{seed}" / "results.jsonl"


def collapse(path):
    G = [json.loads(l) for l in open(path) if l.strip()]
    G = [g for g in G if g["grid"] in TRAIN7]
    assert len(G) == 7, (path, len(G))
    out = {}
    for m in MASKS:
        runs = [r for g in G for r in g["runs"] if r["mask"] == "pre" and r["seed"] == m]
        w = np.array([r["n_low_hidden"] for r in runs], float)
        out[m] = float(np.average([r["frac_pred_lt_0.5"] for r in runs], weights=w))
    return out


def preservation(seed, arm):
    rows = [json.loads(l) for l in open(zs_file(seed)) if l.strip()]
    by = {(r["grid"], r["model"]): r for r in rows if not r.get("error")}
    gs = sorted({g for g, m in by if (g, "retrain_" + arm) in by and (g, "zeroshot") in by})
    a = [by[(g, "retrain_" + arm)]["errors"]["genco"]["branch_dva_mae_deg"] / by[(g, "zeroshot")]["errors"]["genco"]["branch_dva_mae_deg"] for g in gs]
    v = [by[(g, "retrain_" + arm)]["errors"]["genco_pinned"]["vm_mae_pu"] / by[(g, "zeroshot")]["errors"]["genco_pinned"]["vm_mae_pu"] for g in gs]
    return {"n": len(gs), "angle": float(np.median(a)), "V": float(np.median(v)), "angle_better": int(sum(x < 1 for x in a)),
            "V_better": int(sum(x < 1 for x in v))}


def released_row_signature(seed):
    rows = [json.loads(l) for l in open(zs_file(seed)) if l.strip()]
    return {r["grid"]: (r["errors"]["genco"]["branch_dva_mae_deg"], r["errors"]["genco_pinned"]["vm_mae_pu"])
            for r in rows if r["model"] == "zeroshot" and not r.get("error")}


# ---- step 1: seed-200 recomputation equals the published numbers
P = json.load(open(R / "pooled.json"))
PR = json.load(open(R / "preservation.json"))
for arm in ARMS:
    pub = next(p for p in P if p["arm"] == arm and p["state"] == "final")
    mine = collapse(pmask_file(arm, 200))
    for m in MASKS:
        assert abs(mine[m] - pub[str(m)]["frac_lt_0.5"]) < 1e-12, ("collapse mismatch", arm, m)
    pr = preservation(200, arm)
    ref = PR[f"final/retrain_{arm}"]
    assert abs(pr["angle"] - ref["median_angle_ratio_to_released"]) < 1e-12, ("angle mismatch", arm)
    assert abs(pr["V"] - ref["median_V_ratio_to_released"]) < 1e-12, ("V mismatch", arm)
    assert pr["angle_better"] == ref["angle_better_than_released"], ("angle_better mismatch", arm)
print("CHECK PASS: recomputed seed-200 numbers equal pooled.json / preservation.json")
sig = {s: released_row_signature(s) for s in SEEDS}
same = all(sig[s] == sig[200] for s in SEEDS)
print("released ('zeroshot') rows identical across the 3 score runs:", same)

# ---- step 2: all seeds and arms
res = {}
for arm in ARMS:
    for s in SEEDS:
        c = collapse(pmask_file(arm, s))
        res[(arm, s)] = {"collapse": c, "collapse_mean": float(np.mean(list(c.values()))), **preservation(s, arm)}

def spread(arm, f):
    x = [f(res[(arm, s)]) for s in SEEDS]
    return {"median": float(np.median(x)), "min": float(min(x)), "max": float(max(x)), "values": x}

summary = {}
for arm in ARMS:
    summary[arm] = {
        "collapse_mask1000": spread(arm, lambda r: r["collapse"][1000]),
        "collapse_mean_of_masks": spread(arm, lambda r: r["collapse_mean"]),
        "collapse_min_over_masks": spread(arm, lambda r: min(r["collapse"].values())),
        "collapse_max_over_masks": spread(arm, lambda r: max(r["collapse"].values())),
        "angle_ratio": spread(arm, lambda r: r["angle"]),
        "V_ratio": spread(arm, lambda r: r["V"]),
        "angle_better_count": spread(arm, lambda r: r["angle_better"]),
        "V_better_count": spread(arm, lambda r: r["V_better"])}

# per-seed corrected-vs-original comparisons
paired = {s: {"collapse_mean_diff_pub_minus_fix": res[("published", s)]["collapse_mean"] - res[("fixed", s)]["collapse_mean"],
              "angle_diff": res[("published", s)]["angle"] - res[("fixed", s)]["angle"],
              "V_diff": res[("published", s)]["V"] - res[("fixed", s)]["V"]} for s in SEEDS}

out = {"released_rows_identical": same,
       "per_seed": {f"{arm}_s{s}": res[(arm, s)] for arm in ARMS for s in SEEDS},
       "summary": summary, "paired": paired}
json.dump(out, open(S / "pooled_seeds.json", "w"), indent=1)

print("\nper seed/arm: collapse (mask1000/1001/1002) | angle ratio | V ratio | angle<1 of 15")
for arm in ARMS:
    for s in SEEDS:
        r = res[(arm, s)]
        print(f"{arm:9s} s{s}: " + "/".join(f"{r['collapse'][m]:.3f}" for m in MASKS) +
              f" | {r['angle']:.3f} | {r['V']:.3f} | {r['angle_better']}")
print("\n3-seed median [min-max]")
for arm in ARMS:
    for k, v in summary[arm].items():
        print(f"{arm:9s} {k:24s} {v['median']:.3f} [{v['min']:.3f}-{v['max']:.3f}]")
print("\npaired (published - fixed) per seed:", json.dumps(paired, indent=1))
