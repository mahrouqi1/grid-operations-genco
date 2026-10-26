"""Verdict for candidate 8 (GENCO vs classical |V| interpolators), from saved per-scenario files only; protocol.json is the spec. UNAUDITED.
    python verdict.py -> verdict.json, table.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
GR = P["grids"]; IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
ip = pd.read_json(HERE / "interp_per_scenario.jsonl", lines=True)
ip = ip[ip.method.isin(IM)]
ge = pd.read_json(HERE / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
d = pd.concat([ip, ge], ignore_index=True)
b7 = pd.read_json(HERE.parent / "2026-10-21-genco-state-inference" / "baselines_per_scenario.jsonl", lines=True)


TEST_IDS = json.loads((HERE.parent / '2026-10-21-genco-state-inference' / 'test_ids.json').read_text())


def mat(grid, cfgs, method):
    """(n_scenarios, n_cfgs) array in the order of sorted scenario ids shared by every method/config."""
    x = d[(d.grid == grid) & (d.method == method) & (d.config.isin(cfgs))].pivot(index="scenario", columns="config", values="vm_mae_unmeas")
    return x[cfgs]


def test(grid, cfgs):
    mats = {m: mat(grid, cfgs, m) for m in ["G"] + IM}
    sc = sorted(set.intersection(*[set(v.dropna().index) for v in mats.values()]))
    # audit R1-F4: every method x config must carry exactly the pre-registered test scenarios; otherwise stop (no silent shrinking)
    exp = sorted(TEST_IDS[grid]["test"])
    if sc != exp or any(v.shape != (len(exp), len(cfgs)) or v.isna().any().any() for v in mats.values()):
        sys.exit(f"INCOMPLETE: {grid} {cfgs}: {len(sc)} of {len(exp)} scenarios common to all methods/configs")
    A = {m: v.loc[sc].to_numpy() for m, v in mats.items()}
    def stat(idx):
        g = np.median(A["G"][idx]); bs = [np.median(A[m][idx]) for m in IM]
        return g / min(bs), IM[int(np.argmin(bs))], g, min(bs)
    full = np.arange(len(sc)); pt, best, g, bm = stat(full)
    bs = np.array([stat(rng.integers(0, len(sc), len(sc)))[0] for _ in range(NB)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n": len(sc), "ratio": pt, "lo": lo, "hi": hi, "best_interp": best, "genco_median": g, "best_interp_median": bm,
            "per_method_median": {m: float(np.median(A[m])) for m in IM}, "win": bool(pt < 1 and hi < 1), "win_point_only": bool(pt < 1)}


tests = {"primary_f20_pooled3": ["f20_s0", "f20_s1", "f20_s2"], "secondary_f10": ["f10_s0"], "secondary_f40": ["f40_s0"],
         "f20_seed0": ["f20_s0"], "f20_seed1": ["f20_s1"], "f20_seed2": ["f20_s2"]}
res = {t: {g: test(g, c) for g in GR} for t, c in tests.items()}
ref = {}
for g in GR:   # #7's best-of-B1..B3 assumed-dispatch baseline, 20% seed 0 only (the only configuration it was run at)
    ref[g] = float(min(b7[(b7.grid == g) & (b7.method == m)].vm_mae_unmeas.median() for m in ("B1_ac", "B2_ac_meas", "B3_dc")))
out = {"wins": {t: sum(r["win"] for r in v.values()) for t, v in res.items()},
       "wins_point_only": {t: sum(r["win_point_only"] for r in v.values()) for t, v in res.items()}, "threshold": P["threshold_grids"],
       "verdict_primary": None, "verdict_f10": None, "verdict_f40": None, "ref_best_B1B3_f20_seed0": ref, "results": res}
for t, k in (("primary_f20_pooled3", "verdict_primary"), ("secondary_f10", "verdict_f10"), ("secondary_f40", "verdict_f40")):
    out[k] = "PASS" if out["wins"][t] >= P["threshold_grids"] else "FAIL"
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
L = ["| grid | n | 10%: G / best-I (which) ratio [CI] | 20% pooled(3 seeds): G / best-I (which) ratio [CI] | 40%: G / best-I (which) ratio [CI] | #7 best B1-B3 @20% s0 |", "|" + "---|" * 6]
cell = lambda r: f"{r['genco_median']:.4f} / {r['best_interp_median']:.4f} ({r['best_interp']}) {r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{' W' if r['win'] else ''}"
for g in GR:
    L.append(f"| {g.replace('pglib_opf_','')} | {res['primary_f20_pooled3'][g]['n']} | {cell(res['secondary_f10'][g])} | {cell(res['primary_f20_pooled3'][g])} | {cell(res['secondary_f40'][g])} | {ref[g]:.4f} |")
(HERE / "table.md").write_text("\n".join(L) + "\n")
print("PRIMARY", out["verdict_primary"], out["wins"], "point-only", out["wins_point_only"])
