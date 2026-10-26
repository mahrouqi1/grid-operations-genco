"""Verdict for candidate 10 (carved seen-family subgrids control), from saved per-scenario files only; protocol.json is the spec. UNAUDITED.
    python verdict.py -> verdict.json, table.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
OOS = HERE.parent / "2026-10-21-genco-voltage-inference-oos"
IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
ids = json.loads((HERE / "test_ids.json").read_text()); meta = P["subgrids"]
POOL = {"f10": ["f10_s0", "f10_s1", "f10_s2"], "f20": ["f20_s0", "f20_s1", "f20_s2"], "f40": ["f40_s0", "f40_s1", "f40_s2"],
        "n20": ["n20_s0", "n20_s1", "n20_s2"], "n40": ["n40_s0", "n40_s1", "n40_s2"]}
frames, ran, skipped = [], [], {}
for g in meta:
    fi, fg = HERE / "per_grid" / f"interp_{g}.jsonl", HERE / "per_grid" / f"genco_{g}.jsonl"
    if g not in ids or not (fi.exists() and fg.exists()):
        skipped[g] = "no output"; continue
    d = pd.concat([pd.read_json(fi, lines=True), pd.read_json(fg, lines=True).drop(columns=["diag"])], ignore_index=True); need = set(ids[g]["test"])
    if not all(set(d[(d.method == m) & (d.config == c)].scenario) >= need and d[(d.method == m) & (d.config == c)].vm_mae_unmeas.notna().all()
               for m in ["G"] + IM for cs in POOL.values() for c in cs):
        skipped[g] = "incomplete"; continue
    frames.append(d); ran.append(g)
# audit R1-F4: no skipped subgrids; the pre-registered list (protocol 'subgrids') must all have run
if skipped or sorted(ran) != sorted(meta):
    sys.exit(f"INCOMPLETE: skipped={skipped} ran={len(ran)} pre-registered={len(meta)}")
D = pd.concat(frames, ignore_index=True)


def test(grid, cfgs):
    x = D[(D.grid == grid) & D.config.isin(cfgs)]
    A = {m: x[x.method == m].pivot(index="scenario", columns="config", values="vm_mae_unmeas")[cfgs].loc[ids[grid]["test"]].to_numpy() for m in ["G"] + IM}
    def stat(idx):
        g = np.median(A["G"][idx]); bs = [np.median(A[m][idx]) for m in IM]
        return g / min(bs), IM[int(np.argmin(bs))], g, min(bs)
    exp = len(ids[grid]["test"])   # audit R1-F4: n must equal the pre-registered test-scenario count for every method x config
    if any(a.shape != (exp, len(cfgs)) or not np.isfinite(a).all() for a in A.values()): sys.exit(f"INCOMPLETE: {grid} {cfgs}")
    n = A["G"].shape[0]; pt, best, g, bm = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n))[0] for _ in range(NB)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n": n, "ratio": pt, "lo": lo, "hi": hi, "best_interp": best, "genco_median": g, "best_interp_median": bm, "win": bool(pt < 1 and hi < 1), "win_point_only": bool(pt < 1)}


res = {c: {g: test(g, cf) for g in ran} for c, cf in POOL.items()}
thr = P["threshold_fraction"]
def summ(cond, gs, pre=None):
    pre = list(meta) if pre is None else pre   # audit R1-F4: threshold from the PRE-REGISTERED grid list (protocol subgrids), not from the grids that ran
    assert sorted(gs) == sorted(pre), "ran grids differ from the pre-registered list"
    w = sum(res[cond][g]["win"] for g in gs); need = int(np.ceil(thr * len(pre) - 1e-9))
    return {"grids": len(gs), "wins": w, "needed": need, "wins_point_only": sum(res[cond][g]["win_point_only"] for g in gs), "verdict": "PASS" if gs and w >= need else "FAIL"}
tier = {g: meta[g]["tier"] for g in ran}; parent = {g: meta[g]["parent"] for g in ran}
out = {"ran": ran, "skipped": skipped, "primary": {"cond": "f40", **summ("f40", ran)},
       "secondary_f20": summ("f20", ran), "secondary_n40": summ("n40", ran), "descriptive": {c: summ(c, ran) for c in ("f10", "n20")},
       "by_tier": {t: {c: summ(c, [g for g in ran if tier[g] == t], [g for g in meta if meta[g]["tier"] == t]) for c in ("f40", "f20", "n40")} for t in ("a", "b")},
       "by_parent_f40": {p: [sum(res["f40"][g]["win"] for g in ran if parent[g] == p), sum(1 for g in ran if parent[g] == p)] for p in sorted(set(parent.values()))}}
# size-banded comparison with the unseen-family (OOS) results
ov = json.loads((OOS / "verdict.json").read_text()); osz = json.loads((HERE / "oos_nbus.json").read_text())
unseen = [g for g in P["unseen_primary"] if g in ov["results"]["f40"]]
bands = {}
for cond in ("f40", "f20", "n40"):
    b = {}
    for name, sel in (("<=1000", lambda n: n <= 1000), (">1000", lambda n: n > 1000)):
        u = [g for g in unseen if sel(osz[g])]; uw = sum(ov["results"][cond][g]["win"] for g in u)
        cg = ran if name == "<=1000" else []
        cw = sum(res[cond][g]["win"] for g in cg)
        e = {"unseen_grids": len(u), "unseen_wins": uw, "carved_grids": len(cg), "carved_wins": cw}
        if u and cg:
            e["fisher_p_two_sided"] = float(fisher_exact([[cw, len(cg) - cw], [uw, len(u) - uw]])[1])
            e["unseen_median_ratio"] = float(np.median([ov["results"][cond][g]["ratio"] for g in u])); e["carved_median_ratio"] = float(np.median([res[cond][g]["ratio"] for g in cg]))
        b[name] = e
    bands[cond] = b
out["size_bands_vs_unseen"] = bands; out["carved_sizes"] = {g: meta[g]["n_bus"] for g in ran}
out["results"] = res
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
cell = lambda r: f"{r['genco_median']:.4f}/{r['best_interp_median']:.4f} ({r['best_interp']}) {r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{' W' if r['win'] else ''}"
L = ["| parent | subgrid | tier | buses | n scen | 10% | 20% | 40% | 20% + noise | 40% + noise |", "|" + "---|" * 10]
for g in ran:
    L.append(f"| {parent[g]} | {g.replace('carved_','')} | {tier[g]} | {meta[g]['n_bus']} | {res['f40'][g]['n']} | " + " | ".join(cell(res[c][g]) for c in ("f10", "f20", "f40", "n20", "n40")) + " |")
L += ["", "Size-banded win counts (GENCO beats best interpolator, CI rule): carved (all <= 1000 buses) vs unseen families", "", "| condition | band | unseen wins/grids | carved wins/grids | Fisher p |", "|---|---|---|---|---|"]
for cond, b in bands.items():
    for name, e in b.items():
        L.append(f"| {cond} | {name} | {e['unseen_wins']}/{e['unseen_grids']} | {e['carved_wins']}/{e['carved_grids']} | {e.get('fisher_p_two_sided', float('nan')):.3f} |")
(HERE / "table.md").write_text("\n".join(L) + "\n")
p = out["primary"]; print("PRIMARY", p["verdict"], p["wins"], "of", p["grids"], "need", p["needed"], "| f20", out["secondary_f20"]["wins"], "| n40", out["secondary_n40"]["wins"], "| skipped", skipped)
