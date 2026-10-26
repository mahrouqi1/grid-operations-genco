"""Verdict for candidate 9 (out-of-sample |V| confirmation), from the saved per-scenario files only; protocol.json is the spec. UNAUDITED.
    python verdict.py -> verdict.json, table.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
ids = json.loads((HERE / "test_ids.json").read_text())
fam = {g: f for f, gs in P["families"].items() for g in gs}
POOL = {"f10": ["f10_s0", "f10_s1", "f10_s2"], "f20": ["f20_s0", "f20_s1", "f20_s2"], "f40": ["f40_s0", "f40_s1", "f40_s2"],
        "n20": ["n20_s0", "n20_s1", "n20_s2"], "n40": ["n40_s0", "n40_s1", "n40_s2"]}
frames, ran, skipped = [], [], {}
for g in fam:
    fi, fg = HERE / "per_grid" / f"interp_{g}.jsonl", HERE / "per_grid" / f"genco_{g}.jsonl"
    if not (fi.exists() and fg.exists()):
        skipped[g] = "no output files"; continue
    a = pd.read_json(fi, lines=True); b = pd.read_json(fg, lines=True).drop(columns=["diag"])
    d = pd.concat([a, b], ignore_index=True); need = set(ids[g]["test"])
    ok = all(set(d[(d.method == m) & (d.config == c)].scenario) >= need and d[(d.method == m) & (d.config == c)].vm_mae_unmeas.notna().all()
             for m in ["G"] + IM for cs in POOL.values() for c in cs)
    if not ok:
        skipped[g] = "incomplete (a method/config lacks scenarios or has NaN)"; continue
    frames.append(d); ran.append(g)
# audit R1-F4: no skipped grids; the pre-registered grid list (protocol sets) must all have run
PRE = [g for s in P["sets"].values() for g in s]
if skipped or sorted(ran) != sorted(PRE):
    sys.exit(f"INCOMPLETE: skipped={skipped} ran={len(ran)} pre-registered={len(PRE)}")
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
    return {"n": n, "ratio": pt, "lo": lo, "hi": hi, "best_interp": best, "genco_median": g, "best_interp_median": bm,
            "per_method_median": {m: float(np.median(A[m])) for m in IM}, "win": bool(pt < 1 and hi < 1), "win_point_only": bool(pt < 1)}


res = {c: {g: test(g, cf) for g in ran} for c, cf in POOL.items()}
def summ(cond, group):
    gs = [g for g in ran if g in set(P["sets"][group])]
    w = sum(res[cond][g]["win"] for g in gs); need = int(np.ceil(P["threshold_fraction"] * len(P["sets"][group]) - 1e-9))   # from the PRE-REGISTERED list
    per_fam = {f: [sum(res[cond][g]["win"] for g in gs if fam[g] == f), sum(1 for g in gs if fam[g] == f)] for f in sorted({fam[g] for g in gs})}
    return {"grids_ran": len(gs), "wins": w, "wins_point_only": sum(res[cond][g]["win_point_only"] for g in gs), "needed": need,
            "verdict": "PASS" if gs and w >= need else "FAIL", "per_family_wins_total": per_fam}
out = {"ran": ran, "skipped": skipped,
       "primary": {"cond": "f40", **summ("f40", "primary")},
       "secondary_i_noise40": summ("n40", "primary"), "secondary_ii_f20": summ("f20", "primary"),
       "descriptive_primary_set": {c: summ(c, "primary") for c in ("f10", "n20")},
       "seen_family_descriptive": {c: summ(c, "seen") for c in POOL}, "results": res}
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
cell = lambda r: f"{r['genco_median']:.4f}/{r['best_interp_median']:.4f} ({r['best_interp']}) {r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{' W' if r['win'] else ''}"
L = []
for title, group in (("PRIMARY (families unseen in pretraining)", "primary"), ("SEEN family (GOC / EPIGRIDS), descriptive only", "seen")):
    L += [f"### {title}", "", "| family | grid | n scen | 10% | 20% | 40% | 20% + noise | 40% + noise |", "|" + "---|" * 8]
    for g in ran:
        if g in set(P["sets"][group]):
            L.append(f"| {fam[g]} | {g.replace('pglib_opf_','')} | {res['f40'][g]['n']} | " + " | ".join(cell(res[c][g]) for c in ("f10", "f20", "f40", "n20", "n40")) + " |")
    L.append("")
(HERE / "table.md").write_text("\n".join(L))
p = out["primary"]
print("PRIMARY", p["verdict"], p["wins"], "of", p["grids_ran"], "need", p["needed"], "| noise40", out["secondary_i_noise40"]["wins"], out["secondary_i_noise40"]["verdict"], "| f20", out["secondary_ii_f20"]["wins"], out["secondary_ii_f20"]["verdict"], "| skipped", skipped)
