"""Analysis of the 2026-10-26 mask-fraction sweep: per-grid ratio / CI / win flag per fraction (DESCRIPTIVE ONLY, no pass/fail), plus the control
reproduction of f10_s0, f20_s0, f40_s0 against the stored 2026-10-21 per-scenario files. UNAUDITED.  Spec: protocol.json.
The test() body below is copied verbatim from 2026-10-21-genco-voltage-inference*/verdict.py (same stat, same 2000-resample bootstrap, same rule);
the generator is default_rng(7), consumed sequentially in the order stated in protocol.json.
    python sweep_verdict.py -> sweep.csv, sweep_summary.json, control.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_common as S

P = json.loads((S.V_DIR / "protocol.json").read_text())
IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
SETS = S.sets()
TOL_G, TOL_I = 5e-5, 1e-9


def load(sname, g):
    """-> long DataFrame (grid, config, method, scenario, vm_mae_unmeas) with methods G, I1..I5 from the sweep's raw files."""
    d = S.RAW / sname
    fi = sorted(d.glob(f"interp_{g}_*.jsonl"))
    fg = d / "genco.jsonl" if sname == "locked15" else d / f"genco_{g}.jsonl"
    fr = []
    for f in fi:
        x = pd.read_json(f, lines=True) if f.stat().st_size else None
        if x is not None and len(x): fr.append(x[x.method.isin(IM)])
    if fg.exists() and fg.stat().st_size:
        x = pd.read_json(fg, lines=True).drop(columns=["diag"]); fr.append(x[x.grid == g])
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=["grid", "config", "method", "scenario", "vm_mae_unmeas"])


def test_ids(sname, g):
    if sname == "locked15":
        return sorted(json.loads((S.SI_DIR / "test_ids.json").read_text())[g]["test"])
    return sorted(json.loads((SETS[sname][0] / "test_ids.json").read_text())[g]["test"])


def test(D, grid, cfg, exp):
    """verbatim stat/bootstrap/win logic of verdict.py; returns None (no RNG draw) if the cell is incomplete."""
    x = D[(D.grid == grid) & (D.config == cfg)]
    A = {}
    for m in ["G"] + IM:
        y = x[x.method == m].set_index("scenario").vm_mae_unmeas
        if sorted(y.index) != exp or y.isna().any() or len(y) != len(exp): return None
        A[m] = y.loc[exp].to_numpy()[:, None]
    def stat(idx):
        g = np.median(A["G"][idx]); bs = [np.median(A[m][idx]) for m in IM]
        return g / min(bs), IM[int(np.argmin(bs))], g, min(bs)
    n = len(exp); pt, best, g, bm = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n))[0] for _ in range(NB)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n": n, "ratio": float(pt), "ci_lo": lo, "ci_hi": hi, "best_interp": best, "genco_median": float(g), "best_interp_median": float(bm),
            "win": bool(pt < 1 and hi < 1), "win_point_only": bool(pt < 1)}


rows, frames = [], {}
order = [("locked15", "locked15", g) for g in SETS["locked15"][1]]
pieces_order = [("pieces36", s, g) for s in ("control", "fvs") for g in SETS[s][1]]
for p in S.PCTS:                                    # fractions ascending; locked15 first, then pieces (protocol.json order)
    for setname, src, g in order + pieces_order:
        key = (src, g)
        if key not in frames: frames[key] = load(src, g)
        cfg = S.cfg_name(p); exp = test_ids(src, g)
        r = test(frames[key], g, cfg, exp)
        row = {"set": setname, "source": src, "grid": g, "fraction": p / 100}
        row.update(r if r else {"status": "missing"})
        row.setdefault("status", "ok"); rows.append(row)
df = pd.DataFrame(rows)
cols = ["set", "source", "grid", "fraction", "ratio", "ci_lo", "ci_hi", "win", "win_point_only", "best_interp", "genco_median", "best_interp_median", "n", "status"]
df[cols].to_csv(HERE / "sweep.csv", index=False)

summ = {"note": "DESCRIPTIVE ONLY; UNAUDITED; mask seed 0 at every fraction; win = ratio < 1 and 95% CI upper end < 1", "fractions": S.PCTS}
for sname in ("locked15", "pieces36"):
    s = df[df.set == sname]; tot = len(SETS["locked15"][1]) if sname == "locked15" else sum(len(SETS[k][1]) for k in ("control", "fvs"))
    summ[sname] = {"n_grids": tot, "wins": {}, "wins_point_only": {}, "n_complete": {}, "median_ratio": {}}
    for p in S.PCTS:
        c = s[(s.fraction == p / 100) & (s.status == "ok")]
        summ[sname]["wins"][str(p)] = int(c.win.sum()); summ[sname]["wins_point_only"][str(p)] = int(c.win_point_only.sum())
        summ[sname]["n_complete"][str(p)] = int(len(c)); summ[sname]["median_ratio"][str(p)] = float(c.ratio.median()) if len(c) else None
    summ[sname]["ran_any"] = bool((s.status == "ok").any())
(HERE / "sweep_summary.json").write_text(json.dumps(summ, indent=1))

# ---- control reproduction: f10_s0, f20_s0, f40_s0 vs stored 2026-10-21 per-scenario files
ctl = {"tolerance_genco": TOL_G, "tolerance_interp": TOL_I, "sets": {}}
def stored(sname, g):
    if sname == "locked15":
        a = pd.read_json(S.V_DIR / "interp_per_scenario.jsonl", lines=True); b = pd.read_json(S.V_DIR / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
        a = a[a.method.isin(IM)]; return pd.concat([a, b]).query("grid == @g")
    d = SETS[sname][0] / "per_grid"
    return pd.concat([pd.read_json(d / f"interp_{g}.jsonl", lines=True).query("method in @IM"), pd.read_json(d / f"genco_{g}.jsonl", lines=True).drop(columns=["diag"])])
STORED = {}
for sname in ("locked15", "control", "fvs"):
    res = {}
    for p in (10, 20, 40):
        cfg = S.cfg_name(p)
        for m in ["G"] + IM:
            mx, nover, ncell, missing = 0.0, 0, 0, 0
            for g in SETS[sname][1]:
                new = frames.get(("locked15" if sname == "locked15" else sname, g))
                if new is None: new = frames[(sname, g)] = load(sname, g)
                if (sname, g) not in STORED: STORED[(sname, g)] = stored(sname, g)
                o = STORED[(sname, g)]
                a = new[(new.config == cfg) & (new.method == m)].set_index("scenario").vm_mae_unmeas
                b = o[(o.config == cfg) & (o.method == m)].set_index("scenario").vm_mae_unmeas
                if len(a) == 0: missing += 1; continue
                if sorted(a.index) != sorted(b.index): missing += 1; continue
                dd = float((a.loc[b.index] - b).abs().max()); mx = max(mx, dd); ncell += 1
                nover += int(dd > (TOL_G if m == "G" else TOL_I))
            res[f"{cfg}/{m}"] = {"max_abs_diff": mx, "grids_compared": ncell, "grids_over_tolerance": nover, "grids_missing_or_mismatched": missing}
    ctl["sets"][sname] = res
# win flags and ratios vs the recorded locked15 verdict (seed-0 analyses: secondary_f10, f20_seed0, secondary_f40)
rec = json.loads((S.V_DIR / "verdict.json").read_text())["results"]
flags = {}
for p, k in ((10, "secondary_f10"), (20, "f20_seed0"), (40, "secondary_f40")):
    c = df[(df.set == "locked15") & (df.fraction == p / 100) & (df.status == "ok")]
    same = sum(bool(r.win) == rec[k][r.grid]["win"] for r in c.itertuples()); dr = max([abs(r.ratio - rec[k][r.grid]["ratio"]) for r in c.itertuples()] or [float("nan")])
    flags[str(p)] = {"recorded_key": k, "grids_compared": int(len(c)), "win_flag_matches": int(same), "max_abs_ratio_diff": float(dr)}
ctl["locked15_win_flags_vs_recorded"] = flags
ctl["pieces36_note"] = "recorded piece verdicts are pooled over 3 mask seeds, so win flags are not compared for pieces; per-scenario errors are compared instead."
bad = [(s, k) for s, r in ctl["sets"].items() for k, v in r.items() if v["grids_over_tolerance"] or (v["grids_compared"] == 0 and False)]
ctl["all_within_tolerance_for_compared_cells"] = not bad
ctl["cells_over_tolerance"] = bad
(HERE / "control.json").write_text(json.dumps(ctl, indent=1))
print(json.dumps({k: summ[k]["wins"] for k in ("locked15", "pieces36")}))
print("control within tol:", ctl["all_within_tolerance_for_compared_cells"], "over:", bad[:6])
