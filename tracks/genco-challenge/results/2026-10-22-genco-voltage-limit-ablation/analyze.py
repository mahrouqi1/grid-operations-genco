"""Verdict for the limit/cost ablation, from saved per-scenario files only; protocol.json is the spec. UNAUDITED.  python analyze.py -> per_case_*.csv, verdict.json, DONE_<set> markers."""
import json, zlib, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; RES = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
IM = ["I1", "I2", "I3", "I4", "I5"]; ARMS = ["A0", "A1", "A2", "A3"]; CFG = ["f40_s0", "f40_s1", "f40_s2"]; NB = 2000
R = "2026-10-21-genco-voltage-inference"
STUDY = {"control": RES / f"{R}-control", "fvs": RES / f"{R}-family-vs-size", "oos": RES / f"{R}-oos"}
SETS = {"pieces": None, "seen_whole": "oos", "unseen_whole": "oos"}
piece_study = {g: "control" for g in json.loads((STUDY["control"] / "test_ids.json").read_text())}
piece_study.update({g: "fvs" for g in json.loads((STUDY["fvs"] / "test_ids.json").read_text())})
orig_win = {}
for k, d in STUDY.items(): orig_win[k] = json.loads((d / "verdict.json").read_text())["results"]["f40"]
nbus_oos = json.loads((STUDY["control"] / "oos_nbus.json").read_text())
nbus_piece = {}
for k in ("control", "fvs"): nbus_piece.update({g: v["n_bus"] for g, v in json.loads((STUDY[k] / "protocol.json").read_text())["subgrids"].items()})


def load(g, study):
    d = STUDY[study]; ids = json.loads((d / "test_ids.json").read_text())[g]["test"]
    ip = pd.read_json(d / "per_grid" / f"interp_{g}.jsonl", lines=True); gs = pd.read_json(d / "per_grid" / f"genco_{g}.jsonl", lines=True)
    ab = HERE / "per_grid" / f"genco_ablate_{g}.jsonl"
    if not ab.exists(): return None, "not started"
    ab = pd.read_json(ab, lines=True)
    M = {}
    for m in IM:
        x = ip[ip.method == m]; M[m] = x.pivot(index="scenario", columns="config", values="vm_mae_unmeas")[CFG].loc[ids].to_numpy()
    x = gs[gs.method == "G"]; M["A0"] = x.pivot(index="scenario", columns="config", values="vm_mae_unmeas")[CFG].loc[ids].to_numpy()
    for arm in ARMS[1:]:
        x = ab[ab.arm == arm]
        if x.groupby("config").scenario.nunique().reindex(CFG).fillna(0).min() < len(ids): return None, f"incomplete ({arm})"
        M[arm] = x.pivot(index="scenario", columns="config", values="vm_mae_unmeas")[CFG].loc[ids].to_numpy()
    if any(not np.isfinite(v).all() for v in M.values()): return None, "non-finite values"
    return M, None


def test(M, g):
    n = M["A0"].shape[0]; rng = np.random.default_rng([7, zlib.crc32(g.encode())]); idx = rng.integers(0, n, (NB, n))
    best_med = lambda sel: min(np.median(M[m][sel]) for m in IM)
    out = {}
    bm_pt = [np.median(M[m]) for m in IM]
    bmb = np.array([[np.median(M[m][i]) for m in IM] for i in idx]).min(axis=1)   # best interpolator inside each resample
    for arm in ARMS:
        pt = np.median(M[arm]) / min(bm_pt)
        bs = np.array([np.median(M[arm][i]) for i in idx]) / bmb
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[arm] = {"n": n, "genco_median": float(np.median(M[arm])), "best_interp": IM[int(np.argmin(bm_pt))], "best_interp_median": float(min(bm_pt)), "ratio": float(pt), "lo": lo, "hi": hi, "win": bool(pt < 1 and hi < 1)}
    return out


res = {s: {} for s in SETS}; skipped = {s: {} for s in SETS}; rows = {s: [] for s in SETS}
for s in SETS:
    for g in P["sets"][s]["grids"]:
        study = piece_study[g] if s == "pieces" else "oos"
        M, why = load(g, study)
        if M is None: skipped[s][g] = why; continue
        r = test(M, g); res[s][g] = r
        for arm in ARMS:
            x = r[arm]; rows[s].append({"set": s, "grid": g, "arm": arm, "n_scen": x["n"], "buses": (nbus_piece[g] if s == "pieces" else nbus_oos[g]), "genco_median": x["genco_median"], "best_interp": x["best_interp"],
                                         "best_interp_median": x["best_interp_median"], "ratio": x["ratio"], "ci_lo": x["lo"], "ci_hi": x["hi"], "win": int(x["win"]), "original_study_win": int(orig_win[study][g]["win"]) if arm == "A0" else ""})
    pd.DataFrame(rows[s]).to_csv(HERE / f"per_case_{s}.csv", index=False)


def summ(gs_by_set):
    gs = [(s, g) for s, l in gs_by_set.items() for g in l]
    o = {"grids_ran": len(gs)}
    for arm in ARMS:
        o[arm] = {"wins": sum(res[s][g][arm]["win"] for s, g in gs)}
    for arm in ARMS[1:]:
        lost = sum(1 for s, g in gs if res[s][g]["A0"]["win"] and not res[s][g][arm]["win"]); gained = sum(1 for s, g in gs if not res[s][g]["A0"]["win"] and res[s][g][arm]["win"])
        o[arm].update({"delta_wins_A0_minus_arm": o["A0"]["wins"] - o[arm]["wins"], "win_to_loss": lost, "loss_to_win": gained})
        rat = [res[s][g][arm]["ratio"] / res[s][g]["A0"]["ratio"] for s, g in gs]; dif = [res[s][g][arm]["ratio"] - res[s][g]["A0"]["ratio"] for s, g in gs]
        o[arm].update({"median_ratio_shift_factor": float(np.median(rat)) if rat else None, "median_ratio_shift_diff": float(np.median(dif)) if dif else None,
                       "median_ratio": float(np.median([res[s][g][arm]["ratio"] for s, g in gs])) if gs else None})
    o["A0"]["median_ratio"] = float(np.median([res[s][g]["A0"]["ratio"] for s, g in gs])) if gs else None
    o["A0"]["original_study_wins_on_these_grids"] = sum(int(orig_win[piece_study[g] if s == "pieces" else "oos"][g]["win"]) for s, g in gs)
    return o


ran = {s: list(res[s]) for s in SETS}
out = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "set_sizes": {s: len(P["sets"][s]["grids"]) for s in SETS}, "ran": ran, "skipped": skipped,
       "by_set": {s: summ({s: ran[s]}) for s in SETS}, "whole_pooled": summ({"seen_whole": ran["seen_whole"], "unseen_whole": ran["unseen_whole"]})}
thr = {}
for arm in ARMS[1:]:
    dp = out["by_set"]["pieces"][arm]["delta_wins_A0_minus_arm"]; dw = out["whole_pooled"][arm]["delta_wins_A0_minus_arm"]
    thr[arm] = {"delta_pieces": dp, "delta_whole_pooled": dw, "crossed": bool(dp >= 3 or dw >= 3)}
inconcl = {"pieces": len(ran["pieces"]) < 18, "whole_pooled": len(ran["seen_whole"]) + len(ran["unseen_whole"]) < 16}
out["threshold"] = {"per_arm": thr, "THRESHOLD_CROSSED": any(v["crossed"] for v in thr.values()), "inconclusive_less_than_half_ran": inconcl,
                    "complete": {s: len(ran[s]) == len(P["sets"][s]["grids"]) for s in SETS}}
(HERE / "verdict.json").write_text(json.dumps({**out, "results": res}, indent=1))
for s in SETS:
    if len(ran[s]) == len(P["sets"][s]["grids"]): (HERE / f"DONE_{s}").write_text(out["generated"] + "\n")
print(json.dumps({"ran": {s: len(v) for s, v in ran.items()}, "by_set": {s: {a: out["by_set"][s][a].get("wins") for a in ARMS} for s in SETS}, "whole": {a: out["whole_pooled"][a]["wins"] for a in ARMS}, "threshold": out["threshold"]}, indent=1))
