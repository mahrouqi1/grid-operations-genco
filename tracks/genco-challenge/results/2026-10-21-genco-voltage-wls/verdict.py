"""Verdict: does WLS change the GENCO-vs-best-classical counts? From saved per-scenario files only; protocol.json is the spec. UNAUDITED.
python verdict.py -> verdict.json, cells.csv, all_cases.csv, table.md"""
import json, sys, zlib, csv
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; R = HERE.parent
D = lambda s: R / ("2026-10-21-genco-voltage-inference" + s)
IM = ["I1", "I2", "I3", "I4", "I5"]; NB = 2000
CSEED = {"f20": ["f20_s0", "f20_s1", "f20_s2"], "f40": ["f40_s0", "f40_s1", "f40_s2"], "n40": ["n40_s0", "n40_s1", "n40_s2"]}
oos_p = json.loads((D("-oos") / "protocol.json").read_text()); gc_p = json.loads((D("-generator-control") / "protocol.json").read_text())
gc_r = json.loads((D("-generator-control") / "protocol_ran.json").read_text())["ran"]
H15 = json.loads((D("") / "protocol.json").read_text())["grids"]
STUDY = {  # study -> dir suffix, conds, groups {name: [grids]}, scenario limit, orig verdict
    "locked15": dict(dir="", conds={"f20": CSEED["f20"], "f40": ["f40_s0"]}, groups={"locked15": H15}, limit=30),
    "oos": dict(dir="-oos", conds=CSEED, groups={"oos_unseen": oos_p["sets"]["primary"], "oos_seen": oos_p["sets"]["seen"]}, limit=10),
    "control": dict(dir="-control", conds=CSEED, groups={"carved_seen_datakit": None}, limit=30),
    "fvs": dict(dir="-family-vs-size", conds=CSEED, groups={"carved_unseen_datakit": None}, limit=30),
    "genctl": dict(dir="-generator-control", conds={k: CSEED[k] for k in ("f20", "f40")},
                   groups={"carved_seen_ladder": [g for g in gc_r if gc_p["pieces"][g]["group"] == "seen"],
                           "carved_unseen_ladder": [g for g in gc_r if gc_p["pieces"][g]["group"] == "unseen"]}, limit=30),
}


def load(st, g, sp):
    ids = json.loads(((D("-oos") if False else (R / "2026-10-21-genco-state-inference" if st == "locked15" else D(sp["dir"]))) / "test_ids.json").read_text())[g]["test"]
    if sp["limit"]: ids = ids[: sp["limit"]]
    if st == "locked15":
        ip = pd.read_json(D("") / "interp_per_scenario.jsonl", lines=True); ge = pd.read_json(D("") / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
        d = pd.concat([ip[ip.grid == g], ge[ge.grid == g]], ignore_index=True)
    else:
        pg = D(sp["dir"]) / "per_grid"
        d = pd.concat([pd.read_json(pg / f"interp_{g}.jsonl", lines=True), pd.read_json(pg / f"genco_{g}.jsonl", lines=True).drop(columns=["diag"])], ignore_index=True)
    w = pd.read_csv(HERE / "per_case" / st / f"{g}.csv")
    return ids, d, w


def mats(ids, d, w, cfgs):
    A = {}
    for m in ["G"] + IM:
        x = d[(d.method == m) & d.config.isin(cfgs)].pivot(index="scenario", columns="config", values="vm_mae_unmeas")[cfgs]
        A[m] = x.reindex(ids).to_numpy()
    x = w[w.config.isin(cfgs)]
    A["W"] = x.pivot(index="scenario", columns="config", values="wls_mae")[cfgs].reindex(ids).to_numpy()
    ok = x.assign(c=(x.status == "converged")).pivot(index="scenario", columns="config", values="c")[cfgs].reindex(ids).to_numpy()
    obs = x.pivot(index="scenario", columns="config", values="count_observable")[cfgs].reindex(ids).to_numpy()
    A["W_fb"] = np.where(ok.astype(bool), A["W"], A["I2"])
    return A, ok.astype(bool), obs


def cell(A, seed):
    n = A["G"].shape[0]; rng = np.random.default_rng(seed); idx = rng.integers(0, n, (NB, n))
    med = lambda m, ix: np.median(A[m][ix].reshape(len(ix), -1), axis=1) if ix.ndim == 2 else np.median(A[m][ix])
    full = np.arange(n); res = {}
    arms = {"5": IM, "6": IM + ["W"], "6fb": IM + ["W_fb"], "W_only": ["W"]}
    for k, ms in arms.items():
        pt_m = {m: np.median(A[m]) for m in ms}; best = min(pt_m, key=pt_m.get)
        bsd = np.min(np.stack([med(m, idx) for m in ms]), axis=0); ratio = np.median(A["G"][idx].reshape(NB, -1), axis=1) / bsd
        pt = np.median(A["G"]) / pt_m[best]; lo, hi = np.percentile(ratio, 2.5), np.percentile(ratio, 97.5)
        res[k] = {"ratio": float(pt), "lo": float(lo), "hi": float(hi), "best": best, "best_median": float(pt_m[best]), "win": bool(pt < 1 and hi < 1)}
    res["G_median"] = float(np.median(A["G"])); res["W_median"] = float(np.median(A["W"]))
    return res


rows, allc = [], []
for st, sp in STUDY.items():
    orig = None if st == "locked15" else json.loads((D(sp["dir"]) / "verdict.json").read_text())["results"]
    if st == "locked15": origl = json.loads((D("") / "verdict.json").read_text())["results"]
    for grp, gl in sp["groups"].items():
        if gl is None:
            gl = sorted(p.name[len("interp_"):-6] for p in (D(sp["dir"]) / "per_grid").glob("interp_*.jsonl"))
        for g in gl:
            try:
                ids, d, w = load(st, g, sp)
            except FileNotFoundError:
                rows.append({"study": st, "group": grp, "grid": g, "cond": "*", "missing": "WLS or GENCO/interp file"}); continue
            wall = w.assign(study=st); allc.append(wall)
            for cond, cfgs in sp["conds"].items():
                if not set(cfgs) <= set(w.config): rows.append({"study": st, "group": grp, "grid": g, "cond": cond, "missing": "WLS configs incomplete"}); continue
                A, ok, obs = mats(ids, d, w, cfgs)
                if any(np.isnan(a).any() for a in A.values()): rows.append({"study": st, "group": grp, "grid": g, "cond": cond, "missing": "NaN/incomplete scenarios"}); continue
                r = cell(A, zlib.crc32(f"{st}|{g}|{cond}".encode()))
                # published win flag (full scenario set); oos here is a 10-scenario subset so it differs by construction
                if st == "locked15":
                    key = {"f20": "primary_f20_pooled3", "f40": "secondary_f40"}[cond]; ow = origl[key][g]["win"]
                else:
                    ow = orig[cond][g]["win"]
                rows.append({"study": st, "group": grp, "grid": g, "cond": cond, "n_scen": len(ids), "published_win_full_set": ow,
                             "win5": r["5"]["win"], "win6": r["6"]["win"], "win6_fallback": r["6fb"]["win"], "ratio5": r["5"]["ratio"], "best5": r["5"]["best"], "ratio6": r["6"]["ratio"], "best6": r["6"]["best"],
                             "ratio6_hi": r["6"]["hi"], "G_median": r["G_median"], "W_median": r["W_median"], "ratio_G_over_W": r["G_median"] / r["W_median"] if r["W_median"] > 0 else float("nan"),
                             "W_win_alone": r["W_only"]["win"], "frac_not_converged": float(1 - ok.mean()), "frac_not_count_observable": float(1 - obs.mean())})
cells = pd.DataFrame(rows); cells.to_csv(HERE / "cells.csv", index=False)
pd.concat(allc, ignore_index=True).to_csv(HERE / "all_cases.csv", index=False)
miss = cells[cells.get("missing").notna()] if "missing" in cells else cells.iloc[0:0]
ok_c = cells[cells.get("missing").isna()] if "missing" in cells else cells
agg = {}
for (st, grp, cond), x in ok_c.groupby(["study", "group", "cond"]):
    agg[f"{st}/{grp}/{cond}"] = {"n_cells": int(len(x)), "published_wins_full_set": int(x.published_win_full_set.sum()), "wins_5_rederived": int(x.win5.sum()), "wins_6_with_WLS": int(x.win6.sum()),
                                 "wins_6_fallback": int(x.win6_fallback.sum()), "WLS_best_of_six_cells": int((x.best6 == "W").sum()), "WLS_beats_GENCO_alone_cells": int(x.W_win_alone.sum()),
                                 "median_ratio_G_over_W": float(np.nanmedian(x.ratio_G_over_W)), "frac_scen_not_converged": float(x.frac_not_converged.mean()), "frac_scen_not_count_observable": float(x.frac_not_count_observable.mean())}
tot_cells = len(ok_c); wbest = int((ok_c.best6 == "W").sum())
out = {"status": "UNAUDITED", "cells_scored": tot_cells, "cells_missing": miss.to_dict("records"), "WLS_best_of_six_cells": wbest, "WLS_best_of_six_frac": wbest / max(tot_cells, 1),
       "cells_flipping_win_to_loss": int(((ok_c.win5) & (~ok_c.win6)).sum()), "cells_flipping_loss_to_win": int(((~ok_c.win5) & (ok_c.win6)).sum()),
       "add_as_I6_rule_met": bool(wbest / max(tot_cells, 1) >= 0.10 or ((ok_c.win5) != (ok_c.win6)).any()), "aggregates": agg}
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
L = ["| study/group/cond | cells | published wins (full set) | wins I1-I5 (re-derived, same subset) | wins I1-I6 (WLS added) | wins I1-I6 fallback | cells WLS best | cells WLS beats GENCO alone | median G/WLS | not converged | not count-observable |", "|" + "---|" * 11]
for k, v in agg.items():
    L.append(f"| {k} | {v['n_cells']} | {v['published_wins_full_set']} | {v['wins_5_rederived']} | {v['wins_6_with_WLS']} | {v['wins_6_fallback']} | {v['WLS_best_of_six_cells']} | {v['WLS_beats_GENCO_alone_cells']} | {v['median_ratio_G_over_W']:.2f} | {v['frac_scen_not_converged']:.2f} | {v['frac_scen_not_count_observable']:.2f} |")
(HERE / "table.md").write_text("\n".join(L) + "\n")
print("\n".join(L)); print("missing cells", len(miss), "WLS best frac", out["WLS_best_of_six_frac"])
