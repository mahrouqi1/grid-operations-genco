"""Verdict v2 (round-1 audit follow-up). From saved per-scenario files only. UNAUDITED. Spec: protocol_amendment_3.json (written before any flatprior/uncapped result).
python verdict_v2.py -> cells_v2.csv, verdict_v2.json, table_v2.md (published arms, corrected columns), table_v2_arms.md (all arms).
Reproduces the published cells.csv flags (win5/win6/W_win_alone) as a built-in check and prints mismatches. Does NOT overwrite any existing output."""
import json, zlib, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; R = HERE.parent
D = lambda s: R / ("2026-10-21-genco-voltage-inference" + s)
ABL = R / "2026-10-22-genco-voltage-limit-ablation" / "per_grid"
IM = ["I1", "I2", "I3", "I4", "I5"]; NB = 2000
CSEED = {"f20": ["f20_s0", "f20_s1", "f20_s2"], "f40": ["f40_s0", "f40_s1", "f40_s2"], "n40": ["n40_s0", "n40_s1", "n40_s2"]}
oos_p = json.loads((D("-oos") / "protocol.json").read_text()); gc_p = json.loads((D("-generator-control") / "protocol.json").read_text())
gc_r = json.loads((D("-generator-control") / "protocol_ran.json").read_text())["ran"]
H15 = json.loads((D("") / "protocol.json").read_text())["grids"]
STUDY = {
    "locked15": dict(dir="", conds={"f20": CSEED["f20"], "f40": ["f40_s0"]}, groups={"locked15": H15}, limit=30),
    "oos": dict(dir="-oos", conds=CSEED, groups={"oos_unseen": oos_p["sets"]["primary"], "oos_seen": oos_p["sets"]["seen"]}, limit=10),
    "control": dict(dir="-control", conds=CSEED, groups={"carved_seen_datakit": None}, limit=30),
    "fvs": dict(dir="-family-vs-size", conds=CSEED, groups={"carved_unseen_datakit": None}, limit=30),
    "genctl": dict(dir="-generator-control", conds={k: CSEED[k] for k in ("f20", "f40")},
                   groups={"carved_seen_ladder": [g for g in gc_r if gc_p["pieces"][g]["group"] == "seen"],
                           "carved_unseen_ladder": [g for g in gc_r if gc_p["pieces"][g]["group"] == "unseen"]}, limit=30),
}
V2 = {"flat": "flatprior", "unc": "uncapped", "flatunc": "flatuncapped"}


def load(st, g, sp):
    base = R / "2026-10-21-genco-state-inference" if st == "locked15" else D(sp["dir"])
    ids = json.loads((base / "test_ids.json").read_text())[g]["test"]
    if sp["limit"]: ids = ids[: sp["limit"]]
    if st == "locked15":
        ip = pd.read_json(D("") / "interp_per_scenario.jsonl", lines=True); ge = pd.read_json(D("") / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
        d = pd.concat([ip[ip.grid == g], ge[ge.grid == g]], ignore_index=True)
    else:
        pg = D(sp["dir"]) / "per_grid"
        d = pd.concat([pd.read_json(pg / f"interp_{g}.jsonl", lines=True), pd.read_json(pg / f"genco_{g}.jsonl", lines=True).drop(columns=["diag"])], ignore_index=True)
    w = pd.read_csv(HERE / "per_case" / st / f"{g}.csv")
    extra = {}
    for k, arm in V2.items():
        f = HERE / "per_case_v2" / arm / st / f"{g}.csv"
        extra[k] = pd.read_csv(f) if f.exists() else None
    fa = ABL / f"genco_ablate_{g}.jsonl"
    a1 = None
    if fa.exists() and st != "locked15":
        a = pd.read_json(fa, lines=True); a1 = a[a.arm == "A1"]
    return ids, d, w, extra, a1


def piv(df, cfgs, ids, col, extra_cols=None):
    x = df[df.config.isin(cfgs)]
    return x.pivot(index="scenario", columns="config", values=col).reindex(columns=cfgs).reindex(ids).to_numpy()


def mats(ids, d, w, extra, a1, cfgs):
    A = {}; ok = {}
    for m in ["G"] + IM:
        A[m] = piv(d[d.method == m], cfgs, ids, "vm_mae_unmeas")
    A["W"] = piv(w, cfgs, ids, "wls_mae"); ok["W"] = piv(w.assign(c=(w.status == "converged").astype(float)), cfgs, ids, "c")
    obs = piv(w, cfgs, ids, "count_observable")
    A["W_fb"] = np.where(ok["W"] == 1, A["W"], A["I2"])
    for k in V2:
        e = extra[k]
        if e is not None and set(cfgs) <= set(e.config):
            A["W" + k] = piv(e, cfgs, ids, "wls_mae"); ok["W" + k] = piv(e.assign(c=(e.status == "converged").astype(float)), cfgs, ids, "c")
            A["W" + k + "_fb"] = np.where(ok["W" + k] == 1, A["W" + k], A["I2"])
    if a1 is not None and set(cfgs) <= set(a1.config):
        A["G1"] = piv(a1, cfgs, ids, "vm_mae_unmeas")
    return A, ok, obs


def cell(A, seed):
    n = A["G"].shape[0]; rng = np.random.default_rng(seed); idx = rng.integers(0, n, (NB, n))
    med = lambda m, ix: np.median(A[m][ix].reshape(len(ix), -1), axis=1)
    arms = {"5": ("G", IM), "6": ("G", IM + ["W"]), "6fb": ("G", IM + ["W_fb"]), "W_only": ("G", ["W"])}
    if "Wflat" in A: arms.update({"6flat": ("G", IM + ["Wflat"]), "6flat_fb": ("G", IM + ["Wflat_fb"]), "Wflat_only": ("G", ["Wflat"])})
    if "Wunc" in A: arms.update({"6unc": ("G", IM + ["Wunc"]), "6unc_fb": ("G", IM + ["Wunc_fb"]), "Wunc_only": ("G", ["Wunc"])})
    if "Wflatunc" in A: arms.update({"6flatunc": ("G", IM + ["Wflatunc"]), "6flatunc_fb": ("G", IM + ["Wflatunc_fb"]), "Wflatunc_only": ("G", ["Wflatunc"])})
    if "G1" in A:
        arms.update({"G1_5": ("G1", IM), "G1_6": ("G1", IM + ["W"])})
        if "Wflatunc" in A: arms.update({"G1_6flatunc": ("G1", IM + ["Wflatunc"]), "G1_Wflatunc_only": ("G1", ["Wflatunc"])})
        if "Wflat" in A: arms.update({"G1_6flat": ("G1", IM + ["Wflat"]), "G1_Wflat_only": ("G1", ["Wflat"])})
    res = {}
    for k, (gk, ms) in arms.items():
        pt_m = {m: np.median(A[m]) for m in ms}; best = min(pt_m, key=pt_m.get)
        bsd = np.min(np.stack([med(m, idx) for m in ms]), axis=0); ratio = np.median(A[gk][idx].reshape(NB, -1), axis=1) / bsd
        pt = np.median(A[gk]) / pt_m[best]; lo, hi = np.percentile(ratio, 2.5), np.percentile(ratio, 97.5)
        res[k] = {"ratio": float(pt), "lo": float(lo), "hi": float(hi), "best": best, "win": bool(pt < 1 and hi < 1), "other_better": bool(pt > 1 and lo > 1)}
    for m in A: res["med_" + m] = float(np.median(A[m]))
    return res


old = pd.read_csv(HERE / "cells.csv")
rows = []
for st, sp in STUDY.items():
    for grp, gl in sp["groups"].items():
        if gl is None: gl = sorted(p.name[len("interp_"):-6] for p in (D(sp["dir"]) / "per_grid").glob("interp_*.jsonl"))
        for g in gl:
            try: ids, d, w, extra, a1 = load(st, g, sp)
            except FileNotFoundError: continue
            for cond, cfgs in sp["conds"].items():
                if not set(cfgs) <= set(w.config): continue
                A, ok, obs = mats(ids, d, w, extra, a1, cfgs)
                A = {k: v for k, v in A.items() if not np.isnan(v).any()}
                if not {"G", "W"} <= set(A) or not set(IM) <= set(A): continue
                r = cell(A, zlib.crc32(f"{st}|{g}|{cond}".encode()))
                row = {"study": st, "group": grp, "grid": g, "cond": cond, "n_scen": len(ids)}
                for k in ["5", "6", "6fb", "6flat", "6flat_fb", "6unc", "6unc_fb", "6flatunc", "6flatunc_fb", "G1_5", "G1_6", "G1_6flat", "G1_6flatunc"]:
                    if k in r: row["win" + k] = r[k]["win"]; row["best" + k] = r[k]["best"]; row["ratio" + k] = r[k]["ratio"]
                for k in ["W_only", "Wflat_only", "Wunc_only", "Wflatunc_only", "G1_Wflat_only", "G1_Wflatunc_only"]:
                    if k in r: row[k + "_Gbetter"] = r[k]["win"]; row[k + "_Wbetter"] = r[k]["other_better"]; row[k + "_ratio"] = r[k]["ratio"]
                for m in ["G", "G1", "W", "Wflat", "Wunc", "Wflatunc"]:
                    if "med_" + m in r: row["med_" + m] = r["med_" + m]
                row["G_median_below_W_median"] = r["med_G"] < r["med_W"]
                row["frac_nc_W"] = float(1 - np.nanmean(ok["W"]))
                for k in V2:
                    if "W" + k in A: row["frac_nc_W" + k] = float(1 - np.nanmean(ok["W" + k]))
                row["frac_not_count_observable"] = float(1 - np.nanmean(obs))
                rows.append(row)
cells = pd.DataFrame(rows); cells.to_csv(HERE / "cells_v2.csv", index=False)

# built-in reproduction check against the published cells.csv
mm = old[old.missing.isna()] if "missing" in old else old
chk = cells.merge(mm[["study", "group", "grid", "cond", "win5", "win6", "win6_fallback", "W_win_alone"]].rename(columns=lambda c: c + "_old" if c not in ("study", "group", "grid", "cond") else c), on=["study", "group", "grid", "cond"], how="left")
repro = {"cells_v2": int(len(cells)), "cells_in_published": int(chk.win5_old.notna().sum()),
         "mismatch_win5": int((chk.win5 != chk.win5_old.astype(bool)).sum()), "mismatch_win6": int((chk["win6"] != chk.win6_old.astype(bool)).sum()),
         "mismatch_win6fb": int((chk["win6fb"] != chk.win6_fallback_old.astype(bool)).sum()), "mismatch_W_win_alone(G better)": int((chk["W_only_Gbetter"] != chk.W_win_alone_old.astype(bool)).sum()),
         "published_cells_not_reproduced": int(len(mm) - chk.win5_old.notna().sum())}
print("reproduction check", repro)

S = lambda x, c: int(x[c].sum()) if c in x else None
POOL = {"pieces_all_36": ["carved_seen_datakit", "carved_unseen_datakit"], "pieces_ladder_36": ["carved_seen_ladder", "carved_unseen_ladder"]}


def agg_table(df):
    out = {}
    for (st, grp, cond), x in df.groupby(["study", "group", "cond"]):
        out[f"{st}/{grp}/{cond}"] = x
    for pn, gs in POOL.items():
        for cond in ("f20", "f40"):
            x = df[df.group.isin(gs) & (df.cond == cond)]
            if len(x): out[f"POOLED/{pn}/{cond}"] = x
    return out


def row_for(x):
    """counts on all cells in the group (published arms) and on the subset where each new arm exists."""
    r = {"cells": int(len(x)), "win5": S(x, "win5"), "win6": S(x, "win6"), "win6_fb": S(x, "win6fb"),
         "W_best_of_six": int((x.best6 == "W").sum()), "GENCO_beats_WLS_alone": S(x, "W_only_Gbetter"), "WLS_beats_GENCO_alone": S(x, "W_only_Wbetter"),
         "GENCO_median_below_WLS_median": S(x, "G_median_below_W_median"), "nc_W": float(x.frac_nc_W.mean())}
    f = x[x["win6flat"].notna()] if "win6flat" in x else x.iloc[0:0]
    r["flat"] = {"cells_run": int(len(f)), "win5_same_cells": S(f, "win5"), "win6_published_same_cells": S(f, "win6"), "win6_flat": S(f, "win6flat"), "win6_flat_fb": S(f, "win6flat_fb"),
                 "GENCO_beats_Wflat_alone": S(f, "Wflat_only_Gbetter"), "Wflat_beats_GENCO_alone": S(f, "Wflat_only_Wbetter"), "Wflat_best_of_six": int((f.best6flat == "Wflat").sum()) if len(f) else 0,
                 "nc_Wflat": float(f.frac_nc_Wflat.mean()) if len(f) else None}
    g1 = f[f["win_G1_6flat"].notna()] if False else (f[f["winG1_6flat"].notna()] if "winG1_6flat" in f else f.iloc[0:0])
    r["flat_both_sides"] = {"cells_run": int(len(g1)), "win6_flat_GENCO_released": S(g1, "win6flat"), "G1_win5_(GENCO_flat_vs_I1-5)": S(g1, "winG1_5"), "G1_win6flat_(GENCO_flat_vs_I1-5+Wflat)": S(g1, "winG1_6flat"),
                            "G1_beats_Wflat_alone": S(g1, "G1_Wflat_only_Gbetter"), "Wflat_beats_G1_alone": S(g1, "G1_Wflat_only_Wbetter"), "win5_same_cells": S(g1, "win5"), "win6_published_same_cells": S(g1, "win6")}
    u = x[x["win6unc"].notna()] if "win6unc" in x else x.iloc[0:0]
    r["unc"] = {"cells_run": int(len(u)), "win5_same_cells": S(u, "win5"), "win6_published_capped_same_cells": S(u, "win6"), "win6_uncapped": S(u, "win6unc"), "win6_uncapped_fb": S(u, "win6unc_fb"),
                "nc_W_capped": float(u.frac_nc_W.mean()) if len(u) else None, "nc_Wunc": float(u.frac_nc_Wunc.mean()) if len(u) else None,
                "GENCO_beats_Wunc_alone": S(u, "Wunc_only_Gbetter"), "Wunc_beats_GENCO_alone": S(u, "Wunc_only_Wbetter")}
    fu = x[x["win6flatunc"].notna()] if "win6flatunc" in x else x.iloc[0:0]
    g2 = fu[fu["winG1_6flatunc"].notna()] if "winG1_6flatunc" in fu else fu.iloc[0:0]
    both = fu[fu["win6unc"].notna()] if "win6unc" in fu else fu.iloc[0:0]
    r["flatunc"] = {"cells_run": int(len(fu)), "win5_same_cells": S(fu, "win5"), "win6_published_capped_same_cells": S(fu, "win6"), "win6_flatuncapped": S(fu, "win6flatunc"), "win6_flatuncapped_fb": S(fu, "win6flatunc_fb"),
                    "GENCO_beats_Wflatunc_alone": S(fu, "Wflatunc_only_Gbetter"), "Wflatunc_beats_GENCO_alone": S(fu, "Wflatunc_only_Wbetter"), "nc_Wflatunc": float(fu.frac_nc_Wflatunc.mean()) if len(fu) else None,
                    "cells_with_informed_uncapped_too": int(len(both)), "informed_uncapped_wins_on_those": S(both, "win6unc"), "flatuncapped_wins_on_those": S(both, "win6flatunc"),
                    "flat_both_sides_cells": int(len(g2)), "G1_win6flatunc": S(g2, "winG1_6flatunc"), "flat_both_win5_same": S(g2, "win5"), "flat_both_win6flatunc_GENCO_released": S(g2, "win6flatunc")}
    return r


tabs = {k: row_for(x) for k, x in agg_table(cells).items()}
(HERE / "verdict_v2.json").write_text(json.dumps({"status": "UNAUDITED", "reproduction_check": repro, "aggregates": tabs, "grids_run": {
    "flatprior": sorted(set(cells[cells["win6flat"].notna()].grid)) if "win6flat" in cells else [], "uncapped": sorted(set(cells[cells["win6unc"].notna()].grid)) if "win6unc" in cells else []}}, indent=1))

# table_v2.md: the published-arm table with the inverted column corrected (same cell population as table.md)
L = ["| study/group/cond | cells | wins I1-I5 (re-derived, same subset) | wins I1-I6 (WLS added) | wins I1-I6 fallback | cells WLS best | cells GENCO beats WLS alone (CI) | cells WLS beats GENCO alone (CI) | cells GENCO median < WLS median | not converged |", "|" + "---|" * 10]
for k, v in tabs.items():
    if k.startswith("POOLED"): continue
    L.append(f"| {k} | {v['cells']} | {v['win5']} | {v['win6']} | {v['win6_fb']} | {v['W_best_of_six']} | {v['GENCO_beats_WLS_alone']} | {v['WLS_beats_GENCO_alone']} | {v['GENCO_median_below_WLS_median']} | {v['nc_W']:.2f} |")
(HERE / "table_v2.md").write_text("table_v2.md - regenerated by verdict_v2.py. Column 'cells GENCO beats WLS alone' is what table.md called 'cells WLS beats GENCO alone' (inverted label, audit F5). UNAUDITED.\n\n" + "\n".join(L) + "\n")
A2 = ["| set/cond | flat-uncapped cells | I1-I5 | published (capped) | flat-uncapped | flat-uncapped fallback | cells with informed-uncapped too | informed-uncapped wins | flat-uncapped wins (same cells) | A1 flat-both-sides cells | A1 wins vs I1-I5+WLS-flat-unc | not conv (flat-unc) |", "|" + "---|" * 12]
for k, v in tabs.items():
    q = v["flatunc"]; A2.append(f"| {k} | {q['cells_run']} | {q['win5_same_cells']} | {q['win6_published_capped_same_cells']} | {q['win6_flatuncapped']} | {q['win6_flatuncapped_fb']} | {q['cells_with_informed_uncapped_too']} | {q['informed_uncapped_wins_on_those']} | {q['flatuncapped_wins_on_those']} | {q['flat_both_sides_cells']} | {q['G1_win6flatunc']} | {q['nc_Wflatunc']} |")
A = ["| set/cond | cells (run) | I1-I5 wins | I1-I6 published WLS | published fallback | flat-prior cells run | I1-I5 | published WLS | WLS-flat | WLS-flat fallback | GENCO-flat A1 vs I1-I5+WLS-flat (cells) | A1 wins | uncapped cells run | I1-I5 | capped published | uncapped | uncapped fallback |", "|" + "---|" * 17]
for k, v in tabs.items():
    f, b, u = v["flat"], v["flat_both_sides"], v["unc"]
    A.append(f"| {k} | {v['cells']} | {v['win5']} | {v['win6']} | {v['win6_fb']} | {f['cells_run']} | {f['win5_same_cells']} | {f['win6_published_same_cells']} | {f['win6_flat']} | {f['win6_flat_fb']} | {b['cells_run']} | {b['G1_win6flat_(GENCO_flat_vs_I1-5+Wflat)']} | {u['cells_run']} | {u['win5_same_cells']} | {u['win6_published_capped_same_cells']} | {u['win6_uncapped']} | {u['win6_uncapped_fb']} |")
(HERE / "table_v2_arms.md").write_text("\n".join(A) + "\n\n" + "\n".join(A2) + "\n")
print("\n".join(A))
