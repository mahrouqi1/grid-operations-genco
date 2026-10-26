"""v3.5 copy of 2026-10-24-repair-root-audit/q_g_wls_ratios.py (original untouched). CHANGES: mask seed 0 only (CSEED = [f40_s0]),
four variants per cell (informed capped, informed 40/60, limit-blind capped, limit-blind 40/60), cross-check against cells_v3.csv.
Outputs q_g_ratios_per_grid_v35.csv, q_g_results_v35.json in this folder.
Original docstring: GENCO / WLS error ratios per grid (cell) at 40% measured, for informed and limit-blind WLS, per set of figure (a).
From saved per-scenario files only (no new solves). Mirrors verdict_v2.py (same loaders, same cell definition: ratio =
median over (scenarios x seeds) of GENCO vm_mae_unmeas / median of WLS vm_mae_unmeas; 95% CI = 2,000 bootstrap resamples of
scenarios, same crc32 seeds). Cross-checks the point ratio against cells_v2.csv W_only_ratio / Wflat_only_ratio.
Run (any env with numpy+pandas):  python q_g_wls_ratios.py -> q_g_ratios_per_grid.csv, q_g_results.json
Metric: vm_mae_unmeas = mean |V| error (pu) on the unmeasured buses of one scenario."""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent  # results/
W = R / "2026-10-21-genco-voltage-wls"
D = lambda s: R / ("2026-10-21-genco-voltage-inference" + s)
NB = 2000
CSEED = ["f40_s0"]
oos_p = json.loads((D("-oos") / "protocol.json").read_text())
H15 = json.loads((D("") / "protocol.json").read_text())["grids"]
STUDY = {
    "locked15": dict(dir="", cfgs=["f40_s0"], groups={"locked15": H15}, limit=30),
    "oos": dict(dir="-oos", cfgs=CSEED, groups={"oos_unseen": oos_p["sets"]["primary"], "oos_seen": oos_p["sets"]["seen"]}, limit=10),
    "control": dict(dir="-control", cfgs=CSEED, groups={"carved_seen_datakit": None}, limit=30),
    "fvs": dict(dir="-family-vs-size", cfgs=CSEED, groups={"carved_unseen_datakit": None}, limit=30),
}


def piv(df, cfgs, ids, col):
    x = df[df.config.isin(cfgs)]
    return x.pivot(index="scenario", columns="config", values=col).reindex(columns=cfgs).reindex(ids).to_numpy()


def boot(g, w, seed):
    n = g.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (NB, n))
    r = np.median(g[idx].reshape(NB, -1), axis=1) / np.median(w[idx].reshape(NB, -1), axis=1)
    return float(np.median(g) / np.median(w)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))



# four WLS variants per cell, mask seed 0 (the only seed with 40/60-iteration data)
VARIANTS = {  # key: (csv path under W, label)
    "inf": ("per_case/{st}/{g}.csv", "informed, original iteration limit"),
    "infunc": ("per_case_v3/uncapped/{st}/{g}.csv", "informed, limit raised to 40/60"),
    "flat": ("per_case_v2/flatprior/{st}/{g}.csv", "limit-blind, original iteration limit"),
    "flatunc": ("per_case_v3/flatuncapped/{st}/{g}.csv", "limit-blind, limit raised to 40/60"),
}
rows = []
for st, sp in STUDY.items():
    for grp, gl in sp["groups"].items():
        if gl is None:
            gl = sorted(p.name[len("interp_"):-6] for p in (D(sp["dir"]) / "per_grid").glob("interp_*.jsonl"))
        for g in gl:
            base = R / "2026-10-21-genco-state-inference" if st == "locked15" else D(sp["dir"])
            try:
                ids = json.loads((base / "test_ids.json").read_text())[g]["test"]
                if sp["limit"]:
                    ids = ids[: sp["limit"]]
                if st == "locked15":
                    ge = pd.read_json(D("") / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
                    d = ge[ge.grid == g]
                else:
                    d = pd.read_json(D(sp["dir"]) / "per_grid" / f"genco_{g}.jsonl", lines=True).drop(columns=["diag"])
            except FileNotFoundError:
                continue
            G = piv(d[d.method == "G"], sp["cfgs"], ids, "vm_mae_unmeas")
            row = {"study": st, "group": grp, "grid": g, "n_scen": len(ids)}
            for k, (pat, _) in VARIANTS.items():
                f = W / pat.format(st=st, g=g)
                if not f.exists():
                    continue
                wv = pd.read_csv(f)
                if not set(sp["cfgs"]) <= set(wv.config):
                    continue
                Wm = piv(wv, sp["cfgs"], ids, "wls_mae")
                ok = piv(wv.assign(c=(wv.status == "converged").astype(float)), sp["cfgs"], ids, "c")
                if not np.isnan(G).any() and not np.isnan(Wm).any():
                    row["ratio_" + k] = float(np.median(G) / np.median(Wm))
                    row["nc_" + k] = float(1 - np.nanmean(ok))
            rows.append(row)
T = pd.DataFrame(rows)
# cross-check against cells_v3.csv (seed-0 cell ratios computed by verdict_v3.py)
cells = pd.read_csv(W / "cells_v3.csv")
c40 = cells[cells.cond == "f40"][["study", "grid", "W_only_ratio", "Wunc_only_ratio", "Wflat_only_ratio", "Wflatunc_only_ratio"]]
T = T.merge(c40, on=["study", "grid"], how="left")
CHK = {"inf": "W_only_ratio", "infunc": "Wunc_only_ratio", "flat": "Wflat_only_ratio", "flatunc": "Wflatunc_only_ratio"}
for k, c in CHK.items():
    T["chk_" + k] = (T["ratio_" + k] - T[c]).abs()
T.to_csv(HERE / "q_g_ratios_per_grid_v35.csv", index=False)

SETS = [("carved pieces", T[T.study.isin(["control", "fvs"])]), ("locked 15", T[T.study == "locked15"]),
        ("seen whole grids", T[T.group == "oos_seen"]), ("unseen whole grids", T[T.group == "oos_unseen"])]


def summ(x, r, nc):
    x = x[x[r].notna()]
    v = x[r].to_numpy()
    return dict(n=int(len(x)), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                n_genco_better_pt=int((v < 1).sum()), n_ratio_gt120=int((v > 120).sum()), mean_frac_not_converged=float(x[nc].mean()))


out = {"mask_seed": 0, "max_abs_diff_vs_cells_v3": {k: float(T["chk_" + k].max()) for k in CHK}, "sets": {}}
for name, x in SETS:
    out["sets"][name] = {k: summ(x, "ratio_" + k, "nc_" + k) for k in VARIANTS}
(HERE / "q_g_results_v35.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
