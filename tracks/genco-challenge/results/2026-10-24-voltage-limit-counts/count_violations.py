"""D1 counts, exactly as fixed in protocol.json (committed before any prediction). UNAUDITED.
Reads perbus/*.npz; writes counts.csv (per grid, config, method, scope, collapsed policy, tau) and pooled.csv (micro-pooled + per-grid medians).
python count_violations.py"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
CFGS = ["f40_s0", "f20_s0"]; IM = ["I1", "I2", "I3", "I4", "I5"]; TAUS = [0.0, 0.005]
GRIDS = sorted({f.stem.split("__")[1] for f in (HERE / "perbus").glob("*.npz")})


def viol(x, lo, hi, tau):
    return (x < lo - tau) | (x > hi + tau)


rows = []
for g in GRIDS:
    for c in CFGS:
        zi, zg = np.load(HERE / "perbus" / f"interp__{g}__{c}.npz"), np.load(HERE / "perbus" / f"genco__{g}__{c}.npz")
        for k in ["scen", "label", "meas", "pq", "lo", "hi"]:
            assert np.array_equal(zi[k], zg[k]), (g, c, k)
        lab, meas, pq, lo, hi = zg["label"], zg["meas"], zg["pq"], zg["lo"], zg["hi"]
        P = {"G": zg["G"], **{m: zi[m] for m in IM}}
        collapsed = (lab < 0.5).any(1)
        mae = {m: np.median([np.abs(P[m][i] - lab[i])[~meas[i]].mean() for i in range(len(lab))]) for m in IM}   # selection by published MAE only, all scenarios
        best = min(IM, key=lambda m: mae[m])                                                                    # ties -> first
        P["best"] = P[best]
        for m, X in P.items():
            for scope in ["all_unmeas", "pq_unmeas"]:
                sel_b = ~meas if scope == "all_unmeas" else (~meas & pq)
                for pol in ["drop_collapsed", "include_collapsed"]:
                    sel_s = ~collapsed if pol == "drop_collapsed" else np.ones(len(lab), bool)
                    sel = sel_b & sel_s[:, None]
                    for tau in TAUS:
                        L = viol(lab, lo, hi, tau)[sel]; Pv = viol(X, lo, hi, tau)[sel]
                        rows.append({"grid": g, "config": c, "method": m, "best_is": best if m == "best" else "", "scope": scope, "collapsed": pol, "tau": tau,
                                     "n_scen": int(sel_s.sum()), "n_collapsed": int(collapsed.sum()), "scored_buses": int(sel.sum()),
                                     "true_violations": int(L.sum()), "caught": int((L & Pv).sum()), "missed": int((L & ~Pv).sum()), "false_alarms": int((~L & Pv).sum())})
D = pd.DataFrame(rows); D.to_csv(HERE / "counts.csv", index=False)

key = ["config", "method", "scope", "collapsed", "tau"]
S = D.groupby(key)[["n_scen", "scored_buses", "true_violations", "caught", "missed", "false_alarms"]].sum().reset_index()
S["predicted_violations"] = S.caught + S.false_alarms
S["catch_rate"] = S.caught / S.true_violations.replace(0, np.nan)
S["false_alarm_rate"] = S.false_alarms / (S.scored_buses - S.true_violations).replace(0, np.nan)
S["precision"] = S.caught / S.predicted_violations.replace(0, np.nan)
S["false_alarms_per_catch"] = S.false_alarms / S.caught.replace(0, np.nan)
D["catch_rate"] = D.caught / D.true_violations.replace(0, np.nan)
D["false_alarm_rate"] = D.false_alarms / (D.scored_buses - D.true_violations).replace(0, np.nan)
med = D.groupby(key)[["catch_rate", "false_alarm_rate"]].median().reset_index().rename(columns={"catch_rate": "pergrid_median_catch_rate", "false_alarm_rate": "pergrid_median_false_alarm_rate"})
ng = D[D.true_violations > 0].groupby(key).size().rename("grids_with_violation").reset_index()
S = S.merge(med, on=key).merge(ng, on=key, how="left")
S.to_csv(HERE / "pooled.csv", index=False)
print(S[(S.scope == "all_unmeas") & (S.collapsed == "drop_collapsed") & S.method.isin(["G", "best"])].round(4).to_string())
