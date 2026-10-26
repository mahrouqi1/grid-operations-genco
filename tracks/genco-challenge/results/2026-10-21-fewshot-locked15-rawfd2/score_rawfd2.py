"""F1 control (audit genco-make-it-work R1-F1) (UNAUDITED). Re-scores RAW fd2 (2 raw XB iterates from DC, a non-finite step skipped:
the corrector-at-scale/scale.py convention, same fdxb_step as robust-start) next to the GUARDED fd2 (lp2.priors_for, what the few-shot
study scored) on exactly the few-shot eval scenarios of every grid it scored (scenario set = a stored prediction parquet of that grid's eval).
Then computes the few-shot wu100L GENCO ratio to each, per draw, from locked15/results.jsonl (rows are read, not edited).
Run: python score_rawfd2.py   (CPU only)"""
import glob, json, statistics as st, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
L15 = HERE.parent / "2026-10-21-fewshot-locked15"
sys.path.insert(0, str(HERE.parent / "2026-10-09-genco-on-fd2/scripts"))
import common as cm
from genco_challenge import fewshot as fs, hybrid as hy, labelfree_prior2 as lp2
from genco_challenge.fresh_eval_data import EVAL2
EVAL2F = fs.WORK / "eval2_renumbered"
rows = [json.loads(l) for l in (L15 / "results.jsonl").read_text().splitlines()]
rows = [r for r in rows if "error" not in r and r["tag"] == "wu100L"]
grids = sorted({r["grid"] for r in rows}, key=lambda g: int(g.split("case")[1].split("_")[0]))
preds = glob.glob(str(fs.WORK / "fewshot_lk15/*/mlruns_eval/*/*/artifacts/test/*_predictions.parquet"))
out = {}
for g in grids:
    n_want = {r["n_scen_ours"] for r in rows if r["grid"] == g}
    assert len(n_want) == 1; n_want = n_want.pop()
    cand = [f for f in preds if f.endswith(f"/{g}_predictions.parquet")]
    pred = None
    for f in cand:
        p = pd.read_parquet(f)
        if p["scenario"].nunique() == n_want: pred = p; break
    assert pred is not None, g
    EV = EVAL2F if (EVAL2F / g / "meta.json").exists() else EVAL2
    acc = {"dc": [], "fd2_guarded": [], "fd2_raw": []}; n_diff = n_nonfin = 0
    for o in cm.bundles(EV, g, pred):
        sc = o.sc
        acc["dc"].append(cm.score(o, sc.vm_set, o.va_dc))
        vm, va = cm.fd_state(o, "fd2"); acc["fd2_guarded"].append(cm.score(o, vm, va))
        v = sc.vm_set * np.exp(1j * sc.va_dc)
        for _ in range(2):
            try:
                vn = sc.fdxb_step(v)
                if hy._finite(vn): v = vn
                else: n_nonfin += 1
            except Exception:
                n_nonfin += 1
        n_diff += int(not np.array_equal(v, o.pr["fd2"]["v"]))
        acc["fd2_raw"].append(cm.score(o, np.abs(v), np.radians(lp2._unwrapped(sc, v))))
    a = {k: cm.agg(np.array(v)) for k, v in acc.items()}
    mine = {k: a[k]["branch_dva_deg"] for k in a}
    # check against what the few-shot study stored for the same comparators
    stored = rows[[r["grid"] for r in rows].index(g)]["ours"]
    chk = {"dc": abs(mine["dc"] - stored["dc"]["branch_dva_deg"]), "fd2_guarded": abs(mine["fd2_guarded"] - stored["fd2"]["branch_dva_deg"])}
    gd = {r["split_seed"]: r["ours"]["genco_raw"]["branch_dva_deg"] for r in rows if r["grid"] == g}
    out[g] = {"n_scen": n_want, "dc": mine["dc"], "fd2_guarded": mine["fd2_guarded"], "fd2_raw": mine["fd2_raw"],
              "scen_raw_differs_from_guarded": n_diff, "nonfinite_steps_skipped": n_nonfin, "recheck_abs_diff_vs_stored": chk,
              "genco_dva_per_draw": gd,
              "ratio_guarded_per_draw": {s: v / mine["fd2_guarded"] for s, v in gd.items()},
              "ratio_raw_per_draw": {s: v / mine["fd2_raw"] for s, v in gd.items()},
              "ratio_guarded_mean": st.mean(v / mine["fd2_guarded"] for v in gd.values()),
              "ratio_raw_mean": st.mean(v / mine["fd2_raw"] for v in gd.values())}
    print(g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[g].items() if k in ("dc", "fd2_guarded", "fd2_raw", "ratio_raw_mean", "ratio_guarded_mean", "scen_raw_differs_from_guarded", "recheck_abs_diff_vs_stored")}, flush=True)
nb = lambda g: g.split("case")[1].split("_")[0]
L = ["| grid | n scen | dVa DC (deg) | dVa guarded fd2 | dVa raw fd2 | scen raw != guarded | GENCO (mean of draws) | GENCO / guarded fd2 | GENCO / raw fd2 | draws < raw fd2 | GENCO beats raw fd2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
nb_beat = 0
for g, d in out.items():
    gm = st.mean(d["genco_dva_per_draw"].values()); below = sum(x < 1 for x in d["ratio_raw_per_draw"].values()); beat = d["ratio_raw_mean"] < 1; nb_beat += beat
    L.append(f"| {nb(g)} | {d['n_scen']} | {d['dc']:.3f} | {d['fd2_guarded']:.4f} | {d['fd2_raw']:.4f} | {d['scen_raw_differs_from_guarded']} | {gm:.3f} | {d['ratio_guarded_mean']:.2f} | {d['ratio_raw_mean']:.2f} | {below}/{len(d['ratio_raw_per_draw'])} | {'YES' if beat else 'no'} |")
L.append(f"\nGENCO (wu100L, draw-mean branch dVa) beats RAW fd2 on {nb_beat} of {len(out)} scored grids; beats GUARDED fd2 on {sum(d['ratio_guarded_mean'] < 1 for d in out.values())}.")
(HERE / "table.md").write_text("\n".join(L) + "\n"); (HERE / "results.json").write_text(json.dumps(out, indent=1))
print("\n".join(L))
