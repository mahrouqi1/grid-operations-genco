"""Reproduction diff (protocol.json 'reproduction_gate'): per-scenario vm_mae_unmeas recomputed from the SAVED per-bus arrays
and from the rerun logs vs the stored values of the published run. Writes repro_diff.json. Run before count_violations.py.
python repro_diff.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; PUB = HERE.parent / "2026-10-21-genco-voltage-inference"
CFGS = ["f40_s0", "f20_s0"]
ip = pd.read_json(PUB / "interp_per_scenario.jsonl", lines=True); ip = ip[ip.method.isin(["I1", "I2", "I3", "I4", "I5"])]
ge = pd.read_json(PUB / "genco_v_per_scenario.jsonl", lines=True).drop(columns=["diag"])
pub = pd.concat([ip, ge]); pub = pub[pub.config.isin(CFGS)].set_index(["grid", "config", "method", "scenario"]).vm_mae_unmeas
res = {"saved_vs_published": {}, "log_vs_published": {}}
rows = []
for f in sorted((HERE / "perbus").glob("*.npz")):
    kind, g, c = f.stem.split("__"); z = np.load(f)
    meths = ["G"] if kind == "genco" else ["I1", "I2", "I3", "I4", "I5"]
    for m in meths:
        mae = np.array([np.abs(z[m][i] - z["label"][i])[~z["meas"][i]].mean() for i in range(len(z["scen"]))])
        ref = pub.loc[[(g, c, m, int(s)) for s in z["scen"]]].to_numpy()
        rows.append({"kind": kind, "grid": g, "config": c, "method": m, "n": len(mae), "max_abs_diff": float(np.abs(mae - ref).max()), "max_ref": float(ref.max())})
d = pd.DataFrame(rows); d.to_csv(HERE / "repro_diff_by_group.csv", index=False)
for kind in ["interp", "genco"]:
    x = d[d.kind == kind]
    res["saved_vs_published"][kind] = {"groups": len(x), "scenarios": int(x.n.sum()), "max_abs_diff_pu": float(x.max_abs_diff.max()), "worst": x.loc[x.max_abs_diff.idxmax()].to_dict() if len(x) else None}
for fn, kind in [("interp_pb_mae.jsonl", "interp"), ("genco_pb_mae.jsonl", "genco")]:
    p = HERE / fn
    if not p.exists(): continue
    l = pd.read_json(p, lines=True)
    l = l.drop(columns=[c for c in ["diag"] if c in l]).set_index(["grid", "config", "method", "scenario"]).vm_mae_unmeas
    ref = pub.loc[l.index]
    res["log_vs_published"][kind] = {"n": len(l), "max_abs_diff_pu": float((l - ref).abs().max())}
(HERE / "repro_diff.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
