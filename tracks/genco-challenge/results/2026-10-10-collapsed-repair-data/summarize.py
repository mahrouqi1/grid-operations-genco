"""Recompute every count from the per-scenario csv. Writes status.csv (all grids) and tables.md."""
import glob, pandas as pd, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(H / "per_grid/*.csv")))], ignore_index=True)
df.to_csv(H / "status.csv", index=False)
order = ["case4837_goc_idx_35", "case20758_epigrids_idx_49", "case9591_goc_idx_12", "case5658_epigrids_idx_8", "case4837_goc_idx_28",
         "case19402_goc_idx_47", "case10480_goc_idx_40", "case4837_goc_idx_37", "case10480_goc_idx_26"]
rows = []
for g in order:
    d = df[df.grid == g]
    if d.empty:
        continue
    rec, flip, no = (d.status == "recovered").sum(), (d.status == "recovered_signflip").sum(), (d.status == "no_root").sum()
    ph = d[d.status == "recovered"].phase.astype(str).value_counts().to_dict()
    rows.append({"grid": g, "flagged": len(d), "collapsed (|Vm|<0.5)": int((d.label_min_abs_vm < 0.5).sum()),
                 "sign-flip only": int((d.label_reason == "negative_vm_sign_flip").sum()),
                 "recovered": int(rec), "recovered_signflip": int(flip), "no_root": int(no),
                 "recovery rate": round((rec + flip) / len(d), 4), "by phase (1/2/3)": f"{ph.get('1',0)}/{ph.get('2',0)}/{ph.get('3',0)}"})
t = pd.DataFrame(rows)
tot = t.drop(columns=["grid", "recovery rate", "by phase (1/2/3)"]).sum()
out = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
out += ["| " + " | ".join(str(x) for x in r) + " |" for r in t.values.tolist()]
out.append(f"| TOTAL | {tot['flagged']} | {tot['collapsed (|Vm|<0.5)']} | {tot['sign-flip only']} | {tot['recovered']} | {tot['recovered_signflip']} | {tot['no_root']} | "
           f"{round((tot['recovered']+tot['recovered_signflip'])/tot['flagged'],4)} | |")
nr = df[df.status == "no_root"]
txt = "\n".join(out)
if len(nr):
    txt += "\n\nNo-root reasons (relabel.accept_relabel reason):\n\n" + nr.groupby(["grid", "no_root_reason"]).size().to_string()
    txt += "\n\nNo-root load-continuation mu_max (1.0 = reached full load) and homotopy lambda_max:\n\n" + nr.groupby("grid")[["mu_max", "homotopy_lambda_max"]].describe().T.to_string()
(H / "tables.md").write_text(txt + "\n")
print(txt)
