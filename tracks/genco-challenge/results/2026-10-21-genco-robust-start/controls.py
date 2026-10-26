"""Control (ii): released-GENCO prediction errors vs the verified labels, per grid, first 100 scenarios (same as run_starts).
Median |V| error over all scenario-bus pairs (raw prediction and with REF/PV |V| put back), and median / mean branch angle-difference
error (deg, wrapped) for GENCO, DC (datakit Va_dc) and flat, over in-service branches. Also summarises the identity control.
    python controls.py -> controls.csv, controls.md"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / "data/external/genco-robust-start"
P = json.loads((HERE / "protocol.json").read_text())
wrap = lambda d: (d + 180.0) % 360.0 - 180.0
rows = []
for fam, gl in P["grids"].items():
    for g in gl:
        run = max((Path(p).parent.parent for p in glob.glob(str(HERE / "mlruns/*/*/tags/mlflow.runName")) if Path(p).read_text().strip() == f"zeroshot__{g}"), key=lambda p: p.stat().st_mtime)
        pr = pd.read_parquet(run / "artifacts/test" / f"{g}_predictions.parquet")
        scen = sorted(int(s) for s in pr["scenario"].unique())[:100]
        pr = pr[pr["scenario"].isin(scen)]
        raw = ROOT / g / "raw"
        flt = [("scenario", "in", scen)]
        b = pd.read_parquet(raw / "bus_data.parquet", filters=flt, columns=["scenario", "bus", "Vm", "Va", "Va_dc"])
        br = pd.read_parquet(raw / "branch_data.parquet", filters=flt, columns=["scenario", "from_bus", "to_bus", "br_status"])
        df = pr.merge(b, on=["scenario", "bus"]).sort_values(["scenario", "bus"]).reset_index(drop=True)
        assert len(df) == len(pr)
        pin = (df.PV == 1) | (df.REF == 1)
        vm_pin = np.where(pin, df.Vm, df.Vm_pred)
        va_pred = np.degrees(df.Va_pred.to_numpy())
        refva = df.groupby("scenario")["Va"].transform(lambda s: s[df.loc[s.index, "REF"] == 1].mean()).to_numpy()
        br = br[br.br_status == 1]
        key = pd.MultiIndex.from_arrays([df.scenario, df.bus])
        fi = key.get_indexer(pd.MultiIndex.from_arrays([br.scenario, br.from_bus])); ti = key.get_indexer(pd.MultiIndex.from_arrays([br.scenario, br.to_bus]))
        m = (fi >= 0) & (ti >= 0); fi, ti = fi[m], ti[m]
        va_t = df.Va.to_numpy()
        e = lambda v: np.abs(wrap((v[fi] - v[ti]) - (va_t[fi] - va_t[ti])))
        eg, ed, ef = e(va_pred), e(df.Va_dc.to_numpy()), e(refva)
        rows.append(dict(family=fam, grid=g, n_bus=int(df.groupby("scenario").size().iloc[0]), n_scen=len(scen),
                         med_dVm_raw=float(np.median(np.abs(df.Vm_pred - df.Vm))), med_dVm_pinned=float(np.median(np.abs(vm_pin - df.Vm))),
                         med_branch_dVa_genco=float(np.median(eg)), med_branch_dVa_dc=float(np.median(ed)), med_branch_dVa_flat=float(np.median(ef)),
                         mean_branch_dVa_genco=float(eg.mean()), mean_branch_dVa_dc=float(ed.mean()), mean_branch_dVa_flat=float(ef.mean()),
                         med_dVm_dc_flat_setpoint=float(np.median(np.abs(np.where(pin, df.Vm, 1.0) - df.Vm)))))
C = pd.DataFrame(rows); C.to_csv(HERE / "controls.csv", index=False)
L = ["| grid | buses | scen | median abs dVm, GENCO raw (pu) | GENCO REF/PV put back | setpoint/1.0 (flat, DC) | median branch dVa GENCO (deg) | DC | flat | mean branch dVa GENCO | DC | flat |", "|" + "---|" * 12]
for r in rows:
    L.append(f"| {r['grid']} | {r['n_bus']} | {r['n_scen']} | {r['med_dVm_raw']:.4f} | {r['med_dVm_pinned']:.4f} | {r['med_dVm_dc_flat_setpoint']:.4f} | {r['med_branch_dVa_genco']:.3f} | {r['med_branch_dVa_dc']:.3f} | {r['med_branch_dVa_flat']:.3f} | {r['mean_branch_dVa_genco']:.3f} | {r['mean_branch_dVa_dc']:.3f} | {r['mean_branch_dVa_flat']:.3f} |")
L += ["", "Identity control (Newton started AT the label):", "", "| grid | scenarios | max iterations | all converged | max change (pu) |", "|---|---|---|---|---|"]
for f in sorted((HERE / "identity").glob("*.json")):
    d = json.loads(f.read_text()); L.append(f"| {d['grid']} | {d['n']} | {d['max_iterations']} | {d['all_converged']} | {d['max_change_pu']:.1e} |")
(HERE / "controls.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
