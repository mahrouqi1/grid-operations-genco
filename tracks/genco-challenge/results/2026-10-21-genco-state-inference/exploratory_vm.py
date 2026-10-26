"""POST-HOC, NOT PRE-REGISTERED, NOT IN THE VERDICT: two trivial measurement-using |V| predictors, to see whether GENCO's |V| gain
over B1-B3 is just 'uses the 20% measurements'. B4: every unmeasured bus gets the mean measured |V| of its scenario.
B5: harmonic interpolation of the measured |V| over the branch graph (unit weights), measured values held fixed.
Writes exploratory_vm.json (per-grid medians, same per-scenario MAE definition as the verdict). UNAUDITED."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import common as C
ids = json.loads((HERE / "test_ids.json").read_text()); out = {}
g_ = pd.read_json(HERE / "genco_per_scenario.jsonl", lines=True); b_ = pd.read_json(HERE / "baselines_per_scenario.jsonl", lines=True)
for g in C.HELDOUT15:
    tab = C.tables(g, ids[g]["test"]); e4, e5 = [], []
    for s in ids[g]["test"]:
        bus, br = tab["bus"][s], tab["br"][s]; n = len(bus); br = br[br.br_status == 1]
        m = np.zeros(n, bool); m[C.measured_set(s, n)] = True; vt = bus.Vm.to_numpy()
        e4.append(np.abs(vt[m].mean() - vt)[~m].mean())
        A = sp.coo_matrix((np.ones(len(br)), (br.from_bus, br.to_bus)), shape=(n, n)); A = (A + A.T).tocsr(); A.data[:] = 1.0
        Lm = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr(); u = np.where(~m)[0]; k = np.where(m)[0]
        x = vt.copy(); Luu = Lm[u][:, u].tocsc() + 1e-9 * sp.eye(len(u))
        x[u] = sp.linalg.spsolve(Luu, -Lm[u][:, k] @ vt[k]) if len(u) else x[u]
        e5.append(np.abs(x - vt)[u].mean())
    gm = g_[(g_.grid == g) & (g_.method == "genco_det20")].vm_mae_unmeas.median()
    bb = min(b_[(b_.grid == g) & (b_.method == m)].vm_mae_unmeas.median() for m in ("B1_ac", "B2_ac_meas", "B3_dc"))
    out[g] = {"B4_mean_measured": float(np.median(e4)), "B5_harmonic": float(np.median(e5)), "genco": float(gm), "best_B1_B3": float(bb)}
    print(g.replace("pglib_opf_", ""), {k: round(v, 4) for k, v in out[g].items()}, flush=True)
(HERE / "exploratory_vm.json").write_text(json.dumps(out, indent=1))
