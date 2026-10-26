"""Underfit-gate reference: MAE of the flat 1.0 pu guess against the label |V| at PQ buses of NORMAL scenarios (scenario
min |V| >= 0.5), for the 7 training subgrids that run_pretrain_mask scores (original labels = the staged eval tree).
The gate compares this to the model's hidden-PQ |V| MAE on normal scenarios (pmask rows' "normal.mae_vs_label").
All PQ buses are used (the pretraining mask hides a random ~80%, so the expectation is the same). UNAUDITED."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN7 = ["subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case5658_epigrids_idx_8",
          "subnet_case4837_goc_idx_28", "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37", "subnet_case10480_goc_idx_26"]
E = Path(__file__).resolve().parents[4] / "data/external/genco-pf-pretraining-eval"
out = {}
tot, n = 0.0, 0
for g in TRAIN7:
    b = pd.read_parquet(E / g / "raw/bus_data.parquet", columns=["scenario", "Vm", "PQ"])
    mn = b.groupby("scenario")["Vm"].transform("min")
    s = b[(mn >= 0.5) & (b.PQ.astype(bool))]
    mae = float(np.abs(s.Vm - 1.0).mean())
    out[g] = {"n_pq": int(len(s)), "flat1_mae": mae}
    tot += mae * len(s); n += len(s)
out["pooled"] = {"n_pq": n, "flat1_mae": tot / n}
json.dump(out, open(Path(__file__).parent / "flat_guess.json", "w"), indent=1)
print(json.dumps(out, indent=1))
