"""baseMVA (graphkit rule: 95th percentile of non-zero Pd, Qd, gen p_mw, Qg) per affected subgrid under original vs
corrected labels. Why: run_pretrain_mask / score_ckpts normalise with a normalizer fitted on the ORIGINAL-label tree,
but the corrected arm was trained with its own fitted baseMVA for these 9 subgrids. UNAUDITED."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[4] / "data/external"
out = {}
for d in sorted(p.name for p in (E / "genco-pf-pretraining-corrected").iterdir() if p.is_dir()):
    r = {}
    for tag, root in (("original", "genco-pf-pretraining"), ("corrected", "genco-pf-pretraining-corrected")):
        raw = E / root / d / "raw"
        b = pd.read_parquet(raw / "bus_data.parquet", columns=["Pd", "Qd", "Qg"])
        g = pd.read_parquet(raw / "gen_data.parquet", columns=["p_mw"])
        v = pd.concat([b.Pd[b.Pd != 0], b.Qd[b.Qd != 0], g.p_mw[g.p_mw != 0], b.Qg[b.Qg != 0]])
        r[tag] = float(np.percentile(v, 95))
    r["ratio_corrected_over_original"] = r["corrected"] / r["original"]
    out[d] = r
    print(d, r)
json.dump(out, open(Path(__file__).parent / "basemva_check.json", "w"), indent=1)
