"""Run graphkit's datamodule on each staged tree (processes it) and write the test-scenario ids (test_ratio 0.99). CPU. UNAUDITED."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-10-08-zeroshot-shift-decomposition"))
import ocommon as O, grids as G
import shift_eval as S
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
import pandas as pd, lightning as L
out = HERE / "test_ids.json"
for g in sys.argv[1:]:
    ids = json.loads(out.read_text()) if out.exists() else {}
    if g in ids: continue
    n = int(pd.read_parquet(G.ROOT / g / "raw/runtime_data.parquet").scenario.nunique())
    L.seed_everything(0, workers=True)
    dm = LitGridHeteroDataModule(S.make_cfg(g, n), str(G.ROOT)); dm.setup("test")
    ids = json.loads(out.read_text()) if out.exists() else {}
    dmt = sorted(int(i) for i in dm.test_scenario_ids[0]); ver = set(json.loads((HERE / "verified.json").read_text())[g]["pass_ids"])
    ids[g] = {"n_scen": n, "dm_test": dmt, "test": [i for i in dmt if i in ver]}
    out.write_text(json.dumps(ids)); print(g, n, len(dmt), len(ids[g]["test"]), flush=True)
