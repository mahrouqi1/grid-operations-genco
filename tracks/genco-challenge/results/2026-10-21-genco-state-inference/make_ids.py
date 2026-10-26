"""Stage the 15 locked eval trees (symlinked raw/, fresh processed/ in data/external/genco-state-inference) and write the
graphkit test-scenario ids (same cfg/seed as the 15-grid scoring: shift_eval.make_cfg). CPU only. UNAUDITED."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-08-zeroshot-shift-decomposition"))
import common as C
import shift_eval as S
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
import pandas as pd, lightning as L

ROOT = C.EXT / "genco-state-inference"
out = HERE / "test_ids.json"
ids = json.loads(out.read_text()) if out.exists() else {}
for g in (sys.argv[1:] or C.HELDOUT15):
    if g in ids:
        continue
    d = ROOT / g; d.mkdir(parents=True, exist_ok=True)
    if not (d / "raw").exists():
        (d / "raw").symlink_to((C.LAD / "data" / g / "raw").resolve())
    n = int(pd.read_parquet(d / "raw/bus_data.parquet", columns=["scenario"]).scenario.nunique())
    L.seed_everything(0, workers=True)
    dm = LitGridHeteroDataModule(S.make_cfg(g, n), str(ROOT)); dm.setup("test")
    ids[g] = {"n_scen": n, "test": sorted(int(i) for i in dm.test_scenario_ids[0])}
    print(g, n, len(ids[g]["test"]), flush=True)
    out.write_text(json.dumps(ids))
