"""GENCO zero-shot (as candidate 7, same DetMask/inference) under each measurement config. UNAUDITED.
CUDA_VISIBLE_DEVICES=1 python run_genco_v.py"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-10-21-genco-state-inference"))
import vcommon as V
import run_genco as R   # candidate 7's module (imports common as C, same module object as V.C)
import torch, lightning as L
C = V.C; S = R.S
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="*", default=C.HELDOUT15); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", type=Path, default=HERE / "genco_v_per_scenario.jsonl"); a = ap.parse_args()
ids = json.loads((V.SI / "test_ids.json").read_text()); dev = torch.device(a.device)
done = {(r["grid"], r["config"]) for r in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
for g in a.grids:
    todo = [c for c in V.CONFIGS if (g, c[0]) not in done]
    if not todo: continue
    n = ids[g]["n_scen"]; L.seed_everything(0, workers=True)
    cfg = S.make_cfg(g, n, bs=16 if "3120" in g or "2736" in g else 64)
    dm = LitGridHeteroDataModule(cfg, str(R.ROOT)); dm.setup("test")
    assert sorted(int(i) for i in dm.test_scenario_ids[0]) == ids[g]["test"]
    norm = dm.data_normalizers[0]; task = get_task(cfg, dm.data_normalizers)
    task.load_state_dict(adapt_state_dict_for_model(torch.load(S.CKPT, map_location="cpu"), task)); task.to(dev).eval()
    loader = dm.test_dataloader()[0]; tab = C.tables(g, ids[g]["test"])
    for name, frac, si in todo:
        C.measured_set = V.measured_seeded(si)      # patches the mask (DetMask) and C.score alike
        t0 = time.time()
        for attempt in range(6):
            try: rows, diag = R.run_variant(task, loader, dev, norm, frac, tab); break
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))
        else: raise RuntimeError("OOM")
        with a.out.open("a") as fh:
            for r in rows: fh.write(json.dumps({"grid": g, "config": name, "method": "G", "scenario": r["scenario"], "vm_mae_unmeas": r["vm_mae_unmeas"], "diag": diag}) + "\n")
        print(g, name, len(rows), f"{time.time()-t0:.0f}s", json.dumps(diag), flush=True)
