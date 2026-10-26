"""Released GENCO pretraining checkpoint, zero-shot, candidate 7's deterministic mask, for the OOS grids and all configs (incl. noisy |V|).
CUDA_VISIBLE_DEVICES=1 python run_genco_o.py --grids g1 g2 [--limit N] [--device cuda:0|cpu]. UNAUDITED."""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-21-genco-state-inference")); sys.path.insert(0, str(HERE.parent / "2026-10-08-zeroshot-shift-decomposition")); sys.path.insert(0, str(HERE))
import ocommon as O, grids as G
import run_genco as R   # candidate 7's module (its HERE dir is first on sys.path)
import numpy as np, torch, lightning as L
C, V, S = O.C, O.V, R.S
from gridfm_graphkit.datasets import globals as GG
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model

NOISE = {"on": False, "seed": 0}
_Base = R.DetMask


class NoisyDetMask(_Base):
    """Candidate 7's mask; when NOISE['on'], the visible |V| input gets Gaussian noise sigma = 0.5% of the true |V| (same z as the interpolators)."""
    def forward(self, data):
        data = super().forward(data)
        if NOISE["on"]:
            bx = data.x_dict["bus"]; n = bx.size(0); sid = int(data["scenario_id"].view(-1)[0])
            vis = torch.zeros(n, dtype=torch.bool); vis[torch.as_tensor(C.measured_set(sid, n, self.frac))] = True
            z = torch.as_tensor(O.noise_z(sid, n, NOISE["seed"]), dtype=bx.dtype)
            bx[vis, GG.VM_H] = bx[vis, GG.VM_H] * (1.0 + O.SIGMA_REL * z[vis])
        return data


R.DetMask = NoisyDetMask
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="+", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--configs", nargs="*"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
    ids = json.loads((HERE / "test_ids.json").read_text()); dev = torch.device(a.device)
    for g in a.grids:
        out = a.out or HERE / "per_grid" / f"genco_{g}.jsonl"; out.parent.mkdir(exist_ok=True)
        done = {json.loads(l)["config"] for l in out.read_text().splitlines()} if out.exists() else set()
        todo = [c for c in O.CONFIGS if (not a.configs or c[0] in a.configs) and c[0] not in done]
        if not todo: continue
        n = ids[g]["n_scen"]; L.seed_everything(0, workers=True)
        nb = int(__import__("pandas").read_parquet(G.ROOT / g / "raw/bus_data.parquet", columns=["bus"], filters=[("scenario", "==", 0)]).shape[0])
        bs = a.bs or (4 if nb > 5000 else 16 if nb > 2000 else 64)
        cfg = S.make_cfg(g, n, bs=bs); dm = LitGridHeteroDataModule(cfg, str(G.ROOT)); dm.setup("test")
        assert sorted(int(i) for i in dm.test_scenario_ids[0]) == ids[g]["test"]
        norm = dm.data_normalizers[0]; task = get_task(cfg, dm.data_normalizers)
        task.load_state_dict(adapt_state_dict_for_model(torch.load(S.CKPT, map_location="cpu"), task)); task.to(dev).eval()
        loader = dm.test_dataloader()[0]; sc = ids[g]["test"][: a.limit or None]; tab = O.tables(g, ids[g]["test"])
        for name, frac, si, noisy in todo:
            C.measured_set = V.measured_seeded(si); NOISE.update(on=noisy, seed=si); t0 = time.time()
            for attempt in range(6):
                try: rows, diag = R.run_variant(task, loader, dev, norm, frac, tab); break
                except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))
            else: raise RuntimeError("OOM")
            with out.open("a") as fh:
                for r in rows: fh.write(json.dumps({"grid": g, "config": name, "method": "G", "scenario": r["scenario"], "vm_mae_unmeas": r["vm_mae_unmeas"], "diag": diag}) + "\n")
            print(g, name, len(rows), f"{time.time()-t0:.0f}s", json.dumps(diag), flush=True)
