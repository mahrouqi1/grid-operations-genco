"""Run the RELEASED GENCO pretraining checkpoint (no training) under the deterministic state-inference mask. UNAUDITED.

    CUDA_VISIBLE_DEVICES=1 nice -n 10 python run_genco.py [--grids ...] [--variants det20 det100]

det20  = the task: Pg masked at every gen, Qg masked, Vm visible at the seeded 20% of buses, Va masked except REF (visible),
         Pd/Qd/flags/limits/costs visible, branch P/Q masked (same per-feature semantics as graphkit AddPretrainMask, but
         deterministic and with p_low = 0 for the 'other features').
det100 = control: identical but Vm visible at every bus.
Decoder: the OPF decoder of pretraining (as results/2026-10-09-collapsed-label-pretrain-mask). One pass per variant (the mask
is deterministic, so there is no draw-to-draw variance).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-08-zeroshot-shift-decomposition"))
import numpy as np, pandas as pd, torch, lightning as L
from torch_geometric.transforms import BaseTransform
import common as C
import shift_eval as S
from gridfm_graphkit.datasets import globals as GG
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model

ROOT = C.EXT / "genco-state-inference"


class DetMask(BaseTransform):
    def __init__(self, frac):
        super().__init__(); self.frac = frac

    def forward(self, data):
        bx, gx = data.x_dict["bus"], data.x_dict["gen"]
        n = bx.size(0)
        sid = int(data["scenario_id"].view(-1)[0])
        mb = torch.zeros_like(bx, dtype=torch.bool); mg = torch.zeros_like(gx, dtype=torch.bool)
        vis = torch.zeros(n, dtype=torch.bool); vis[torch.as_tensor(C.measured_set(sid, n, self.frac))] = True
        pq, pv, ref = bx[:, GG.PQ_H] == 1, bx[:, GG.PV_H] == 1, bx[:, GG.REF_H] == 1
        mb[:, GG.VM_H] = ~vis
        mb[:, GG.VA_H] = ~ref                  # REF Va visible (at its label value); every other angle masked
        mb[:, GG.QG_H] = True                  # Qg always masked
        mg[:, GG.PG_H] = True                  # Pg masked at every generator
        be = data.edge_attr_dict[("bus", "connects", "bus")]
        mbr = torch.zeros_like(be, dtype=torch.bool); mbr[:, GG.P_E] = True; mbr[:, GG.Q_E] = True
        data.mask_dict = {"bus": mb, "gen": mg, "branch": mbr, "PQ": pq, "PV": pv, "REF": ref}
        return data


@torch.no_grad()
def run_variant(task, loader, dev, norm, frac, tab):
    S.STATE["mask"] = "pre"; S.STATE["pre"] = DetMask(frac)
    task.model.physics_decoder = PHYSICS_DECODER_REGISTRY.create("OptimalPowerFlow").to(dev)
    rows, diag = [], {"vm_visible_dev": 0.0, "va_ref_dev": 0.0, "n_gen_mismatch": 0, "pg_label_dev": 0.0}
    for batch in loader:
        batch = batch.to(dev)
        out = task.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
        norm.inverse_transform(batch); norm.inverse_output(out, batch)
        bb = batch.batch_dict["bus"]; sid_b = batch["scenario_id"][bb].cpu().numpy()
        gb = batch["gen"].batch; sid_g = batch["scenario_id"][gb].cpu().numpy()
        vm = out["bus"][:, 0].cpu().numpy(); va = np.degrees(out["bus"][:, 1].cpu().numpy())
        pg = out["gen"][:, 0].cpu().numpy(); pg_t = batch.y_dict["gen"][:, GG.PG_H].cpu().numpy()
        mk = batch.mask_dict["bus"][:, GG.VM_H].cpu().numpy()
        vm_t = batch.y_dict["bus"][:, GG.VM_H].cpu().numpy()
        va_t = np.degrees(batch.y_dict["bus"][:, GG.VA_H].cpu().numpy())
        ref = batch.mask_dict["REF"].cpu().numpy()
        for s in np.unique(sid_b):
            s = int(s); mb_, mg_ = sid_b == s, sid_g == s
            gen = tab["gen"][s]
            ins = gen.in_service.to_numpy() == 1
            assert mg_.sum() == ins.sum(), (s, int(mg_.sum()), int(ins.sum()))   # graph carries the in-service gens, in idx order
            diag["pg_label_dev"] = max(diag["pg_label_dev"], float(np.abs(pg_t[mg_] - gen.p_mw.to_numpy()[ins]).max()))
            pg_full = np.zeros(len(gen)); pg_full[ins] = pg[mg_]
            diag["vm_visible_dev"] = max(diag["vm_visible_dev"], float(np.abs(vm[mb_] - vm_t[mb_])[~mk[mb_]].max()))
            diag["va_ref_dev"] = max(diag["va_ref_dev"], float(np.abs(va[mb_] - va_t[mb_])[ref[mb_]].max()))
            sc = C.score(tab, s, vm[mb_], va[mb_], pg_full, frac)
            rows.append({"scenario": s, **sc})
    return rows, diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=C.HELDOUT15)
    ap.add_argument("--variants", nargs="*", default=["det20", "det100"])
    ap.add_argument("--out", type=Path, default=HERE / "genco_per_scenario.jsonl")
    a = ap.parse_args()
    ids = json.loads((HERE / "test_ids.json").read_text())
    dev = torch.device("cuda:0")
    done = {(r["grid"], r["method"]) for r in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
    for g in a.grids:
        todo = [v for v in a.variants if (g, "genco_" + v) not in done]
        if not todo:
            continue
        n = ids[g]["n_scen"]
        L.seed_everything(0, workers=True)
        cfg = S.make_cfg(g, n, bs=16 if "3120" in g or "2736" in g else 64)
        dm = LitGridHeteroDataModule(cfg, str(ROOT)); dm.setup("test")
        assert sorted(int(i) for i in dm.test_scenario_ids[0]) == ids[g]["test"]
        norm = dm.data_normalizers[0]; task = get_task(cfg, dm.data_normalizers)
        task.load_state_dict(adapt_state_dict_for_model(torch.load(S.CKPT, map_location="cpu"), task)); task.to(dev).eval()
        loader = dm.test_dataloader()[0]
        tab = C.tables(g, ids[g]["test"])
        for v in todo:
            t0 = time.time()
            for attempt in range(6):
                try:
                    rows, diag = run_variant(task, loader, dev, norm, 1.0 if v == "det100" else C.FRAC, tab); break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))
            else:
                raise RuntimeError("OOM x6")
            with a.out.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps({"grid": g, "method": "genco_" + v, **r, "diag": diag}) + "\n")
            print(g, v, len(rows), f"{time.time()-t0:.0f}s", json.dumps(diag), flush=True)
