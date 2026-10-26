"""D1: copy of ../2026-10-21-genco-voltage-inference/run_genco_v.py that ALSO saves per-bus |V| predictions. UNAUDITED.
run_variant below is a copy of ../2026-10-21-genco-state-inference/run_genco.py::run_variant with ONE addition: it keeps vm[mb_] per scenario.
CUDA_VISIBLE_DEVICES=<free gpu> python run_genco_pb.py"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-10-21-genco-state-inference"))
import vcommon as V
import run_genco as R
import numpy as np, torch, lightning as L
C = V.C; S = R.S; GG = R.GG
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
CFGS = ["f40_s0", "f20_s0"]


@torch.no_grad()
def run_variant_pb(task, loader, dev, norm, frac, tab):
    S.STATE["mask"] = "pre"; S.STATE["pre"] = R.DetMask(frac)
    task.model.physics_decoder = R.PHYSICS_DECODER_REGISTRY.create("OptimalPowerFlow").to(dev)
    rows, diag, keep = [], {"vm_visible_dev": 0.0, "va_ref_dev": 0.0, "n_gen_mismatch": 0, "pg_label_dev": 0.0}, {}
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
            assert mg_.sum() == ins.sum(), (s, int(mg_.sum()), int(ins.sum()))
            diag["pg_label_dev"] = max(diag["pg_label_dev"], float(np.abs(pg_t[mg_] - gen.p_mw.to_numpy()[ins]).max()))
            pg_full = np.zeros(len(gen)); pg_full[ins] = pg[mg_]
            diag["vm_visible_dev"] = max(diag["vm_visible_dev"], float(np.abs(vm[mb_] - vm_t[mb_])[~mk[mb_]].max()))
            diag["va_ref_dev"] = max(diag["va_ref_dev"], float(np.abs(va[mb_] - va_t[mb_])[ref[mb_]].max()))
            sc = C.score(tab, s, vm[mb_], va[mb_], pg_full, frac)
            rows.append({"scenario": s, **sc})
            keep[s] = vm[mb_].astype(np.float64)       # the one addition: per-bus predicted |V|, bus-index order
    return rows, diag, keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="*", default=C.HELDOUT15); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", type=Path, default=HERE / "perbus"); a = ap.parse_args(); a.outdir.mkdir(exist_ok=True)
    log = HERE / "genco_pb_mae.jsonl"
    ids = json.loads((V.SI / "test_ids.json").read_text()); dev = torch.device(a.device)
    for g in a.grids:
        cs = [c for c in V.CONFIGS if c[0] in CFGS and not (a.outdir / f"genco__{g}__{c[0]}.npz").exists()]
        if not cs: continue
        n = ids[g]["n_scen"]; L.seed_everything(0, workers=True)
        cfg = S.make_cfg(g, n, bs=16 if "3120" in g or "2736" in g else 64)
        dm = LitGridHeteroDataModule(cfg, str(R.ROOT)); dm.setup("test")
        assert sorted(int(i) for i in dm.test_scenario_ids[0]) == ids[g]["test"]
        norm = dm.data_normalizers[0]; task = get_task(cfg, dm.data_normalizers)
        task.load_state_dict(adapt_state_dict_for_model(torch.load(S.CKPT, map_location="cpu"), task)); task.to(dev).eval()
        loader = dm.test_dataloader()[0]; tab = C.tables(g, ids[g]["test"])
        for name, frac, si in cs:
            C.measured_set = V.measured_seeded(si)
            t0 = time.time()
            for attempt in range(6):
                try: rows, diag, keep = run_variant_pb(task, loader, dev, norm, frac, tab); break
                except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))
            else: raise RuntimeError("OOM")
            sc = ids[g]["test"]; assert sorted(keep) == sorted(sc)
            lab, meas, pq = [], [], []
            for s in sc:
                bus = tab["bus"][s]; nb = len(bus); m = np.zeros(nb, bool); m[C.measured_set(s, nb, frac)] = True
                lab.append(bus.Vm.to_numpy()); meas.append(m); pq.append(bus.PQ.to_numpy().astype(bool)); lo, hi = bus.min_vm_pu.to_numpy(), bus.max_vm_pu.to_numpy()
            np.savez_compressed(a.outdir / f"genco__{g}__{name}.npz", scen=np.array(sc, int), label=np.array(lab), meas=np.array(meas), pq=np.array(pq),
                                lo=lo, hi=hi, G=np.array([keep[s] for s in sc]))
            with log.open("a") as fh:
                for r in rows: fh.write(json.dumps({"grid": g, "config": name, "method": "G", "scenario": r["scenario"], "vm_mae_unmeas": r["vm_mae_unmeas"], "diag": diag}) + "\n")
            print(g, name, len(rows), f"{time.time()-t0:.0f}s", json.dumps(diag), flush=True)
