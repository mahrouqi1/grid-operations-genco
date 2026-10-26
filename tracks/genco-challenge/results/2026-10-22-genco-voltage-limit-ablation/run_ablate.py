"""Limit/cost ablation of zero-shot GENCO |V| inference at 40% measured buses. CPU only. UNAUDITED. See protocol.json (written first).
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 python run_ablate.py --study control|fvs|oos --set pieces|seen_whole|unseen_whole [--grids g ...] [--arms A1 A2 A3] [--configs f40_s0 ..]
Each worker claims grids by mkdir-lock in ./locks; grids run smallest first. Reuses each study's run_genco_o.py machinery unchanged except the mask transform."""
import argparse, json, os, sys, time, datetime
from pathlib import Path
OUT = Path(__file__).resolve().parent
RES = OUT.parent
STUDY = {"control": RES / "2026-10-21-genco-voltage-inference-control", "fvs": RES / "2026-10-21-genco-voltage-inference-family-vs-size", "oos": RES / "2026-10-21-genco-voltage-inference-oos"}
ap = argparse.ArgumentParser()
ap.add_argument("--study", required=True, choices=list(STUDY)); ap.add_argument("--set", required=True)
ap.add_argument("--grids", nargs="*"); ap.add_argument("--arms", nargs="*", default=["A1", "A2", "A3"]); ap.add_argument("--configs", nargs="*", default=["f40_s0", "f40_s1", "f40_s2"])
ap.add_argument("--deadline", default="2026-10-22 07:30"); ap.add_argument("--worker", default="w"); ap.add_argument("--no-lock", action="store_true"); ap.add_argument("--nocap", action="store_true"); ap.add_argument("--cap", type=int, default=0)
ap.add_argument("--outdir", type=Path, default=OUT / "per_grid"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(STUDY[a.study]))
import run_genco_o as RO          # inserts the study dir + candidate 7 paths itself
import numpy as np, pandas as pd, torch, lightning as L
O, G, R, S, C, V = RO.O, RO.G, RO.R, RO.S, RO.C, RO.V
from gridfm_graphkit.datasets import globals as GG
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
DEADLINE = datetime.datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
ARM = {"name": "A0"}
_Base = RO.NoisyDetMask


class AblMask(_Base):
    """The studies' mask, then (arms A1-A3) overwrite generator limit / cost inputs. Originals are stashed on the object so a reused sample is never contaminated."""
    def forward(self, data):
        data = super().forward(data)
        gx = data.x_dict["gen"]
        orig = getattr(data["gen"], "_abl_orig", None)
        if orig is None: orig = gx.clone(); data["gen"]._abl_orig = orig
        new = orig.clone(); arm = ARM["name"]
        if arm in ("A1", "A3") and new.size(0):
            new[:, GG.MAX_PG] = new[:, GG.MAX_PG].median(); new[:, GG.MIN_PG] = 0.0
        if arm in ("A2", "A3"):
            new[:, GG.C0_H] = 0.0; new[:, GG.C1_H] = 0.0; new[:, GG.C2_H] = 0.0
        gx.copy_(new)
        return data


R.DetMask = AblMask
LOG = []
def nbus(g):
    return int(pd.read_parquet(G.ROOT / g / "raw/bus_data.parquet", columns=["bus"], filters=[("scenario", "==", 0)]).shape[0]) if a.study == "oos" else int(json.loads((STUDY[a.study] / "protocol.json").read_text())["subgrids"][g.replace("pglib_opf_", "")]["n_bus"] if False else json.loads((STUDY[a.study] / "protocol.json").read_text())["subgrids"][g]["n_bus"])
PROTO = json.loads((OUT / "protocol.json").read_text())
grids = a.grids or PROTO["sets"][a.set]["grids"]
ids = json.loads((STUDY[a.study] / "test_ids.json").read_text())
grids = [g for g in grids if g in ids]
nb = {g: nbus(g) for g in grids}; order = sorted(grids, key=lambda g: nb[g])
CAPS = json.loads((OUT / "caps.json").read_text()) if (OUT / "caps.json").exists() else {}   # protocol_amendment_1: {set: max buses}
cap = a.cap or (None if a.nocap else CAPS.get(a.set)); print(f"[{a.worker}] {a.set}: cap={cap}", flush=True)
if cap: order = [g for g in order if nb[g] <= cap]
a.outdir.mkdir(exist_ok=True); (OUT / "locks").mkdir(exist_ok=True); (OUT / "timing").mkdir(exist_ok=True)
CF = {c[0]: c for c in O.CONFIGS}
dev = torch.device("cpu")
for g in order:
    out = a.outdir / f"genco_ablate_{g}.jsonl"
    done = {(r["arm"], r["config"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [(arm, c) for arm in a.arms for c in a.configs if (arm, c) not in done]
    if not todo: continue
    if datetime.datetime.now() >= DEADLINE: print("deadline reached, stopping", flush=True); break
    tl = [json.loads(l) for l in (OUT / "timing" / "timing.jsonl").read_text().splitlines()] if (OUT / "timing" / "timing.jsonl").exists() else []
    rate = float(np.median([t["sec"] / (t["buses"] * t["n_scen"]) for t in tl[-30:]])) if tl else 1.0e-3
    est = 1.15 * rate * nb[g] * len(ids[g]["test"]) * len(todo)
    if datetime.datetime.now() + datetime.timedelta(seconds=est) > DEADLINE:
        print(f"[{a.worker}] {a.set} {g}: projected {est/3600:.2f} h passes the deadline; not started (remaining larger grids are skipped too)", flush=True); break
    if not a.no_lock:
        try: (OUT / "locks" / f"{a.set}__{g}").mkdir()
        except FileExistsError: continue
    n = ids[g]["n_scen"]; t_all = time.time(); L.seed_everything(0, workers=True)
    bs = a.__dict__.get("bs") or (4 if nb[g] > 5000 else 16 if nb[g] > 2000 else 64)
    cfg = S.make_cfg(g, n, bs=bs); dm = LitGridHeteroDataModule(cfg, str(G.ROOT)); dm.setup("test")
    key = "dm_test" if "dm_test" in ids[g] else "test"
    assert sorted(int(i) for i in dm.test_scenario_ids[0]) == ids[g][key]
    norm = dm.data_normalizers[0]; task = get_task(cfg, dm.data_normalizers)
    task.load_state_dict(adapt_state_dict_for_model(torch.load(S.CKPT, map_location="cpu"), task)); task.to(dev).eval()
    loader = dm.test_dataloader()[0]; tab = O.tables(g, ids[g][key]); keepset = set(ids[g]["test"])
    print(f"[{a.worker}] {a.set} {g} buses={nb[g]} n_scen={len(keepset)} setup {time.time()-t_all:.0f}s", flush=True)
    for arm, name in todo:
        _, frac, si, noisy = CF[name]; assert not noisy and abs(frac - 0.4) < 1e-9
        C.measured_set = V.measured_seeded(si); RO.NOISE.update(on=False, seed=si); ARM["name"] = arm; t0 = time.time()
        rows, diag = R.run_variant(task, loader, dev, norm, frac, tab)
        with out.open("a") as fh:
            for r in rows:
                if r["scenario"] in keepset: fh.write(json.dumps({"grid": g, "arm": arm, "config": name, "scenario": r["scenario"], "vm_mae_unmeas": r["vm_mae_unmeas"]}) + "\n")
        dt = time.time() - t0
        print(f"[{a.worker}] {g} {arm} {name} {len(rows)} scen {dt:.0f}s  vm_visible_dev={diag['vm_visible_dev']:.2e}", flush=True)
        with (OUT / "timing" / "timing.jsonl").open("a") as fh:
            fh.write(json.dumps({"set": a.set, "grid": g, "buses": nb[g], "arm": arm, "config": name, "n_scen": len(keepset), "sec": dt, "t": time.strftime("%H:%M:%S")}) + "\n")
    print(f"[{a.worker}] {g} finished in {time.time()-t_all:.0f}s", flush=True)
