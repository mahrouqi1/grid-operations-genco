"""Block G (UNAUDITED): GENCO timings and the pipelines that contain a GENCO forward. GPU 0 only, CPU work pinned by the caller.
    CUDA_VISIBLE_DEVICES=0 taskset -c 16-19 python t_genco.py <grid> [--ckpt tiny_s0|small_s0] [--reps 30] [--skip-cpu]
Checkpoint: ladder grids (case14..case1354) use the pretrained GENCO checkpoint zero-shot (cost does not depend on the weights; the
supervised corrector has the identical architecture, see report); goc2000/goc10000 use the released GOC checkpoint (--ckpt).
Rows (kind):
  model_load        once: datamodule setup + weights (s), plus first-CUDA-call s
  forward           (a) forward only, data resident, per-scenario ms = call / B, device + batch size, repeats >= reps after 5 warm-ups
  convert_gk        graphkit's shipped path for one scenario: parquet read + process_scenarios (writes .pt): reference only
  e2e               (b) at batch B>1: in-memory tables -> graph (merge, build, normalise+mask, collate) -> H2D -> forward -> D2H, per scenario
  pipeline          B=1 per-scenario pipelines with their components (conversion, h2d, fwd, d2h) AND the classical parts:
                    genco_only, corrector (= fd2 + conversion + forward + add), vmdc_newton (= DC + conversion + forward + Newton from GENCO |V| + DC angle)
"""
import argparse, os, sys, tempfile, shutil, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
os.environ.setdefault("FEWSHOT_INPUT_NORM", "1")
from tcommon import *
import t_classical as tc
from genco_challenge import hybrid as hy
from genco_challenge import labelfree_prior2 as lp2
from genco_challenge import fewshot as fs
from genco_challenge import gk_freeze  # noqa: F401  (input-only normaliser plugin, as in the earlier ladder runs)
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.hetero_preprocess import build_hetero_data_for_scenario, merge_agg_gen_into_bus, process_scenarios
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
from torch_geometric.data import Batch

WARM = 5
SCRATCH = Path(os.environ.get("TT_SCRATCH", tempfile.gettempdir()))


def load_model(grid):
    root, name, fam = GRIDS[grid]
    t0 = time.perf_counter()
    if fam == "goc":
        sys.path.insert(0, str(OUT.parent / "2026-10-10-goc-large-scoring/scripts"))
        import s4_timing as s4
        from common import checkpoints
        short = {"goc2000": "case2000_goc", "goc10000": "case10000_goc"}[grid]
        c = [c for c in checkpoints() if c["short"] == short and f"{c['variant']}_s{c['seed']}" == ARGS.ckpt][0]
        dm, task, _, _ = s4.load_model(c)
        ck = c["key"]
    else:
        cfg = yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
        cfg["data"]["networks"] = [name]
        cfg["data"]["scenarios"] = [json.loads((root / name / "meta.json").read_text())["scenarios_written"]]
        cfg["data"]["workers"] = 0
        args = NestedNamespace(**cfg)
        dm = LitGridHeteroDataModule(args, str(root), normalizer_stats_path=None)
        dm.setup("test")
        task = get_task(args, dm.data_normalizers)
        sd = adapt_state_dict_for_model(torch.load(str(fs.PRETRAINED), map_location="cpu"), task)
        task.load_state_dict(sd)
        ck = "pretrained_zeroshot"
    task.eval()
    return dm, task, ck, time.perf_counter() - t0


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def fwd(task, b):
    with torch.no_grad():
        return task.forward(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)


def main():
    global ARGS
    ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--ckpt", default="tiny_s0")
    ap.add_argument("--reps", type=int, default=30); ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--cpu-batches", nargs="*", type=int, default=[1, 16]); ap.add_argument("--gpu-batches", nargs="*", type=int, default=[1, 16, 64])
    ap.add_argument("--e2e-batches", nargs="*", type=int, default=[16, 64])
    ARGS = a = ap.parse_args()
    torch.set_num_threads(4)
    grid = a.grid
    root, name, fam = GRIDS[grid]
    tag = {"grid": grid, "n_bus": N_BUS[grid]}
    with Block("genco", **tag, ckpt=a.ckpt if fam == "goc" else "pretrained"):
        dm, task, ck, t_load = load_model(grid)
        t0 = time.perf_counter(); torch.zeros(1).cuda(); torch.cuda.synchronize(); t_cuda = time.perf_counter() - t0
        tag["ckpt"] = ck
        emit(kind="model_load", **tag, s_datamodule_and_weights=t_load, s_first_cuda_call=t_cuda, n_params=sum(p.numel() for p in task.parameters()),
             gpu=torch.cuda.get_device_name(0), torch_threads=torch.get_num_threads())
        ds, nrm = dm.datasets[0], dm.data_normalizers[0]
        raw = root / name / "raw"
        bus = pd.read_parquet(raw / "bus_data.parquet"); gen = pd.read_parquet(raw / "gen_data.parquet"); br = pd.read_parquet(raw / "branch_data.parquet")
        bus_by, gen_by, br_by = (dict(tuple(x.groupby("scenario"))) for x in (bus, gen, br))
        scen = sorted(bus_by)
        lp_bus, lp_y, lp_br = tc.load_tables(root, name)

        def convert(sids, parts=None):
            ds_ = []
            t0 = time.perf_counter()
            for s in sids:
                bdf = merge_agg_gen_into_bus(bus_by[s], gen_by[s])
                ds_.append(build_hetero_data_for_scenario(s, bdf, gen_by[s], br_by[s]))
            t1 = time.perf_counter()
            for d in ds_:
                nrm.transform(d)
            ds_ = [ds.transform(d) if ds.transform else d for d in ds_]
            t2 = time.perf_counter()
            b = Batch.from_data_list(ds_)
            t3 = time.perf_counter()
            if parts is not None:
                parts += [t1 - t0, t2 - t1, t3 - t2]
            return b

        # in-memory conversion must reproduce the shipped dataset item exactly
        s0 = scen[0]; ref = ds[s0]; mine = convert([s0])
        same = all(torch.allclose(ref.x_dict[k], mine.x_dict[k]) for k in ref.x_dict) and all(
            torch.allclose(ref.edge_attr_dict[k], mine.edge_attr_dict[k]) for k in ref.edge_attr_dict) and all(
            torch.equal(ref.mask_dict[k], mine.mask_dict[k]) for k in ref.mask_dict)
        emit(kind="convert_check", **tag, scenario=int(s0), in_memory_equals_dataset_item=bool(same))
        assert same, "in-memory conversion differs from the dataset item"

        def sids(i, B):
            return [scen[(i * B + j) % len(scen)] for j in range(B)]

        # ---- (a) forward only, data resident
        def fwd_block(dev, B):
            nb = a.reps + WARM
            batches = [convert(sids(i, B)) for i in range(min(nb, 8))]
            try:
                task.to(dev)
                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                db = [b.to(dev) for b in batches]
                ts = []
                t_first = None
                reps = a.reps
                i = 0
                while i < WARM + reps:
                    b = db[i % len(db)]
                    sync(dev); t = time.perf_counter(); fwd(task, b); sync(dev); dt = time.perf_counter() - t
                    if i == WARM:  # decide repeats from the first timed call
                        reps = a.reps if dt < 1.5 else (12 if dt < 12 else 6)
                    if i >= WARM:
                        ts.append(dt / B)
                    i += 1
                emit(kind="forward", **tag, device=dev, batch=B, repeats_target=a.reps, warmups=WARM,
                     peak_gpu_gb=(torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None),
                     **stats(ts))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                emit(kind="forward", **tag, device=dev, batch=B, oom=True)
            finally:
                task.to("cpu"); torch.cuda.empty_cache()

        for B in a.gpu_batches:
            fwd_block("cuda", B)
        if not a.skip_cpu:
            for B in a.cpu_batches:
                fwd_block("cpu", B)

        # ---- graphkit's shipped conversion path (reference): parquet read of one scenario + process_scenarios to disk
        td = tempfile.mkdtemp(dir=str(SCRATCH))
        try:
            rd, pr = [], []
            for i in range(WARM + 12):
                s = scen[i % len(scen)]
                flt = [("scenario", "in", [int(s)])]
                t0 = time.perf_counter()
                b_ = pd.read_parquet(raw / "bus_data.parquet", filters=flt); g_ = pd.read_parquet(raw / "gen_data.parquet", filters=flt)
                r_ = pd.read_parquet(raw / "branch_data.parquet", filters=flt)
                t1 = time.perf_counter()
                b_ = merge_agg_gen_into_bus(b_, g_)
                process_scenarios(b_, g_, r_, td, skip_existing=False, show_progress=False)
                t2 = time.perf_counter()
                if i >= WARM:
                    rd.append(t1 - t0); pr.append(t2 - t1)
            emit(kind="convert_gk", **tag, read_parquet_one_scenario=stats(rd), merge_build_save_pt=stats(pr),
                 total=stats([x + y for x, y in zip(rd, pr)]))
        finally:
            shutil.rmtree(td, ignore_errors=True)

        # ---- (b) end to end at batch > 1 (GPU)
        task.to("cuda")
        for B in a.e2e_batches:
            comps = {k: [] for k in ("build", "norm_mask", "collate", "h2d", "fwd", "d2h", "total")}
            try:
                for i in range(WARM + a.reps):
                    parts = []
                    sync("cuda"); t0 = time.perf_counter()
                    b = convert(sids(i, B), parts)
                    t1 = time.perf_counter(); b = b.to("cuda"); sync("cuda"); t2 = time.perf_counter()
                    out = fwd(task, b); sync("cuda"); t3 = time.perf_counter()
                    o = out["bus"].cpu().numpy(); t4 = time.perf_counter()
                    if i >= WARM:
                        for k, v in zip(comps, (*parts, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                            comps[k].append(v / B)
                emit(kind="e2e", **tag, device="cuda", batch=B, **{k: stats(v) for k, v in comps.items()})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); emit(kind="e2e", **tag, device="cuda", batch=B, oom=True)

        # ---- B=1 pipelines with classical parts
        comps = {k: [] for k in ("asm", "dc", "fd2", "build", "norm_mask", "collate", "h2d", "fwd", "d2h", "add", "newton_vmdc",
                                 "genco_only", "corrector", "vmdc_newton")}
        meta = {"it": [], "ok": [], "dv": []}
        for i in range(WARM + a.reps):
            s = scen[i % len(scen)]
            b_, y_, r_ = lp_bus[s], lp_y[s], lp_br[s]
            sc, t_asm = tc.assemble(b_, y_, r_)
            v_dc, t_dc = tc.dc_state(sc)
            v2, t_fd2 = tc.fd_iters(sc, v_dc, 2)
            parts = []
            sync("cuda"); t0 = time.perf_counter()
            b = convert([s], parts)
            t1 = time.perf_counter(); b = b.to("cuda"); sync("cuda"); t2 = time.perf_counter()
            out = fwd(task, b); sync("cuda"); t3 = time.perf_counter()
            o = out["bus"].cpu().numpy(); t4 = time.perf_counter()
            vm_c = np.abs(v2) + o[:, 0]; va_c = np.angle(v2) + o[:, 1]
            t5 = time.perf_counter()
            # Newton from GENCO |V| (PV/REF pinned to their setpoints) with the DC angle
            pvref = sc.is_t["PV"] | sc.is_t["REF"]
            vm0 = np.where(pvref, sc.vm_set, o[:, 0])
            lab = tc.label_of(b_)
            sc.v_label, sc.vm_label, sc.va_label = lab, np.abs(lab), np.angle(lab)
            sc.v_ref, sc.ref_kind = None, "input_only"
            t6 = time.perf_counter()
            w = hy.solve(sc, vm0 * np.exp(1j * sc.va_dc), "nr")
            t_nr = w["seconds"]
            if i < WARM:
                continue
            g_only = (t4 - t0)
            vals = dict(asm=t_asm, dc=t_dc, fd2=t_fd2, build=parts[0], norm_mask=parts[1], collate=parts[2], h2d=t2 - t1, fwd=t3 - t2,
                        d2h=t4 - t3, add=t5 - t4, newton_vmdc=t_nr, genco_only=g_only,
                        corrector=t_asm + t_dc + t_fd2 + g_only + (t5 - t4), vmdc_newton=t_asm + t_dc + g_only + t_nr)
            for k, v in vals.items():
                comps[k].append(v)
            meta["it"].append(w["iterations"]); meta["ok"].append(bool(w["converged"])); meta["dv"].append(w["max_dv_label"])
        emit(kind="pipeline", **tag, device="cuda", batch=1, warmups=WARM, **{k: stats(v) for k, v in comps.items()},
             vmdc_newton_iters_median=float(np.median(meta["it"])), vmdc_newton_converged_frac=float(np.mean(meta["ok"])),
             vmdc_newton_max_dv_label_median=float(np.nanmedian(meta["dv"])))
        # quality of the vmdc start is not the point here; convergence is recorded above


if __name__ == "__main__":
    main()
