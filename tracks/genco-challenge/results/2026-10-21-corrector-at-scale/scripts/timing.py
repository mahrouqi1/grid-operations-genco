"""Per-scenario wall time of the corrector pipeline vs fd_k, k = 2..30, on one large grid (UNAUDITED). Protocol: protocol.json 'timing'.
Reuses results/2026-10-10-timing-table/scripts (tcommon.Block/emit/stats, t_classical.assemble/dc_state, t_genco.load_model/convert logic).
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 taskset -c 16-19 python timing.py <case> [--reps 5] [--batch 64] [--smoke]
Rows go to $TIMING_JSONL (default ../timing.jsonl). One repeat =
  fd_k:      mean over the 64 scenarios of (assembly + DC + k fast-decoupled iterations, LU included), k = 1..30 (cumulative stamps)
  corrector: mean over the 64 scenarios of (assembly + DC + 2 iterations) + (GENCO end to end at batch B on the GPU) / 64 + add
  fwd_only:  GENCO forward on resident data at batch B, per scenario (lower bound on the GPU part)"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
TT = HERE.parents[1] / "2026-10-10-timing-table/scripts"
sys.path.insert(0, str(TT))
os.environ.setdefault("TIMING_JSONL", str(OUT / "timing.jsonl"))
os.environ.setdefault("FEWSHOT_INPUT_NORM", "1")
import tcommon as tcm
from tcommon import emit, Block, stats
import t_classical as tc
from genco_challenge import labelfree_prior2 as lp2
from genco_challenge import fewshot as fs
from genco_challenge import gk_freeze  # noqa: F401
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.hetero_preprocess import build_hetero_data_for_scenario, merge_agg_gen_into_bus
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
from torch_geometric.data import Batch

KMAX = 30
D = fs.REPO / "data/external/genco-corrector-scale"


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root, name = D / "trees/eval", a.case
    torch.set_num_threads(4)
    ids_file = OUT / "per_scenario" / f"{a.case}_n100.npz"
    scored = sorted(int(x) for x in np.load(ids_file)["ids"]) if ids_file.exists() else list(range(int(a.batch)))
    ids = scored[:64]
    tag = {"case": a.case, "smoke": a.smoke, "n_timed": len(ids)}
    with Block("corrector_at_scale", **tag):
        cfg = yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
        cfg["data"]["networks"] = [name]
        cfg["data"]["scenarios"] = [json.loads((root / name / "meta.json").read_text())["scenarios_written"]]
        cfg["data"]["workers"] = 0
        args = NestedNamespace(**cfg)
        dm = LitGridHeteroDataModule(args, str(root), normalizer_stats_path=None)
        dm.setup("test")
        task = get_task(args, dm.data_normalizers)
        task.load_state_dict(adapt_state_dict_for_model(torch.load(str(fs.PRETRAINED), map_location="cpu"), task))
        task.eval().to("cuda")
        ds, nrm = dm.datasets[0], dm.data_normalizers[0]
        raw = root / name / "raw"
        bus = pd.read_parquet(raw / "bus_data.parquet"); gen = pd.read_parquet(raw / "gen_data.parquet"); br = pd.read_parquet(raw / "branch_data.parquet")
        bus_by, gen_by, br_by = (dict(tuple(x.groupby("scenario"))) for x in (bus, gen, br))
        lp_bus, lp_y, lp_br = tc.load_tables(root, name)
        n_bus = len(lp_bus[ids[0]])

        def convert(sids):
            ds_ = []
            for s in sids:
                bdf = merge_agg_gen_into_bus(bus_by[s], gen_by[s])
                ds_.append(build_hetero_data_for_scenario(s, bdf, gen_by[s], br_by[s]))
            for d in ds_:
                nrm.transform(d)
            ds_ = [ds.transform(d) if ds.transform else d for d in ds_]
            return Batch.from_data_list(ds_)

        def fwd(b):
            with torch.no_grad():
                return task.forward(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)

        B = a.batch
        while True:        # largest batch (<= requested) that fits the GPU
            try:
                b = convert(ids[:B]).to("cuda"); fwd(b); sync(); del b; torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); B //= 2
        emit(kind="setup", **tag, n_bus=n_bus, batch_used=B, gpu=torch.cuda.get_device_name(0),
             gpu_state=os.popen("nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader").read().strip(),
             affinity=sorted(os.sched_getaffinity(0)))
        chunks = [ids[i:i + B] for i in range(0, len(ids), B)]

        def cpu_fd():
            asm, dc, cum = [], [], []
            for s in ids:
                sc, t_asm = tc.assemble(lp_bus[s], lp_y[s], lp_br[s])
                v, t_dc = tc.dc_state(sc)
                sc._bp_lu = sc._bpp_lu = None
                t0 = time.perf_counter(); c = []
                for _ in range(KMAX):
                    v = sc.fdxb_step(v); c.append(time.perf_counter() - t0)
                asm.append(t_asm); dc.append(t_dc); cum.append(c)
            return np.array(asm), np.array(dc), np.array(cum)      # (S,), (S,), (S,KMAX)

        def gpu_e2e():
            t0 = time.perf_counter()
            for ch in chunks:
                b = convert(ch).to("cuda"); sync()
                o = fwd(b); sync()
                o["bus"].cpu().numpy()
            return (time.perf_counter() - t0) / len(ids)

        def gpu_fwd_only():
            dev = [convert(ch).to("cuda") for ch in chunks]
            sync(); t0 = time.perf_counter()
            for b in dev:
                fwd(b)
            sync()
            return (time.perf_counter() - t0) / len(ids)

        cpu_fd(); gpu_e2e(); gpu_fwd_only()       # warm-up pass, untimed
        rows = []
        for r in range(a.reps):
            l0 = os.getloadavg()
            g0 = os.popen("nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader").read().strip()
            asm, dc, cum = cpu_fd()
            e2e, fo = gpu_e2e(), gpu_fwd_only()
            t_add = 1e-4 * 0 + 0.0
            tfd = asm.mean() + dc.mean() + cum.mean(axis=0)          # seconds, k = 1..KMAX
            rec = {"rep": r, "gpu_before": g0, "gpu_after": os.popen("nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader").read().strip(), "loadavg_before": l0, "loadavg_after": os.getloadavg(), "asm_ms": asm.mean() * 1e3, "dc_ms": dc.mean() * 1e3,
                   "fd_total_ms": (tfd * 1e3).tolist(), "gpu_e2e_ms": e2e * 1e3, "gpu_fwd_only_ms": fo * 1e3,
                   "corrector_ms": (tfd[1] + e2e) * 1e3, "corrector_fwd_only_ms": (tfd[1] + fo) * 1e3}
            emit(kind="repeat", **tag, **rec)
            rows.append(rec)
        med = lambda k: float(np.median([x[k] for x in rows]))
        fd_med = np.median(np.array([x["fd_total_ms"] for x in rows]), axis=0).tolist()
        emit(kind="summary", **tag, n_bus=n_bus, batch_used=B, reps=a.reps, corrector_ms=med("corrector_ms"),
             corrector_range=[min(x["corrector_ms"] for x in rows), max(x["corrector_ms"] for x in rows)],
             corrector_fwd_only_ms=med("corrector_fwd_only_ms"), gpu_e2e_ms=med("gpu_e2e_ms"), gpu_fwd_only_ms=med("gpu_fwd_only_ms"),
             fd_total_ms_median=fd_med,
             fd_total_ms_range=[np.min([x["fd_total_ms"] for x in rows], axis=0).tolist(), np.max([x["fd_total_ms"] for x in rows], axis=0).tolist()])


if __name__ == "__main__":
    main()
