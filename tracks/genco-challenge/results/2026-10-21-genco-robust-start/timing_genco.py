"""Per-scenario cost of the released GENCO as a start (UNAUDITED). One grid per call; GPU 0; same recipe as
2026-10-10-timing-table/scripts/t_genco.py 'e2e' (in-memory tables -> graph -> normalise/mask -> collate -> H2D -> forward -> D2H),
at batch B (64, halved on OOM). Writes timing/<grid>.json: median per-scenario seconds (e2e and forward-only), batch used, load,
and a control that the in-memory forward reproduces graphkit's saved predictions (max |dVm|, |dVa|).
    CUDA_VISIBLE_DEVICES=0 python timing_genco.py <grid>"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
HERE = Path(__file__).resolve().parent
from genco_challenge import fewshot as fs
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.hetero_preprocess import build_hetero_data_for_scenario, merge_agg_gen_into_bus
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
from torch_geometric.data import Batch

REPO = fs.REPO
ROOT = REPO / "data/external/genco-robust-start"
grid = sys.argv[1]
torch.set_num_threads(4)

cfg = yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
cfg["data"]["networks"] = [grid]
cfg["data"]["scenarios"] = [json.loads((ROOT / grid / "meta.json").read_text())["scenarios_written"]]
cfg["data"]["workers"] = 0
args = NestedNamespace(**cfg)
dm = LitGridHeteroDataModule(args, str(ROOT), normalizer_stats_path=None)
dm.setup("test")
task = get_task(args, dm.data_normalizers)
task.load_state_dict(adapt_state_dict_for_model(torch.load(str(fs.PRETRAINED), map_location="cpu"), task))
task.eval().to("cuda")
ds, nrm = dm.datasets[0], dm.data_normalizers[0]
raw = ROOT / grid / "raw"
bus = pd.read_parquet(raw / "bus_data.parquet"); gen = pd.read_parquet(raw / "gen_data.parquet"); br = pd.read_parquet(raw / "branch_data.parquet")
bus_by, gen_by, br_by = (dict(tuple(x.groupby("scenario"))) for x in (bus, gen, br))
# the scenarios used by run_starts: first 100 ids in graphkit's predictions
run = max((Path(p).parent.parent for p in glob.glob(str(HERE / "mlruns/*/*/tags/mlflow.runName")) if Path(p).read_text().strip() == f"zeroshot__{grid}"), key=lambda p: p.stat().st_mtime)
pred = pd.read_parquet(run / "artifacts/test" / f"{grid}_predictions.parquet")
scen = sorted(int(s) for s in pred["scenario"].unique())[:100]


def convert(sids, parts=None):
    t0 = time.perf_counter()
    ds_ = [build_hetero_data_for_scenario(s, merge_agg_gen_into_bus(bus_by[s], gen_by[s]), gen_by[s], br_by[s]) for s in sids]
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


def fwd(b):
    with torch.no_grad():
        return task.forward(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)


def sync():
    torch.cuda.synchronize()


# control: in-memory forward == graphkit predictions
b = convert(scen[:4]).to("cuda"); o = fwd(b)["bus"].cpu().numpy()
n = len(bus_by[scen[0]]); ctrl = {}
p0 = pred[pred["scenario"] == scen[0]].sort_values("bus")
ctrl = {"max_abs_dVm": float(np.abs(o[:n, 0] - p0["Vm_pred"].to_numpy()).max()), "max_abs_dVa_rad": float(np.abs(o[:n, 1] - p0["Va_pred"].to_numpy()).max())}

B = 64
while True:
    try:
        comps = {k: [] for k in ("build", "norm_mask", "collate", "h2d", "fwd", "d2h", "total")}
        fo = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(2 + 6):
            sids = [scen[(i * B + j) % len(scen)] for j in range(B)]
            parts = []
            sync(); t0 = time.perf_counter()
            bb = convert(sids, parts)
            t1 = time.perf_counter(); bb = bb.to("cuda"); sync(); t2 = time.perf_counter()
            out = fwd(bb); sync(); t3 = time.perf_counter()
            out["bus"].cpu().numpy(); t4 = time.perf_counter()
            if i >= 2:
                for k, v in zip(comps, (*parts, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                    comps[k].append(v / B)
        break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); B //= 2
        if B < 1:
            raise
res = {"grid": grid, "n_bus": n, "batch": B, "per_scenario_s": {k: float(np.median(v)) for k, v in comps.items()},
       "forward_only_per_scenario_s": float(np.median(comps["fwd"])), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
       "loadavg": list(os.getloadavg()), "gpu": torch.cuda.get_device_name(0), "n_params": sum(p.numel() for p in task.parameters()),
       "control_inmemory_vs_graphkit": ctrl, "n_scenarios_used": len(scen)}
(HERE / "timing").mkdir(exist_ok=True)
(HERE / "timing" / f"{grid}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
