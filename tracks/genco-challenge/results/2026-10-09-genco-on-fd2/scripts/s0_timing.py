"""Wall time per scenario: fd2, nr1, GENCO forward (GPU batched 64, CPU batched 64), checkpoint load.
    FEWSHOT_INPUT_NORM=1 python s0_timing.py   (writes ../timing.jsonl)"""
import json, os, statistics, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse.linalg as spla
import torch, yaml
os.environ.setdefault("FEWSHOT_INPUT_NORM", "1")
from genco_challenge import fewshot as fs
from genco_challenge import hybrid as hy
from genco_challenge import labelfree_prior2 as lp2
from genco_challenge import gk_freeze  # noqa: F401  (plugins by env)
from genco_challenge.fresh_eval_data import EVAL2
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model

OUT = fs.REPO / "tracks/genco-challenge/results/2026-10-09-genco-on-fd2/timing.jsonl"
NS = 60


def classical(grid):
    bus, ys, brs = lp2._grid_tables(EVAL2, grid)
    scen = sorted(bus["scenario"].unique())[:NS]
    rows = {"setup": [], "dc": [], "fd2_iter": [], "nr1_iter": []}
    for rep in range(3):
        for s in scen:
            b = bus[bus["scenario"] == s].sort_values("bus")
            t0 = time.perf_counter(); sc = lp2.input_problem(b, ys[s], brs[s]); t1 = time.perf_counter()
            p = sc.sbus.real
            va = np.full(sc.n, sc.va_ref)
            va[sc.pvpq] = sc.va_ref + spla.spsolve(sc._bp, p[sc.pvpq]); t2 = time.perf_counter()
            v0 = sc.vm_set * np.exp(1j * sc.va_dc)
            v = v0.copy()
            for _ in range(2):
                v = sc.fdxb_step(v)
            t3 = time.perf_counter()
            hy.nr_ls_step(sc, v0); t4 = time.perf_counter()
            for k, d in zip(rows, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                rows[k].append(d)
    med = {k: statistics.median(v) * 1e3 for k, v in rows.items()}
    return {"grid": grid, "n_timed_scenarios": len(scen), "ms_setup": med["setup"], "ms_dc": med["dc"],
            "ms_fd2_iterations_incl_lu": med["fd2_iter"], "ms_nr1_iteration": med["nr1_iter"],
            "ms_fd2_total": med["setup"] + med["dc"] + med["fd2_iter"], "ms_nr1_total": med["setup"] + med["dc"] + med["nr1_iter"]}


def genco(grid, device):
    cfg = yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [json.loads((EVAL2 / grid / "meta.json").read_text())["scenarios_written"]]
    cfg["data"]["workers"] = 0
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, str(EVAL2), normalizer_stats_path=None)
    dm.setup("test")
    task = get_task(args, dm.data_normalizers)
    t0 = time.perf_counter()
    sd = torch.load(str(fs.PRETRAINED), map_location="cpu")
    sd = adapt_state_dict_for_model(sd, task)
    task.load_state_dict(sd)
    t_load = time.perf_counter() - t0
    task.eval().to(device)
    t0 = time.perf_counter()
    batches = list(dm.test_dataloader()[0])
    t_data = time.perf_counter() - t0
    n_g = sum(b.num_graphs for b in batches)
    batches = [b.to(device) for b in batches]
    def run():
        if device == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            for b in batches:
                task.forward(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)
        if device == "cuda": torch.cuda.synchronize()
        return time.perf_counter() - t
    run()  # warmup
    ts = [run() for _ in range(5)]
    return {"grid": grid, "device": device, "n_graphs": n_g, "batch_size": cfg["training"]["batch_size"],
            "ms_forward_per_scenario": statistics.median(ts) / n_g * 1e3, "s_ckpt_load": t_load,
            "ms_data_to_graph_per_scenario_cached_dataloader": t_data / n_g * 1e3}


if __name__ == "__main__":
    what = sys.argv[1]
    gpu = None
    if what == "gpu":
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        free = sorted(((int(f), i.strip()) for i, f in (ln.split(",") for ln in out.strip().splitlines())), reverse=True)
        print("using gpu", free[0])
    torch.set_num_threads(4)
    for g in fs.GRIDS:
        if what == "classical":
            r = classical(g)
        else:
            r = genco(g, "cuda" if what == "gpu" else "cpu")
        r |= {"kind": what, "loadavg": os.getloadavg(), "host": os.uname().nodename, "t": time.strftime("%F %T")}
        with OUT.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
