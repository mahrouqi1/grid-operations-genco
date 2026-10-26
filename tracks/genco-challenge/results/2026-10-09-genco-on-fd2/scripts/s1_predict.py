"""Stage 1 step 1: build validation sets, run zero-shot GENCO (graphkit evaluate, unchanged) on validation + eval2.
Predictions land in genco-ladder/genco_on_fd2/mlruns_eval. Run from repo root with PYTHONPATH=tracks/genco-challenge/src.
    python results/.../scripts/s1_predict.py val|eval2|eval2-defaultnorm
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from genco_challenge import fewshot as fs
from genco_challenge import labelfree as lf
from genco_challenge.fresh_eval_data import EVAL2

OUT = Path(__file__).resolve().parents[1]
W = fs.WORK / "genco_on_fd2"
VAL = W / "val"
DEV = lf.DEV


def build_val(grid):
    dst = VAL / grid
    if (dst / "meta.json").exists():
        return json.loads((dst / "val_build.json").read_text())
    rng = np.random.default_rng(20261009)
    pool = lf.pool_lf(grid)
    bad = fs.collapsed(lf.ORIG_TRAIN / grid / "raw/bus_data.parquet")
    pool2 = [s for s in pool if s not in bad]
    pick = sorted(rng.choice(pool2, size=200, replace=False).tolist())
    remap = {s: i for i, s in enumerate(pick)}
    (dst / "raw").mkdir(parents=True, exist_ok=True)
    src = lf.ORIG_TRAIN / grid / "raw"
    for f in ["bus_data", "gen_data", "branch_data", "y_bus_data", "runtime_data"]:
        d = pd.read_parquet(src / f"{f}.parquet")
        d = d[d["scenario"].isin(pick)].copy()
        d["scenario"] = d["scenario"].map(remap)
        if "scenario_partition" in d:                 # graphkit's AC/DC metrics filter on scenario // 200
            d["scenario_partition"] = d["scenario"] // 200
        d.sort_values("scenario", kind="stable").reset_index(drop=True).to_parquet(dst / "raw" / f"{f}.parquet")
    shutil.copy(src / "scenarios_agg_load_profile.parquet", dst / "raw")
    meta = json.loads((lf.ORIG_TRAIN / grid / "meta.json").read_text())
    meta |= {"scenarios_written": 200, "val_source_scenarios": pick}
    (dst / "meta.json").write_text(json.dumps(meta))
    info = {"grid": grid, "pool": len(pool), "pool_after_collapsed_filter": len(pool2), "picked": 200,
            "profile_idx_range": None}
    (dst / "val_build.json").write_text(json.dumps(info))
    return info


def pick_gpu(min_free=3500):
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = sorted(((int(f), i.strip()) for i, f in (ln.split(",") for ln in out.strip().splitlines())), reverse=True)
        if free and free[0][0] >= min_free:
            return free[0][1]
        time.sleep(60)


def evaluate_zs(grid, rec, input_norm):
    """fs.evaluate, but ALWAYS through genco_challenge.gk_freeze so FEWSHOT_INPUT_NORM takes effect (fs.evaluate only
    routes through the plugin module when slack_vm is on; the first zero-shot run of this study therefore used graphkit's
    default normalizer fit by mistake, and is kept as the 'defaultnorm' sensitivity set)."""
    meta = json.loads((fs.EVAL_DATA / grid / "meta.json").read_text())
    cfg = fs.yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [meta["scenarios_written"]]
    cfg_path = fs.FS / "configs" / f"eval_{grid}.yaml"
    cfg_path.write_text(fs.yaml.safe_dump(cfg))
    name = f"eval__{rec['key']}"
    mlruns = fs.FS / "mlruns_eval"
    cmd = [sys.executable, "-m", "genco_challenge.gk_freeze", "evaluate", "--config", str(cfg_path), "--data_path", str(fs.EVAL_DATA),
           "--model_path", rec["model"], "--compute_dc_ac_metrics", "--save_output", "--num_workers", "2",
           "--log_dir", str(mlruns), "--run_name", name]
    rc, sec = fs._gk(cmd, fs.FS / "logs" / f"{name}.log", fs._plugin_env(None, False) | {"FEWSHOT_INPUT_NORM": "1" if input_norm else "0"})
    if rc != 0:
        raise RuntimeError(f"evaluate {name} failed rc={rc}")
    art = fs._find_run(mlruns, name) / "artifacts/test"
    if not (art / f"{grid}_metrics.csv").exists():
        raise RuntimeError(f"evaluate {name}: no metrics")
    log = (fs.FS / "logs" / f"{name}.log").read_text()
    return {"eval_seconds": round(sec, 1), "slack_vm": False, "input_norm_plugin_seen": "input-only normalizer" in log} | fs._score(grid, art)


def run(root: Path, grids, tag, input_norm):
    fs.EVAL_DATA, fs.FS = root, W / f"fs_{tag}"
    (fs.FS / "configs").mkdir(parents=True, exist_ok=True)
    fs.ENV = {k: v for k, v in fs.ENV.items() if k != "FEWSHOT_INPUT_NORM"} | ({"FEWSHOT_INPUT_NORM": "1"} if input_norm else {})
    rec = {"key": f"zeroshot_{tag}", "model": str(fs.PRETRAINED), "normalizer_stats": None, "slack_vm": False}
    res = W / f"score_{tag}.jsonl"
    done = {json.loads(l)["grid"] for l in res.read_text().splitlines() if "error" not in json.loads(l)} if res.exists() else set()
    for g in grids:
        if g in done:
            continue
        row = {"grid": g, "tag": tag, "input_norm": input_norm}
        for attempt in range(4):
            try:
                fs.ENV = fs.ENV | {"CUDA_VISIBLE_DEVICES": pick_gpu()}
                rec2 = dict(rec, key=f"zeroshot_{tag}_{g}")
                row |= evaluate_zs(g, rec2, input_norm)
                row.pop("error", None)
                break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(60)
        with res.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("grid", "genco_active_res_mw", "eval_seconds", "error")}), flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "val":
        for g in DEV:
            print(build_val(g))
        run(VAL, DEV, "val", True)
    elif what == "eval2":
        run(EVAL2, fs.GRIDS, "eval2", True)

