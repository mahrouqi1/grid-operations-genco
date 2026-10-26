"""Few-shot fine-tuning of GENCO on held-out grids, scored on the ladder EVAL sets.

Stages (all resumable; each skips work whose output already exists):

``gen``      Build a SEPARATE training set per grid with GENCO's datakit: the grid's eval config
             (``genco-ladder/configs/<grid>.yaml``, itself datakit's ``default_pf.yaml`` with one
             topology variant per load scenario) with only the scenario count, a fixed seed and the
             output dir changed. Output: ``genco-ladder/train/<grid>/raw``.
``evalcopy`` Copy each grid's ladder eval ``raw/`` + ``meta.json`` (read only) to ``fewshot/eval_data``
             so graphkit processes it fresh: the ladder's case30 ``processed/`` predates its ``raw/``.
``disjoint`` Check train and eval share no operating point (seeds, aggregate-profile index,
             per-bus load vectors). Writes ``disjoint.json`` in the results dir.
``splits``   Nested n-sample subsets (n=10 within n=100 within n=1000) drawn from the train pool,
             split 80/10/10 the way graphkit's ``split_by_load_scenario_idx`` does.
``run``      Fine-tune (GENCO's ``gridfm_graphkit finetune`` with its case118_1000 recipe) or train
             from scratch, then evaluate on the grid's eval set exactly like ``ladder_eval``
             (graphkit ``evaluate --compute_dc_ac_metrics --save_output`` with the run's own
             normalizer stats, plus ``ladder_eval.errors`` for Vm/Va by bus type and the wrapped branch
             angle-difference error). ``--slack-vm`` trains AND evaluates with the slack |V| setpoint
             fed as a known input (``gk_freeze`` plugin, ``FEWSHOT_SLACK_VM=1``).
``rescore``  Re-score the v1 rows (``OLD_OUT``) from their saved predictions into ``OUT``.
``table`` / ``summary``  Per-run table; per (grid, variant, n) mean and min-max over sample draws.

The training pool excludes every scenario whose aggregate-load-profile index the eval set used
(datakit cuts the deterministic ``default`` profile to its first N points, so the eval set's 200
scenarios sit on profile indices 0..199): we generate 1,200 and keep indices >= 200.

Run (env ``grid-operations-genco``, from the repo root):

    python -m genco_challenge.fewshot gen --grids pglib_opf_case14_ieee
    python -m genco_challenge.fewshot disjoint
    python -m genco_challenge.fewshot splits
    python -m genco_challenge.fewshot run --modes finetune scratch --ns 10 100 1000
    python -m genco_challenge.fewshot run --variant lr1e-4 --set optimizer.learning_rate=1e-4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from genco_challenge.ladder_eval import errors

REPO = Path(__file__).resolve().parents[4]
WORK = REPO / "data/external/genco-ladder"
LADDER_DATA = WORK / "data"          # read only (another process owns it)
TRAIN_DATA = WORK / "train"
FS = WORK / "fewshot"
EVAL_DATA = FS / "eval_data"         # verbatim copy of each grid's ladder raw/ + meta.json, processed fresh
GK = REPO / "data/external/gridfm-graphkit-repro"
T = REPO / "data/external/genco-pf-transfer"
PRETRAINED = T / "pretraining/best_model_state_dict.pt"
RECIPE = T / "finetune/case118_1000/case118_1000.yaml"   # what GENCO ran for its 1,000-sample fine-tune
OUT = REPO / "tracks/genco-challenge/results/2026-10-07-genco-fewshot-v2"
OLD_OUT = REPO / "tracks/genco-challenge/results/2026-10-07-genco-fewshot"   # v1 rows, re-scored into OUT
ZEROSHOT = REPO / "tracks/genco-challenge/results/2026-10-07-genco-heldout-zeroshot/results.jsonl"

GRIDS = ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee", "pglib_opf_case39_epri", "pglib_opf_case57_ieee",
         "pglib_opf_case73_ieee_rts", "pglib_opf_case89_pegase", "pglib_opf_case118_ieee",
         "pglib_opf_case162_ieee_dtc", "pglib_opf_case300_ieee"]
# pglib_opf_case1354_pegase skipped: 69 of 97 eval labels and ~76% of generated training labels collapsed
NS = [10, 100, 1000]
N_TRAIN_GEN = 1200
TRAIN_SEED = 7007          # datakit global seed for every training set (eval runs drew random seeds)
SPLIT_SEED = 0
# datakit's PF labels can be a collapsed (low-voltage) root: zero-injection buses at Vm ~0 or below
# (hybrid-warmstart report section 7; case1354_pegase 69 of 97 eval scenarios). Such scenarios are
# dropped from training pools and refused in eval sets.
COLLAPSED_VM = 0.5


def collapsed(bus_parquet: Path) -> set[int]:
    """Scenario ids whose AC label has any bus with Vm < COLLAPSED_VM pu."""
    b = pd.read_parquet(bus_parquet, columns=["scenario", "Vm"])
    vmin = b.groupby("scenario")["Vm"].min()
    return {int(s) for s in vmin.index[vmin < COLLAPSED_VM]}
# Lightning writes a 241 MB last.ckpt every epoch; on NFS that dominated wall time (~4 s per one-batch
# epoch). Training logs go to local disk; best model, normalizer stats and metrics are copied to FS/runs.
LOCAL_MLRUNS = Path("<LOCAL_SCRATCH>/genco-fewshot/mlruns")
ENV = os.environ | {"MLFLOW_ALLOW_FILE_STORE": "true", "PYTHONNOUSERSITE": "1"}


def _nice(cmd: list[str]) -> list[str]:
    return ["nice", "-n", "10", *cmd]


# ----------------------------------------------------------------------------- data

def eval_seed(grid: str) -> int | None:
    log = WORK / "logs" / f"{grid}.log"
    for line in log.read_text().splitlines() if log.exists() else []:
        if "Using seed=" in line or "seed set to:" in line:
            return int(line.rsplit("=", 1)[-1].rsplit(":", 1)[-1].strip())
    return None


def gen(grid: str, procs: int) -> dict:
    raw = TRAIN_DATA / grid / "raw"
    meta_path = TRAIN_DATA / grid / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("status") == "ok":
        return json.loads(meta_path.read_text()) | {"cached": True}
    es = eval_seed(grid)
    if es == TRAIN_SEED:
        raise RuntimeError(f"{grid}: eval set used seed {es}, pick another TRAIN_SEED")
    cfg = yaml.safe_load((WORK / "configs" / f"{grid}.yaml").read_text())
    cfg["load"]["scenarios"] = N_TRAIN_GEN
    cfg["settings"].update(data_dir=str(TRAIN_DATA), num_processes=procs, seed=TRAIN_SEED,
                           large_chunk_size=200, enable_solver_logs=False)
    cfg_path = FS / "configs" / f"train_{grid}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    log = FS / "logs" / f"gen_{grid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(_nice(["gridfm_datakit", "generate", str(cfg_path)]), stdout=fh,
                            stderr=subprocess.STDOUT, env=ENV | {"JULIA_CPU_TARGET": "generic"}).returncode
    meta = {"grid": grid, "scenarios_requested": N_TRAIN_GEN, "seed": TRAIN_SEED, "eval_seed": es,
            "returncode": rc, "seconds": round(time.time() - t0, 1), "config": str(cfg_path.relative_to(REPO))}
    if rc == 0 and (raw / "bus_data.parquet").exists():
        b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
        meta.update(status="ok", scenarios_written=int(b["scenario"].nunique()))
    else:
        meta["status"] = "failed"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def eval_copy(grid: str) -> dict:
    import shutil
    src, dst = LADDER_DATA / grid, EVAL_DATA / grid
    if not (dst / "meta.json").exists():
        shutil.copytree(src / "raw", dst / "raw", dirs_exist_ok=True)
        shutil.copy(src / "meta.json", dst / "meta.json")
    b = pd.read_parquet(dst / "raw/bus_data.parquet", columns=["scenario"])
    meta = json.loads((dst / "meta.json").read_text())
    assert b["scenario"].nunique() == meta["scenarios_written"], grid
    bad = collapsed(dst / "raw/bus_data.parquet")
    if bad:   # graphkit scores every scenario in raw/: refuse rather than silently score collapsed labels
        raise RuntimeError(f"{grid}: {len(bad)} eval scenarios have collapsed labels (Vm < {COLLAPSED_VM})")
    return {"grid": grid, "scenarios": meta["scenarios_written"], "collapsed": 0}


def pool(grid: str) -> list[int]:
    """Training-set scenario ids whose aggregate-profile index the eval set never used."""
    ev = pd.read_parquet(EVAL_DATA / grid / "raw/bus_data.parquet", columns=["load_scenario_idx"])  # copy of ladder raw
    max_eval_idx = int(ev["load_scenario_idx"].max())
    tr = pd.read_parquet(TRAIN_DATA / grid / "raw/bus_data.parquet", columns=["scenario", "load_scenario_idx"])
    per = tr.groupby("scenario")["load_scenario_idx"].first()
    bad = collapsed(TRAIN_DATA / grid / "raw/bus_data.parquet")
    return sorted(int(s) for s, li in per.items() if li > max_eval_idx and int(s) not in bad)


def disjoint(grid: str) -> dict:
    cols = ["scenario", "load_scenario_idx", "bus", "Pd", "Qd"]
    ev = pd.read_parquet(EVAL_DATA / grid / "raw/bus_data.parquet", columns=cols)
    tr = pd.read_parquet(TRAIN_DATA / grid / "raw/bus_data.parquet", columns=cols)
    P = {k: d.pivot(index="scenario", columns="bus", values="Pd").to_numpy() for k, d in (("ev", ev), ("tr", tr))}
    keep = pool(grid)
    ptr = P["tr"][keep]
    pev = P["ev"]
    scale = np.abs(pev).sum(axis=1).mean()
    # minimum L1 distance between any training and any eval load vector, relative to total load
    d = np.abs(ptr[:, None, :] - pev[None, :, :]).sum(axis=2) / scale
    ev_idx = set(ev["load_scenario_idx"].astype(int))
    tr_idx = set(tr.loc[tr["scenario"].isin(keep), "load_scenario_idx"].astype(int))
    meta = json.loads((TRAIN_DATA / grid / "meta.json").read_text())
    return {"grid": grid, "train_seed": meta["seed"], "eval_seed": meta["eval_seed"],
            "collapsed_eval": len(collapsed(EVAL_DATA / grid / "raw/bus_data.parquet")),
            "collapsed_train_generated": len(collapsed(TRAIN_DATA / grid / "raw/bus_data.parquet")),
            "n_eval": int(pev.shape[0]), "n_train_generated": int(P["tr"].shape[0]), "n_train_pool": len(keep),
            "eval_profile_idx_range": [min(ev_idx), max(ev_idx)],
            "train_pool_profile_idx_range": [min(tr_idx), max(tr_idx)],
            "shared_profile_idx": len(ev_idx & tr_idx),
            "min_rel_L1_train_to_eval": float(d.min()),
            "median_nn_rel_L1_train_to_eval": float(np.median(d.min(axis=1))),
            "exact_duplicate_load_vectors": int((d < 1e-9).sum()),
            "total_load_mw_eval": [float(pev.sum(1).min()), float(pev.sum(1).max())],
            "total_load_mw_train_pool": [float(ptr.sum(1).min()), float(ptr.sum(1).max())]}


def split_dir(grid: str, n: int, seed: int = SPLIT_SEED) -> Path:
    return FS / "splits" / grid / (f"n{n}" if seed == SPLIT_SEED else f"n{n}_s{seed}")


def write_splits(grid: str, seed: int = SPLIT_SEED) -> dict:
    """Nested subsets: perm[:10] within perm[:100] within perm[:1000]; per-n split by graphkit's rule.
    ``seed`` changes which training scenarios are drawn (resampling check for small n)."""
    import torch
    ids = pool(grid)
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    out = {}
    for n in NS:
        sub = perm[:n]          # a few failed datakit scenarios can leave the pool just short of n
        n_act = len(sub)
        n_val, n_test = int(0.1 * n_act), int(0.1 * n_act)
        n_tr = n_act - n_val - n_test
        parts = {"train": sub[:n_tr], "val": sub[n_tr:n_tr + n_val], "test": sub[n_tr + n_val:]}
        d = split_dir(grid, n, seed)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in parts.items():
            if not (d / f"{k}.pt").exists():   # never rewrite a split a run may already be using
                torch.save(torch.tensor(v, dtype=torch.long), d / f"{k}.pt")
        out[n] = {"n_actual": n_act, **{k: len(v) for k, v in parts.items()}}
        (d / "split.json").write_text(json.dumps(out[n] | {"ids": parts}))
    return out


# ----------------------------------------------------------------------------- train + eval

def _set(cfg: dict, dotted: str) -> None:
    key, val = dotted.split("=", 1)
    node = cfg
    *path, last = key.split(".")
    for p in path:
        node = node[p]
    v = yaml.safe_load(val)
    if isinstance(v, str):      # YAML 1.1 reads "1e-4" as a string
        try:
            v = float(v)
        except ValueError:
            pass
    node[last] = v


def train_config(grid: str, n: int, overrides: list[str], seed: int = SPLIT_SEED) -> dict:
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [n]           # ignored by graphkit when split files are given
    cfg["data"].pop("split_by_load_scenario_idx", None)
    cfg["data"]["split_from_existing_files"] = str(split_dir(grid, n, seed))
    cfg["data"]["workers"] = 2
    for o in overrides:
        _set(cfg, o)
    return cfg


def _find_run(mlruns: Path, run_name: str) -> Path:
    runs = [Path(p).parent.parent for p in glob.glob(str(mlruns / "*/*/tags/mlflow.runName"))
            if Path(p).read_text().strip() == run_name]
    if not runs:
        raise FileNotFoundError(f"no mlflow run named {run_name} in {mlruns}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _csv(path: Path) -> dict:
    return {r["Metric"]: float(r["Value"]) for r in csv.DictReader(path.open())}


def _gk(cmd: list[str], log: Path, env: dict | None = None) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(_nice(cmd), cwd=GK, stdout=fh, stderr=subprocess.STDOUT, env=env or ENV).returncode
    return rc, time.time() - t0


REFERENCE = {   # evaluated as-is, no training on the target grid
    "zeroshot": (PRETRAINED, None),
    "ft118_1000": (T / "finetune/case118_1000/best_model_state_dict.pt", T / "finetune/case118_1000/normalizer_stats.pt"),
}


def train(grid: str, n: int, mode: str, variant: str, overrides: list[str], init: Path | None,
          trainable: str | None = None, seed: int = SPLIT_SEED, slack_vm: bool = False) -> dict:
    key = f"{variant}__{mode}__{grid}__n{n}"
    if mode in REFERENCE:
        mp, ns = REFERENCE[mode]
        return {"key": key, "model": str(mp), "normalizer_stats": ns and str(ns),
                "train_seconds": 0.0, "epochs_run": 0, "returncode": 0, "slack_vm": slack_vm}
    rec_path = FS / "runs" / f"{key}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    lock = FS / "runs" / f"{key}.lock"     # parallel per-grid pipelines must never train one key twice
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise RuntimeError(f"locked: {key} is being trained by another process ({lock})") from None
    try:
        return _train(grid, n, mode, variant, overrides, init, trainable, key, rec_path, seed, slack_vm)
    finally:
        lock.unlink(missing_ok=True)


def _train(grid, n, mode, variant, overrides, init, trainable, key, rec_path, seed, slack_vm=False) -> dict:
    cfg = train_config(grid, n, overrides, seed)
    cfg_path = FS / "configs" / f"{key}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    mlruns = LOCAL_MLRUNS
    mlruns.mkdir(parents=True, exist_ok=True)
    # additive (2026-10-09): FEWSHOT_FIX_SANITY=1 in the environment also routes full fine-tunes through the plugin
    plugin = trainable or slack_vm or os.environ.get("FEWSHOT_FIX_SANITY") == "1"
    exe = [sys.executable, "-m", "genco_challenge.gk_freeze"] if plugin else ["gridfm_graphkit"]
    cmd = [*exe, "train" if mode == "scratch" else "finetune", "--config", str(cfg_path),
           "--data_path", str(TRAIN_DATA), "--log_dir", str(mlruns), "--exp_name", "genco_fewshot",
           "--run_name", key, "--num_workers", "2"]
    if mode != "scratch":
        cmd += ["--model_path", str(init or PRETRAINED)]
    env = _plugin_env(trainable, slack_vm)
    rc, sec = _gk(cmd, FS / "logs" / f"train_{key}.log", env)
    import shutil
    run = _find_run(mlruns, key)
    src_model = run / "artifacts/model/best_model_state_dict.pt"
    src_stats = run / "artifacts/stats/normalizer_stats.pt"
    if rc != 0 or not (src_model.exists() and src_stats.exists()):
        # a crashed run (e.g. CUDA OOM on the shared GPUs) can still leave the checkpoint graphkit's
        # saver wrote during the pre-training sanity validation: never accept it
        shutil.rmtree(run, ignore_errors=True)
        raise RuntimeError(f"train {key} failed rc={rc}, see {FS / 'logs' / f'train_{key}.log'}")
    keep = FS / "runs" / key
    keep.mkdir(parents=True, exist_ok=True)
    model, stats = keep / src_model.name, keep / src_stats.name
    shutil.copy(src_model, model)
    shutil.copy(src_stats, stats)
    shutil.copytree(run / "metrics", keep / "metrics", dirs_exist_ok=True)
    for f in (run / "artifacts/stats").glob("*.json"):
        shutil.copy(f, keep / f.name)
    epochs = _epochs_run(run)
    shutil.rmtree(run)     # our own local run dir (last.ckpt etc.); what we need is now in FS/runs/<key>
    rec = {"key": key, "grid": grid, "n": n, "mode": mode, "variant": variant, "overrides": overrides,
           "split_seed": seed, "slack_vm": slack_vm,
           "trainable": trainable, "init": str(init or PRETRAINED) if mode != "scratch" else None,
           "returncode": rc, "train_seconds": round(sec, 1), "model": str(model), "normalizer_stats": str(stats),
           "config": str(cfg_path), "epochs_run": epochs}
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec


def _epochs_run(run: Path) -> int | None:
    f = run / "metrics" / "epoch"
    if not f.exists():
        return None
    vals = [float(line.split()[1]) for line in f.read_text().splitlines() if line.strip()]
    return int(max(vals)) + 1 if vals else None


def _plugin_env(trainable: str | None = None, slack_vm: bool = False) -> dict:
    return ENV | {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "FEWSHOT_TRAINABLE": trainable or "",
                  "FEWSHOT_SLACK_VM": "1" if slack_vm else "0"}


def _score(grid: str, art: Path) -> dict:
    """Metrics from a graphkit evaluate run's saved artefacts (no GPU): graphkit's residuals plus
    ``ladder_eval.errors`` (Vm/Va by bus type and the wrapped branch angle-difference error)."""
    pred = pd.read_parquet(art / f"{grid}_predictions.parquet")
    bus = pd.read_parquet(EVAL_DATA / grid / "raw/bus_data.parquet", columns=["scenario", "bus", "Va_dc", "Vm", "Pd"])
    # join check (as ladder_eval): graphkit's targets must equal raw/ (guards a stale processed/ cache)
    j = pred.merge(bus, on=["scenario", "bus"], how="left", suffixes=("", "_raw"))
    vm_gap = float((j["Vm_target"] - j["Vm"]).abs().max())
    pd_gap = float(((j["Pd"] - j["Pd_raw"]).abs() / np.maximum(j["Pd_raw"].abs(), 100.0)).max())
    if not (np.isfinite(vm_gap) and np.isfinite(pd_gap)) or vm_gap > 1e-4 or pd_gap > 1e-5:
        raise RuntimeError(f"{grid}: graphkit targets differ from raw (|V| {vm_gap:.3g}, Pd rel {pd_gap:.3g})")
    branch = pd.read_parquet(EVAL_DATA / grid / "raw/branch_data.parquet",
                             columns=["scenario", "from_bus", "to_bus", "br_status"])
    m = _csv(art / f"{grid}_metrics.csv")
    ref = _csv(art / f"{grid}_ac_dc_metrics.csv")
    n_bad = len(set(pred["scenario"].unique()) & collapsed(EVAL_DATA / grid / "raw/bus_data.parquet"))
    if n_bad:
        raise RuntimeError(f"{grid}: {n_bad} scored scenarios have collapsed labels")
    return {"n_test_scenarios": int(pred["scenario"].nunique()), "collapsed_scored": 0,
            "genco_active_res_mw": m["Avg. active res. (MW)"], "genco_reactive_res_mvar": m["Avg. reactive res. (MVar)"],
            "genco_pbe_max": m["PBE Max"], "dc_active_res_mw": ref["DC Avg. active res. (MW)"],
            "label_active_res_mw": ref["AC Avg. active res. (MW)"], "errors": errors(pred, bus, branch)}


def evaluate(grid: str, rec: dict) -> dict:
    meta = json.loads((EVAL_DATA / grid / "meta.json").read_text())
    cfg = yaml.safe_load((GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [meta["scenarios_written"]]
    cfg_path = FS / "configs" / f"eval_{grid}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    name = f"eval__{rec['key']}"
    mlruns = FS / "mlruns_eval"
    slack_vm = bool(rec.get("slack_vm"))
    exe = [sys.executable, "-m", "genco_challenge.gk_freeze"] if slack_vm else ["gridfm_graphkit"]
    cmd = [*exe, "evaluate", "--config", str(cfg_path), "--data_path", str(EVAL_DATA),
           "--model_path", rec["model"], "--compute_dc_ac_metrics", "--save_output", "--num_workers", "2",
           "--log_dir", str(mlruns), "--run_name", name]
    if rec["normalizer_stats"]:   # zero-shot has none: graphkit fits baseMVA on the eval set's 2-scenario train split
        cmd += ["--normalizer_stats", rec["normalizer_stats"]]
    rc, sec = _gk(cmd, FS / "logs" / f"{name}.log", _plugin_env(None, slack_vm))
    if rc != 0:
        raise RuntimeError(f"evaluate {name} failed rc={rc}")
    art = _find_run(mlruns, name) / "artifacts/test"
    if not (art / f"{grid}_metrics.csv").exists():
        raise RuntimeError(f"evaluate {name} rc={rc}: no metrics")
    return {"eval_seconds": round(sec, 1), "slack_vm": slack_vm} | _score(grid, art)


def _wait_gpu(min_free_mb: int = 6000, timeout_s: int = 3 * 3600) -> None:
    """Shared GPUs: wait until the visible GPU has ``min_free_mb`` free before launching (OOM avoidance)."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            if int(out.splitlines()[0]) >= min_free_mb:
                return
        except Exception:
            return
        time.sleep(60)


def run(args) -> None:
    global TRAIN_DATA
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = set(_latest_ok(res_path)) if res_path.exists() else set()
    if args.train_root:   # e.g. dcprior.DC_TRAIN: same scenarios and splits, different training targets
        TRAIN_DATA = args.train_root
    if args.slack_vm and "slackvm" not in args.variant:
        args.variant = f"{args.variant}_slackvm"
    if args.split_seed != SPLIT_SEED and not args.variant.endswith(f"_s{args.split_seed}"):
        args.variant = f"{args.variant}_s{args.split_seed}"
    for grid in args.grids:
        if args.split_seed != SPLIT_SEED:
            write_splits(grid, args.split_seed)
        for n in args.ns:
            for mode in args.modes:
                key = f"{args.variant}__{mode}__{grid}__n{n}"
                if key in done:
                    continue
                r: dict = {"key": key, "grid": grid, "n": n, "mode": mode, "variant": args.variant,
                           "overrides": args.set, "trainable": args.trainable, "split_seed": args.split_seed,
                           "slack_vm": args.slack_vm, "train_root": str(TRAIN_DATA)}
                for attempt in range(args.retries + 1):
                    if mode not in REFERENCE and not (FS / "runs" / f"{key}.json").exists():
                        _wait_gpu()
                    try:
                        rec = train(grid, n, mode, args.variant, args.set, args.init, args.trainable,
                                    args.split_seed, args.slack_vm)
                        r |= {k: rec[k] for k in ("train_seconds", "epochs_run", "returncode")}
                        r |= evaluate(grid, rec)
                        r.pop("error", None)
                        break
                    except Exception as exc:  # recorded, retried (shared GPUs: transient OOM happens)
                        r["error"] = f"{type(exc).__name__}: {exc}"
                        if "locked:" in r["error"]:
                            break
                        stale = FS / "runs" / f"{key}.json"
                        if "train" in r["error"] and stale.exists():
                            stale.unlink()
                        time.sleep(120)
                with res_path.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(json.dumps({k: r.get(k) for k in ("key", "train_seconds", "genco_active_res_mw",
                                                        "dc_active_res_mw", "error")}), flush=True)


def _latest_ok(path: Path) -> dict:
    """The last row per key, kept only if it succeeded (a later error/invalidation row voids it)."""
    out: dict = {}
    for r in map(json.loads, path.read_text().splitlines()):
        if "error" in r:
            out.pop(r["key"], None)
        else:
            out[r["key"]] = r
    return out


def ckpt_check(out_dir: Path | None = None, only_existing: bool = False) -> dict:
    """Per fine-tune run: is the selected checkpoint bit-identical to the pretrained one, and the
    last-layer validation residual (graphkit's selection metric) after epoch 1 and at its minimum.
    graphkit selects on ``Validation layer_11_residual`` and its saver also fires in Lightning's
    pre-training sanity validation, so a run that never beats the pretrained residual returns
    the pretrained weights."""
    import torch
    path = (out_dir or OUT) / "ckpt_check.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    pre = None
    for f in sorted((FS / "runs").glob("*.json")):
        rec = json.loads(f.read_text())
        # skip only rows that already carry the pretrained bound; older rows get the missing fields filled
        # in place (audit genco-adapt R2-F4). only_existing: refresh a frozen result dir without adding keys.
        if "pretrained_val_l11_bound" in out.get(rec["key"], {}) or rec["mode"] != "finetune":
            continue
        if only_existing and rec["key"] not in out:
            continue
        if not Path(rec["model"]).exists():   # checkpoint lost: leave any existing row as it was
            continue
        if pre is None:
            pre = torch.load(PRETRAINED, map_location="cpu")
        m = torch.load(rec["model"], map_location="cpu")
        same = all(torch.equal(m[k], pre[k]) for k in pre if k in m)
        met = f.with_suffix("") / "metrics" / "Validation layer_11_residual"
        row = {"selected_is_pretrained": same}
        if met.exists():
            v = [float(x.split()[1]) for x in met.read_text().splitlines()]
            row |= {"val_l11_epoch1": v[0], "val_l11_min": min(v), "val_l11_argmin": v.index(min(v))}
            # The pretrained model's own residual (Lightning's sanity validation, step 0) is NOT written to the
            # metric log, so it cannot be read back (audit genco-adapt F6, 2026-10-08). What the log does
            # bound: a kept-pretrained run has pretrained <= every logged epoch; any other run's selected epoch
            # beat it, so pretrained > val_l11_min.
            row |= {"pretrained_val_l11": None,
                    "pretrained_val_l11_bound": ("<=" if same else ">") + f" {min(v):.6g}"}
        out.setdefault(rec["key"], {}).update(row)
    path.write_text(json.dumps(out, indent=1))
    return out


def _n_actual(r: dict) -> int | str:
    if r["mode"] in REFERENCE:
        return "-"
    sp = split_dir(r["grid"], r["n"], r.get("split_seed") or SPLIT_SEED) / "split.json"
    return json.loads(sp.read_text())["n_actual"] if sp.exists() else r["n"]


def table() -> str:
    """Markdown table: per grid and run, power residual (MW), branch angle-difference error (deg,
    wrapped), slack-referenced Va MAE (deg) and |V| MAE (pu), each beside its ratio to DC."""
    best = _latest_ok(OUT / "results.jsonl")
    ck = json.loads((OUT / "ckpt_check.json").read_text()) if (OUT / "ckpt_check.json").exists() else {}
    lines = ["| grid | model | variant | n (actual) | P res. MW | / DC | branch dVa deg | / DC | Va MAE deg | / DC "
             "| Vm MAE pu | / DC | train s |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for grid in GRIDS:
        rs = sorted((r for r in best.values() if r["grid"] == grid),
                    key=lambda r: (r["mode"] not in REFERENCE, r["variant"], r["mode"], r["n"]))
        if not rs:
            continue
        r0 = rs[0]
        dc = r0["errors"]["dc"]
        lines.append(f"| {grid} | DC | | | {r0['dc_active_res_mw']:.3f} | 1 | {dc['branch_dva_mae_deg']:.3f} | 1 "
                     f"| {dc['va_mae_deg']:.3f} | 1 | {dc['vm_mae_pu']:.4f} | 1 | |")
        for r in rs:
            e, d = r["errors"]["genco"], r["errors"]["dc"]
            flag = " (sel.=pretrained)" if ck.get(r["key"], {}).get("selected_is_pretrained") else ""
            lines.append(
                f"| {grid} | {r['mode']}{flag} | {r['variant']} | {_n_actual(r)} "
                f"| {r['genco_active_res_mw']:.3f} | {r['genco_active_res_mw'] / r['dc_active_res_mw']:.2f} "
                f"| {e['branch_dva_mae_deg']:.3f} | {e['branch_dva_mae_deg'] / d['branch_dva_mae_deg']:.2f} "
                f"| {e['va_mae_deg']:.3f} | {e['va_mae_deg'] / d['va_mae_deg']:.2f} "
                f"| {e['vm_mae_pu']:.4f} | {e['vm_mae_pu'] / d['vm_mae_pu']:.2f} | {r.get('train_seconds', 0):.0f} |")
    return "\n".join(lines)


METRICS = (("res", lambda r: r["genco_active_res_mw"], lambda r: r["dc_active_res_mw"]),
           ("dva", lambda r: r["errors"]["genco"]["branch_dva_mae_deg"], lambda r: r["errors"]["dc"]["branch_dva_mae_deg"]),
           ("vm", lambda r: r["errors"]["genco"]["vm_mae_pu"], lambda r: r["errors"]["dc"]["vm_mae_pu"]))


def summary(variants: list[str] | None = None) -> list[dict]:
    """Per (grid, variant family, mode, n): mean, min and max over sample draws (variant suffix
    ``_s<seed>`` = draw) of each metric, absolute and as a ratio to DC, and the count of draws
    beating DC. Written to ``summary.json``."""
    import re
    best = _latest_ok(OUT / "results.jsonl")
    groups: dict = {}
    for r in best.values():
        fam = re.sub(r"_s\d+$", "", r["variant"])
        if variants and fam not in variants:
            continue
        groups.setdefault((r["grid"], fam, r["mode"], r["n"]), []).append(r)
    out = []
    for (grid, fam, mode, n), rs in sorted(groups.items(), key=lambda kv: (GRIDS.index(kv[0][0]) if kv[0][0] in GRIDS
                                                                            else 99, kv[0][1], kv[0][2], kv[0][3])):
        row = {"grid": grid, "variant": fam, "mode": mode, "n": n, "draws": len(rs),
               "keys": sorted(r["key"] for r in rs), "n_actual": sorted({str(_n_actual(r)) for r in rs})}
        for name, f, fdc in METRICS:
            v = np.array([f(r) for r in rs])
            dcv = np.array([fdc(r) for r in rs])
            row[name] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "dc": float(dcv.mean()),
                         "ratio_mean": float((v / dcv).mean()), "ratio_min": float((v / dcv).min()),
                         "ratio_max": float((v / dcv).max()), "beats_dc": int((v < dcv).sum())}
        out.append(row)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def summary_md(variants: list[str] | None = None) -> str:
    """Error vs n per grid: mean over draws (min to max) and the mean ratio to DC; k/d = draws beating DC."""
    rows = summary(variants)
    lines = ["| grid | variant | mode | n | draws | P res. MW (min-max) | / DC | k/d | branch dVa deg (min-max) | / DC "
             "| k/d | Vm MAE pu (min-max) | / DC | k/d |", "|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|"]
    last = None
    for r in rows:
        if r["grid"] != last:
            lines.append(f"| {r['grid']} | DC | | | | {r['res']['dc']:.3f} | 1 | | {r['dva']['dc']:.3f} | 1 | "
                         f"| {r['vm']['dc']:.4f} | 1 | |")
            last = r["grid"]
        cells = []
        for name, fmt in (("res", ".3f"), ("dva", ".3f"), ("vm", ".4f")):
            m = r[name]
            rng = f" ({m['min']:{fmt}}-{m['max']:{fmt}})" if r["draws"] > 1 else ""
            cells.append(f"{m['mean']:{fmt}}{rng} | {m['ratio_mean']:.2f} | {m['beats_dc']}/{r['draws']}")
        lines.append(f"| {r['grid']} | {r['variant']} | {r['mode']} | {r['n']} | {r['draws']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rescore() -> dict:
    """Re-score every successful v1 row (``OLD_OUT/results.jsonl``) from its saved evaluate artefacts
    (``FS/mlruns_eval``, no retraining, no GPU) with the branch angle-difference metric, and append
    the rows to ``OUT/results.jsonl`` (keys already there are skipped)."""
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    have = set(_latest_ok(res_path)) if res_path.exists() else set()
    old = _latest_ok(OLD_OUT / "results.jsonl")
    stats = {"rescored": 0, "skipped": 0, "max_abs_change_res": 0.0, "max_abs_change_va": 0.0}
    with res_path.open("a") as fh:
        for key, r in old.items():
            if key in have:
                stats["skipped"] += 1
                continue
            art = _find_run(FS / "mlruns_eval", f"eval__{key}") / "artifacts/test"
            new = _score(r["grid"], art)
            # the re-score must reproduce the v1 numbers it shares (same artefacts): a mismatch means
            # the newest eval run under that name is not the one v1 scored
            stats["max_abs_change_res"] = max(stats["max_abs_change_res"],
                                              abs(new["genco_active_res_mw"] - r["genco_active_res_mw"]))
            stats["max_abs_change_va"] = max(stats["max_abs_change_va"],
                                             abs(new["errors"]["genco"]["va_mae_deg"] - r["errors"]["genco"]["va_mae_deg"]))
            row = r | new | {"slack_vm": False, "rescored": "branch dVa from saved predictions (v2, 2026-10-07)"}
            fh.write(json.dumps(row) + "\n")
            stats["rescored"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--grids", nargs="+", default=GRIDS)
    g.add_argument("--procs", type=int, default=6)
    c = sub.add_parser("evalcopy")
    c.add_argument("--grids", nargs="+", default=GRIDS)
    d = sub.add_parser("disjoint")
    d.add_argument("--grids", nargs="+", default=GRIDS)
    s = sub.add_parser("splits")
    s.add_argument("--grids", nargs="+", default=GRIDS)
    sub.add_parser("table")
    sm = sub.add_parser("summary")
    sm.add_argument("--variants", nargs="*", default=None)
    sub.add_parser("rescore")
    sub.add_parser("ckptcheck")
    r = sub.add_parser("run")
    r.add_argument("--grids", nargs="+", default=GRIDS)
    r.add_argument("--ns", nargs="+", type=int, default=NS)
    r.add_argument("--modes", nargs="+", default=["finetune"], choices=["finetune", "scratch", *REFERENCE])
    r.add_argument("--variant", default="genco")
    r.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. optimizer.learning_rate=1e-4")
    r.add_argument("--init", type=Path, default=None, help="checkpoint to fine-tune from (default: GENCO pretrained)")
    r.add_argument("--trainable", default=None, help="comma-separated param-name regexes; freeze the rest")
    r.add_argument("--split-seed", type=int, default=SPLIT_SEED, help="resample which scenarios form each n")
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--train-root", type=Path, default=None,
                   help="training-data root other than genco-ladder/train (same scenario ids), e.g. train_dcprior")
    r.add_argument("--slack-vm", action="store_true",
                   help="feed the slack |V| setpoint as a known input (gk_freeze plugin, train and eval)")
    args = ap.parse_args()
    if args.cmd == "gen":
        for grid in args.grids:
            print(json.dumps(gen(grid, args.procs)), flush=True)
    elif args.cmd == "evalcopy":
        for grid in args.grids:
            print(json.dumps(eval_copy(grid)), flush=True)
    elif args.cmd == "disjoint":
        OUT.mkdir(parents=True, exist_ok=True)
        rows = [disjoint(grid) for grid in args.grids]
        (OUT / "disjoint.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows, indent=1))
    elif args.cmd == "splits":
        print(json.dumps({grid: write_splits(grid) for grid in args.grids}, indent=1))
    elif args.cmd == "ckptcheck":
        print(json.dumps(ckpt_check(), indent=1))
    elif args.cmd == "table":
        print(table())
    elif args.cmd == "summary":
        print(summary_md(args.variants))
    elif args.cmd == "rescore":
        print(json.dumps(rescore(), indent=1))
    else:
        run(args)


if __name__ == "__main__":
    main()
