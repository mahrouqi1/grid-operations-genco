"""Label-free GENCO adaptation with hyperparameters selected on VALIDATION signals only.

The v2 label-free recipe (DC-prior weight 1, lr 1e-4, all parameters) was picked after looking at
EVAL numbers: v2 scored label-free variants (frozen trunk vs all parameters, weak vs strong DC prior,
physics-only) on the eval sets of all 9 grids, and its particular weight/lr were chosen after seeing
case14 eval numbers. This driver removes the leakage from the HYPERPARAMETER choice (validation signals
on the two dev grids only), but the method FAMILY it searches (prior weight x frozen/all) was shaped by
those v2 eval-set comparisons, and its ``final`` step scores on the same eval sets. So the numbers here
are on eval sets re-used from v2, not on held-out ones. Audit 2026-10-08 genco-adapt F1; the fix is
the fresh eval sets of ``results/2026-10-08-adapt-fresh-eval/`` (``fresh_eval.py``), which re-score
these checkpoints without retraining. This driver:

``protocol``  Write the hyperparameter grid and the selection rule to the results dir (before any run).
``pools``     Input-only training pools (no AC label read) and the input-only normalizer, compared
              with the v2 label-reading versions. Writes ``pools.json``.
``splits``    Nested splits for draws 0, 1, 2 from the input-only pool (``fewshot/splits_lf``).
``train``     Train a list of configs on a list of grids/draws/n WITHOUT evaluating on the eval set
              (selection runs never produce an eval number). Several workers may share one key list:
              a per-key lock in ``fewshot.train`` makes each key train once.
``select``    Selection table from the dev grids' validation metrics only; writes
              ``selection.json`` and ``selection_table.csv``.
``final``     Train (if needed) and evaluate the selected configuration on every grid, draws 0-2,
              n = 1000 and 100; appends ``results.jsonl``.
``summary``   ``summary.json``: eval metrics per grid and n (mean, min-max over draws, ratio to DC),
              dev and untouched grids separately, and the v2 provisional numbers beside them.

Every run: slack |V| fed as a known input (``--slack-vm``), training targets = DC prior
(``dcprior.DC_TRAIN``; the physics loss is label-free, the MSE term pulls toward the DC state),
normalizer fit on inputs only (``gk_freeze`` ``FEWSHOT_INPUT_NORM=1``), training pool filtered by an
input-only criterion. The AC solution is read only by ``fewshot.evaluate`` to score the eval set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from genco_challenge import fewshot as fs
from genco_challenge.dcprior import DC_TRAIN

OUT = fs.REPO / "tracks/genco-challenge/results/2026-10-08-genco-labelfree-clean"
V2 = fs.REPO / "tracks/genco-challenge/results/2026-10-07-genco-fewshot-v2/results.jsonl"
ORIG_TRAIN = fs.WORK / "train"            # same inputs as DC_TRAIN; only Va/Vm labels differ
SPLITS = fs.FS / "splits_lf"
DEV = ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee"]
HELDOUT = [g for g in fs.GRIDS if g not in DEV]
DRAWS = [0, 1, 2]
SELECT_N, SELECT_DRAW = 1000, 0
FROZEN = r"model\.mlp_bus,model\.mlp_gen,model\.input_proj"      # 46,275 trainable params
GRID = {"prior_weight": [0.0, 0.1, 0.3, 1.0, 3.0],
        "lr": [3e-5, 1e-4, 3e-4],
        "trainable": {"frozen": FROZEN, "all": None}}
DC_ANGLE_MAX_DEG = 60.0   # input-only pool filter: drop a scenario whose DC branch angle difference exceeds this
GUARD_FACTOR = 10.0       # selection guard: prior distance > 10x the median config on a dev grid = wrong root
TIE_TOL = 0.02            # configs within 2% of the best residual score: prefer the one closer to the prior

# all runs go through these patched fewshot globals (results dir, pool, splits, training root)
fs.OUT = OUT
fs.TRAIN_DATA = DC_TRAIN
fs.ENV = fs.ENV | {"FEWSHOT_INPUT_NORM": "1"}


def pool_lf(grid: str) -> list[int]:
    """Training scenarios whose load-profile index the eval set never used, minus those whose DC
    power-flow solution is pathological (max branch angle difference > DC_ANGLE_MAX_DEG). Inputs only."""
    ev = pd.read_parquet(fs.EVAL_DATA / grid / "raw/bus_data.parquet", columns=["load_scenario_idx"])
    max_eval_idx = int(ev["load_scenario_idx"].max())
    tr = pd.read_parquet(ORIG_TRAIN / grid / "raw/bus_data.parquet",
                         columns=["scenario", "bus", "load_scenario_idx", "Va_dc"])
    per = tr.groupby("scenario")["load_scenario_idx"].first()
    bad = dc_pathological(grid, tr)
    return sorted(int(s) for s, li in per.items() if li > max_eval_idx and int(s) not in bad)


def dc_max_angle(grid: str, bus: pd.DataFrame | None = None) -> pd.Series:
    """Per scenario: max over in-service branches of |Va_dc(from) - Va_dc(to) - shift| (deg); inf if NaN."""
    if bus is None:
        bus = pd.read_parquet(ORIG_TRAIN / grid / "raw/bus_data.parquet", columns=["scenario", "bus", "Va_dc"])
    br = pd.read_parquet(ORIG_TRAIN / grid / "raw/branch_data.parquet",
                         columns=["scenario", "from_bus", "to_bus", "shift", "br_status"])
    br = br[br["br_status"] == 1]
    va = bus.set_index(["scenario", "bus"])["Va_dc"]
    f = va.reindex(pd.MultiIndex.from_arrays([br["scenario"], br["from_bus"]])).to_numpy()
    t = va.reindex(pd.MultiIndex.from_arrays([br["scenario"], br["to_bus"]])).to_numpy()
    d = np.abs(f - t - br["shift"].to_numpy())
    d = np.where(np.isfinite(d), d, np.inf)
    return pd.Series(d, index=br["scenario"].to_numpy()).groupby(level=0).max()


def dc_pathological(grid: str, bus: pd.DataFrame | None = None) -> set[int]:
    mx = dc_max_angle(grid, bus)
    return {int(s) for s in mx.index[mx > DC_ANGLE_MAX_DEG]}


def split_dir_lf(grid: str, n: int, seed: int = fs.SPLIT_SEED) -> Path:
    return SPLITS / grid / f"n{n}_s{seed}"


fs.pool = pool_lf
fs.split_dir = split_dir_lf


# ----------------------------------------------------------------------------- protocol

def configs() -> list[dict]:
    out = []
    for w, lr, (tname, tpat) in itertools.product(GRID["prior_weight"], GRID["lr"], GRID["trainable"].items()):
        out.append({"name": f"lfc_w{w:g}_lr{lr:g}_{tname}", "prior_weight": w, "lr": lr, "trainable_set": tname,
                    "trainable": tpat, "overrides": [f"training.loss_weights=[1.0,{w:g}]",
                                                     f"optimizer.learning_rate={lr:g}"]})
    return out


def write_protocol() -> dict:
    path = OUT / "hyperparameter_grid.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the protocol is written once, before any run")
    OUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "grid": {"prior_weight (loss_weights=[1, w]; w=0 = physics only)": GRID["prior_weight"],
                 "learning_rate": GRID["lr"], "trainable_set": {k: v or "all parameters" for k, v in GRID["trainable"].items()}},
        "fixed": {"slack_vm": True, "training_targets": "DC prior (Va := Va_dc, PQ Vm := 1.0; dcprior.py)",
                  "recipe": "GENCO case118_1000 fine-tune recipe (200 epochs, batch 64, early stop patience 100)",
                  "init": "GENCO pretrained checkpoint", "normalizer": "input-only fit (gk_freeze FEWSHOT_INPUT_NORM=1)",
                  "pool": f"profile index > eval max, minus scenarios with DC max branch angle diff > {DC_ANGLE_MAX_DEG} deg"},
        "configs": configs(),
        "dev_grids": DEV, "selection_runs": {"n": SELECT_N, "draw": SELECT_DRAW, "eval_set": "never evaluated"},
        "selection_rule": [
            "R[c,g] = min over epochs of 'Validation layer_11_residual' (graphkit's checkpoint-selection metric, "
            "the kept checkpoint); D[c,g] = 'Validation Masked bus MSE loss' at that epoch (distance to the DC "
            "prior on the validation split). Both label-free. A run that kept the pretrained checkpoint is flagged.",
            "r[c,g] = R[c,g] / min_c R[c,g]; d[c,g] = D[c,g] / median_c D[c,g] (per dev grid).",
            f"Guard: c is ineligible if any dev run failed, or d[c,g] > {GUARD_FACTOR} on any dev grid (wrong-root states "
            "such as v2's rotated case14 solution sit far from the DC prior).",
            "Score S[c] = geometric mean over dev grids of r[c,g]; choose argmin S among eligible configs.",
            f"Tie: among eligible configs with S <= (1 + {TIE_TOL}) * min S, choose the smallest geometric-mean d.",
            "No eval-set number and no AC label enters selection. The chosen config is frozen for all grids, draws, n.",
        ],
        "final": {"grids": fs.GRIDS, "draws": DRAWS, "n": [1000, 100],
                  "report": "dev grids (case14, case30) separately from the 7 untouched grids"},
    }
    text = json.dumps(proto, indent=1)
    path.write_text(text)
    (OUT / "hyperparameter_grid.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  hyperparameter_grid.json\n")
    return proto


def protocol() -> dict:
    return json.loads((OUT / "hyperparameter_grid.json").read_text())


def config_by_name(name: str) -> dict:
    return next(c for c in protocol()["configs"] if c["name"] == name)


# ----------------------------------------------------------------------------- data checks

def pools_report() -> list[dict]:
    """Input-only pool vs v2's label-filtered pool, and input-only vs graphkit baseMVA on the draw-0
    n=1000 training split (the latter reads AC labels here, for the record only)."""
    rows = []
    for g in fs.GRIDS:
        new = pool_lf(g)
        ev = pd.read_parquet(fs.EVAL_DATA / g / "raw/bus_data.parquet", columns=["load_scenario_idx"])
        tr = pd.read_parquet(ORIG_TRAIN / g / "raw/bus_data.parquet", columns=["scenario", "load_scenario_idx"])
        per = tr.groupby("scenario")["load_scenario_idx"].first()
        by_idx = {int(s) for s, li in per.items() if li > int(ev["load_scenario_idx"].max())}
        label_bad = fs.collapsed(ORIG_TRAIN / g / "raw/bus_data.parquet")   # v2 filter, record only
        old = sorted(by_idx - label_bad)
        mx = dc_max_angle(g)
        ids = list(map(int, __import__("torch").load(split_dir_lf(g, 1000, 0) / "train.pt")))
        b = pd.read_parquet(ORIG_TRAIN / g / "raw/bus_data.parquet", columns=["scenario", "Pd", "Qd", "Qg"])
        gn = pd.read_parquet(ORIG_TRAIN / g / "raw/gen_data.parquet", columns=["scenario", "p_mw", "p_mw_dc", "is_slack_gen"])
        b, gn = b[b.scenario.isin(ids)], gn[gn.scenario.isin(ids)]
        nz = lambda s: s[s != 0]  # noqa: E731
        base_gk = float(np.percentile(pd.concat([nz(b.Pd), nz(b.Qd), nz(gn.p_mw), nz(b.Qg)]), 95))
        pg = pd.Series(np.where(gn.is_slack_gen.to_numpy() == 1, gn.p_mw_dc, gn.p_mw))
        base_in = float(np.percentile(pd.concat([nz(b.Pd), nz(b.Qd), nz(pg)]), 95))
        rows.append({"grid": g, "pool_input_only": len(new), "pool_v2_label_filter": len(old),
                     "identical": new == old, "dropped_by_dc_criterion": len(by_idx) - len(set(new) & by_idx),
                     "dropped_by_v2_label_filter": len(by_idx & label_bad),
                     "dc_max_branch_angle_deg": {"median": float(mx.median()), "max": float(mx[mx.index.isin(list(by_idx))].max())},
                     "baseMVA_graphkit_with_labels": base_gk, "baseMVA_input_only": base_in,
                     "baseMVA_rel_change": base_in / base_gk - 1})
    # does the DC criterion catch collapsed labels where they exist? (case1354, not used here)
    g = "pglib_opf_case1354_pegase"
    if (ORIG_TRAIN / g / "raw/bus_data.parquet").exists():
        lab = fs.collapsed(ORIG_TRAIN / g / "raw/bus_data.parquet")
        dcb = dc_pathological(g)
        rows.append({"grid": g, "note": "not in this study; checks whether the input-only criterion detects collapsed labels",
                     "scenarios": int(dc_max_angle(g).size), "collapsed_labels": len(lab),
                     "dc_pathological": len(dcb), "overlap": len(lab & dcb),
                     "dc_max_branch_angle_deg_max": float(dc_max_angle(g).max())})
    (OUT / "pools.json").write_text(json.dumps(rows, indent=1))
    return rows


# ----------------------------------------------------------------------------- runs

def _pick_gpu(min_free_mb: int = 6000) -> str:
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = sorted(((int(f), i.strip()) for i, f in (ln.split(",") for ln in out.strip().splitlines())), reverse=True)
        if free and free[0][0] >= min_free_mb:
            return free[0][1]
        time.sleep(60)


def _variant(cfg: dict, seed: int) -> str:
    v = f"{cfg['name']}_slackvm"
    return v if seed == fs.SPLIT_SEED else f"{v}_s{seed}"


def _key(cfg: dict, grid: str, n: int, seed: int) -> str:
    return f"{_variant(cfg, seed)}__finetune__{grid}__n{n}"


def train_one(cfg: dict, grid: str, n: int, seed: int, retries: int = 4) -> dict | None:
    """Train one key (no eval-set evaluation). Returns the run record, or None if another worker holds it."""
    key = _key(cfg, grid, n, seed)
    rec_path = fs.FS / "runs" / f"{key}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    log = OUT / "train_log.jsonl"
    for attempt in range(retries + 1):
        gpu = _pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        fs.ENV = fs.ENV | {"CUDA_VISIBLE_DEVICES": gpu}
        t0 = time.time()
        try:
            rec = fs.train(grid, n, "finetune", _variant(cfg, seed), cfg["overrides"], None, cfg["trainable"], seed, True)
            row = {"key": key, "attempt": attempt, "gpu": gpu, "ok": True, "train_seconds": rec["train_seconds"],
                   "epochs_run": rec["epochs_run"]}
            with log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            return rec
        except Exception as exc:  # shared GPUs: transient OOM; a crashed run is never scored
            err = f"{type(exc).__name__}: {exc}"
            with log.open("a") as fh:
                fh.write(json.dumps({"key": key, "attempt": attempt, "gpu": gpu, "ok": False, "error": err,
                                     "seconds": round(time.time() - t0, 1)}) + "\n")
            print(err, flush=True)
            if "locked:" in err:
                return None
            rec_path.unlink(missing_ok=True)
            time.sleep(120)
    return None


def train_many(cfg_names: list[str], grids: list[str], ns: list[int], draws: list[int]) -> None:
    cfgs = [config_by_name(c) for c in cfg_names]
    for g in grids:
        for d in draws:
            if not (split_dir_lf(g, max(fs.NS), d) / "train.pt").exists():
                fs.write_splits(g, d)
    for cfg, g, n, d in itertools.product(cfgs, grids, ns, draws):
        train_one(cfg, g, n, d)


# ----------------------------------------------------------------------------- selection

def _series(run_dir: Path, metric: str) -> dict[int, float]:
    f = run_dir / "metrics" / metric
    out = {}
    for line in f.read_text().splitlines():
        _, v, step = line.split()
        out[int(step)] = float(v)
    return out


def _is_pretrained(model_path: str) -> bool:
    import torch
    pre = torch.load(fs.PRETRAINED, map_location="cpu")
    m = torch.load(model_path, map_location="cpu")
    return all(torch.equal(m[k], pre[k]) for k in pre if k in m)


def select() -> dict:
    cfgs = protocol()["configs"]
    table = []
    for c in cfgs:
        for g in DEV:
            key = _key(c, g, SELECT_N, SELECT_DRAW)
            rec_path = fs.FS / "runs" / f"{key}.json"
            row = {"config": c["name"], "grid": g, "key": key}
            if not rec_path.exists():
                table.append(row | {"failed": True})
                continue
            rec = json.loads(rec_path.read_text())
            run_dir = Path(rec["model"]).parent
            res = _series(run_dir, "Validation layer_11_residual")
            mse = _series(run_dir, "Validation Masked bus MSE loss")
            step = min(res, key=res.get)
            row |= {"failed": False, "val_residual": res[step], "val_prior_mse": mse[step], "best_step": step,
                    "epochs_logged": len(res), "val_residual_last": res[max(res)],
                    "kept_pretrained": _is_pretrained(rec["model"])}
            table.append(row)
    ok = [r for r in table if not r["failed"]]
    for g in DEV:
        rs = [r for r in ok if r["grid"] == g]
        rmin = min(r["val_residual"] for r in rs)
        dmed = float(np.median([r["val_prior_mse"] for r in rs]))
        for r in rs:
            r["r"] = r["val_residual"] / rmin
            r["d"] = r["val_prior_mse"] / dmed
    agg = []
    for c in cfgs:
        rs = [r for r in table if r["config"] == c["name"]]
        if len(rs) != len(DEV) or any(r["failed"] for r in rs):
            agg.append({"config": c["name"], "eligible": False, "why": "missing/failed dev run"})
            continue
        s = float(np.exp(np.mean([np.log(r["r"]) for r in rs])))
        dg = float(np.exp(np.mean([np.log(r["d"]) for r in rs])))
        guard = any(r["d"] > GUARD_FACTOR for r in rs)
        # R is the best LOGGED epoch; a run that kept the pretrained checkpoint (saved at Lightning's sanity
        # check) is not described by it, so such a config is ineligible (audit genco-adapt F5, 2026-10-08)
        kept = any(r["kept_pretrained"] for r in rs)
        why = " and ".join(w for w, on in ((f"prior distance > {GUARD_FACTOR}x median", guard),
                                           ("kept the pretrained checkpoint (R does not describe it)", kept)) if on)
        agg.append({"config": c["name"], "S": s, "d_geo": dg, "eligible": not (guard or kept),
                    "why": why,
                    **{f"r_{r['grid']}": r["r"] for r in rs}, **{f"d_{r['grid']}": r["d"] for r in rs}})
    el = [a for a in agg if a["eligible"]]
    smin = min(a["S"] for a in el)
    near = [a for a in el if a["S"] <= (1 + TIE_TOL) * smin]
    chosen = min(near, key=lambda a: a["d_geo"])
    out = {"chosen": chosen["config"], "best_S": smin, "tie_set": [a["config"] for a in near],
           "per_run": table, "per_config": sorted(agg, key=lambda a: a.get("S", np.inf))}
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    with (OUT / "selection_table.csv").open("w", newline="") as fh:
        cols = ["config", "eligible", "S", "d_geo", *[f"r_{g}" for g in DEV], *[f"d_{g}" for g in DEV], "why"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a in out["per_config"]:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in a.items()})
    return out


# ----------------------------------------------------------------------------- final

def final(ns: list[int], grids: list[str], draws: list[int]) -> None:
    cfg = config_by_name(json.loads((OUT / "selection.json").read_text())["chosen"])
    res_path = OUT / "results.jsonl"
    for g in grids:
        for d in draws:
            if not (split_dir_lf(g, max(fs.NS), d) / "train.pt").exists():
                fs.write_splits(g, d)
    for g, n, d in itertools.product(grids, ns, draws):
        key = _key(cfg, g, n, d)
        done = fs._latest_ok(res_path) if res_path.exists() else {}
        if key in done:
            continue
        # one worker per key from training through evaluation: a second worker used to skip straight to
        # evaluate and could score the first worker's artifacts (audit genco-adapt F4, 2026-10-08)
        flock = fs.FS / "runs" / f"{key}.final.lock"
        flock.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(flock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            print(f"final: {key} is held by another worker, skipping", flush=True)
            continue
        try:
            _final_one(cfg, key, g, n, d, res_path)
        finally:
            flock.unlink(missing_ok=True)


def _final_one(cfg: dict, key: str, g: str, n: int, d: int, res_path: Path) -> None:
    if res_path.exists() and key in fs._latest_ok(res_path):   # scored by another worker before we took the lock
        return
    rec = train_one(cfg, g, n, d)
    if rec is None:
        return
    r = {"key": key, "grid": g, "n": n, "draw": d, "config": cfg["name"], "variant": _variant(cfg, d),
         "mode": "finetune", "split_seed": d, "overrides": cfg["overrides"], "trainable": cfg["trainable"],
         "slack_vm": True, "train_root": str(DC_TRAIN), "dev_grid": g in DEV,
         **{k: rec[k] for k in ("train_seconds", "epochs_run", "returncode")}}
    for attempt in range(3):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = fs.ENV["CUDA_VISIBLE_DEVICES"] = _pick_gpu(3000)
            r |= fs.evaluate(g, rec)
            r.pop("error", None)
            break
        except Exception as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(60)
    r["n_actual"] = json.loads((split_dir_lf(g, n, d) / "split.json").read_text())["n_actual"]
    with res_path.open("a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: r.get(k) for k in ("key", "genco_active_res_mw", "dc_active_res_mw", "error")}), flush=True)


# ----------------------------------------------------------------------------- summary

def _stats(rs: list[dict]) -> dict:
    out = {"draws": len(rs)}
    for name, f, fdc in fs.METRICS:
        v = np.array([f(r) for r in rs])
        dcv = np.array([fdc(r) for r in rs])
        out[name] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "dc": float(dcv.mean()),
                     "ratio_mean": float((v / dcv).mean()), "ratio_min": float((v / dcv).min()),
                     "ratio_max": float((v / dcv).max()), "beats_dc": int((v < dcv).sum())}
    return out


def summary() -> dict:
    new = fs._latest_ok(OUT / "results.jsonl")
    old = fs._latest_ok(V2)
    chosen = json.loads((OUT / "selection.json").read_text())["chosen"]
    old_fam = {1000: "physdcW1_slackvm", 100: "freezephysdc_slackvm"}   # v2 provisional (n=100: weak prior, frozen)
    rows = []
    for g in fs.GRIDS:
        for n in (1000, 100):
            rs = [r for r in new.values() if r["grid"] == g and r["n"] == n]
            if not rs:
                continue
            o = [r for r in old.values() if r["grid"] == g and r["n"] == n
                 and r["variant"] in {old_fam[n], f"{old_fam[n]}_s1", f"{old_fam[n]}_s2"}]
            rows.append({"grid": g, "n": n, "group": "dev" if g in DEV else "untouched",
                         "n_actual": sorted({r["n_actual"] for r in rs}), "new": _stats(rs) | {"keys": sorted(r["key"] for r in rs)},
                         "v2_provisional": (_stats(o) | {"variant": old_fam[n], "keys": sorted(r["key"] for r in o)}) if o else None})
    held = [r for r in rows if r["group"] == "untouched" and r["n"] == 1000]
    verdict = {
        n: {"grids_beating_dc_on_both_mean_ratio": [r["grid"] for r in rows if r["group"] == "untouched" and r["n"] == n
                                                    and r["new"]["res"]["ratio_mean"] < 1 and r["new"]["dva"]["ratio_mean"] < 1],
            "grids_beating_dc_on_both_every_draw": [r["grid"] for r in rows if r["group"] == "untouched" and r["n"] == n
                                                    and r["new"]["res"]["beats_dc"] == r["new"]["draws"]
                                                    and r["new"]["dva"]["beats_dc"] == r["new"]["draws"]]}
        for n in (1000, 100)}
    out = {"chosen_config": chosen, "rows": rows, "verdict_untouched": verdict, "n_untouched_n1000": len(held)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("protocol")
    sub.add_parser("pools")
    sp = sub.add_parser("splits")
    sp.add_argument("--grids", nargs="+", default=fs.GRIDS)
    t = sub.add_parser("train")
    t.add_argument("--configs", nargs="+", default=None, help="config names (default: whole grid)")
    t.add_argument("--grids", nargs="+", default=DEV)
    t.add_argument("--ns", nargs="+", type=int, default=[SELECT_N])
    t.add_argument("--draws", nargs="+", type=int, default=[SELECT_DRAW])
    sub.add_parser("select")
    f = sub.add_parser("final")
    f.add_argument("--ns", nargs="+", type=int, default=[1000, 100])
    f.add_argument("--grids", nargs="+", default=fs.GRIDS)
    f.add_argument("--draws", nargs="+", type=int, default=DRAWS)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "protocol":
        print(json.dumps(write_protocol(), indent=1))
    elif a.cmd == "pools":
        print(json.dumps(pools_report(), indent=1))
    elif a.cmd == "splits":
        print(json.dumps({g: {d: fs.write_splits(g, d) for d in DRAWS} for g in a.grids}, indent=1))
    elif a.cmd == "train":
        train_many(a.configs or [c["name"] for c in protocol()["configs"]], a.grids, a.ns, a.draws)
    elif a.cmd == "select":
        s = select()
        print(json.dumps({k: s[k] for k in ("chosen", "best_S", "tie_set")}, indent=1))
    elif a.cmd == "final":
        final(a.ns, a.grids, a.draws)
    else:
        print(json.dumps(summary()["verdict_untouched"], indent=1))


if __name__ == "__main__":
    main()
