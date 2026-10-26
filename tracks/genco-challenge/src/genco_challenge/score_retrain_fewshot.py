"""Case118 n = 1,000 few-shot fine-tune FROM each OSC retrain arm (job B), scored like the isolation study.

Fills the "case118 n = 1,000 (of DC)" column of the post's retrain table. For each arm checkpoint
(``experiments/genco-retrain/<arm>/final/best_model_state_dict.pt``; ``released`` = GENCO's released pretrained
checkpoint, the reference) and each seed, run GENCO's own case118_1000 fine-tune recipe through ``fewshot.train``
with ``init`` = that checkpoint, under two recipes: ``published`` (as released) and ``warm100`` (100-step linear
warm-up, ``FEWSHOT_WARMUP_STEPS=100``, the fix of results/2026-10-10-case118-retrain-isolation). Each run is checked
as iso.py checks it (return code, epochs run, finite validation curve, checkpoint differs from its init) and scored
on the fresh eval set ``eval2`` (ratio to DC, branch angle). One row per run in ``<out>/results.jsonl``; ``table.md``
gives per (arm, recipe) the per-seed ratios and the median.

    CUDA_VISIBLE_DEVICES=1 nice -n 10 python -m genco_challenge.score_retrain_fewshot \\
        --out tracks/genco-challenge/results/<date>-genco-retrain-fewshot [--arms fixed published released] \\
        [--recipes published warm100] [--seeds 0 1 2] [--epochs 200]

Cost: about 20 to 40 minutes per run on one RTX 3090 (isolation study), so the default 3 arms x 2 recipes x 3 seeds
is 18 runs, roughly 6 to 12 GPU-hours; run two processes on two GPUs with disjoint --arms. ``--epochs 2`` is a smoke
test of the plumbing only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

os.environ["FEWSHOT_FIX_SANITY"] = "1"
os.environ.setdefault("PYTHONNOUSERSITE", "1")
from genco_challenge import fewshot as fs  # noqa: E402
from genco_challenge.fresh_eval_data import EVAL2  # noqa: E402

GRID = "pglib_opf_case118_ieee"
REPO = fs.REPO if hasattr(fs, "REPO") else Path(__file__).resolve().parents[4]
RECIPES = {"published": {}, "warm100": {"FEWSHOT_WARMUP_STEPS": "100"}}


def arm_ckpt(run_root: Path, arm: str) -> Path:
    if arm == "released":
        return fs.PRETRAINED
    return run_root / arm / "final" / "best_model_state_dict.pt"


def val_curve(rundir: Path) -> list[float]:
    f = rundir / "metrics" / "Validation layer_11_residual"
    if not f.exists():
        return []
    rows = sorted((ln.split() for ln in f.read_text().splitlines() if ln.strip()), key=lambda r: int(r[2]))
    return [float(r[1]) for r in rows]


def same_weights(a: Path, b: Path) -> bool:
    import torch
    x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def score_eval2(rec: dict) -> dict:
    fsdir = fs.WORK / "fewshot_fresh"
    (fsdir / "configs").mkdir(parents=True, exist_ok=True)
    old = fs.FS, fs.EVAL_DATA
    fs.FS, fs.EVAL_DATA = fsdir, EVAL2
    try:
        r = fs.evaluate(GRID, rec)
    finally:
        fs.FS, fs.EVAL_DATA = old
    e = r.get("errors") if isinstance(r.get("errors"), dict) else {}
    return {"eval2_res_mw": r["genco_active_res_mw"], "eval2_dc_res_mw": r["dc_active_res_mw"],
            "eval2_ratio_to_dc": r["genco_active_res_mw"] / r["dc_active_res_mw"],
            "eval2_dva_mae_deg": (e.get("genco") or {}).get("branch_dva_mae_deg", e.get("branch_dva_mae_deg")), "eval2_n_scored": r["n_test_scenarios"]}


def run_one(arm: str, init: Path, recipe: str, seed: int, epochs: int, tag: str) -> dict:
    base = dict(fs.ENV)
    fs.ENV = base | RECIPES[recipe]
    variant = f"{tag}_{arm}_{recipe}_s{seed}_e{epochs}"
    row = {"arm": arm, "recipe": recipe, "seed": seed, "epochs": epochs, "init": str(init), "variant": variant}
    try:
        fs._wait_gpu()
        rec = fs.train(GRID, 1000, "finetune", variant, [f"seed={seed}", f"training.epochs={epochs}"], init)
        if rec["returncode"] != 0 or rec["epochs_run"] != epochs:
            raise RuntimeError(f"train check failed: rc={rec['returncode']} epochs_run={rec['epochs_run']}")
        val = val_curve(Path(rec["model"]).parent)
        if len(val) != epochs or not all(math.isfinite(v) for v in val):
            raise RuntimeError(f"validation curve check failed: n={len(val)}")
        if same_weights(Path(rec["model"]), init):
            raise RuntimeError("selected checkpoint equals its init")
        row |= {"train_seconds": rec["train_seconds"], "val_epoch0": val[0], "val_min": min(val),
                "best_epoch": val.index(min(val)), "starts_above_1": val[0] > 1.0} | score_eval2(rec)
    except Exception as exc:  # recorded, never scored
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        fs.ENV = base
    return row


def table(rows: list[dict]) -> str:
    out = ["# case118 n = 1,000 fine-tune from each retrain arm: eval2 residual ratio to DC (generated)", "",
           "| arm | recipe | per seed | median | branch angle deg (median) | errors |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["arm"], r["recipe"]) for r in rows})
    for a, rc in keys:
        ok = [r for r in rows if (r["arm"], r["recipe"]) == (a, rc) and "error" not in r]
        bad = [r for r in rows if (r["arm"], r["recipe"]) == (a, rc) and "error" in r]
        ratios = [r["eval2_ratio_to_dc"] for r in sorted(ok, key=lambda r: r["seed"])]
        dva = [r["eval2_dva_mae_deg"] for r in ok if r.get("eval2_dva_mae_deg") is not None]
        out.append(f"| {a} | {rc} | {', '.join(f'{x:.2f}' for x in ratios) or 'n/a'} | "
                   f"{statistics.median(ratios):.2f} | {(f'{statistics.median(dva):.3f}' if dva else 'n/a')} | {len(bad)} |" if ratios else
                   f"| {a} | {rc} | n/a | n/a | n/a | {len(bad)} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, default=Path(REPO) / "experiments/genco-retrain")
    ap.add_argument("--arms", nargs="+", default=["fixed", "published", "released"])
    ap.add_argument("--recipes", nargs="+", default=["published", "warm100"], choices=list(RECIPES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--tag", default="retrainfs")
    a = ap.parse_args()
    a.run_root = a.run_root.resolve()   # absolute: graphkit runs in another cwd (found 10-19)
    a.out.mkdir(parents=True, exist_ok=True)
    res = a.out / "results.jsonl"
    done = set()
    if res.exists():
        done = {(r["arm"], r["recipe"], r["seed"], r["epochs"]) for r in map(json.loads, res.read_text().splitlines())
                if "error" not in r}
    for arm in a.arms:
        init = arm_ckpt(a.run_root, arm)
        if not init.is_file():
            print(f"skip {arm}: {init} missing (pull back with: bash slurm/sync_from_osc.sh genco-retrain)")
            continue
        for rc in a.recipes:
            for s in a.seeds:
                if (arm, rc, s, a.epochs) in done:
                    continue
                row = run_one(arm, init, rc, s, a.epochs, a.tag)
                with res.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in ("arm", "recipe", "seed", "eval2_ratio_to_dc", "error")}), flush=True)
    rows = [json.loads(line) for line in res.read_text().splitlines()]
    (a.out / "table.md").write_text(table(rows))
    print((a.out / "table.md").read_text())


if __name__ == "__main__":
    main()
