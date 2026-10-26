"""Transfer experiment: validation selection, then ONE scoring of the 15 (protocol.json, frozen + amendments).

  python run_selection.py val  --runs D_s0 a_s0 ... --gpu 0     # RTE validation for every 2k checkpoint + final
  python run_selection.py pick                                   # per-run and per-arm selection -> selection.json/.md
  python run_selection.py test --runs ... --gpu 0                # the 15 locked-benchmark grids, selected ckpt only

Runs live in experiments/genco-transfer/<arm>_s<seed>/steps (OSC runs pulled back by pull_back.sh; C00/C01 local).
step 0 is the released weights (identical in every run) and is not re-scored: val-released/ holds it.
Metric: mean over the RTE validation grids of the branch-angle error ratio to DC (genco branch_dva_mae_deg /
dc branch_dva_mae_deg), lower is better. UNAUDITED.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
# RESCUE=1 (protocol-rescue.json): runs under experiments/genco-transfer-rescue, outputs suffixed -rescue
RESCUE = os.environ.get("RESCUE") == "1"
# SIMBENCH=1 (arm H, an internal note): runs under experiments/genco-transfer-simbench, outputs suffixed -simbench, 3 runs;
# validation grids, STEPS and the released-weights rows are the main round's (val-released/ is shared, no -simbench copy).
SIMBENCH = os.environ.get("SIMBENCH") == "1"
RUNS = REPO / ("experiments/genco-transfer-rescue" if RESCUE else "experiments/genco-transfer-simbench" if SIMBENCH else "experiments/genco-transfer")
SUF = "-rescue" if RESCUE else "-simbench" if SIMBENCH else ""
RELSUF = "-rescue" if RESCUE else ""   # suffix of the val-released directory
VAL_ROOT = REPO / "data/external/genco-transfer-val"
VAL_GRIDS = ["case1888rte", "case2848rte"]
if os.environ.get("RESCUE") == "1":   # rescue amendment (audit R1-F3 noise floor): all 5 RTE grids
    VAL_GRIDS = ["case1888rte", "case2848rte", "case6470rte", "case6495rte", "case6515rte"]
STEPS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
if os.environ.get("RESCUE") == "1":
    STEPS = [2000, 4000, 6000, 8000, 10000, 12000]   # every 2,000 (amendment 02:45, before any rescue validation value was read)
PY = "<CONDA_ENVS>/grid-operations-genco/bin/python"


def ckpts(run: str) -> dict[str, Path]:
    d = RUNS / run / "steps"
    out = {f"step{s}": d / f"state_step{s}.pt" for s in STEPS if (d / f"state_step{s}.pt").exists()}
    if (d / "state_final.pt").exists():
        out["final"] = d / "state_final.pt"
    return out


def score(out: Path, specs: list[str], grids: list[str] | None, data_root: Path | None, gpu: str) -> None:
    cmd = [PY, "-m", "genco_challenge.score_ckpts", "--out", str(out), "--ckpt", *specs]
    if grids:
        cmd += ["--grids", *grids]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    env = os.environ | {"PYTHONNOUSERSITE": "1", "MLFLOW_ALLOW_FILE_STORE": "true", "CUDA_VISIBLE_DEVICES": gpu,
                        "PYTHONPATH": str(REPO / "tracks/genco-challenge/src"),
                        "PATH": f"{Path(PY).parent}:{os.environ['PATH']}"}
    subprocess.run(cmd, cwd=REPO, env=env, check=False)


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def angle_ratio(r: dict) -> float:
    e = r["errors"]
    return e["genco"]["branch_dva_mae_deg"] / e["dc"]["branch_dva_mae_deg"]


def pick() -> dict:
    sel = {"runs": {}, "arms": {}}
    dirs = [d for d in sorted((HERE / f"val{SUF}").glob("*")) if d.is_dir()]   # transfer R2-F2: directories only
    expected = 9 if RESCUE else 3 if SIMBENCH else 21
    if len(dirs) != expected:
        raise SystemExit(f"expected {expected} runs in val{SUF}/, found {len(dirs)}: {[d.name for d in dirs]}")
    for d in dirs:
        if not (d / "results.jsonl").exists() or not rows(d / "results.jsonl"):
            raise SystemExit(f"{d.name}: no validation results")
        allowed = {f"step{x}" for x in STEPS} | {"final"}   # only the protocol's checkpoint set (extra rows ignored)
        R = [r for r in rows(d / "results.jsonl") if "error" not in r and r["model"].rsplit("_", 1)[1] in allowed]
        by = {}
        for r in R:
            by.setdefault(r["model"], {})[r["grid"]] = angle_ratio(r)
        partial = [m for m, v in by.items() if not all(g in v for g in VAL_GRIDS)]
        if partial:                      # never drop a checkpoint silently (audit R1-F6)
            raise SystemExit(f"{d.name}: checkpoints missing a validation grid: {partial}")
        full = {m: float(np.mean([v[g] for g in VAL_GRIDS])) for m, v in by.items()}
        on_disk = {f"{d.name}_{t}" for t in ckpts(d.name)}           # transfer R2-F2: every on-disk protocol checkpoint
        if on_disk - set(full):                                       # must be scored before step 0 can win by default
            raise SystemExit(f"{d.name}: on-disk checkpoints not scored on validation: {sorted(on_disk - set(full))}")
        if True:   # audit R1-F1 (approved by the owner 2026-10-20 23:09): step 0 = released weights is a candidate in every mode
            rel = [r for r in rows(HERE / f"val-released{RELSUF}/results.jsonl") if "error" not in r and r["grid"] in VAL_GRIDS]
            if len(rel) != len(VAL_GRIDS):
                raise SystemExit(f"val-released{RELSUF}/ incomplete: score the released weights on every validation grid first")
            full[f"{d.name}_step0"] = float(np.mean([angle_ratio(r) for r in rel]))
        if not full:
            continue
        best = min(full, key=full.get)
        sel["runs"][d.name] = {"selected": best, "val_metric": full[best], "all": full}
    for run, s in sel["runs"].items():
        arm = run.rsplit("_s", 1)[0]
        sel["arms"].setdefault(arm, {"seeds": {}})["seeds"][run] = s["val_metric"]
    for arm, a in sel["arms"].items():
        a["seed_mean_val_metric"] = float(np.mean(list(a["seeds"].values())))
    ladder = {k: v for k, v in sel["arms"].items() if k in (("Rlr", "Rfrz", "Rrep") if RESCUE else ("H",) if SIMBENCH else ("D", "a", "b", "c", "m"))}
    if ladder:
        sel["go_no_go_arm"] = min(ladder, key=lambda k: ladder[k]["seed_mean_val_metric"])
    rel = [r for r in rows(HERE / f"val-released{RELSUF}/results.jsonl") if "error" not in r]
    if rel:
        sel["released_val_metric"] = float(np.mean([angle_ratio(r) for r in rel if r["grid"] in VAL_GRIDS]))
    (HERE / f"selection{SUF}.json").write_text(json.dumps(sel, indent=1))
    L = ["# Validation selection (generated by run_selection.py pick; UNAUDITED)", "",
         f"Metric: mean over {VAL_GRIDS} of the branch-angle error ratio to DC. Released: {sel.get('released_val_metric', float('nan')):.3f}", "",
         "| run | selected checkpoint | validation metric |", "|---|---|---|"]
    for run, s in sorted(sel["runs"].items()):
        L.append(f"| {run} | {s['selected']} | {s['val_metric']:.3f} |")
    L += ["", "| arm | seed-mean validation metric |", "|---|---|"]
    for arm, a in sorted(sel["arms"].items()):
        L.append(f"| {arm} | {a['seed_mean_val_metric']:.3f} |")
    L += ["", f"Go/no-go arm (lowest seed-mean among ladder arms): **{sel.get('go_no_go_arm', 'n/a')}**"]
    (HERE / f"selection{SUF}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["val", "pick", "test"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    if a.stage == "val":
        for run in a.runs:
            c = ckpts(run)
            if not c:
                print("no checkpoints for", run); continue
            score(HERE / f"val{SUF}" / run, [f"{run}_{t}={p}" for t, p in c.items()], VAL_GRIDS, VAL_ROOT, a.gpu)
    elif a.stage == "pick":
        pick()
    else:
        sel = json.loads((HERE / f"selection{SUF}.json").read_text())
        for run in a.runs:
            s = sel["runs"].get(run)
            if not s:
                print("not selected yet:", run); continue
            tag = s["selected"].rsplit("_", 1)[1]
            if tag == "step0":   # released weights (rescue amendment): held-out rows = released
                print(run, "selects step0 = released"); continue
            p = ckpts(run)[tag]
            score(HERE / f"heldout{SUF}" / run, [f"{run}={p}"], None, None, a.gpu)


if __name__ == "__main__":
    sys.exit(main())
