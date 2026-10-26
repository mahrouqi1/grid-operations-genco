"""Re-score existing adaptation checkpoints on FRESH eval sets (audit 2026-10-08 genco-adapt F1/F2). EVAL ONLY.

No training happens here. Every row to re-score comes from an earlier study's results.jsonl (v1 rows as
re-scored in v2, v2, label-free-clean). Its checkpoint and its own normalizer stats come from the run
record ``fewshot/runs/<key>.json`` (or the reference checkpoints), and ``fewshot.evaluate`` /
``fewshot._score`` (graphkit ``evaluate`` + ``ladder_eval.errors``: residual, branch dVa wrapped, |V|,
DC and flat baselines) score it on ``genco-ladder/eval2/<grid>`` (see ``fresh_eval_data``). In this
process only: ``fewshot.EVAL_DATA`` -> eval2 and ``fewshot.FS`` -> ``genco-ladder/fewshot_fresh`` so no
config, log or MLflow run of the earlier studies (or of a concurrently running study) is touched.

    python -m genco_challenge.fresh_eval score --grids pglib_opf_case14_ieee ...     # appends results.jsonl
    python -m genco_challenge.fresh_eval analyze                                      # tables + summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import numpy as np

from genco_challenge import fewshot as fs
from genco_challenge.fresh_eval_data import EVAL2

OUT = fs.REPO / "tracks/genco-challenge/results/2026-10-08-adapt-fresh-eval"
FS2 = fs.WORK / "fewshot_fresh"
RUNS = fs.FS / "runs"                 # run records of the earlier studies (read only); fs.FS is redirected in score()
V2 = fs.REPO / "tracks/genco-challenge/results/2026-10-07-genco-fewshot-v2/results.jsonl"
LFC = fs.REPO / "tracks/genco-challenge/results/2026-10-08-genco-labelfree-clean/results.jsonl"
LFC_VARIANT = re.compile(r"^lfc_w1_lr3e-05_all_slackvm(_s\d+)?$")
DEV = ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee"]
HELD = [g for g in fs.GRIDS if g not in DEV]


def plan() -> list[dict]:
    """Rows to re-score: every successful v2 row (includes v1 re-scored) + the 54 final label-free rows."""
    rows = [dict(r, study="v2") for r in fs._latest_ok(V2).values()]
    rows += [dict(r, study="labelfree-clean") for r in fs._latest_ok(LFC).values() if LFC_VARIANT.match(r["variant"])]
    return rows


def record(r: dict) -> dict:
    if r["mode"] in fs.REFERENCE:
        mp, ns = fs.REFERENCE[r["mode"]]
        return {"key": r["key"], "model": str(mp), "normalizer_stats": ns and str(ns), "slack_vm": bool(r.get("slack_vm"))}
    rec = json.loads((RUNS / f"{r['key']}.json").read_text())
    assert rec["key"] == r["key"] and bool(rec.get("slack_vm")) == bool(r.get("slack_vm")), r["key"]
    return rec


def _pick_gpu(min_free_mb: int = 3500) -> str:
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = sorted(((int(f), i.strip()) for i, f in (ln.split(",") for ln in out.strip().splitlines())), reverse=True)
        if free and free[0][0] >= min_free_mb:
            return free[0][1]
        time.sleep(60)


def score(grids: list[str]) -> None:
    fs.EVAL_DATA, fs.FS = EVAL2, FS2
    (FS2 / "configs").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    for r in plan():
        if r["grid"] not in grids:
            continue
        done = fs._latest_ok(res_path) if res_path.exists() else {}
        if r["key"] in done:
            continue
        lock = FS2 / "locks" / f"{r['key']}.lock"     # several workers may share a grid: one scores each key
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        rec = record(r)
        lf = r["study"] == "labelfree-clean"
        # the label-free runs were evaluated with the input-only-normalizer plugin env on (harmless with
        # --normalizer_stats given); supervised runs without it. Same env as when each was first scored.
        fs.ENV = {k: v for k, v in fs.ENV.items() if k != "FEWSHOT_INPUT_NORM"} | ({"FEWSHOT_INPUT_NORM": "1"} if lf else {})
        row = {k: r[k] for k in ("key", "grid", "n", "mode", "variant", "split_seed", "slack_vm", "study") if k in r}
        row |= {"train_root": r.get("train_root"), "trainable": r.get("trainable"), "overrides": r.get("overrides")}
        if "n_actual" in r:
            row["n_actual"] = r["n_actual"]
        for attempt in range(4):
            try:
                g = _pick_gpu()
                fs.ENV = fs.ENV | {"CUDA_VISIBLE_DEVICES": g}
                row |= fs.evaluate(r["grid"], rec)
                row.pop("error", None)
                break
            except Exception as exc:   # shared GPUs: OOM happens. A crashed evaluation is never scored.
                row["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(60)
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if "error" in row:
            lock.unlink(missing_ok=True)   # a failed key can be retried by the next run
        print(json.dumps({k: row.get(k) for k in ("key", "genco_active_res_mw", "dc_active_res_mw", "error")}), flush=True)


# ------------------------------------------------------------------------------------------ analysis

def fam(v: str) -> str:
    return re.sub(r"_s\d+$", "", v)


def metrics(r: dict) -> dict:
    g, d, f = r["errors"]["genco"], r["errors"]["dc"], r["errors"]["flat"]
    return {"res": r["genco_active_res_mw"], "res_dc": r["dc_active_res_mw"],
            "dva": g["branch_dva_mae_deg"], "dva_dc": d["branch_dva_mae_deg"], "dva_flat": f["branch_dva_mae_deg"],
            "vm": g["vm_mae_pu"], "vm_dc": d["vm_mae_pu"], "vm_flat": f["vm_mae_pu"],
            "va": g["va_mae_deg"], "va_dc": d["va_mae_deg"], "va_flat": f["va_mae_deg"]}


def group(rows: list[dict]) -> dict:
    """(grid, family, mode, n) -> mean over draws of each metric and of the ratio to DC, draws beating DC."""
    g: dict = {}
    for r in rows:
        g.setdefault((r["grid"], fam(r["variant"]), r["mode"], r["n"]), []).append(r)
    out = {}
    for k, rs in g.items():
        m = [metrics(r) for r in rs]
        o = {"draws": len(rs), "n_test": sorted({r["n_test_scenarios"] for r in rs}),
             "n_actual": sorted({str(r.get("n_actual", "-")) for r in rs})}
        for q in ("res", "dva", "vm"):
            v, dc = np.array([x[q] for x in m]), np.array([x[f"{q}_dc"] for x in m])
            o[q] = {"mean": float(v.mean()), "dc": float(dc.mean()), "ratio": float((v / dc).mean()),
                    "ratio_min": float((v / dc).min()), "ratio_max": float((v / dc).max()), "beats": int((v < dc).sum())}
        out[k] = o
    return out


def load_old(study: str) -> list[dict]:
    p = V2 if study == "v2" else LFC
    return [r for r in fs._latest_ok(p).values() if study == "v2" or LFC_VARIANT.match(r["variant"])]


def load_new() -> list[dict]:
    return list(fs._latest_ok(OUT / "results.jsonl").values())


def lf_headline(groups: dict) -> dict:
    """Label-free chosen config, 7 non-dev grids, n = 1000, from grouped rows."""
    res = {}
    for g in HELD:
        k = next((k for k in groups if k[0] == g and k[1] == "lfc_w1_lr3e-05_all_slackvm" and k[3] == 1000), None)
        if k is None:
            continue
        o = groups[k]
        res[g] = {"res": o["res"]["ratio"], "dva": o["dva"]["ratio"], "vm": o["vm"]["ratio"],
                  "res_every_draw": o["res"]["beats"] == o["draws"], "dva_every_draw": o["dva"]["beats"] == o["draws"],
                  "draws": o["draws"]}
    return {"per_grid": res,
            "res_beats": [g for g, x in res.items() if x["res"] < 1], "dva_beats": [g for g, x in res.items() if x["dva"] < 1],
            "both_beats": [g for g, x in res.items() if x["res"] < 1 and x["dva"] < 1],
            "both_every_draw": [g for g, x in res.items() if x["res_every_draw"] and x["dva_every_draw"]],
            "vm_beats": [g for g, x in res.items() if x["vm"] < 1]}


SUP = {"freeze": "frozen trunk (46k)", "freezeN": "freezeN (66k)"}


def sup_wins(groups: dict, variant: str) -> dict:
    """(grid, n) -> beats DC on residual / on all three metrics (mean ratio over draws < 1), for n in 10, 100, 1000."""
    out = {}
    for (g, v, mode, n), o in groups.items():
        if v == variant and mode == "finetune":
            out[(g, n)] = {"res": o["res"]["ratio"], "dva": o["dva"]["ratio"], "vm": o["vm"]["ratio"], "draws": o["draws"],
                           "beats_res": o["res"]["ratio"] < 1, "beats_all": all(o[q]["ratio"] < 1 for q in ("res", "dva", "vm"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "analyze", "plan"])
    ap.add_argument("--grids", nargs="+", default=fs.GRIDS)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.grids)
    elif a.cmd == "plan":
        rows = plan()
        print(len(rows), {g: sum(r["grid"] == g for r in rows) for g in fs.GRIDS})
    else:
        from genco_challenge import fresh_eval_report
        fresh_eval_report.build()


if __name__ == "__main__":
    main()
