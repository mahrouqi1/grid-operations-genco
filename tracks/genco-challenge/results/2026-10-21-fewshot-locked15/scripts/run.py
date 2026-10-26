"""Driver for the locked-15 few-shot study (candidate 4) (UNAUDITED). One worker = one training stream; run at most two.

    CUDA_VISIBLE_DEVICES=0 python scripts/run.py worker --name w1 --deadline EPOCH_SECONDS

Specs come from queue.json (written with protocol.json by plan.py); workers claim specs in priority order through O_EXCL files in
claims/. A spec is trained through fewshot.train (non-zero exit raises; the run is NEVER scored), checked, evaluated on the fresh
eval2 set (graphkit evaluate, own normalizer stats) and scored with genco_on_fd2/scripts/common.py (dc, fd2, genco on the same
scenarios: branch dVa, |V|, our AC residual). One row per spec lands in results.jsonl; failed specs get an 'error' row, no scores.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ["FEWSHOT_FIX_SANITY"] = "1"
os.environ.setdefault("PYTHONNOUSERSITE", "1")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "2026-10-09-genco-on-fd2/scripts"))
import numpy as np, pandas as pd
import common as cm
from genco_challenge import fewshot as fs
from genco_challenge.fresh_eval_data import EVAL2

EVAL2F = fs.WORK / "eval2_renumbered"   # case240 only: eval2 minus its 5 collapsed-label scenarios (scripts/filter_eval_case240.py)
RESULTS = HERE / "results.jsonl"
CURVES = HERE / "curves.jsonl"
CLAIMS = HERE / "claims"
FS_REAL = fs.FS
EVAL_REAL = fs.EVAL_DATA
BASE_ENV = dict(fs.ENV)
DC_TRAIN_REAL = fs.TRAIN_DATA       # = train_dcprior, set as a side effect of importing common (labelfree)
SPLIT_DIR_REAL = fs.split_dir
EST = {100: 1700}


def read_metric(rundir: Path, name: str) -> list[float]:
    f = rundir / "metrics" / name
    if not f.exists():
        return []
    rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: int(r[2]))
    return [float(r[1]) for r in rows]


def same_as_pretrained(model_path) -> bool:
    import torch
    pre = torch.load(fs.PRETRAINED, map_location="cpu")
    m = torch.load(model_path, map_location="cpu")
    return all(torch.equal(m[k], pre[k]) for k in pre if k in m)


def run_spec(spec: dict, wname: str) -> dict:
    grid, n, s = spec["grid"], spec["n"], spec["split_seed"]
    variant = f"lk15_{spec['tag']}_s{s}"
    lab = spec.get("labels", "dcprior")
    if lab == "ac":      # amendment1: the labelled recipe: AC training labels, graphkit's own (label-reading) normalizer
        fs.TRAIN_DATA = fs.WORK / "train"
        fs.split_dir = lambda g, n_, seed=0: FS_REAL / "splits_L" / g / (f"n{n_}" if seed == 0 else f"n{n_}_s{seed}")
        base_env = {k: v for k, v in BASE_ENV.items() if k != "FEWSHOT_INPUT_NORM"}
    else:                # as B4 ran it: importing common patches fewshot to the DC-prior training set + input-only normalizer
        fs.TRAIN_DATA = DC_TRAIN_REAL
        fs.split_dir = SPLIT_DIR_REAL
        base_env = dict(BASE_ENV)
    fs.write_splits(grid, s)                              # additive: never rewrites an existing split file
    fs.FS, fs.EVAL_DATA = FS_REAL, EVAL_REAL
    train_env = {"FEWSHOT_FIX_SANITY": "1"} | {k: str(v) for k, v in spec["env"].items()}
    fs.ENV = base_env | train_env
    row = {"id": spec["id"], "tag": spec["tag"], "grid": grid, "n": n, "init_seed": spec["seed"], "split_seed": s,
           "env": spec["env"], "epochs_requested": spec["epochs"], "variant": variant, "labels": lab, "note": spec.get("note", "")}
    fs._wait_gpu()
    rec = fs.train(grid, n, "finetune", variant, [f"seed={spec['seed']}", f"training.epochs={spec['epochs']}"], None, None, s, False)
    if rec["returncode"] != 0:
        raise RuntimeError(f"train check failed: rc={rec['returncode']}")
    keep = Path(rec["model"]).parent
    val = read_metric(keep, "Validation layer_11_residual")
    vloss = read_metric(keep, "Validation loss")
    if len(val) < 5 or not all(math.isfinite(v) for v in val):
        raise RuntimeError(f"metrics check failed: n={len(val)} finite={all(math.isfinite(v) for v in val)}")
    if same_as_pretrained(rec["model"]):
        raise RuntimeError("selected checkpoint equals the pretrained one")
    row |= {"train_seconds": rec["train_seconds"], "epochs_run": rec["epochs_run"], "returncode": rec["returncode"],
            "val_epoch0": val[0], "val_final": val[-1], "val_min": min(val), "best_epoch": val.index(min(val)),
            "starts_above_1": val[0] > 1.0}
    with CURVES.open("a") as fh:
        fh.write(json.dumps({"id": spec["id"], "val_layer_11_residual": val, "val_loss": vloss}) + "\n")
    # ---- evaluation on eval2 (clean env: no training switches), per-worker scratch dir
    evfs = fs.WORK / "fewshot_lk15" / wname
    (evfs / "configs").mkdir(parents=True, exist_ok=True)
    EV = EVAL2F if (EVAL2F / grid / 'meta.json').exists() else EVAL2
    fs.FS, fs.EVAL_DATA = evfs, EV
    fs.ENV = base_env
    try:
        ev = fs.evaluate(grid, rec)
    finally:
        fs.FS, fs.EVAL_DATA = FS_REAL, EVAL_REAL
    row |= {k: ev[k] for k in ("n_test_scenarios", "genco_active_res_mw", "genco_reactive_res_mvar", "dc_active_res_mw", "errors")}
    pred = pd.read_parquet(cm.pred_file(evfs, rec["key"], grid))
    acc = {}
    for o in cm.bundles(EV, grid, pred):
        for k, (vm, va) in cm.build_states(o, None).items():
            if k in ("dc", "fd2", "genco_raw"):
                acc.setdefault(k, []).append(cm.score(o, vm, va))
    row["ours"] = {k: cm.agg(np.array(v)) for k, v in acc.items()}
    row["n_scen_ours"] = len(acc["dc"])
    for k in ("genco_active_res_mw", "dc_active_res_mw"):
        assert math.isfinite(row[k]), k
    if not all(math.isfinite(x) for st in row["ours"].values() for x in st.values()):
        raise RuntimeError("non-finite scored metric")
    return row


def worker(a):
    queue = json.loads((HERE / "queue.json").read_text())
    if (HERE / "queue_amendment1.json").exists():
        queue += json.loads((HERE / "queue_amendment1.json").read_text())
    queue = sorted(queue, key=lambda q: q["prio"])
    CLAIMS.mkdir(exist_ok=True)
    os.nice(0)
    skipped = []
    for spec in queue:
        done = {json.loads(l)["id"] for l in RESULTS.read_text().splitlines() if "error" not in json.loads(l)} if RESULTS.exists() else set()
        if spec["id"] in done:
            continue
        est = EST[spec["n"]] * (1.3 if "118" in spec["grid"] else 1.0) + 400
        if a.deadline and time.time() + est > a.deadline:
            skipped.append(spec["id"]); continue
        g = spec["grid"]
        if not (((fs.WORK / "train" if spec.get("labels") == "ac" else DC_TRAIN_REAL) / g / "meta.json").exists() and (EVAL2 / g / "meta.json").exists() and (fs.EVAL_DATA / g / "meta.json").exists()):
            skipped.append(spec["id"] + " (data not ready)"); continue
        claim = CLAIMS / f"{spec['id']}.claim"
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        t0 = time.time()
        row = None
        for attempt in range(3):
            try:
                row = run_spec(spec, a.name)
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                print(f"[{a.name}] {spec['id']} attempt {attempt} failed: {err}", flush=True)
                row = {"id": spec["id"], "tag": spec["tag"], "grid": spec["grid"], "n": spec["n"], "split_seed": spec["split_seed"], "error": err}
                stale = FS_REAL / "runs" / f"lk15_{spec['tag']}_s{spec['split_seed']}__finetune__{spec['grid']}__n{spec['n']}.json"
                if "pretrained one" in err or "metrics check" in err:
                    break
                if stale.exists() and "train" in err:
                    stale.unlink()
                time.sleep(120)
        row["wall_seconds"] = round(time.time() - t0, 1)
        row["worker"] = a.name
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        o = row.get("ours", {})
        print(json.dumps({"id": row["id"], "val0": row.get("val_epoch0"), "ep": row.get("epochs_run"), "dva_ratio_dc": (o["genco_raw"]["branch_dva_deg"] / o["dc"]["branch_dva_deg"]) if o else None,
                          "err": row.get("error"), "sec": row["wall_seconds"]}), flush=True)
    print(f"[{a.name}] finished; skipped for deadline: {skipped}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("worker"); w.add_argument("--name", required=True); w.add_argument("--deadline", type=float, default=0)
    worker(ap.parse_args())
