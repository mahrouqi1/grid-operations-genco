"""Corrector-at-scale driver (UNAUDITED). A trimmed copy of results/2026-10-10-corrector-above-162/scripts/corr3.py: same training/evaluation
mechanics (fs.train / fs.evaluate with the corr3_plugins, same env switches, same scoring code), with ONLY these changes:
  * settings/paths: one setting per grid under data/external/genco-corrector-scale (trees/, priors/, splits/, fs_<case>/),
  * splits come from splits_info/<case>.json (train ids, fixed val ids) instead of the case1354 load-profile blocks,
  * per-grid training and evaluation batch sizes from protocol.json,
  * the classical comparators are fd_k for every k <= 30 (protocol.json), not only dc/fd2/nr1/nr2/fd3/fd4,
  * the case300/legacy settings, the probes, the selection stage and the mismatch-feature option are removed (the variant is fixed).
Env grid-operations-genco, CUDA_VISIBLE_DEVICES=1, repo root, PYTHONPATH=tracks/genco-challenge/src.
  python scale.py train  --case case2869pegase --n 100     train only (validation metrics are logged, no test scoring)
  python scale.py eval   --case case2869pegase --n 100     evaluate the trained run on the test split + all comparators"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[1] / "2026-10-09-genco-on-fd2/scripts"))
from genco_challenge import fewshot as fs
from genco_challenge import labelfree as lf           # patches fs.pool / split_dir / TRAIN_DATA; undone below
from genco_challenge import labelfree_prior2 as lp2
from genco_challenge import hybrid as hy
from common import bundles, fd_state, score, pred_file, wrap_rad, boot_ratio                  # noqa: E402
from s1_score import rows_for                                                      # noqa: E402

D = fs.REPO / "data/external/genco-corrector-scale"
PROTO = json.loads((OUT / "protocol.json").read_text())
BATCH = json.loads((OUT / "protocol_addendum3.json").read_text())["batch_size"]   # addendum 3 (2869 from addendum 2, 9241 lowered)
EVAL_BATCH = PROTO["corrector"]["eval_batch_size"]
FROZEN = r"model\.mlp_bus,model\.mlp_gen,model\.input_proj"
ENV0 = dict(fs.ENV)
VARIANTS = {"p0.1_lr0.0005": {"name": "p0.1_lr0.0005", "lr": 5e-4, "wp": 0.1},          # PRIMARY (pre-registered)
            "p0.1_lr5e-05": {"name": "p0.1_lr5e-05", "lr": 5e-5, "wp": 0.1},               # POST HOC (labelled), see report
            "p0.1_lr0.0005_rawfd2": {"name": "p0.1_lr0.0005_rawfd2", "lr": 5e-4, "wp": 0.1, "prior_dir": "priors_raw"}}   # POST HOC: raw fd2 input, no guard
VARIANT = VARIANTS["p0.1_lr0.0005"]
EPOCHS = {100: 250, 1000: 120}
KMAX = 30
CUR = {"case": None}
_orig_gk = fs._gk


def _gk_eval_small(cmd, log, env=None):
    if CUR["case"] and "evaluate" in cmd and "--batch_size" not in cmd:
        cmd = list(cmd) + ["--batch_size", str(EVAL_BATCH[CUR["case"]])]
    return _orig_gk(cmd, log, env)


fs._gk = _gk_eval_small


def use(case):
    CUR["case"] = case
    fs.TRAIN_DATA, fs.EVAL_DATA, fs.FS, fs.OUT = D / "trees/train", D / "trees/eval", D / "fs" / case, OUT
    fs.LOCAL_MLRUNS = Path("<LOCAL_SCRATCH>/genco-corrector/mlruns_scale_" + case)
    (fs.FS / "configs").mkdir(parents=True, exist_ok=True)
    (D / "scen").mkdir(parents=True, exist_ok=True)
    fs.split_dir = lambda grid, n, seed=0: D / "splits" / grid / f"n{n}_s{seed}"


def spec_of(case, n, draw=0, variant="p0.1_lr0.0005"):
    VARIANT = VARIANTS[variant]
    return {"name": VARIANT["name"], "case": case, "grid": case, "n": n, "draw": draw, "lr": VARIANT["lr"], "wp": VARIANT["wp"],
            "epochs": EPOCHS[n], "split_seed": draw}


def key_of(sp):
    return f"c3_{sp['name']}_d{sp['draw']}__finetune__{sp['grid']}__n{sp['n']}"


def pick_gpu(min_free=9000):
    while True:
        out = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        if out.strip() and int(out.strip().split(",")[1]) >= min_free:
            return "1"
        time.sleep(60)


PRIOR_DIR = {"name": "priors"}


def make_env(gpu, evaluating):
    e = dict(ENV0) | {"FEWSHOT_PRIOR_DIR": str(D / PRIOR_DIR["name"]), "FEWSHOT_FIX_SANITY": "1", "FEWSHOT_INPUT_NORM": "1", "FEWSHOT_VAL_DVA": "1",
                      "FEWSHOT_MONITOR": "Validation branch_dva", "PYTHONDONTWRITEBYTECODE": "1", "FEWSHOT_PRIOR_MODE": "corr",
                      "CUDA_VISIBLE_DEVICES": gpu, "FEWSHOT_PRIOR_ZERO_HEAD": "0" if evaluating else "1",
                      "FEWSHOT_EXTRA_PLUGIN": "genco_challenge.corr3_plugins", "FEWSHOT_TRAIN_DVA": "1"}
    fs.ENV = e


def overrides(sp):
    ep, wp = sp["epochs"], sp["wp"]
    return [f"optimizer.learning_rate={sp['lr']:g}", f"training.epochs={ep}", f"callbacks.patience={ep + 10}",
            f"training.loss_weights=[{wp:g},{1 - wp:g}]", f"seed={sp['draw']}", f"training.batch_size={BATCH[sp['case']]}"]


def write_split(sp):
    import torch
    d = fs.split_dir(sp["grid"], sp["n"], sp["split_seed"])
    if (d / "train.pt").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    info = json.loads((D / "splits_info" / f"{sp['case']}.json").read_text())
    ids = info["train_ids"]
    rng = np.random.default_rng(sp["draw"])
    tr = sorted(ids[i] for i in rng.permutation(len(ids))[:sp["n"]])
    va = info["val_ids"]
    for k, v in (("train", tr), ("val", va), ("test", va)):
        torch.save(torch.tensor(v, dtype=torch.long), d / f"{k}.pt")
    (d / "split.json").write_text(json.dumps({"n_train": len(tr), "n_val": len(va), "train": tr, "val": va}))
    return d


def val_metrics(rec):
    run_dir = Path(rec["model"]).parent
    S = lf._series
    dva, pri = S(run_dir, "Validation branch_dva"), S(run_dir, "Validation branch_dva_prior")
    steps = sorted(dva)
    step = min(dva, key=dva.get)
    ep = steps.index(step) + 1
    m = {"val_dva": dva[step], "val_dva_prior": float(np.mean(list(pri.values()))), "best_epoch": ep, "last_epoch": len(steps),
         "kept_pretrained": lf._is_pretrained(rec["model"]), "val_dva_curve": [dva[k] for k in steps]}
    m["val_dva_ratio"] = m["val_dva"] / m["val_dva_prior"]
    return m


def train_one(sp, retries=3):
    use(sp["case"])
    key = key_of(sp)
    rp = fs.FS / "runs" / f"{key}.json"
    if rp.exists():
        return json.loads(rp.read_text())
    write_split(sp)
    log = OUT / "train_log.jsonl"
    for a in range(retries + 1):
        gpu = pick_gpu()
        make_env(gpu, False)
        t0 = time.time()
        try:
            rec = fs.train(sp["grid"], sp["n"], "finetune", f"c3_{sp['name']}_d{sp['draw']}", overrides(sp), None, FROZEN, sp["split_seed"], True)
            if rec["epochs_run"] != sp["epochs"]:
                (fs.FS / "runs" / f"{key}.json").rename(fs.FS / "runs" / f"{key}.json.rejected")
                raise RuntimeError(f"ASSERT budget: {key} ran {rec['epochs_run']} of {sp['epochs']}")
            with log.open("a") as fh:
                fh.write(json.dumps({"key": key, "attempt": a, "gpu": gpu, "ok": True, "train_seconds": rec["train_seconds"], "epochs_run": rec["epochs_run"]}) + "\n")
            return rec
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            with log.open("a") as fh:
                fh.write(json.dumps({"key": key, "attempt": a, "gpu": gpu, "ok": False, "error": err[:500], "seconds": round(time.time() - t0, 1)}) + "\n")
            print(err, flush=True)
            if "locked:" in err:
                return None
            rp.unlink(missing_ok=True)
            time.sleep(90)
    return None


class Lite:
    pass


def light_bundles(root, grid, pred):
    scen = sorted(int(s) for s in pred["scenario"].unique())
    bus, ys, brs = lp2._grid_tables(root, grid, scen)
    keyed = pred.set_index(["scenario", "bus"]).sort_index()
    for s in scen:
        b = bus[bus["scenario"] == s].sort_values("bus")
        sc = lp2.input_problem(b, ys[s], brs[s])
        on = brs[s][brs[s]["br_status"] == 1]
        g = keyed.loc[s].sort_index()
        o = Lite()
        o.s, o.sc, o.n = s, sc, sc.n
        o.f, o.t = on["from_bus"].to_numpy(), on["to_bus"].to_numpy()
        o.vm_t, o.va_t = g["Vm_target"].to_numpy(), g["Va_target"].to_numpy()
        o.vm_g, o.va_g = g["Vm_pred"].to_numpy(), g["Va_pred"].to_numpy()
        yield o


def classical(grid, pred):
    """Scores of dc and fd_k, k = 2..KMAX (cached per grid). fd_k is the k-th XB fast-decoupled iterate from the DC start."""
    cp = D / "scen" / f"classical_{grid}.npz"
    ids = sorted(int(s) for s in pred["scenario"].unique())
    if cp.exists():
        z = np.load(cp)
        assert z["ids"].tolist() == ids, "scenario list differs from the cached classical scores"
        return {k: z[k] for k in z.files if k != "ids"}
    acc = {}
    for o in bundles(fs.EVAL_DATA, grid, pred):
        sc = o.sc
        acc.setdefault("dc", []).append(score(o, sc.vm_set, o.va_dc))
        vm, va = fd_state(o, "fd2")                      # fd2_input = the GUARDED fd2 state of lp2.priors_for, i.e. what the corrector receives
        acc.setdefault("fd2_input", []).append(score(o, vm, va))
        acc.setdefault("fd2_input_is_dc", []).append([float(np.array_equal(o.pr["fd2"]["v"], sc.vm_set * np.exp(1j * sc.va_dc)))] + [0.0] * 6)
        v = (sc.vm_set * np.exp(1j * sc.va_dc))          # fd_k = k-th RAW XB iterate from the DC start (addendum 4; matches the timing path)
        for k in range(1, KMAX + 1):
            try:
                vn = sc.fdxb_step(v)
                if hy._finite(vn):
                    v = vn
            except Exception:
                pass
            if k >= 2:
                acc.setdefault(f"fd{k}", []).append(score(o, np.abs(v), np.radians(lp2._unwrapped(sc, v))))
    acc = {k: np.array(v) for k, v in acc.items()}
    np.savez_compressed(cp, ids=np.array(ids), **acc)
    return acc


def eval_one(sp):
    use(sp["case"])
    key = key_of(sp)
    rec = json.loads((fs.FS / "runs" / f"{key}.json").read_text())
    row = {"key": key, "train_seconds": rec["train_seconds"], "epochs_run": rec["epochs_run"]} | val_metrics(rec)
    for a in range(5):
        gpu = pick_gpu(6000)
        make_env(gpu, True)
        try:
            r = fs.evaluate(sp["grid"], rec | {"key": key})
            break
        except Exception as exc:
            print("eval retry", exc, flush=True)
            time.sleep(60)
    else:
        raise RuntimeError(f"evaluate failed for {key}")
    pred = pd.read_parquet(pred_file(fs.FS, key, sp["grid"]))
    acc_c = classical(sp["grid"], pred)
    prior = np.load(D / "priors/eval" / f"{sp['grid']}.npy")
    mine, rms = [], []
    for o in light_bundles(fs.EVAL_DATA, sp["grid"], pred):
        mine.append(score(o, o.vm_g, o.va_g))
    acc = acc_c | {"corrector": np.array(mine)}
    assert len(acc["corrector"]) == len(acc["fd2"]) == len(acc["dc"])
    np.savez_compressed(D / "scen" / f"{key}.npz", ids=np.array(sorted(int(s) for s in pred["scenario"].unique())), **acc)
    # per-scenario results kept in the results dir too (small): ids + one (S,7) array per state
    (OUT / "per_scenario").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "per_scenario" / f"{sp['case']}_n{sp['n']}{'' if sp['name'] == VARIANT['name'] else '_' + sp['name']}.npz", ids=np.array(sorted(int(s) for s in pred["scenario"].unique())), **acc)
    keep = {k: acc[k] for k in ("dc", "fd2_input", "fd2", "fd3", "fd4", "fd5", "corrector")}
    rows = rows_for("test", sp["grid"], keep, extra={"variant": sp["name"], "case": sp["case"], "n": sp["n"], "draw": sp["draw"], "key": key})
    with (OUT / "results.jsonl").open("a") as fh:
        for rr in rows:
            fh.write(json.dumps(rr | {"graphkit_active_res_mw": r["genco_active_res_mw"], "epochs_run": row["epochs_run"], "budget": sp["epochs"],
                                      "train_seconds": row["train_seconds"], "lr": sp["lr"], "wp": sp["wp"], "n_scored": len(mine),
                                      **{k: row[k] for k in row if k not in ("key", "train_seconds", "epochs_run") and not k.endswith("_curve")}}) + "\n")
    with (OUT / "curves.jsonl").open("a") as fh:
        fh.write(json.dumps({"key": key, "case": sp["case"], "n": sp["n"], **{k: row[k] for k in row if k.endswith("_curve")}, "val_dva_prior": row["val_dva_prior"]}) + "\n")
    print(json.dumps({"key": key, "n_scored": len(mine)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--case", required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variant", default="p0.1_lr0.0005")
    a = ap.parse_args()
    sp = spec_of(a.case, a.n, 0, a.variant)
    PRIOR_DIR["name"] = VARIANTS[a.variant].get("prior_dir", "priors")
    if a.cmd == "train":
        train_one(sp)
    elif a.cmd == "eval":
        eval_one(sp)
