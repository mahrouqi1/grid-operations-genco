"""SUPPLEMENTARY (outside the pre-registered gate; no new training): existing label-free adapted GENCO checkpoints
scored in the 'GENCO on top of fd2' framing on eval2.
  score-dc : the 27 clean-run checkpoints (lfc_w1_lr3e-05_all, DC prior, 9 grids x draws 0-2, n=1000), predictions
             from fewshot_fresh/mlruns_eval (already produced by fresh_eval.py); variants raw, pinned, b, c, e50.
  eval-fd2 : evaluate the two existing fd2-prior runs (lfp2_fd2_w1_lr3e-05_all, case14/case30, draw 0) on eval2.
  score-fd2: same variants on those predictions.
"""
import argparse, json, subprocess, time
import numpy as np, pandas as pd
from common import *
from s1_score import rows_for, write
from genco_challenge.fresh_eval_data import EVAL2
import s1_predict as sp1

FRESH = fs.WORK / "fewshot_fresh"
DEVG = ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee"]


def score_keys(tag, items):
    """items: [(grid, draw, fsdir, key)] -> rows with split=tag"""
    for g, d, fsdir, key in items:
        pred = pd.read_parquet(pred_file(fsdir, key, g))
        acc = {}
        for o in bundles(EVAL2, g, pred):
            for k, (vm, va) in build_states(o, None, e50=True).items():
                acc.setdefault(k, []).append(score(o, vm, va))
        acc = {k: np.array(v) for k, v in acc.items()}
        np.savez_compressed(W / f"scen_{tag}_{g}_d{d}.npz", **acc)
        write(rows_for(tag, g, acc, extra={"draw": d, "key": key}))
        print(tag, g, d, flush=True)


def keys_dc():
    out = []
    for g in fs.GRIDS:
        for d, suf in enumerate(["", "_s1", "_s2"]):
            out.append((g, d, FRESH, f"lfc_w1_lr3e-05_all_slackvm{suf}__finetune__{g}__n1000"))
    return out


def eval_fd2():
    fs.EVAL_DATA, fs.FS = EVAL2, W / "fs_s2fd2"
    (fs.FS / "configs").mkdir(parents=True, exist_ok=True)
    fs.ENV = {k: v for k, v in fs.ENV.items() if k != "FEWSHOT_INPUT_NORM"} | {"FEWSHOT_INPUT_NORM": "1"}
    for g in DEVG:
        key = f"lfp2_fd2_w1_lr3e-05_all_slackvm__finetune__{g}__n1000"
        rec = json.loads((fs.WORK / "fewshot/runs" / f"{key}.json").read_text())
        for attempt in range(4):
            try:
                fs.ENV = fs.ENV | {"CUDA_VISIBLE_DEVICES": sp1.pick_gpu()}
                r = fs.evaluate(g, rec)
                (W / f"score_s2fd2_{g}.json").write_text(json.dumps(r))
                break
            except Exception as exc:
                print("retry", exc, flush=True); time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); a = ap.parse_args()
    if a.cmd == "score-dc":
        score_keys("supp_dcprior_adapted", keys_dc())
    elif a.cmd == "eval-fd2":
        eval_fd2()
    else:
        score_keys("supp_fd2prior_adapted", [(g, 0, W / "fs_s2fd2", f"lfp2_fd2_w1_lr3e-05_all_slackvm__finetune__{g}__n1000") for g in DEVG])
