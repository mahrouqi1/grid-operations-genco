"""Stage 1 scoring. `val`: blend selection + gate input on validation sets. `eval`: report on eval2.
    python s1_score.py val
    python s1_score.py eval [--tag eval2|eval2defaultnorm]
Appends rows to ../results.jsonl (one row per split, grid, state)."""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd
from common import *
from genco_challenge import labelfree as lf
from genco_challenge.fresh_eval_data import EVAL2

GRID5 = [0, 0.25, 0.5, 0.75, 1.0]
RES = OUT / "results.jsonl"
SEL = OUT / "stage1_blend_selection.json"


def collect(root, grid, tag, blend_list, e50=False):
    """returns {state: (S,7) array}. blend_list: list of (w_vm,w_pv,w_pq) to evaluate as 'd[...]' states."""
    pred = pd.read_parquet(pred_file(W / f"fs_{tag}", f"zeroshot_{tag}_{grid}", grid))
    acc = {}
    for o in bundles(root, grid, pred):
        st = build_states(o, None, e50)
        for bl in blend_list:
            st[f"d{tuple(bl)}"] = build_states(o, bl)["d_blend"]
        for k, (vm, va) in st.items():
            acc.setdefault(k, []).append(score(o, vm, va))
    return {k: np.array(v) for k, v in acc.items()}


def rows_for(split, grid, acc, n_pred_info=None, extra=None):
    ref = acc["fd2"]
    out = []
    for k, a in acc.items():
        r = {"stage": 1, "split": split, "grid": grid, "state": k, "n_scen": len(a), **agg(a)}
        for name, kind, key in (("dva", "dva", "branch_dva"), ("va", "va", "va"), ("vm", "vm", "vm"), ("p", "p", "p_res"), ("q", "q", "q_res")):
            pt, lo, hi = boot_ratio(a, ref, kind)
            r[f"ratio_{key}_vs_fd2"], r[f"ratio_{key}_lo"], r[f"ratio_{key}_hi"] = pt, lo, hi
        r["win_frac_dva_vs_fd2"] = float(np.mean(a[:, 0] / a[:, 1] < ref[:, 0] / ref[:, 1]))
        if extra:
            r.update(extra)
        out.append(r)
    return out


def write(rows):
    with RES.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def val():
    angle_grid = list(itertools.product(GRID5, GRID5))
    blends = [(w, 0.0, 0.0) for w in GRID5 if w > 0] + [(0.0, a, b) for a, b in angle_grid if (a, b) != (0, 0)]
    accs = {g: collect(lf.fs.WORK / "genco_on_fd2/val", g, "val", blends) for g in lf.DEV}
    # selection
    def gm(f):
        return float(np.exp(np.mean([np.log(f(accs[g])) for g in lf.DEV])))
    vm_sc = {w: gm(lambda A: agg(A[f"d{(w,0.0,0.0)}"] if w > 0 else A["fd2"])["vm_pu"] / agg(A["fd2"])["vm_pu"]) for w in GRID5}
    an_sc = {(a, b): gm(lambda A: agg(A[f"d{(0.0,a,b)}"] if (a, b) != (0, 0) else A["fd2"])["branch_dva_deg"] / agg(A["fd2"])["branch_dva_deg"])
             for a, b in angle_grid}
    w_vm = min(GRID5, key=lambda w: (vm_sc[w], w))
    wa = min(angle_grid, key=lambda ab: (an_sc[ab], ab[0] + ab[1]))
    sel = {"w_vm": w_vm if vm_sc[w_vm] <= 0.98 else 0.0, "w_pv": wa[0] if an_sc[wa] <= 0.98 else 0.0,
           "w_pq": wa[1] if an_sc[wa] <= 0.98 else 0.0, "vm_scores": {str(k): v for k, v in vm_sc.items()},
           "angle_scores": {str(k): v for k, v in an_sc.items()}, "best_vm_score": vm_sc[w_vm], "best_angle_score": an_sc[wa],
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    SEL.write_text(json.dumps(sel, indent=1))
    bl = (sel["w_vm"], sel["w_pv"], sel["w_pq"])
    # val rows for the gate: a, b, c, d (selected). d at selected weights recomputed from the grid when possible
    for g in lf.DEV:
        A = accs[g]
        keep = {k: A[k] for k in ("dc", "fd2", "nr1", "genco_raw", "genco_pinned", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va")}
        # d at the combined selected weights needs a fresh pass
        pred = pd.read_parquet(pred_file(W / "fs_val", f"zeroshot_val_{g}", g))
        dd = [score(o, *build_states(o, bl)["d_blend"]) for o in bundles(lf.fs.WORK / "genco_on_fd2/val", g, pred)]
        keep["d_blend"] = np.array(dd)
        write(rows_for("val", g, keep, extra={"blend": list(bl)}))
    print(json.dumps({k: sel[k] for k in ("w_vm", "w_pv", "w_pq", "best_vm_score", "best_angle_score")}))


def evalset(tag):
    sel = json.loads(SEL.read_text())
    bl = (sel["w_vm"], sel["w_pv"], sel["w_pq"])
    for g in fs.GRIDS:
        pred = pd.read_parquet(pred_file(W / f"fs_{tag}", f"zeroshot_{tag}_{g}", g))
        acc = {}
        for o in bundles(EVAL2, g, pred):
            for k, (vm, va) in build_states(o, bl, e50=True).items():
                acc.setdefault(k, []).append(score(o, vm, va))
        acc = {k: np.array(v) for k, v in acc.items()}
        np.savez_compressed(W / f"scen_{tag}_{g}.npz", **acc)
        write(rows_for(tag, g, acc, extra={"blend": list(bl)}))
        print(g, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--tag", default="eval2")
    a = ap.parse_args()
    val() if a.cmd == "val" else evalset(a.tag)
