"""F2 control (audit genco-make-it-work R1-F2) (UNAUDITED). Rebuilds the wu100L (AC-labelled) splits with the collapse-filtering
fewshot.pool (run.py's `import common` -> labelfree patches fs.pool to the label-free pool; the original is saved BEFORE that import)
into fewshot/splits_L_poolfix, and reruns the same recipe (run.py run_spec, unchanged except the splits dir and the variant tag
'wu100Lpf') for the requested grid/draws. Results: ../results.jsonl. Nothing in locked15 is read-modified.
    python rerun.py splits                       # write + compare splits only
    python rerun.py run --grid G --draws 0 1 2   # one worker, sequential
"""
import argparse, inspect, json, sys, textwrap, time
from pathlib import Path
from genco_challenge import fewshot as fs
ORIG_POOL = fs.pool                       # the collapse-filtering pool, before labelfree patches it
HERE = Path(__file__).resolve().parent.parent
L15 = HERE.parent / "2026-10-21-fewshot-locked15"
sys.path.insert(0, str(L15 / "scripts"))
import run as R                           # noqa  (patches fs.pool; run.py's own module-level constants)
fs.pool = ORIG_POOL
R.RESULTS, R.CURVES = HERE / "results.jsonl", HERE / "curves.jsonl"
SPL = "splits_L_poolfix"
src = textwrap.dedent(inspect.getsource(R.run_spec)).replace('"splits_L"', f'"{SPL}"')
exec(compile(src, "run_spec_poolfix", "exec"), R.__dict__)       # redefines R.run_spec with the new splits dir


def compare(grid):
    fs.TRAIN_DATA = fs.WORK / "train"
    bad = fs.collapsed(fs.TRAIN_DATA / grid / "raw/bus_data.parquet")
    res = {"pool_size_fixed": len(ORIG_POOL(grid)), "pool_size_labelfree_used": len(R.lf_pool(grid)) if hasattr(R, "lf_pool") else None, "draws": {}}
    for s in (0, 1, 2):
        sub = f"n100" if s == 0 else f"n100_s{s}"
        old = json.load(open(R.FS_REAL / "splits_L" / grid / sub / "split.json"))["ids"]
        fs.split_dir = lambda g, n_, seed=0: R.FS_REAL / SPL / g / (f"n{n_}" if seed == 0 else f"n{n_}_s{seed}")
        fs.write_splits(grid, s)
        new = json.load(open(R.FS_REAL / SPL / grid / sub / "split.json"))["ids"]
        o, n = set(sum(old.values(), [])), set(sum(new.values(), []))
        res["draws"][s] = {"old_collapsed": len(o & bad), "new_collapsed": len(n & bad), "same_split": old == new,
                           "overlap_old_new": len(o & n), "old_train_collapsed": len(set(old["train"]) & bad), "old_test_collapsed": len(set(old["test"]) & bad)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--grid", default="pglib_opf_case240_pserc"); ap.add_argument("--draws", type=int, nargs="*", default=[0, 1, 2]); ap.add_argument("--name", default="pf1")
    a = ap.parse_args()
    if a.cmd == "splits":
        out = compare(a.grid); (HERE / "splits_comparison.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); sys.exit()
    for s in a.draws:
        spec = {"id": f"wu100Lpf_{a.grid}_n100_s{s}", "prio": 0, "grid": a.grid, "n": 100, "seed": s, "split_seed": s, "tag": "wu100Lpf", "labels": "ac",
                "env": {"FEWSHOT_WARMUP_STEPS": "100"}, "epochs": 200, "note": "F2 pool-fix rerun of wu100L"}
        t0 = time.time()
        try:
            row = R.run_spec(spec, a.name)
        except Exception as exc:
            row = {"id": spec["id"], "error": f"{type(exc).__name__}: {exc}"}
        row["wall_seconds"] = round(time.time() - t0, 1)
        with R.RESULTS.open("a") as fh: fh.write(json.dumps(row) + "\n")
        print(spec["id"], row.get("error"), row["wall_seconds"], flush=True)
