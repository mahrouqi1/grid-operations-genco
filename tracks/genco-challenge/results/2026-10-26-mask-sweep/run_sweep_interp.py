"""One interpolator task (set, grid, pct): runs the UNCHANGED 2026-10-21 run_interp.py with CONFIGS overridden to the sweep fractions
and --configs f{pct}_s0, writing raw/<set>/interp_<grid>_<cfg>.jsonl. UNAUDITED.
    python run_sweep_interp.py --set locked15|control|fvs --grid G --pcts 30 40"""
import argparse, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as S

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--grid", required=True); ap.add_argument("--pcts", type=int, nargs="+", required=True)
a = ap.parse_args()
src, _ = S.sets()[a.set]
sys.path.insert(0, str(src))
tag = S.cfg_name(a.pcts[0]) if len(a.pcts) == 1 else "all"
out = S.RAW / a.set / f"interp_{a.grid}_{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): out.unlink()                       # a partial file from an interrupted task is never reused
if a.set == "locked15":
    import vcommon as V; V.CONFIGS = S.configs3()
else:
    import ocommon as O; O.CONFIGS = S.configs4()
sys.argv = ["run_interp.py", "--grids", a.grid, "--configs", *[S.cfg_name(p) for p in a.pcts], "--out", str(out)]
runpy.run_path(str(src / "run_interp.py"), run_name="__main__")
