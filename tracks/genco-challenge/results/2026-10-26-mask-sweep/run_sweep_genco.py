"""GENCO zero-shot at the sweep fractions: runs the UNCHANGED 2026-10-21 run_genco_v.py (locked15) or run_genco_o.py (pieces) with CONFIGS
overridden. locked15: all grids in one process -> raw/locked15/genco.jsonl. pieces: one grid per call -> raw/<set>/genco_<grid>.jsonl.
CUDA_VISIBLE_DEVICES=<g> python run_sweep_genco.py --set locked15 [--grid G] ; UNAUDITED."""
import argparse, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as S

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--grid", default=None); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
src, _ = S.sets()[a.set]
sys.path.insert(0, str(src))
(S.RAW / a.set).mkdir(parents=True, exist_ok=True)
if a.set == "locked15":
    import vcommon as V; V.CONFIGS = S.configs3()
    sys.argv = ["run_genco_v.py", "--device", a.device, "--out", str(S.RAW / a.set / "genco.jsonl")]
    script = "run_genco_v.py"
else:
    import ocommon as O; O.CONFIGS = S.configs4()
    out = S.RAW / a.set / f"genco_{a.grid}.jsonl"
    if out.exists(): out.unlink()
    sys.argv = ["run_genco_o.py", "--grids", a.grid, "--device", a.device, "--out", str(out)]
    script = "run_genco_o.py"
runpy.run_path(str(src / script), run_name="__main__")
