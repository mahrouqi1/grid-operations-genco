"""Shared constants for the 2026-10-26 mask-fraction sweep (descriptive only, UNAUDITED). Spec: protocol.json in this folder.
Thin layer: the 2026-10-21 modules are imported/run unchanged; only CONFIGS is overridden and output paths are redirected."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
V_DIR = R / "2026-10-21-genco-voltage-inference"
SI_DIR = R / "2026-10-21-genco-state-inference"
PIECE_DIRS = {"control": R / "2026-10-21-genco-voltage-inference-control", "fvs": R / "2026-10-21-genco-voltage-inference-family-vs-size"}
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
PY = "<CONDA_ENVS>/grid-operations-genco/bin/python"
RAW = HERE / "raw"
LOGS = HERE / "logs"


def cfg_name(p):
    return f"f{p}_s0"


def configs3():   # vcommon.CONFIGS format
    return [(cfg_name(p), p / 100, 0) for p in PCTS]


def configs4():   # ocommon.CONFIGS format (name, frac, seed_idx, noisy)
    return [(cfg_name(p), p / 100, 0, False) for p in PCTS]


def sets():
    """{set_name: (source_dir, [grids])}; locked15 from the 2026-10-21 protocol, pieces from their protocols' pre-registered subgrid lists."""
    out = {"locked15": (V_DIR, json.loads((V_DIR / "protocol.json").read_text())["grids"])}
    for k, d in PIECE_DIRS.items():
        out[k] = (d, list(json.loads((d / "protocol.json").read_text())["subgrids"]))
    return out
