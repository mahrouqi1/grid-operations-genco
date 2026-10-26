"""CPU-only: run graphkit's own raw -> processed conversion for a tree so the training start does not wait for it (same class and config as training/eval).
usage: preprocess.py <case> <train|eval>"""
import json, sys, yaml
from pathlib import Path
from genco_challenge import fewshot as fs
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
case, tree = sys.argv[1:3]
root = fs.REPO / "data/external/genco-corrector-scale/trees" / tree
cfg = yaml.safe_load((fs.GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
cfg["data"]["networks"] = [case]
cfg["data"]["scenarios"] = [json.loads((root / case / "meta.json").read_text())["scenarios_written"]]
cfg["data"]["workers"] = 0
dm = LitGridHeteroDataModule(NestedNamespace(**cfg), str(root), normalizer_stats_path=None)
dm.setup("test")
print("processed", case, tree, len(dm.datasets[0]))
