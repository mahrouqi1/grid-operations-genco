"""Run the PRETRAINED GENCO checkpoint (zero-shot, HF gridfm/genco-pf-transfer pretraining/) with graphkit's own
``evaluate --save_output`` on a data dir of unpartitioned raw/ tables. Same command and config as
``genco_challenge.ladder_eval.run_graphkit`` (GENCO's metric path unchanged), only the data path and the split differ:
test_ratio 0.9999, val_ratio 0 (the splitter rejects 1.0; this leaves ~1 load scenario out) so nearly every scenario is predicted. UNAUDITED.

    python run_genco_inference.py --data <dir containing GRID/raw> --mlruns <dir> GRID...
"""
import argparse, glob, os, subprocess, sys
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[4]
GK = REPO / "data/external/gridfm-graphkit-repro"
CKPT = REPO / "data/external/genco-pf-transfer/pretraining/best_model_state_dict.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--mlruns", type=Path, required=True)
ap.add_argument("--cfgdir", type=Path, required=True)
ap.add_argument("grids", nargs="+")
a = ap.parse_args()
a.data, a.mlruns, a.cfgdir = a.data.resolve(), a.mlruns.resolve(), a.cfgdir.resolve()
a.cfgdir.mkdir(parents=True, exist_ok=True)
for g in a.grids:
    n = len(set(__import__("pandas").read_parquet(a.data / g / "raw/bus_data.parquet", columns=["scenario"])["scenario"]))
    cfg = yaml.safe_load((GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
    cfg["data"].update(networks=[g], scenarios=[n], test_ratio=0.9999, val_ratio=0.0)
    cp = a.cfgdir / f"eval_{g}.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    cmd = ["gridfm_graphkit", "evaluate", "--config", str(cp), "--data_path", str(a.data), "--model_path", str(CKPT),
           "--save_output", "--num_workers", "4", "--log_dir", str(a.mlruns), "--run_name", f"zeroshot__{g}"]
    env = os.environ | {"MLFLOW_ALLOW_FILE_STORE": "true", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    log = a.cfgdir / f"eval_{g}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=GK, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    print(g, "rc", rc, flush=True)
