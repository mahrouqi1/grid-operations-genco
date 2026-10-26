"""Fetch branch_data + y_bus_data for the same partitions of the pretraining subgrids that genco_data_check.py
sampled bus_data/gen_data for (needed to run GENCO and Newton on those scenarios). UNAUDITED."""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import re
REPO = Path(__file__).resolve().parents[4]
PRE = REPO / "data/external/genco-pf-pretraining"
grids = sys.argv[1:]
fs = HfApi().dataset_info("gridfm/genco-pf-pretraining").siblings
for g in grids:
    parts = sorted(int(p.name.split("=")[1]) for p in (PRE / g / "raw/bus_data.parquet").glob("scenario_partition=*"))
    for f in fs:
        m = re.match(rf"{g}/raw/(branch_data|y_bus_data)\.parquet/scenario_partition=(\d+)/", f.rfilename)
        if m and int(m.group(2)) in parts:
            hf_hub_download("gridfm/genco-pf-pretraining", f.rfilename, repo_type="dataset", local_dir=PRE)
    print("done", g, parts, flush=True)
