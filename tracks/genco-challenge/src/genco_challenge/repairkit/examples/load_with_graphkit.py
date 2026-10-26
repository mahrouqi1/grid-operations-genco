"""Load a repaired raw dir with gridfm_graphkit's HeteroGridDatasetDisk and iterate a few samples (CPU only).

graphkit reads ``data_index_<idx>.pt`` for idx in 0..N-1, so scenario ids must be contiguous. The repaired dir keeps the
original ids (gaps in a sampled dataset), therefore this script works on a COPY: it copies the raw dir into
``<work>/<name>/raw``, renumbers the copy with ``genco_challenge.renumber_raw`` (writes ``scenario_map.csv`` with the
original ids), and builds the dataset there. The repaired dir itself is never modified.

    PYTHONPATH=tracks/genco-challenge/src:core:data/external/gridfm-graphkit-repro-pretraining \\
      python tracks/genco-challenge/src/genco_challenge/repairkit/examples/load_with_graphkit.py \\
        data/external/genco-repairkit-demo/subnet_case10480_goc_idx_40/raw --work /path/to/scratch --n 3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

GRAPHKIT = Path(__file__).resolve().parents[6] / "data/external/gridfm-graphkit-repro-pretraining"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path, help="a repaired (or any datakit-format) raw dir")
    ap.add_argument("--work", type=Path, required=True, help="scratch dir for the renumbered copy")
    ap.add_argument("--n", type=int, default=3, help="samples to iterate")
    ap.add_argument("--only-repaired", action="store_true", help="iterate the repaired scenarios (needs repair_log.csv)")
    a = ap.parse_args(argv)
    if str(GRAPHKIT) not in sys.path:
        sys.path.insert(0, str(GRAPHKIT))
    from genco_challenge import renumber_raw
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk

    root = a.work / "graphkit_root"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(a.raw_dir, root / "raw", ignore=shutil.ignore_patterns("originals.parquet"))
    renumber_raw.renumber(root / "raw")
    smap = pd.read_csv(root / "raw/scenario_map.csv") if (root / "raw/scenario_map.csv").exists() else None
    norm = HeteroDataMVANormalizer(SimpleNamespace(data=SimpleNamespace(baseMVA=100, normalizationMVA=100)))
    ds = HeteroGridDatasetDisk(str(root), "mvanorm", norm)
    print("samples:", len(ds))
    idx = list(range(min(a.n, len(ds))))
    log = a.raw_dir / "repair_log.csv"
    if a.only_repaired and smap is not None and log.exists():
        rec = set(pd.read_csv(log).query("status == 'recovered'").scenario)
        idx = smap.loc[smap.original_scenario.isin(rec), "scenario"].tolist()[: a.n]
    for i in idx:
        d = ds[i]
        orig = int(smap.original_scenario.iloc[i]) if smap is not None else i
        print(f"idx {i} (original scenario {orig}):", {k: tuple(d[k].x.shape) for k in d.node_types},
              "min bus Vm (normalised feature 3):", float(d["bus"].x[:, 3].min()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
