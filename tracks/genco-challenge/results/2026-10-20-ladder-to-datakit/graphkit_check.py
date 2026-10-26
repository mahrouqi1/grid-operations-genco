"""Build graphkit graphs from a converted ladder raw dir (HeteroGridDatasetDisk) and compare per-feature
ranges with the published reference's processed graphs. CPU only.

    python graphkit_check.py <converted case dir> <scratch dir> [<reference subnet dir>]
Uses the repro graphkit under data/external/gridfm-graphkit-repro-pretraining (put on sys.path here).
"""
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "data/external/gridfm-graphkit-repro-pretraining"))
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer  # noqa: E402
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk  # noqa: E402

BUS = ["Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF", "min_vm_pu", "max_vm_pu", "min_q", "max_q", "GS", "BS", "vn_kv"]
GEN = ["p_mw", "min_p", "max_p", "cp0", "cp1", "cp2", "in_service"]
EDGE = ["pf", "qf", "Yff_r", "Yff_i", "Yft_r", "Yft_i", "tap", "ang_min", "ang_max", "rate_a", "br_status"]


def ranges(files):
    acc = {"bus": [], "gen": [], "edge": []}
    for f in files:
        d = torch.load(f, weights_only=False)   # HeteroData (ours) or its dict form (published processed/)
        g = (lambda k, a: d[k][a]) if isinstance(d, dict) else (lambda k, a: getattr(d[k], a))
        acc["bus"].append(g("bus", "x").numpy()); acc["gen"].append(g("gen", "x").numpy())
        acc["edge"].append(g(("bus", "connects", "bus"), "edge_attr").numpy())
    out = {}
    for k, names in (("bus", BUS), ("gen", GEN), ("edge", EDGE)):
        M = np.concatenate(acc[k])
        out[k] = {n: [float(M[:, i].min()), float(M[:, i].max())] for i, n in enumerate(names)}
    return out


def main():
    case_dir, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    ref = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else REPO / "data/external/genco-pf-pretraining-eval/subnet_case4837_goc_idx_7"
    (scratch / "raw").mkdir(parents=True, exist_ok=True)
    for t in ("bus_data", "gen_data", "branch_data"):
        dst = scratch / "raw" / f"{t}.parquet"
        if not dst.exists():
            os.symlink(case_dir / "raw" / f"{t}.parquet", dst)
    args = SimpleNamespace(data=SimpleNamespace(baseMVA=100, normalizationMVA=100))
    ds = HeteroGridDatasetDisk(root=str(scratch), norm_method="mvanorm", data_normalizer=HeteroDataMVANormalizer(args))
    n = len(ds)
    g = ds[0]  # also exercises transform()
    proc = sorted(glob.glob(str(scratch / "processed/data_index_*.pt")))[:50]
    mine = ranges(proc)
    theirs = ranges(sorted(glob.glob(str(ref / "processed/data_index_*.pt")))[:50])
    nan_or_inf = any(not np.isfinite(g[k].x.numpy()).all() for k in ("bus", "gen"))
    res = {"case": case_dir.name, "n_graphs": n, "x_bus_shape": list(g["bus"].x.shape), "finite": not nan_or_inf,
           "ranges_mine": mine, "ranges_reference": theirs}
    print(json.dumps({k: res[k] for k in ("case", "n_graphs", "x_bus_shape", "finite")}))
    for grp in mine:
        for n_, (lo, hi) in mine[grp].items():
            rlo, rhi = theirs[grp][n_]
            print(f"{grp:5s} {n_:10s} mine [{lo:11.4g},{hi:11.4g}]  ref [{rlo:11.4g},{rhi:11.4g}]")
    (scratch / "graphkit_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
