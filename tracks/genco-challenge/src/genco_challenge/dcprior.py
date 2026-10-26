"""Label-free training data for GENCO: AC voltage labels replaced by a DC power-flow prior.

Physics-only (label-free) adaptation can settle on a wrong low-residual state: with the slack |V|
pinned, case14 converged to all non-slack angles rotated ~-2.2 rad from the slack and a 0.55 pu bus
beside it (the slack bus's unknown Pg/Qg absorb the mismatch). A label-free anchor exists: the DC
power-flow solution, computed from inputs only (datakit stores it as ``Va_dc``). This module writes
a copy of a grid's training set whose bus voltage labels are that prior:

- ``Va`` := ``Va_dc`` (DC power-flow angle, degrees, slack-referenced);
- ``Vm`` := 1.0 pu at PQ buses; the PV / slack setpoints, which power flow specifies, are kept.

Every other column (inputs) is unchanged, so training with ``loss_weights=[physics, w]`` pulls the
prediction toward the DC state with weight ``w`` while the physics loss corrects it. The AC
solution is never read as a training target. Other raw files are symlinked.

    python -m genco_challenge.dcprior --grids pglib_opf_case14_ieee pglib_opf_case30_ieee
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from genco_challenge.fewshot import TRAIN_DATA, WORK

DC_TRAIN = WORK / "train_dcprior"


def build(grid: str) -> dict:
    src, dst = TRAIN_DATA / grid, DC_TRAIN / grid
    (dst / "raw").mkdir(parents=True, exist_ok=True)
    for f in (src / "raw").iterdir():
        if f.name != "bus_data.parquet" and not (dst / "raw" / f.name).exists():
            os.symlink(f, dst / "raw" / f.name)
    if not (dst / "meta.json").exists():
        os.symlink(src / "meta.json", dst / "meta.json")
    b = pd.read_parquet(src / "raw/bus_data.parquet")
    pq = b["PQ"].to_numpy() == 1
    out = b.copy()
    out["Va"] = b["Va_dc"]
    out["Vm"] = np.where(pq, 1.0, b["Vm"])
    out.to_parquet(dst / "raw/bus_data.parquet")
    # how far the prior is from the AC solution (for the record only; not used in training)
    return {"grid": grid, "rows": len(out), "prior_va_mae_deg": float((b["Va"] - b["Va_dc"]).abs().mean()),
            "prior_vm_mae_pu_pq": float((b.loc[pq, "Vm"] - 1.0).abs().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", required=True)
    args = ap.parse_args()
    for g in args.grids:
        print(json.dumps(build(g)), flush=True)


if __name__ == "__main__":
    main()
