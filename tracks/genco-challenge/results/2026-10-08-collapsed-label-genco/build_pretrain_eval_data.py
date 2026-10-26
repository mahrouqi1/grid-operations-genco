"""Assemble the sampled pretraining partitions of one subgrid into a single unpartitioned graphkit raw/ dir
(scenarios renumbered 0..N-1, map saved) so ``gridfm_graphkit evaluate`` can run the PRETRAINED GENCO on them.
No scenario is dropped or altered. Run (env grid-operations-genco, repo root):

    python tracks/genco-challenge/results/2026-10-08-collapsed-label-genco/build_pretrain_eval_data.py SUBGRID...
UNAUDITED.
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "data/external/genco-pf-pretraining"
DST = REPO / "data/external/genco-pf-pretraining-eval"
TABLES = ("bus_data", "gen_data", "branch_data", "y_bus_data")

for g in sys.argv[1:]:
    parts = sorted(int(p.name.split("=")[1]) for p in (SRC / g / "raw/bus_data.parquet").glob("scenario_partition=*"))
    out = DST / g / "raw"
    out.mkdir(parents=True, exist_ok=True)
    smap = None
    for t in TABLES:
        df = pd.concat([pd.read_parquet(SRC / g / f"raw/{t}.parquet/scenario_partition={p}") for p in parts], ignore_index=True)
        df = df.drop(columns=[c for c in ("scenario_partition",) if c in df.columns])
        if smap is None:
            ids = sorted(df["scenario"].unique())
            smap = pd.DataFrame({"new": range(len(ids)), "orig": ids})
        df["scenario"] = df["scenario"].map(dict(zip(smap["orig"], smap["new"]))).astype("int64")
        df = df.sort_values(["scenario"] + (["bus"] if t == "bus_data" else [])).reset_index(drop=True)
        df.to_parquet(out / f"{t}.parquet", index=False)
    smap.to_csv(out / "scenario_map.csv", index=False)
    print(g, "partitions", parts, "scenarios", len(smap), flush=True)
