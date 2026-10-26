"""Rewrite the RTE validation trees (data/external/genco-transfer-val/<case>/raw) in the layout graphkit `evaluate
--compute_dc_ac_metrics` reads: hive-partitioned parquet dirs with scenario_partition = scenario // 200
(gridfm_graphkit.tasks.pf_ac_dc_baseline.N_SCENARIO_PER_PARTITION) and a runtime_data table. The converter writes
single files (training does not need partitions). runtime_data carries no ac/dc timing columns (we have none); the
metric code skips absent timing modes. Values are unchanged. 2026-10-20."""
import shutil
import sys
from pathlib import Path

import pandas as pd

for d in sys.argv[1:]:
    raw = Path(d) / "raw"
    for t in ("bus_data", "branch_data", "gen_data", "y_bus_data"):
        f = raw / f"{t}.parquet"
        if f.is_dir():
            continue                                   # already partitioned
        df = pd.read_parquet(f)
        df["scenario_partition"] = (df["scenario"] // 200).astype("int64")
        tmp = raw / f"{t}.parquet.part"
        df.to_parquet(tmp, partition_cols=["scenario_partition"], index=False)
        f.unlink()
        tmp.rename(f)
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario")
    rt = b.sort_values("scenario").reset_index(drop=True)
    rt["scenario_partition"] = (rt["scenario"] // 200).astype("int64")
    if not (raw / "runtime_data.parquet").exists():
        rt.to_parquet(raw / "runtime_data.parquet", partition_cols=["scenario_partition"], index=False)
    print(d, "ok", len(rt), "scenarios")
