"""Add runtime_data.parquet (scenario, load_scenario_idx) to each converted tree so graphkit's datamodule accepts it; asserts scenarios are 0..N-1. UNAUDITED."""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grids as G
for g in G.SEL:
    raw = G.ROOT / g / "raw"
    if not (raw / "bus_data.parquet").exists(): print(g, "no data"); continue
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
    assert list(b.scenario) == list(range(len(b))), g
    b.to_parquet(raw / "runtime_data.parquet", index=False); print(g, len(b))
