"""Not a timing run: copies the PowerModels solve times recorded by the datakit at data generation (runtime_data.parquet, 'ac' and 'dc'
columns, seconds, recorded under load 50 to 70 per the earlier reports) into timing.jsonl so every table number comes from one file."""
import pandas as pd
from tcommon import *
for g, (root, name, fam) in GRIDS.items():
    r = pd.read_parquet(root / name / "raw" / "runtime_data.parquet")
    emit(kind="pm_recorded", grid=g, n_bus=N_BUS[g], **{f"ac_{k}": v for k, v in stats(r["ac"].to_numpy()).items()},
         **{f"dc_{k}": v for k, v in stats(r["dc"].dropna().to_numpy()).items()}, n_scen=int(len(r)))
