"""eval2 of case240 has scenarios whose AC label is a collapsed root (Vm<0.5); fewshot._score refuses to score them (by design).
Build a filtered copy (those scenarios dropped from every raw/*.parquet) at eval2_filtered/<grid>; run.py uses it for this grid. UNAUDITED."""
import json, shutil, sys
from pathlib import Path
import pandas as pd
from genco_challenge import fewshot as fs
g = "pglib_opf_case240_pserc"
src = fs.WORK / "eval2" / g; dst = fs.WORK / "eval2_renumbered" / g
bad = fs.collapsed(src / "raw/bus_data.parquet"); print("collapsed scenarios:", sorted(bad))
keep = sorted(set(pd.read_parquet(src / 'raw/bus_data.parquet', columns=['scenario'])['scenario']) - bad)
remap = {int(o): i for i, o in enumerate(keep)}   # graphkit needs scenario ids 0..n-1
assert not dst.exists()
(dst / "raw").mkdir(parents=True)
for f in (src / "raw").iterdir():
    if f.name.endswith(".parquet") and (f.is_dir() or True):
        df = pd.read_parquet(f)
        if "scenario" in df.columns:
            n0 = df["scenario"].nunique(); df = df[~df["scenario"].isin(bad)].copy(); df['scenario'] = df['scenario'].map(remap).astype(int)
            print(f.name, n0, "->", df["scenario"].nunique())
        df.to_parquet(dst / "raw" / f.name)
    elif f.is_file():
        shutil.copy(f, dst / "raw" / f.name)
meta = json.loads((src / "meta.json").read_text()); n = int(pd.read_parquet(dst / "raw/bus_data.parquet", columns=["scenario"])["scenario"].nunique())
meta |= {"scenarios_written_unfiltered": meta["scenarios_written"], "scenarios_written": n, "dropped_collapsed": sorted(bad), "scenario_remap_old_to_new": remap}
(dst / "meta.json").write_text(json.dumps(meta, indent=1)); (dst / "raw/n_scenarios.txt").write_text(f"{n}\n")
