"""Renumber the scenarios of one graphkit raw dir to 0..N-1, in place, consistently across its four tables.

Why: graphkit's ``HeteroGridDatasetDisk`` saves ``data_index_<scenario>.pt`` but reads ``data_index_<idx>.pt`` for
idx in ``range(len(files))``. The published pretraining tables we hold are a sample of partitions, so their scenario
ids have gaps (e.g. 1,060 scenarios with ids up to 9,659), and 860 of 1,060 ``get()`` calls then fail (measured
2026-10-18 on subnet_case10480_goc_idx_40). The eval trees were renumbered the same way
(``results/2026-10-08-collapsed-label-genco/build_pretrain_eval_data.py``); the OSC retrain body had not been.

Writes each table back as one parquet file (the hive ``scenario_partition`` column is kept as a plain column, as
pandas returns it when reading the partitioned dir) and ``scenario_map.csv`` (new, original). Refuses a dir whose
tables disagree on the scenario set. A dir that is already 0..N-1 is left unchanged.

Run: python -m genco_challenge.renumber_raw <raw dir> [<raw dir> ...]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

TABLES = ("bus_data", "gen_data", "branch_data", "y_bus_data")


def renumber(raw: Path) -> int:
    dfs = {t: pd.read_parquet(raw / f"{t}.parquet") for t in TABLES}
    sets = {t: set(map(int, d.scenario.unique())) for t, d in dfs.items()}
    ref = sets["bus_data"]
    for t, s in sets.items():
        if s != ref:
            raise SystemExit(f"{raw}: {t} has {len(s)} scenarios, bus_data {len(ref)}; refusing to renumber")
    old = sorted(ref)
    if old == list(range(len(old))):
        print(f"{raw}: already 0..{len(old) - 1}, unchanged")
        return 0
    m = {o: n for n, o in enumerate(old)}
    for t, d in dfs.items():
        d = d.copy()
        d["scenario"] = d.scenario.astype(int).map(m).astype("int64")
        if "scenario_partition" in d.columns:
            d["scenario_partition"] = d["scenario_partition"].astype(str).astype("int64")
        p = raw / f"{t}.parquet"
        tmp = raw / f".{t}.renumbered.parquet"
        old_aside = raw / f".{t}.original.parquet"
        d.to_parquet(tmp, index=False)
        # audit genco-retrain R1-F4: never leave the table missing. Move the old one aside, swap the new one in,
        # then delete the old one; a crash leaves either the old or the new table in place.
        p.rename(old_aside)
        tmp.rename(p)
        if old_aside.is_dir():
            shutil.rmtree(old_aside)
        else:
            old_aside.unlink()
    pd.DataFrame({"scenario": range(len(old)), "original_scenario": old}).to_csv(raw / "scenario_map.csv", index=False)
    print(f"{raw}: renumbered {len(old)} scenarios (original max id {old[-1]})")
    return len(old)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        renumber(Path(a))
