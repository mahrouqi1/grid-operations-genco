"""Opt-in data variant for OSC job B (audit genco-collapsed-repair R2-F2): drop scenarios whose label has an in-service
branch angle above 90 degrees (the repo's relabel.ANG_MAX) from the corrected pretraining tables of job B's subgrids.

Writes data/external/genco-pf-pretraining-filtered90/<subgrid>/raw/{bus,gen,branch,y_bus}_data.parquet only for subgrids
that have at least one such scenario, keeping the hive partitioning (scenario_partition=N) of the source. Source per
subgrid: the corrected tree when it has the subgrid, else the original. Writes scan.csv (every subgrid) and
dropped.csv (every dropped scenario with its max angle and min PQ Vm) next to this file.

Run (env grid-operations-genco, repo root): python tracks/genco-challenge/results/2026-10-18-angle90-filter/build_filter90.py
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXT = REPO / "data/external"
SRC, COR, OUT = EXT / "genco-pf-pretraining", EXT / "genco-pf-pretraining-corrected", EXT / "genco-pf-pretraining-filtered90"
ALL9 = ["subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case9591_goc_idx_12",
        "subnet_case5658_epigrids_idx_8", "subnet_case4837_goc_idx_28", "subnet_case19402_goc_idx_47",
        "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37", "subnet_case10480_goc_idx_26"]   # = genco_retrain_body.sh
ANG_MAX = 90.0
TABLES = ["bus_data", "gen_data", "branch_data", "y_bus_data"]


def max_angle(raw: Path) -> pd.DataFrame:
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Vm", "Va", "PQ"])
    br = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "from_bus", "to_bus", "br_status"])
    br = br[br.br_status == 1]
    va = b.set_index(["scenario", "bus"]).Va
    a = (va.reindex(pd.MultiIndex.from_arrays([br.scenario, br.from_bus])).to_numpy()
         - va.reindex(pd.MultiIndex.from_arrays([br.scenario, br.to_bus])).to_numpy())
    a = np.abs((a + 180) % 360 - 180)
    mx = pd.Series(a, index=br.scenario.to_numpy()).groupby(level=0).max().rename("max_angle_deg")
    mn = b[b.PQ == 1].groupby("scenario").Vm.min().rename("min_pq_vm")
    return pd.concat([mx, mn], axis=1)


def main():
    scan, dropped = [], []
    for g in ALL9:
        src = (COR if (COR / g / "raw").is_dir() else SRC) / g / "raw"
        s = max_angle(src)
        bad = s.index[s.max_angle_deg > ANG_MAX]
        scan.append({"subgrid": g, "source": src.parents[1].name, "n_scen": len(s), "gt90": len(bad),
                     "max_angle_deg": round(float(s.max_angle_deg.max()), 1)})
        print(scan[-1], flush=True)
        if len(bad) == 0:
            continue
        dropped += [{"subgrid": g, "scenario": int(i), **s.loc[i].round(4).to_dict()} for i in bad]
        dst = OUT / g / "raw"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        badset = set(int(i) for i in bad)
        for t in TABLES:
            ds = pq.ParquetDataset(src / f"{t}.parquet")
            tab = ds.read()
            df = tab.to_pandas()
            keep = df[~df.scenario.isin(badset)]
            assert keep.scenario.nunique() == len(s) - len(bad), (g, t)
            part = "scenario_partition" in df.columns
            out = pa.Table.from_pandas(keep, preserve_index=False)
            if part:
                pq.write_to_dataset(out, root_path=str(dst / f"{t}.parquet"), partition_cols=["scenario_partition"])
            else:
                pq.write_table(out, dst / f"{t}.parquet")
        for extra in src.iterdir():
            if extra.suffix != ".parquet":
                shutil.copy2(extra, dst / extra.name)
    pd.DataFrame(scan).to_csv(HERE / "scan.csv", index=False)
    pd.DataFrame(dropped).to_csv(HERE / "dropped.csv", index=False)
    print("dropped", len(dropped), "scenarios; written:", sorted(p.name for p in OUT.iterdir()) if OUT.exists() else [])


if __name__ == "__main__":
    main()
