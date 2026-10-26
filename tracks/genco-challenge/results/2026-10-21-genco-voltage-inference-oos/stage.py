"""Stage a scenario-subset tree per grid into data/external/genco-voltage-oos/<grid>/raw (single-file parquet, scenarios 0..K-1,
runtime_data + ladder_scenario_map.csv), so graphkit's datamodule (test_ratio 0.99) yields ~K-1 test scenarios. Values unchanged. CPU. UNAUDITED."""
import sys, shutil
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grids as G
want = sys.argv[1:] or [g for _, _, g in G.ALL]
for fam, src, g in G.ALL:
    if g not in want: continue
    out = G.ROOT / g / "raw"
    if (out / "runtime_data.parquet").exists(): continue
    out.mkdir(parents=True, exist_ok=True)
    raw = src / "raw"
    nb = int(pq.read_table(raw / "bus_data.parquet", columns=["scenario"], filters=[("scenario", "==", 0)]).num_rows)
    K = G.N_BIG_STAGED if nb > G.BIG_BUS else G.N_SMALL_STAGED
    for t in ("bus_data", "branch_data", "gen_data", "y_bus_data"):
        df = pd.read_parquet(raw / f"{t}.parquet", filters=[("scenario", "<", K)])
        df = df.drop(columns=["scenario_partition"], errors="ignore")
        df.to_parquet(out / f"{t}.parquet", index=False)
        if t == "bus_data":
            rt = df[["scenario", "load_scenario_idx"]].drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
            rt.to_parquet(out / "runtime_data.parquet", index=False); ns = len(rt); assert sorted(rt.scenario) == list(range(ns)), g
    sm = raw / "ladder_scenario_map.csv"
    shutil.copy(sm if sm.exists() else raw / "scenario_map.csv", out / "ladder_scenario_map.csv")
    print(g, fam, "n_bus", nb, "staged", ns, flush=True)
