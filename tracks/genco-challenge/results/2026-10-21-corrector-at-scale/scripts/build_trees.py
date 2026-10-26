"""Split the converted ladder points of one grid into train / val / test and write graphkit trees (UNAUDITED).
usage: build_trees.py <case> [--ntrain 1000 --nval 100 --ntest 220]   (env grid-operations-genco, PYTHONPATH=tracks/genco-challenge/src)
Input  data/external/genco-corrector-scale/conv/<case>/raw (ladder_to_datakit output, 1,320 verified points).
Output data/external/genco-corrector-scale/trees/{train,eval}/<case>/{raw,meta.json}; the train tree holds train AND validation
scenarios (renumbered 0..N-1; ids of each part in splits_info/<case>.json), as in corrector-above-162. Hive partitions of 200 scenarios
(graphkit evaluate layout, as results/2026-10-20-genco-transfer/partition_val_trees.py); raw/scenario_map.csv is written as
raw/ladder_scenario_map.csv (new id -> original ladder sample).
Split: the ladder points are independent random load draws (load_scenario_idx unique per scenario), so a scenario-level split is leak-free
in profile terms; fixed permutation numpy default_rng(0). Collapsed-root check (min Vm < 0.5 pu) applied as in ladder_data.label_diagnostics."""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
from genco_challenge import ladder_data as ld

ap = argparse.ArgumentParser(); ap.add_argument("case")
ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--nval", type=int, default=100); ap.add_argument("--ntest", type=int, default=220)
a = ap.parse_args()
D = ld.REPO / "data/external/genco-corrector-scale"
raw = D / "conv" / a.case / "raw"
bus = pd.read_parquet(raw / "bus_data.parquet")
gen = pd.read_parquet(raw / "gen_data.parquet")
diag = ld.label_diagnostics(bus[["scenario", "bus", "Vm", "PV", "REF", "load_scenario_idx"]], gen[["scenario", "bus", "q_mvar", "min_q_mvar", "max_q_mvar", "in_service"]])
ids = diag.index[~diag["invalid"].to_numpy()].to_numpy()
print(a.case, "scenarios", bus["scenario"].nunique(), "valid", len(ids), "collapsed", int(diag["invalid"].sum()), flush=True)
assert len(ids) >= a.ntrain + a.nval + a.ntest, "not enough valid points"
perm = ids[np.random.default_rng(0).permutation(len(ids))]
test, val, train = perm[:a.ntest], perm[a.ntest:a.ntest + a.nval], perm[a.ntest + a.nval:a.ntest + a.nval + a.ntrain]
assert not (set(test) & set(val)) and not (set(test) & set(train)) and not (set(val) & set(train))
smap = pd.read_csv(raw / "scenario_map.csv").set_index("new")["orig"]
TABLES = ["bus_data", "gen_data", "branch_data", "y_bus_data"]
tabs = {"bus_data": bus, "gen_data": gen}
for t in ("branch_data", "y_bus_data"):
    tabs[t] = pd.read_parquet(raw / f"{t}.parquet")


def write(name, order):
    order = np.array(order)
    new = pd.Series(np.arange(len(order)), index=order)
    gdir = D / "trees" / name / a.case
    if gdir.exists():
        shutil.rmtree(gdir)
    (gdir / "raw").mkdir(parents=True)
    for t in TABLES:
        df = tabs[t].drop(columns=["scenario_partition"], errors="ignore")
        df = df[df["scenario"].isin(order)].copy()
        df["scenario"] = new.reindex(df["scenario"].to_numpy()).to_numpy()
        df = df.sort_values("scenario", kind="stable").reset_index(drop=True)
        df["scenario_partition"] = (df["scenario"] // ld.N_PER_PARTITION).astype("int64")
        df.to_parquet(gdir / "raw" / f"{t}.parquet", partition_cols=["scenario_partition"], engine="pyarrow", index=False)
        if t == "bus_data":
            rt = df[["scenario", "load_scenario_idx"]].drop_duplicates("scenario").sort_values("scenario").reset_index(drop=True)
            rt["scenario_partition"] = (rt["scenario"] // ld.N_PER_PARTITION).astype("int64")
            rt.to_parquet(gdir / "raw" / "runtime_data.parquet", partition_cols=["scenario_partition"], index=False)
    (gdir / "raw" / "n_scenarios.txt").write_text(str(len(order)))
    pd.DataFrame({"new": np.arange(len(order)), "orig": [int(smap.loc[s]) for s in order]}).to_csv(gdir / "raw" / "ladder_scenario_map.csv", index=False)
    (gdir / "meta.json").write_text(json.dumps({"name": a.case, "n_bus": int(bus[bus.scenario == bus.scenario.iloc[0]].shape[0]),
                                                "scenarios_written": int(len(order)), "status": "ok",
                                                "variant": "corrector-at-scale 2026-10-21, ladder_to_datakit conversion"}))
    return new

newT = write("train", list(train) + list(val))
write("eval", list(test))
info = {"case": a.case, "written": time.strftime("%F %T"), "train_ids": list(range(len(train))), "val_ids": list(range(len(train), len(train) + len(val))),
        "eval_n": int(len(test)), "n_valid": int(len(ids)), "n_collapsed_dropped": int(diag["invalid"].sum())}
(D / "splits_info").mkdir(exist_ok=True)
(D / "splits_info" / f"{a.case}.json").write_text(json.dumps(info))
print("done", {k: (len(v) if isinstance(v, list) else v) for k, v in info.items()}, flush=True)
