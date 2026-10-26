"""Classical baselines (B1 AC with assumed dispatch, B2 +measured PV |V|, B3 DC), oracle-dispatch control, flat null.
CPU. One row per (grid, scenario, method). UNAUDITED.   python run_baselines.py [--grids ...]"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--grids", nargs="*", default=C.HELDOUT15)
ap.add_argument("--out", type=Path, default=HERE / "baselines_per_scenario.jsonl"); a = ap.parse_args()
ids = json.loads((HERE / "test_ids.json").read_text())
done = {r["grid"] for r in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
for g in a.grids:
    if g in done or g not in ids:
        continue
    t0 = time.time()
    tab = C.tables(g, ids[g]["test"]); base = C.base_case(g)
    # case-file bus/gen mapping must agree with datakit's tables (gen bus of scenario 0)
    s0 = ids[g]["test"][0]; gen0 = tab["gen"][s0]
    assert (gen0.bus.to_numpy().astype(int) == base["gen_bus"][gen0.idx.to_numpy().astype(int)]).all(), "gen-bus mapping"
    rows = []
    for s in ids[g]["test"]:
        r = C.baselines(tab, s, base)
        for name, v in r.items():
            rows.append({"grid": g, "method": name, "scenario": s, "converged": bool(v["converged"]),
                         **C.score(tab, s, v["vm"], v["va"], v["pg"])})
    with a.out.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(g, len(ids[g]["test"]), "loss_frac", round(base["loss_frac"], 4), f"{time.time()-t0:.0f}s", flush=True)
