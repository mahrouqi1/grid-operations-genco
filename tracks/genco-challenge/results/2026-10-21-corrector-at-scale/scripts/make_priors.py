"""fd2 prior tables for the corrector-at-scale study: [scenarios, buses, 2] = (|V| pu, angle rad) of the fd2 state, per tree and grid ->
data/external/genco-corrector-scale/priors/<tree>/<case>.npy. Same code path as corrector-above-162/scripts/make_priors.py (lp2.priors_for)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from genco_challenge import ladder_data as ld
from genco_challenge import labelfree_prior2 as lp2

D = ld.REPO / "data/external/genco-corrector-scale"
RAW = False


def one(args):
    root, grid, lo, hi = args
    scen = list(range(lo, hi))
    bus, ys, brs = lp2._grid_tables(root, grid, scen)
    out = []
    for s in scen:
        b = bus[bus["scenario"] == s].sort_values("bus")
        sc = lp2.input_problem(b, ys[s], brs[s])
        if RAW:      # POST HOC variant: the raw second XB iterate from DC, without the mismatch guard of lp2.priors_for
            v = sc.vm_set * np.exp(1j * sc.va_dc)
            for _ in range(2):
                v = sc.fdxb_step(v)
        else:
            v = lp2.priors_for(sc)["fd2"]["v"]
        out.append((s, np.abs(v), np.radians(lp2._unwrapped(sc, v))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    RAW = a.raw
    for tree in ("train", "eval"):
        root = D / "trees" / tree
        ns = int(pd.read_parquet(root / a.case / "raw/bus_data.parquet", columns=["scenario"])["scenario"].max()) + 1
        jobs = [(root, a.case, lo, min(ns, lo + 10)) for lo in range(0, ns, 10)]
        with ProcessPoolExecutor(a.procs) as ex:
            res = [r for ch in ex.map(one, jobs) for r in ch]
        res.sort(key=lambda r: r[0])
        assert [r[0] for r in res] == list(range(ns))
        tab = np.stack([np.stack([r[1], r[2]], -1) for r in res])
        PD = D / ("priors_raw" if a.raw else "priors")
        (PD / tree).mkdir(parents=True, exist_ok=True)
        np.save(PD / tree / f"{a.case}.npy", tab)
        print(tree, ns, "scenarios", tab.shape[1], "buses; non-finite fd2 scenarios", int((~np.isfinite(tab).all(axis=(1, 2))).sum()), flush=True)
