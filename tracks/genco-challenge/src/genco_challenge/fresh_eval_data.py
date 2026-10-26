"""Fresh evaluation sets for the nine few-shot / label-free grids (audit 2026-10-08 genco-adapt F1, F2).

The eval sets used to choose recipes (v1/v2/label-free) sit on aggregate-profile indices 0..199 and the
training pools on 200..1199. A fresh set must share neither. It is generated with GENCO's own datakit
and the SAME config as the old eval set (``genco-ladder/configs/<grid>.yaml``: one topology variant per
load scenario, 200 scenarios requested), with only the data dir, the global seed and the profile window
changed. datakit cuts the deterministic ``default`` profile to its first N points, so a window elsewhere
needs the load generator to build the first ``PROFILE_END`` points and keep ``PROFILE_START:PROFILE_END``;
``pf_post_processing`` is wrapped so the stored ``load_scenario_idx`` is the TRUE profile index. Nothing
else in datakit is touched. Window 1600..1799 was picked because its mean profile level (0.615) matches
the training pool's (0.608), not the old eval window's (0.672); it leaves a 400-hour gap after the pool.

    python -m genco_challenge.fresh_eval_data gen --grids pglib_opf_case14_ieee ...
    python -m genco_challenge.fresh_eval_data validate --grids ...
    python -m genco_challenge.fresh_eval_data disjoint            # writes disjoint.json in the results dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from genco_challenge import fewshot as fs
from genco_challenge import ladder_data as ld

EVAL2 = fs.WORK / "eval2"
SEED = 41113               # datakit global seed; old eval seeds were random draws, the training seed is 7007
PROFILE_START, PROFILE_END = 1600, 1800
RESULTS = fs.REPO / "tracks/genco-challenge/results/2026-10-08-adapt-fresh-eval"


def config(grid: str, procs: int) -> Path:
    cfg = yaml.safe_load((fs.WORK / "configs" / f"{grid}.yaml").read_text())   # the old eval set's config
    cfg["settings"].update(data_dir=str(EVAL2), seed=SEED, num_processes=procs, large_chunk_size=200,
                           enable_solver_logs=False)
    assert cfg["load"]["scenarios"] == 200 and cfg["topology_perturbation"]["n_topology_variants"] == 1
    out = EVAL2 / "_configs" / f"{grid}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg))
    return out


def _patch() -> None:
    from gridfm_datakit.perturbations import load_perturbation as lp
    from gridfm_datakit.process import process_network as pn
    orig_call, orig_post = lp.LoadScenariosFromAggProfile.__call__, pn.pf_post_processing

    def call(self, net, n_scenarios, scenarios_log, max_iter, seed):
        full = orig_call(self, net, PROFILE_END, scenarios_log, max_iter, seed)
        assert n_scenarios == PROFILE_END - PROFILE_START and full.shape[1] == PROFILE_END
        return full[:, PROFILE_START:PROFILE_END, :]

    def post(scenario_index, *a, **k):
        return orig_post(scenario_index + PROFILE_START, *a, **k)

    lp.LoadScenariosFromAggProfile.__call__ = call
    pn.pf_post_processing = post


def gen(grid: str, procs: int) -> dict:
    meta_p = EVAL2 / grid / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text()) | {"cached": True}
    if (EVAL2 / grid / "raw").exists():
        raise RuntimeError(f"{grid}: eval2 raw/ exists without meta.json (interrupted run); move it aside")
    old = fs.eval_seed(grid)
    assert SEED != old and SEED != fs.TRAIN_SEED, (grid, old)
    cfg = config(grid, procs)
    os.environ |= {"PYTHONNOUSERSITE": "1", "JULIA_CPU_TARGET": "generic"}
    from gridfm_datakit.generate import generate_power_flow_data_distributed
    _patch()
    t0 = time.time()
    generate_power_flow_data_distributed(str(cfg))
    raw = EVAL2 / grid / "raw"
    b = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
    per = b.groupby("scenario")["load_scenario_idx"].first()
    assert per.min() >= PROFILE_START and per.max() < PROFILE_END, (per.min(), per.max())
    meta = {"name": grid, "n_bus": ld.bus_count_m(ld.PGLIB / f"{grid}.m"), "scenarios_requested": 200,
            "returncode": 0, "seconds": round(time.time() - t0, 1), "status": "ok", "seed": SEED,
            "old_eval_seed": old, "profile_window": [PROFILE_START, PROFILE_END - 1],
            "scenarios_written": int(per.size), "config": str(cfg.relative_to(fs.REPO)),
            "note": "load_scenario_idx = true aggregate-profile index; scenarios_agg_load_profile.parquet is keyed 0..199 (window-relative)"}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def validate(grid: str) -> dict:
    return ld.validate_grid(grid, root=EVAL2)


def disjoint(grid: str) -> dict:
    """Fresh eval set vs old eval set and vs the training pool (the pool actually used by the label-free
    study, i.e. ``labelfree.pool_lf``, and v2's ``fewshot.pool``): profile indices and per-bus load vectors."""
    from genco_challenge import labelfree as lf   # patches fs.pool; only used for pool_lf
    cols = ["scenario", "load_scenario_idx", "bus", "Pd"]
    ev2 = pd.read_parquet(EVAL2 / grid / "raw/bus_data.parquet", columns=cols)
    ev1 = pd.read_parquet(fs.EVAL_DATA / grid / "raw/bus_data.parquet", columns=cols)
    tr = pd.read_parquet(lf.ORIG_TRAIN / grid / "raw/bus_data.parquet", columns=cols)
    pools = {"labelfree_pool": lf.pool_lf(grid)}
    pools["all_generated_train"] = sorted(set(tr["scenario"]))
    pools["v2_pool"] = sorted(int(s) for s in tr.groupby("scenario")["load_scenario_idx"].first().index
                              if tr.groupby("scenario")["load_scenario_idx"].first()[s] > 199
                              and int(s) not in fs.collapsed(lf.ORIG_TRAIN / grid / "raw/bus_data.parquet"))
    piv = lambda d: d.pivot(index="scenario", columns="bus", values="Pd").to_numpy()  # noqa: E731
    P2, P1, PT = piv(ev2), piv(ev1), piv(tr)
    scale = np.abs(P2).sum(axis=1).mean()
    idx2 = set(ev2["load_scenario_idx"].astype(int))
    out = {"grid": grid, "n_eval_fresh": int(P2.shape[0]), "n_eval_old": int(P1.shape[0]),
           "fresh_profile_idx_range": [min(idx2), max(idx2)],
           "old_eval_profile_idx_range": [int(ev1["load_scenario_idx"].min()), int(ev1["load_scenario_idx"].max())],
           "shared_profile_idx_with_old_eval": len(idx2 & set(ev1["load_scenario_idx"].astype(int))),
           "total_load_mw_fresh": [float(P2.sum(1).min()), float(P2.sum(1).max())],
           "total_load_mw_old_eval": [float(P1.sum(1).min()), float(P1.sum(1).max())]}
    d = np.abs(P1[:, None, :] - P2[None, :, :]).sum(axis=2) / scale
    out |= {"min_rel_L1_old_eval_to_fresh": float(d.min()), "exact_duplicates_vs_old_eval": int((d < 1e-9).sum())}
    trpiv = pd.DataFrame(PT, index=sorted(set(tr["scenario"])))
    first = tr.groupby("scenario")["load_scenario_idx"].first()
    for k, ids in pools.items():
        if k == "all_generated_train":
            ids_ = list(trpiv.index)
        else:
            ids_ = ids
        P = trpiv.loc[ids_].to_numpy()
        dd = np.abs(P[:, None, :] - P2[None, :, :]).sum(axis=2) / scale
        tri = set(first.loc[ids_].astype(int))
        out[k] = {"n": len(ids_), "profile_idx_range": [min(tri), max(tri)],
                  "shared_profile_idx": len(tri & idx2), "min_rel_L1_to_fresh": float(dd.min()),
                  "median_nn_rel_L1_to_fresh": float(np.median(dd.min(axis=1))),
                  "exact_duplicate_load_vectors": int((dd < 1e-9).sum()),
                  "total_load_mw": [float(P.sum(1).min()), float(P.sum(1).max())]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "validate", "disjoint"])
    ap.add_argument("--grids", nargs="+", default=fs.GRIDS)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "gen":
        for g in a.grids:
            print(json.dumps(gen(g, a.procs)), flush=True)
    elif a.cmd == "validate":
        for g in a.grids:
            m = validate(g)
            v = m.get("validation", {})
            print(json.dumps({"name": g, "status": m.get("status"), "generated": v.get("generated"),
                              "dropped": v.get("dropped"), "kept": v.get("kept"), "reasons": v.get("reasons")}), flush=True)
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        rows = [disjoint(g) for g in a.grids]
        (RESULTS / "disjoint.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
