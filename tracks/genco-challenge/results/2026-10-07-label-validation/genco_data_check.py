"""Collapsed-root check on GENCO's own data (task 3 of the 2026-10-07 label validation).

Applies ``ladder_data.label_diagnostics`` (min Vm < 0.5 pu, any negative Vm; PV/REF reactive-limit
violations as a diagnostic) to:

(a) GENCO's case118 PF-transfer evaluation set, ``data/external/genco-pf-transfer/data/case118_ieee/raw``
    (all scenarios);
(b) GENCO's pretraining subgrids on Hugging Face (``gridfm/genco-pf-pretraining``): for EVERY subgrid
    present on the hub, a fixed sample of ``bus_data`` partitions (first, evenly spaced, last; each
    partition = 200 scenarios) is downloaded to ``data/external/genco-pf-pretraining/`` (bus_data
    only, plus gen_data for the Q diagnostic when ``--with-gen``), staying under ``--max-gb``.

Writes ``genco_data_check.json`` next to this script. Run (env ``grid-operations-genco``, repo root,
PYTHONPATH=tracks/genco-challenge/src):

    python tracks/genco-challenge/results/2026-10-07-label-validation/genco_data_check.py --parts 6
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from genco_challenge.ladder_data import REPO, VM_MIN, label_diagnostics

HERE = Path(__file__).resolve().parent
EVAL118 = REPO / "data/external/genco-pf-transfer/data/case118_ieee/raw"
PRE = REPO / "data/external/genco-pf-pretraining"
LISTS = REPO / "data/external/genco-pf-pretraining-lists"
HF_REPO = "gridfm/genco-pf-pretraining"


def low_bus_stats(bus: pd.DataFrame) -> dict:
    """What the collapsed buses are (audit genco-eval R1-F11): ``bus`` needs scenario, Vm, PQ, PV, REF, Pd.
    A collapsed bus is Vm < VM_MIN. Reports their bus types, how many per affected scenario, and their
    active load Pd (MW), so statements like 'all PQ, 1 to 7 per scenario, median Pd X MW' have a source."""
    low = bus[bus["Vm"] < VM_MIN]
    if low.empty:
        return {}
    per_scen = low.groupby("scenario").size()
    a = low["Vm"].abs()
    return {"low_buses": int(len(low)),
            "low_buses_negative": int((low["Vm"] < 0).sum()),
            "low_buses_abs_vm_ge_vmmin": int((a >= VM_MIN).sum()),  # negative sign flips, not collapsed
            "low_bus_abs_vm": {"min": float(a.min()), "median": float(a.median()), "max": float(a.max())},
            "low_bus_types": {t: int((low[t] == 1).sum()) for t in ("PQ", "PV", "REF")},
            "low_buses_per_scenario": {"min": int(per_scen.min()), "median": float(per_scen.median()),
                                       "max": int(per_scen.max())},
            "low_bus_pd_mw": {"median": float(low["Pd"].median()), "p05": float(low["Pd"].quantile(0.05)),
                              "p95": float(low["Pd"].quantile(0.95)), "frac_zero": float((low["Pd"] == 0).mean())}}


def summarize(d: pd.DataFrame, bus: pd.DataFrame | None = None) -> dict:
    n = len(d)
    out = {"scenarios": n,
           "min_vm_below_0.5": int((d["min_vm"] < VM_MIN).sum()),
           "negative_vm": int((d["n_neg"] > 0).sum()),
           "collapsed_abs_vm_below_0.5": int((d["n_collapsed"] > 0).sum()),  # a bus with |Vm| < VM_MIN
           "negative_only_abs_vm_ge_0.5": int(((d["n_neg"] > 0) & (d["n_collapsed"] == 0)).sum()),
           "frac_collapsed_abs_vm": float((d["n_collapsed"] > 0).mean()) if n else None,
           "frac_min_vm_below_0.5": float((d["min_vm"] < VM_MIN).mean()) if n else None,
           "frac_negative_vm": float((d["n_neg"] > 0).mean()) if n else None,
           "min_vm_overall": float(d["min_vm"].min()) if n else None,
           "min_vm_p01": float(d["min_vm"].quantile(0.01)) if n else None,
           "min_vm_below_0.8": int((d["min_vm"] < 0.8).sum()),
           "max_vm_overall": float(d["max_vm"].max()) if n else None}
    if "q_viol_pv_mvar" in d:
        out["pv_q_violation_gt_1mvar"] = int((d["q_viol_pv_mvar"] > 1.0).sum())
        out["pv_q_violation_mvar_median"] = float(d["q_viol_pv_mvar"].median())
        out["pv_q_violation_mvar_max"] = float(d["q_viol_pv_mvar"].max())
    if bus is not None:
        out |= low_bus_stats(bus)
    return out


def check_eval118() -> dict:
    bus = pd.read_parquet(EVAL118 / "bus_data.parquet", columns=["scenario", "bus", "Vm", "PV", "REF", "PQ", "Pd"])
    gen = pd.read_parquet(EVAL118 / "gen_data.parquet",
                          columns=["scenario", "bus", "q_mvar", "min_q_mvar", "max_q_mvar", "in_service"])
    return {"path": str(EVAL118.relative_to(REPO)), **summarize(label_diagnostics(bus, gen), bus)}


def hub_index() -> dict[str, dict]:
    from huggingface_hub import HfApi
    files = HfApi().dataset_info(HF_REPO, files_metadata=True).siblings
    idx: dict[str, dict] = defaultdict(lambda: {"bus": defaultdict(list), "gen": defaultdict(list)})
    for f in files:
        m = re.match(r"([^/]+)/raw/(bus|gen)_data\.parquet/scenario_partition=(\d+)/", f.rfilename)
        if m:
            idx[m.group(1)][m.group(2)][int(m.group(3))].append((f.rfilename, f.size or 0))
    return idx


def pick(parts: list[int], k: int) -> list[int]:
    parts = sorted(parts)
    if len(parts) <= k:
        return parts
    return sorted({parts[i] for i in np.linspace(0, len(parts) - 1, k).round().astype(int)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=6, help="bus_data partitions sampled per subgrid")
    ap.add_argument("--with-gen", action="store_true", help="also fetch gen_data (Q diagnostic)")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    out: dict = {"rule": f"min Vm < {VM_MIN} pu; negative Vm", "eval_case118": check_eval118()}
    print(json.dumps(out["eval_case118"]), flush=True)

    from huggingface_hub import hf_hub_download
    idx = hub_index()
    split = {}
    for name in ("train", "validation", "test"):
        p = LISTS / f"{name}_networks.txt"
        if p.exists():
            split |= {s.strip(): name for s in p.read_text().split() if s.strip()}
    plan, total = [], 0
    for grid in sorted(idx):
        for part in pick(list(idx[grid]["bus"]), args.parts):
            for kind in ("bus", "gen") if args.with_gen else ("bus",):
                for fn, size in idx[grid][kind].get(part, []):
                    plan.append((grid, part, kind, fn))
                    total += size
    out["hub"] = {"subgrids_on_hub": len(idx), "listed_in_split_files": len(split),
                  "on_hub_by_split": {s: sum(split.get(g) == s for g in idx) for s in ("train", "validation", "test")},
                  "on_hub_not_in_lists": sorted(g for g in idx if g not in split),
                  "partitions_sampled_per_subgrid": args.parts, "download_gb": round(total / 1e9, 3)}
    print(json.dumps(out["hub"]), flush=True)
    if total / 1e9 > args.max_gb:
        raise SystemExit(f"planned download {total / 1e9:.2f} GB exceeds --max-gb {args.max_gb}")
    if not args.skip_download:
        for i, (_, _, _, fn) in enumerate(plan):
            hf_hub_download(HF_REPO, fn, repo_type="dataset", local_dir=PRE)
            if i % 25 == 0:
                print(f"downloaded {i + 1}/{len(plan)}", flush=True)
    rows = []
    for grid in sorted(idx):
        raw = PRE / grid / "raw"
        parts = pick(list(idx[grid]["bus"]), args.parts)
        bus = pd.concat([pd.read_parquet(raw / f"bus_data.parquet/scenario_partition={p}",
                                         columns=["scenario", "bus", "Vm", "PV", "REF", "PQ", "Pd"]) for p in parts])
        gen = None
        if args.with_gen:
            gen = pd.concat([pd.read_parquet(raw / f"gen_data.parquet/scenario_partition={p}",
                                             columns=["scenario", "bus", "q_mvar", "min_q_mvar", "max_q_mvar",
                                                      "in_service"]) for p in parts])
        d = label_diagnostics(bus, gen)
        r = {"subgrid": grid, "split": split.get(grid), "n_bus": int(bus.groupby("scenario")["bus"].size().max()),
             "partitions_on_hub": len(idx[grid]["bus"]), "partitions_checked": parts, **summarize(d, bus)}
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("subgrid", "split", "n_bus", "scenarios", "min_vm_below_0.5",
                                            "negative_vm", "min_vm_overall")}), flush=True)
    out["pretraining_subgrids"] = rows
    tot = sum(r["scenarios"] for r in rows)
    out["pretraining_total"] = {"subgrids": len(rows), "scenarios_checked": tot,
                                "min_vm_below_0.5": sum(r["min_vm_below_0.5"] for r in rows),
                                "negative_vm": sum(r["negative_vm"] for r in rows),
                                "collapsed_abs_vm_below_0.5": sum(r["collapsed_abs_vm_below_0.5"] for r in rows),
                                "negative_only_abs_vm_ge_0.5": sum(r["negative_only_abs_vm_ge_0.5"] for r in rows),
                                "subgrids_with_any": sum(r["min_vm_below_0.5"] > 0 for r in rows)}
    (HERE / "genco_data_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["pretraining_total"]), flush=True)


if __name__ == "__main__":
    main()
