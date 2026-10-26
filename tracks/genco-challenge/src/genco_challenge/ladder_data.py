"""Build GENCO-format evaluation data for the held-out ladder with GENCO's own datakit.

Each grid gets a datakit config cloned from datakit's ``default_pf.yaml`` with the same edits
GENCO made for its own evaluation sets (``launch_pretraining_eval_data_gen_pf.py``): one topology
variant per load scenario. Only the network, scenario count, process count and output dir change.
So a held-out grid is perturbed exactly the way GENCO's case118 eval set was.

Scenario counts shrink with size to keep the run affordable on a shared box; they are recorded
in each grid's ``meta.json`` and in the summary.

**Label validation (2026-10-07).** datakit's AC power flow can converge to a COLLAPSED root:
zero-injection buses at Vm ~ 0 or negative. Such a point satisfies the power-flow equations, so a
residual check passes, but it is not an operating point (found on case1354_pegase, case240_pserc and
the RTE grids; see ``results/2026-10-07-genco-hybrid-warmstart/report.md`` section 7). After a grid
is generated, ``validate_grid`` drops every scenario whose label has min Vm < 0.5 pu or any negative
Vm, rewrites the scenario-keyed raw tables (bus, gen, branch, y_bus, runtime) with scenario ids
renumbered 0..N-1 (graphkit asserts contiguity), keeps datakit's ``load_scenario_idx`` unchanged
(so ``scenarios_agg_load_profile.parquet``, keyed by load scenario, stays a valid lookup and is
left as datakit wrote it, as datakit itself does after its own solver failures), and writes
``raw/scenario_map.csv`` (new id, original id, load scenario) and ``raw/dropped_scenarios.csv``.
The original ``raw/`` is kept as ``raw.unvalidated-<time>`` and a stale ``processed/`` is moved
aside. PV/REF reactive-limit violations are recorded as a diagnostic only: datakit's PF mode
(like GENCO's own data) does not enforce Q limits, so a violation is the problem as posed, not a
wrong solution. A grid left with fewer than ``MIN_VALID`` scenarios gets status
``insufficient_valid_scenarios`` (``ladder_eval`` then skips it).

Run (env ``grid-operations-genco``, from the repo root):

    python -m genco_challenge.ladder_data --status heldout --max-bus 3500
    python -m genco_challenge.ladder_data --names pglib_opf_case30_ieee --scenarios 100
    python -m genco_challenge.ladder_data --validate-only            # every generated grid
    python -m genco_challenge.ladder_data --validate-only --names pglib_opf_case1354_pegase
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
WORK = REPO / "data/external/genco-ladder"
PGLIB = REPO / "data/external/pglib-opf"
DATAKIT = REPO / "data/external/gridfm-datakit-repro"
SPLIT = REPO / "tracks/genco-challenge/results/2026-10-06-genco-heldout-split/split.json"

VM_MIN = 0.5           # pu; a label with any bus below this (or negative) is a collapsed root
MIN_VALID = 20         # fewer valid scenarios than this: the grid is not scored
Q_TOL_MVAR = 1.0       # reactive-limit violation threshold for the (diagnostic-only) Q check
SCEN_TABLES = ("bus_data", "gen_data", "branch_data", "y_bus_data", "runtime_data")
N_PER_PARTITION = 200  # datakit's n_scenario_per_partition (gridfm_datakit/utils/utils.py)


def scenarios_for(n_bus: int) -> int:
    if n_bus <= 500:
        return 200
    if n_bus <= 3500:
        return 100
    return 40


def bus_count_m(path: Path) -> int:
    """Count rows of mpc.bus in a MATPOWER file without loading it."""
    n, inside = 0, False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("mpc.bus") and "=" in s and "_name" not in s:
            inside = True
            continue
        if inside:
            if s.startswith("];"):
                break
            if s and not s.startswith("%"):
                n += 1
    return n


def make_config(name: str, n_scen: int, n_proc: int) -> Path:
    cfg = yaml.safe_load((DATAKIT / "scripts/config/default_pf.yaml").read_text())
    cfg["network"].update(name=name, source="file", network_dir=str(WORK / "grids"))
    cfg["load"]["scenarios"] = n_scen
    cfg["topology_perturbation"]["n_topology_variants"] = 1
    cfg["settings"].update(num_processes=n_proc, data_dir=str(WORK / "data"),
                           large_chunk_size=min(200, n_scen), enable_solver_logs=False)
    out = WORK / "configs" / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def build(name: str, n_scen: int | None, n_proc: int, timeout_s: int) -> dict:
    src = PGLIB / f"{name}.m"
    if not src.exists():
        return {"name": name, "status": "skipped", "reason": "no PGLib .m file (export not wired yet)"}
    raw = WORK / "data" / name / "raw"
    meta_p = WORK / "data" / name / "meta.json"
    if meta_p.exists() and ((raw / "bus_data.parquet").exists() or
                            json.loads(meta_p.read_text()).get("status") == "insufficient_valid_scenarios"):
        return json.loads(meta_p.read_text()) | {"status": "cached"}
    (WORK / "grids").mkdir(parents=True, exist_ok=True)
    (WORK / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, WORK / "grids" / src.name)
    # graphkit caches its processed tensors next to raw/ and does NOT notice a regenerated raw/:
    # on 2026-10-06 case30 was scored against a stale cache (caught by hybrid.py's join check).
    stale = WORK / "data" / name / "processed"
    if stale.exists():
        stale.rename(stale.with_name(f"processed.stale-{time.strftime('%Y%m%d-%H%M%S')}"))
    n_bus = bus_count_m(src)
    n_scen = n_scen or scenarios_for(n_bus)
    cfg = make_config(name, n_scen, n_proc)
    env = os.environ | {"PYTHONNOUSERSITE": "1", "JULIA_CPU_TARGET": "generic"}
    t0 = time.time()
    log = WORK / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as fh:
            rc = subprocess.run(["gridfm_datakit", "generate", str(cfg)], stdout=fh,
                                stderr=subprocess.STDOUT, env=env, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    meta = {"name": name, "n_bus": n_bus, "scenarios_requested": n_scen, "returncode": rc,
            "seconds": round(time.time() - t0, 1), "config": str(cfg.relative_to(REPO))}
    if rc == 0 and (raw / "bus_data.parquet").exists():
        bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario"])
        meta["scenarios_written"] = int(bus["scenario"].nunique())
        meta["status"] = "ok"
    else:
        meta["status"] = "failed"
    (WORK / "data" / name).mkdir(parents=True, exist_ok=True)
    (WORK / "data" / name / "meta.json").write_text(json.dumps(meta, indent=1))
    if meta["status"] == "ok":
        meta = validate_grid(name)
    return meta


# ---------------------------------------------------------------------------- label validation

def label_diagnostics(bus: pd.DataFrame, gen: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-scenario label checks. ``bus`` needs scenario, bus, Vm (and PV, REF for the Q check);
    ``gen`` needs scenario, bus, q_mvar, min_q_mvar, max_q_mvar, in_service.

    Columns: min_vm, max_vm, argmin_bus, n_low (Vm < VM_MIN), n_neg (Vm < 0), n_collapsed (|Vm| < VM_MIN),
    min_abs_vm, invalid, reason, and
    when ``gen`` is given q_viol_pv_mvar / q_viol_ref_mvar: the largest reactive-limit violation
    over PV / REF buses (gens at one bus aggregated, limits summed), 0 if none.
    """
    b = bus[["scenario", "bus", "Vm"]].reset_index(drop=True)
    g = b.groupby("scenario")["Vm"]
    out = pd.DataFrame({"min_vm": g.min(), "max_vm": g.max(),
                        "n_low": (b["Vm"] < VM_MIN).groupby(b["scenario"]).sum().astype(int),
                        "n_neg": (b["Vm"] < 0).groupby(b["scenario"]).sum().astype(int),
                        "n_nonfinite": (~np.isfinite(b["Vm"])).groupby(b["scenario"]).sum().astype(int)})
    # A negative Vm with |Vm| >= VM_MIN is not a collapsed magnitude: datakit can write a normal operating
    # point as -|V| with the angle shifted by 180 degrees (audit genco-eval R2-F1). Still unusable as a
    # Vm/Va regression target, so still invalid, but diagnosed separately from a collapsed root.
    absvm = b["Vm"].abs()
    out["n_collapsed"] = (absvm < VM_MIN).groupby(b["scenario"]).sum().astype(int)
    out["min_abs_vm"] = absvm.where(np.isfinite(absvm), np.inf).groupby(b["scenario"]).min()
    # idxmin on a copy with non-finite entries pushed to +inf: an all-NaN scenario would raise KeyError
    # on the raw column (audit R1-F7).
    b_fill = b.assign(Vm=b["Vm"].where(np.isfinite(b["Vm"]), np.inf))
    out["argmin_bus"] = b.loc[b_fill.groupby("scenario")["Vm"].idxmin(), "bus"].to_numpy()
    out["invalid"] = (out["n_low"] > 0) | (out["n_neg"] > 0) | (out["n_nonfinite"] > 0)
    out["reason"] = np.where(out["n_nonfinite"] > 0, "nonfinite",
                             np.where((out["n_neg"] > 0) & (out["n_collapsed"] > 0), "negative_vm",
                                      np.where(out["n_neg"] > 0, "negative_vm_sign_flip",
                                               np.where(out["n_low"] > 0, f"min_vm_below_{VM_MIN}", ""))))
    if gen is not None:
        on = gen[gen["in_service"] == 1]
        q = on.groupby(["scenario", "bus"])[["q_mvar", "min_q_mvar", "max_q_mvar"]].sum()
        viol = np.maximum(0.0, np.maximum(q["q_mvar"] - q["max_q_mvar"], q["min_q_mvar"] - q["q_mvar"]))
        t = bus.set_index(["scenario", "bus"])[["PV", "REF"]].reindex(q.index)
        for col, name in (("PV", "q_viol_pv_mvar"), ("REF", "q_viol_ref_mvar")):
            v = viol[t[col].to_numpy() == 1]
            out[name] = v.groupby(level="scenario").max().reindex(out.index).fillna(0.0)
    return out


def _summ(x: pd.Series) -> dict:
    return {"median": float(x.median()), "p95": float(x.quantile(0.95)), "max": float(x.max())} if len(x) else {}


def oracle_check(raw: Path, scen: list[int]) -> list[dict]:
    """Re-solve ``scen`` with this repo's Newton-Raphson (core/grid_ops/reference_pf) from a flat and
    a DC start on each scenario's own Ybus/Sbus (hybrid.Scenario builds them), tol 1e-8 pu, 30 its.
    ``agree`` = converged and max |V - V_label| < 1e-4 pu; ``physical`` = every Vm >= VM_MIN."""
    import warnings

    from genco_challenge.hybrid import Scenario, newton_raphson_pf

    flt = [("scenario", "in", [int(s) for s in scen])]
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    yb = pd.read_parquet(raw / "y_bus_data.parquet", filters=flt)
    br = pd.read_parquet(raw / "branch_data.parquet", filters=flt,
                         columns=["scenario", "from_bus", "to_bus", "x", "br_status"])
    by, bb = dict(tuple(yb.groupby("scenario"))), dict(tuple(br.groupby("scenario")))
    rows = []
    for s, b in bus.groupby("scenario"):
        sc = Scenario(b, by[s], bb[s])
        row = {"scenario": int(s), "label_min_vm": float(sc.vm_label.min()), "hybrid_ref_kind": sc.ref_kind}
        for start, va0 in (("flat", np.full(sc.n, sc.va_ref)), ("dc", sc.va_dc)):
            v0 = sc.vm_set * np.exp(1j * va0)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    r = newton_raphson_pf(sc.ybus, sc.sbus, v0, sc.pv, sc.pq, tol=1e-8, max_it=30)
                    ok = bool(r.converged and np.all(np.isfinite(r.v)))
                except Exception:
                    r, ok = None, False
            gap = float(np.abs(r.v - sc.v_label).max()) if ok else None
            row[start] = {"converged": ok, "iterations": r.iterations if r is not None else None,
                          "max_dv_vs_label_pu": gap, "agree": bool(ok and gap < 1e-4),
                          "root_min_vm": float(np.abs(r.v).min()) if ok else None,
                          "physical": bool(ok and np.abs(r.v).min() >= VM_MIN)}
        rows.append(row)
    return rows


def _sample(ids: list[int], k: int) -> list[int]:
    if len(ids) <= k:
        return list(ids)
    return [ids[i] for i in np.linspace(0, len(ids) - 1, k).round().astype(int)]


def _rewrite_table(src: Path, dst: Path, keep: np.ndarray) -> int:
    """Copy one scenario-partitioned table keeping ``keep`` (original ids, in new-id order)."""
    df = pd.read_parquet(src)
    df = df.drop(columns=["scenario_partition"], errors="ignore")
    new_id = pd.Series(np.arange(len(keep), dtype="int64"), index=keep)
    df = df[df["scenario"].isin(keep)].copy()
    df["scenario"] = new_id.reindex(df["scenario"].to_numpy()).to_numpy()
    df = df.sort_values("scenario", kind="stable").reset_index(drop=True)
    df["scenario_partition"] = (df["scenario"] // N_PER_PARTITION).astype("int64")
    df.to_parquet(dst, partition_cols=["scenario_partition"], engine="pyarrow", index=False)
    return len(df)


def validate_grid(name: str, *, force: bool = False, oracle_k: int = 10, root: Path | None = None) -> dict:
    """Drop collapsed-root scenarios from one generated grid and record it in meta.json.
    ``root`` = directory holding ``<name>/{raw,meta.json}`` (default ``WORK/data``; the fresh eval sets
    of 2026-10-08 live in ``WORK/eval2``)."""
    gdir = (root or WORK / "data") / name
    raw, meta_p = gdir / "raw", gdir / "meta.json"
    meta = json.loads(meta_p.read_text())
    if meta.get("status") not in ("ok", "insufficient_valid_scenarios") or not (raw / "bus_data.parquet").exists():
        return meta | {"validation_skipped": f"status {meta.get('status')!r} or no raw tables"}
    if "validation" in meta and not force:
        return meta
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "Vm", "PV", "REF"])
    gen = pd.read_parquet(raw / "gen_data.parquet",
                          columns=["scenario", "bus", "q_mvar", "min_q_mvar", "max_q_mvar", "in_service"])
    d = label_diagnostics(bus, gen)
    ids = d.index.to_numpy()
    if not (ids == np.arange(len(ids))).all():
        raise RuntimeError(f"{name}: scenario ids are not 0..N-1 before validation")
    keep = ids[~d["invalid"].to_numpy()]
    drop = ids[d["invalid"].to_numpy()]
    for t in SCEN_TABLES:  # every scenario-keyed table must carry the same scenarios
        s = pd.read_parquet(raw / f"{t}.parquet", columns=["scenario"])["scenario"].unique()
        if set(s) != set(ids):
            raise RuntimeError(f"{name}: {t} scenarios differ from bus_data")
    n_gen = len(ids)
    prev = raw / "scenario_map.csv"  # an earlier validation's mapping: new id -> id in the original raw/
    old = pd.read_csv(prev).set_index("scenario")["orig_scenario"] if prev.exists() else None

    def orig_id(s: int) -> int:  # chain through the earlier mapping (audit R1-F8)
        return int(old.loc[s]) if old is not None else int(s)

    val = {"rule": f"drop if min Vm < {VM_MIN} pu or any Vm < 0", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "generated": int(meta.get("scenarios_generated", n_gen)), "checked": n_gen,
           "dropped": int(len(drop)), "kept": int(len(keep)),
           "reasons": {k: int(v) for k, v in d.loc[drop, "reason"].value_counts().items()},
           "dropped_scenarios": [{"orig_scenario": orig_id(s), "reason": d.at[s, "reason"],
                                  "min_vm": float(d.at[s, "min_vm"]), "argmin_bus": int(d.at[s, "argmin_bus"]),
                                  "n_low": int(d.at[s, "n_low"]), "n_neg": int(d.at[s, "n_neg"])} for s in drop],
           "kept_min_vm": _summ(d.loc[keep, "min_vm"]) | {"min": float(d.loc[keep, "min_vm"].min())} if len(keep) else {},
           "q_limit_diagnostic": {
               "tol_mvar": Q_TOL_MVAR,
               "kept_scenarios_with_pv_violation": int((d.loc[keep, "q_viol_pv_mvar"] > Q_TOL_MVAR).sum()),
               "kept_pv_violation_mvar": _summ(d.loc[keep, "q_viol_pv_mvar"]),
               "kept_scenarios_with_ref_violation": int((d.loc[keep, "q_viol_ref_mvar"] > Q_TOL_MVAR).sum()),
               "dropped_pv_violation_mvar": _summ(d.loc[drop, "q_viol_pv_mvar"]),
               "note": "diagnostic only: datakit PF mode does not enforce Q limits"}}
    if oracle_k:
        rows = oracle_check(raw, _sample(list(map(int, keep)), oracle_k) + _sample(list(map(int, drop)), oracle_k))
        kept_rows = [r for r in rows if r["scenario"] in set(keep.tolist())]
        drop_rows = [r for r in rows if r["scenario"] in set(drop.tolist())]
        val["oracle"] = {
            "method": "core/grid_ops/reference_pf.newton_raphson_pf, flat and DC starts, tol 1e-8, max_it 30",
            "kept_sampled": len(kept_rows),
            "kept_agree_flat": sum(r["flat"]["agree"] for r in kept_rows),
            "kept_agree_dc": sum(r["dc"]["agree"] for r in kept_rows),
            "dropped_sampled": len(drop_rows),
            "dropped_physical_root_flat": sum(r["flat"]["physical"] for r in drop_rows),
            "dropped_physical_root_dc": sum(r["dc"]["physical"] for r in drop_rows),
            "rows": rows}
    changed = len(drop) > 0
    if changed:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tmp = gdir / f"raw.validating-{stamp}"
        shutil.copytree(raw, tmp, ignore=shutil.ignore_patterns(*(f"{t}.parquet" for t in SCEN_TABLES)))
        for t in SCEN_TABLES:  # no valid scenario: the tables are left out entirely
            if len(keep):
                _rewrite_table(raw / f"{t}.parquet", tmp / f"{t}.parquet", keep)
        smap = pd.DataFrame({"scenario": np.arange(len(keep)), "orig_scenario": keep})
        ls = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"]) \
            .groupby("scenario")["load_scenario_idx"].first()
        smap["load_scenario_idx"] = ls.reindex(keep).to_numpy()
        if old is not None:
            smap["orig_scenario"] = old.reindex(keep).to_numpy()
        smap.to_csv(tmp / "scenario_map.csv", index=False)
        dropped = d.loc[drop, ["reason", "min_vm", "argmin_bus", "n_low", "n_neg"]].rename_axis("orig_scenario")
        if old is not None:
            dropped.index = old.reindex(drop).to_numpy()
            old_drop = raw / "dropped_scenarios.csv"
            if old_drop.exists():
                dropped = pd.concat([pd.read_csv(old_drop, index_col=0), dropped])
        dropped.index.name = "orig_scenario"
        dropped.to_csv(tmp / "dropped_scenarios.csv")
        (tmp / "n_scenarios.txt").write_text(str(len(keep)))
        if len(keep):
            check = pd.read_parquet(tmp / "bus_data.parquet", columns=["scenario", "Vm"])
            assert check["scenario"].nunique() == len(keep) and check["scenario"].max() == len(keep) - 1
            assert check["Vm"].notna().all() and check["Vm"].min() >= VM_MIN
        raw.rename(gdir / f"raw.unvalidated-{stamp}")
        tmp.rename(raw)
        stale = gdir / "processed"
        if stale.exists():
            stale.rename(stale.with_name(f"processed.stale-{stamp}"))
        val["original_raw"] = f"raw.unvalidated-{stamp}"
    meta["scenarios_generated"] = val["generated"]
    meta["scenarios_written"] = int(len(keep))
    meta["scenarios_dropped_invalid"] = int(meta.get("scenarios_dropped_invalid", 0) + len(drop))
    meta["validation"] = val
    if len(keep) < MIN_VALID:
        meta["status"] = "insufficient_valid_scenarios"
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--status", choices=["heldout", "seen", "family-seen", "all"], default="heldout")
    ap.add_argument("--max-bus", type=int, default=10**9)
    ap.add_argument("--scenarios", type=int, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    ap.add_argument("--validate-only", action="store_true",
                    help="validate already-generated grids (all under data/ unless --names); no generation")
    ap.add_argument("--force", action="store_true", help="re-validate grids that already carry a validation")
    ap.add_argument("--relabel", action="store_true",
                    help="build the SEPARATE data-relabeled/ variant: collapsed labels replaced by the physical "
                         "root (genco_challenge.relabel); the validated data under data/ is not touched")
    ap.add_argument("--oracle-k", type=int, default=10, help="kept and dropped scenarios re-solved per grid")
    args = ap.parse_args()
    if args.relabel:
        from genco_challenge.relabel import main as relabel_main
        relabel_main(["--names", *args.names] if args.names else [])
        return
    if args.validate_only:
        names = args.names or sorted(p.parent.name for p in (WORK / "data").glob("*/meta.json"))
        for n in names:
            m = validate_grid(n, force=args.force, oracle_k=args.oracle_k)
            v = m.get("validation", {})
            print(json.dumps({"name": n, "status": m.get("status"), "generated": v.get("generated"),
                              "dropped": v.get("dropped"), "kept": v.get("kept"), "reasons": v.get("reasons"),
                              "skipped": m.get("validation_skipped")}), flush=True)
        return
    if args.names:
        names = args.names
    else:
        split = json.loads(SPLIT.read_text())
        want = {"heldout": "held-out"}.get(args.status, args.status)
        names = [c["name"] for c in split["cases"]
                 if c["name"].startswith("pglib_opf_") and (want == "all" or c["genco_pretrain"] == want)]
        names = [n for n in names if bus_count_m(PGLIB / f"{n}.m") <= args.max_bus]
        names.sort(key=lambda n: bus_count_m(PGLIB / f"{n}.m"))
    summary = []
    for n in names:
        m = build(n, args.scenarios, args.procs, args.timeout)
        summary.append(m)
        print(json.dumps(m), flush=True)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
