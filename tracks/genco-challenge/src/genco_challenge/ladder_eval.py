"""Score GENCO checkpoints on the held-out ladder data built by ``ladder_data``.

For every grid and checkpoint: run graphkit's own ``evaluate`` (GENCO's metric path, unchanged)
with ``--save_output``, then compute, on the same test scenarios, voltage errors against the AC
label by bus type for:

- ``genco``: GENCO's raw prediction;
- ``genco_pinned``: GENCO with the quantities a power-flow problem SPECIFIES put back (slack Vm and
  Va, PV Vm). A pure post-processing fix; it uses only inputs, never labels of unknowns;
- ``dc``: DC power flow angles (datakit's ``Va_dc``), Vm at setpoints for PV/slack and 1.0 at PQ;
- ``flat``: Va = slack angle everywhere, Vm as in ``dc``.

GENCO's own power-balance residual (MW) for the raw prediction and DC come from graphkit's
metric files. Angles are reported in degrees; graphkit predictions are in radians, ``bus_data``
in degrees.

Run (env ``grid-operations-genco``, from the repo root):

    python -m genco_challenge.ladder_eval --models zeroshot ft118_1000 --out <result dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
WORK = REPO / "data/external/genco-ladder"
GK = REPO / "data/external/gridfm-graphkit-repro"
T = REPO / "data/external/genco-pf-transfer"

MODELS = {
    "zeroshot": (T / "pretraining/best_model_state_dict.pt", None),
    "ft118_1000": (T / "finetune/case118_1000/best_model_state_dict.pt",
                   T / "finetune/case118_1000/normalizer_stats.pt"),
    "ft118_250000": (T / "finetune/case118_250000/best_model_state_dict.pt",
                     T / "finetune/case118_250000/normalizer_stats.pt"),
}
TYPES = ("PQ", "PV", "REF")


def run_graphkit(grid: str, n_scen: int, model: str, mlruns: Path, data_root: Path | None = None) -> Path:
    """``data_root`` (optional, default ``WORK/data``): another label variant, e.g. ``WORK/data-relabeled``."""
    cfg = yaml.safe_load((GK / "scripts/pretraining_advantage/case118_eval.yaml").read_text())
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [n_scen]
    if os.environ.get("EVAL_BATCH"):   # opt-in (2026-10-21): smaller inference batch for 6k+-bus grids on a shared GPU;
        cfg["training"]["batch_size"] = int(os.environ["EVAL_BATCH"])   # inference-only, metrics unchanged
    tag = "" if data_root is None else "_" + data_root.name
    cfg_path = WORK / "configs" / f"eval_{grid}{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    mp, ns = MODELS[model]
    cmd = ["gridfm_graphkit", "evaluate", "--config", str(cfg_path), "--data_path", str(data_root or WORK / "data"),
           "--model_path", str(mp), "--compute_dc_ac_metrics", "--save_output", "--num_workers", "4",
           "--log_dir", str(mlruns), "--run_name", f"{model}__{grid}"]
    if ns is not None:
        cmd += ["--normalizer_stats", str(ns)]
    log = WORK / "logs" / f"eval_{model}__{grid}{tag}.log"
    env = os.environ | {"MLFLOW_ALLOW_FILE_STORE": "true", "PYTHONNOUSERSITE": "1"}
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=GK, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"graphkit evaluate failed ({rc}), see {log}")
    runs = [Path(p).parent.parent for p in glob.glob(str(mlruns / "*/*/tags/mlflow.runName"))
            if Path(p).read_text().strip() == f"{model}__{grid}"]
    return max(runs, key=lambda p: p.stat().st_mtime) / "artifacts/test"


def _csv(path: Path) -> dict:
    return {r["Metric"]: float(r["Value"]) for r in csv.DictReader(path.open())}


def _wrap(d: np.ndarray) -> np.ndarray:
    """Angle difference in degrees wrapped to (-180, 180]."""
    return -((-d + 180.0) % 360.0 - 180.0)


def shift_dc_table(grid_dir: Path, scenarios=None) -> pd.DataFrame:
    """Shift-aware DC angles (degrees) per (scenario, bus) from each scenario's OWN Ybus/Sbus and full branch
    table, via ``relabel.shift_dc_angles`` (MATPOWER makeBdc with phase shifters). datakit's written ``Va_dc``
    ignores phase shifters, so this is the DC baseline that a grid with shifters is owed. Columns:
    scenario, bus, Va_dc_shift. Pass it to ``errors`` by merging into ``bus`` (additive; absent = old behaviour)."""
    from genco_challenge.relabel import shift_dc_angles
    from genco_challenge.hybrid import Scenario
    raw = Path(grid_dir) / "raw"
    flt = None if scenarios is None else [("scenario", "in", [int(x) for x in scenarios])]
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    yb = pd.read_parquet(raw / "y_bus_data.parquet", filters=flt)
    br = pd.read_parquet(raw / "branch_data.parquet", filters=flt)
    by, bb = dict(tuple(yb.groupby("scenario"))), dict(tuple(br.groupby("scenario")))
    rows = []
    for s, b in bus.groupby("scenario"):
        sc = Scenario(b, by[s], bb[s])
        th = np.degrees(shift_dc_angles(sc, bb[s]))
        rows.append(pd.DataFrame({"scenario": s, "bus": np.arange(sc.n), "Va_dc_shift": th}))
    return pd.concat(rows, ignore_index=True)


def errors(pred: pd.DataFrame, bus: pd.DataFrame, branch: pd.DataFrame | None = None) -> dict:
    """Voltage errors per predictor. If ``bus`` carries an optional ``Va_dc_shift`` column (see
    ``shift_dc_table``), a ``dc_shift`` predictor is added; without it the output is exactly as before."""
    has_shift = "Va_dc_shift" in bus.columns
    df = pred.merge(bus[["scenario", "bus", "Va_dc"] + (["Va_dc_shift"] if has_shift else [])],
                    on=["scenario", "bus"], how="left")
    vm_t, va_t = df["Vm_target"].to_numpy(), np.degrees(df["Va_target"].to_numpy())
    is_ref, is_pv = df["REF"].to_numpy() == 1, df["PV"].to_numpy() == 1
    ref_va = df.groupby("scenario")["Va_target"].transform(lambda s: s[df.loc[s.index, "REF"] == 1].mean())
    ref_va = np.degrees(ref_va.to_numpy())
    vm_set = np.where(is_ref | is_pv, vm_t, 1.0)
    preds = {
        "genco": (df["Vm_pred"].to_numpy(), np.degrees(df["Va_pred"].to_numpy())),
        "dc": (vm_set, df["Va_dc"].to_numpy()),
        "flat": (vm_set, ref_va),
    }
    if has_shift:
        preds["dc_shift"] = (vm_set, df["Va_dc_shift"].to_numpy())
    vm_p, va_p = preds["genco"]
    preds["genco_pinned"] = (np.where(is_ref | is_pv, vm_t, vm_p), np.where(is_ref, va_t, va_p))
    # Branch angle-difference error (audit null-baseline R2-F1): power flow depends on angle
    # DIFFERENCES across lines, so a uniform offset relative to the slack is not a flow error.
    br = None
    if branch is not None:
        br = branch[branch["br_status"] == 1][["scenario", "from_bus", "to_bus"]]
        br = br[br["scenario"].isin(df["scenario"].unique())]
        key = pd.MultiIndex.from_arrays([df["scenario"], df["bus"]])
        fi = key.get_indexer(pd.MultiIndex.from_arrays([br["scenario"], br["from_bus"]]))
        ti = key.get_indexer(pd.MultiIndex.from_arrays([br["scenario"], br["to_bus"]]))
        ok = (fi >= 0) & (ti >= 0)
        fi, ti = fi[ok], ti[ok]
        d_true = va_t[fi] - va_t[ti]
    out = {}
    for name, (vm, va) in preds.items():
        e_vm, e_va = np.abs(vm - vm_t), np.abs(_wrap(va - va_t))
        row = {"vm_mae_pu": float(e_vm.mean()), "va_mae_deg": float(e_va.mean()),
               "vm_max_pu": float(e_vm.max()), "va_max_deg": float(e_va.max())}
        if br is not None and len(fi):
            e_br = np.abs(_wrap((va[fi] - va[ti]) - d_true))
            row["branch_dva_mae_deg"] = float(e_br.mean())
            row["branch_dva_max_deg"] = float(e_br.max())
        for t in TYPES:
            m = df[t].to_numpy() == 1
            if m.any():
                row[f"vm_mae_pu_{t}"] = float(e_vm[m].mean())
                row[f"va_mae_deg_{t}"] = float(e_va[m].mean())
        out[name] = row
    return out


def data_stamp(grid: str) -> str:
    """Version of a grid's labels: validation time, scenario count and raw/ mtime. Stored in every
    results row; a row whose stamp differs from the current one was scored on other labels and is
    not 'done' (audit R1-F5)."""
    gdir = WORK / "data" / grid
    meta = json.loads((gdir / "meta.json").read_text())
    when = (meta.get("validation") or {}).get("when", "")
    return f"{when}|{meta.get('scenarios_written')}|{int((gdir / 'raw').stat().st_mtime)}"


def join_check(pred: pd.DataFrame, bus: pd.DataFrame) -> None:
    """graphkit's targets must equal raw/ (a stale processed/ cache once broke this); bus needs
    scenario, bus, Vm, Pd."""
    j = pred.merge(bus, on=["scenario", "bus"], how="left", suffixes=("", "_raw"))
    # Tolerances sit above graphkit's float32 storage (~1e-4 MW on large loads) and far below a
    # real mismatch (the stale case30 cache was 0.35 off).
    vm_gap = float((j["Vm_target"] - j["Vm"]).abs().max())
    pd_gap = float(((j["Pd"] - j["Pd_raw"]).abs() / np.maximum(j["Pd_raw"].abs(), 100.0)).max())
    if not (np.isfinite(vm_gap) and np.isfinite(pd_gap)) or vm_gap > 1e-4 or pd_gap > 1e-5:
        raise RuntimeError(f"graphkit targets differ from raw bus_data (|V| {vm_gap:.3g} pu, "
                           f"Pd rel {pd_gap:.3g}): stale processed/ cache?")


def evaluate(grid: str, model: str, mlruns: Path, data_root: Path | None = None, shift: bool = False) -> dict:
    """Score one checkpoint on one grid. Defaults reproduce the original behaviour exactly. Optional:
    ``data_root`` scores another label variant (e.g. ``WORK/data-relabeled``); ``shift`` adds the shift-aware
    DC baseline ``dc_shift`` to ``errors``; when the variant has ``raw/scenario_map.csv`` the row also carries
    ``errors_by_relabeled`` ({"0": datakit-label scenarios, "1": relabeled}) and ``n_by_relabeled``."""
    root = data_root or WORK / "data"
    meta = json.loads((root / grid / "meta.json").read_text())
    n_scen = meta["scenarios_written"]
    art = run_graphkit(grid, n_scen, model, mlruns, data_root)
    pred = pd.read_parquet(art / f"{grid}_predictions.parquet")
    bus = pd.read_parquet(root / grid / "raw/bus_data.parquet",
                          columns=["scenario", "bus", "Va_dc", "Vm", "Pd"])
    join_check(pred, bus)
    branch = pd.read_parquet(root / grid / "raw/branch_data.parquet",
                             columns=["scenario", "from_bus", "to_bus", "br_status"])
    if shift:
        bus = bus.merge(shift_dc_table(root / grid, pred["scenario"].unique()), on=["scenario", "bus"], how="left")
    m = _csv(art / f"{grid}_metrics.csv")
    ref = _csv(art / f"{grid}_ac_dc_metrics.csv")
    row = {"grid": grid, "model": model, "data_stamp": data_stamp(grid) if data_root is None else None,
           "n_bus": meta["n_bus"], "n_test_scenarios": int(pred["scenario"].nunique()),
           "genco_active_res_mw": m["Avg. active res. (MW)"], "genco_reactive_res_mvar": m["Avg. reactive res. (MVar)"],
           "genco_pbe_max": m["PBE Max"], "dc_active_res_mw": ref["DC Avg. active res. (MW)"],
           "label_active_res_mw": ref["AC Avg. active res. (MW)"], "errors": errors(pred, bus, branch)}
    smap_path = root / grid / "raw/scenario_map.csv"
    if data_root is not None and smap_path.exists():
        smap = pd.read_csv(smap_path)
        flag = dict(zip(smap["scenario"], smap["relabeled"]))
        row["errors_by_relabeled"], row["n_by_relabeled"] = {}, {}
        for f in (0, 1):
            sub = pred[pred["scenario"].map(flag) == f]
            if len(sub):
                row["errors_by_relabeled"][str(f)] = errors(sub, bus, branch)
                row["n_by_relabeled"][str(f)] = int(sub["scenario"].nunique())
    return row


def _artifacts(grid: str, model: str, mlruns: Path) -> Path:
    runs = [Path(p).parent.parent for p in glob.glob(str(mlruns / "*/*/tags/mlflow.runName"))
            if Path(p).read_text().strip() == f"{model}__{grid}"]
    return max(runs, key=lambda p: p.stat().st_mtime) / "artifacts/test"


def rescore(res_path: Path) -> None:
    """Recompute ``errors`` for every successful row from its saved predictions (CPU only) and
    rewrite the file with one row per (grid, model), the latest successful one winning."""
    latest: dict = {}
    for r in map(json.loads, res_path.read_text().splitlines()):
        if "error" not in r or (r["grid"], r["model"]) not in latest:
            latest[(r["grid"], r["model"])] = r
    out = []
    for (g, mdl), r in sorted(latest.items()):
        if "error" not in r:
            art = _artifacts(g, mdl, WORK / "mlruns")
            pred = pd.read_parquet(art / f"{g}_predictions.parquet")
            bus = pd.read_parquet(WORK / "data" / g / "raw/bus_data.parquet",
                                  columns=["scenario", "bus", "Va_dc", "Vm", "Pd"])
            join_check(pred, bus)  # refuse to rescore predictions made on other labels (audit R1-F5)
            branch = pd.read_parquet(WORK / "data" / g / "raw/branch_data.parquet",
                                     columns=["scenario", "from_bus", "to_bus", "br_status"])
            r["errors"] = errors(pred, bus, branch)
            r["rescored"] = "wrapped angles + branch dVa (2026-10-07)"
        out.append(r)
    res_path.write_text("".join(json.dumps(r) + "\n" for r in out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["zeroshot", "ft118_1000"])
    ap.add_argument("--grids", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rescore", action="store_true", help="recompute metrics from saved predictions only")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.out / "results.jsonl")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    grids = args.grids or sorted(
        p.parent.name for p in WORK.glob("data/*/meta.json")
        if json.loads(p.read_text()).get("status") in ("ok", "cached"))
    res_path = args.out / "results.jsonl"
    done = set()
    if res_path.exists():
        # Failed rows (e.g. a transient OOM on the shared GPUs) are retried; so are rows scored on other
        # labels than the grid has now (data_stamp missing or different, e.g. after a re-validation).
        done = {(r["grid"], r["model"]) for r in map(json.loads, res_path.read_text().splitlines())
                if "error" not in r and r.get("data_stamp") == data_stamp(r["grid"])}
    for g in grids:
        for mdl in args.models:
            if (g, mdl) in done:
                continue
            try:
                r = evaluate(g, mdl, WORK / "mlruns")
            except Exception as exc:  # recorded, not raised: one grid must not stop the ladder
                r = {"grid": g, "model": mdl, "error": f"{type(exc).__name__}: {exc}"}
            with res_path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: r.get(k) for k in ("grid", "model", "genco_active_res_mw", "dc_active_res_mw", "error")}), flush=True)


if __name__ == "__main__":
    main()
