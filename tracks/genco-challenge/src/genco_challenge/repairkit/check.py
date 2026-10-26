"""``repairkit check``: classify every scenario of a raw dir. Read-only.

Per scenario: min |Vm| over PQ buses and over all buses, negative-Vm flag, max in-service branch angle, label power-mismatch
residual (needs y_bus_data), and a category:

  collapsed      some |Vm| < 0.5 (or non-finite): an unintended low-voltage root, repairable
  negative_only  no |Vm| < 0.5 but some Vm < 0: a sign defect (-|V| written with a 180 degree angle shift), repairable
  angle_gt90     a valid-looking label with an in-service branch angle > 90 degrees: FLAGGED FOR REVIEW, never repaired
  healthy        none of the above

Precedence collapsed > negative_only > angle_gt90 > healthy; ``angle_gt90`` is also a boolean column on every row.
Definitions reuse ``ladder_data.label_diagnostics`` (collapsed / negative) and ``relabel.ANG_MAX`` / ``relabel.RES_TOL``.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genco_challenge import relabel as R
from genco_challenge.ladder_data import label_diagnostics

from . import _io

CATEGORIES = ("healthy", "collapsed", "negative_only", "angle_gt90")
DEFECTS = ("collapsed", "negative_only")
_RAW: Path | None = None


def label_voltage(b: pd.DataFrame) -> np.ndarray:
    """Complex label voltage Vm * exp(j Va) (a negative Vm with a shifted angle is the same complex voltage)."""
    return b["Vm"].to_numpy() * np.exp(1j * np.radians(b["Va"].to_numpy()))


def max_branch_angle(v: np.ndarray, br: pd.DataFrame) -> float:
    on = br["br_status"].to_numpy() == 1
    f, t = br["from_bus"].to_numpy()[on], br["to_bus"].to_numpy()[on]
    if not len(f):
        return 0.0
    return float(np.abs(np.degrees(np.angle(v[f] * np.conj(v[t])))).max())


def ybus_matrix(y: pd.DataFrame, n: int) -> sp.csr_matrix:
    return sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(), (y["index1"].to_numpy(), y["index2"].to_numpy())), shape=(n, n))


def label_residual(b: pd.DataFrame, y: pd.DataFrame) -> float:
    """Power mismatch (pu, max-norm of P at PV+PQ and Q at PQ) of the label as written, from its own y_bus and injections."""
    n = len(b)
    v = label_voltage(b)
    sbus = (b["Pg"].to_numpy() - b["Pd"].to_numpy() + 1j * (b["Qg"].to_numpy() - b["Qd"].to_numpy())) / 100.0
    pv, pq = np.where(b["PV"].to_numpy() == 1)[0], np.where(b["PQ"].to_numpy() == 1)[0]
    if not np.all(np.isfinite(v)):
        return float("nan")
    return R.mismatch_max(ybus_matrix(y, n), sbus, v, pv, pq)


def _batch(arg) -> pd.DataFrame:
    raw, b, with_y = arg
    bus = _io.read_batch(raw, "bus_data", b)
    br = _io.read_batch(raw, "branch_data", b, columns=["scenario", "from_bus", "to_bus", "br_status"])
    yb = _io.read_batch(raw, "y_bus_data", b) if with_y else None
    if bus is None or br is None or (with_y and yb is None):
        raise SystemExit(f"{raw}: batch {b.label} is missing from a table")
    bus = bus.sort_values(["scenario", "bus"]).reset_index(drop=True)
    diag = label_diagnostics(bus[["scenario", "bus", "Vm", "PV", "REF"]])
    pqabs = bus.loc[bus["PQ"] == 1].assign(a=lambda d: d["Vm"].abs()).groupby("scenario")["a"].min()
    bg, brg = dict(tuple(bus.groupby("scenario"))), dict(tuple(br.groupby("scenario")))
    yg = dict(tuple(yb.groupby("scenario"))) if with_y else {}
    rows = []
    for s, g in bg.items():
        ok_idx = bool((g["bus"].to_numpy() == np.arange(len(g))).all())
        v = label_voltage(g)
        ang = max_branch_angle(v, brg[s]) if s in brg and ok_idx else float("nan")
        res = float("nan")
        if with_y and s in yg and ok_idx:
            res = label_residual(g, yg[s])
        d = diag.loc[s]
        rows.append({"scenario": int(s), "load_scenario_idx": int(g["load_scenario_idx"].iloc[0]) if "load_scenario_idx" in g else -1,
                     "n_bus": len(g), "min_vm": float(d["min_vm"]), "min_abs_vm": float(d["min_abs_vm"]),
                     "min_pq_abs_vm": float(pqabs.get(s, np.nan)), "n_neg_vm": int(d["n_neg"]), "n_collapsed": int(d["n_collapsed"]),
                     "n_nonfinite": int(d["n_nonfinite"]), "max_branch_angle_deg": ang, "residual": res,
                     "angle_gt90": bool(ang > R.ANG_MAX)})
    return pd.DataFrame(rows)


def categorize(df: pd.DataFrame) -> pd.Series:
    cat = np.where((df["n_collapsed"] > 0) | (df["n_nonfinite"] > 0), "collapsed",
                   np.where(df["n_neg_vm"] > 0, "negative_only", np.where(df["angle_gt90"], "angle_gt90", "healthy")))
    return pd.Series(cat, index=df.index)


def check(raw: Path, workers: int = 1) -> pd.DataFrame:
    """Per-scenario report (see module doc). Never modifies ``raw``."""
    raw = Path(raw)
    for t in _io.CORE:
        if not _io.has_table(raw, t):
            raise SystemExit(f"{raw}: missing {t}.parquet")
    with_y = _io.has_table(raw, "y_bus_data")
    bl = _io.batches(raw)
    args = [(raw, b, with_y) for b in bl]
    if workers > 1 and len(args) > 1:
        with Pool(min(workers, len(args))) as p:
            parts = p.map(_batch, args, chunksize=1)
    else:
        parts = [_batch(a) for a in args]
    df = pd.concat(parts, ignore_index=True).sort_values("scenario").reset_index(drop=True)
    df["category"] = categorize(df)
    df["residual_gt_tol"] = df["residual"] > R.RES_TOL
    df.attrs["has_residual"] = with_y
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {"n_scenarios": int(len(df)), "has_residual": bool(df.attrs.get("has_residual", df["residual"].notna().any())),
           "categories": {c: int((df["category"] == c).sum()) for c in CATEGORIES},
           "angle_gt90_any_category": int(df["angle_gt90"].sum()),
           "min_abs_vm_over_dataset": float(df["min_abs_vm"].min()) if len(df) else None}
    if out["has_residual"]:
        out["residual_gt_1e-9_total"] = int(df["residual_gt_tol"].sum())
        out["residual_gt_1e-9_in_healthy"] = int((df["residual_gt_tol"] & (df["category"] == "healthy")).sum())
        out["max_residual_healthy"] = float(df.loc[df["category"] == "healthy", "residual"].max()) if (df["category"] == "healthy").any() else None
    else:
        out["note"] = "y_bus_data absent: residual column is NaN and was not checked"
    return out


def run(raw: Path, out: Path | None, workers: int = 1) -> dict:
    df = check(raw, workers)
    s = summarize(df)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        Path(str(out) + ".summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return s
