"""``repairkit verify``: independent read-back of a repaired dir against its original. Exit status 1 on any failure.

(a) every scenario the log marks ``recovered``: the WRITTEN Vm/Va solve the power-flow equations with the WRITTEN Pg/Qg/Pd/Qd and
    the scenario's y_bus (residual < 1e-9), all Vm >= 0.5 (and positive), max in-service branch angle <= 90 degrees, bus Pg/Qg
    equal the in-service generator sums at PV/REF buses, branch pf equals the Ybus branch formula, the label actually changed,
    and no column outside the label set (Vm, Va, Pg, Qg / p_mw, q_mvar / pf, qf, pt, qt) differs from the original;
(b) HEALTHY-case regression: every scenario that is not ``recovered`` (including the ones listed ``unresolved``) is byte-identical,
    or equal to 1e-12, in all four tables;
(c) the same tables, columns, partitions, row counts, and scenario / bus / idx / index keys in the same order.
"""
from __future__ import annotations

import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genco_challenge import relabel as R

from . import _io
from .check import label_residual, label_voltage, max_branch_angle
from .repair import PATCH

TOL_EQ = 1e-12
TOL_RES = R.RES_TOL


def _same_files(a: Path, b: Path) -> bool:
    fa, fb = _io.leaf_files(a) if a.is_dir() else [a], _io.leaf_files(b) if b.is_dir() else [b]
    return len(fa) == len(fb) and all(x.name == y.name and filecmp.cmp(x, y, shallow=False) for x, y in zip(fa, fb))


def _max_diff(a: pd.DataFrame, b: pd.DataFrame, cols) -> float:
    worst = 0.0
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            xf, yf = x.astype(float), y.astype(float)
            both_nan = np.isnan(xf) & np.isnan(yf)
            d = np.where(both_nan | (xf == yf), 0.0, np.abs(xf - yf))
            d = np.where(np.isnan(d), np.inf, d)
        else:
            d = np.where(x == y, 0.0, np.inf)
        worst = max(worst, float(d.max()) if len(d) else 0.0)
    return worst


def verify(orig: Path, rep: Path, log_path: Path | None = None, say=print) -> dict:
    orig, rep = Path(orig), Path(rep)
    log_path = Path(log_path) if log_path else rep / "repair_log.csv"
    fails: list[str] = []

    def fail(msg):
        if len(fails) < 200:
            fails.append(msg)

    lg = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["scenario", "status"])
    if not log_path.exists():
        fail(f"repair log {log_path} not found")
    recovered = set(lg.loc[lg["status"] == "recovered", "scenario"].astype(int))
    unresolved = set(lg.loc[lg["status"] == "unresolved", "scenario"].astype(int))
    tabs = [t for t in _io.TABLES if _io.has_table(orig, t)]
    for t in _io.TABLES:
        if _io.has_table(orig, t) != _io.has_table(rep, t):
            fail(f"(c) table {t} present in only one dir")
    if _io.is_hive(orig) != _io.is_hive(rep):
        fail("(c) partition layout differs")
    stats = {"recovered_logged": len(recovered), "unresolved_logged": len(unresolved), "scenarios_compared": 0,
             "max_residual_recovered": 0.0, "min_vm_recovered": np.inf, "max_angle_recovered": 0.0, "max_diff_unchanged": 0.0}
    seen_rec: set[int] = set()
    for b in _io.batches(orig):
        d = {t: (_io.read_batch(orig, t, b), _io.read_batch(rep, t, b)) for t in tabs}
        for t, (o, r) in d.items():
            if o is None or r is None:
                fail(f"(c) {t} batch {b.label} missing in one dir")
                continue
            if list(o.columns) != list(r.columns):
                fail(f"(c) {t} {b.label}: columns differ")
                continue
            if len(o) != len(r):
                fail(f"(c) {t} {b.label}: row count {len(o)} vs {len(r)}")
                continue
            kc = [k for k in _io.KEYS[t] if k in o.columns]
            if _max_diff(o, r, kc) != 0.0:
                fail(f"(c) {t} {b.label}: scenario/key columns differ or reordered")
                continue
            fast = b.ids is None and _same_files(_io.tpath(orig, t) / b.label, _io.tpath(rep, t) / b.label)
            scen = o["scenario"].to_numpy()
            isrec = np.isin(scen, list(recovered))
            if fast:
                continue
            others = [c for c in o.columns if c not in (PATCH[t][1] if t in PATCH else [])]
            # (b) non-recovered rows: everything; (a) recovered rows: all columns outside the label set
            dd = _max_diff(o[~isrec], r[~isrec], list(o.columns))
            stats["max_diff_unchanged"] = max(stats["max_diff_unchanged"], dd)
            if dd > TOL_EQ:
                fail(f"(b) {t} {b.label}: a non-recovered scenario differs (max diff {dd:.3g})")
            if isrec.any():
                dd = _max_diff(o[isrec], r[isrec], others)
                if dd > TOL_EQ:
                    fail(f"(a) {t} {b.label}: a column outside the label set changed in a recovered scenario ({dd:.3g})")
        ids_here = set(map(int, d["bus_data"][1]["scenario"].unique())) if d["bus_data"][1] is not None else set()
        stats["scenarios_compared"] += len(ids_here)
        mine = sorted(recovered & ids_here)
        seen_rec |= set(mine)
        if not mine:
            continue
        if "y_bus_data" not in d:
            fail("(a) cannot check residual: no y_bus_data")
            continue
        bo, bn = d["bus_data"]; gn = d["gen_data"][1]; brn = d["branch_data"][1]; yn = d["y_bus_data"][1]
        bg, og, gg, rg, yg = (dict(tuple(x.groupby("scenario"))) for x in (bn, bo, gn, brn, yn))
        for s in mine:
            bb = bg[s].sort_values("bus").reset_index(drop=True)
            res = label_residual(bb, yg[s])
            vm = bb["Vm"].to_numpy()
            ang = max_branch_angle(label_voltage(bb), rg[s])
            stats["max_residual_recovered"] = max(stats["max_residual_recovered"], res)
            stats["min_vm_recovered"] = min(stats["min_vm_recovered"], float(vm.min()))
            stats["max_angle_recovered"] = max(stats["max_angle_recovered"], ang)
            if not res < TOL_RES:
                fail(f"(a) scenario {s}: residual {res:.3g} >= {TOL_RES}")
            if not (np.all(np.isfinite(vm)) and vm.min() >= R.VM_MIN):
                fail(f"(a) scenario {s}: min Vm {vm.min():.4g} (need >= {R.VM_MIN} and positive)")
            if not ang <= R.ANG_MAX:
                fail(f"(a) scenario {s}: max branch angle {ang:.2f} deg > {R.ANG_MAX}")
            if _max_diff(og[s], bg[s], ["Vm", "Va"]) == 0.0:
                fail(f"(a) scenario {s}: logged as recovered but its labels are unchanged")
            g = gg[s]; on = g["in_service"] == 1
            gs = g[on].groupby("bus")[["p_mw", "q_mvar"]].sum().reindex(bb["bus"].to_numpy()).fillna(0.0)
            isg = ((bb["PV"] == 1) | (bb["REF"] == 1)).to_numpy()
            gap = max(np.abs(gs["p_mw"].to_numpy()[isg] - bb["Pg"].to_numpy()[isg]).max(),
                      np.abs(gs["q_mvar"].to_numpy()[isg] - bb["Qg"].to_numpy()[isg]).max()) if isg.any() else 0.0
            if gap > 1e-6:
                fail(f"(a) scenario {s}: bus Pg/Qg differ from in-service generator sums by {gap:.3g} MW")
            br = rg[s]; onb = br["br_status"].to_numpy() == 1
            v = label_voltage(bb)
            c = lambda k: (br[f"{k}_r"] + 1j * br[f"{k}_i"]).to_numpy()  # noqa: E731
            vf, vt = v[br["from_bus"].to_numpy()], v[br["to_bus"].to_numpy()]
            sf = vf * np.conj(c("Yff") * vf + c("Yft") * vt) * 100
            st = vt * np.conj(c("Ytf") * vf + c("Ytt") * vt) * 100
            bgap = max(np.abs(sf.real - br["pf"].to_numpy())[onb].max(), np.abs(sf.imag - br["qf"].to_numpy())[onb].max(),
                       np.abs(st.real - br["pt"].to_numpy())[onb].max(), np.abs(st.imag - br["qt"].to_numpy())[onb].max()) if onb.any() else 0.0
            if bgap > 1e-6:
                fail(f"(a) scenario {s}: branch flows differ from the Ybus branch formula by {bgap:.3g} MW")
    for s in sorted(recovered - seen_rec):
        fail(f"(a) scenario {s} is logged as recovered but is not in the dataset")
    out = {"ok": not fails, "n_failures": len(fails), "failures": fails, **{k: (None if v == np.inf else v) for k, v in stats.items()}}
    say(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))
    for f in fails[:50]:
        say("FAIL " + f)
    return out


def run(orig, rep, log_path=None, out=None) -> int:
    res = verify(orig, rep, log_path)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:3]))
