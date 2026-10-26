"""Label-free GENCO adaptation with a stronger INPUT-ONLY prior (follow-up to ``labelfree``).

The clean label-free run (``results/2026-10-08-genco-labelfree-clean``) pulls GENCO toward the DC
power-flow state. It loses to DC on branch angle difference exactly where that prior is weakest
(case162, case300). This driver keeps every setting of that run (prior weight 1, lr 3e-5, all
parameters, slack |V| known, input-only normalizer, input-only pool, the same ``splits_lf`` splits)
and changes ONE thing: the training target. Candidates, all computed from a scenario's inputs only
(Ybus, P/Q injections at the buses where power flow specifies them, PV/REF |V| setpoints, REF angle):

``dc``   the clean run's prior (Va := Va_dc, PQ |V| := 1). Its dev-grid runs are reused, not retrained.
``fd1``  DC, then 1 fast-decoupled XB iteration (``hybrid.Scenario.fdxb_step``).
``fd2``  DC, then 2 XB iterations.
``nr1``  DC, then 1 Newton step, damped by backtracking when it does not reduce the mismatch
         (``hybrid.nr_ls_step``).
``nr2``  DC, then 2 such steps.
``nr3``  DC, then 3 such steps (a truncated solver; reported with its cost and how often it has
         already converged, since a converged prior is the power-flow solution computed by us).

Per-scenario guard (input-only): if a candidate state is non-finite, has any |V| outside
[0.5, 1.5] pu, or a larger max power mismatch than the DC state, that scenario keeps the DC prior
(counted per grid). Priors are computed on TRAINING inputs only; the eval set is never touched by
training. The AC solution is read only to score the eval set.

Stages (env ``grid-operations-genco``, repo root, PYTHONPATH = tracks/genco-challenge/src):

``protocol``  candidate list + selection rule -> ``hyperparameter_grid.json`` (+ sha256), refuses overwrite.
``build``     training sets ``genco-ladder/train_prior2_<variant>/<grid>`` -> ``priors_build.jsonl``.
``train``     selection runs: every new candidate on the dev grids (case14, case30), n = 1000, draw 0.
``select``    choice from validation physics residual only -> ``selection.json``, ``selection_table.csv``.
``final``     the chosen candidate on all 9 grids, draws 0-2, n = 1000 (``labelfree.final``).
``control``   every candidate prior ITSELF as a predictor on the eval sets (after selection only).
``summary``   ``summary.json``: clean run (old) -> new, dev and untouched grids separately, controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genco_challenge import fewshot as fs
from genco_challenge import hybrid as hy
from genco_challenge import labelfree as lf
from genco_challenge.dcprior import DC_TRAIN
from genco_challenge.ladder_eval import errors

OUT = fs.REPO / "tracks/genco-challenge/results/2026-10-08-genco-labelfree-prior2"
CLEAN = lf.OUT                                   # read only
WORK = fs.WORK
VARIANTS = {"dc": ("none", 0), "fd1": ("fdxb", 1), "fd2": ("fdxb", 2),
            "nr1": ("nr_ls", 1), "nr2": ("nr_ls", 2), "nr3": ("nr_ls", 3)}
COST_ORDER = list(VARIANTS)                      # tie-break: the cheaper prior
CLEAN_CHOSEN = "lfc_w1_lr3e-05_all"              # the clean run's selected optimizer settings
OVERRIDES = ["training.loss_weights=[1.0,1]", "optimizer.learning_rate=3e-05"]
VM_BOUNDS = (0.5, 1.5)
TIE_TOL = 0.02
FINAL_N = 1000

# every labelfree / fewshot side effect lands in THIS results dir, never in the clean one
lf.OUT = OUT
fs.OUT = OUT


def train_root(variant: str) -> Path:
    return DC_TRAIN if variant == "dc" else WORK / f"train_prior2_{variant}"


def cfg_for(variant: str) -> dict:
    name = CLEAN_CHOSEN if variant == "dc" else f"lfp2_{variant}_w1_lr3e-05_all"
    return {"name": name, "variant_prior": variant, "prior_weight": 1.0, "lr": 3e-5, "trainable_set": "all",
            "trainable": None, "overrides": OVERRIDES, "train_root": str(train_root(variant))}


def use_variant(variant: str) -> dict:
    """Point the shared fewshot/labelfree globals at this variant's training root."""
    fs.TRAIN_DATA = lf.DC_TRAIN = train_root(variant)
    return cfg_for(variant)


# ----------------------------------------------------------------------------- protocol

def write_protocol() -> dict:
    path = OUT / "hyperparameter_grid.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the protocol is written once, before any run")
    OUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "question": "Does a stronger INPUT-ONLY prior than DC fix label-free GENCO's branch-angle losses "
                    "(case162 1.90x, case300 1.23x DC in the clean run)?",
        "candidates": {
            "dc": "Va := Va_dc, PQ |V| := 1 (the clean run's prior; dev-grid runs reused)",
            "fd1": "DC state, then 1 fast-decoupled XB iteration (hybrid.Scenario.fdxb_step)",
            "fd2": "DC state, then 2 XB iterations",
            "nr1": "DC state, then 1 Newton step with backtracking (Armijo, halving to 1/64; hybrid.nr_ls_step)",
            "nr2": "DC state, then 2 such Newton steps",
            "nr3": "DC state, then 3 such Newton steps (truncated solver; cost and converged fraction reported)",
        },
        "prior_inputs": "Ybus (y_bus_data), P = Pg - Pd at PV+PQ, Q = -Qd at PQ (Qg = 0 at every PQ bus, checked), "
                        "|V| setpoint at PV/REF and REF angle (bus_data Vm/Va at those buses, as dcprior.py), "
                        "branch x for XB B'. No slack Pg, no PV Qg, no PQ |V|, no non-REF angle label is read.",
        "guard": f"per scenario: non-finite, any |V| outside {list(VM_BOUNDS)} pu, or max mismatch above the DC "
                 "state's -> that scenario keeps the DC prior (counted).",
        "fixed": {"optimizer_settings": f"{CLEAN_CHOSEN} (prior weight 1, lr 3e-5, all parameters), chosen in the clean run",
                  "overrides": OVERRIDES, "slack_vm": True, "normalizer": "input-only (FEWSHOT_INPUT_NORM=1)",
                  "pool_and_splits": "labelfree.pool_lf / splits_lf (identical to the clean run)",
                  "recipe": "GENCO case118_1000 fine-tune recipe, pretrained init"},
        "configs": [cfg_for(v) for v in VARIANTS],
        "dev_grids": lf.DEV, "selection_runs": {"n": FINAL_N, "draw": 0, "eval_set": "never evaluated"},
        "selection_rule": [
            "R[v,g] = min over epochs of 'Validation layer_11_residual' (label-free physics residual on the "
            "validation split; graphkit's checkpoint metric) for candidate v on dev grid g.",
            "r[v,g] = R[v,g] / min_v R[v,g]; S[v] = geometric mean over the 2 dev grids of r[v,g].",
            "Eligible: both dev runs finished (a crashed run is retried, never scored).",
            f"Choose argmin S; among candidates with S <= (1 + {TIE_TOL}) * min S, choose the cheapest in the "
            f"order {COST_ORDER}.",
            "If dc is chosen, the clean run's prior stands and the final runs reproduce the clean run's n=1000 rows.",
            "No eval-set number and no AC label enters selection; the prior-as-predictor controls are computed "
            "only after selection.json exists.",
        ],
        "final": {"grids": fs.GRIDS, "draws": lf.DRAWS, "n": [FINAL_N],
                  "report": "dev (case14, case30) vs 7 untouched grids; residual, branch dVa (lead), |V|; ratio to "
                            "DC; min-max over draws; clean run (old) -> new; every candidate prior as a predictor."},
    }
    text = json.dumps(proto, indent=1)
    path.write_text(text)
    (OUT / "hyperparameter_grid.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  hyperparameter_grid.json\n")
    return proto


# ----------------------------------------------------------------------------- input-only priors

def input_problem(b: pd.DataFrame, y: pd.DataFrame, br: pd.DataFrame) -> hy.Scenario:
    """A ``hybrid.Scenario`` built from inputs only (no ``reference_root``, no AC outputs)."""
    sc = hy.Scenario.__new__(hy.Scenario)
    b = b.sort_values("bus")
    n = len(b)
    if not (b["bus"].to_numpy() == np.arange(n)).all():
        raise ValueError("bus indices are not 0..n-1")
    sc.n = n
    sc.ybus = sp.csr_matrix((y["G"].to_numpy() + 1j * y["B"].to_numpy(),
                             (y["index1"].to_numpy(), y["index2"].to_numpy())), shape=(n, n))
    sc.is_t = {t: b[t].to_numpy() == 1 for t in hy.TYPES}
    pq, pv, ref = sc.is_t["PQ"], sc.is_t["PV"], sc.is_t["REF"]
    if np.abs(b["Qg"].to_numpy()[pq]).max(initial=0) > 1e-9 or np.abs(b["Pg"].to_numpy()[pq]).max(initial=0) > 1e-9:
        raise ValueError("generation at a PQ bus: Q injection would not be an input")
    p = np.where(ref, 0.0, b["Pg"].to_numpy() - b["Pd"].to_numpy())     # slack P unknown: never used
    q = np.where(pq, -b["Qd"].to_numpy(), 0.0)                          # PV/REF Q unknown: never used
    sc.sbus = (p + 1j * q) / hy.BASE_MVA
    sc.pv, sc.pq, sc.ref = np.where(pv)[0], np.where(pq)[0], np.where(ref)[0]
    sc.pvpq = np.r_[sc.pv, sc.pq]
    sc.vm_set = np.where(pv | ref, b["Vm"].to_numpy(), 1.0)
    sc.va_ref = float(np.radians(b["Va"].to_numpy()[ref]).mean())
    sc.va_dc = np.radians(b["Va_dc"].to_numpy())
    on = br[br["br_status"] == 1]
    f, t = on["from_bus"].to_numpy(), on["to_bus"].to_numpy()
    bx = 1.0 / on["x"].to_numpy()
    bp = sp.coo_matrix((np.r_[bx, bx, -bx, -bx], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)).tocsr()
    sc._bp = bp[sc.pvpq, :][:, sc.pvpq].tocsc()
    sc._bpp = (-sc.ybus.imag)[sc.pq, :][:, sc.pq].tocsc()
    sc._bp_lu = sc._bpp_lu = None
    sc.v_ref, sc.ref_kind = None, "input_only"
    return sc


def priors_for(sc: hy.Scenario) -> dict[str, dict]:
    """Every candidate state for one scenario: {variant: {v, seconds, fallback, maxmis_pu, converged}}."""
    v_dc = sc.vm_set * np.exp(1j * sc.va_dc)
    m_dc = hy.maxmis(sc, v_dc)
    out = {"dc": {"v": v_dc, "seconds": 0.0, "fallback": False, "maxmis_pu": m_dc, "converged": m_dc <= hy.TOL}}
    for method in ("fdxb", "nr_ls"):
        v, t0 = v_dc.copy(), time.perf_counter()
        step = hy.STEPS[method]
        for k in range(1, 4 if method == "nr_ls" else 3):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    if hy._finite(v) and hy.maxmis(sc, v) > hy.TOL:
                        v = step(sc, v)
                except Exception:
                    v = v * np.nan
            sec = time.perf_counter() - t0
            name = {"fdxb": "fd", "nr_ls": "nr"}[method] + str(k)
            ok = hy._finite(v) and VM_BOUNDS[0] <= np.abs(v).min() and np.abs(v).max() <= VM_BOUNDS[1]
            m = hy.maxmis(sc, v) if ok else np.inf
            ok = ok and m <= m_dc
            out[name] = {"v": v.copy() if ok else v_dc, "seconds": sec, "fallback": not ok,
                         "maxmis_pu": m if ok else m_dc, "converged": bool(ok and m <= hy.TOL)}
    return out


def _unwrapped(sc: hy.Scenario, v: np.ndarray) -> np.ndarray:
    """Angle in degrees, continuous with the DC angle (no +-180 wrap in the training target)."""
    return np.degrees(sc.va_dc + np.angle(v * np.exp(-1j * sc.va_dc)))


def _grid_tables(root: Path, grid: str, scen: list[int] | None = None):
    raw = root / grid / "raw"
    flt = [("scenario", "in", list(map(int, scen)))] if scen is not None else None
    bus = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    y = pd.read_parquet(raw / "y_bus_data.parquet", filters=flt, columns=["scenario", "index1", "index2", "G", "B"])
    br = pd.read_parquet(raw / "branch_data.parquet", filters=flt,
                         columns=["scenario", "from_bus", "to_bus", "x", "br_status"])
    return bus, dict(tuple(y.groupby("scenario"))), dict(tuple(br.groupby("scenario")))


def _build_grid(grid: str) -> list[dict]:
    src = lf.ORIG_TRAIN
    bus, ys, brs = _grid_tables(src, grid)
    pool = set(lf.pool_lf(grid))
    new = {v: {} for v in VARIANTS if v != "dc"}
    stats: dict[str, list] = {v: [] for v in VARIANTS}
    for s, b in bus.groupby("scenario"):
        b = b.sort_values("bus")
        sc = input_problem(b, ys[s], brs[s])
        pr = priors_for(sc)
        for v, r in pr.items():
            if v != "dc":
                new[v][int(s)] = (_unwrapped(sc, r["v"]), np.abs(r["v"]))
            if int(s) in pool:
                stats[v].append((r["seconds"], r["fallback"], r["maxmis_pu"] * hy.BASE_MVA, r["converged"]))
    rows = []
    pq = bus["PQ"].to_numpy() == 1
    for v, d in new.items():
        dst = train_root(v) / grid
        (dst / "raw").mkdir(parents=True, exist_ok=True)
        for f in (src / grid / "raw").iterdir():
            if f.name != "bus_data.parquet" and not (dst / "raw" / f.name).exists():
                os.symlink(f, dst / "raw" / f.name)
        if not (dst / "meta.json").exists():
            os.symlink(src / grid / "meta.json", dst / "meta.json")
        out = bus.copy()
        va = np.empty(len(out))
        vm = np.empty(len(out))
        for s, idx in out.groupby("scenario").groups.items():
            order = out.loc[idx, "bus"].to_numpy()
            a, m = d[int(s)]
            va[out.index.get_indexer(idx)] = a[order]
            vm[out.index.get_indexer(idx)] = m[order]
        out["Va"] = va
        out["Vm"] = np.where(pq, vm, bus["Vm"].to_numpy())      # PV/REF |V|: the setpoint, unchanged
        out.to_parquet(dst / "raw/bus_data.parquet")
    for v, st in stats.items():
        a = np.array(st, dtype=float)
        rows.append({"grid": grid, "variant": v, "pool_scenarios": len(a), "fallback_to_dc": int(a[:, 1].sum()),
                     "prior_ms_median": float(np.median(a[:, 0]) * 1e3), "prior_ms_max": float(a[:, 0].max() * 1e3),
                     "maxmis_mw_median": float(np.median(a[:, 2])), "maxmis_mw_max": float(a[:, 2].max()),
                     "converged_frac_1e-8pu": float(a[:, 3].mean())})
    return rows


def build(grids: list[str], procs: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(procs) as ex:
        for r in ex.map(_build_grid, grids):
            rows += r
            for x in r:
                print(json.dumps(x), flush=True)
    with (OUT / "priors_build.jsonl").open("a") as fh:
        for x in rows:
            fh.write(json.dumps(x | {"written": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return rows


# ----------------------------------------------------------------------------- runs

def protocol() -> dict:
    return json.loads((OUT / "hyperparameter_grid.json").read_text())


def train(variants: list[str], grids: list[str], draws: list[int]) -> None:
    for v in variants:
        cfg = use_variant(v)
        for g in grids:
            if not (train_root(v) / g / "raw/bus_data.parquet").exists():
                raise SystemExit(f"{v}/{g}: prior not built")
            for d in draws:
                lf.train_one(cfg, g, FINAL_N, d)


def select() -> dict:
    table = []
    for v in VARIANTS:
        cfg = cfg_for(v)
        for g in lf.DEV:
            key = lf._key(cfg, g, FINAL_N, 0)
            rec_path = fs.FS / "runs" / f"{key}.json"
            row = {"variant": v, "grid": g, "key": key, "failed": not rec_path.exists()}
            if rec_path.exists():
                rec = json.loads(rec_path.read_text())
                run_dir = Path(rec["model"]).parent
                res = lf._series(run_dir, "Validation layer_11_residual")
                step = min(res, key=res.get)
                row |= {"val_residual": res[step], "best_step": step, "epochs_logged": len(res),
                        "val_residual_last": res[max(res)], "kept_pretrained": lf._is_pretrained(rec["model"])}
            table.append(row)
    ok = [r for r in table if not r["failed"]]
    for g in lf.DEV:
        rs = [r for r in ok if r["grid"] == g]
        rmin = min(r["val_residual"] for r in rs)
        for r in rs:
            r["r"] = r["val_residual"] / rmin
    agg = []
    for v in VARIANTS:
        rs = [r for r in table if r["variant"] == v]
        if any(r["failed"] for r in rs):
            agg.append({"variant": v, "eligible": False, "why": "missing/failed dev run"})
            continue
        agg.append({"variant": v, "eligible": True, "S": float(np.exp(np.mean([np.log(r["r"]) for r in rs]))),
                    **{f"r_{r['grid']}": r["r"] for r in rs}, **{f"R_{r['grid']}": r["val_residual"] for r in rs}})
    el = [a for a in agg if a["eligible"]]
    smin = min(a["S"] for a in el)
    near = [a for a in el if a["S"] <= (1 + TIE_TOL) * smin]
    chosen = min(near, key=lambda a: COST_ORDER.index(a["variant"]))
    out = {"chosen_variant": chosen["variant"], "chosen": cfg_for(chosen["variant"])["name"], "best_S": smin,
           "tie_set": [a["variant"] for a in near], "per_run": table,
           "per_variant": sorted(agg, key=lambda a: a.get("S", np.inf)), "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    with (OUT / "selection_table.csv").open("w", newline="") as fh:
        cols = ["variant", "eligible", "S", *[f"r_{g}" for g in lf.DEV], *[f"R_{g}" for g in lf.DEV], "why"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a in out["per_variant"]:
            w.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in a.items()})
    return out


def final(grids: list[str], draws: list[int]) -> None:
    sel = json.loads((OUT / "selection.json").read_text())
    use_variant(sel["chosen_variant"])
    # labelfree.final resolves the chosen config through OUT/hyperparameter_grid.json "configs"
    lf.final([FINAL_N], grids, draws)


# ----------------------------------------------------------------------------- controls

def _eval_pred(grid: str) -> Path:
    """Predictions of the clean run's draw-0 n=1000 eval (its scenario set and targets)."""
    key = f"eval__{CLEAN_CHOSEN}_slackvm__finetune__{grid}__n{FINAL_N}"
    return fs._find_run(fs.FS / "mlruns_eval", key) / "artifacts/test" / f"{grid}_predictions.parquet"


def _ac_residual_mw(sc: hy.Scenario, v: np.ndarray) -> float:
    """graphkit's 'Avg. active res. (MW)' for a voltage-only predictor: mean |P mismatch| over ALL buses,
    with the slack Pg completed from the injection (its residual 0, as GENCO's GNS head does)."""
    p, _ = sc.f(v)
    return float(np.abs(p).sum() * hy.BASE_MVA / sc.n)


def _control_grid(grid: str) -> dict:
    pred = pd.read_parquet(_eval_pred(grid))
    scen = sorted(pred["scenario"].unique())
    bus, ys, brs = _grid_tables(fs.EVAL_DATA, grid, scen)
    branch = pd.read_parquet(fs.EVAL_DATA / grid / "raw/branch_data.parquet",
                             columns=["scenario", "from_bus", "to_bus", "br_status"])
    lab = bus[["scenario", "bus", "Va_dc", "Vm", "Pd"]]
    cols = {v: {"Vm": [], "Va": []} for v in VARIANTS}
    res = {v: [] for v in VARIANTS} | {"genco_check": []}
    fb = {v: 0 for v in VARIANTS}
    keyed = pred.set_index(["scenario", "bus"]).sort_index()
    order = []
    for s, b in bus.groupby("scenario"):
        b = b.sort_values("bus")
        sc = input_problem(b, ys[s], brs[s])
        pr = priors_for(sc)
        order.append(int(s))
        for v, r in pr.items():
            cols[v]["Vm"].append(np.abs(r["v"]))
            cols[v]["Va"].append(np.radians(_unwrapped(sc, r["v"])))
            res[v].append(_ac_residual_mw(sc, r["v"]))
            fb[v] += int(r["fallback"])
        g = keyed.loc[int(s)].sort_index()
        res["genco_check"].append(_ac_residual_mw(sc, g["Vm_pred"].to_numpy() * np.exp(1j * g["Va_pred"].to_numpy())))
    base = pred.set_index(["scenario", "bus"]).loc[[(s, i) for s in order for i in range(len(cols["dc"]["Vm"][0]))]]
    base = base.reset_index()
    out = {"grid": grid, "scenarios": len(order),
           "graphkit_genco_active_res_mw": float(pred["active res. (MW)"].abs().mean()),
           "recomputed_genco_active_res_mw": float(np.mean(res["genco_check"]))}
    for v in VARIANTS:
        p = base.copy()
        p["Vm_pred"] = np.concatenate(cols[v]["Vm"])
        p["Va_pred"] = np.concatenate(cols[v]["Va"])
        e = errors(p, lab, branch)["genco"]
        out[v] = {"active_res_mw": float(np.mean(res[v])), "branch_dva_mae_deg": e["branch_dva_mae_deg"],
                  "vm_mae_pu": e["vm_mae_pu"], "va_mae_deg": e["va_mae_deg"], "fallback_to_dc": fb[v]}
    e_dc = errors(base, lab, branch)["dc"]
    out["dc_reference"] = {"branch_dva_mae_deg": e_dc["branch_dva_mae_deg"], "vm_mae_pu": e_dc["vm_mae_pu"]}
    return out


def control(grids: list[str], procs: int) -> list[dict]:
    if not (OUT / "selection.json").exists():
        raise SystemExit("controls are computed only after selection.json exists")
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(_control_grid, grids))
    (OUT / "control_priors.json").write_text(json.dumps(rows, indent=1))
    return rows


# ----------------------------------------------------------------------------- summary

def summary() -> dict:
    sel = json.loads((OUT / "selection.json").read_text())
    new = fs._latest_ok(OUT / "results.jsonl")
    old = fs._latest_ok(CLEAN / "results.jsonl")
    ctrl = {r["grid"]: r for r in json.loads((OUT / "control_priors.json").read_text())}
    rows = []
    for g in fs.GRIDS:
        nr = [r for r in new.values() if r["grid"] == g and r["n"] == FINAL_N]
        orr = [r for r in old.values() if r["grid"] == g and r["n"] == FINAL_N and r["config"] == CLEAN_CHOSEN]
        if not nr:
            continue
        dc = {"res": float(np.mean([r["dc_active_res_mw"] for r in nr])),
              "dva": float(np.mean([r["errors"]["dc"]["branch_dva_mae_deg"] for r in nr])),
              "vm": float(np.mean([r["errors"]["dc"]["vm_mae_pu"] for r in nr]))}
        c = ctrl.get(g, {})
        rows.append({"grid": g, "group": "dev" if g in lf.DEV else "untouched", "draws": len(nr),
                     "new": lf._stats(nr) | {"keys": sorted(r["key"] for r in nr)},
                     "old_clean": lf._stats(orr) | {"keys": sorted(r["key"] for r in orr)} if orr else None,
                     "prior_as_predictor_ratio_to_dc": {
                         v: {"res": c[v]["active_res_mw"] / dc["res"], "dva": c[v]["branch_dva_mae_deg"] / dc["dva"],
                             "vm": c[v]["vm_mae_pu"] / dc["vm"]} for v in VARIANTS if v in c},
                     "prior_as_predictor_abs": {v: c[v] for v in VARIANTS if v in c}, "dc_abs": dc})
    unt = [r for r in rows if r["group"] == "untouched"]
    verdict = {"chosen_variant": sel["chosen_variant"],
               "untouched_beat_dc_res": [r["grid"] for r in unt if r["new"]["res"]["ratio_mean"] < 1],
               "untouched_beat_dc_dva": [r["grid"] for r in unt if r["new"]["dva"]["ratio_mean"] < 1],
               "untouched_beat_dc_both_every_draw": [r["grid"] for r in unt
                                                     if r["new"]["res"]["beats_dc"] == r["new"]["draws"]
                                                     and r["new"]["dva"]["beats_dc"] == r["new"]["draws"]],
               "old_untouched_beat_dc_dva": [r["grid"] for r in unt if r["old_clean"]
                                             and r["old_clean"]["dva"]["ratio_mean"] < 1]}
    out = {"verdict": verdict, "rows": rows}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("protocol")
    b = sub.add_parser("build")
    b.add_argument("--grids", nargs="+", default=fs.GRIDS)
    b.add_argument("--procs", type=int, default=5)
    t = sub.add_parser("train")
    t.add_argument("--variants", nargs="+", default=[v for v in VARIANTS if v != "dc"])
    t.add_argument("--grids", nargs="+", default=lf.DEV)
    t.add_argument("--draws", nargs="+", type=int, default=[0])
    sub.add_parser("select")
    f = sub.add_parser("final")
    f.add_argument("--grids", nargs="+", default=fs.GRIDS)
    f.add_argument("--draws", nargs="+", type=int, default=lf.DRAWS)
    c = sub.add_parser("control")
    c.add_argument("--grids", nargs="+", default=fs.GRIDS)
    c.add_argument("--procs", type=int, default=5)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "protocol":
        print(json.dumps(write_protocol(), indent=1))
    elif a.cmd == "build":
        build(a.grids, a.procs)
    elif a.cmd == "train":
        train(a.variants, a.grids, a.draws)
    elif a.cmd == "select":
        s = select()
        print(json.dumps({k: s[k] for k in ("chosen_variant", "best_S", "tie_set")}, indent=1))
    elif a.cmd == "final":
        final(a.grids, a.draws)
    elif a.cmd == "control":
        print(json.dumps(control(a.grids, a.procs), indent=1))
    else:
        print(json.dumps(summary()["verdict"], indent=1))


if __name__ == "__main__":
    main()
