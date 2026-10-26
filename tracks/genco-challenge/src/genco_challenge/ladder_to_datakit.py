"""Convert real-grid "ladder" operating-point shards into gridfm-datakit raw tables.

The ladder shards (``experiments/2026-10-18-genladder/<case>.npz``, made by
``powerflow_surrogate.generate``) hold, per sample, the per-load demand (``p_mw``, ``q_mvar``, in
``net.load`` order) and the verified AC solution (``vm_pu``, ``va_deg``, in ``net.bus`` order). Only
loads were perturbed; generator dispatch is the base case's (``generate.prepare_base_case``), and the
slack(s) absorb the imbalance. This module turns them into the four tables GENCO / gridfm-graphkit read:
``<out>/<case>/raw/{bus,branch,gen,y_bus}_data.parquet``.

How the tables are made (so conventions match the published ones BY CONSTRUCTION):

1. ``prepare_base_case`` rebuilds the exact base net the generator used; one ``runpp`` gives pandapower's
   internal ppc (MATPOWER layout), from which a datakit ``Network`` is built once per case.
2. Per sample: loads are set, ``pp.runpp(init="results")`` starts from the stored voltages and is held to
   them (|dVm| < 1e-6 pu, |dVa| < 1e-4 deg, else the sample is dropped and counted). The converged state
   supplies Pd/Qd (net of sgens, see below), per-generator Pg/Qg, and the voltages.
3. The tables are produced by datakit's OWN ``process_network.pf_post_processing`` (Ybus, branch
   admittances, flows, bus-type one-hots, DC columns). Its inputs are a PowerModels-shaped result dict,
   which is filled here from the solved state (branch flows from datakit's own ``makeYbus`` Yf/Yt).
4. The DC columns follow PowerModels' ``solve_dc_pf`` as used by datakit (identified empirically against
   the published subgrid, max error 2e-10): branch susceptance x/(r^2+x^2), no tap, no phase shift,
   reference angle 0, non-reference generators fixed, the first generator at the reference bus takes the
   balance (datakit's ``apply_slack_single_gen``).

Conventions that differ from a plain pandapower->MATPOWER mapping, each forced by the datakit format:

* One REF bus (datakit asserts it). Extra pandapower ext_grids (case6495rte has 6, SimBench 3) become PV
  buses whose generator P is the solved value; the voltage solution is unchanged.
* Va is reported relative to the REF bus (REF Va = 0, as PowerModels fixes it), wrapped to [-180, 180).
  A global angle shift is not a physical change; the stored labels are compared BEFORE the shift.
* baseMVA = 100 (graphkit assumes it). SimBench nets carry sn_mva = 1; branch r, x, b are rescaled.
* Generators, two regimes.
  - PGLib cases (pglib_opf_*, default): datakit semantics. pandapower's from_mpc turns only the first gen at each
    type-2 bus into a gen (first at the REF bus into the ext_grid) and EVERY other .m gen (2nd+ at a bus, gens at
    type-1 buses) into an sgen; folding those into Pd gave Pd down to -2.8 GW and shrank graphkit's baseMVA rule.
    Here the .m file's gen rows are the gen_data rows (all of them, .m order, status/limits/costs from the file);
    each is mapped back to its pandapower element by replaying from_mpc's rule (limits cross-checked), and its P is
    the base dispatch (gen/sgen p, i.e. the OPF-dispatched value where the case needed one; the slack row takes the
    remainder). Bus Qg is the solved injection plus Qd, split among that bus's in-service gens proportionally to
    their Q range (published tables split arbitrarily - Ipopt - so no rule exists to reproduce; equal split if all
    ranges are 0). Pd/Qd are loads only. Bus types as PowerModels: REF stays, PV iff an in-service gen is there, else
    PQ. A slack gen that prepare_base_case re-enabled (its .m status was 0) is written in service. Branch and bus
    shunt physics stay the ppc-derived network pandapower SOLVED (it models some tapped branches slightly
    differently from MATPOWER: max |dY| 34 pu on case500), so the labels remain exact roots of the written tables.
  - Non-PGLib cases (SimBench, RTE pandapower copies): sgens (fixed P,Q injections) are folded into Pd/Qd as negative
    demand, as pandapower's ppc does; a PQ bus never carries Pg, which datakit asserts. Pd can therefore be negative
    (SimBench: PV/wind feed-in). This is the known remaining deviation from datakit generator semantics.
* Non-PGLib cases: out-of-service generators are not in pandapower's ppc and are not written.
* Generator limits the base case does not define (ext_grid rows of SimBench; pandapower otherwise
  writes +-1e9) get +-cap, cap = max(100, total base demand) rounded up to 100 MW. Cost columns are
  zero where the case has no poly_cost (SimBench, most RTE rows).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import scipy.sparse.linalg as spl

warnings.filterwarnings("ignore")

from gridfm_datakit.network import Network, makeYbus  # noqa: E402
from gridfm_datakit.process.process_network import pf_post_processing  # noqa: E402
from gridfm_datakit.utils.column_names import (  # noqa: E402
    BRANCH_COLUMNS, BUS_COLUMNS, DC_BRANCH_COLUMNS, DC_BUS_COLUMNS, DC_GEN_COLUMNS, GEN_COLUMNS,
    YBUS_COLUMNS,
)
from gridfm_datakit.utils.idx_brch import (  # noqa: E402
    ANGMAX, ANGMIN, BR_B, BR_R, BR_STATUS, BR_X, F_BUS, RATE_A, SHIFT, T_BUS, TAP,
)
from gridfm_datakit.utils.idx_bus import (  # noqa: E402
    BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE, GS, PD, PQ, PV, QD, REF, VM, VMAX, VMIN, ZONE,
)
from gridfm_datakit.utils.idx_bus import VA as VA_COL  # noqa: E402
from gridfm_datakit.utils.idx_cost import COST, MODEL, NCOST, POLYNOMIAL  # noqa: E402
from gridfm_datakit.utils.idx_gen import (  # noqa: E402
    GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN,
)

BASE_MVA = 100.0
TOL_VM = 1e-6          # pu, vs the stored label
TOL_VA = 1e-4          # degrees, vs the stored label
SENTINEL_MVA = 9999.0   # rate_a at or above this (or <= 0) is a placeholder, not a rating
TOL_RESID = 1e-6       # pu on 100 MVA, AC power mismatch from the written tables

BUS_COLS = BUS_COLUMNS + DC_BUS_COLUMNS
GEN_COLS = GEN_COLUMNS + DC_GEN_COLUMNS
BRANCH_COLS = BRANCH_COLUMNS + DC_BRANCH_COLUMNS
YBUS_COLS = YBUS_COLUMNS
_INT_COLS = {"scenario", "bus", "from_bus", "to_bus", "index1", "index2"}   # everything else float64
TABLES = {"bus_data": BUS_COLS, "branch_data": BRANCH_COLS, "gen_data": GEN_COLS, "y_bus_data": YBUS_COLS}
_ARR_KEY = {"bus_data": "bus", "branch_data": "branch", "gen_data": "gen", "y_bus_data": "Y_bus"}


def _schema(cols):
    return pa.schema([("scenario", pa.int64())] + [
        (c, pa.int64() if c in _INT_COLS else pa.float64()) for c in cols])


# --------------------------------------------------------------------------------------------------
# Core: one solved state -> datakit arrays (shared by the ladder path and the published round trip)
# --------------------------------------------------------------------------------------------------
def dc_power_flow(net: Network, pg_gen: np.ndarray, res_pu_base: float | None = None):
    """PowerModels-style DC power flow as datakit calls it (see module docstring). Returns
    (theta_rad, pf_pu, ok). Reference angle 0; fixed injections = Pg - Pd - Gs at non-reference buses."""
    n = net.buses.shape[0]
    base = net.baseMVA
    idx = net.idx_branches_in_service
    f = net.branches[idx, F_BUS].astype(int)
    t = net.branches[idx, T_BUS].astype(int)
    r, x = net.branches[idx, BR_R].real, net.branches[idx, BR_X].real
    b = x / (r * r + x * x)
    gb = net.gens[net.idx_gens_in_service, GEN_BUS].astype(int)
    pg_bus = np.bincount(gb, weights=pg_gen, minlength=n)
    inj = (pg_bus - net.buses[:, PD] - net.buses[:, GS]) / base
    B = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)).tocsr()
    ref = net.ref_bus_idx
    keep = np.setdiff1d(np.arange(n), [ref])
    theta = np.zeros(n)
    try:
        theta[keep] = spl.spsolve(B[keep][:, keep].tocsc(), inj[keep])
    except Exception:
        return theta, None, False
    if not np.all(np.isfinite(theta)):
        return theta, None, False
    return theta, b * (theta[f] - theta[t]), True


def tables_from_state(net: Network, scenario_index: int, vm: np.ndarray, va_rad: np.ndarray,
                      pg_gen: np.ndarray, qg_gen: np.ndarray, with_dc: bool = True):
    """Datakit tables for one scenario. ``net`` must already carry this scenario's Pd/Qd. Voltages are in
    Network (0-based contiguous) bus order; pg_gen/qg_gen in ``net.idx_gens_in_service`` order, MW/MVAr."""
    base = net.baseMVA
    nb = net.buses.shape[0]
    n_br = net.branches.shape[0]
    ing = net.idx_gens_in_service
    inb = net.idx_branches_in_service
    V = vm * np.exp(1j * va_rad)
    _, Yf, Yt = makeYbus(base, net.buses, net.branches)
    Sf, St = V[net.branches[:, F_BUS].astype(int)] * np.conj(Yf @ V), V[net.branches[:, T_BUS].astype(int)] * np.conj(Yt @ V)
    br = {str(i + 1): {"pf": Sf.real[i], "qf": Sf.imag[i], "pt": St.real[i], "qt": St.imag[i]} for i in range(n_br)}
    res = {
        "solution": {
            "pf": True,
            "branch": br,
            "gen": {str(i + 1): {"pg": pg_gen[k] / base, "qg": qg_gen[k] / base} for k, i in enumerate(ing)},
            "bus": {str(i): {"vm": vm[i], "va": va_rad[i]} for i in range(nb)},
        },
        "solve_time": 0.0,
    }
    res_dc = None
    if with_dc:
        th, pf_dc, ok = dc_power_flow(net, pg_gen)
        if ok:
            brd = {str(i + 1): {"pf": 0.0, "pt": 0.0} for i in range(n_br)}
            for k, i in enumerate(inb):
                brd[str(i + 1)] = {"pf": pf_dc[k], "pt": -pf_dc[k]}
            res_dc = {"solution": {"pf": True, "branch": brd,
                                   "bus": {str(i): {"va": th[i]} for i in range(nb)}}, "solve_time": 0.0}
    return pf_post_processing(scenario_index, net, res, res_dc, True)


def residual_from_tables(bus: np.ndarray, Y: np.ndarray, base: float = BASE_MVA) -> float:
    """Max |V conj(Y V) - ((Pg - Pd) + j(Qg - Qd))/base| over all buses, from raw table arrays
    (bus: BUS_COLS layout, Y: [scenario, index1, index2, G, B])."""
    c = {k: i for i, k in enumerate(["scenario"] + BUS_COLS)} if bus.shape[1] == len(BUS_COLS) + 1 else {k: i for i, k in enumerate(BUS_COLS)}
    n = bus.shape[0]
    V = bus[:, c["Vm"]] * np.exp(1j * np.deg2rad(bus[:, c["Va"]]))
    Ym = sp.coo_matrix((Y[:, -2] + 1j * Y[:, -1], (Y[:, -4].astype(int), Y[:, -3].astype(int))), shape=(n, n)).tocsr()
    S = V * np.conj(Ym @ V)
    inj = ((bus[:, c["Pg"]] - bus[:, c["Pd"]]) + 1j * (bus[:, c["Qg"]] - bus[:, c["Qd"]])) / base
    return float(np.abs(S - inj).max())


# --------------------------------------------------------------------------------------------------
# Ladder path: pandapower net -> template Network, per-sample conversion
# --------------------------------------------------------------------------------------------------
class Template:
    """Everything static about a case: the pandapower net, the datakit Network, and bookkeeping."""

    def __init__(self, case: str, fill_defaults: bool = True, rate_cap: float | None = None,
                 rate_fix: bool = True):
        import pandapower as pp
        from pandapower.pypower.idx_brch import BR_B_ASYM, BR_G, BR_G_ASYM, BR_R_ASYM, BR_X_ASYM
        from pandapower.pypower.makeYbus import makeYbus as pp_makeYbus
        from powerflow_surrogate.generate import prepare_base_case

        self.case = case
        net, self.how = prepare_base_case(case)
        pp.runpp(net, numba=True)
        ppc = net._ppc
        lk = net._pd2ppc_lookups
        nb = len(net.bus)
        if ppc["bus"].shape[0] != nb or not np.array_equal(lk["bus"][net.bus.index.values], np.arange(nb)):
            raise RuntimeError("ppc bus order differs from net.bus order (aux buses / reordering); not handled")
        if (ppc["bus"][:, BUS_TYPE] == 4).any():
            raise RuntimeError("isolated/out-of-service buses present; datakit format has none")
        for col in ("const_z_percent", "const_i_percent"):
            if col in net.load and (net.load[col].fillna(0) != 0).any():
                raise RuntimeError(f"voltage-dependent loads ({col}) not representable")
        if np.abs(ppc["branch"][:, [BR_R_ASYM, BR_X_ASYM, BR_G_ASYM, BR_B_ASYM]]).max() > 0:
            raise RuntimeError("asymmetric branch impedance not representable")
        old_base = float(ppc["baseMVA"])
        bus = ppc["bus"][:, :13].copy()
        brn = ppc["branch"][:, :13].copy()
        # pandapower branches carry a shunt conductance BR_G (transformer iron loss) that MATPOWER/datakit
        # branches cannot hold. It enters Yff as G/2/|tap|^2 and Ytt as G/2, i.e. exactly a bus shunt at each
        # end, so it is moved into the bus GS (MW at V = 1 pu): the physics, and the stored labels, are kept.
        g_pu = ppc["branch"][:, BR_G].real * ppc["branch"][:, BR_STATUS].real
        tap_ = np.where(ppc["branch"][:, TAP].real != 0, ppc["branch"][:, TAP].real, 1.0)
        np.add.at(bus[:, GS], ppc["branch"][:, F_BUS].real.astype(int), g_pu / 2 / tap_ ** 2 * old_base)
        np.add.at(bus[:, GS], ppc["branch"][:, T_BUS].real.astype(int), g_pu / 2 * old_base)
        self.gs_from_branch_g_mw = float(np.sum(g_pu) * old_base)
        brn[:, [BR_R, BR_X]] *= BASE_MVA / old_base
        brn[:, BR_B] *= old_base / BASE_MVA
        gen = ppc["gen"][:, :21].copy()
        gen[:, MBASE] = BASE_MVA
        bus[:, BUS_I] = np.arange(nb)
        # single REF: the first ext_grid's bus; other REF buses become PV
        ref_bus = int(gen[0, GEN_BUS])
        assert bus[ref_bus, BUS_TYPE] == REF
        bus[bus[:, BUS_TYPE] == REF, BUS_TYPE] = PV
        bus[ref_bus, BUS_TYPE] = REF
        self.n_extra_ref = int(len(net.ext_grid) - 1)
        # limits for ext_grid rows (pandapower writes +-1e9 / 0 there) and costs
        cap = float(np.ceil(max(100.0, net.load.p_mw.sum()) / 100.0) * 100.0)
        self.cap = cap
        cost = np.zeros((gen.shape[0], 7))
        cost[:, MODEL] = POLYNOMIAL
        cost[:, NCOST] = 3
        pc = net.poly_cost
        n_unbounded = 0
        for et, key in (("ext_grid", "ext_grid"), ("gen", "gen")):
            tab = getattr(net, et)
            for el, row in zip(tab.index, lk[key][tab.index.values] if len(tab) else []):
                if not bool(tab.at[el, "in_service"]):
                    continue
                if et == "ext_grid":
                    for col, lim in (("min_p_mw", PMIN), ("max_p_mw", PMAX), ("min_q_mvar", QMIN), ("max_q_mvar", QMAX)):
                        v = tab.at[el, col] if col in tab else np.nan
                        if np.isfinite(v):
                            gen[row, lim] = v
                        else:
                            gen[row, lim] = cap if col.startswith("max") else -cap
                            n_unbounded += 1
                sel = pc[(pc.et == et) & (pc.element == el)]
                if len(sel):
                    c = sel.iloc[0]
                    cost[row, COST:COST + 3] = [c["cp2_eur_per_mw2"], c["cp1_eur_per_mw"], c["cp0_eur"]]
        self.n_unbounded_limits = n_unbounded
        if not np.isfinite(gen[:, [PMIN, PMAX, QMIN, QMAX]]).all() or (np.abs(gen[:, [PMIN, PMAX, QMIN, QMAX]]) >= 1e8).any():
            raise RuntimeError("non-finite or >=1e8 generator limits remain")
        for c_ in (VMIN, VMAX):
            if not np.isfinite(bus[:, c_]).all():
                raise RuntimeError("non-finite bus voltage limits")
        # Limits the base case does not define. pandapower writes vmin = 0 / vmax = 2 for a bus without limits
        # and +-360 deg for every branch angle limit (it has none); the published datakit tables carry
        # 0.9/1.1 pu and +-30 deg. With fill_defaults (default) the undefined ones take those values so the
        # input features are not wildly out of distribution; the counts are recorded in the stats.
        self.fill_defaults = fill_defaults
        self.n_vm_limits_filled = self.n_angle_limits_filled = 0
        if fill_defaults:
            lo = net.bus["min_vm_pu"] if "min_vm_pu" in net.bus else pd.Series(np.nan, index=net.bus.index)
            hi = net.bus["max_vm_pu"] if "max_vm_pu" in net.bus else pd.Series(np.nan, index=net.bus.index)
            undef = (lo.isna() | hi.isna()).to_numpy()
            # a non-OPF ppc always carries vmin 0 / vmax 2; the case's real limits live in net.bus
            bus[~undef, VMIN], bus[~undef, VMAX] = lo.to_numpy()[~undef], hi.to_numpy()[~undef]
            bus[undef, VMIN], bus[undef, VMAX] = 0.9, 1.1
            self.n_vm_limits_filled = int(undef.sum())
            und = (brn[:, ANGMIN] <= -360) & (brn[:, ANGMAX] >= 360)
            brn[und, ANGMIN], brn[und, ANGMAX] = -30.0, 30.0
            self.n_angle_limits_filled = int(und.sum())
        # Branch ratings. rate_a is a GENCO edge INPUT feature. pandapower carries sentinels for "unrated"
        # (99999 on PGLib-derived nets, up to 6.6e7 on RTE); the published tables max out at 3620 MVA.
        # Unrated branches (rate_a <= 0 or >= SENTINEL_MVA) get PowerModels' calc_thermal_limits! value
        # (verified in PowerModels.jl src/core/data.jl, _calc_thermal_limits!):
        #   theta_max = max(|angmin|, |angmax|) [rad];  y = |1/(r + jx)|;  v = max(fr_vmax, to_vmax)
        #   c_max = sqrt(fr_vmax^2 + to_vmax^2 - 2 fr_vmax to_vmax cos(theta_max));  rate = y v c_max (pu) * baseMVA
        # It additionally caps by c_rating_a * v only when the case has a current rating, which a pandapower
        # net does not carry, so no such cap applies here.
        self.rate_fix = rate_fix
        r0 = brn[:, RATE_A].copy()
        q = lambda a: [float(x) for x in np.percentile(a, [0, 50, 99, 100])] if len(a) else None  # noqa: E731
        self.rate_before = q(r0)
        unrated = (r0 <= 0) | (r0 >= SENTINEL_MVA)
        self.n_rate_unrated = int(unrated.sum())
        self.n_rate_computed = 0
        if rate_fix and unrated.any():
            fb, tb = brn[:, F_BUS].astype(int), brn[:, T_BUS].astype(int)
            th = np.deg2rad(np.maximum(np.abs(brn[:, ANGMIN]), np.abs(brn[:, ANGMAX])))
            zmag = np.hypot(brn[:, BR_R], brn[:, BR_X])
            fv, tv = bus[fb, VMAX], bus[tb, VMAX]
            cmax = np.sqrt(np.maximum(fv ** 2 + tv ** 2 - 2 * fv * tv * np.cos(th), 0.0))
            new = (1.0 / zmag) * np.maximum(fv, tv) * cmax * BASE_MVA
            ok = unrated & np.isfinite(new) & (new > 0)
            brn[ok, RATE_A] = new[ok]
            self.n_rate_computed = int(ok.sum())
            self.n_rate_unresolved = int((unrated & ~ok).sum())
        else:
            self.n_rate_unresolved = int(unrated.sum()) if rate_fix else 0
        if rate_cap is not None:
            brn[brn[:, RATE_A] > rate_cap, RATE_A] = rate_cap
        self.rate_after = q(brn[:, RATE_A])
        self.rate_a_max = float(brn[:, RATE_A].max())
        self.dk = Network({"baseMVA": BASE_MVA, "bus": bus, "gen": gen, "branch": brn, "gencost": cost})
        # Ybus agreement with pandapower's own (rescaled to 100 MVA)
        Yp = pp_makeYbus(old_base, ppc["bus"], ppc["branch"])[0] * (old_base / BASE_MVA)
        Ym = makeYbus(BASE_MVA, self.dk.buses, self.dk.branches)[0]
        self.ybus_vs_pandapower = float(abs(Yp - Ym).max())
        self.net = net
        self.pg = None
        self.n_bus = nb
        self.n_load = len(net.load)
        self.ref_bus = ref_bus


PGLIB_DIR = Path(__file__).resolve().parents[4] / "data/external/pglib-opf"


class PglibGens:
    """Datakit-faithful generator semantics for PGLib cases: the Network is built from the .m file (every .m gen is
    a gen row, in .m order, status/limits/costs from the file); pandapower is used only to solve the state."""


def attach_pglib(tpl: Template) -> None:
    """Rebuild ``tpl.dk`` from the PGLib .m file and map each .m gen row to its pandapower element.
    from_mpc (pandapower/converter/pypower/from_ppc.py, _gen_to_which) makes: the first gen at a REF bus an ext_grid,
    the first gen at each type-2 bus a gen, and EVERY other gen (second+ gen at a bus, gens at type-1 buses) an sgen,
    all in .m row order; sgens made from negative-Pd buses come first and are non-controllable. That rule is
    replayed here, and checked against the elements' limits."""
    from matpowercaseframes import CaseFrames
    from pandapower.converter.pypower.from_ppc import _gen_to_which

    cf = CaseFrames(str(PGLIB_DIR / f"{tpl.case}.m"))
    bus, gen, brn = cf.bus.values.astype(float).copy(), cf.gen.values.astype(float).copy(), cf.branch.values.astype(float).copy()
    gc = cf.gencost.values.astype(float).copy()
    net = tpl.net
    m_bus, m_brn = bus.copy(), brn.copy()      # the .m network, kept only for the comparison below
    ids = bus[:, BUS_I].astype(int)
    if not np.array_equal(ids, np.sort(ids)):
        raise RuntimeError("PGLib .m bus ids are not sorted; rank mapping would differ from datakit's")
    rank = {int(b): k for k, b in enumerate(ids)}
    # Physics must stay the network pandapower SOLVED (it models some tapped branches slightly differently from
    # MATPOWER, so the .m Ybus is not the labels' Ybus): buses and branches come from the ppc-derived Network;
    # generators, bus types, loads and costs come from the .m file.
    bus, brn = tpl.dk.buses.copy(), tpl.dk.branches.copy()
    # pandapower bus labels: 0..n-1 (ext2int renumbering) for some cases, the original .m ids for others. Either way
    # the Network is the ppc-derived one, whose bus i is net.bus position i; datakit's order (sorted ids) must agree.
    idx = net.bus.index.values
    if len(idx) != len(bus) or not (np.diff(idx) > 0).all():
        raise RuntimeError("pandapower bus index is not strictly increasing / wrong length")
    if np.array_equal(idx, ids):
        m2lab = {int(b): int(b) for b in ids}
    elif np.array_equal(idx, ids - 1):
        m2lab = {int(b): int(b) - 1 for b in ids}
    elif np.array_equal(idx, np.arange(len(idx))):
        m2lab = {int(b): k for k, b in enumerate(ids)}
    else:
        raise RuntimeError("cannot map pandapower bus labels to .m bus ids")
    lab2pos = {int(l): k for k, l in enumerate(idx)}
    pos_rank = np.arange(len(idx))
    n = len(bus)
    # which pandapower element is each .m gen row
    is_eg, is_g, is_sg = _gen_to_which({"bus": m_bus, "gen": gen})
    if len(gen) != is_eg.sum() + is_g.sum() + is_sg.sum():
        raise RuntimeError("some .m gens map to no pandapower element")
    eg_idx, g_idx = list(net.ext_grid.index), list(net.gen.index)
    sg_idx = list(net.sgen.index[net.sgen.controllable.fillna(False).astype(bool)]) if "controllable" in net.sgen else []
    if (len(eg_idx), len(g_idx), len(sg_idx)) != (is_eg.sum(), is_g.sum(), is_sg.sum()):
        raise RuntimeError(f"element counts differ: pp {(len(eg_idx), len(g_idx), len(sg_idx))} vs .m "
                           f"{(is_eg.sum(), is_g.sum(), is_sg.sum())}")
    kind = np.empty(len(gen), dtype=object)
    elem = np.zeros(len(gen), dtype=int)
    for mask, lst, name in ((is_eg, eg_idx, "ext_grid"), (is_g, g_idx, "gen"), (is_sg, sg_idx, "sgen")):
        rows = np.where(mask)[0]
        kind[rows] = name
        elem[rows] = lst
    # limits cross-check (ext_grid rows carry limits too in from_mpc)
    for r in range(len(gen)):
        tab = getattr(net, kind[r])
        for col, gc_ in (("max_p_mw", PMAX), ("min_p_mw", PMIN), ("max_q_mvar", QMAX), ("min_q_mvar", QMIN)):
            if col in tab and np.isfinite(tab.at[elem[r], col]) and abs(tab.at[elem[r], col] - gen[r, gc_]) > 1e-6:
                raise RuntimeError(f"gen row {r} ({kind[r]} {elem[r]}) {col} differs from the .m file")
        if m2lab[int(gen[r, GEN_BUS])] != int(tab.at[elem[r], "bus"]):
            raise RuntimeError(f"gen row {r} bus mismatch")
    # status: .m status, except the ext_grid that prepare_base_case re-enabled
    status = np.array([bool(getattr(net, kind[r]).at[elem[r], "in_service"]) for r in range(len(gen))])
    gen[:, GEN_STATUS] = status.astype(float)
    tpl.n_slack_reenabled = int(((cf.gen.values[:, GEN_STATUS] <= 0) & status).sum())
    if "scaling" in net.sgen and not (net.sgen.scaling.fillna(1) == 1).all():
        raise RuntimeError("sgen scaling != 1")
    # base P of every non-slack row (loads are the only thing perturbed)
    p0 = np.zeros(len(gen))
    for r in range(len(gen)):
        if kind[r] != "ext_grid" and status[r]:
            p0[r] = getattr(net, kind[r]).at[elem[r], "p_mw"]
    ref_bus_rank = int(lab2pos[m2lab[int(m_bus[m_bus[:, BUS_TYPE] == REF][0, BUS_I])]])
    ext_row = int(np.where(is_eg)[0][0]) if is_eg.any() else None
    if ext_row is None or lab2pos[m2lab[int(gen[ext_row, GEN_BUS])]] != ref_bus_rank:
        raise RuntimeError("no gen at the REF bus")
    # bus types as datakit/PowerModels: REF stays, PV iff an in-service gen sits there, else PQ
    gbus = np.array([lab2pos[m2lab[int(b)]] for b in gen[:, GEN_BUS]])
    has_gen = np.zeros(n, bool)
    has_gen[gbus[status]] = True
    bt = np.where(has_gen, PV, PQ).astype(float)
    bt[ref_bus_rank] = REF
    bus[:, BUS_TYPE] = bt
    gen[:, GEN_BUS] = gbus          # bus ids of the (ppc-derived, 0-based) bus table
    dk = Network({"baseMVA": BASE_MVA, "bus": bus, "gen": gen, "branch": brn, "gencost": gc})
    pg = PglibGens()
    pg.dk, pg.kind, pg.elem, pg.status, pg.p0 = dk, kind, elem, status, p0
    pg.pos_rank, pg.gbus, pg.ref_rank, pg.ext_row = pos_rank, gbus, ref_bus_rank, ext_row
    pg.load_rank = np.array([lab2pos[int(l)] for l in net.load.bus.values])
    nc = net.sgen[~net.sgen.controllable.fillna(False).astype(bool)] if "controllable" in net.sgen else net.sgen.iloc[:0]
    pg.neg_rank = np.array([lab2pos[int(l)] for l in nc.bus.values], dtype=int)
    pg.neg_p, pg.neg_q = nc.p_mw.values.astype(float), nc.q_mvar.values.astype(float)
    pg.n_m_gens, pg.n_pv = len(gen), int((bt == PV).sum())
    pg.in_service_rows = np.where(status)[0]
    pg.qmin, pg.qmax = gen[:, QMIN], gen[:, QMAX]
    pg.counts = {"ext_grid": int(is_eg.sum()), "gen": int(is_g.sum()), "sgen": int(is_sg.sum())}
    tpl.pg = pg
    tpl.dk = dk
    tpl.ref_bus = ref_bus_rank
    # information only: how far the pure-.m network is from the one pandapower solved (tapped branches)
    mb = m_bus.copy()
    mb[:, BUS_I] = np.arange(n)
    mbr = m_brn.copy()
    mbr[:, F_BUS] = [rank[int(v)] for v in m_brn[:, F_BUS]]
    mbr[:, T_BUS] = [rank[int(v)] for v in m_brn[:, T_BUS]]
    tpl.ybus_m_vs_pandapower = float(abs(makeYbus(BASE_MVA, mb, mbr)[0] - makeYbus(BASE_MVA, dk.buses, dk.branches)[0]).max())


def split_q(pg: PglibGens, qbus: np.ndarray) -> np.ndarray:
    """Solved bus Qg -> per in-service gen, proportional to each gen's Q range (equal split if all ranges are 0)."""
    rows = pg.in_service_rows
    b = pg.gbus[rows]
    qmin, qmax = pg.qmin[rows], pg.qmax[rows]
    rng = qmax - qmin
    n = len(qbus)
    srng, sqmin = np.bincount(b, rng, n), np.bincount(b, qmin, n)
    cnt = np.bincount(b, minlength=n)
    f = np.where(srng > 0, (qbus - sqmin) / np.where(srng > 0, srng, 1), 0.0)
    q = np.where(srng[b] > 0, qmin + f[b] * rng, qbus[b] / cnt[b])
    return q


def _finish_pglib(i, tpl, vm_pp, va_pp, diag):
    pg, dk = tpl.pg, tpl.dk
    net = tpl.net
    n = dk.buses.shape[0]
    vm = np.empty(n)
    va = np.empty(n)
    vm[pg.pos_rank], va[pg.pos_rank] = vm_pp, va_pp
    pd_ = np.bincount(pg.load_rank, net.load.p_mw.values, n) - np.bincount(pg.neg_rank, pg.neg_p, n)
    qd_ = np.bincount(pg.load_rank, net.load.q_mvar.values, n) - np.bincount(pg.neg_rank, pg.neg_q, n)
    dk.buses[:, PD], dk.buses[:, QD] = pd_, qd_
    V = vm * np.exp(1j * np.deg2rad(va))
    S = V * np.conj(makeYbus(BASE_MVA, dk.buses, dk.branches)[0] @ V) * BASE_MVA
    pg_bus, qg_bus = S.real + pd_, S.imag + qd_
    prow = pg.p0.copy()
    at_ref = [r for r in pg.in_service_rows if pg.gbus[r] == pg.ref_rank and r != pg.ext_row]
    prow[pg.ext_row] = pg_bus[pg.ref_rank] - prow[at_ref].sum()
    sum_p = np.bincount(pg.gbus, prow, n)
    err_p = np.abs(sum_p - pg_bus)
    err_p[pg.ref_rank] = 0.0
    nogen = np.ones(n, bool)
    nogen[pg.gbus[pg.status]] = False
    diag["dispatch_err_mw"] = float(err_p.max())
    diag["nogen_inj_mvar"] = float(max(np.abs(qg_bus[nogen]).max() if nogen.any() else 0.0, np.abs(pg_bus[nogen]).max() if nogen.any() else 0.0))
    if err_p.max() > 1e-4 or diag["nogen_inj_mvar"] > 1e-4:
        return i, "dispatch_mismatch", None, diag
    qrow = split_q(pg, qg_bus)
    pin = prow[pg.in_service_rows]
    va_rel = np.deg2rad(va - va[pg.ref_rank])
    try:
        out = tables_from_state(dk, i, vm, va_rel, pin, qrow)
    except Exception as e:  # noqa: BLE001
        return i, f"table_error:{type(e).__name__}", None, diag
    res = residual_from_tables(out["bus"], out["Y_bus"])
    diag["resid"] = res
    diag["min_pd"] = float(pd_.min())
    if not res < TOL_RESID:
        return i, "residual", None, diag
    out.pop("runtime", None)
    return i, "ok", out, diag


_TPL: Template | None = None
_DATA: dict | None = None


def _convert_one(i: int):
    """Worker: sample ``i`` of the shard -> (i, status, arrays|None, diag). Fork-inherited globals."""
    import pandapower as pp
    tpl, d = _TPL, _DATA
    net, dk = tpl.net, tpl.dk
    p, q = d["p_mw"][i].astype(float), d["q_mvar"][i].astype(float)
    vm0, va0 = d["vm_pu"][i].astype(float), d["va_deg"][i].astype(float)
    if not (np.isfinite(vm0).all() and np.isfinite(va0).all()):
        return i, "stored_nonfinite", None, {}
    net.load["p_mw"], net.load["q_mvar"] = p, q
    net.res_bus["vm_pu"], net.res_bus["va_degree"] = vm0, va0
    try:
        pp.runpp(net, init="results", numba=True)
    except Exception:
        return i, "pandapower_failed", None, {}
    ppc = net._ppc
    vm, va = ppc["bus"][:, VM], ppc["bus"][:, VA_COL]
    dvm, dva = float(np.abs(vm - vm0).max()), float(np.abs(va - va0).max())
    diag = {"dvm": dvm, "dva": dva}
    if dvm > TOL_VM or dva > TOL_VA:
        return i, "label_mismatch", None, diag
    if tpl.pg is not None:
        return _finish_pglib(i, tpl, vm, va, diag)
    if ppc["gen"].shape[0] != dk.gens.shape[0]:
        return i, "gen_set_changed", None, diag
    dk.buses[:, PD] = ppc["bus"][:, PD]
    dk.buses[:, QD] = ppc["bus"][:, QD]
    va_rel = np.deg2rad(va - va[tpl.ref_bus])
    try:
        out = tables_from_state(dk, i, vm, va_rel, ppc["gen"][:, PG].copy(), ppc["gen"][:, QG].copy())
    except Exception as e:  # noqa: BLE001
        return i, f"table_error:{type(e).__name__}", None, diag
    res = residual_from_tables(out["bus"], out["Y_bus"])
    diag["resid"] = res
    if not res < TOL_RESID:
        return i, "residual", None, diag
    out.pop("runtime", None)
    return i, "ok", out, diag


class _Writers:
    def __init__(self, raw: Path):
        raw.mkdir(parents=True, exist_ok=True)
        self.w = {t: pq.ParquetWriter(raw / f"{t}.parquet", _schema(c)) for t, c in TABLES.items()}
        self.cols = TABLES

    def write(self, scen_ids: list[int], outs: list[dict]):
        for t, cols in self.cols.items():
            arrs = [o[_ARR_KEY[t]] for o in outs]
            sc = np.concatenate([np.full(a.shape[0], s, dtype=np.int64) for a, s in zip(arrs, scen_ids)])
            M = np.concatenate(arrs, axis=0)
            data = {"scenario": sc}
            for j, c in enumerate(cols):
                data[c] = M[:, j].astype(np.int64) if c in _INT_COLS else M[:, j].astype(np.float64)
            self.w[t].write_table(pa.table(data, schema=_schema(cols)))

    def close(self):
        for w in self.w.values():
            w.close()


def convert_case(case: str, shards: Path, out: Path, n: int, seed: int, workers: int, log=print,
                 fill_defaults: bool = True, rate_cap: float | None = None, rate_fix: bool = True,
                 pglib_gens: bool = True) -> dict:
    global _TPL, _DATA
    t0 = time.time()
    z = np.load(shards / f"{case}.npz")
    arrays = {k: z[k] for k in ("p_mw", "q_mvar", "vm_pu", "va_deg")}
    n_avail = arrays["p_mw"].shape[0]
    tpl = Template(case, fill_defaults, rate_cap, rate_fix)
    if arrays["p_mw"].shape[1] != tpl.n_load or arrays["vm_pu"].shape[1] != tpl.n_bus:
        raise RuntimeError(f"{case}: shard shape {arrays['p_mw'].shape}/{arrays['vm_pu'].shape} vs net "
                           f"{tpl.n_load} loads / {tpl.n_bus} buses")
    if case.startswith("pglib_opf_") and pglib_gens:
        attach_pglib(tpl)
    t_setup = time.time() - t0
    perm = np.random.default_rng(seed).permutation(n_avail)
    _TPL, _DATA = tpl, arrays
    raw = out / case / "raw"
    wr = _Writers(raw)
    kept, scen_map, drops = [], [], {}
    mx = {"dvm": 0.0, "dva": 0.0, "resid": 0.0, "dispatch_err_mw": 0.0, "nogen_inj_mvar": 0.0}
    min_pd = float("inf")
    cursor, batch_ids, batch_out = 0, [], []
    t1 = time.time()
    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers) if workers > 1 else None
    try:
        while len(kept) < n and cursor < n_avail:
            need = n - len(kept)
            chunk = perm[cursor:cursor + need].tolist()
            cursor += len(chunk)
            it = pool.imap(_convert_one, chunk, chunksize=2) if pool else map(_convert_one, chunk)
            for i, status, outarr, diag in it:
                for k in mx:
                    if k in diag:
                        mx[k] = max(mx[k], diag[k]) if status == "ok" else mx[k]
                if status == "ok" and "min_pd" in diag:
                    min_pd = min(min_pd, diag["min_pd"])
                if status != "ok":
                    drops[status] = drops.get(status, 0) + 1
                    continue
                sid = len(kept)
                kept.append(i)
                scen_map.append((sid, i))
                batch_ids.append(sid)
                batch_out.append(outarr)
                if len(batch_ids) >= 25:
                    wr.write(batch_ids, batch_out)
                    batch_ids, batch_out = [], []
            if cursor >= 3 * n + 50 and len(kept) < 0.2 * cursor:
                raise RuntimeError(f"{case}: only {len(kept)} of {cursor} attempted samples convert "
                                   f"(drops {drops}); aborting instead of scanning the shard")
    finally:
        if batch_ids:
            wr.write(batch_ids, batch_out)
        wr.close()
        if pool:
            pool.close()
            pool.join()
    dt = time.time() - t1
    # load_scenario_idx must be the shard index; patch it in via the map (datakit put it in col 0 of arrays
    # as scenario_index = shard index already), so nothing to do. Write the map for provenance.
    pd.DataFrame(scen_map, columns=["new", "orig"]).to_csv(raw / "scenario_map.csv", index=False)
    stats = {
        "case": case, "n_bus": tpl.n_bus, "n_branch": int(tpl.dk.branches.shape[0]),
        "n_gen": int(tpl.dk.gens.shape[0]), "base_how": tpl.how, "requested": n, "kept": len(kept),
        "attempted": cursor, "dropped": drops, "seed": seed,
        "tol_vm_pu": TOL_VM, "tol_va_deg": TOL_VA, "max_dvm_pu": mx["dvm"], "max_dva_deg": mx["dva"], "max_resid_pu": mx["resid"],
        "ybus_vs_pandapower_max_abs": tpl.ybus_vs_pandapower, "ybus_m_file_vs_solved_net_max_abs": getattr(tpl, "ybus_m_vs_pandapower", None), "branch_g_moved_to_bus_gs_mw": tpl.gs_from_branch_g_mw, "extra_ref_buses_made_pv": tpl.n_extra_ref,
        "rate_fix": tpl.rate_fix, "rate_a_before_min_p50_p99_max": tpl.rate_before,
        "rate_a_after_min_p50_p99_max": tpl.rate_after, "rate_a_max_mva": tpl.rate_a_max,
        "n_rate_a_unrated_sentinel": tpl.n_rate_unrated, "n_rate_a_computed": tpl.n_rate_computed,
        "n_rate_a_unresolved": tpl.n_rate_unresolved, "published_rate_a_max_mva": 3620.0,
        "rate_cap": rate_cap, "fill_defaults": tpl.fill_defaults,
        "gens_semantics": "PGLib .m gen rows (datakit)" if tpl.pg is not None else "pandapower ppc gens; sgens folded into Pd",
        "n_gens_written": int(tpl.dk.gens.shape[0]),
        "n_m_gens": tpl.pg.n_m_gens if tpl.pg is not None else None,
        "pandapower_elements_per_m_gen": tpl.pg.counts if tpl.pg is not None else None,
        "n_pv_buses": int((tpl.dk.buses[:, BUS_TYPE] == PV).sum()),
        "n_slack_gens_reenabled": getattr(tpl, "n_slack_reenabled", None),
        "min_pd_mw": None if min_pd == float("inf") else min_pd,
        "max_dispatch_err_mw": mx["dispatch_err_mw"], "max_nogen_bus_injection": mx["nogen_inj_mvar"], "n_vm_limits_filled": tpl.n_vm_limits_filled,
        "n_angle_limits_filled": tpl.n_angle_limits_filled, "n_unbounded_limits_filled": tpl.n_unbounded_limits, "limit_cap": tpl.cap,
        "seconds_setup": t_setup, "seconds_convert": dt, "samples_per_s": len(kept) / dt if dt > 0 else None,
        "workers": workers,
    }
    (out / case / "convert_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"[{case}] kept {len(kept)}/{n} (dropped {drops}) {len(kept) / max(dt, 1e-9):.2f} samples/s "
        f"max resid {mx['resid']:.2e} dvm {mx['dvm']:.1e} dva {mx['dva']:.1e}")
    return stats


# --------------------------------------------------------------------------------------------------
# Read-back validation (independent of the in-worker check): residual from the WRITTEN parquet files
# --------------------------------------------------------------------------------------------------
def validate_written(raw: Path) -> dict:
    bus = pd.read_parquet(raw / "bus_data.parquet")
    Y = pd.read_parquet(raw / "y_bus_data.parquet")
    out = {"max_resid_pu": 0.0, "n_scenarios": int(bus.scenario.nunique())}
    ids = sorted(bus.scenario.unique())
    out["contiguous"] = ids == list(range(len(ids)))
    gb, gy = bus.groupby("scenario"), Y.groupby("scenario")
    for s in ids:
        b, y = gb.get_group(s), gy.get_group(s)
        out["max_resid_pu"] = max(out["max_resid_pu"], residual_from_tables(
            b.sort_values("bus")[BUS_COLS].to_numpy(), y[["scenario", "index1", "index2", "G", "B"]].to_numpy()))
    return out


# --------------------------------------------------------------------------------------------------
# Round trip against a published datakit raw dir
# --------------------------------------------------------------------------------------------------
def network_from_published(b: pd.DataFrame, g: pd.DataFrame, e: pd.DataFrame) -> Network:
    """Rebuild a datakit Network for ONE published scenario from its own tables."""
    b, g, e = b.sort_values("bus"), g.sort_values("idx"), e.sort_values("idx")
    nb = len(b)
    bus = np.zeros((nb, 13))
    bus[:, BUS_I] = b.bus.values
    bus[:, BUS_TYPE] = 1 * (b.PQ.values == 1) + 2 * (b.PV.values == 1) + 3 * (b.REF.values == 1)
    bus[:, PD], bus[:, QD] = b.Pd.values, b.Qd.values
    bus[:, GS], bus[:, BS] = b.GS.values * BASE_MVA, b.BS.values * BASE_MVA
    bus[:, BUS_AREA] = 1
    bus[:, ZONE] = 1
    bus[:, VM], bus[:, VA_COL] = b.Vm.values, b.Va.values
    bus[:, BASE_KV], bus[:, VMIN], bus[:, VMAX] = b.vn_kv.values, b.min_vm_pu.values, b.max_vm_pu.values
    gen = np.zeros((len(g), 21))
    gen[:, GEN_BUS], gen[:, PG], gen[:, QG] = g.bus.values, g.p_mw.values, g.q_mvar.values
    gen[:, QMAX], gen[:, QMIN] = g.max_q_mvar.values, g.min_q_mvar.values
    gen[:, MBASE], gen[:, GEN_STATUS] = BASE_MVA, g.in_service.values
    gen[:, PMAX], gen[:, PMIN] = g.max_p_mw.values, g.min_p_mw.values
    cost = np.zeros((len(g), 7))
    cost[:, MODEL], cost[:, NCOST] = POLYNOMIAL, 3
    cost[:, COST], cost[:, COST + 1], cost[:, COST + 2] = g.cp2_eur_per_mw2.values, g.cp1_eur_per_mw.values, g.cp0_eur.values
    br = np.zeros((len(e), 13))
    br[:, F_BUS], br[:, T_BUS] = e.from_bus.values, e.to_bus.values
    br[:, BR_R], br[:, BR_X], br[:, BR_B] = e.r.values, e.x.values, e.b.values
    br[:, RATE_A], br[:, TAP], br[:, SHIFT] = e.rate_a.values, e.tap.values, e["shift"].values
    br[:, BR_STATUS], br[:, ANGMIN], br[:, ANGMAX] = e.br_status.values, e.ang_min.values, e.ang_max.values
    return Network({"baseMVA": BASE_MVA, "bus": bus, "gen": gen, "branch": br, "gencost": cost})


def roundtrip(raw: Path, n: int = 20, healthy_tol: float = TOL_RESID, scan: int = 80) -> dict:
    """Rebuild ``n`` healthy published scenarios through ``tables_from_state`` and diff every column."""
    B = pd.read_parquet(raw / "bus_data.parquet")
    G = pd.read_parquet(raw / "gen_data.parquet")
    E = pd.read_parquet(raw / "branch_data.parquet")
    Y = pd.read_parquet(raw / "y_bus_data.parquet")
    gB, gG, gE, gY = (x.groupby("scenario") for x in (B, G, E, Y))
    scen = sorted(B.scenario.unique())[:scan]
    diffs = {t: {c: 0.0 for c in cols} for t, cols in TABLES.items()}
    struct_bad, used, resid_pub, skipped = [], [], [], []
    for s in scen:
        b, g, e, y = gB.get_group(s), gG.get_group(s), gE.get_group(s), gY.get_group(s)
        r = residual_from_tables(b.sort_values("bus")[BUS_COLS].to_numpy(),
                                 y[["scenario", "index1", "index2", "G", "B"]].to_numpy())
        if r > healthy_tol:
            skipped.append((int(s), r))
            continue
        net = network_from_published(b, g, e)
        bs, gs = b.sort_values("bus"), g.sort_values("idx")
        ing = net.idx_gens_in_service
        out = tables_from_state(net, int(b.load_scenario_idx.iloc[0]), bs.Vm.values, np.deg2rad(bs.Va.values),
                                gs.p_mw.values[ing], gs.q_mvar.values[ing])
        for t, pubdf in (("bus_data", b.sort_values("bus")), ("gen_data", g.sort_values("idx")),
                         ("branch_data", e.sort_values("idx")), ("y_bus_data", y.sort_values(["index1", "index2"]))):
            mine = out[_ARR_KEY[t]]
            if t == "y_bus_data":
                mine = mine[np.lexsort((mine[:, 2], mine[:, 1]))]
            if mine.shape[0] != len(pubdf):
                struct_bad.append((int(s), t, mine.shape[0], len(pubdf)))
                continue
            for j, c in enumerate(TABLES[t]):
                d = np.abs(mine[:, j] - pubdf[c].to_numpy(dtype=float))
                if c == "Va" or c == "Va_dc":
                    d = np.abs((mine[:, j] - pubdf[c].to_numpy(dtype=float) + 180) % 360 - 180)
                diffs[t][c] = max(diffs[t][c], float(np.nanmax(d)) if d.size else 0.0)
        used.append(int(s))
        resid_pub.append(r)
        if len(used) >= n:
            break
    return {"scenarios_used": used, "n_used": len(used), "skipped_unhealthy": skipped[:10],
            "n_skipped_unhealthy": len(skipped), "structure_mismatch": struct_bad,
            "max_abs_diff": diffs, "published_resid_max": max(resid_pub) if resid_pub else None}


# --------------------------------------------------------------------------------------------------
def main(argv=None):
    global TOL_VM, TOL_VA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", default="experiments/2026-10-18-genladder")
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=200, help="samples per case (first N of a seeded random draw)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/external/genco-ladder-datakit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-defaults", action="store_true",
                    help="keep pandapower's placeholder limits (vm 0/2 pu, angle +-360) instead of 0.9/1.1 pu, +-30 deg")
    ap.add_argument("--tol-vm", type=float, default=TOL_VM, help="max |dVm| (pu) vs the stored label")
    ap.add_argument("--tol-va", type=float, default=TOL_VA, help="max |dVa| (deg) vs the stored label")
    ap.add_argument("--pandapower-gens", action="store_true",
                    help="PGLib cases: keep pandapower's gens/sgens (sgens folded into Pd) instead of the .m gen rows")
    ap.add_argument("--no-rate-fix", action="store_true",
                    help="keep pandapower's placeholder rate_a (>= 9999 MVA or <= 0) instead of PowerModels' "
                         "calc_thermal_limits! value")
    ap.add_argument("--n-map", default=None,
                    help="per-case sample counts: 'case=n,case2=n2' or a JSON file; others use --n")
    ap.add_argument("--rate-cap", type=float, default=None,
                    help="clip branch rate_a above this many MVA (default: keep the case's values)")
    ap.add_argument("--validate", action="store_true", help="re-read the written parquet and recompute residuals")
    ap.add_argument("--roundtrip", default=None, help="published raw dir: round-trip check, then exit")
    ap.add_argument("--roundtrip-n", type=int, default=20)
    a = ap.parse_args(argv)
    if a.roundtrip:
        print(json.dumps(roundtrip(Path(a.roundtrip), a.roundtrip_n), indent=1))
        return 0
    TOL_VM, TOL_VA = a.tol_vm, a.tol_va
    nmap = {}
    if a.n_map:
        nmap = (json.loads(Path(a.n_map).read_text()) if Path(a.n_map).exists()
                else {k: int(v) for k, v in (kv.split('=') for kv in a.n_map.split(','))})
    out = Path(a.out)
    failed = 0
    for c in a.cases:
        try:
            st = convert_case(c, Path(a.shards), out, nmap.get(c, a.n), a.seed, a.workers, log=lambda m: print(m, flush=True),
                              fill_defaults=not a.no_defaults, rate_cap=a.rate_cap,
                              rate_fix=not a.no_rate_fix, pglib_gens=not a.pandapower_gens)
            if a.validate:
                v = validate_written(out / c / "raw")
                print(f"[{c}] written-file validation: {v}", flush=True)
                st["written_validation"] = v
                (out / c / "convert_stats.json").write_text(json.dumps(st, indent=1))
            if st["kept"] < nmap.get(c, a.n):
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{c}] FAILED: {type(e).__name__}: {e}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
