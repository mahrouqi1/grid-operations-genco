"""PyG ``HeteroData`` graph construction from a solved pandapower net.

Builds the graph directly from a pandapower net's own internal PYPOWER-format
``ppc`` - the same solver-ready representation :func:`reference_pf.from_pandapower`
already consumes - reusing pandapower's own computed admittance-model
parameters and solved state rather than re-deriving them (per the "read the
producer's own artifact" convention).

Node/edge TYPING and per-type feature semantics deliberately follow PF∆'s
(Rivera et al., NeurIPS'25 D&B; ``data/external/pfdelta``) schema - see
``core/datasets/pfdelta_dataset.py: PFDeltaDataset.build_heterodata`` - so a
graph built here is topology/feature-schema-comparable to PF∆'s bundled
baselines (GNS-Strict, CANOS-PF, PowerFlowNet): reuse their node/edge typing
rather than invent a new one (2026-07-31)::

    bus={x=[N,2], y=[N,2], bus_gen=[N,2], bus_demand=[N,2],
         bus_voltages=[N,2], bus_type=[N], shunt=[N,2], limits=[N,2]}
    gen={limits=[G,4], generation=[G,2], slack_gen=[G]}
    load={demand=[L,2]}
    (bus, branch, bus)={edge_index=[2,E], edge_attr=[E,8],
                         edge_label=[E,4], edge_limits=[E,1]}
    (gen, gen_link, bus) / (bus, gen_link, gen)   = {edge_index=[2,G]}
    (load, load_link, bus) / (bus, load_link, load) = {edge_index=[2,L]}

ONE deliberate deviation from PF∆: units. PF∆ (via PowerModels.jl) stores P/Q
in PER-UNIT of baseMVA and angles in RADIANS; this module keeps this repo's
existing convention throughout (``testbeds.py`` / ``reference_pf.py`` /
``baselines.py``) instead - MW/MVAr and DEGREES - since that is what every
other substrate module here already uses and validates against.
``data.base_mva`` is attached so a caller needing literal PF∆-unit
compatibility (e.g. to run a PF∆ model checkpoint on a graph built here) can
convert on the fly: ``P_pu = P_mw / base_mva``, ``angle_rad =
np.radians(angle_deg)``.

Bus-type input/target masking (the classical PF known/unknown split) matches
PF∆ exactly::

    PQ bus (type 1):    x=[Pd, Qd]          y=[Va, Vm]
    PV bus (type 2):    x=[Pg-Pd, Vm]       y=[Qg-Qd, Va]
    slack bus (type 3): x=[Va, Vm]          y=[Pg-Pd, Qg-Qd]

``branch``'s 8-column ``edge_attr`` [r, x, g_fr, b_fr, g_to, b_to, tap, shift]
mirrors PowerModels' more general pi-model split; PYPOWER's branch model
(what this module reads from) has no line-charging conductance and a single
symmetric charging susceptance, so ``g_fr = g_to = 0`` and
``b_fr = b_to = br_b / 2`` exactly - not an approximation, this is literally
how PYPOWER's own ``makeYbus`` applies the B column (half the total
susceptance at each end).

Requires ``net`` to have been solved (``pandapower.runpp(net)``) and
converged first - this module builds ground-truth LABEL graphs (mirroring
PF∆'s always-solved PowerModels input), so a non-converged solve is a hard
error rather than a silently-poisoned label (see numerical-stability.md's
"guard plausibility inside the loop" family).

Not yet handled (no registered testbed needs them today; extend here if one
does): ward/xward/motor elements, DC lines, multiple parallel transformers
with independent tap control. ``storage`` elements ARE handled, but only
implicitly: pandapower folds storage's P/Q (like ``sgen``'s) into the bus-level
net PQ injection (ppc's ``PD``/``QD`` columns) exactly like a classical MATPOWER
PQ bus, so ``bus.x``/``bus_demand`` already reflect it correctly - but storage
gets no dedicated per-element node type (unlike ``gen``/``load``), so a
downstream consumer cannot recover per-unit storage features (state of
charge, capacity). Verified against SimBench grids, the only registered
testbeds with nonzero ``net.storage`` (2026-07-31).

SimBench (switch-heavy) nets also surface a ppc quirk this module corrects
for: pandapower's internal bus array can be LARGER than ``net.bus`` - some
extra rows are genuine branch-connected auxiliary buses (switch-topology
bookkeeping; kept), others are ``BUS_TYPE==NONE`` ("isolated") rows with no
branch/gen/load reference and NaN solved VM/VA (dropped, with indices
remapped) - see the isolated-bus filter in :func:`build_hetero_data`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData


def _solved_ppc(net) -> dict:
    """The ppc dict for a solved net, or a clear error.

    Raises rather than surfacing a cryptic ``AttributeError``/``KeyError`` if
    ``pandapower.runpp(net)`` was never called or didn't converge - a graph
    built from an unsolved/diverged net would silently poison every
    downstream label.
    """
    ppc = net.get("_ppc")
    if ppc is None:
        raise ValueError(
            "net has not been solved yet; call pandapower.runpp(net) before "
            "build_hetero_data(net) -- this module builds ground-truth label "
            "graphs and needs a converged AC solution."
        )
    if not bool(ppc.get("success", False)):
        raise ValueError(
            "net's power flow did NOT converge; refusing to build a label "
            "graph from a diverged solve (would silently poison the dataset)."
        )
    return ppc


def build_hetero_data(net, *, add_bus_type: bool = False) -> HeteroData:
    """Build a PF∆-schema ``HeteroData`` graph from a solved pandapower net.

    Parameters
    ----------
    net : pandapower net
        Must already have been solved via ``pandapower.runpp(net)`` (any
        settings) and converged.
    add_bus_type : bool
        Mirrors PF∆'s own ``add_bus_type`` dataset flag exactly. When True,
        adds ``PQ``/``PV``/``slack`` node types (subsets of ``bus``, same
        ``x``/``y`` plus ``demand``/``generation`` for PV/slack) and their
        link edges to/from ``bus`` - this is what CANOS-PF consumes.

    Returns
    -------
    HeteroData
    """
    from pandapower.pypower.idx_bus import BUS_TYPE, NONE, PD, QD, GS, BS, VM, VA, VMIN, VMAX
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, PF, QF, PT, QT,
    )
    from pandapower.pypower.idx_gen import (
        GEN_BUS, PG, QG, QMAX, QMIN, PMAX, PMIN, GEN_STATUS,
    )

    ppc = _solved_ppc(net)
    base_mva = float(ppc["baseMVA"])
    bus, branch, gen = ppc["bus"], ppc["branch"], ppc["gen"]

    # ppc's internal bus array can be LARGER than net.bus: pandapower adds
    # auxiliary rows for switch-topology bookkeeping. Two kinds, verified
    # empirically on switch-heavy nets (SimBench MV/HV grids -- not seen on
    # any registered TESTBEDS/PGLib case, which are switch-free): (a) real,
    # branch-connected auxiliary buses with a genuine solved VM/VA (kept --
    # dropping them would orphan the real branches that terminate there);
    # (b) BUS_TYPE==NONE ("isolated") rows with NO branch/gen/load reference
    # at all and NaN VM/VA -- pure dead bookkeeping the solver never touched.
    # Drop only (b), or a downstream consumer silently gets NaN node features.
    keep = bus[:, BUS_TYPE] != NONE
    if not keep.all():
        old_to_new = np.full(bus.shape[0], -1, dtype=int)
        old_to_new[keep] = np.arange(int(keep.sum()))
        bus = bus[keep]

        def _remap(idx: np.ndarray) -> np.ndarray:
            new_idx = old_to_new[idx]
            if (new_idx < 0).any():
                raise ValueError(
                    "an active branch/gen/load references a bus this module "
                    "classified as isolated (BUS_TYPE==NONE, dropped) -- the "
                    "empirical assumption that isolated ppc buses are always "
                    "unreferenced does not hold for this net; investigate "
                    "before trusting the graph."
                )
            return new_idx
    else:
        def _remap(idx: np.ndarray) -> np.ndarray:
            return idx

    n_bus = bus.shape[0]

    bus_type = bus[:, BUS_TYPE].astype(int)  # 1=PQ, 2=PV, 3=REF/slack
    pd, qd = bus[:, PD], bus[:, QD]
    gs, bs = bus[:, GS], bus[:, BS]
    vm_sol, va_sol = bus[:, VM], bus[:, VA]
    vmin, vmax = bus[:, VMIN], bus[:, VMAX]

    # Aggregate in-service generation onto its bus: several generators
    # (gen/ext_grid/sgen all fold into ppc's single gen matrix) can share a
    # bus, matching PF∆'s per-bus sum-over-active-gens.
    on = gen[:, GEN_STATUS] > 0
    gbus = _remap(gen[on, GEN_BUS].astype(int))
    pg_bus = np.zeros(n_bus)
    qg_bus = np.zeros(n_bus)
    np.add.at(pg_bus, gbus, gen[on, PG])
    np.add.at(qg_bus, gbus, gen[on, QG])

    is_pq, is_pv, is_ref = bus_type == 1, bus_type == 2, bus_type == 3

    pf_x = np.zeros((n_bus, 2))
    pf_y = np.zeros((n_bus, 2))
    pf_x[is_pq] = np.c_[pd, qd][is_pq]
    pf_y[is_pq] = np.c_[va_sol, vm_sol][is_pq]
    pf_x[is_pv] = np.c_[pg_bus - pd, vm_sol][is_pv]
    pf_y[is_pv] = np.c_[qg_bus - qd, va_sol][is_pv]
    pf_x[is_ref] = np.c_[va_sol, vm_sol][is_ref]
    pf_y[is_ref] = np.c_[pg_bus - pd, qg_bus - qd][is_ref]

    data = HeteroData()
    data.base_mva = base_mva

    data["bus"].x = torch.tensor(pf_x, dtype=torch.float32)
    data["bus"].y = torch.tensor(pf_y, dtype=torch.float32)
    data["bus"].bus_gen = torch.tensor(np.c_[pg_bus, qg_bus], dtype=torch.float32)
    data["bus"].bus_demand = torch.tensor(np.c_[pd, qd], dtype=torch.float32)
    data["bus"].bus_voltages = torch.tensor(np.c_[va_sol, vm_sol], dtype=torch.float32)
    data["bus"].bus_type = torch.tensor(bus_type, dtype=torch.long)
    data["bus"].shunt = torch.tensor(np.c_[gs, bs], dtype=torch.float32)
    data["bus"].limits = torch.tensor(np.c_[vmin, vmax], dtype=torch.float32)

    # --- gen ----------------------------------------------------------
    gi = np.where(on)[0]
    n_gen = len(gi)
    data["gen"].limits = torch.tensor(
        gen[gi][:, [PMIN, PMAX, QMIN, QMAX]], dtype=torch.float32
    )
    data["gen"].generation = torch.tensor(gen[gi][:, [PG, QG]], dtype=torch.float32)
    data["gen"].slack_gen = torch.tensor(is_ref[gbus], dtype=torch.bool)

    # --- load -----------------------------------------------------------
    # ppc has no per-load array (loads are pre-aggregated into bus PD/QD),
    # so per-load nodes come from pandapower's own element table, mapped to
    # ppc-internal bus indices via its bus lookup; solved (not input) P/Q so
    # any load scaling/control is reflected, matching every other solved field.
    lookup = net._pd2ppc_lookups["bus"]
    load_df = net.load[net.load.in_service] if len(net.load) else net.load
    n_load = len(load_df)
    load_bus_ppc = (
        _remap(lookup[load_df.bus.to_numpy()]) if n_load else np.zeros(0, dtype=int)
    )
    load_pq = (
        net.res_load.loc[load_df.index, ["p_mw", "q_mvar"]].to_numpy()
        if n_load else np.zeros((0, 2))
    )
    data["load"].demand = torch.tensor(load_pq, dtype=torch.float32)

    # --- (bus, branch, bus) ---------------------------------------------
    active = branch[:, BR_STATUS] > 0
    br = branch[active]
    f_bus = _remap(br[:, F_BUS].astype(int))
    t_bus = _remap(br[:, T_BUS].astype(int))
    r, x, b_total = br[:, BR_R], br[:, BR_X], br[:, BR_B]
    # pandapower's ppc builder already normalizes the "no tap" sentinel to
    # 1.0 (verified empirically: line rows carry TAP=1.0, not the raw
    # MATPOWER 0-sentinel); the `where` guard is defensive, not load-bearing.
    tap = np.where(br[:, TAP] == 0, 1.0, br[:, TAP])
    shift = br[:, SHIFT]
    b_half = b_total / 2.0
    zeros = np.zeros_like(r)
    edge_attr = np.c_[r, x, zeros, b_half, zeros, b_half, tap, shift]

    data["bus", "branch", "bus"].edge_index = torch.tensor(
        np.c_[f_bus, t_bus].T, dtype=torch.long
    )
    data["bus", "branch", "bus"].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    data["bus", "branch", "bus"].edge_label = torch.tensor(
        br[:, [PF, QF, PT, QT]], dtype=torch.float32
    )
    data["bus", "branch", "bus"].edge_limits = torch.tensor(
        br[:, [RATE_A]], dtype=torch.float32
    )

    # --- gen_link / load_link connectivity -------------------------------
    gen_edge = torch.tensor(np.c_[np.arange(n_gen), gbus].T, dtype=torch.long)
    data["gen", "gen_link", "bus"].edge_index = gen_edge
    data["bus", "gen_link", "gen"].edge_index = gen_edge.flip(0)

    load_edge = torch.tensor(np.c_[np.arange(n_load), load_bus_ppc].T, dtype=torch.long)
    data["load", "load_link", "bus"].edge_index = load_edge
    data["bus", "load_link", "load"].edge_index = load_edge.flip(0)

    if add_bus_type:
        _add_bus_type_nodes(data, is_pq=is_pq, is_pv=is_pv, is_ref=is_ref)

    return data


def _add_bus_type_nodes(
    data: HeteroData, *, is_pq: np.ndarray, is_pv: np.ndarray, is_ref: np.ndarray
) -> None:
    """PQ/PV/slack node-type subsets of ``bus`` + their link edges.

    Mirrors ``PFDeltaCANOS``'s ``add_bus_type=True`` path: each subset carries
    the same ``x``/``y`` as its parent ``bus`` rows, and PV/slack additionally
    carry ``demand``/``generation`` (PQ's ``x`` already *is* its demand, so no
    separate field is needed there - matches PF∆).
    """
    for type_name, mask in (("PQ", is_pq), ("PV", is_pv), ("slack", is_ref)):
        idx = torch.tensor(np.where(mask)[0], dtype=torch.long)
        data[type_name].x = data["bus"].x[idx]
        data[type_name].y = data["bus"].y[idx]
        if type_name != "PQ":
            data[type_name].demand = data["bus"].bus_demand[idx]
            data[type_name].generation = data["bus"].bus_gen[idx]
        edge = torch.stack([torch.arange(len(idx)), idx])
        rel = f"{type_name}_link"
        data[type_name, rel, "bus"].edge_index = edge
        data["bus", rel, type_name].edge_index = edge.flip(0)


if __name__ == "__main__":
    import pandapower as pp

    from grid_ops.testbeds import load_case

    net = load_case("case14")
    pp.runpp(net, numba=False)
    print(build_hetero_data(net, add_bus_type=True))
