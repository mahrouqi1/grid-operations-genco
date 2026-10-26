"""genco_challenge.hybrid: right-root reference when a power-flow label is a collapsed root.

On case1354_pegase / case240_pserc / case1888_rte / case1951_rte, datakit's PF labels carry Vm ~ 0 (or
negative) at zero-injection buses. Such a point satisfies the power-flow equations exactly (the
identity check passes) but is not an operating point; scoring warm starts against it called every
start that reached the physical root "wrong". These tests pin the v4 behaviour on a 3-bus toy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/genco-challenge/src"))
from genco_challenge import hybrid as H  # noqa: E402
from grid_ops.reference_pf import newton_raphson_pf  # noqa: E402

Z01, Z12 = 0.01 + 0.1j, 0.01 + 0.05j
LOAD = 0.8 + 0.3j  # pu at bus 1; bus 2 is a zero-injection leaf hanging off bus 1


def _ybus(isolate_2: bool = False) -> np.ndarray:
    y01, y12 = 1 / Z01, (0.0 if isolate_2 else 1 / Z12)
    y = np.array([[y01, -y01, 0], [-y01, y01 + y12, -y12], [0, -y12, y12]], complex)
    if isolate_2:
        y[2, 2] = 0.1j  # a shunt only, no branch: nothing to repair from
    return y


def _scenario(v_label: np.ndarray, ybus: np.ndarray) -> H.Scenario:
    n = 3
    b = pd.DataFrame({"bus": range(n), "Pg": [0.0] * n, "Qg": [0.0] * n,
                      "Pd": [0.0, LOAD.real * 100, 0.0], "Qd": [0.0, LOAD.imag * 100, 0.0],
                      "PQ": [0, 1, 1], "PV": [0, 0, 0], "REF": [1, 0, 0],
                      "Vm": np.abs(v_label), "Va": np.degrees(np.angle(v_label)), "Va_dc": [0.0] * n,
                      "min_vm_pu": [0.9] * n, "max_vm_pu": [1.1] * n})
    c = sp.coo_matrix(ybus)
    y = pd.DataFrame({"index1": c.row, "index2": c.col, "G": c.data.real, "B": c.data.imag})
    br = pd.DataFrame({"from_bus": [0, 1], "to_bus": [1, 2], "x": [Z01.imag, Z12.imag], "br_status": [1, 1]})
    return H.Scenario(b, y, br)


def _physical_root() -> np.ndarray:
    y = _ybus()
    r = newton_raphson_pf(sp.csr_matrix(y), np.array([0, -LOAD, 0]), np.ones(3, complex),
                          np.array([], int), np.array([1, 2]), tol=1e-12, max_it=30)
    assert r.converged
    return r.v


def _collapsed_root() -> np.ndarray:
    """V2 = 0 at the leaf: bus 2's equation holds trivially, bus 1 sees the leaf branch as a short."""
    y1 = 1 / Z01 + 1 / Z12  # bus 1 self-admittance with V2 = 0
    y = np.array([[1 / Z01, -1 / Z01], [-1 / Z01, y1]])
    r = newton_raphson_pf(sp.csr_matrix(y), np.array([0, -LOAD]), np.ones(2, complex),
                          np.array([], int), np.array([1]), tol=1e-12, max_it=50)
    assert r.converged
    return np.r_[r.v, 0.0]


def test_collapsed_label_is_a_genuine_root_that_the_identity_check_accepts():
    sc = _scenario(_collapsed_root(), _ybus())
    assert H.maxmis(sc, sc.v_label) < 1e-9
    assert sc.vm_label.min() < H.LABEL_VM_MIN


def test_operating_label_is_its_own_reference():
    v = _physical_root()
    sc = _scenario(v, _ybus())
    assert sc.ref_kind == "label" and np.array_equal(sc.v_ref, sc.v_label)
    r = H.solve(sc, H.starts_for(sc, None)["flat"], "nr")
    assert r["outcome"] == "right_root" and r["max_dv"] == r["max_dv_label"]


def test_collapsed_label_is_replaced_by_the_repaired_physical_root():
    sc = _scenario(_collapsed_root(), _ybus())
    assert sc.ref_kind == "repaired"
    assert np.abs(sc.v_ref - _physical_root()).max() < 1e-8
    r = H.solve(sc, H.starts_for(sc, None)["flat"], "nr")
    # v3 called this a wrong root (max |dV| vs the label ~ 1 pu); it is the physical root
    assert r["max_dv_label"] > 0.5
    assert r["outcome"] == "right_root"


def test_no_physical_root_reachable_means_no_reference():
    v = _collapsed_root()
    sc = _scenario(v, _ybus(isolate_2=True))
    assert sc.ref_kind == "none" and sc.v_ref is None
    w = pd.DataFrame([{"scenario": 0, "model": "-", "start": "flat", "method": "nr"}
                      | H.solve(sc, H.starts_for(sc, None)["flat"], "nr")])
    a = H.agg_warm(w)[0]
    assert a["n_ref"] == 0 and np.isnan(a["right_root_frac"])
    row = H.score(sc, H.starts_for(sc, None)["flat"])
    assert sum(row[f"n_{t}"] for t in H.TYPES) == 0
