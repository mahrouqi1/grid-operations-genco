"""Additive shift-aware DC baseline (``dc_shift``) in ladder_eval.errors and hybrid.starts_for.

Defaults must not change: without a ``Va_dc_shift`` column / ``va_dc_shift`` attribute the outputs carry no
shift entry. 2-bus toy: bus 0 REF, bus 1 PQ with load 0.5 pu, one line x = 0.1 pu with a 10 degree phase shifter.
DC (MATPOWER makeBdc): Pf = (th_f - th_t - shift)/x, so th_1 = th_0 - shift - 0.5*x (radians).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/genco-challenge/src"))
sys.path.insert(0, str(ROOT / "core"))
from genco_challenge import hybrid, ladder_eval  # noqa: E402
from genco_challenge.relabel import shift_dc_angles  # noqa: E402

SHIFT_DEG, X, LOAD = 10.0, 0.1, 0.5


def _scenario():
    y = 1.0 / (0.01 + X * 1j)
    b = pd.DataFrame({"bus": [0, 1], "Pg": [50.0, 0.0], "Pd": [0.0, 50.0], "Qg": [0.0, 0.0], "Qd": [0.0, 10.0],
                      "PQ": [0, 1], "PV": [0, 0], "REF": [1, 0], "Vm": [1.0, 0.98], "Va": [0.0, -12.0],
                      "Va_dc": [0.0, -2.9], "min_vm_pu": [0.9, 0.9], "max_vm_pu": [1.1, 1.1]})
    yb = pd.DataFrame({"index1": [0, 0, 1, 1], "index2": [0, 1, 0, 1],
                       "G": [y.real, -y.real, -y.real, y.real], "B": [y.imag, -y.imag, -y.imag, y.imag]})
    br = pd.DataFrame({"from_bus": [0], "to_bus": [1], "x": [X], "br_status": [1], "tap": [0.0], "shift": [SHIFT_DEG]})
    return hybrid.Scenario(b, yb, br), br


def test_shift_dc_angles_match_hand_value_and_differ_from_unshifted():
    sc, br = _scenario()
    th = shift_dc_angles(sc, br)
    assert th[0] == pytest.approx(0.0)
    assert th[1] == pytest.approx(-np.radians(SHIFT_DEG) - LOAD * X, abs=1e-12)
    th0 = shift_dc_angles(sc, br.assign(shift=0.0))
    assert th0[1] == pytest.approx(-LOAD * X, abs=1e-12)


def test_starts_for_adds_shift_starts_only_when_attached():
    sc, br = _scenario()
    pred = pd.DataFrame({"Vm_pred": [1.0, 0.97], "Va_pred": [0.0, -0.2]})
    base = hybrid.starts_for(sc, pred)
    assert "dc_shift" not in base and "vmdc_shift" not in base
    sc.va_dc_shift = shift_dc_angles(sc, br)
    st = hybrid.starts_for(sc, pred)
    assert set(st) - set(base) == {"dc_shift", "vmdc_shift"}
    for k in base:                       # every original start is untouched
        assert np.array_equal(st[k], base[k])
    assert np.allclose(np.angle(st["dc_shift"]), sc.va_dc_shift)
    assert np.allclose(np.abs(st["vmdc_shift"]), [1.0, 0.97]) and np.allclose(np.angle(st["vmdc_shift"]), sc.va_dc_shift)
    assert set(hybrid.starts_for(sc, None)) == {"flat", "dc", "dc_shift"}


def _pred_bus():
    pred = pd.DataFrame({"scenario": [0, 0], "bus": [0, 1], "Vm_pred": [1.0, 0.97], "Va_pred": [0.0, -0.2],
                         "Vm_target": [1.0, 0.98], "Va_target": [0.0, -0.21], "REF": [1, 0], "PV": [0, 0], "PQ": [0, 1]})
    bus = pd.DataFrame({"scenario": [0, 0], "bus": [0, 1], "Va_dc": [0.0, -2.9]})
    branch = pd.DataFrame({"scenario": [0], "from_bus": [0], "to_bus": [1], "br_status": [1]})
    return pred, bus, branch


def test_errors_default_has_no_dc_shift_and_is_unchanged_by_the_option():
    pred, bus, branch = _pred_bus()
    base = ladder_eval.errors(pred, bus, branch)
    assert "dc_shift" not in base
    th_deg = -SHIFT_DEG - np.degrees(LOAD * X)
    withs = ladder_eval.errors(pred, bus.assign(Va_dc_shift=[0.0, th_deg]), branch)
    assert set(withs) - set(base) == {"dc_shift"}
    for k in base:
        assert withs[k] == base[k]
    true_d = np.degrees(-0.21)
    assert withs["dc_shift"]["branch_dva_mae_deg"] == pytest.approx(abs((0 - th_deg) - (0 - true_d)))
    assert withs["dc"]["branch_dva_mae_deg"] == pytest.approx(abs((0 + 2.9) - (0 - true_d)))
