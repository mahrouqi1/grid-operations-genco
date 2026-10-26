"""Acceptance rule for relabeling collapsed datakit labels (genco_challenge.relabel.accept_relabel).

A new label needs TWO different starts at physical roots: power-mismatch < res_tol, every |V| >= 0.5,
and max |dV| < agree_tol between them. 3-bus toy network, bus 0 REF, bus 1 PV, bus 2 PQ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")  # relabel imports ladder_data/hybrid (genco env)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/genco-challenge/src"))
sys.path.insert(0, str(ROOT / "core"))
from genco_challenge.relabel import accept_relabel, mismatch_max  # noqa: E402
from grid_ops.reference_pf import newton_raphson_pf  # noqa: E402

PV, PQ = np.array([1]), np.array([2])


def _problem():
    y = 1.0 / (0.01 + 0.1j)
    Y = np.array([[2 * y, -y, -y], [-y, 2 * y, -y], [-y, -y, 2 * y]])
    S = np.array([0, 0.2, -0.5 - 0.1j])
    v0 = np.array([1.0, 1.02, 1.0], dtype=complex)
    r = newton_raphson_pf(Y, S, v0, PV, PQ, tol=1e-12, max_it=30)
    assert r.converged
    return Y, S, r.v


def test_two_agreeing_physical_roots_are_accepted_and_label_is_first_start():
    Y, S, v = _problem()
    assert mismatch_max(Y, S, v, PV, PQ) < 1e-9
    out = accept_relabel([("flat", v), ("dc", v * (1 + 1e-9))], Y, S, PV, PQ, res_tol=1e-6)
    assert out["accepted"] and out["pair"] == ["flat", "dc"] and np.allclose(out["v"], v)


def test_single_physical_root_is_rejected():
    Y, S, v = _problem()
    out = accept_relabel([("flat", v), ("dc", None)], Y, S, PV, PQ)
    assert not out["accepted"] and out["reason"] == "one_physical_root"


def test_no_convergence_is_rejected():
    Y, S, v = _problem()
    out = accept_relabel([("flat", v * 1.3), ("dc", None)], Y, S, PV, PQ)
    assert not out["accepted"] and out["reason"] == "no_start_converged"


def test_roots_that_disagree_are_rejected():
    Y, S, v = _problem()
    bumped = v.copy()
    bumped[2] *= 1 + 1e-3   # loose res_tol so both count as roots; they differ by ~1e-3 pu
    out = accept_relabel([("flat", v), ("dc", bumped)], Y, S, PV, PQ, res_tol=1e-1)
    assert not out["accepted"] and out["reason"] == "physical_roots_disagree"
    assert out["max_dv"] > 1e-6


def test_collapsed_roots_are_not_physical_even_if_they_agree():
    Y, S, v = _problem()
    low = v.copy()
    low[2] = 0.3 * np.exp(1j * np.angle(v[2]))       # |V| < 0.5 at a PQ bus
    out = accept_relabel([("flat", low), ("dc", low)], Y, S, PV, PQ, res_tol=1e9)
    assert not out["accepted"] and out["reason"] == "collapsed_roots_only"
    assert all(c["why"] == "collapsed" for c in out["candidates"])


def test_first_agreeing_pair_wins_when_a_third_start_is_wrong():
    Y, S, v = _problem()
    out = accept_relabel([("flat", None), ("dc", v), ("damped_flat", v), ("repaired_label", v * 2)],
                         Y, S, PV, PQ, res_tol=1e-6)
    assert out["accepted"] and out["pair"] == ["dc", "damped_flat"]


def test_plain_and_damped_newton_from_one_initial_point_are_not_two_starts():
    Y, S, v = _problem()
    out = accept_relabel([("dc_shift", v), ("damped_dc_shift", v)], Y, S, PV, PQ, res_tol=1e-6)
    assert not out["accepted"]   # same initial point, plain vs damped iteration: one start, not two
    out2 = accept_relabel([("dc_shift", v), ("damped_dc_shift", v), ("neighbor3", v)], Y, S, PV, PQ, res_tol=1e-6)
    assert out2["accepted"] and out2["pair"] == ["dc_shift", "neighbor3"]


def test_root_with_implausible_branch_angle_is_not_physical():
    Y, S, v = _problem()
    far = lambda vv: 150.0   # noqa: E731  stand-in for "largest branch angle difference"
    out = accept_relabel([("flat", v), ("dc", v)], Y, S, PV, PQ, res_tol=1e-6, ang_fn=far)
    assert not out["accepted"] and out["reason"] == "implausible_angle_roots_only"
    ok = accept_relabel([("flat", v), ("dc", v)], Y, S, PV, PQ, res_tol=1e-6, ang_fn=lambda vv: 20.0)
    assert ok["accepted"] and ok["candidates"][0]["max_branch_angle_deg"] == 20.0


# ---------------------------------------------------------------------------- write-back (audit R1-F4)

import pandas as pd  # noqa: E402
from genco_challenge import relabel as rl  # noqa: E402
from genco_challenge.hybrid import Scenario  # noqa: E402

_LINES = [(0, 1), (1, 2), (0, 2)]
_Y = 1.0 / (0.01 + 0.1j)


def _tables():
    """3-bus triangle, bus 0 REF (1.0 pu, 0 deg), bus 1 PV (1.02 pu, Pg 20 MW), bus 2 PQ (50 MW + 10 MVAr load).
    The branch table holds the same admittances the Ybus is built from. Starting labels are deliberately wrong."""
    Y = np.zeros((3, 3), dtype=complex)
    rows = []
    for k, (f, t) in enumerate(_LINES):
        Y[f, f] += _Y; Y[t, t] += _Y; Y[f, t] -= _Y; Y[t, f] -= _Y
        rows.append({"idx": k, "from_bus": f, "to_bus": t, "br_status": 1, "x": 0.1, "tap": 0.0, "shift": 0.0,
                     "Yff_r": _Y.real, "Yff_i": _Y.imag, "Yft_r": -_Y.real, "Yft_i": -_Y.imag,
                     "Ytf_r": -_Y.real, "Ytf_i": -_Y.imag, "Ytt_r": _Y.real, "Ytt_i": _Y.imag,
                     "pf": 9.0, "qf": 9.0, "pt": 9.0, "qt": 9.0})
    br = pd.DataFrame(rows)
    bus = pd.DataFrame({"bus": [0, 1, 2], "Pd": [0.0, 0.0, 50.0], "Qd": [0.0, 0.0, 10.0],
                        "PQ": [0, 0, 1], "PV": [0, 1, 0], "REF": [1, 0, 0], "GS": [0.0] * 3, "BS": [0.0] * 3,
                        "min_vm_pu": [0.9] * 3, "max_vm_pu": [1.1] * 3, "Va_dc": [0.0, -1.0, -2.0],
                        "Pg_dc": [30.0, 20.0, 0.0], "Vm": [1.0, 1.02, 0.4], "Va": [0.0, -5.0, -9.0],
                        "Pg": [31.0, 20.0, 0.0], "Qg": [1.0, 2.0, 0.0]})
    gen = pd.DataFrame({"bus": [0, 0, 1], "in_service": [1, 1, 1], "is_slack_gen": [1, 0, 0],
                        "p_mw": [20.0, 11.0, 20.0], "q_mvar": [0.5, 0.5, 2.0],
                        "max_q_mvar": [99.0] * 3, "min_q_mvar": [-99.0] * 3})
    yb = pd.DataFrame({"index1": np.repeat(np.arange(3), 3), "index2": np.tile(np.arange(3), 3),
                       "G": Y.real.ravel(), "B": Y.imag.ravel()})
    return bus, gen, br, yb


def _solved():
    bus, gen, br, yb = _tables()
    sc = Scenario(bus, yb, br)
    out = rl.solve_scenario(sc, br)
    assert out["accepted"], out["reason"]
    return sc, bus, gen, br, out["v"]


def test_writeback_keeps_specified_quantities_and_is_consistent():
    sc, bus, gen, br, v = _solved()
    b2, g2, br2, dg = rl.new_label_tables(sc, v, bus, gen, br)
    cc = rl.check_consistency(sc, bus, b2, g2)
    assert cc["ok"], cc
    # bus types and specified quantities unchanged
    for c in ("PQ", "PV", "REF", "Pd", "Qd", "GS", "BS"):
        assert np.array_equal(b2[c].to_numpy(), bus[c].to_numpy())
    assert b2.loc[[0, 1], "Vm"].tolist() == pytest.approx([1.0, 1.02], abs=1e-12)   # Vm at REF and PV
    assert b2.at[0, "Va"] == pytest.approx(0.0, abs=1e-9)                           # Va at REF
    assert b2.at[1, "Pg"] == 20.0 and b2.at[2, "Pg"] == 0.0 and b2.at[2, "Qg"] == 0.0   # P at PV, P and Q at PQ
    assert b2.at[2, "Vm"] > 0.5                                                      # the bad label is replaced
    # slack P and PV/REF Q follow from the root: bus injection = V conj(Y V)
    s = v * np.conj(sc.ybus @ v) * 100.0
    assert b2.at[0, "Pg"] == pytest.approx(s.real[0] + bus.at[0, "Pd"], abs=1e-9)
    assert b2.at[0, "Qg"] == pytest.approx(s.imag[0], abs=1e-9)
    assert b2.at[1, "Qg"] == pytest.approx(s.imag[1], abs=1e-9)
    # generator rows add up to the bus values; the non-slack generator at the REF bus keeps its value
    assert g2["p_mw"].tolist()[1] == 11.0 and g2["q_mvar"].tolist()[1] == 0.5
    assert g2.loc[g2.bus == 0, "p_mw"].sum() == pytest.approx(b2.at[0, "Pg"], abs=1e-9)
    assert g2.loc[g2.bus == 0, "q_mvar"].sum() == pytest.approx(b2.at[0, "Qg"], abs=1e-9)
    assert g2.at[2, "q_mvar"] == pytest.approx(b2.at[1, "Qg"], abs=1e-9) and g2.at[2, "p_mw"] == 20.0
    # power balance at every bus from the written branch flows (no shunts here)
    inj = np.zeros(3, dtype=complex)
    for r in br2.itertuples():
        inj[r.from_bus] += r.pf + 1j * r.qf
        inj[r.to_bus] += r.pt + 1j * r.qt
    net = (b2["Pg"] - b2["Pd"]).to_numpy() + 1j * (b2["Qg"] - b2["Qd"]).to_numpy()
    assert np.abs(inj - net).max() < 1e-7


def test_check_consistency_flags_a_changed_specified_quantity():
    sc, bus, gen, br, v = _solved()
    b2, g2, _, _ = rl.new_label_tables(sc, v, bus, gen, br)
    b2.loc[1, "Vm"] += 1e-6          # PV setpoint moved
    assert not rl.check_consistency(sc, bus, b2, g2)["ok"]
    b3 = b2.copy(); b3.loc[2, "Pd"] += 1.0
    with pytest.raises(AssertionError):
        rl.check_consistency(sc, bus, b3, g2)


def test_shift_dc_angles_three_bus_hand_value():
    """Line 0-1 (x 0.1, 10 deg shifter), 1-2 (x 0.2, tap 2 -> effective x 0.4); bus 0 REF; bus 2 draws 0.3 pu, bus 1 none.
    Branch 1-2 carries 0.3 pu: th1 - th2 = 0.3*0.4 = 0.12. Branch 0-1 carries 0.3: (th0 - th1 - shift)/0.1 = 0.3,
    so th1 = -shift - 0.03 and th2 = th1 - 0.12 (radians)."""
    bus, _, br, yb = _tables()
    bus = bus.assign(Pd=[0.0, 0.0, 30.0], Pg=[30.0, 0.0, 0.0], PV=[0, 0, 0], PQ=[0, 1, 1])
    br = br.iloc[:2].copy()
    br["from_bus"], br["to_bus"] = [0, 1], [1, 2]
    br["x"], br["tap"], br["shift"] = [0.1, 0.2], [0.0, 2.0], [10.0, 0.0]
    sc = Scenario(bus, yb, br)
    th = rl.shift_dc_angles(sc, br)
    sh = np.radians(10.0)
    assert th == pytest.approx([0.0, -sh - 0.03, -sh - 0.15], abs=1e-12)
    th_ns = rl.shift_dc_angles(sc, br.assign(shift=0.0))
    assert th_ns == pytest.approx([0.0, -0.03, -0.15], abs=1e-12)
    assert th[1] - th_ns[1] == pytest.approx(-sh, abs=1e-12)


def test_relabel_grid_moves_stale_processed_aside(tmp_path, monkeypatch):
    """Rewriting raw/ renames processed/ to processed.stale-<stamp> (nothing deleted)."""
    bus, gen, br, yb = _tables()
    n = 2
    def rep(df): return pd.concat([df.assign(scenario=s) for s in range(n)], ignore_index=True)
    data = tmp_path / "data" / "g"
    src = data / "raw.unvalidated-1"
    src.mkdir(parents=True)
    bus2 = bus.assign(load_scenario_idx=0, Vm=[1.0, 1.02, 1.0], Va=[0.0, -1.0, -2.0])
    rep(bus2).to_parquet(src / "bus_data.parquet"); rep(gen).to_parquet(src / "gen_data.parquet")
    rep(br).to_parquet(src / "branch_data.parquet"); rep(yb).to_parquet(src / "y_bus_data.parquet")
    pd.DataFrame({"scenario": range(n), "x": 0.0}).to_parquet(src / "runtime_data.parquet")
    (data / "meta.json").write_text("{}")
    root = tmp_path / "out"
    (root / "g" / "processed").mkdir(parents=True)
    (root / "g" / "processed" / "marker.txt").write_text("old")
    (root / "g" / "raw").mkdir()
    summ = rl.relabel_grid("g", root=root, data_root=tmp_path / "data")
    assert not (root / "g" / "processed").exists()
    stale = list((root / "g").glob("processed.stale-*"))
    assert len(stale) == 1 and (stale[0] / "marker.txt").read_text() == "old"
    assert list((root / "g").glob("raw.superseded-*")) and (root / "g" / "raw" / "scenario_map.csv").exists()
    # F1: provenance stamped in the summary, meta.json and the sidecar
    assert set(summ["provenance"]) >= {"git_sha", "code_dirty", "dirty_modules"}
    assert json.loads((root / "g" / "meta.json").read_text())["relabel"]["provenance"] == summ["provenance"]
    assert json.loads((root / "g" / "provenance.json").read_text())["git_sha"] == summ["provenance"]["git_sha"]
