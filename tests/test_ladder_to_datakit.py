"""ladder_to_datakit: shard -> gridfm-datakit raw tables (n = 5 per case).

case14 (fast) and case118 (slack reference angle 30 degrees, so the Va-relative-to-REF convention is exercised).
The shard is the lab's own (experiments/2026-10-18-genladder/<case>.npz) when present; otherwise the first 8 samples of
the same shards, committed as JSON under tests/fixtures/ladder_shards (set GENCO_HIDE_PRIVATE_DATA=1 to force that path).
The 2-scenario round trip against a PUBLISHED subgrid needs that private data and is skipped without it; a round trip of
the converter's own case14 output (a self-consistency check of tables_from_state, not a comparison with published
tables) runs everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("gridfm_datakit")
ROOT = Path(__file__).resolve().parents[1]
for p in ("core", "tracks/genco-challenge/src", "tracks/powerflow-surrogate/src"):
    sys.path.insert(0, str(ROOT / p))
from genco_challenge import ladder_to_datakit as L  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import _datakit_fixture as FX  # noqa: E402

LAB_SHARDS = ROOT / "experiments/2026-10-18-genladder"
PUBLISHED = ROOT / "data/external/genco-pf-pretraining-eval/subnet_case4837_goc_idx_7/raw"


def _shards(case, tmp_path):
    """The lab's shard dir when it has ``case``, else the committed 8-sample slice written to ``tmp_path/shards``."""
    if FX.private_data_visible(LAB_SHARDS / f"{case}.npz"):
        return LAB_SHARDS
    return FX.materialize_shard(case, tmp_path / "shards")
REF_COLS = {
    "bus_data": ["scenario", "load_scenario_idx", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF", "vn_kv",
                 "min_vm_pu", "max_vm_pu", "GS", "BS", "Va_dc", "Pg_dc"],
    "branch_data": ["scenario", "load_scenario_idx", "idx", "from_bus", "to_bus", "pf", "qf", "pt", "qt", "r", "x", "b",
                    "Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i", "tap", "shift", "ang_min",
                    "ang_max", "rate_a", "br_status", "pf_dc", "pt_dc"],
    "gen_data": ["scenario", "load_scenario_idx", "idx", "bus", "p_mw", "q_mvar", "min_p_mw", "max_p_mw", "min_q_mvar",
                 "max_q_mvar", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2", "in_service", "is_slack_gen", "p_mw_dc"],
    "y_bus_data": ["scenario", "load_scenario_idx", "index1", "index2", "G", "B"],
}


@pytest.mark.parametrize("case", ["case14", "case118"])
def test_convert_small_case(case, tmp_path):
    st = L.convert_case(case, _shards(case, tmp_path), tmp_path, n=5, seed=0, workers=1, log=lambda m: None)
    assert st["kept"] == 5 and not st["dropped"]
    assert st["max_resid_pu"] < 1e-6 and st["max_dvm_pu"] < 1e-6 and st["max_dva_deg"] < 1e-4
    raw = tmp_path / case / "raw"
    for t, cols in REF_COLS.items():
        df = pd.read_parquet(raw / f"{t}.parquet")
        assert list(df.columns) == cols
        assert sorted(df.scenario.unique()) == list(range(5))
        assert df.scenario.dtype == np.int64
    bus = pd.read_parquet(raw / "bus_data.parquet")
    assert (bus.loc[bus.REF == 1, "Va"].abs() < 1e-9).all()                 # REF angle 0, as PowerModels fixes it
    assert (bus[["PQ", "PV", "REF"]].sum(axis=1) == 1).all()
    assert (bus.loc[bus.PQ == 1, "Pg"] == 0).all()                           # datakit's own assertion
    assert L.validate_written(raw)["max_resid_pu"] < 1e-6


def test_roundtrip_own_output(tmp_path):
    st = L.convert_case("case14", FX.materialize_shard("case14", tmp_path / "shards"), tmp_path, n=5, seed=0, workers=1, log=lambda m: None)
    assert st["kept"] == 5
    r = L.roundtrip(tmp_path / "case14" / "raw", n=3)
    assert r["n_used"] == 3 and not r["structure_mismatch"] and r["n_skipped_unhealthy"] == 0
    assert max(v for t in r["max_abs_diff"].values() for v in t.values()) < 1e-6


def test_roundtrip_published_subgrid():
    if not FX.private_data_visible(PUBLISHED):
        pytest.skip("published subgrid absent")
    r = L.roundtrip(PUBLISHED, n=2)
    assert r["n_used"] == 2 and not r["structure_mismatch"]
    worst = max(v for t in r["max_abs_diff"].values() for v in t.values())
    assert worst < 1e-6
