"""genco_challenge.ladder_data label validation on a fake 3-bus grid (audit genco-eval R1-F7, R1-F8).

F7: a NaN Vm label must be dropped (it never trips ``Vm < VM_MIN``), and an all-NaN scenario must not
raise KeyError in the diagnostics. F8: on a second (``force``) validation, ``meta.json``'s
``dropped_scenarios[].orig_scenario`` must hold the id in the ORIGINAL raw/, matching the chained CSV.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")  # parquet tables; present in the grid-operations-genco env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/genco-challenge/src"))
import genco_challenge.ladder_data as ld  # noqa: E402

N, NB = 6, 3


def _write(raw: Path, vm: list[list[float]]) -> None:
    rows = [dict(scenario=s, bus=b, Vm=vm[s][b], Va=0.0, PV=int(b == 1), REF=int(b == 0), PQ=int(b == 2),
                 load_scenario_idx=s, scenario_partition=0) for s in range(N) for b in range(NB)]
    pd.DataFrame(rows).to_parquet(raw / "bus_data.parquet", partition_cols=["scenario_partition"])
    gen = pd.DataFrame([dict(scenario=s, bus=b, q_mvar=0.0, min_q_mvar=-1.0, max_q_mvar=1.0, in_service=1,
                             scenario_partition=0) for s in range(N) for b in (0, 1)])
    gen.to_parquet(raw / "gen_data.parquet", partition_cols=["scenario_partition"])
    for t in ld.SCEN_TABLES:
        if t not in ("bus_data", "gen_data"):
            pd.DataFrame(dict(scenario=list(range(N)), x=1.0, scenario_partition=0)).to_parquet(
                raw / f"{t}.parquet", partition_cols=["scenario_partition"])


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "WORK", tmp_path)
    raw = tmp_path / "data" / "g" / "raw"
    raw.mkdir(parents=True)
    (raw.parent / "meta.json").write_text(json.dumps({"name": "g", "status": "ok"}))
    return raw


def test_nan_label_is_dropped(fake):
    vm = [[1.0, 1.0, 1.0] for _ in range(N)]
    vm[1][2] = -0.1  # collapsed
    vm[3][2] = np.nan  # NaN label
    _write(fake, vm)
    m = ld.validate_grid("g", oracle_k=0)
    assert m["validation"]["dropped"] == 2 and m["validation"]["kept"] == 4
    assert m["validation"]["reasons"] == {"negative_vm": 1, "nonfinite": 1}
    assert pd.read_parquet(fake / "bus_data.parquet")["Vm"].notna().all()


def test_all_nan_scenario_does_not_raise():
    bus = pd.DataFrame(dict(scenario=[0, 0, 1, 1], bus=[0, 1, 0, 1], Vm=[np.nan, np.nan, 1.0, 1.0],
                            PV=[0, 0, 0, 0], REF=[1, 0, 1, 0]))
    d = ld.label_diagnostics(bus)
    assert bool(d.loc[0, "invalid"]) and d.loc[0, "reason"] == "nonfinite"
    assert not bool(d.loc[1, "invalid"])


def test_second_validation_records_original_ids(fake):
    vm = [[1.0, 1.0, 1.0] for _ in range(N)]
    vm[1][2] = -0.1
    _write(fake, vm)
    ld.validate_grid("g", oracle_k=0)  # drops orig 1; kept orig [0,2,3,4,5] -> new [0..4]
    b = pd.read_parquet(fake / "bus_data.parquet").drop(columns=["scenario_partition"])
    b.loc[(b.scenario == 1) & (b.bus == 2), "Vm"] = -0.2  # new id 1 == orig 2
    time.sleep(1.1)  # backup dirs are stamped to the second
    shutil.rmtree(fake / "bus_data.parquet")
    b["scenario_partition"] = 0
    b.to_parquet(fake / "bus_data.parquet", partition_cols=["scenario_partition"])
    m = ld.validate_grid("g", force=True, oracle_k=0)
    assert [d["orig_scenario"] for d in m["validation"]["dropped_scenarios"]] == [2]
    smap = pd.read_csv(fake / "scenario_map.csv")
    assert smap["orig_scenario"].tolist() == [0, 3, 4, 5]
    dropped = pd.read_csv(fake / "dropped_scenarios.csv")
    assert dropped.columns[0] == "orig_scenario" and dropped["orig_scenario"].tolist() == [1, 2]


def test_negative_vm_sign_flip_is_diagnosed_separately(fake):
    """Audit genco-eval R2-F1: a negative Vm with |Vm| >= VM_MIN is a sign-flipped normal point, not a
    collapsed root. It is still dropped (unusable regression target); only the reason differs."""
    bus = pd.DataFrame(dict(scenario=[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3, bus=[0, 1, 2] * 4,
                            Vm=[1.0, 1.0, 1.0,  1.0, 1.0, -1.07,  1.0, 1.0, -0.1,  1.0, 0.3, 1.0],
                            PV=0, REF=[1, 0, 0] * 4))
    d = ld.label_diagnostics(bus)
    assert list(d["reason"]) == ["", "negative_vm_sign_flip", "negative_vm", f"min_vm_below_{ld.VM_MIN}"]
    assert list(d["invalid"]) == [False, True, True, True]
    assert list(d["n_collapsed"]) == [0, 0, 1, 1]
    assert d.loc[1, "min_abs_vm"] == pytest.approx(1.0) and d.loc[2, "min_abs_vm"] == pytest.approx(0.1)
    # validation behavior: the sign-flip scenario is still dropped
    vm = [[1.0, 1.0, 1.0] for _ in range(N)]
    vm[2][2] = -1.07
    _write(fake, vm)
    m = ld.validate_grid("g", oracle_k=0)
    assert m["validation"]["dropped"] == 1 and m["validation"]["reasons"] == {"negative_vm_sign_flip": 1}


def test_low_bus_stats_reports_abs_vm():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/genco-challenge/results/2026-10-07-label-validation"))
    import genco_data_check as gdc
    bus = pd.DataFrame(dict(scenario=[0, 0, 1, 1], Vm=[-1.07, 1.0, 0.2, -0.3], PQ=1, PV=0, REF=0, Pd=5.0))
    s = gdc.low_bus_stats(bus)
    assert s["low_buses"] == 3 and s["low_buses_negative"] == 2 and s["low_buses_abs_vm_ge_vmmin"] == 1
    assert s["low_bus_abs_vm"]["min"] == pytest.approx(0.2) and s["low_bus_abs_vm"]["max"] == pytest.approx(1.07)
