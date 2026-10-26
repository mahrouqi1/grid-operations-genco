"""Small regression tests for the genco-eval audit R1 fixes (F4 aggregation, F5 resume stamp + join)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/genco-challenge/src"))
from genco_challenge import hybrid as H  # noqa: E402
from genco_challenge import ladder_eval as E  # noqa: E402


def test_agg_hybrid_counts_diverged_and_matches_mae_rows():
    """F4: a diverged (NaN) scenario is counted in nonfinite and leaves the MAE denominator."""
    rows = []
    for i, ok in enumerate([True, True, False]):
        r = {"start": "vmdc", "method": "nr", "k": 1, "finite": ok, "mis_max": 1.0 if ok else np.nan,
             "mis_mean": 1.0 if ok else np.nan, "p_max": 1.0, "q_max": 1.0}
        for t in H.TYPES:
            r[f"n_{t}"] = 2
            r[f"vm_sum_{t}"] = 0.2 if ok else np.nan
            r[f"va_sum_{t}"] = 2.0 if ok else np.nan
        rows.append(r)
    out = H.agg_hybrid(pd.DataFrame(rows))[0]
    assert out["n"] == 3 and out["nonfinite"] == 1
    assert out["vm_mae_pu"] == pytest.approx(0.1) and out["va_mae_deg"] == pytest.approx(1.0)  # not 2/3 of that


def test_data_stamp_changes_on_revalidation_and_join_check(tmp_path, monkeypatch):
    """F5: a re-validated grid gets a new stamp (so its old row is not 'done'); join_check rejects stale targets."""
    monkeypatch.setattr(E, "WORK", tmp_path)
    g = tmp_path / "data" / "g"
    (g / "raw").mkdir(parents=True)
    meta = {"scenarios_written": 10, "validation": {"when": "2026-10-07 10:00:00"}}
    (g / "meta.json").write_text(json.dumps(meta))
    s1 = E.data_stamp("g")
    meta["scenarios_written"], meta["validation"]["when"] = 8, "2026-10-07 11:00:00"
    (g / "meta.json").write_text(json.dumps(meta))
    assert E.data_stamp("g") != s1

    bus = pd.DataFrame({"scenario": [0, 0], "bus": [0, 1], "Vm": [1.0, 1.0], "Pd": [0.0, 50.0]})
    pred = bus.rename(columns={"Vm": "Vm_target"}).drop(columns=[]).assign(Pd=bus["Pd"])
    E.join_check(pred, bus)
    with pytest.raises(RuntimeError):
        E.join_check(pred.assign(Vm_target=[1.0, 0.5]), bus)
