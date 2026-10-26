"""genco_challenge.hybrid resume key (audit genco-eval R2-F7).

A per_grid result is 'done' only if it was computed from the grid's current labels (ladder_eval.data_stamp)
and the current prediction files (mtimes); a result with no stamp is not done. upgrade_v3 applies the same
freshness rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")  # present in the grid-operations-genco env
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/genco-challenge/src"))
from genco_challenge import hybrid as H  # noqa: E402
from genco_challenge import ladder_eval as LE  # noqa: E402


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(LE, "WORK", tmp_path)
    monkeypatch.setattr(H, "WORK", tmp_path)
    gdir = tmp_path / "data" / "g"
    (gdir / "raw").mkdir(parents=True)
    (gdir / "meta.json").write_text(json.dumps({"scenarios_written": 5, "validation": {"when": "t0"}}))
    pred = tmp_path / "pred.parquet"
    pred.write_text("x")
    return gdir, {"m": pred}, tmp_path


def test_unstamped_result_is_not_current(world):
    gdir, preds, _ = world
    assert not H.is_current({"version": H.VERSION, "models": ["m"]}, "g", preds)


def test_stamped_result_is_current_until_labels_or_predictions_change(world):
    gdir, preds, _ = world
    prev = {"version": H.VERSION, "models": ["m"], **H.run_stamp("g", preds)}
    assert H.is_current(prev, "g", preds)
    # re-validated labels (new validation time)
    meta = json.loads((gdir / "meta.json").read_text())
    (gdir / "meta.json").write_text(json.dumps({**meta, "validation": {"when": "t1"}}))
    assert not H.is_current(prev, "g", preds)
    # restore labels, then a newer GENCO run rewrites the prediction file
    (gdir / "meta.json").write_text(json.dumps(meta))
    assert H.is_current(prev, "g", preds)
    st = preds["m"].stat()
    os.utime(preds["m"], (st.st_atime, st.st_mtime + 100))
    assert not H.is_current(prev, "g", preds)


def test_upgrade_v3_requires_fresh_rows(world):
    gdir, preds, root = world
    out = root / "out"
    (out / "per_grid").mkdir(parents=True)
    pd.DataFrame({"scenario": [0], "Vm": [1.0], "bus": [0], "Va": [0.0]}).to_parquet(gdir / "raw" / "bus_data.parquet")
    w = pd.DataFrame({"scenario": [0], "model": ["m"], "start": ["dc"], "method": ["nr"], "converged": [True],
                      "iterations": [3], "seconds": [0.0], "max_dv": [0.0], "outcome": ["right_root"]})
    wf = out / "per_grid" / "g_warm.parquet"
    w.to_parquet(wf)
    prev = {"version": 3, "models": ["m"], "warm": {"m": []}}
    base = wf.stat().st_mtime
    os.utime(gdir / "raw", (base + 50, base + 50))  # labels regenerated after the rows were written
    assert H.upgrade_v3("g", out, dict(prev), preds) is None
    os.utime(gdir / "raw", (base - 50, base - 50))
    os.utime(preds["m"], (base + 50, base + 50))  # predictions newer than the rows
    assert H.upgrade_v3("g", out, dict(prev), preds) is None
    os.utime(preds["m"], (base - 50, base - 50))
    r = H.upgrade_v3("g", out, dict(prev), preds)
    assert r is not None and r["version"] == H.VERSION and H.is_current(r, "g", preds)
