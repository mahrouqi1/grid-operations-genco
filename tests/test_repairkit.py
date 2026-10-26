"""GENCO RepairKit: check categories, repair round-trip, verify (and a deliberately broken repaired dir).

Every test runs in two modes (module parameter ``case``):

* ``private``: a few scenarios of data/external/genco-pf-pretraining/subnet_case10480_goc_idx_40 (355 buses each; three
  genuinely collapsed ones), plus two synthetic defects made from healthy scenarios: a sign defect (-|V| with the angle
  shifted by 180 degrees, the same complex voltage) and an in-service branch angle above 90 degrees. Not collected when the data
  is absent or when GENCO_HIDE_PRIVATE_DATA=1 (which simulates the clean public export without touching the data).
* ``fixture``: the committed 10-scenario case14 slice (tests/fixtures/datakit_case14, see tests/_datakit_fixture.py) with
  one collapsed |V| label, one negative-Vm label and one branch angle above 90 degrees injected. Always runs.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/genco-challenge/src"))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _datakit_fixture as FX  # noqa: E402

PRIVATE_SRC = ROOT / "data/external/genco-pf-pretraining/subnet_case10480_goc_idx_40/raw"

from genco_challenge.repairkit import check as ck  # noqa: E402
from genco_challenge.repairkit import repair as rp  # noqa: E402
from genco_challenge.repairkit import verify as vf  # noqa: E402
from genco_challenge.repairkit import _io  # noqa: E402

TABLES = _io.TABLES


@dataclass(frozen=True)
class Case:
    name: str
    collapsed: list
    healthy: list
    sign: int          # healthy scenario that gets a synthetic sign defect
    angle: int         # healthy scenario that gets a synthetic branch-angle defect
    damage: tuple      # (scenario, bus): a healthy scenario / bus whose Pd the verify test perturbs
    bad_label: tuple   # (scenario, bus): a repaired scenario / bus whose Vm the verify test perturbs
    src: Path


CASES = {
    "private": Case("private", [2029, 3852, 5836], [2030, 3851, 3853, 5835, 5837], 3851, 5837, (2030, 5), (2029, 7), PRIVATE_SRC),
    "fixture": Case("fixture", FX.FX_COLLAPSED, FX.FX_HEALTHY, FX.FX_SIGN, FX.FX_ANGLE, (0, 5), (FX.FX_COLLAPSED[0], 7), FX.FIXTURE_RAW),
}
# The private variants are not collected at all when the data is absent or hidden (no skip noise in the clean export).
PARAMS = ["fixture"] + (["private"] if FX.private_data_visible(PRIVATE_SRC / "y_bus_data.parquet") else [])


@pytest.fixture(scope="module", params=PARAMS)
def case(request) -> Case:
    return CASES[request.param]


def _rd(case, t, ids):
    return _io.read_scenarios(case.src, t, ids)


def _write(raw: Path, tabs: dict[str, pd.DataFrame], hive: bool) -> None:
    for t, df in tabs.items():
        p = raw / f"{t}.parquet"
        if hive:
            for k, g in df.groupby(df["scenario"] // 200):
                (p / f"scenario_partition={k}").mkdir(parents=True, exist_ok=True)
                g.to_parquet(p / f"scenario_partition={k}" / "part-0.parquet", index=False)
        else:
            raw.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)


@pytest.fixture(scope="module")
def tabs(case):
    ids = case.collapsed + case.healthy
    t = {n: _rd(case, n, ids) for n in TABLES}
    if case.name == "fixture":
        return FX.inject_defects(t)          # the committed slice is all healthy; the defects are made here
    b = t["bus_data"]
    m = (b.scenario == case.sign) & b.bus.isin([20, 21, 22])
    b.loc[m, "Va"] += 180.0
    b.loc[m, "Vm"] *= -1.0
    br = t["branch_data"]
    f = br[(br.scenario == case.angle) & (br.br_status == 1)].from_bus.iloc[0]
    b.loc[(b.scenario == case.angle) & (b.bus == f), "Va"] += 100.0
    return t


@pytest.fixture
def raw(tmp_path, tabs):
    d = tmp_path / "in" / "raw"
    _write(d, tabs, hive=True)
    return d


def _snapshot(d: Path):
    return sorted((str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file())


def test_check_categories_and_readonly(case, raw):
    before = _snapshot(raw)
    df = ck.check(raw, workers=2)
    assert _snapshot(raw) == before
    cat = df.set_index("scenario")["category"]
    SIGN, ANGLE = case.sign, case.angle
    assert all(cat[s] == "collapsed" for s in case.collapsed)
    assert cat[SIGN] == "negative_only"
    assert cat[ANGLE] == "angle_gt90" and bool(df.set_index("scenario").at[ANGLE, "angle_gt90"])
    plain = [s for s in case.healthy if s not in (SIGN, ANGLE)]
    assert all(cat[s] == "healthy" for s in plain)
    row = df.set_index("scenario").loc[plain[0]]
    assert row.min_pq_abs_vm > 0.5 and row.residual < 1e-7 and row.max_branch_angle_deg < 90
    assert df.set_index("scenario").loc[SIGN, "residual"] < 1e-7      # same complex voltage, residual unaffected
    s = ck.summarize(df)
    assert s["categories"] == {"healthy": len(plain), "collapsed": len(case.collapsed), "negative_only": 1, "angle_gt90": 1}


def test_check_flat_layout_and_missing_ybus(case, tmp_path, tabs):
    d = tmp_path / "flat" / "raw"
    _write(d, tabs, hive=False)
    assert not _io.is_hive(d)
    df = ck.check(d)
    assert df.set_index("scenario")["category"].to_dict()[case.collapsed[0]] == "collapsed"
    assert df["residual"].notna().all()
    (d / "y_bus_data.parquet").unlink()
    df2 = ck.check(d)
    assert df2["residual"].isna().all() and "not checked" in ck.summarize(df2)["note"]
    assert (df2["category"] == df["category"]).all()


@pytest.fixture(scope="module")
def repaired(tmp_path_factory, tabs):
    base = tmp_path_factory.mktemp("rt")
    src = base / "in" / "raw"
    _write(src, tabs, hive=True)
    out = base / "out" / "raw"
    log = rp.run(src, out, workers=2, log=lambda *_: None)
    return src, out, log


def test_repair_roundtrip(case, repaired):
    src, out, log = repaired
    COLLAPSED, HEALTHY, SIGN, ANGLE = case.collapsed, case.healthy, case.sign, case.angle
    assert sorted(log.scenario) == sorted(COLLAPSED + [SIGN])
    assert (log.status == "recovered").all()
    # identity check (phase 0) when the label's own residual is < 1e-9, else a normal two-start solve (phase 1)
    assert str(log.set_index("scenario").at[SIGN, "phase"]) in ("0", "1")
    assert (log.residual < 1e-9).all() and (log.min_vm_after >= 0.5).all() and (log.max_branch_angle_deg <= 90).all()
    ob, nb = _rd(case, "bus_data", COLLAPSED + HEALTHY), _io.read_scenarios(out, "bus_data", COLLAPSED + HEALTHY)
    assert len(ob) == len(nb) and (ob[["scenario", "bus"]].to_numpy() == nb[["scenario", "bus"]].to_numpy()).all()
    for s in COLLAPSED:
        assert nb[nb.scenario == s].Vm.min() >= 0.5
    # the angle_gt90 scenario is flagged, not repaired
    a_old = pd.read_parquet(src / "bus_data.parquet").query(f"scenario == {ANGLE}")
    a_new = pd.read_parquet(out / "bus_data.parquet").query(f"scenario == {ANGLE}")
    assert a_old[["Vm", "Va"]].equals(a_new[["Vm", "Va"]])
    # sign-flip: magnitudes come back positive and equal to the unflipped originals (genuine healthy label)
    sf = nb[nb.scenario == SIGN].sort_values("bus")
    assert (sf.Vm > 0).all()
    assert np.abs(sf.Vm.to_numpy() - ob[ob.scenario == SIGN].sort_values("bus").Vm.abs().to_numpy()).max() < 1e-9
    orig = pd.read_parquet(out / "originals.parquet" / "bus_data.parquet")
    assert sorted(orig.scenario.unique()) == sorted(COLLAPSED + [SIGN])
    assert all((out / "originals.parquet" / f"{t}.parquet").exists() for t in TABLES)
    assert (out / "repair_log.csv").exists()


def test_verify_passes(repaired):
    src, out, _ = repaired
    res = vf.verify(src, out, say=lambda *_: None)
    assert res["ok"], res["failures"]
    assert res["max_diff_unchanged"] == 0.0 and res["max_residual_recovered"] < 1e-9
    assert vf.run(src, out) == 0


def _broken(repaired, tmp_path):
    src, out, _ = repaired
    d = tmp_path / "broken"
    shutil.copytree(out, d)
    return src, d


def _edit(d, table, mask_fn, col, delta):
    for f in (d / f"{table}.parquet").rglob("*.parquet"):
        df = pd.read_parquet(f)
        m = mask_fn(df)
        if m.any():
            df.loc[m, col] = df.loc[m, col] + delta
            df.to_parquet(f, index=False)
            return
    raise AssertionError("nothing edited")


def test_verify_catches_damaged_healthy_scenario(case, repaired, tmp_path):
    src, d = _broken(repaired, tmp_path)
    sid, bus = case.damage
    _edit(d, "bus_data", lambda df: (df.scenario == sid) & (df.bus == bus), "Pd", 1e-6)
    assert not vf.verify(src, d, say=lambda *_: None)["ok"]
    assert vf.run(src, d) == 1


def test_verify_catches_bad_repaired_label(case, repaired, tmp_path):
    src, d = _broken(repaired, tmp_path)
    sid, bus = case.bad_label
    _edit(d, "bus_data", lambda df: (df.scenario == sid) & (df.bus == bus), "Vm", 1e-3)
    res = vf.verify(src, d, say=lambda *_: None)
    assert not res["ok"] and any("residual" in f for f in res["failures"])


def test_verify_catches_dropped_row(repaired, tmp_path):
    src, d = _broken(repaired, tmp_path)
    f = next((d / "gen_data.parquet").rglob("*.parquet"))
    df = pd.read_parquet(f)
    df.iloc[:-1].to_parquet(f, index=False)
    res = vf.verify(src, d, say=lambda *_: None)
    assert not res["ok"] and any("row count" in x for x in res["failures"])


def test_repair_refuses_existing_output_and_missing_ybus(raw, tmp_path):
    out = tmp_path / "o"
    out.mkdir(); (out / "x").write_text("1")
    with pytest.raises(SystemExit):
        rp.run(raw, out, workers=1, log=lambda *_: None)
    shutil.rmtree(raw / "y_bus_data.parquet")
    with pytest.raises(SystemExit):
        rp.run(raw, tmp_path / "o2", workers=1, log=lambda *_: None)


def test_repair_and_verify_flat_layout(case, tmp_path, tabs):
    COLLAPSED, SIGN = case.collapsed, case.sign
    src = tmp_path / "flat" / "raw"
    _write(src, tabs, hive=False)
    out = tmp_path / "flat_out" / "raw"
    log = rp.run(src, out, workers=2, categories="collapsed", log=lambda *_: None)
    assert sorted(log.scenario) == COLLAPSED and (log.status == "recovered").all()   # sign defect not selected
    assert (out / "bus_data.parquet").is_file()
    assert vf.run(src, out) == 0
    neg = pd.read_parquet(out / "bus_data.parquet").query(f"scenario == {SIGN}").Vm
    assert (neg < 0).any()                                                           # left as it was


def test_fixture_repairs_to_the_known_physical_root(case, repaired):
    """Fixture only: the committed slice is the ground truth, so the repair is checked against it and against an
    independent residual computed from the repaired tables (not from the repair log)."""
    if case.name != "fixture":
        pytest.skip("ground truth is known only for the committed fixture")
    src, out, log = repaired
    truth = _rd(case, "bus_data", case.collapsed + [case.sign]).sort_values(["scenario", "bus"]).reset_index(drop=True)
    new = _io.read_scenarios(out, "bus_data", case.collapsed + [case.sign]).sort_values(["scenario", "bus"]).reset_index(drop=True)
    ybus = _io.read_scenarios(out, "y_bus_data", case.collapsed + [case.sign])
    for s in case.collapsed + [case.sign]:
        g = new[new.scenario == s]
        assert g.Vm.min() >= 0.5 and (g.Vm > 0).all()
        assert ck.label_residual(g, ybus[ybus.scenario == s]) < 1e-9
        t = truth[truth.scenario == s]
        assert np.abs(g.Vm.to_numpy() - t.Vm.to_numpy()).max() < 1e-6
        assert np.abs(g.Va.to_numpy() - t.Va.to_numpy()).max() < 1e-4
    # the collapsed scenario had a flagged label before and a clean one after
    before = ck.check(src).set_index("scenario")
    after = ck.check(out).set_index("scenario")
    assert (before.loc[case.collapsed, "category"] == "collapsed").all() and (after.loc[case.collapsed, "category"] == "healthy").all()
    assert before.at[case.sign, "category"] == "negative_only" and after.at[case.sign, "category"] == "healthy"
    assert after.at[case.angle, "category"] == "angle_gt90"          # flagged for review, not repaired
