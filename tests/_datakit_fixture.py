"""Shared helpers for the tests that need a datakit-format dataset (RepairKit, ladder_to_datakit).

``FIXTURE_RAW`` is a committed, 10-scenario slice of pandapower case14 in the exact gridfm-datakit raw layout (flat
``bus_data.parquet`` / ``gen_data.parquet`` / ``branch_data.parquet`` / ``y_bus_data.parquet``, written by
``genco_challenge.ladder_to_datakit`` from verified AC solutions; about 45 KB; regenerate with
``tests/fixtures/make_datakit_fixture.py``). Every committed scenario is healthy. ``inject_defects`` makes three
deliberate, deterministic defects in a copy of the tables:

* a collapsed |V| label: the Vm of every PQ bus scaled by 0.35 (min |Vm| about 0.35, below the 0.5 threshold). Newton from
  a flat start does not reach a low-voltage root on case14 at its loading, so the collapsed label is not itself a power-flow
  root, which the private data's collapsed labels are. The RepairKit never relies on that (it flags by |Vm| and re-solves);
* a negative Vm: three PQ buses written as -|V| with the angle shifted by 180 degrees (the same complex voltage);
* an in-service branch angle above 90 degrees (Va of one branch end shifted by 100 degrees), flagged and not repaired.

``private_data_visible`` is False when the env var ``GENCO_HIDE_PRIVATE_DATA`` is set to 1, which lets the no-private-data
path be exercised on a machine that has the data (nothing is moved or deleted).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RAW = ROOT / "tests/fixtures/datakit_case14/raw"
TABLES = ("bus_data", "gen_data", "branch_data", "y_bus_data")

# scenario ids of the 10 committed scenarios, and which one gets which defect
FX_COLLAPSED, FX_SIGN, FX_ANGLE = [2], 5, 7
FX_HEALTHY = [0, 1, 3, 4, 5, 6, 7, 8, 9]       # includes the two scenarios that get the sign and angle defects


def hidden() -> bool:
    return os.environ.get("GENCO_HIDE_PRIVATE_DATA", "") not in ("", "0")


def private_data_visible(path: Path) -> bool:
    return (not hidden()) and Path(path).exists()


def read_fixture() -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(FIXTURE_RAW / f"{t}.parquet") for t in TABLES}


def inject_defects(tabs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Copy of ``tabs`` with the collapsed, negative and angle defects described in the module doc."""
    t = {k: v.copy() for k, v in tabs.items()}
    b = t["bus_data"]
    for s in FX_COLLAPSED:
        m = (b.scenario == s) & (b.PQ == 1)
        b.loc[m, "Vm"] *= 0.35
    pq = b[(b.scenario == FX_SIGN) & (b.PQ == 1)].bus.sort_values().iloc[:3]
    m = (b.scenario == FX_SIGN) & b.bus.isin(pq)
    b.loc[m, "Va"] += 180.0
    b.loc[m, "Vm"] *= -1.0
    br = t["branch_data"]
    ref = set(b.loc[(b.scenario == FX_ANGLE) & (b.REF == 1), "bus"])
    f = br[(br.scenario == FX_ANGLE) & (br.br_status == 1) & ~br.from_bus.isin(ref)].from_bus.iloc[0]
    b.loc[(b.scenario == FX_ANGLE) & (b.bus == f), "Va"] += 100.0
    return t


def materialize_shard(case: str, dst: Path) -> Path:
    """Write ``<dst>/<case>.npz`` (the layout ``ladder_to_datakit.convert_case`` reads) from the committed JSON copy of the
    first 8 samples of the lab's real shard (float32 values, exact round trip). JSON because the export excludes *.npz."""
    import json
    d = json.loads((ROOT / "tests/fixtures/ladder_shards" / f"{case}.json").read_text())
    Path(dst).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(dst) / f"{case}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})
    return Path(dst)
