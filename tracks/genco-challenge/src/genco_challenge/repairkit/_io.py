"""Layout handling for gridfm-datakit-format raw dirs (hive-partitioned ``scenario_partition=N`` or plain files).

A raw dir holds ``bus_data.parquet``, ``gen_data.parquet``, ``branch_data.parquet`` and (optionally) ``y_bus_data.parquet``;
each is either a directory of ``scenario_partition=N/`` leaves or a single file / flat directory. Work is done one
"batch" at a time: a batch is one partition (hive layout) or a block of 200 scenario ids (flat layout).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("bus_data", "gen_data", "branch_data", "y_bus_data")
CORE = ("bus_data", "gen_data", "branch_data")
KEYS = {"bus_data": ["scenario", "bus"], "gen_data": ["scenario", "idx"], "branch_data": ["scenario", "idx"],
        "y_bus_data": ["scenario", "index1", "index2"]}
FLAT_BLOCK = 200


@dataclass(frozen=True)
class Batch:
    label: str                      # partition name ("scenario_partition=7") or "ids<k>"
    ids: tuple[int, ...] | None     # None for a hive batch (read the leaf dir), else the scenario ids of the block


def tpath(raw: Path, t: str) -> Path:
    return Path(raw) / f"{t}.parquet"


def has_table(raw: Path, t: str) -> bool:
    return tpath(raw, t).exists()


def is_hive(raw: Path, t: str = "bus_data") -> bool:
    p = tpath(raw, t)
    return p.is_dir() and any(c.is_dir() and c.name.startswith("scenario_partition=") for c in p.iterdir())


def _pnum(name: str) -> int:
    return int(name.split("=", 1)[1])


def batches(raw: Path) -> list[Batch]:
    raw = Path(raw)
    if is_hive(raw):
        names = sorted((c.name for c in tpath(raw, "bus_data").iterdir() if c.name.startswith("scenario_partition=")), key=_pnum)
        return [Batch(n, None) for n in names]
    ids = np.sort(pd.read_parquet(tpath(raw, "bus_data"), columns=["scenario"])["scenario"].unique())
    return [Batch(f"ids{i}", tuple(int(x) for x in ids[i:i + FLAT_BLOCK])) for i in range(0, len(ids), FLAT_BLOCK)]


def read_batch(raw: Path, t: str, b: Batch, columns=None) -> pd.DataFrame | None:
    """The rows of table ``t`` in batch ``b`` (None when the table has no such partition)."""
    p = tpath(raw, t)
    if b.ids is None:
        leaf = p / b.label
        if not leaf.exists():
            return None
        return pd.read_parquet(leaf, columns=columns)
    return pd.read_parquet(p, filters=[("scenario", "in", list(b.ids))], columns=columns)


def read_scenarios(raw: Path, t: str, ids) -> pd.DataFrame:
    """Rows of the given scenarios, without any hive ``scenario_partition`` column."""
    d = pd.read_parquet(tpath(raw, t), filters=[("scenario", "in", [int(i) for i in ids])])
    return d.drop(columns=["scenario_partition"], errors="ignore") if is_hive(raw, t) else d


def leaf_files(leaf: Path) -> list[Path]:
    return sorted(p for p in Path(leaf).rglob("*") if p.is_file())


def write_like(df: pd.DataFrame, dst: Path, schema_from: Path | None) -> None:
    """Write ``df`` to ``dst`` keeping the arrow schema of ``schema_from`` (a parquet file) when possible."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tab = None
    if schema_from is not None:
        try:
            sch = pq.read_schema(schema_from).remove_metadata()
            tab = pa.Table.from_pandas(df, schema=sch, preserve_index=False)
        except Exception:  # noqa: BLE001 (fall back to the inferred schema)
            tab = None
    if tab is None:
        tab = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tab, dst, compression="snappy")


def copy_entry(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
