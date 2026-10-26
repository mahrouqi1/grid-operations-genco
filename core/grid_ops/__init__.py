"""grid_ops - shared substrate for graph-ML on power-grid operations.

Lives in ``core/`` and is reused by every track (powerflow-surrogate,
and future opf-learning / dispatch-uc). Put ``core/`` on PYTHONPATH so
``import grid_ops`` resolves (the parent of the package, per the
migration convention).
"""
from __future__ import annotations

from grid_ops.testbeds import (
    TESTBEDS,
    FAMILIES,
    load_case,
    list_testbeds,
    inventory,
)
from grid_ops.graph import build_hetero_data

__all__ = [
    "TESTBEDS", "FAMILIES", "load_case", "list_testbeds", "inventory",
    "build_hetero_data",
]
