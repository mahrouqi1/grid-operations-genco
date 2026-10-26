"""Batch packing of the node-budget plugin (genco_challenge.node_budget.pack)."""
import random

import pytest

_m = pytest.importorskip("gridfm_graphkit.datasets.hetero_powergrid_datamodule")


def test_pack_respects_budget_and_cap():
    from genco_challenge.node_budget import pack
    sizes = [64] * 500 + [3000] * 200 + [9591] * 50
    b = pack(sizes, 40000, 32, random.Random(0))
    assert sorted(i for x in b for i in x) == list(range(len(sizes)))      # every sample exactly once
    for x in b:
        assert len(x) <= 32
        assert sum(sizes[i] for i in x) <= 40000 or len(x) == 1
    assert b != pack(sizes, 40000, 32, random.Random(1))                    # reshuffled by seed


def test_oversized_graph_alone():
    from genco_challenge.node_budget import pack
    b = pack([50000, 10, 10], 1000, 32, random.Random(0))
    assert [0] in b
