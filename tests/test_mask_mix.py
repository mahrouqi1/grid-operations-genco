"""Schedules of the transfer experiment's mask policies (genco_challenge.mask_mix.pf_fraction).

Skipped in the public export, deliberately: ``genco_challenge.mask_mix`` monkey-patches ``AddPretrainMask.forward`` and
reads ``AddPFHeteroMask`` from ``gridfm_graphkit.datasets.masking`` AT IMPORT, and both classes exist only on the graphkit
pretraining branch (data/external/gridfm-graphkit-repro-pretraining), not in released gridfm-graphkit. The schedule
function itself is pure, but it cannot be imported without that branch, and stubbing the graphkit classes would test the
stub rather than the plugin, so the test stays behind the branch check.
"""
import importlib
import os
import sys

import pytest

_m = pytest.importorskip("gridfm_graphkit.datasets.masking")
if not hasattr(_m, "AddPretrainMask"):
    pytest.skip("needs the graphkit pretraining branch on PYTHONPATH (data/external/gridfm-graphkit-repro-pretraining)",
                allow_module_level=True)


def _load(policy, epochs):
    os.environ["MASK_POLICY"], os.environ["MASK_EPOCHS"] = policy, str(epochs)
    sys.modules.pop("genco_challenge.mask_mix", None)
    return importlib.import_module("genco_challenge.mask_mix")


def test_schedules():
    m = _load("curriculum", 8)
    f = m.pf_fraction
    assert [f(e, "pretrain", 8) for e in range(8)] == [0.0] * 8
    assert [f(e, "pf", 8) for e in range(8)] == [1.0] * 8
    assert [f(e, "halfmix", 8) for e in range(8)] == [0.0] * 4 + [0.5] * 4
    assert [f(e, "mix50", 8) for e in range(8)] == [0.5] * 8
    c = [f(e, "curriculum", 8) for e in range(8)]
    assert c[0] == 0.0 and c[-1] == 1.0 and all(b > a for a, b in zip(c, c[1:]))


def test_bad_policy():
    with pytest.raises(ValueError):
        _load("nonsense", 8)
    with pytest.raises(ValueError):
        _load("halfmix", 1)
