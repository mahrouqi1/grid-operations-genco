"""Regenerate tests/fixtures/datakit_case14/raw (10 healthy scenarios of pandapower case14 in gridfm-datakit raw layout).

Not run by the test suite (about 45 s, needs pandapower + gridfm_datakit). The committed parquet files are the output of
this script at seed 0; defects (a collapsed |V|, a negative Vm, a branch angle above 90 degrees) are injected by the tests
themselves (tests/_datakit_fixture.py), so the committed data stays healthy.

    python tests/fixtures/make_datakit_fixture.py <scratch_dir>      # then copy <scratch_dir>/dk/case14/raw/*.parquet here
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("core", "tracks/genco-challenge/src", "tracks/powerflow-surrogate/src"):
    sys.path.insert(0, str(ROOT / p))
from genco_challenge import ladder_to_datakit as L  # noqa: E402
from powerflow_surrogate.generate import generate_shard  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
samples, _ = generate_shard("case14", n_samples=12, seed=0)
np.savez_compressed(out / "case14.npz", **{k: np.stack([s[k] for s in samples]) for k in ("p_mw", "q_mvar", "vm_pu", "va_deg")})
L.convert_case("case14", out, out / "dk", n=10, seed=0, workers=1)
