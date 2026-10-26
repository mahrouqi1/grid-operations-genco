"""Registry for the generator control (candidate 12): the carved pieces of candidates 10 (seen families) and 11 (unseen families),
regenerated through the LADDER generator. UNAUDITED."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXT = HERE.parents[3] / "data/external"
WORK = EXT / "genco-generator-control"
PGLIB = WORK / "pglib"            # carved .m files copied as pglib_opf_<piece>.m
SHARDS = WORK / "shards"
ROOT = WORK / "tree"              # converted datakit-layout trees, <ROOT>/pglib_opf_<piece>/raw
CTRL = HERE.parent / "2026-10-21-genco-voltage-inference-control"; FVS = HERE.parent / "2026-10-21-genco-voltage-inference-family-vs-size"
_s = json.loads((CTRL / "protocol.json").read_text())["subgrids"]; _u = json.loads((FVS / "protocol.json").read_text())["subgrids"]
PIECES = {f"pglib_opf_{n}": {"group": "seen", "family": "seen", "n_bus": v["n_bus"], "source": n} for n, v in _s.items()}
PIECES.update({f"pglib_opf_{n}": {"group": "unseen", "family": v["family"], "n_bus": v["n_bus"], "source": n} for n, v in _u.items()})
SEL = list(PIECES)
