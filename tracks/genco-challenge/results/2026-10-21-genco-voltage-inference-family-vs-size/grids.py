"""Registry for the carved-subgrid control (candidate 10). UNAUDITED."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXT = HERE.parents[3] / "data/external"
CARVED = EXT / "genco-carved-unseen"
ROOT = CARVED / "work/data"
SEL = json.loads((HERE / "selected.json").read_text())["selected"] if (HERE / "selected.json").exists() else []
PARENT = {n: n.replace("carved_", "").rsplit("_c", 1)[0] for n in SEL}
import glob as _g
_man = {r["name"]: r for f in _g.glob(str(CARVED / "manifest_*.json")) for r in json.loads(open(f).read())}
TIER = {n: ("b" if _man[n].get("tie_max_rule", 1.0) > 1.0 else "a") for n in SEL if n in _man}
FAMILY = {n: ("RTE" if "rte" in n else "PEGASE" if "pegase" in n else "Polish") for n in SEL}
