"""Registry for the carved-subgrid control (candidate 10). UNAUDITED."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXT = HERE.parents[3] / "data/external"
CARVED = EXT / "genco-carved-subgrids"
ROOT = CARVED / "work/data"
SEL = json.loads((HERE / "selected.json").read_text())["selected"] if (HERE / "selected.json").exists() else []
PARENT = {n: n.replace("carved_", "").rsplit("_c", 1)[0] for n in SEL}
TIER = {n: ("b" if int(n.rsplit("_c", 1)[1]) >= 10 else "a") for n in SEL}
