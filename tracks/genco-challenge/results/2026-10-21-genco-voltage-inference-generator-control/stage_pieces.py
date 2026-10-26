"""Copy each carved piece .m into data/external/genco-generator-control/pglib/pglib_opf_<piece>.m (function name = file stem) so the repo's
pglib loaders (testbeds, ladder_to_datakit) treat it as a PGLib case. Values unchanged. UNAUDITED."""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grids as G
G.PGLIB.mkdir(parents=True, exist_ok=True)
for stem, v in G.PIECES.items():
    src = (G.EXT / ("genco-carved-subgrids" if v["group"] == "seen" else "genco-carved-unseen") / "grids" / f"{v['source']}.m")
    t = src.read_text(); t = re.sub(r"function mpc = \S+", f"function mpc = {stem}", t, count=1); (G.PGLIB / f"{stem}.m").write_text(t)
print(len(G.PIECES), "pieces staged")
