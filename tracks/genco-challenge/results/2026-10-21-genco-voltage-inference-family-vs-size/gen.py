"""Generate datakit PF scenarios for the carved subgrids with the repo's ladder_data pipeline (default_pf.yaml with GENCO's eval edits:
one topology variant per load scenario) and its collapsed-label validation. python gen.py NAME... [--scenarios N]. UNAUDITED."""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tracks/genco-challenge/src"))
import genco_challenge.ladder_data as LD
OUT = REPO / "data/external/genco-carved-unseen"
LD.PGLIB = OUT / "grids"; LD.WORK = OUT / "work"
ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--scenarios", type=int, default=100); ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--timeout", type=int, default=3600); a = ap.parse_args()
for n in a.names:
    m = LD.build(n, a.scenarios, a.procs, a.timeout)
    print(json.dumps({k: m.get(k) for k in ("name", "n_bus", "status", "returncode", "seconds", "scenarios_requested", "scenarios_written", "scenarios_dropped_invalid")}), flush=True)
