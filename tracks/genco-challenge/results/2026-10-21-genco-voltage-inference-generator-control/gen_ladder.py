"""Ladder-generator shards (load-only perturbation, PV |V| fixed, the repo's oracle) for each carved piece, then conversion to the datakit layout
and the rate_a fix, exactly the route the whole-grid arm took. python gen_ladder.py PIECE... [--n 120]. UNAUDITED."""
import argparse, os, subprocess, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]; sys.path.insert(0, str(HERE))
import grids as G
ap = argparse.ArgumentParser(); ap.add_argument("pieces", nargs="+"); ap.add_argument("--n", type=int, default=120); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
env = os.environ | {"GRID_OPS_PGLIB_DIR": str(G.PGLIB), "PYTHONNOUSERSITE": "1", "PYTHONPATH": f"{REPO/'core'}:{REPO/'tracks/powerflow-surrogate/src'}:{REPO/'tracks/genco-challenge/src'}", "OMP_NUM_THREADS": "1"}
PY = sys.executable
G.SHARDS.mkdir(parents=True, exist_ok=True)
for p in a.pieces:
    r = subprocess.run([PY, "-m", "powerflow_surrogate.run_generate", "--cases", p, "--index", "0", "--out", str(G.SHARDS), "--n", str(a.n), "--budget-hours", "0.6", "--seed", "0"], env=env, capture_output=True, text=True)
    print(p, "generate rc", r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:150], flush=True)
    if r.returncode: continue
    code = ("import sys; import genco_challenge.ladder_to_datakit as L; from pathlib import Path; L.PGLIB_DIR = Path(sys.argv[1]); sys.exit(L.main(sys.argv[2:]))")
    r = subprocess.run([PY, "-c", code, str(G.PGLIB), "--cases", p, "--n", str(a.n), "--shards", str(G.SHARDS), "--out", str(G.ROOT), "--workers", str(a.workers)], env=env, capture_output=True, text=True)
    print(p, "convert rc", r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:150], flush=True)
    if r.returncode in (0, 1):   # audit R2-F4: rc 1 = some samples dropped but the tree was written; the rate fix must still run
        r = subprocess.run([PY, "-m", "genco_challenge.pglib_rate_a_fix", str(G.ROOT), "--pglib", str(G.PGLIB), "--cases", p], env=env, capture_output=True, text=True)
        print(p, "rate_fix rc", r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:150], flush=True)
