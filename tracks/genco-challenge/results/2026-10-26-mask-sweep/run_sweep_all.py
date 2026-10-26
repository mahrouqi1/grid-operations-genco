"""Driver: interpolator tasks on <=5 CPU worker processes + GENCO on one GPU process (6 workers total), logging timings. UNAUDITED.
    nice -n 15 python run_sweep_all.py --gpu 1 [--sets locked15 control fvs] [--stop-before HH:MM]
Stops launching new tasks at --stop-before (EDT, same day); running tasks are given until --kill-at, then terminated."""
import argparse, datetime as dt, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as S

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1"); ap.add_argument("--sets", nargs="+", default=["locked15", "control", "fvs"])
ap.add_argument("--stop-before", default="03:05"); ap.add_argument("--kill-at", default="03:25"); ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--pcts", type=int, nargs="+", default=S.PCTS); ap.add_argument("--no-genco", action="store_true"); ap.add_argument("--no-interp", action="store_true")
a = ap.parse_args()
S.LOGS.mkdir(exist_ok=True)
today = dt.date.today()
hm = lambda s: dt.datetime.combine(today, dt.time(*map(int, s.split(":"))))
STOP, KILL = hm(a.stop_before), hm(a.kill_at)
ENV = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
lock = threading.Lock(); procs = set()


def log(msg):
    with lock, (S.LOGS / "timings.tsv").open("a") as f:
        f.write(f"{dt.datetime.now():%H:%M:%S}\t{msg}\n")


def run(tag, cmd, env):
    if dt.datetime.now() > STOP:
        log(f"SKIPPED(after stop)\t{tag}"); return False
    t0 = time.time()
    with (S.LOGS / f"{tag}.log").open("w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(S.HERE)); procs.add(p)
        while p.poll() is None:
            if dt.datetime.now() > KILL:
                p.terminate(); log(f"KILLED\t{tag}"); procs.discard(p); return False
            time.sleep(5)
    procs.discard(p); log(f"rc={p.returncode}\t{time.time()-t0:.0f}s\t{tag}")
    return p.returncode == 0


BIG = {"pglib_opf_case1354_pegase", "pglib_opf_case2736sp_k", "pglib_opf_case3120sp_k"}
def interp_tasks():
    out = []
    for sname in a.sets:
        _, grids = S.sets()[sname]
        if sname == "locked15":   # big grids split by fraction (largest first) for load balance; the rest one task per grid
            for g in sorted(BIG, key=lambda g: -int(''.join(c for c in g.split('case')[1].split('_')[0] if c.isdigit()))):
                for p in sorted(a.pcts, reverse=True): out.append((sname, g, [p]))
            out += [(sname, g, a.pcts) for g in grids if g not in BIG]
        else:
            out += [(sname, g, a.pcts) for g in grids]
    return out


def itask(t):
    sname, g, pcts = t
    tag = f"interp_{sname}_{g}_{'-'.join(map(str, pcts))}"
    return run(tag, ["nice", "-n", "15", S.PY, "run_sweep_interp.py", "--set", sname, "--grid", g, "--pcts", *map(str, pcts)], ENV)


def genco_all():
    env = dict(ENV, CUDA_VISIBLE_DEVICES=a.gpu)
    for sname in a.sets:
        _, grids = S.sets()[sname]
        if sname == "locked15":
            run("genco_locked15", ["nice", "-n", "15", S.PY, "run_sweep_genco.py", "--set", sname], env)
        else:
            for g in grids: run(f"genco_{sname}_{g}", ["nice", "-n", "15", S.PY, "run_sweep_genco.py", "--set", sname, "--grid", g], env)


log("START\t" + " ".join(sys.argv[1:]))
th = threading.Thread(target=genco_all) if not a.no_genco else None
if th: th.start()
if not a.no_interp:
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(itask, interp_tasks()))
if th: th.join()
log("DONE")
