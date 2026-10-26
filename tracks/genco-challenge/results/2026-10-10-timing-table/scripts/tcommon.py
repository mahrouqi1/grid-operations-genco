"""Shared helpers for the H3 timing table (UNAUDITED). Rows are appended to ../timing.jsonl.

Every block script wraps its work in `Block(...)`, which samples the load average every 5 s on a thread and writes a
`block` row with load before / after / max during, so each number can be traced to the load it was measured under.
"""
from __future__ import annotations
import json, os, platform, statistics, subprocess, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
TIMING = Path(os.environ.get("TIMING_JSONL", OUT / "timing.jsonl"))
REPO = HERE.parents[4]
LAD = REPO / "data/external/genco-ladder"
EVAL2, RELAB, GOCD = LAD / "eval2", LAD / "data-relabeled", LAD / "data-goc"

# short name -> (data root, dir name, family)
GRIDS = {
    "case14": (EVAL2, "pglib_opf_case14_ieee", "ladder"),
    "case57": (EVAL2, "pglib_opf_case57_ieee", "ladder"),
    "case118": (EVAL2, "pglib_opf_case118_ieee", "ladder"),
    "case300": (EVAL2, "pglib_opf_case300_ieee", "ladder"),
    "case1354": (RELAB, "pglib_opf_case1354_pegase", "ladder"),
    "goc2000": (GOCD, "pglib_opf_case2000_goc", "goc"),
    "goc10000": (GOCD, "pglib_opf_case10000_goc", "goc"),
}
N_BUS = {"case14": 14, "case57": 57, "case118": 118, "case300": 300, "case1354": 1354, "goc2000": 2000, "goc10000": 10000}


def machine() -> dict:
    cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    d = {"host": os.uname().nodename, "cpu": cpu, "n_cpus": os.cpu_count(), "gpus_now": gpu, "kernel": platform.release(),
         "python": platform.python_version(), "affinity": sorted(os.sched_getaffinity(0))}
    try:
        import torch
        d |= {"torch": torch.__version__, "cuda": torch.version.cuda}
    except Exception:
        pass
    return d


def emit(**kw):
    kw.setdefault("t", time.strftime("%F %T"))
    kw.setdefault("loadavg", list(os.getloadavg()))
    with open(TIMING, "a") as fh:
        fh.write(json.dumps(kw) + "\n")


class Block:
    """with Block('classical', grid='case14'): ...  -> a `block` row with load before/after/max."""

    def __init__(self, name, **kw):
        self.name, self.kw, self.samples, self._stop = name, kw, [], threading.Event()

    def _run(self):
        while not self._stop.wait(5.0):
            self.samples.append(os.getloadavg()[0])

    def __enter__(self):
        self.l0, self.t0 = list(os.getloadavg()), time.time()
        emit(kind="block_start", block=self.name, **self.kw, machine=machine())
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, et, ev, tb):
        self._stop.set()
        self.th.join()
        l1 = list(os.getloadavg())
        mx = max([self.l0[0], l1[0], *self.samples])
        emit(kind="block", block=self.name, **self.kw, load_before=self.l0, load_after=l1, load1_max_during=mx,
             load1_samples=self.samples[:200], seconds=time.time() - self.t0, noisy=bool(mx > 25), error=repr(ev) if ev else None)
        return False


def stats(xs_s):
    """seconds -> ms summary dict (median, q1, q3, iqr, min, mean, n)."""
    a = np.asarray(xs_s, float) * 1e3
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "iqr_ms": float(q3 - q1),
            "min_ms": float(a.min()), "mean_ms": float(a.mean())}
