"""Does datakit's fast path (trust-region, flat start) reproduce GENCO's PUBLISHED collapsed labels, and does
method=:newton give the physical root? (an owner decision, audit post-claims R1-F1.) Per affected subgrid: up to
N_COLL collapsed scenarios (relabeled.csv rows with status recovered and label_reason min_vm_below_0.5 or
negative_vm, evenly spaced) and 1 healthy control (lowest scenario id not in relabeled.csv). Inputs rebuilt from the
published raw tables with build_mpc.build (as in 2026-10-10-datakit-collapse-cause); physical root from the
corrected tree. Writes out.jsonl (one row per scenario x method) and cases.csv. Run: python run.py"""
import os, subprocess, json, numpy as np, pandas as pd
from build_mpc import build
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
EXT = f"{REPO}/data/external"
JULIA = "<CONDA_ENVS>/grid-operations-genco/julia_env/pyjuliapkg/install/bin/julia"
PROJ = "<CONDA_ENVS>/grid-operations-genco/julia_env"
ENV = dict(os.environ, JULIA_DEPOT_PATH="<JULIA_DEPOT>")
ALL9 = ["subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case9591_goc_idx_12",
        "subnet_case5658_epigrids_idx_8", "subnet_case4837_goc_idx_28", "subnet_case19402_goc_idx_47",
        "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37", "subnet_case10480_goc_idx_26"]
N_COLL = 6
WORK = f"{EXT}/genco-published-newton-work"

def vcsv(raw, sc, path):
    b = pd.read_parquet(f"{raw}/bus_data.parquet", filters=[("scenario", "==", sc)]).sort_values("bus")
    pd.DataFrame({"bus": b.bus + 1, "vm": b.Vm, "va": np.radians(b.Va)}).to_csv(path, index=False)
    return b

def main():
    os.makedirs(WORK, exist_ok=True)
    out = f"{HERE}/out.jsonl"; done = set()
    if os.path.exists(out):
        # audit R1-F4: a tag is done only when every method wrote its row
        seen = {}
        for l in open(out):
            r = json.loads(l); seen.setdefault(r["tag"], set()).add(r["variant"])
        done = {t for t, v in seen.items() if {"trust_region_default", "newton"} <= v}
    cases = []
    for g in ALL9:
        src, cor = f"{EXT}/genco-pf-pretraining/{g}/raw", f"{EXT}/genco-pf-pretraining-corrected/{g}/raw"
        rl = pd.read_csv(f"{cor}/relabeled.csv")
        coll = rl[(rl.status == "recovered") & rl.label_reason.isin(["min_vm_below_0.5", "negative_vm"])].scenario.to_numpy()
        coll = np.sort(coll)[np.linspace(0, len(coll) - 1, min(N_COLL, len(coll))).round().astype(int)] if len(coll) else []
        allsc = np.sort(pd.read_parquet(f"{src}/bus_data.parquet", columns=["scenario"]).scenario.unique())
        healthy = [s for s in allsc if s not in set(rl.scenario)][:1]
        for kind, scs in (("collapsed", coll), ("healthy", healthy)):
            for sc in scs:
                sc = int(sc); tag = f"{g}|{sc}|{kind}"
                cases.append({"subgrid": g, "scenario": sc, "kind": kind})
                if tag in done:
                    continue
                m = f"{WORK}/{g}_s{sc}.m"; lab = f"{WORK}/{g}_s{sc}_label.csv"; phy = f"{WORK}/{g}_s{sc}_phys.csv"
                build(src, sc, m); vcsv(src, sc, lab)
                if kind == "collapsed":
                    vcsv(cor, sc, phy)
                p = subprocess.run(["nice", "-n", "10", JULIA, f"--project={PROJ}", f"{HERE}/probe2.jl", m, lab, phy, out, tag],
                                   env=ENV, capture_output=True, text=True)
                if p.returncode:
                    with open(out, "a") as fh:
                        fh.write(json.dumps({"tag": tag, "variant": "error", "status": "julia_error", "err": p.stderr[-400:]}) + "\n")
                print(tag, "done", flush=True)
    pd.DataFrame(cases).to_csv(f"{HERE}/cases.csv", index=False)

if __name__ == "__main__":
    main()
