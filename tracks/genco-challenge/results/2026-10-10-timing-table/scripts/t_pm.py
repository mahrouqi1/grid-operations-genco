"""Block P (UNAUDITED): PowerModels timings at low load, single process, 1 Julia thread, pinned by the caller.
    taskset -c 16-19 python t_pm.py <grid> [--n-fast 30] [--n-ipopt 5] [--no-ipopt]
Per scenario a MATPOWER case is rebuilt from the datakit raw tables (bus type, Pd/Qd/GS/BS, gen p/q/limits, Vg = bus Vm, branch r/x/b/tap/
shift/rate/status): the operating point the PF solvers saw at generation. Initial voltages: PQ buses flat (Vm 1, Va 0) and PV/REF Vm = setpoint
(the datakit used the OPF |V| at every bus, which is not stored; the variant 'warm_labels' starts from the solved state and so is a lower
bound on the time). The Julia code is the datakit's own (process/process_network.py init_julia), copied verbatim:
  parse         PowerModels.parse_file(.m)
  cpf           compute_ac_pf(network) on the parsed network (NLsolve Newton): wall ms, and the 'solve_time' field it reports
  fast_path     run_pf_fast(file): parse + compute_ac_pf + update_data! + calc_branch_flow_ac (what datakit called with pf_fast)
  ipopt         run_pf core (solve_ac_pf with Ipopt, tol 1e-6, print_level 0) incl. parse + update_data! + flows
  write_mpc     python side: tables -> .m text (setup, not part of the PowerModels solve)
"""
import argparse, os, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from tcommon import *

WARM = 5
BASE = 100.0


def mpc_text(b, g, r, warm=False):
    b = b.sort_values("bus")
    n = len(b)
    typ = np.where(b["REF"].to_numpy() == 1, 3, np.where(b["PV"].to_numpy() == 1, 2, 1))
    vm = b["Vm"].to_numpy().copy(); va = np.radians(0) + b["Va"].to_numpy().copy()
    if not warm:
        vm = np.where(typ == 1, 1.0, vm); va = np.zeros(n)
    bus = np.c_[np.arange(1, n + 1), typ, b["Pd"], b["Qd"], b["GS"] * BASE, b["BS"] * BASE, np.ones(n), vm, va, b["vn_kv"], np.ones(n),
                b["max_vm_pu"], b["min_vm_pu"]]
    vmap = b["Vm"].to_numpy()
    gg = g.sort_values("idx")
    gb = gg["bus"].to_numpy().astype(int)
    gen = np.c_[gb + 1, gg["p_mw"], gg["q_mvar"], gg["max_q_mvar"], gg["min_q_mvar"], vmap[gb], np.full(len(gg), BASE), gg["in_service"],
                gg["max_p_mw"], gg["min_p_mw"]]
    cost = np.c_[np.full(len(gg), 2), np.zeros(len(gg)), np.zeros(len(gg)), np.full(len(gg), 3), gg["cp2_eur_per_mw2"], gg["cp1_eur_per_mw"],
                 gg["cp0_eur"]]
    rr = r.sort_values("idx")
    br = np.c_[rr["from_bus"] + 1, rr["to_bus"] + 1, rr["r"], rr["x"], rr["b"], rr["rate_a"], np.zeros(len(rr)), np.zeros(len(rr)), rr["tap"],
               rr["shift"], rr["br_status"], rr["ang_min"], rr["ang_max"]]
    def mat(a, nm):
        return f"mpc.{nm} = [\n" + "\n".join("  " + " ".join(f"{x:.12g}" for x in row) + ";" for row in a) + "\n];\n"
    return ("function mpc = case_from_dict\nmpc.version = '2';\nmpc.baseMVA = 100.0;\n" + mat(bus, "bus") + mat(gen, "gen") + mat(br, "branch")
            + mat(cost, "gencost"))


JL_SRC = """
using PowerModels
using Ipopt
using Memento
Memento.config!("not_set")
function run_pf_fast(case_file)
    network = PowerModels.parse_file(case_file)
    result = compute_ac_pf(network)
    if result["termination_status"] == false
        return result
    end
    update_data!(network, result["solution"])
    flows = calc_branch_flow_ac(network)
    result["solution"]["branch"] = flows["branch"]
    result["solution"]["pf"] = true
    return result
end
function _run_pf_core(case_file, max_iter)
    network = PowerModels.parse_file(case_file)
    result = solve_ac_pf(
        network,
        optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0, "max_iter" => max_iter),
    )
    if string(result["termination_status"]) != "LOCALLY_SOLVED"
        return result
    end
    update_data!(network, result["solution"])
    flows = calc_branch_flow_ac(network)
    result["solution"]["branch"] = flows["branch"]
    result["solution"]["pf"] = true
    return result
end
const LAST = Ref{Any}(nothing)
function t_parse(f)
    t0 = time_ns(); net = PowerModels.parse_file(f); t1 = time_ns()
    LAST[] = net
    return (t1 - t0) / 1e9
end
function t_cpf()
    t0 = time_ns(); r = compute_ac_pf(LAST[]); t1 = time_ns()
    LAST[] = r
    return (t1 - t0) / 1e9, Float64(r["solve_time"]), string(r["termination_status"])
end
function t_fast(f)
    t0 = time_ns(); r = run_pf_fast(f); t1 = time_ns()
    return (t1 - t0) / 1e9, string(r["termination_status"])
end
function t_ipopt(f, mi)
    t0 = time_ns(); r = _run_pf_core(f, mi); t1 = time_ns()
    LAST[] = r
    return (t1 - t0) / 1e9, string(r["termination_status"])
end
function last_vm(n)
    s = LAST[]["solution"]["bus"]
    return [s[string(i)]["vm"] for i in 1:n], [s[string(i)]["va"] for i in 1:n]
end
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--n-fast", type=int, default=30)
    ap.add_argument("--n-ipopt", type=int, default=5); ap.add_argument("--no-ipopt", action="store_true"); ap.add_argument("--only-ipopt", action="store_true"); ap.add_argument("--ipopt-max-iter", type=int, default=200)
    a = ap.parse_args()
    root, name, fam = GRIDS[a.grid]
    tag = {"grid": a.grid, "n_bus": N_BUS[a.grid]}
    td = Path(tempfile.mkdtemp(dir=os.environ.get("TT_SCRATCH", tempfile.gettempdir())))
    with Block("powermodels", **tag):
        t0 = time.perf_counter()
        from juliacall import Main as jl
        jl.seval(JL_SRC)
        emit(kind="pm_load", **tag, s_julia_start_and_using=time.perf_counter() - t0, julia_threads=int(jl.seval("Threads.nthreads()")),
             julia_version=str(jl.seval("VERSION")))
        raw = root / name / "raw"
        bus = pd.read_parquet(raw / "bus_data.parquet"); gen = pd.read_parquet(raw / "gen_data.parquet"); br = pd.read_parquet(raw / "branch_data.parquet")
        bb, gb, rb = (dict(tuple(x.groupby("scenario"))) for x in (bus, gen, br))
        scen = sorted(bb)
        rec = {k: [] for k in ("write", "parse", "cpf", "cpf_reported", "fast", "cpf_warm", "ipopt")}
        status = {"cpf": [], "fast": [], "ipopt": []}
        fid = None
        total = 0 if a.only_ipopt else WARM + a.n_fast
        for i in range(total):
            s = scen[i % len(scen)]
            t0 = time.perf_counter(); txt = mpc_text(bb[s], gb[s], rb[s]); f = td / f"c{i}.m"; f.write_text(txt); tw = time.perf_counter() - t0
            tp = float(jl.t_parse(str(f)))
            tc_, rep, st = jl.t_cpf()
            if fid is None and str(st) == 'true':
                vm, va = jl.last_vm(len(bb[s]))
                lab = bb[s].sort_values("bus")
                fid = {"max_dvm_vs_label": float(np.abs(np.asarray(vm) - lab["Vm"].to_numpy()).max()),
                       "max_dva_deg_vs_label": float(np.abs(np.degrees(np.asarray(va)) - lab["Va"].to_numpy()).max())}
                emit(kind="pm_fidelity", **tag, scenario=int(s), status=str(st), **fid)
            if str(st) != 'true' and i == 0:
                emit(kind='pm_fidelity', **tag, scenario=int(s), status=str(st), note='compute_ac_pf did not converge from this start')
            tf, stf = jl.t_fast(str(f))
            fw = td / f"w{i}.m"; fw.write_text(mpc_text(bb[s], gb[s], rb[s], warm=True))
            jl.t_parse(str(fw)); twc, _, _ = jl.t_cpf()
            if i >= WARM:
                for k, v in zip(("write", "parse", "cpf", "cpf_reported", "fast", "cpf_warm"), (tw, tp, float(tc_), float(rep), float(tf), float(twc))):
                    rec[k].append(v)
                status["cpf"].append(str(st)); status["fast"].append(str(stf))
            f.unlink(); fw.unlink()
        if not a.only_ipopt: emit(kind="pm", **tag, warmups=WARM, **{k: stats(v) for k, v in rec.items() if v and k != "ipopt"},
             cpf_status_counts={s: status["cpf"].count(s) for s in set(status["cpf"])},
             fast_status_counts={s: status["fast"].count(s) for s in set(status["fast"])})
        if not a.no_ipopt:
            tw_ = []
            for i in range(2 + a.n_ipopt):
                s = scen[i % len(scen)]
                f = td / f"i{i}.m"; f.write_text(mpc_text(bb[s], gb[s], rb[s]))
                ti, sti = jl.t_ipopt(str(f), a.ipopt_max_iter)
                if i == 0 and str(sti) == 'LOCALLY_SOLVED':
                    vm, va = jl.last_vm(len(bb[s])); lab = bb[s].sort_values('bus')
                    emit(kind='pm_fidelity_ipopt', **tag, scenario=int(s), max_dvm_vs_label=float(np.abs(np.asarray(vm) - lab['Vm'].to_numpy()).max()), max_dva_deg_vs_label=float(np.abs(np.degrees(np.asarray(va)) - lab['Va'].to_numpy()).max()), max_iter=a.ipopt_max_iter)
                if i >= 2:
                    rec["ipopt"].append(float(ti)); status["ipopt"].append(str(sti))
                f.unlink()
            emit(kind="pm_ipopt", **tag, warmups=2, **stats(rec["ipopt"]), status_counts={s: status["ipopt"].count(s) for s in set(status["ipopt"])})
    import shutil; shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
