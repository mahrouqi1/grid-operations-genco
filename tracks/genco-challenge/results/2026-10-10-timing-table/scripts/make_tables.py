"""Builds ../tables.md and ../summary.json from ../timing.jsonl only (UNAUDITED). Every number in report.md is copied from tables.md.
    python make_tables.py"""
import json, collections
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(OUT / "timing.jsonl")]
GR = ["case14", "case57", "case118", "case300", "case1354", "goc2000", "goc10000"]
NB = dict(case14=14, case57=57, case118=118, case300=300, case1354=1354, goc2000=2000, goc10000=10000)


def last(kind, **kw):
    """latest row of a kind matching kw (reruns supersede)"""
    r = [x for x in rows if x.get("kind") == kind and all(x.get(k) == v for k, v in kw.items())]
    return r[-1] if r else None


def f(x, d=2):
    if x is None:
        return "n/a"
    return f"{x:.{d}f}" if x < 100 else f"{x:.0f}"


def cell(st):
    return "n/a" if not st or "median_ms" not in st else f"{f(st['median_ms'])} [{f(st['q1_ms'])}, {f(st['q3_ms'])}]"


def med(st):
    return None if not st or "median_ms" not in st else st["median_ms"]


S = {}  # summary: S[grid][name] = median ms (or dict)
md = []
# ---- block loads
md.append("## T0. Blocks and the load average they ran under\n")
md.append("| block | grid | ckpt | load1 before | load1 after | load1 max during | seconds | noisy (>25) | error |\n|---|---|---|---:|---:|---:|---:|---|---|")
blocks = [r for r in rows if r["kind"] == "block"]
for b in blocks:
    md.append(f"| {b['block']} | {b.get('grid')} | {b.get('ckpt', '')} | {b['load_before'][0]:.1f} | {b['load_after'][0]:.1f} | {b['load1_max_during']:.1f} | {b['seconds']:.0f} | {b['noisy']} | {b['error'] or ''} |")
if blocks:
    md.append(f"\nMax load1 over all blocks: {max(b['load1_max_during'] for b in blocks):.1f}; blocks above 25: {sum(b['noisy'] for b in blocks)}; blocks with an error: {sum(1 for b in blocks if b['error'])}.\n")
bm = last("block_start")
if bm:
    md.append("Machine (first block): " + json.dumps(bm["machine"]) + "\n")

# ---- per grid summary
CK = {"goc2000": ["tiny_s0", "small_s0"], "goc10000": ["tiny_s0", "small_s0"]}
for g in GR:
    d = S.setdefault(g, {})
    for comp in ("ybus_csr", "assembly", "dc", "fd2_iters", "fd3_extra", "nr_flat", "nr_dc", "fd2nr", "dc_total", "fd2_total", "fd3_total",
                 "newton_flat_total", "newton_dc_total", "fd2nr_total"):
        r = last("classical", grid=g, component=comp)
        d["c_" + comp] = r
    d["c_meta"] = last("classical_meta", grid=g)
    d["pm"] = last("pm", grid=g)
    d["pm_ipopt"] = last("pm_ipopt", grid=g)
    d["pm_rec"] = last("pm_recorded", grid=g)
    d["pm_fid"] = last("pm_fidelity", grid=g)
    d["pm_load"] = last("pm_load", grid=g)
    for ck in (CK.get(g) or ["pretrained_zeroshot"]):
        pass

def genco_rows(g):
    """list of (ckpt label, dict) for each GENCO checkpoint timed on grid g"""
    cks = sorted({r["ckpt"] for r in rows if r.get("grid") == g and r["kind"] == "pipeline"})
    out = []
    for ck in cks:
        d = {"ckpt": ck}
        d["load"] = last("model_load", grid=g, ckpt=ck)
        for dev, bs in (("cuda", 1), ("cuda", 16), ("cuda", 64), ("cpu", 1), ("cpu", 16)):
            d[f"fwd_{dev}_{bs}"] = last("forward", grid=g, ckpt=ck, device=dev, batch=bs)
        for bs in (16, 64):
            d[f"e2e_{bs}"] = last("e2e", grid=g, ckpt=ck, batch=bs)
        d["pipe"] = last("pipeline", grid=g, ckpt=ck)
        d["gk"] = last("convert_gk", grid=g, ckpt=ck)
        out.append((ck, d))
    return out


def fwdmed(r):
    return None if not r or r.get("oom") else r.get("median_ms")


md.append("\n## T1. Per-scenario wall time, ms: median (Q1 to Q3) over scenarios (n in tables.json); one scenario at a time unless a batch is named\n")
for g in GR:
    c = S[g]
    md.append(f"\n### {g} ({NB[g]} buses)\n")
    md.append("| item | ms median (Q1 to Q3) | n | note |\n|---|---:|---:|---|")
    def line(name, st, note=""):
        md.append(f"| {name} | {cell(st)} | {st.get('n', '') if st else ''} | {note} |")
    line("setup: Ybus csr from table", c["c_ybus_csr"], "part of the assembly")
    line("setup: assembly (Ybus + Sbus + B' + B'')", c["c_assembly"])
    line("DC solve (ours)", c["c_dc"])
    line("fd2 iterations incl. 2 LU", c["c_fd2_iters"])
    line("**fd2 total** (assembly + DC + 2 iter)", c["c_fd2_total"])
    line("**fd3 total**", c["c_fd3_total"])
    line("sparse Newton from flat, solve only", c["c_nr_flat"])
    line("**sparse Newton from flat, total** (assembly + solve)", c["c_newton_flat_total"], f"converged frac {f(c['c_meta']['converged_frac_flat']) if c['c_meta'] else 'n/a'}; median iters {c['c_meta']['it_flat'] if c['c_meta'] else 'n/a'}")
    line("sparse Newton from DC, solve only", c["c_nr_dc"])
    line("**sparse Newton from DC, total** (assembly + DC + solve)", c["c_newton_dc_total"], f"converged frac {f(c['c_meta']['converged_frac_dc']) if c['c_meta'] else 'n/a'}; median iters {c['c_meta']['it_dc'] if c['c_meta'] else 'n/a'}")
    line("fd2nr solve only (2 XB iter then Newton from DC)", c["c_fd2nr"])
    line("**fd2nr total** (assembly + DC + fd2nr)", c["c_fd2nr_total"], f"converged frac {f(c['c_meta']['converged_frac_fd2nr']) if c['c_meta'] else 'n/a'}; median iters {c['c_meta']['it_fd2nr'] if c['c_meta'] else 'n/a'}")
    pm = c["pm"]
    if pm:
        line("PowerModels parse_file (setup)", pm.get("parse"))
        line("PowerModels compute_ac_pf, wall", pm.get("cpf"), f"status {pm['cpf_status_counts']}")
        line("PowerModels compute_ac_pf, its own reported solve_time", pm.get("cpf_reported"))
        line("PowerModels datakit fast path (parse + solve + update + flows)", pm.get("fast"))
        line("PowerModels compute_ac_pf from solved start (lower bound)", pm.get("cpf_warm"))
        line("(python: tables to .m file, not PowerModels)", pm.get("write"))
    if c["pm_ipopt"]:
        line("PowerModels Ipopt path (solve_ac_pf, parse + flows)", c["pm_ipopt"], f"status {c['pm_ipopt']['status_counts']}")
    if c["pm_rec"]:
        r = c["pm_rec"]
        md.append(f"| PowerModels AC time recorded at generation (load 50 to 70), mean / median | {f(r['ac_mean_ms'])} / {f(r['ac_median_ms'])} | {r['n_scen']} | datakit `solve_time`, single worker |")
    for ck, d in genco_rows(g):
        md.append(f"| **GENCO {ck}** | | | model load once {f(d['load']['s_datamodule_and_weights'])} s (+ first CUDA call {f(d['load']['s_first_cuda_call'])} s), {d['load']['n_params']/1e6:.2f}M params |")
        for dev, bs in (("cuda", 1), ("cuda", 16), ("cuda", 64), ("cpu", 1), ("cpu", 16)):
            r = d[f"fwd_{dev}_{bs}"]
            if r:
                md.append(f"| GENCO {ck} (a) forward only, {'GPU' if dev == 'cuda' else 'CPU 4 thr'}, batch {bs}, data resident | {'OOM' if r.get('oom') else cell(r)} | {r.get('n', '')} | {('peak GPU %.2f GB' % r['peak_gpu_gb']) if r.get('peak_gpu_gb') else ''} |")
        p = d["pipe"]
        if p:
            for k, nm in (("build", "merge + build graph (in memory)"), ("norm_mask", "normalise + mask"), ("collate", "collate"), ("h2d", "host to device"),
                          ("fwd", "forward"), ("d2h", "device to host")):
                md.append(f"| GENCO {ck} (b) B=1 component: {nm} | {cell(p[k])} | {p[k]['n']} | |")
            md.append(f"| **GENCO {ck} (b) B=1 conversion + transfer + forward + D2H** | {cell(p['genco_only'])} | {p['genco_only']['n']} | |")
        for bs in (16, 64):
            r = d[f"e2e_{bs}"]
            if r:
                md.append(f"| **GENCO {ck} (b) batch {bs} conversion + transfer + forward + D2H, per scenario** | {'OOM' if r.get('oom') else cell(r['total'])} | {'' if r.get('oom') else r['total']['n']} | |")
        if d["gk"]:
            md.append(f"| GENCO {ck} graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | {cell(d['gk']['total'])} | {d['gk']['total']['n']} | read {f(d['gk']['read_parquet_one_scenario']['median_ms'])} + build/save {f(d['gk']['merge_build_save_pt']['median_ms'])} |")
        if p:
            md.append(f"| **(f) corrector pipeline = fd2 + (b) B=1 + add** | {cell(p['corrector'])} | {p['corrector']['n']} | GENCO {ck}; cost-equal stand-in for the trained corrector |")
            md.append(f"| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | {cell(p['vmdc_newton'])} | {p['vmdc_newton']['n']} | Newton alone {cell(p['newton_vmdc'])}; converged frac {f(p['vmdc_newton_converged_frac'])}, median iters {p['vmdc_newton_iters_median']} (zero-shot/released weights) |")

# ---- speedups
md.append("\n## T2. Speedup ratios (reference time / method time; > 1 means the method is faster)\n")
md.append("NOTE: on goc2000 and goc10000 the reference columns use the Ipopt path (the solver that produced those labels, config pf_fast: false; compute_ac_pf on the rebuilt case lands on another root at 2,000 buses and does not converge at 10,000), so cpf and fast columns are equal there. Reference PowerModels = `compute_ac_pf` wall (solve only), the datakit fast path (parse + solve + flows) and the Ipopt path, single process, measured here at low load; `/64` is the graphkit convention (reference divided by 64). Methods: our fd2 total, our fd2nr total, GENCO forward only (best GPU batch, resident), GENCO B=1 (b), GENCO best-batch (b).\n")
md.append("| grid | GENCO ckpt | PM cpf / GENCO fwd best-batch | PM cpf/64 / GENCO fwd best-batch | PM fast / GENCO fwd best-batch | PM fast/64 / GENCO fwd best-batch | PM fast / GENCO (b) B=1 | PM fast / GENCO (b) best batch | PM fast / fd2 total | PM fast / fd2nr total | PM fast / Newton best (flat or DC) | fd2nr total / GENCO (b) best batch | Newton best / GENCO fwd best-batch |\n|---|---|" + "---:|" * 11)
SP = {}
for g in GR:
    c = S[g]; pm = c["pm"]
    if not pm:
        continue
    goc = g.startswith("goc")
    ipm = med(c["pm_ipopt"]) if c["pm_ipopt"] else None
    cpf, fast = (ipm, ipm) if goc else (med(pm["cpf"]), med(pm["fast"]))   # GOC labels came from the Ipopt path (pf_fast false)
    fd2, fd2nr = med(c["c_fd2_total"]), med(c["c_fd2nr_total"])
    nf, nd = med(c["c_newton_flat_total"]), med(c["c_newton_dc_total"])
    mf, md_ = c["c_meta"]["converged_frac_flat"], c["c_meta"]["converged_frac_dc"]
    cands = [x for x, ok in ((nf, mf >= 0.95), (nd, md_ >= 0.95)) if x and ok]
    nbest = min(cands) if cands else None
    for ck, d in genco_rows(g):
        fw = [fwdmed(d.get(f"fwd_cuda_{b}")) for b in (1, 16, 64)]
        fw = [x for x in fw if x]
        fbest = min(fw) if fw else None
        e2e = [med(d["e2e_%d" % b]["total"]) for b in (16, 64) if d.get("e2e_%d" % b) and not d["e2e_%d" % b].get("oom")]
        b1 = med(d["pipe"]["genco_only"]) if d["pipe"] else None
        ebest = min([x for x in e2e + [b1] if x]) if (e2e or b1) else None
        r = lambda a, b: (a / b if a and b else None)
        row = dict(cpf_over_fwd=r(cpf, fbest), cpf64_over_fwd=r(cpf / 64, fbest), fast_over_fwd=r(fast, fbest), fast64_over_fwd=r(fast / 64, fbest),
                   fast_over_b1=r(fast, b1), fast_over_ebest=r(fast, ebest), fast_over_fd2=r(fast, fd2), fast_over_fd2nr=r(fast, fd2nr),
                   fast_over_nbest=r(fast, nbest), fd2nr_over_ebest=r(fd2nr, ebest), nbest_over_fwd=r(nbest, fbest),
                   ipopt_over_fwd=r(med(c["pm_ipopt"]) if c["pm_ipopt"] else None, fbest), fbest=fbest, ebest=ebest, b1=b1, nbest=nbest)
        SP[(g, ck)] = row
        fx = lambda v: "n/a" if v is None else f"{v:.2f}"
        md.append(f"| {g} | {ck} | {fx(row['cpf_over_fwd'])} | {fx(row['cpf64_over_fwd'])} | {fx(row['fast_over_fwd'])} | {fx(row['fast64_over_fwd'])} | {fx(row['fast_over_b1'])} | {fx(row['fast_over_ebest'])} | {fx(row['fast_over_fd2'])} | {fx(row['fast_over_fd2nr'])} | {fx(row['fast_over_nbest'])} | {fx(row['fd2nr_over_ebest'])} | {fx(row['nbest_over_fwd'])} |")
md.append("\nRecorded-at-generation reference (mean, load 50 to 70) over the same GENCO times, and the Ipopt path:\n")
md.append("| grid | GENCO ckpt | recorded PM mean / GENCO fwd best-batch | recorded PM mean/64 / GENCO fwd best-batch | recorded PM mean/64 / GENCO (b) best batch | Ipopt path / GENCO fwd best-batch | Ipopt path / fd2nr total |\n|---|---|---:|---:|---:|---:|---:|")
for (g, ck), row in SP.items():
    c = S[g]; rec = c["pm_rec"]["ac_mean_ms"]
    ip = med(c["pm_ipopt"]) if c["pm_ipopt"] else None
    fx = lambda v: "n/a" if v is None else f"{v:.2f}"
    fd2nr = med(c["c_fd2nr_total"])
    md.append(f"| {g} | {ck} | {fx(rec / row['fbest'] if row['fbest'] else None)} | {fx(rec / 64 / row['fbest'] if row['fbest'] else None)} | {fx(rec / 64 / row['ebest'] if row['ebest'] else None)} | {fx(row['ipopt_over_fwd'])} | {fx(ip / fd2nr if ip and fd2nr else None)} |")

md.append("\n## T3. Fidelity of the rebuilt PowerModels inputs (max abs difference of the solved state against the datakit label, scenario 0)\n")
for r in rows:
    if r["kind"] in ("pm_fidelity", "pm_fidelity_ipopt"):
        md.append(f"- {r['kind']} {r['grid']}: status {r.get('status', 'LOCALLY_SOLVED')} |dVm| {r.get('max_dvm_vs_label', 'n/a')} |dVa| deg {r.get('max_dva_deg_vs_label', 'n/a')} {r.get('note', '')}")
(OUT / "tables.md").write_text("\n".join(md) + "\n")
json.dump({f"{g}|{ck}": v for (g, ck), v in SP.items()}, open(OUT / "summary.json", "w"), indent=1)
print("\n".join(md)[:3000])
