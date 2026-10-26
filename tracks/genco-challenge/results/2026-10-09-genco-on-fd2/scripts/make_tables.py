"""Recompute every table in report.md from results.jsonl / timing.jsonl / stage1_blend_selection.json. Writes ../tables.md."""
import json, collections
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parents[1]
R = [json.loads(l) for l in open(OUT / "results.jsonl")]
G = ["case14_ieee", "case30_ieee", "case39_epri", "case57_ieee", "case73_ieee_rts", "case89_pegase", "case118_ieee", "case162_ieee_dtc", "case300_ieee"]
SH = lambda g: g.split("_")[0]
idx = collections.defaultdict(list)
for r in R:
    idx[(r["split"], r["grid"].replace("pglib_opf_", ""), r["state"])].append(r)
def one(split, g, st):
    return idx[(split, g, st)][0]
def mean(split, g, st, key):
    return float(np.mean([r[key] for r in idx[(split, g, st)]]))
L = []
def table(title, head, rows):
    L.append(f"\n**{title}**\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
f4 = lambda x: f"{x:.4f}"
f5 = lambda x: f"{x:.5f}"
f3 = lambda x: f"{x:.3f}"
metrics = [("branch_dva_deg", "Branch dVa MAE (deg), lead metric", f4, "dva"), ("va_deg", "Slack-referenced Va MAE (deg)", f4, "va"),
           ("vm_pu", "|V| MAE (pu)", f5, "vm"), ("p_res_mw", "AC active-power residual (MW, own Ybus/Sbus)", f3, "p_res"), ("q_res_mvar", "AC reactive-power residual (MVAr, own Ybus/Sbus)", f3, "q_res")]
S1 = ["dc", "fd2", "nr1", "genco_raw", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va", "d_blend", "e50"]
names = {"dc": "DC", "fd2": "fd2 (=a)", "nr1": "nr1", "genco_raw": "GENCO raw", "b_genco_vm_fd2_va": "b", "c_fd2_vm_genco_va": "c", "d_blend": "d (w=0)", "e50": "e50 (not selected)"}
for key, title, f, _ in metrics:
    table(f"Stage 1, zero-shot GENCO, eval2: {title}", ["grid", *[names[s] for s in S1]],
          [[SH(g), *[f(one("eval2", g, s)[key]) for s in S1]] for g in G])
# beats fd2 summary stage 1
rows = []
for s in ["dc", "nr1", "genco_raw", "genco_pinned", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va", "d_blend", "e50"]:
    c = []
    for key in ("dva", "va", "vm", "p_res", "q_res"):
        k = {"dva": "branch_dva", "va": "va", "vm": "vm", "p_res": "p_res", "q_res": "q_res"}[key]
        c.append(sum(one("eval2", g, s)[f"ratio_{k}_hi"] < 1 for g in G))
    rows.append([s, *[f"{x}/9" for x in c]])
table("Stage 1: grids where the state beats fd2 with the 95% bootstrap upper bound of the ratio below 1 (zero-shot GENCO)", ["state", "branch dVa", "Va", "|V|", "P residual", "Q residual"], rows)
sel = json.loads((OUT / "stage1_blend_selection.json").read_text())
vrows = []
for s in ["dc", "nr1", "genco_raw", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va", "d_blend"]:
    c = []
    for g in ("case14_ieee", "case30_ieee"):
        r = one("val", g, s)
        c.append(f'{r["ratio_branch_dva_vs_fd2"]:.3g} / {r["ratio_vm_vs_fd2"]:.3g} / {r["ratio_p_res_vs_fd2"]:.3g}')
    vrows.append([s, *c])
table("Validation sets (200 training-pool scenarios per dev grid, labels used only here): ratio to fd2, dVa / |V| / P residual", ["state", "case14", "case30"], vrows)
L.append(f'\nBlend selection on validation: best |V| score {sel["best_vm_score"]:.3f} at w_vm 0 (w_vm=0.25 gives {sel["vm_scores"]["0.25"]:.2f}); best angle score {sel["best_angle_score"]:.3f} at (0,0) (next best (0.25,0.25) = {sel["angle_scores"]["(0.25, 0.25)"]:.1f}). Adopted weights: w_vm {sel["w_vm"]}, w_pv {sel["w_pv"]}, w_pq {sel["w_pq"]}; so d equals fd2.')
# ratios with CI for b, c, e50 dva
table("Stage 1, ratio to fd2 of branch dVa with 95% paired-bootstrap interval, and share of scenarios where the state beats fd2", ["grid", "GENCO raw", "e50", "GENCO raw win%", "e50 win%"],
      [[SH(g), *[f'{one("eval2",g,s)["ratio_branch_dva_vs_fd2"]:.1f} [{one("eval2",g,s)["ratio_branch_dva_lo"]:.1f}, {one("eval2",g,s)["ratio_branch_dva_hi"]:.1f}]' for s in ("genco_raw", "e50")],
        *[f'{100*one("eval2",g,s)["win_frac_dva_vs_fd2"]:.1f}' for s in ("genco_raw", "e50")]] for g in G])
# sensitivity default norm
DN = [json.loads(l) for l in open(OUT / "results_defaultnorm_zeroshot.jsonl")]
dn = {(r["split"], r["grid"].replace("pglib_opf_", ""), r["state"]): r for r in DN}
table("Sensitivity: first zero-shot run, graphkit DEFAULT normalizer fit (baseMVA from AC Qg and slack Pg of 2 scenarios) vs the input-only fit used above. GENCO raw: branch dVa (deg) | |V| MAE (pu) | P residual (MW)",
      ["grid", "default fit", "input-only fit (primary)"],
      [[SH(g), f'{dn[("eval2_defaultnorm",g,"genco_raw")]["branch_dva_deg"]:.4f} | {dn[("eval2_defaultnorm",g,"genco_raw")]["vm_pu"]:.5f} | {dn[("eval2_defaultnorm",g,"genco_raw")]["p_res_mw"]:.3f}',
        f'{one("eval2",g,"genco_raw")["branch_dva_deg"]:.4f} | {one("eval2",g,"genco_raw")["vm_pu"]:.5f} | {one("eval2",g,"genco_raw")["p_res_mw"]:.3f}'] for g in G])
# supplementary DC-prior adapted
SS = ["fd2", "genco_raw", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va", "e50"]
for key, title, f, k in [metrics[0][:3] + (metrics[0][3],), metrics[2][:3] + (metrics[2][3],), metrics[3][:3] + (metrics[3][3],), metrics[4][:3] + (metrics[4][3],)]:
    kk = {"dva": "branch_dva", "vm": "vm", "p_res": "p_res", "q_res": "q_res"}[k]
    rows = []
    for g in G:
        row = [SH(g)]
        for s in SS:
            v = mean("supp_dcprior_adapted", g, s, key)
            row.append(f(v) + ("" if s == "fd2" else f' ({np.mean([r[f"ratio_{kk}_vs_fd2"] for r in idx[("supp_dcprior_adapted", g, s)]]):.2f}x)'))
        rows.append(row)
    table(f"Supplementary (existing label-free DC-prior adapted checkpoints, mean of 3 draws, n=1000, eval2): {title}; ratio to fd2 in brackets", ["grid", "fd2", "adapted raw", "b", "c", "e50"], rows)
rows = []
for s in ["genco_raw", "b_genco_vm_fd2_va", "c_fd2_vm_genco_va", "e50"]:
    c = []
    for key in ("branch_dva", "va", "vm", "p_res", "q_res"):
        n = 0
        for g in G:
            rs = idx[("supp_dcprior_adapted", g, s)]
            n += all(r[f"ratio_{key}_vs_fd2"] < 1 for r in rs) and len(rs) == 3
        c.append(f"{n}/9")
    rows.append([s, *c])
table("Supplementary DC-prior adapted: grids where the state beats fd2 on all 3 draws (point ratio < 1)", ["state", "branch dVa", "Va", "|V|", "P residual", "Q residual"], rows)
rows = []
for g in ("case14_ieee", "case30_ieee"):
    for s in SS:
        r = one("supp_fd2prior_adapted", g, s)
        rows.append([SH(g), s, f4(r["branch_dva_deg"]), f5(r["vm_pu"]), f3(r["p_res_mw"]), f3(r["q_res_mvar"]),
                     f'{r["ratio_branch_dva_vs_fd2"]:.2f} [{r["ratio_branch_dva_lo"]:.2f},{r["ratio_branch_dva_hi"]:.2f}]', f'{r["ratio_p_res_vs_fd2"]:.2f} [{r["ratio_p_res_lo"]:.2f},{r["ratio_p_res_hi"]:.2f}]'])
table("Supplementary: existing fd2-prior adapted checkpoints (lfp2_fd2_w1_lr3e-05_all, draw 0, dev grids only), eval2", ["grid", "state", "dVa", "|V|", "P res", "Q res", "dVa ratio to fd2 [CI]", "P res ratio [CI]"], rows)
# cost-matched classical
CS = ["dc", "fd1", "fd2", "fd3", "fd4", "nr1", "nr2", "nr3"]
for key, title, f in [("branch_dva_deg", "branch dVa (deg)", f4), ("p_res_mw", "AC active residual (MW)", f4)]:
    table(f"Cost-matched classical states on eval2: {title}", ["grid", *CS], [[SH(g), *[f(one("eval2_classical", g, s)[key]) for s in CS]] for g in G])
# timing
T = [json.loads(l) for l in open(OUT / "timing.jsonl")] if (OUT / "timing.jsonl").exists() else []
if T:
    cl = {r["grid"]: r for r in T if r["kind"] == "classical"}
    gp = {r["grid"]: r for r in T if r["kind"] == "gpu"}
    cp = {r["grid"]: r for r in T if r["kind"] == "cpu"}
    rows = []
    for g in G:
        gg = "pglib_opf_" + g
        rows.append([SH(g), f'{cl[gg]["ms_fd2_total"]:.2f}', f'{cl[gg]["ms_nr1_total"]:.2f}', f'{cl[gg]["ms_setup"]:.2f}', f'{cl[gg]["ms_dc"]:.2f}',
                     f'{gp[gg]["ms_forward_per_scenario"]:.2f}' if gg in gp else "-", f'{cp[gg]["ms_forward_per_scenario"]:.2f}' if gg in cp else "-",
                     f'{gp[gg]["s_ckpt_load"]:.2f}' if gg in gp else "-"])
    table("Wall time per scenario (ms; median; same shared box, load 45-60 on 24 cores, GPU shared). fd2 and nr1 totals include building the problem from the tables (setup) and the DC solve", ["grid", "fd2 total", "nr1 total", "of which setup", "of which DC solve", "GENCO forward, GPU, batch 64", "GENCO forward, CPU 4 threads, batch 64", "ckpt load (s, once)"], rows)
(OUT / "tables.md").write_text("\n".join(L) + "\n")
print("ok", len(L))
