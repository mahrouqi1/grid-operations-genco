"""Tables, counts and ``summary.json`` for ``results/2026-10-08-adapt-fresh-eval`` (see ``fresh_eval``).

Every count is computed from the grouped rows of ``results.jsonl`` (fresh) and of the original studies'
results.jsonl (old); ``verify()`` recomputes the headline counts a second, independent way.
"""
from __future__ import annotations

import json

import numpy as np

from genco_challenge import fewshot as fs
from genco_challenge import fresh_eval as fe

LF = "lfc_w1_lr3e-05_all_slackvm"
SHORT = lambda g: g.replace("pglib_opf_", "").replace("_ieee_dtc", "").replace("_ieee_rts", "").replace("_ieee", "").replace("_epri", "").replace("_pegase", "")  # noqa: E731
# grids whose EVAL numbers shaped the recipe (audit F1/F2). Label-free: case14/case30 chose the hyperparameters
# (validation only) AND v2 compared the family on all 9 eval sets. Supervised frozen trunk: variants ranked on
# case14/case30 eval residuals (v1 'what was tried'); freezeN: more capacity chosen against eval plateaus on
# case73/89/39/57 (v2 report).
SHAPED = {"freeze": ["case14_ieee", "case30_ieee"], "freezeN": ["case14_ieee", "case30_ieee", "case39_epri",
                                                                  "case57_ieee", "case73_ieee_rts", "case89_pegase"]}
# SHAPED holds short names; the tables iterate full 'pglib_opf_' grid names (audit genco-adapt R2-F1)
SHAPED = {v: ["pglib_opf_" + g for g in gs] for v, gs in SHAPED.items()}


# draws per cell, counted from results.jsonl (audit genco-adapt R2-F2): seed variants _s1/_s2 exist only for
# frozen trunk n = 10, 100 and freezeN; frozen trunk n = 1000, GENCO recipe and scratch are single runs.
DRAWS = {"freeze": "mean over 3 draws at n = 10 and 100, a SINGLE draw at n = 1000",
         "freezeN": "mean over 3 draws",
         "genco": "a SINGLE draw per cell",
         "scratch": "a SINGLE draw per cell"}


def r3(x):
    return f"{x:.2f}"


def cell(o, n_):
    if o is None:
        return "-"
    return " / ".join(f"{o[q]['ratio']:.2f}" for q in ("res", "dva", "vm"))


def build() -> None:
    new_rows, old_v2, old_lf = fe.load_new(), fe.load_old("v2"), fe.load_old("labelfree-clean")
    assert not [r for r in new_rows if "error" in r]
    gn, go_v2, go_lf = fe.group(new_rows), fe.group(old_v2), fe.group(old_lf)
    # new rows of the label-free study (study tag) vs old
    gn_lf = fe.group([r for r in new_rows if r["study"] == "labelfree-clean"])
    gn_v2 = fe.group([r for r in new_rows if r["study"] == "v2"])
    out: dict = {"n_rows_fresh": len(new_rows), "n_rows_planned": len(fe.plan())}
    # ---------------------------------------------------------------- label-free headline
    h_old, h_new = fe.lf_headline(go_lf), fe.lf_headline(gn_lf)
    out["labelfree_headline"] = {"old_eval_reused": {k: v for k, v in h_old.items() if k != "per_grid"} | {"per_grid": h_old["per_grid"]},
                                 "fresh_eval": {k: v for k, v in h_new.items() if k != "per_grid"} | {"per_grid": h_new["per_grid"]}}
    L = ["| grid | role | n | n_actual | scored scenarios old / fresh | residual old -> fresh | branch dVa old -> fresh | "
         "|V| old -> fresh | draws beating DC on res, dVa (old -> fresh) |", "|---|---|---:|---:|---|---|---|---|---|"]
    for g in fs.GRIDS:
        for n in (1000, 100):
            ko = (g, LF, "finetune", n)
            o, w = go_lf.get(ko), gn_lf.get(ko)
            if not o or not w:
                continue
            L.append(f"| {SHORT(g)} | {'dev' if g in fe.DEV else 'non-dev'} | {n} | {', '.join(w['n_actual'])} "
                     f"| {o['n_test'][0]} / {w['n_test'][0]} | "
                     + " | ".join(f"{o[q]['ratio']:.2f} -> {w[q]['ratio']:.2f}" for q in ("res", "dva", "vm"))
                     + f" | {o['res']['beats']}/{o['draws']}, {o['dva']['beats']}/{o['draws']} -> "
                       f"{w['res']['beats']}/{w['draws']}, {w['dva']['beats']}/{w['draws']} |")
    out["table_labelfree"] = "\n".join(L)
    # n=100 non-dev: both-metric wins
    out["labelfree_n100"] = {tag: [g for g in fe.HELD if (g, LF, "finetune", 100) in gg
                                   and gg[(g, LF, "finetune", 100)]["res"]["ratio"] < 1 and gg[(g, LF, "finetune", 100)]["dva"]["ratio"] < 1]
                             for tag, gg in (("old", go_lf), ("fresh", gn_lf))}
    # ---------------------------------------------------------------- supervised
    sup = {}
    for v in ("freeze", "freezeN"):
        so, sn = fe.sup_wins(go_v2, v), fe.sup_wins(gn_v2, v)
        sup[v] = {}
        for n in (10, 100, 1000):
            sup[v][n] = {}
            for tag, s in (("old", so), ("fresh", sn)):
                sup[v][n][tag] = {"res_beats": sorted(g for (g, nn), x in s.items() if nn == n and x["beats_res"]),
                                  "all3_beats": sorted(g for (g, nn), x in s.items() if nn == n and x["beats_all"]),
                                  "grids": sorted(g for (g, nn) in s if nn == n)}
    out["supervised"] = sup
    out["freezeN_n100_case162_draws"] = {t: {q: gg[("pglib_opf_case162_ieee_dtc", "freezeN", "finetune", 100)][q]["beats"] for q in ("res", "dva", "vm")}
                                         for t, gg in (("old", go_v2), ("fresh", gn_v2))}
    T = []
    for v, title in (("freeze", "frozen trunk (46k)"), ("freezeN", "freezeN (66k)"), ("genco", "GENCO recipe fine-tune"),
                     ("scratch", "from scratch")):
        mode = "scratch" if v == "scratch" else "finetune"
        vv = "genco" if v == "scratch" else v
        T.append(f"\n**{title}**: ratio to DC (residual / branch dVa / |V|), old eval -> fresh eval; {DRAWS[v]}. "
                 "`*` = grid shaped this recipe.\n")
        T.append("| grid | n = 10 | n = 100 | n = 1000 |\n|---|---|---|---|")
        for g in fs.GRIDS:
            cells = []
            for n in (10, 100, 1000):
                k = (g, vv, mode, n)
                o, w = go_v2.get(k), gn_v2.get(k)
                cells.append("-" if not o or not w else f"{cell(o, n)} -> {cell(w, n)}")
            if all(c == "-" for c in cells):
                continue
            star = "*" if g in SHAPED.get(v, fe.DEV if v in ("genco", "scratch") else []) else ""
            T.append(f"| {SHORT(g)}{star} | " + " | ".join(cells) + " |")
    out["table_supervised"] = "\n".join(T)
    # reference rows
    R = ["| grid | model | residual ratio old -> fresh | branch dVa ratio old -> fresh | |V| ratio old -> fresh |", "|---|---|---|---|---|"]
    for g in fs.GRIDS:
        for mode in ("zeroshot", "ft118_1000"):
            for fam_ in ("ref", "ref_slackvm"):
                k = (g, fam_, mode, 0)
                if k in go_v2 and k in gn_v2:
                    o, w = go_v2[k], gn_v2[k]
                    R.append(f"| {SHORT(g)} | {mode}{' (slack |V| fix)' if fam_ == 'ref_slackvm' else ''} | "
                             + " | ".join(f"{o[q]['ratio']:.2f} -> {w[q]['ratio']:.2f}" for q in ("res", "dva", "vm")) + " |")
    out["table_reference"] = "\n".join(R)
    # ---------------------------------------------------------------- DC and flat baselines
    B = ["| grid | scored scenarios old / fresh | DC residual MW old -> fresh | DC branch dVa deg old -> fresh | flat branch dVa deg old -> fresh "
         "| DC |V| MAE old -> fresh | flat |V| MAE old -> fresh |", "|---|---|---|---|---|---|---|"]
    base = {}
    for g in fs.GRIDS:
        ro = next(r for r in old_lf + old_v2 if r["grid"] == g)
        rn = next(r for r in new_rows if r["grid"] == g)
        mo, mn = fe.metrics(ro), fe.metrics(rn)
        base[g] = {"n_test_old": ro["n_test_scenarios"], "n_test_fresh": rn["n_test_scenarios"],
                   **{f"{q}_old": mo[q] for q in ("res_dc", "dva_dc", "dva_flat", "vm_dc", "vm_flat")},
                   **{f"{q}_fresh": mn[q] for q in ("res_dc", "dva_dc", "dva_flat", "vm_dc", "vm_flat")}}
        # DC / flat must be identical across all rows of one grid within a study (same scenarios)
        for rs in (old_lf + old_v2, new_rows):
            vals = {round(fe.metrics(r)["res_dc"], 9) for r in rs if r["grid"] == g}
            assert len(vals) == 1, (g, vals)
        B.append(f"| {SHORT(g)} | {ro['n_test_scenarios']} / {rn['n_test_scenarios']} | "
                 f"{mo['res_dc']:.3f} -> {mn['res_dc']:.3f} | {mo['dva_dc']:.3f} -> {mn['dva_dc']:.3f} | "
                 f"{mo['dva_flat']:.3f} -> {mn['dva_flat']:.3f} | {mo['vm_dc']:.4f} -> {mn['vm_dc']:.4f} | "
                 f"{mo['vm_flat']:.4f} -> {mn['vm_flat']:.4f} |")
    out["baselines"] = base
    out["table_baselines"] = "\n".join(B)
    # v2 provisional label-free family on the fresh sets (strong prior, n = 1000)
    P = {}
    for tag, gg in (("old", go_v2), ("fresh", gn_v2)):
        ks = {g: gg.get((g, "physdcW1_slackvm", "finetune", 1000)) for g in fe.HELD}
        P[tag] = {"grids": [g for g, o in ks.items() if o], "both_beats": [g for g, o in ks.items() if o and o["res"]["ratio"] < 1 and o["dva"]["ratio"] < 1]}
    out["v2_provisional_physdcW1_n1000"] = P
    (fe.OUT / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    (fe.OUT / "tables.md").write_text("\n\n".join(["### Label-free (chosen config)", out["table_labelfree"], "### Supervised and reference",
                                                    out["table_supervised"], out["table_reference"], "### DC and flat",
                                                    out["table_baselines"]]))
    verify(out)
    print(json.dumps({"labelfree": out["labelfree_headline"]["old_eval_reused"] | {"per_grid": None},
                      "labelfree_fresh": out["labelfree_headline"]["fresh_eval"] | {"per_grid": None},
                      "n100": out["labelfree_n100"], "sup": sup, "v2prov": P}, indent=1, default=str))


def verify(out: dict) -> None:
    """Independent recomputation of the label-free headline counts straight from the jsonl files."""
    def counts(rows):
        res = {}
        for r in rows:
            if r["grid"] in fe.HELD and r["n"] == 1000 and fe.LFC_VARIANT.match(r["variant"]):
                res.setdefault(r["grid"], []).append(r)
        wins = {"res": 0, "dva": 0, "both": 0, "vm": 0}
        for g, rs in res.items():
            m = [fe.metrics(r) for r in rs]
            a = np.mean([x["res"] / x["res_dc"] for x in m]) < 1
            b = np.mean([x["dva"] / x["dva_dc"] for x in m]) < 1
            c = np.mean([x["vm"] / x["vm_dc"] for x in m]) < 1
            wins["res"] += a
            wins["dva"] += b
            wins["both"] += a and b
            wins["vm"] += c
        return {k: int(v) for k, v in wins.items()} | {"grids": len(res)}
    new = [r for r in fs._latest_ok(fe.OUT / "results.jsonl").values() if r["study"] == "labelfree-clean"]
    old = [r for r in fs._latest_ok(fe.LFC).values() if fe.LFC_VARIANT.match(r["variant"])]
    for tag, rows, h in (("old", old, out["labelfree_headline"]["old_eval_reused"]), ("fresh", new, out["labelfree_headline"]["fresh_eval"])):
        c = counts(rows)
        assert c["res"] == len(h["res_beats"]) and c["dva"] == len(h["dva_beats"]) and c["both"] == len(h["both_beats"]) \
            and c["vm"] == len(h["vm_beats"]), (tag, c, h)
        print("verified", tag, c)
