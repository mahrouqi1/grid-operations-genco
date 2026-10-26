import sys
"""Verdict for candidate 12 (generator control): carved pieces regenerated through the LADDER generator vs the ladder-generated whole grids of candidate 9,
from saved per-scenario files only; protocol.json (+ protocol_ran.json, the pieces that ran) is the spec. UNAUDITED.
    python verdict.py -> verdict.json, table.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text()); R = json.loads((HERE / "protocol_ran.json").read_text())
OOS = HERE.parent / "2026-10-21-genco-voltage-inference-oos"; CTRL = HERE.parent / "2026-10-21-genco-voltage-inference-control"; FVS = HERE.parent / "2026-10-21-genco-voltage-inference-family-vs-size"
IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
ids = json.loads((HERE / "test_ids.json").read_text()); meta = R["ran"]          # pre-registered (before any comparison) list of pieces that ran
CONFIGS = ["f20_s0", "f20_s1", "f20_s2", "f40_s0", "f40_s1", "f40_s2"]; POOL = {"f20": CONFIGS[:3], "f40": CONFIGS[3:]}
frames = []
for g in meta:
    fi, fg = HERE / "per_grid" / f"interp_{g}.jsonl", HERE / "per_grid" / f"genco_{g}.jsonl"
    if not (fi.exists() and fg.exists()): sys.exit(f"INCOMPLETE: no output for {g}")
    d = pd.concat([pd.read_json(fi, lines=True), pd.read_json(fg, lines=True).drop(columns=["diag"])], ignore_index=True); need = set(ids[g]["test"])
    if not all(set(d[(d.method == m) & (d.config == c)].scenario) >= need and d[(d.method == m) & (d.config == c)].vm_mae_unmeas.notna().all() for m in ["G"] + IM for c in CONFIGS):
        sys.exit(f"INCOMPLETE: {g}")
    frames.append(d)
D = pd.concat(frames, ignore_index=True)


def test(grid, cfgs):
    x = D[(D.grid == grid) & D.config.isin(cfgs)]
    A = {m: x[x.method == m].pivot(index="scenario", columns="config", values="vm_mae_unmeas")[cfgs].loc[ids[grid]["test"]].to_numpy() for m in ["G"] + IM}
    exp = len(ids[grid]["test"])
    if any(a.shape != (exp, len(cfgs)) or not np.isfinite(a).all() for a in A.values()): sys.exit(f"INCOMPLETE: {grid} {cfgs}")
    def stat(idx):
        g = np.median(A["G"][idx]); bs = [np.median(A[m][idx]) for m in IM]
        return g / min(bs), IM[int(np.argmin(bs))], g, min(bs)
    n = exp; pt, best, g, bm = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n))[0] for _ in range(NB)]); lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n": n, "ratio": pt, "lo": lo, "hi": hi, "best_interp": best, "genco_median": g, "best_interp_median": bm, "win": bool(pt < 1 and hi < 1)}


res = {c: {g: test(g, cf) for g in meta} for c, cf in POOL.items()}
fam = {g: P["pieces"][g]["family"] for g in meta}; grp = {g: P["pieces"][g]["group"] for g in meta}; size = {g: P["pieces"][g]["n_bus"] for g in meta}
ov = json.loads((OOS / "verdict.json").read_text())
whole_unseen = {c: [ov["results"][c][g]["win"] for g in ov["results"][c] if g in set(json.loads((OOS / "protocol.json").read_text())["sets"]["primary"])] for c in ("f40", "f20")}
whole_seen = {c: [ov["results"][c][g]["win"] for g in ov["results"][c] if g in set(json.loads((OOS / "protocol.json").read_text())["sets"]["seen"])] for c in ("f40", "f20")}
dk = {"seen": json.loads((CTRL / "verdict.json").read_text())["results"], "unseen": json.loads((FVS / "verdict.json").read_text())["results"]}
alpha = P["alpha"]


def wins(c, pieces): return [res[c][g]["win"] for g in pieces]
def fish(a, b, alt):   # a, b = lists of bool; alt relative to a (rows: a, b)
    return float(fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]], alternative=alt)[1])
U = [g for g in meta if grp[g] == "unseen"]; S = [g for g in meta if grp[g] == "seen"]
out = {"ran": meta, "n_unseen_pieces": len(U), "n_seen_pieces": len(S)}
for c in ("f40", "f20"):
    out["unseen_pieces_vs_whole_" + c] = {"pieces_wins": sum(wins(c, U)), "pieces_n": len(U), "whole_wins": sum(whole_unseen[c]), "whole_n": len(whole_unseen[c]),
                                         "fisher_one_sided_p_pieces_win_more": fish(wins(c, U), whole_unseen[c], "greater") if U else None}
    out["seen_pieces_vs_whole_" + c] = {"pieces_wins": sum(wins(c, S)), "pieces_n": len(S), "whole_wins": sum(whole_seen[c]), "whole_n": len(whole_seen[c]),
                                       "fisher_one_sided_p_pieces_win_more": fish(wins(c, S), whole_seen[c], "greater") if S else None}
    out["seen_vs_unseen_pieces_" + c] = {"seen_wins": sum(wins(c, S)), "seen_n": len(S), "unseen_wins": sum(wins(c, U)), "unseen_n": len(U),
                                        "fisher_two_sided_p": float(fisher_exact([[sum(wins(c, S)), len(S) - sum(wins(c, S))], [sum(wins(c, U)), len(U) - sum(wins(c, U))]])[1]) if S and U else None}
    # generator effect on the SAME pieces (descriptive): win flag under datakit (candidates 10/11) vs ladder
    pr = {"both_win": 0, "datakit_only": 0, "ladder_only": 0, "neither": 0}
    for g in meta:
        k = P["pieces"][g]["source"]; dkw = dk[P["pieces"][g]["group"]][c][k if k.startswith("carved_") else "carved_" + k]["win"] if (P["pieces"][g]["source"] in dk[P["pieces"][g]["group"]][c] or "carved_" + P["pieces"][g]["source"] in dk[P["pieces"][g]["group"]][c]) else None
        if dkw is None: sys.exit(f"INCOMPLETE: no datakit-arm result for a piece (audit R2-F5)")
        lw = res[c][g]["win"]; pr["both_win" if dkw and lw else "datakit_only" if dkw else "ladder_only" if lw else "neither"] += 1
    out["generator_effect_same_pieces_" + c] = pr
    out["median_ratio_" + c] = {"unseen_pieces_ladder": float(np.median([res[c][g]["ratio"] for g in U])) if U else None, "seen_pieces_ladder": float(np.median([res[c][g]["ratio"] for g in S])) if S else None,
                               "unseen_whole_ladder": float(np.median([ov["results"][c][g]["ratio"] for g in ov["results"][c] if g in set(json.loads((OOS / "protocol.json").read_text())["sets"]["primary"])]))}
p = out["unseen_pieces_vs_whole_f40"]
out["primary"] = {**p, "verdict": "SIZE CONTRAST PERSISTS within the ladder generator" if p["fisher_one_sided_p_pieces_win_more"] is not None and p["fisher_one_sided_p_pieces_win_more"] < alpha else "NOT SHOWN within the ladder generator"}
out["results"] = res
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
cell = lambda r: f"{r['genco_median']:.4f}/{r['best_interp_median']:.4f} ({r['best_interp']}) {r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{' W' if r['win'] else ''}"
L = ["| group | family | piece | buses | n scen | 20% | 40% |", "|---|---|---|---|---|---|---|"]
for g in meta: L.append(f"| {grp[g]} | {fam[g]} | {g.replace('pglib_opf_carved_','')} | {size[g]} | {res['f40'][g]['n']} | {cell(res['f20'][g])} | {cell(res['f40'][g])} |")
L += ["", "| comparison (ladder generator throughout) | 40% | 20% |", "|---|---|---|"]
def r_(k, c):
    e = out[k + "_" + c]; a = [x for x in e if x.endswith("wins")]
    pv = e.get("fisher_one_sided_p_pieces_win_more", e.get("fisher_two_sided_p")); n1, n2 = [x for x in e if x.endswith("_n")]
    return f"{e[a[0]]}/{e[n1]} vs {e[a[1]]}/{e[n2]} (p={pv:.4f})"
for k, name in (("unseen_pieces_vs_whole", "unseen: pieces vs whole grids (one-sided, pieces win more)"), ("seen_pieces_vs_whole", "seen: pieces vs whole grids (one-sided)"), ("seen_vs_unseen_pieces", "pieces: seen vs unseen (two-sided)")):
    L.append(f"| {name} | {r_(k, 'f40')} | {r_(k, 'f20')} |")
L += ["", "Same pieces, datakit-generated (candidates 10/11) vs ladder-generated win flags: " + json.dumps({c: out["generator_effect_same_pieces_" + c] for c in ("f40", "f20")})]
(HERE / "table.md").write_text("\n".join(L) + "\n")
print("PRIMARY", out["primary"]["verdict"], f"unseen pieces {p['pieces_wins']}/{p['pieces_n']} vs whole {p['whole_wins']}/{p['whole_n']} p={p['fisher_one_sided_p_pieces_win_more']:.4f}" if U else "")
