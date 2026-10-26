"""Verdict for candidate 11 (family vs size: carved UNSEEN-family subgrids vs the carved SEEN-family subgrids), from saved per-scenario files
only; protocol.json is the spec. Seen-family results are reused from ../2026-10-21-genco-voltage-inference-control/verdict.json, not rerun. UNAUDITED.
    python verdict.py -> verdict.json, table.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, ks_2samp

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
CTRL = HERE.parent / "2026-10-21-genco-voltage-inference-control"
IM = P["interpolators"]; NB = P["bootstrap"]["n"]; rng = np.random.default_rng(P["bootstrap"]["seed"])
ids = json.loads((HERE / "test_ids.json").read_text()); meta = P["subgrids"]
POOL = {"f10": ["f10_s0", "f10_s1", "f10_s2"], "f20": ["f20_s0", "f20_s1", "f20_s2"], "f40": ["f40_s0", "f40_s1", "f40_s2"],
        "n20": ["n20_s0", "n20_s1", "n20_s2"], "n40": ["n40_s0", "n40_s1", "n40_s2"]}
frames, ran, skipped = [], [], {}
for g in meta:
    fi, fg = HERE / "per_grid" / f"interp_{g}.jsonl", HERE / "per_grid" / f"genco_{g}.jsonl"
    if g not in ids or not (fi.exists() and fg.exists()):
        skipped[g] = "no output"; continue
    d = pd.concat([pd.read_json(fi, lines=True), pd.read_json(fg, lines=True).drop(columns=["diag"])], ignore_index=True); need = set(ids[g]["test"])
    if not all(set(d[(d.method == m) & (d.config == c)].scenario) >= need and d[(d.method == m) & (d.config == c)].vm_mae_unmeas.notna().all()
               for m in ["G"] + IM for cs in POOL.values() for c in cs):
        skipped[g] = "incomplete"; continue
    frames.append(d); ran.append(g)
# audit R1-F4: no skipped subgrids; the pre-registered list (protocol 'subgrids') must all have run
if skipped or sorted(ran) != sorted(meta):
    sys.exit(f"INCOMPLETE: skipped={skipped} ran={len(ran)} pre-registered={len(meta)}")
D = pd.concat(frames, ignore_index=True)


def test(grid, cfgs):
    x = D[(D.grid == grid) & D.config.isin(cfgs)]
    A = {m: x[x.method == m].pivot(index="scenario", columns="config", values="vm_mae_unmeas")[cfgs].loc[ids[grid]["test"]].to_numpy() for m in ["G"] + IM}
    def stat(idx):
        g = np.median(A["G"][idx]); bs = [np.median(A[m][idx]) for m in IM]
        return g / min(bs), IM[int(np.argmin(bs))], g, min(bs)
    exp = len(ids[grid]["test"])   # audit R1-F4: n must equal the pre-registered test-scenario count for every method x config
    if any(a.shape != (exp, len(cfgs)) or not np.isfinite(a).all() for a in A.values()): sys.exit(f"INCOMPLETE: {grid} {cfgs}")
    n = A["G"].shape[0]; pt, best, g, bm = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n))[0] for _ in range(NB)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n": n, "ratio": pt, "lo": lo, "hi": hi, "best_interp": best, "genco_median": g, "best_interp_median": bm, "win": bool(pt < 1 and hi < 1)}


res = {c: {g: test(g, cf) for g in ran} for c, cf in POOL.items()}
sv = json.loads((CTRL / "verdict.json").read_text()); sres = sv["results"]; ssz = sv["carved_sizes"]; seen = list(ssz)
fam = {g: meta[g]["family"] for g in ran}; size = {g: meta[g]["n_bus"] for g in ran}


def fisher_less(w_u, n_u, w_s, n_s):
    return float(fisher_exact([[w_u, n_u - w_u], [w_s, n_s - w_s]], alternative="less")[1])


def cmp(cond, gu, gs):
    wu = sum(res[cond][g]["win"] for g in gu); ws = sum(sres[cond][g]["win"] for g in gs)
    return {"unseen_wins": wu, "unseen_n": len(gu), "seen_wins": ws, "seen_n": len(gs), "fisher_one_sided_p_unseen_lower": fisher_less(wu, len(gu), ws, len(gs)) if gu and gs else None,
            "unseen_median_ratio": float(np.median([res[cond][g]["ratio"] for g in gu])) if gu else None, "seen_median_ratio": float(np.median([sres[cond][g]["ratio"] for g in gs])) if gs else None}


alpha = P["alpha"]
out = {"ran": ran, "skipped": skipped, "unseen_sizes": size, "seen_sizes": ssz}
out["primary"] = {"cond": "f40", **cmp("f40", ran, seen)}; out["primary"]["verdict"] = "FAMILY MATTERS (unseen pieces lose)" if out["primary"]["fisher_one_sided_p_unseen_lower"] < alpha else "NOT SHOWN (unseen pieces are not significantly lower)"
out["secondary_f20"] = {**cmp("f20", ran, seen)}; out["secondary_f20"]["significant"] = out["secondary_f20"]["fisher_one_sided_p_unseen_lower"] < alpha
out["descriptive_n40"] = cmp("n40", ran, seen); out["descriptive_f10"] = cmp("f10", ran, seen)
out["per_family"] = {f: {c: cmp(c, [g for g in ran if fam[g] == f], seen) for c in ("f40", "f20")} for f in sorted(set(fam.values()))}
# size distributions and size-matched subset
su, ss = np.array([size[g] for g in ran]), np.array([ssz[g] for g in seen])
ks = ks_2samp(su, ss)
out["size_distributions"] = {"unseen": {"n": len(su), "min": int(su.min()), "median": float(np.median(su)), "max": int(su.max())}, "seen": {"n": len(ss), "min": int(ss.min()), "median": float(np.median(ss)), "max": int(ss.max())},
                              "ks_p": float(ks.pvalue), "median_ratio": float(np.median(su) / np.median(ss))}
differ = bool(ks.pvalue < P["size_differ_ks_p"] or abs(np.log(np.median(su) / np.median(ss))) > np.log(1 + P["size_differ_median_frac"]))
out["size_distributions"]["differ_by_rule"] = differ
# greedy nearest-neighbour matching in log(bus count), without replacement, caliper |log ratio| <= P['caliper']
pairs, used = [], set()
for g in sorted(ran, key=lambda g: size[g]):
    best = min((h for h in seen if h not in used), key=lambda h: abs(np.log(ssz[h] / size[g])), default=None)
    if best is not None and abs(np.log(ssz[best] / size[g])) <= P["caliper_log"]: pairs.append((g, best)); used.add(best)
mu, ms = [p[0] for p in pairs], [p[1] for p in pairs]
out["size_matched"] = {"pairs": [[a, b, size[a], ssz[b]] for a, b in pairs], **{c: cmp(c, mu, ms) for c in ("f40", "f20")}}
out["results"] = res
(HERE / "verdict.json").write_text(json.dumps(out, indent=1))
cell = lambda r: f"{r['genco_median']:.4f}/{r['best_interp_median']:.4f} ({r['best_interp']}) {r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{' W' if r['win'] else ''}"
L = ["| family | subgrid | tier | buses | n scen | 10% | 20% | 40% | 20% + noise | 40% + noise |", "|" + "---|" * 10]
for g in ran:
    L.append(f"| {fam[g]} | {g.replace('carved_','')} | {meta[g]['tier']} | {size[g]} | {res['f40'][g]['n']} | " + " | ".join(cell(res[c][g]) for c in ("f10", "f20", "f40", "n20", "n40")) + " |")
L += ["", "Unseen vs seen carved subgrids (seen results reused from the control run):", "", "| comparison | unseen wins | seen wins | one-sided Fisher p (unseen lower) | median ratio unseen / seen |", "|---|---|---|---|---|"]
def row(name, e): return f"| {name} | {e['unseen_wins']}/{e['unseen_n']} | {e['seen_wins']}/{e['seen_n']} | {e['fisher_one_sided_p_unseen_lower']:.4f} | {e['unseen_median_ratio']:.2f} / {e['seen_median_ratio']:.2f} |"
L += [row("40% (primary)", out["primary"]), row("20%", out["secondary_f20"]), row("40% + noise", out["descriptive_n40"]), row("10%", out["descriptive_f10"])]
for f, d in out["per_family"].items(): L += [row(f"{f} 40%", d["f40"]), row(f"{f} 20%", d["f20"])]
L += [row(f"size-matched ({len(pairs)} pairs) 40%", out["size_matched"]["f40"]), row(f"size-matched ({len(pairs)} pairs) 20%", out["size_matched"]["f20"])]
(HERE / "table.md").write_text("\n".join(L) + "\n")
p = out["primary"]; print("PRIMARY", p["verdict"], f"unseen {p['unseen_wins']}/{p['unseen_n']} vs seen {p['seen_wins']}/{p['seen_n']} p={p['fisher_one_sided_p_unseen_lower']:.4f}", "| f20 p", round(out["secondary_f20"]["fisher_one_sided_p_unseen_lower"], 4), "| sizes differ", differ, "matched pairs", len(pairs), "| skipped", skipped)
