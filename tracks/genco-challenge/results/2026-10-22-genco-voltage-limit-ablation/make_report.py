"""Builds report.md from verdict.json + per_case_*.csv. UNAUDITED."""
import json, datetime
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
v = json.loads((H / "verdict.json").read_text()); P = json.loads((H / "protocol.json").read_text())
ARMS = ["A0", "A1", "A2", "A3"]; NAME = {"A0": "A0 released", "A1": "A1 limits constant", "A2": "A2 costs zeroed", "A3": "A3 both"}
L = ["# GENCO |V| inference: generator-limit / cost ablation (2026-10-22)", "",
     f"> **UNAUDITED.** Pre-registered control (`protocol.json`, sha256 in `protocol.sha256`, written 2026-10-22 00:52 EDT before any inference; amendments 1 and 2 written 01:16 and 01:37 EDT, before any whole grid ran). Generated {v['generated']}. CPU only.", ""]
def tbl(name, d, ranlist, total):
    o = [f"| arm | wins on the {len(ranlist)} grids that ran (of {total}) | wins lost vs A0 | wins gained vs A0 | median ratio | median shift factor (arm/A0) |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        x = d[a]
        o.append(f"| {NAME[a]} | {x['wins']} | {x.get('win_to_loss','-')} | {x.get('loss_to_win','-')} | {x['median_ratio']:.3f} | {x.get('median_ratio_shift_factor', 1.0):.3f} |" if x['median_ratio'] is not None else f"| {NAME[a]} | 0 | - | - | - | - |")
    return o
bs = v["by_set"]; sz = v["set_sizes"]
L += ["## Bottom line", ""]
T = v["threshold"]
E_ = {s: len(v["ran"][s]) for s in v["ran"]}
L.append(f"Threshold (pre-registered): crossed if any ablated arm loses >= 3 wins vs the released arm on the pieces, or on the whole grids pooled. **THRESHOLD_CROSSED = {T['THRESHOLD_CROSSED']}.**")
for a, x in T["per_arm"].items(): L.append(f"- {NAME[a]}: wins lost on pieces {x['delta_pieces']}, on pooled whole grids {x['delta_whole_pooled']} -> {'CROSSED' if x['crossed'] else 'not crossed'}")
L += ["", f"Coverage: pieces {E_['pieces']}/{sz['pieces']}, seen whole grids {E_['seen_whole']}/{sz['seen_whole']}, unseen whole grids {E_['unseen_whole']}/{sz['unseen_whole']}. Inconclusive flags (fewer than half run): {T['inconclusive_less_than_half_ran']}.", ""]
L += ["## Win counts per arm", "", "Win = GENCO median error below the best of five interpolators and the upper end of the 95% scenario-bootstrap CI of the ratio below 1, 40% measured, 3 mask seeds pooled. Counts are on the grids that ran (same grids for every arm).", ""]
for s, title in (("pieces", "Carved pieces (datakit-generated)"), ("seen_whole", "Seen-family whole grids"), ("unseen_whole", "Unseen-family whole grids")):
    L += [f"### {title}", ""] + tbl(s, bs[s], v["ran"][s], sz[s]) + [f"", f"Original study wins on these same grids: {bs[s]['A0'].get('original_study_wins_on_these_grids', 0)} (recomputed A0: {bs[s]['A0']['wins']})." if v['ran'][s] else "Nothing ran.", ""]
L += ["### Whole grids pooled", ""] + tbl("w", v["whole_pooled"], v["ran"]["seen_whole"] + v["ran"]["unseen_whole"], sz["seen_whole"] + sz["unseen_whole"]) + [""]
L += ["## Not run", ""]
for s in v["skipped"]:
    if v["skipped"][s]:
        L.append(f"- **{s}** ({len(v['skipped'][s])}): " + "; ".join(f"{g} ({w})" for g, w in v["skipped"][s].items()))
L += ["", "## Per-case results", "", "`per_case_pieces.csv`, `per_case_seen_whole.csv`, `per_case_unseen_whole.csv`: one row per grid per arm (ratio, CI, win).", ""]
for s in ("pieces", "seen_whole", "unseen_whole"):
    f = H / f"per_case_{s}.csv"
    if f.exists() and f.stat().st_size > 80:
        d = pd.read_csv(f); p = d.pivot(index="grid", columns="arm", values="ratio"); w = d.pivot(index="grid", columns="arm", values="win"); b = d.drop_duplicates("grid").set_index("grid").buses
        L += [f"### {s}", "", "| grid | buses | ratio A0 | A1 | A2 | A3 | wins A0/A1/A2/A3 |", "|---|---|---|---|---|---|---|"]
        for g in sorted(p.index, key=lambda g: b[g]):
            L.append(f"| {g} | {b[g]} | " + " | ".join(f"{p.loc[g,a]:.3f}" for a in ARMS) + " | " + "/".join(str(int(w.loc[g, a])) for a in ARMS) + " |")
        L.append("")
(H / "report_tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L)[:3000])
