"""Computes the pre-registered success test from saved rows; writes verdict.json and table.md. UNAUDITED.
Rows: results.jsonl (this study) + results/2026-10-10-warmup-adaptation/results.jsonl (reused B4 rows: wu100, n=100, 14/30/57/118)."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
B4 = HERE.parent / "2026-10-10-warmup-adaptation/results.jsonl"
proto = json.loads((HERE / "protocol.json").read_text())
G = proto["grids"]; REUSE = proto["grid_data_status"]["reuse_B4_rows"]["grids"]; SKIP = proto["grid_data_status"]["skipped_no_data"]
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "wu100L"      # wu100L = AC-labelled (primary), wu100 = as-run B4 (DC-prior targets, label-free)
import re
nb = lambda g: re.search(r'case(\d+)', g).group(1)
rows = {}
def add(path, only=None):
    if not path.exists(): return
    for l in path.read_text().splitlines():
        r = json.loads(l)
        if "error" in r or r.get("tag") != TAG[:5] + ("L" if TAG.endswith("L") else "") or r.get("n") != 100 or (only and r["grid"] not in only): continue
        rows[(r["grid"], r["split_seed"])] = r | {"source": path.parent.name}
if TAG == "wu100": add(B4, set(REUSE))
add(HERE / "results.jsonl")
out = {}; lines = ["| grid | buses | source | draws | dVa DC (deg) | dVa fd2 (deg) | dVa GENCO (deg) | ratio to DC (mean) | draws <DC | ratio to fd2 (mean) | |V| ratio DC | p-res ratio DC | verdict |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for g in G:
    rs = [rows[(g, s)] for s in (0, 1, 2) if (g, s) in rows]
    if not rs:
        out[g] = {"status": "no data: " + SKIP.get(g, "no completed run"), "beats_dc": False}
        lines.append(f"| {g[10:]} | {nb(g)} | - | 0 | | | | | | | | | NOT RUN: {SKIP.get(g, 'failed/none')} |"); continue
    o = [r["ours"] for r in rs]
    rdc = [x["genco_raw"]["branch_dva_deg"] / x["dc"]["branch_dva_deg"] for x in o]
    rfd = [x["genco_raw"]["branch_dva_deg"] / x["fd2"]["branch_dva_deg"] for x in o]
    rv = [x["genco_raw"]["vm_pu"] / x["dc"]["vm_pu"] for x in o]
    rp = [x["genco_raw"]["p_res_mw"] / x["dc"]["p_res_mw"] for x in o]
    m = st.mean(rdc); full = len(rs) == 3
    out[g] = {"status": "ok" if full else f"only {len(rs)} draws", "draws": len(rs), "ratio_dc": m, "ratio_dc_per_draw": rdc, "draws_below_dc": sum(r < 1 for r in rdc),
              "ratio_fd2": st.mean(rfd), "draws_below_fd2": sum(r < 1 for r in rfd), "vm_ratio_dc": st.mean(rv), "pres_ratio_dc": st.mean(rp),
              "beats_dc": m < 1, "beats_dc_2of3": sum(r < 1 for r in rdc) >= 2, "beats_fd2": st.mean(rfd) < 1, "source": rs[0]["source"],
              "dva_dc": st.mean(x["dc"]["branch_dva_deg"] for x in o), "dva_fd2": st.mean(x["fd2"]["branch_dva_deg"] for x in o), "dva_genco": st.mean(x["genco_raw"]["branch_dva_deg"] for x in o)}
    d = out[g]
    lines.append(f"| {g[10:]} | {nb(g)} | {d['source'][:10]} | {len(rs)} | {d['dva_dc']:.3f} | {d['dva_fd2']:.4f} | {d['dva_genco']:.3f} | {m:.2f} | {d['draws_below_dc']}/{len(rs)} | {d['ratio_fd2']:.1f} | {d['vm_ratio_dc']:.2f} | {d['pres_ratio_dc']:.2f} | {'beats DC' if d['beats_dc'] else 'loses to DC'} |")
n_beat = sum(d["beats_dc"] for d in out.values())
res = {"variant": TAG, "success_test": ">= 10 of 15 grids with draw-mean branch dVa ratio to DC < 1 at n=100", "n_beating_dc": n_beat, "of": len(G),
       "n_evaluated": sum("ratio_dc" in d for d in out.values()), "n_beating_dc_2of3": sum(d.get("beats_dc_2of3", False) for d in out.values()),
       "n_beating_fd2": sum(d.get("beats_fd2", False) for d in out.values()), "PASS": n_beat >= 10, "per_grid": out}
(HERE / f"verdict_{TAG}.json").write_text(json.dumps(res, indent=1)); (HERE / f"table_{TAG}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nGENCO beats DC (draw-mean dVa) on {n_beat} of {len(G)} (evaluated {res['n_evaluated']}); 2-of-3 rule: {res['n_beating_dc_2of3']}; beats fd2: {res['n_beating_fd2']}; PASS={res['PASS']}")
