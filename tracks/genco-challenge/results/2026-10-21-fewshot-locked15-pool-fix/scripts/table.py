"""Builds table.md for the F2 pool-fix rerun from results.jsonl (new, pool-fixed) vs locked15 results.jsonl (old, label-free pool)."""
import json, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent.parent; G = "pglib_opf_case240_pserc"
new = {r["split_seed"]: r for r in map(json.loads, (H / "results.jsonl").read_text().splitlines()) if "error" not in r}
old = {r["split_seed"]: r for r in map(json.loads, (H.parent / "2026-10-21-fewshot-locked15/results.jsonl").read_text().splitlines()) if "error" not in r and r["grid"] == G and r["tag"] == "wu100L"}
raw = json.loads((H.parent / "2026-10-21-fewshot-locked15-rawfd2/results.json").read_text())[G]["fd2_raw"]
L = ["| draw | split | n scen | dVa DC | dVa guarded fd2 | dVa GENCO | ratio to DC | ratio to guarded fd2 | ratio to raw fd2 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
S = {}
for lab, d in (("old (label-free pool, collapsed labels in split)", old), ("pool-fix (collapse-filtering pool)", new)):
    rs = []
    for s in sorted(d):
        o = d[s]["ours"]; g, dc, fd = (o[k]["branch_dva_deg"] for k in ("genco_raw", "dc", "fd2"))
        rs.append((g / dc, g / fd, g / raw))
        L.append(f"| {s} | {lab.split(' ')[0]} | {d[s]['n_scen_ours']} | {dc:.3f} | {fd:.4f} | {g:.3f} | {g/dc:.3f} | {g/fd:.2f} | {g/raw:.2f} |")
    S[lab] = {"mean_ratio_dc": st.mean(r[0] for r in rs), "draws_below_dc": sum(r[0] < 1 for r in rs), "mean_ratio_guarded_fd2": st.mean(r[1] for r in rs), "mean_ratio_raw_fd2": st.mean(r[2] for r in rs)}
    L.append(f"| mean | {lab} | | | | | {S[lab]['mean_ratio_dc']:.3f} ({S[lab]['draws_below_dc']}/3 below DC) | {S[lab]['mean_ratio_guarded_fd2']:.2f} | {S[lab]['mean_ratio_raw_fd2']:.2f} |")
(H / "table.md").write_text("\n".join(L) + "\n"); (H / "summary.json").write_text(json.dumps(S, indent=1)); print("\n".join(L))
