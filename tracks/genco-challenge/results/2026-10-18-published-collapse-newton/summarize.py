"""Tally out.jsonl per subgrid and method (thresholds from protocol.json). Writes table.md, summary.json."""
import json, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f"{HERE}/out.jsonl")]
# audit R2-F2: one row per (tag, variant), the last one written wins; every case has both methods; no error rows
last = {}
for r in rows:
    if r["variant"] == "error":
        raise SystemExit(f"error row for {r['tag']}: {r.get('err', '')[:200]}")
    last[(r["tag"], r["variant"])] = r
rows = list(last.values())
import csv as _csv
for c in _csv.DictReader(open(f"{HERE}/cases.csv")):
    t = f"{c['subgrid']}|{c['scenario']}|{c['kind']}"
    for v in ("trust_region_default", "newton"):
        assert (t, v) in last, f"missing row {t} {v}"
T = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    g, sc, kind = r["tag"].split("|"); v = r["variant"]; c = T[(g, kind, v)]
    c["n"] += 1
    if r["status"] == "julia_exception":
        c["julia_exception"] += 1; continue
    if r["status"] != "converged":
        c["not_converged"] += 1; continue
    if r.get("max_dV_vs_label", 9) < 1e-6: c["equals_label"] += 1
    if r.get("max_dV_vs_phys", 9) < 1e-4: c["physical_root"] += 1
    if r["min_vm"] < 0.5: c["collapsed"] += 1
L = ["| subgrid | kind | method | n | equals published label | our physical root | collapsed (min Vm < 0.5) | not converged |", "|---|---|---|---|---|---|---|---|"]
tot = collections.defaultdict(collections.Counter)
for (g, kind, v), c in sorted(T.items()):
    L.append(f"| {g.replace('subnet_case','')} | {kind} | {v} | {c['n']} | {c['equals_label']} | {c['physical_root'] if kind=='collapsed' else 'n/a'} | {c['collapsed']} | {c['not_converged']} |")
    tot[(kind, v)].update(c)
L += ["", "| kind | method | n | equals published label | our physical root | collapsed | not converged |", "|---|---|---|---|---|---|---|"]
for (kind, v), c in sorted(tot.items()):
    L.append(f"| {kind} | {v} | {c['n']} | {c['equals_label']} | {c['physical_root'] if kind=='collapsed' else 'n/a'} | {c['collapsed']} | {c['not_converged']} |")
open(f"{HERE}/table.md", "w").write("\n".join(L) + "\n")
json.dump({f"{k[0]}|{k[1]}": dict(v) for k, v in tot.items()}, open(f"{HERE}/summary.json", "w"), indent=1)
print("\n".join(L))
