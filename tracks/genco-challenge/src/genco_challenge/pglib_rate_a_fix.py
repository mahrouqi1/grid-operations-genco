"""Post-hoc: make branch ``rate_a`` of converted PGLib cases equal the PGLib .m file's RATE_A column.

Datakit passes the case file's rate_a through unchanged (PGLib files carry 99999 sentinels), so for every
``pglib_opf_*`` case the converted ``branch_data.parquet`` gets the .m values, in datakit's branch order (the .m
row order). Our branches come from pandapower's ppc (lines, then transformers, then impedances), so the order is
recovered by matching each converted branch to a .m row on (unordered bus pair, r, x); bus ids map by rank, as
datakit's ``Network`` does. Ties (electrically identical parallel branches) are assigned in order; the number of
ties whose .m ratings differ is reported (0 means the assignment is unambiguous).

    python -m genco_challenge.pglib_rate_a_fix <converted root> [--pglib DIR] [--cases ...]

Rewrites only the rate_a column; records before/after in ``<case>/convert_stats.json`` and asserts max abs diff 0.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]


def fix_case(case_dir: Path, pglib_dir: Path) -> dict:
    from matpowercaseframes import CaseFrames
    cf = CaseFrames(str(pglib_dir / f"{case_dir.name}.m"))
    ids = np.sort(cf.bus.BUS_I.values.astype(int))
    rank = {int(b): i for i, b in enumerate(ids)}
    mb = cf.branch
    mf = np.array([rank[int(v)] for v in mb.F_BUS.values])
    mt = np.array([rank[int(v)] for v in mb.T_BUS.values])
    mr, mx, mrate = mb.BR_R.values.astype(float), mb.BR_X.values.astype(float), mb.RATE_A.values.astype(float)

    path = case_dir / "raw" / "branch_data.parquet"
    tab = pq.read_table(path)
    sc = tab["scenario"].to_numpy()
    first = sc == sc.min()
    idx = tab["idx"].to_numpy()[first].astype(int)
    f, t = tab["from_bus"].to_numpy()[first], tab["to_bus"].to_numpy()[first]
    r, x = tab["r"].to_numpy()[first], tab["x"].to_numpy()[first]
    oldr = tab["rate_a"].to_numpy()[first]
    n = len(idx)
    if n != len(mb):
        raise RuntimeError(f"{case_dir.name}: {n} converted branches vs {len(mb)} in the .m file")
    groups = defaultdict(list)
    for j in range(len(mb)):
        groups[(min(mf[j], mt[j]), max(mf[j], mt[j]))].append(j)
    used = np.zeros(len(mb), bool)
    rate_for = np.full(n, np.nan)
    amb = 0
    worst = 0.0
    for k in np.argsort(idx):
        cand = [j for j in groups[(min(f[k], t[k]), max(f[k], t[k]))] if not used[j]]
        if not cand:
            raise RuntimeError(f"{case_dir.name}: converted branch {idx[k]} has no .m counterpart")
        cost = [abs(mr[j] - r[k]) + abs(mx[j] - x[k]) for j in cand]
        # electrically identical parallels are told apart by the rating pandapower carried over (it keeps real
        # ratings, only sentinels are replaced), then by file order
        rd = [abs(mrate[j] - oldr[k]) / max(1.0, abs(mrate[j])) for j in cand]
        key = [(round(c, 9), round(q, 9)) for c, q in zip(cost, rd)]
        best = min(range(len(cand)), key=lambda i: key[i])
        j = cand[best]
        tie = [i for i in range(len(cand)) if key[i] == key[best]]
        if len(tie) > 1 and len({mrate[cand[i]] for i in tie}) > 1:
            amb += 1
        worst = max(worst, cost[best])
        used[j] = True
        rate_for[k] = mrate[j]
    if worst > 1e-4:
        raise RuntimeError(f"{case_dir.name}: r/x of matched branches differ by up to {worst:.2e}")
    by_idx = np.empty(n)
    by_idx[idx] = rate_for
    old = tab["rate_a"].to_numpy()
    new = by_idx[tab["idx"].to_numpy().astype(int)]
    ci = tab.schema.get_field_index("rate_a")
    tab = tab.set_column(ci, "rate_a", pa.array(new, type=pa.float64()))
    tmp = path.with_suffix(".tmp")
    pq.write_table(tab, tmp, row_group_size=1 << 20)
    tmp.replace(path)
    chk = pq.read_table(path, columns=["idx", "rate_a"])
    diff = float(np.abs(chk["rate_a"].to_numpy() - by_idx[chk["idx"].to_numpy().astype(int)]).max())
    assert diff == 0.0, diff
    rec = {"source": "PGLib .m RATE_A (datakit passes it through)", "before_max": float(old.max()),
           "after_max": float(new.max()), "n_branches_changed_gt_1e-6_rel": int((np.abs(old[first] - new[first]) > 1e-6 * np.maximum(1.0, np.abs(new[first]))).sum()),
           "n_branches_changed_any": int((np.abs(old[first] - new[first]) > 0).sum()),   # includes ~1e-16 float noise from pandapower's kA round trip
           "n_branches": n, "match_max_rx_diff": worst, "n_ambiguous_ties_with_different_rates": amb,
           "max_abs_diff_vs_m_after": diff}
    sp = case_dir / "convert_stats.json"
    st = json.loads(sp.read_text())
    st["rate_a_pglib_posthoc"] = rec
    st["rate_a_max_mva"] = rec["after_max"]
    sp.write_text(json.dumps(st, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pglib", default=str(REPO / "data/external/pglib-opf"))
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    cases = a.cases or sorted(p.name for p in root.iterdir() if p.name.startswith("pglib_opf_") and (p / "raw").is_dir())
    for c in cases:
        print(c, json.dumps(fix_case(root / c, Path(a.pglib))), flush=True)


if __name__ == "__main__":
    main()
