"""Collect the 2026-10-07 label validation into per-grid JSON and an old -> new comparison.

Entry points for this result dir (env ``grid-operations-genco``, repo root,
PYTHONPATH=tracks/genco-challenge/src):

    python -m genco_challenge.ladder_data --validate-only          # task 1-2: drop collapsed labels
    python -m genco_challenge.ladder_eval --models zeroshot ft118_1000 ft118_250000 \\
        --grids pglib_opf_case1354_pegase pglib_opf_case240_pserc \\
        --out tracks/genco-challenge/results/2026-10-07-genco-heldout-zeroshot
    python -m genco_challenge.ladder_eval --rescore --out tracks/genco-challenge/results/2026-10-07-genco-heldout-zeroshot
    python tracks/genco-challenge/results/2026-10-07-label-validation/genco_data_check.py --parts 6 --with-gen
    python tracks/genco-challenge/results/2026-10-07-label-validation/summarize.py

Writes ``per_grid/<grid>.json`` (the meta.json validation block) and ``old_vs_new.json``.

Note (audit genco-eval R1-F5): the pre-validation case1354 / case240 rows were removed from
``heldout-zeroshot/results.jsonl`` BY HAND (the pre-validation copy is
``heldout-zeroshot.results.pre-validation.jsonl`` here); ladder_eval did not detect them as stale
then. Since 2026-10-08 every row carries a ``data_stamp`` and a stamp mismatch counts as not done, so
the ``--grids`` command above reruns a re-validated grid by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from genco_challenge.ladder_data import WORK

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "2026-10-07-genco-heldout-zeroshot/results.jsonl"
OLD = HERE / "heldout-zeroshot.results.pre-validation.jsonl"
KEYS = {"res_mw": ("genco_active_res_mw", "dc_active_res_mw"),
        "branch_dva_deg": ("branch_dva_mae_deg",), "vm_pu": ("vm_mae_pu",)}


def metrics(r: dict) -> dict:
    e = r["errors"]
    return {"n_test": r["n_test_scenarios"],
            "res_mw": {"genco": r["genco_active_res_mw"], "dc": r["dc_active_res_mw"]},
            "branch_dva_deg": {"genco": e["genco"]["branch_dva_mae_deg"], "dc": e["dc"]["branch_dva_mae_deg"]},
            "vm_pu": {"genco_pinned": e["genco_pinned"]["vm_mae_pu"], "genco_raw": e["genco"]["vm_mae_pu"],
                      "dc": e["dc"]["vm_mae_pu"]}}


def main() -> None:
    (HERE / "per_grid").mkdir(exist_ok=True)
    changed = []
    for p in sorted((WORK / "data").glob("*/meta.json")):
        m = json.loads(p.read_text())
        if "validation" not in m:
            continue
        (HERE / "per_grid" / f"{m['name']}.json").write_text(json.dumps(m, indent=1))
        if m["validation"]["dropped"]:
            changed.append(m["name"])
    old = {(r["grid"], r["model"]): r for r in map(json.loads, OLD.read_text().splitlines()) if "error" not in r}
    new = {(r["grid"], r["model"]): r for r in map(json.loads, RES.read_text().splitlines()) if "error" not in r}
    cmp = []
    for g in changed:
        meta = json.loads((WORK / "data" / g / "meta.json").read_text())
        for mdl in ("zeroshot", "ft118_1000", "ft118_250000"):
            row = {"grid": g, "model": mdl, "status": meta["status"], "kept": meta["validation"]["kept"],
                   "old": metrics(old[(g, mdl)]) if (g, mdl) in old else None,
                   "new": metrics(new[(g, mdl)]) if (g, mdl) in new else None}
            if meta["status"] == "insufficient_valid_scenarios":
                row["new"] = "insufficient valid scenarios"
            cmp.append(row)
    (HERE / "old_vs_new.json").write_text(json.dumps(cmp, indent=1))
    for r in cmp:
        def f(x: dict | str | None) -> str:
            if not isinstance(x, dict):
                return str(x)
            return (f"n={x['n_test']} res G/DC {x['res_mw']['genco']:.2f}/{x['res_mw']['dc']:.2f} "
                    f"dVa G/DC {x['branch_dva_deg']['genco']:.3f}/{x['branch_dva_deg']['dc']:.3f} "
                    f"|V| Gp/DC {x['vm_pu']['genco_pinned']:.4f}/{x['vm_pu']['dc']:.4f}")
        print(f"{r['grid']:28s} {r['model']:13s} OLD {f(r['old'])}\n{'':42s} NEW {f(r['new'])}")


if __name__ == "__main__":
    main()
