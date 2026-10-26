#!/usr/bin/env python
"""Check a regenerated big data set against the checksums recorded for the private working copy.

    python tools/check_external.py <set> <path-to-the-set-root>     # e.g. relabeled-ladder  data/external/genco-ladder/data-relabeled
    python tools/check_external.py --list

The sets and their expected sha256 are in EXTERNAL-DATA.json (see EXTERNAL-DATA.md for how each is regenerated).
Reports per-file MATCH / DIFFERENT / MISSING; exit 1 on any difference. Parquet written by a different
pyarrow or numba build can differ in bytes while the table is equal: a DIFFERENT file needs a table-level
comparison (pandas.read_parquet and DataFrame.equals) before it is called a failure.
"""
import hashlib, json, sys
from pathlib import Path

DATA = json.loads((Path(__file__).resolve().parents[1] / "EXTERNAL-DATA.json").read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        for k, v in DATA.items():
            print(f"{k}: {v['files']} files, {v['bytes']} bytes, tree sha256 {v['tree_sha256']}")
        return 0
    if len(sys.argv) != 3 or sys.argv[1] not in DATA:
        sys.exit(__doc__)
    spec, root = DATA[sys.argv[1]], Path(sys.argv[2])
    res = {"MATCH": 0, "DIFFERENT": 0, "MISSING": 0}
    for rel, want in sorted(spec["per_file"].items()):
        p = root / rel
        if not p.is_file():
            r = "MISSING"
        elif sha(p) == want["sha256"]:
            r = "MATCH"
        else:
            r = "DIFFERENT"
        res[r] += 1
        if r != "MATCH":
            print(f"  {r}: {rel}")
    print(f"check_external {sys.argv[1]}: {res}  (expected {spec['files']} files)")
    return 0 if res["DIFFERENT"] == 0 and res["MISSING"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
