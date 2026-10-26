"""CLI: python -m genco_challenge.repairkit {check,repair,verify} ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m genco_challenge.repairkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="classify every scenario (read-only)")
    c.add_argument("raw_dir", type=Path)
    c.add_argument("--out", type=Path, default=None, help="write the per-scenario report to this csv (+ .summary.json)")
    c.add_argument("--workers", type=int, default=1)
    r = sub.add_parser("repair", help="relabel flagged scenarios and write a full repaired copy")
    r.add_argument("raw_dir", type=Path)
    r.add_argument("out_dir", type=Path)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--categories", default="collapsed,negative", help="comma list of collapsed, negative (default both)")
    v = sub.add_parser("verify", help="check a repaired dir against its original; exit 1 on any failure")
    v.add_argument("orig_dir", type=Path)
    v.add_argument("repaired_dir", type=Path)
    v.add_argument("--log", type=Path, default=None, help="repair_log.csv (default: inside repaired_dir)")
    v.add_argument("--out", type=Path, default=None, help="write the verification result as json")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        from . import check
        check.run(a.raw_dir, a.out, a.workers)
        return 0
    if a.cmd == "repair":
        from . import repair
        df = repair.run(a.raw_dir, a.out_dir, a.workers, a.categories)
        print(df["status"].value_counts().to_dict() if len(df) else "nothing flagged")
        return 0
    from . import verify
    return verify.run(a.orig_dir, a.repaired_dir, a.log, a.out)


if __name__ == "__main__":
    sys.exit(main())
