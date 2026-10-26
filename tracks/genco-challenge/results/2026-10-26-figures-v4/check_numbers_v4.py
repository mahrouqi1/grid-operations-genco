"""v4 checker for fig1_voltage_inference_v4.png (single GENCO / best-interpolator panel). Copy of check_numbers_v35.py with the WLS-panel checks dropped.
    python make_fig1_v4.py; python check_numbers_v4.py"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
R = HERE.parent
PL = json.loads((HERE / "plotted_numbers_fig1_v4.json").read_text())
n_ok = 0


def check(cond, msg):
    global n_ok
    assert cond, "FAIL: " + msg
    n_ok += 1


# ---- (a) swarm columns: post section 3 summary (V1-V4)
cols = PL["a_columns"]
want = [("seen", 19, 19, (12, 19)), ("unseen", 14, 17, (11, 17)), ("seen", 19, 19, None), ("unseen", 11, 17, None),
        ("locked 15", 12, 15, None), ("seen", 15, 15, (8, 15)), ("unseen", 8, 17, (2, 17))]
for c, (nm, w, t, nz) in zip(cols, want):
    check(c["column"] == nm and c["wins"] == w and c["total"] == t, f"swarm {nm}: {c['wins']}/{c['total']} vs {w}/{t}")
    if nz is None:
        check(c["noise"] is None, f"{nm} noise should be absent")
    else:
        check((c["noise"]["wins"], c["noise"]["total"]) == nz, f"{nm} noise {c['noise']} vs {nz}")
check(cols[0]["wins"] + cols[1]["wins"] == 33 and cols[0]["total"] + cols[1]["total"] == 36, "V1: 33 of 36 datakit pieces")
# recompute win counts and ratios from the verdict files
J = lambda d: json.loads((R / d / "verdict.json").read_text())
ctl = J("2026-10-21-genco-voltage-inference-control")["results"]["f40"]
check(sorted(round(e["ratio"], 9) for e in ctl.values()) == sorted(round(r, 9) for r in cols[0]["ratios"]), "ratios of column 1 equal the verdict file")
check(abs(cols[0]["median_ratio"] - float(np.median([e["ratio"] for e in ctl.values()]))) < 1e-12, "median of column 1")
# the pieces-versus-whole claim on the unseen noise: 11/17 for unseen pieces (V4)
check(cols[1]["noise"] == {"wins": 11, "total": 17}, "V4 unseen pieces noise 11/17")
src = (HERE / "make_fig1_v4.py").read_text()
# win counts shown == sum of the win flags in the source verdict files, and every point above 1 is a non-win (hollow)
P = lambda d: json.loads((R / d / "protocol.json").read_text())
oos, fvs, gen, c8 = (J(d) for d in ("2026-10-21-genco-voltage-inference-oos", "2026-10-21-genco-voltage-inference-family-vs-size",
                                    "2026-10-21-genco-voltage-inference-generator-control", "2026-10-21-genco-voltage-inference"))
gp = P("2026-10-21-genco-voltage-inference-generator-control"); op = P("2026-10-21-genco-voltage-inference-oos")
src_cols = [list(ctl.values()), list(fvs["results"]["f40"].values()),
            [gen["results"]["f40"][g] for g in gen["ran"] if gp["pieces"][g]["group"] == "seen"],
            [gen["results"]["f40"][g] for g in gen["ran"] if gp["pieces"][g]["group"] == "unseen"],
            list(c8["results"]["secondary_f40"].values()),
            [oos["results"]["f40"][g] for g in op["sets"]["seen"]], [oos["results"]["f40"][g] for g in op["sets"]["primary"]]]
for c, ent in zip(cols, src_cols):
    check(c["wins"] == sum(bool(e["win"]) for e in ent) and c["total"] == len(ent), f"win count shown {c['wins']}/{c['total']} equals verdict file")
    check(all(not e["win"] for e in ent if e["ratio"] >= 1), f"{c['column']}: every point with ratio >= 1 is hollow (not a win)")
    check(sorted(round(e["ratio"], 9) for e in ent) == sorted(round(r, 9) for r in c["ratios"]), f"{c['column']}: plotted ratios equal the verdict file")
check("Every point above 1 is hollow" in src and "hollow = not a win" in src, "annotation text present")
check("GENCO error / best interpolator error" in src, "y label")
check("axw" not in src and "QG" not in src, "no WLS panel code left")
check('BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"' in src, "v3.1 palette kept")
check("\u2014" not in src and "\u2013" not in src, "no em or en dashes in make_fig1_v4.py")
im = Image.open(HERE / "new" / "fig1_voltage_inference_v4.png")
check(im.width == 1500, f"fig width {im.width} (750 px at 2x)")
print(f"check_numbers_v4: ALL {n_ok} CHECKS PASSED")
