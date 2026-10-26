"""Asserts the values plotted in fig_corrected_v51.png against the source json (pooled.json, seeds/pooled_seeds.json).
    python make_fig_corrected_v51.py; python check_numbers_corrected_v51.py"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CP = HERE.parent / "2026-10-20-genco-continued-pretraining"
PL = json.loads((HERE / "plotted_numbers_corrected_v51.json").read_text())
P = json.loads((CP / "pooled.json").read_text())
S = json.loads((CP / "seeds" / "pooled_seeds.json").read_text())["per_seed"]
n = 0


def check(c, m):
    global n
    assert c, "FAIL: " + m
    n += 1


STEPS = [0, 25, 50, 100, 250, 500, 1000, PL["final_step"]]
NAME = {0: "step0", 25: "step25", 50: "step50", 100: "step100", 250: "step250", 500: "step500", 1000: "step1000", PL["final_step"]: "final"}
check(PL["final_step"] == 20 * 248, "final step = 20 epochs x 248 updates")
for arm in ("fixed", "published"):
    c = PL["c_curves"][arm]
    check(c["updates"] == STEPS, f"{arm}: plotted update steps {c['updates']}")
    for u, m, sd in zip(c["updates"], c["mean"], c["seeds"]):
        row = [p for p in P if p["arm"] == arm and p["state"] == NAME[u]]
        check(len(row) == 1, f"{arm} {u}: one source row")
        v = [row[0][s]["frac_lt_0.5"] for s in ("1000", "1001", "1002")]
        check(np.allclose(sd, v, atol=1e-12) and abs(m - np.mean(v)) < 1e-12, f"{arm} step {u}: per-seed values and mean equal pooled.json")
    check(len(P) == 16, "pooled.json has exactly the 16 plotted states (8 per arm); no unplotted intermediate")
# endpoint labels as drawn (2 decimals): corrected 0.84, original 0.94, released 0.97
fx, pb = PL["c_curves"]["fixed"], PL["c_curves"]["published"]
check(f"{fx['mean'][-1]:.2f}" == "0.84" and f"{pb['mean'][-1]:.2f}" == "0.94", "endpoint labels 0.84 / 0.94")
check(f"{fx['mean'][0]:.2f}" == "0.97" and f"{pb['mean'][0]:.2f}" == "0.97", "released checkpoint 0.97 (step 0 of both arms)")
check(PL["c_ylim"][0] == 0.0, "y axis from 0")
# three training seeds, mask seed 1000, final state (not plotted; reported in the text)
f = [S[f"fixed_s{k}"]["collapse"]["1000"] for k in (200, 201, 202)]
p = [S[f"published_s{k}"]["collapse"]["1000"] for k in (200, 201, 202)]
check(round(float(np.median(f)), 2) == 0.80 and round(min(f), 2) == 0.70 and round(max(f), 2) == 0.83, f"corrected seeds median 0.80, range 0.70-0.83: {f}")
check(round(float(np.median(p)), 2) == 0.94 and round(min(p), 2) == 0.92 and round(max(p), 2) == 0.97, f"original seeds median 0.94, range 0.92-0.97: {p}")
src = (HERE / "make_fig_corrected_v51.py").read_text()
check("dashed: no scored checkpoint between 1,000 and 4,960" in src and "thin lines:" not in src, "single in-figure note")
check("set_xscale" not in src, "linear x axis")
check("\u2014" not in src and "\u2013" not in src, "no em or en dashes in make_fig_corrected_v51.py")
check(Image.open(HERE / "new" / "fig_corrected_v51.png").width == 1500, "png width")
print(f"check_numbers_corrected_v51: ALL {n} CHECKS PASSED")
