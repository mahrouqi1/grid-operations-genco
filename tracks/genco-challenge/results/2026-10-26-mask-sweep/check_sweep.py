"""Asserts the values plotted in fig_sweep.png (plotted_numbers_fig_sweep.json, written by make_fig_sweep.py) equal sweep_summary.json,
and that sweep_summary.json equals a fresh count from sweep.csv. UNAUDITED.   python check_sweep.py"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "sweep_summary.json").read_text()); Pl = json.loads((HERE / "plotted_numbers_fig_sweep.json").read_text())
df = pd.read_csv(HERE / "sweep.csv")
assert Pl["x_pct"] == S["fractions"]
assert Pl["locked15_wins"] == [S["locked15"]["wins"][str(p)] for p in S["fractions"]], "locked15 plotted != summary"
if S["pieces36"]["ran_any"]:
    pp = Pl["pieces36_x_pct"]
    assert all(S["pieces36"]["n_complete"][str(p)] == S["pieces36"]["n_grids"] for p in pp), "plotted a fraction with incomplete pieces"
    assert Pl["pieces36_wins"] == [S["pieces36"]["wins"][str(p)] for p in pp], "pieces36 plotted != summary"
else:
    assert "pieces36_wins" not in Pl
# summary vs csv
for sname in ("locked15", "pieces36"):
    for p in S["fractions"]:
        c = df[(df.set == sname) & (df.fraction.round(2) == p / 100) & (df.status == "ok")]
        assert int(c.win.sum()) == S[sname]["wins"][str(p)], (sname, p)
        assert len(c) == S[sname]["n_complete"][str(p)], (sname, p)
print("check_sweep: OK")
