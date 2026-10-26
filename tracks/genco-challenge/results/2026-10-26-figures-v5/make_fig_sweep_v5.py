"""fig_sweep_v5.png: share of grids where GENCO beats the best interpolator, against the measured fraction, one axis
(locked 15 and the 36 carved pieces as percentages, so the 10-of-15 bar is not read on a second axis).
Reads ../2026-10-26-mask-sweep/sweep_summary.json only. Style from make_fig_voltage_v5.py.
    python make_fig_sweep_v5.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "new"
OUT.mkdir(exist_ok=True)
INK, INK2, SURF, GRID, GREY = "#111111", "#4a4945", "#fcfcfb", "#e4e3df", "#8a8984"
BLUE, GREEN = "#0072B2", "#009E73"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.unicode_minus": False})
S = json.loads((HERE.parent / "2026-10-26-mask-sweep" / "sweep_summary.json").read_text())
X = S["fractions"]
lw = [S["locked15"]["wins"][str(p)] for p in X]
pw = [S["pieces36"]["wins"][str(p)] for p in X]
ly = [100 * w / 15 for w in lw]
py = [100 * w / 36 for w in pw]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.plot(X, py, "-s", color=BLUE, lw=2.4, ms=7.5, mec=SURF, mew=1.2, zorder=4)
ax.plot(X, ly, "-o", color=GREEN, lw=2.4, ms=8.5, mec=SURF, mew=1.4, zorder=5)
for x, y, w in zip(X, ly, lw):
    if x == 90:
        ax.text(x, y + 3.5, f"{w}/15", ha="center", va="bottom", fontsize=9.5, color=GREEN)
    elif x == 30:
        ax.text(x + 1.0, y - 7, f"{w}/15", ha="left", va="top", fontsize=9.5, color=GREEN)
    else:
        ax.text(x, y - 6.5, f"{w}/15", ha="center", va="top", fontsize=9.5, color=GREEN)
for x, y, w in zip(X, py, pw):
    if x == 90:
        ax.text(x, y - 4, f"{w}/36", ha="center", va="top", fontsize=9.5, color=BLUE)
    elif x == 10:
        ax.text(x - 1.4, y, f"{w}/36", ha="right", va="center", fontsize=9.5, color=BLUE)
    else:
        ax.text(x, y + 3.5, f"{w}/36", ha="center", va="bottom", fontsize=9.5, color=BLUE)
ax.text(70, 60, "36 carved pieces", color=BLUE, fontsize=11, fontweight="bold", ha="center", va="center")
ax.text(70, 48, "locked 15 grids", color=GREEN, fontsize=11, fontweight="bold", ha="center", va="center")
ax.plot([15, 25], [100 * 10 / 15] * 2, color=INK, lw=1.4, ls="--", zorder=3)
ax.axvline(20, color=GREY, lw=1.0, ls=":", zorder=1)
ax.text(27, 6, "dotted line: GENCO's pretraining level (about 20% of |V| visible)\n"
        "dashed line: bar set in advance for the locked-15 test at 20% (10/15)",
        fontsize=9.5, color=INK2, ha="left", va="bottom", linespacing=1.1)
ax.set_xlim(3, 95)
ax.set_ylim(0, 105)
ax.set_xticks(X)
ax.set_xticklabels([f"{p}%" for p in X])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("measured fraction of buses (|V| visible), mask seed 0", fontsize=11)
ax.set_ylabel("grids where GENCO beats\nthe best interpolator", fontsize=11)
ax.tick_params(length=0)
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
fig.savefig(OUT / "fig_sweep_v5.png", dpi=200, facecolor=SURF)
(HERE / "plotted_numbers_sweep_v5.json").write_text(json.dumps({"x": X, "locked15_wins": lw, "pieces36_wins": pw}, indent=1))
print("done", lw, pw)
