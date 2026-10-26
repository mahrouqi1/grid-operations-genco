"""GENCO post figure 2, v4: collapse-reproduction curve only (fig_c top panel of make_fig_v3.py, copy; originals untouched). Linear update axis, all scored steps.
    python make_fig2_v4.py; python check_numbers_fig2_v4.py"""
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
R = HERE.parent                      # .../results
OUT = HERE / "new"
OUT.mkdir(exist_ok=True)

INK, INK2, SURF, GRID, GREY = "#111111", "#4a4945", "#fcfcfb", "#e4e3df", "#8a8984"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"   # Okabe-Ito: seen family / unseen family / locked benchmark
PURP, SKY, CHAR = "#CC79A7", "#56B4E9", "#3b3b3b"     # Okabe-Ito purple: power-flow-task mask; sky blue: corrected labels; charcoal: original labels
# One colour per concept, same in all three figures (random mask = hollow bar, no colour; grey = neutral baseline):
#   seen family BLUE | unseen family VERM | locked benchmark GREEN | power-flow-task mask PURP | corrected labels SKY | original labels CHAR
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "text.color": INK, "axes.unicode_minus": False})
DPI = 200
PLOTTED = {}


def draw_key(fig, y, items, x0=0.02, fs=10.5):
    """Small shared key along the figure top. items: (label, facecolor, edgecolor). Same swatch style in every figure."""
    x = x0
    for lab, fc, ec in items:
        fig.add_artist(Rectangle((x, y - 0.007), 0.022, 0.014, transform=fig.transFigure, facecolor=fc, edgecolor=ec, lw=1.6))
        t = fig.text(x + 0.03, y, lab, fontsize=fs, va="center", ha="left", color=INK)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        x = bb.x1 + 0.035


def clean(ax, left=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if not left: ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)



# ------------------------------------------------------------------ figure 2: collapse reproduction, linear update axis
FINAL = 4960   # 20 epochs x 248 updates per epoch (progress bar of the training log: "Epoch 19: 100% 248/248"); report.md says "about 5,000"


def fig_c():
    CP = R / "2026-10-20-genco-continued-pretraining"
    Pj = json.load(open(CP / "pooled.json"))
    steps = {"step0": 0, "step25": 25, "step50": 50, "step100": 100, "step250": 250, "step500": 500, "step1000": 1000, "final": FINAL}
    arms = (("fixed", SKY, "corrected labels"), ("published", CHAR, "original labels"))
    fig = plt.figure(figsize=(7.5, 4.3))
    ax = fig.add_axes([0.14, 0.17, 0.66, 0.70])
    out = {}
    for arm, col, lab in arms:
        pts = sorted((steps[p["state"]], [p[s]["frac_lt_0.5"] for s in ("1000", "1001", "1002")]) for p in Pj if p["arm"] == arm and p["state"] in steps)
        x = [u for u, _ in pts]
        seeds = np.array([v for _, v in pts]); mean = seeds.mean(axis=1)
        for j in range(3):
            ax.plot(x[:-1], seeds[:-1, j], color=col, lw=1.0, alpha=0.35, zorder=2); ax.plot(x[-2:], seeds[-2:, j], color=col, lw=1.0, alpha=0.35, ls=(0, (3, 3)), zorder=2)
        ax.plot(x[:-1], mean[:-1], color=col, lw=3.0, zorder=4); ax.plot(x[-2:], mean[-2:], color=col, lw=2.0, ls=(0, (3, 3)), zorder=4)
        ax.plot(x, mean, color=col, lw=0, marker="o", ms=8, mec=SURF, mew=1.6, zorder=5)
        out[arm] = {"updates": x, "mean": [float(m) for m in mean], "seeds": seeds.tolist()}
    m_fix, m_pub = out["fixed"]["mean"], out["published"]["mean"]
    ax.text(FINAL + 200, m_fix[-1], "corrected  " + f"{m_fix[-1]:.2f}", color=INK, fontsize=11.5, fontweight="bold", va="center", ha="left", clip_on=False)
    ax.text(FINAL + 200, m_pub[-1], "original  " + f"{m_pub[-1]:.2f}", color=INK, fontsize=11.5, fontweight="bold", va="center", ha="left", clip_on=False)
    ax.annotate(f"released checkpoint {m_fix[0]:.2f}", xy=(0, m_fix[0]), xytext=(1250, 0.78), fontsize=11, color=INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1), linespacing=0.95)
    ax.text(2900, 0.58, "thin lines: the 3 mask seeds\ndashed: no scored step between 1k and final", fontsize=10, color=INK2, ha="center", va="center")
    ax.set_xlim(-120, FINAL + 120); ax.set_ylim(0.0, 1.04)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, FINAL]); ax.set_xticklabels(["0", "1k", "2k", "3k", "4k", f"{FINAL:,}\n(final)"], fontsize=11)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=11)
    ax.tick_params(length=0); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); clean(ax)
    ax.set_xlabel("continued-pretraining updates (scored at 0, 25, 50, 100, 250, 500, 1,000 and final)", fontsize=10.5)
    ax.set_ylabel("fraction of hidden collapsed buses\nstill predicted below 0.5 pu (axis from 0)", fontsize=11, linespacing=1.0)
    draw_key(fig, 0.955, [("trained on corrected labels", SKY, SKY), ("trained on original labels", CHAR, CHAR)])
    PLOTTED["c_curves"] = out; PLOTTED["c_ylim"] = list(ax.get_ylim()); PLOTTED["final_step"] = FINAL
    fig.savefig(OUT / "fig2_corrected_labels_v4.png", dpi=DPI, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    fig_c()
    (HERE / "plotted_numbers_fig2_v4.json").write_text(json.dumps(PLOTTED, indent=1))
    print("done")
