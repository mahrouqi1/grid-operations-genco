"""GENCO post figure 1, v4: ONLY the GENCO / best-interpolator panel (swarm + win counts + 0.5%-noise row) of make_fig_a_v35.py. Copy; originals untouched.
    python make_fig1_v4.py; python check_numbers_v4.py
(older docstring follows) GENCO post figure (a), v3.5 PREVIEW. Copy of 2026-10-24-figure-redesign/make_figs_v3.py (original untouched), figure (a) only.
Top two panels unchanged. Bottom (WLS) panel now shows MASK SEED 0 ratios at the original iteration limit (hollow) and with the limit raised to 40/60 (filled),
read from q_g_ratios_per_grid_v35.csv (q_g_wls_ratios_v35.py; cross-checked to cells_v3.csv). Win counts from cells_v3.csv.
    python q_g_wls_ratios_v35.py; python make_fig_a_v35.py; python check_numbers_v35.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image

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


# ------------------------------------------------------------------ (a) voltage inference + WLS layer
def fig_a():  # v4: single panel
    J = lambda d: json.loads((R / d / "verdict.json").read_text())
    P = lambda d: json.loads((R / d / "protocol.json").read_text())
    c8, oos, ctl, fvs, gen = (J(d) for d in ("2026-10-21-genco-voltage-inference", "2026-10-21-genco-voltage-inference-oos",
                                              "2026-10-21-genco-voltage-inference-control", "2026-10-21-genco-voltage-inference-family-vs-size",
                                              "2026-10-21-genco-voltage-inference-generator-control"))
    oos_p = P("2026-10-21-genco-voltage-inference-oos"); gen_p = P("2026-10-21-genco-voltage-inference-generator-control")
    take = lambda res: [res[k] for k in res]
    cols = [  # key, family colour, entries, noise count (wins, total) or None
        ("seen", BLUE, take(ctl["results"]["f40"]), (ctl["secondary_n40"]["wins"], ctl["secondary_n40"]["grids"])),
        ("unseen", VERM, take(fvs["results"]["f40"]), (fvs["descriptive_n40"]["unseen_wins"], fvs["descriptive_n40"]["unseen_n"])),
        ("seen", BLUE, [gen["results"]["f40"][g] for g in gen["ran"] if gen_p["pieces"][g]["group"] == "seen"], None),
        ("unseen", VERM, [gen["results"]["f40"][g] for g in gen["ran"] if gen_p["pieces"][g]["group"] == "unseen"], None),
        ("locked 15", GREEN, take(c8["results"]["secondary_f40"]), None),
        ("seen", BLUE, [oos["results"]["f40"][g] for g in oos_p["sets"]["seen"]],
         (oos["seen_family_descriptive"]["n40"]["wins"], oos["seen_family_descriptive"]["n40"]["grids_ran"])),
        ("unseen", VERM, [oos["results"]["f40"][g] for g in oos_p["sets"]["primary"]],
         (oos["secondary_i_noise40"]["wins"], oos["secondary_i_noise40"]["grids_ran"])),
    ]
    xs = [0, 1, 2.35, 3.35, 5.1, 6.1, 7.1]


    H = 6.05   # inches; single panel (heights in inches from the v3.5 layout: swarm 3.72, noise strip 0.5, label gap 0.87, top margin 0.56)
    fig = plt.figure(figsize=(7.5, H))
    ax = fig.add_axes([0.13, 1.27 / H, 0.84, 3.72 / H]); axn = fig.add_axes([0.13, 0.25 / H, 0.84, 0.5 / H], sharex=ax)

    # panel 1: swarm
    ax.axhspan(0.06, 1, color="#eef3f7", zorder=0)
    ax.axhline(1, color=INK, lw=1.4, zorder=1)
    out = []
    for (name, col, ent, noise), x in zip(cols, xs):
        r = np.array([e["ratio"] for e in ent]); win = np.array([e["win"] for e in ent])
        order = np.argsort(r); jit = np.zeros(len(r)); placed = []
        for i in order:
            k = 0
            while k < 9 and any(abs(np.log(r[i]) - np.log(r[j])) < 0.20 and abs(jit[j] - (0.11 * ((k + 1) // 2) * (1 if k % 2 else -1))) < 0.105 for j in placed): k += 1
            jit[i] = float(np.clip(0.11 * ((k + 1) // 2) * (1 if k % 2 else -1), -0.47, 0.47)); placed.append(i)
        ax.plot(x + jit[win], r[win], "o", ms=9.5, mfc=col, mec=SURF, mew=1.4, ls="none", zorder=3)
        ax.plot(x + jit[~win], r[~win], "o", ms=9.5, mfc=SURF, mec=col, mew=2.2, ls="none", zorder=3)
        med = float(np.median(r))
        ax.plot([x - 0.46, x + 0.46], [med, med], color=INK, lw=2.4, solid_capstyle="butt", zorder=4)
        ax.text(x, 12.0, f"{int(win.sum())}/{len(win)}", ha="center", va="bottom", fontsize=14, fontweight="bold")
        out.append({"column": name, "x": x, "wins": int(win.sum()), "total": int(len(win)), "median_ratio": med, "ratios": [float(v) for v in r],
                    "noise": None if noise is None else {"wins": int(noise[0]), "total": int(noise[1])}})
        ax.plot([x - 0.4, x + 0.4], [0.052, 0.052], color=col, lw=5, clip_on=False, solid_capstyle="butt")
        ax.text(x, 0.046, name.replace(" 15", "\n15"), ha="center", va="top", fontsize=11.5, color=INK)
    ax.set_yscale("log"); ax.set_ylim(0.06, 34); ax.set_xlim(-0.6, 7.75); ax.minorticks_off()
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"], fontsize=11.5)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); clean(ax)
    ax.set_ylabel("GENCO error / best interpolator error", fontsize=11.5)
    ax.text(4.2, 0.12, "GENCO\nbetter", fontsize=10.5, color=INK2, fontweight="bold", va="bottom", ha="center", linespacing=0.95)
    ax.text(4.2, 1.12, "interpolator\nbetter", fontsize=10.5, color=INK2, fontweight="bold", va="bottom", ha="center", linespacing=0.95)
    # legend-like key in the band under the colour key (touches no data)
    for yk, kind in ((1 - 0.52 / H, "fill"), (1 - 0.78 / H, "hollow")):
        fig.add_artist(Line2D([0.035], [yk], transform=fig.transFigure, marker="o", ms=9.5, ls="none", mfc=INK2 if kind == "fill" else SURF, mec=SURF if kind == "fill" else INK2, mew=1.4 if kind == "fill" else 2.2))
        fig.text(0.06, yk, "significant GENCO win (95% CI upper end below 1)" if kind == "fill" else "not a significant win; above 1, GENCO's error is higher", fontsize=10.5, color=INK2, va="center", ha="left")
    fig.add_artist(Line2D([0.62, 0.66], [1 - 0.52 / H] * 2, transform=fig.transFigure, color=INK, lw=2.4, solid_capstyle="butt"))
    fig.text(0.67, 1 - 0.52 / H, "bar = group median", fontsize=10.5, color=INK2, va="center", ha="left")
    # headers over the groups
    for (x0, x1, t) in ((-0.4, 3.75, ""), (4.7, 7.5, "")):
        ax.plot([x0, x1], [19, 19], color=GREY, lw=1.2, clip_on=False)
    for (xc, t) in ((1.675, "carved subgrids, 159 to 903 buses"), (6.1, "whole grids")):
        ax.text(xc, 21.5, t, ha="center", va="bottom", fontsize=11.5, color=INK2, fontweight="bold")
    for xc, t in ((0.5, "GENCO's own\nscenario generator"), (2.85, "one common\nscenario generator")):
        ax.text(xc, 0.068, t, ha="center", va="bottom", fontsize=9.5, color=INK2, style="italic", linespacing=0.95)
    PLOTTED["a_columns"] = out

    # panel 2: noise strip
    axn.set_ylim(0, 1); axn.axis("off")
    axn.text(-0.45, 0.97, "with 0.5% noise on the measured voltages", fontsize=10.5, color=INK2, ha="left", va="top", style="italic")
    for o in out:
        t = "n/a" if o["noise"] is None else f"{o['noise']['wins']}/{o['noise']['total']}"
        axn.text(o["x"], 0.28, t, ha="center", va="center", fontsize=12.5 if o["noise"] else 10.5, fontweight="bold" if o["noise"] else "normal",
                 color=INK if o["noise"] else GREY)

    draw_key(fig, 1 - 0.16 / H, [("seen family", BLUE, BLUE), ("unseen family", VERM, VERM), ("locked benchmark (15 grids)", GREEN, GREEN)])
    fig.savefig(OUT / "fig_voltage_v5.png", dpi=DPI, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    fig_a()
    (HERE / "plotted_numbers_voltage_v5.json").write_text(json.dumps(PLOTTED, indent=1))
    print("done")
