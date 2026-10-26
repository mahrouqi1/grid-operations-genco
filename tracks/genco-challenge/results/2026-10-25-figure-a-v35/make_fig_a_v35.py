"""GENCO post figure (a), v3.5 PREVIEW. Copy of 2026-10-24-figure-redesign/make_figs_v3.py (original untouched), figure (a) only.
Top two panels unchanged. Bottom (WLS) panel now shows MASK SEED 0 ratios at the original iteration limit (hollow) and with the limit raised to 40/60 (filled),
read from q_g_ratios_per_grid_v35.csv (q_g_wls_ratios_v35.py; cross-checked to cells_v3.csv). Win counts from cells_v3.csv.
    python q_g_wls_ratios_v35.py; python make_fig_a_v35.py; python check_numbers_v35.py"""
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
def fig_a():
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

    cells = pd.read_csv(R / "2026-10-21-genco-voltage-wls/cells_v2.csv"); c40 = cells[cells.cond == "f40"]
    sets = [("carved pieces", c40[c40.study.isin(["control", "fvs"])], True),
            ("locked 15", c40[c40.study == "locked15"], True),
            ("seen whole grids", c40[c40.group == "oos_seen"], False),
            ("unseen whole grids", c40[c40.group == "oos_unseen"], False)]

    fig = plt.figure(figsize=(7.5, 13.4))
    ax = fig.add_axes([0.13, 0.6807, 0.84, 0.2777]); axn = fig.add_axes([0.13, 0.6160, 0.84, 0.0371], sharex=ax)
    axw = fig.add_axes([0.13, 0.2273, 0.84, 0.3099])

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
    ax.set_ylabel("GENCO error / best classical error", fontsize=11.5)
    ax.text(4.45, 0.12, "GENCO\nbetter", fontsize=12, color=INK2, fontweight="bold", va="bottom", ha="center", linespacing=0.95)
    ax.text(4.45, 1.12, "classical\nbetter", fontsize=12, color=INK2, fontweight="bold", va="bottom", ha="center", linespacing=0.95)
    # direct annotations replacing a legend
    lock = [e for n, c, e, _ in cols if n == "locked 15"][0]
    rmax = max(lock, key=lambda e: e["ratio"])["ratio"]
    ax.annotate("hollow = not a win\n(95% CI reaches 1)", xy=(5.1 + 0.05, rmax * 0.93), xytext=(3.75, 2.9), fontsize=10.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0, shrinkA=0, shrinkB=3))
    ax.annotate("filled = win", xy=(1.0, 0.43), xytext=(1.0, 0.1325), fontsize=10.5, color=INK2, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0, shrinkA=0, shrinkB=2))
    ax.annotate("bar = median", xy=(0.46, float(np.median([e['ratio'] for e in cols[0][2]]))), xytext=(-0.55, 0.175), fontsize=10.5, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0, shrinkA=0, shrinkB=1))
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

    # panel 3 (v3.5): GENCO / WLS error ratio per grid, MASK SEED 0, informed vs limit-blind WLS, each at the original iteration
    # limit (hollow) and with the limit raised to 40/60 (filled). Source: q_g_ratios_per_grid_v35.csv (cross-checked to cells_v3.csv).
    QG = pd.read_csv(HERE / "q_g_ratios_per_grid_v35.csv")
    qsets = [("carved\npieces", QG[QG.study.isin(["control", "fvs"])]), ("locked 15", QG[QG.study == "locked15"]),
             ("seen\nwhole grids", QG[QG.group == "oos_seen"]), ("unseen\nwhole grids", QG[QG.group == "oos_unseen"])]
    YTOP = 120.0
    axw.set_yscale("log"); axw.set_ylim(0.06, YTOP); axw.set_xlim(-2.3, 21.2); axw.minorticks_off()
    axw.axhspan(0.06, 1, color="#eef3f7", zorder=0); axw.axhline(1, color=INK, lw=1.4, zorder=1)
    axw.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 100]); axw.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "30", "100"], fontsize=11)
    axw.tick_params(axis="y", length=0); axw.tick_params(axis="x", length=0, labelbottom=False)
    axw.grid(axis="y", color=GRID, lw=0.7, zorder=0); axw.set_axisbelow(True); clean(axw)
    axw.set_ylabel("GENCO error / WLS error (mask seed 0)", fontsize=11.5)
    # (column key, marker, face colour, edge colour, label); hollow = original iteration limit, filled = limit raised to 40/60
    variants = [("ratio_inf", "s", SURF, INK, "informed WLS, original limit"), ("ratio_infunc", "s", INK, INK, "informed WLS, limit 40/60"),
                ("ratio_flat", "^", SURF, GREY, "limit-blind WLS, original limit"), ("ratio_flatunc", "^", GREY, GREY, "limit-blind WLS, limit 40/60")]
    DX = 1.08
    offs = [-1.5 * DX, -0.5 * DX, 0.5 * DX, 1.5 * DX]
    qx0 = [0.0, 5.6, 11.2, 16.8]
    lab_fmt = lambda m: f"{m:.2f}" if m < 2 else f"{m:.1f}"
    qout = []
    for (lab, d), xc in zip(qsets, qx0):
        rec = {"set": lab.replace("\n", " "), "n": int(len(d))}
        for (col, mk, fc, ec, vname), off in zip(variants, offs):
            x = xc + off
            r = d[col].to_numpy(dtype=float)
            order = np.argsort(r); jit = np.zeros(len(r)); placed = []
            step = 0.1
            for i in order:
                k = 0
                while k < 9 and any(abs(np.log(r[i]) - np.log(r[j])) < 0.16 and abs(jit[j] - (step * ((k + 1) // 2) * (1 if k % 2 else -1))) < 0.095 for j in placed): k += 1
                jit[i] = float(np.clip(step * ((k + 1) // 2) * (1 if k % 2 else -1), -0.4, 0.4)); placed.append(i)
            yv = np.minimum(r, YTOP)
            axw.plot(x + jit, yv, mk, ms=6.3, mfc=fc, mec=ec, mew=1.3, ls="none", zorder=3, clip_on=False)
            med = float(np.median(r))
            axw.plot([x - 0.5, x + 0.5], [med, med], color=INK, lw=2.2, solid_capstyle="butt", zorder=4)
            axw.text(x, 0.0565 if off in (offs[0], offs[2]) else 0.0385, lab_fmt(med), ha="center", va="top", fontsize=9, fontweight="bold", color=INK, clip_on=False)
            big = r > YTOP
            if big.any():
                axw.text(x, YTOP * 1.1, f"+{int(big.sum())}", ha="center", va="bottom", fontsize=8.5, color=INK2, linespacing=0.95, clip_on=False)
            rec[vname] = {"median": med, "n": int(len(r)), "ratios": [float(v) for v in r], "n_off_scale": int(big.sum()), "max": float(r.max())}
        qout.append(rec)
        axw.text(xc, 0.0185, lab, ha="center", va="top", fontsize=11.5, color=INK, linespacing=0.95, clip_on=False)
    axw.text(21.1, 0.55, "GENCO\nbetter", fontsize=11, color=INK2, fontweight="bold", va="top", ha="right", linespacing=0.95)
    axw.text(21.1, 1.1, "WLS\nbetter", fontsize=11, color=INK2, fontweight="bold", va="bottom", ha="right", linespacing=0.95)
    # direct key (replaces a legend): square = informed, triangle = limit-blind; hollow = original limit, filled = limit raised to 40/60
    for xk, (mk, c_h, c_f, ttl) in zip((-2.0, 7.4), (("s", INK, INK, "informed WLS (true limits)"), ("^", GREY, GREY, "limit-blind WLS"))):
        yk = 400
        axw.plot(xk, yk, mk, ms=7, mfc=SURF, mec=c_h, mew=1.3, clip_on=False)
        axw.plot(xk + 0.7, yk, mk, ms=7, mfc=c_f, mec=c_h, mew=1.3, clip_on=False)
        axw.text(xk + 1.2, yk, ttl, fontsize=9.5, color=INK, va="center", ha="left", clip_on=False)
    axw.text(-2.0, 210, "hollow = original iteration limit, filled = limit 40/60; bar and number = median", fontsize=9.5, color=INK2, va="center", ha="left", style="italic", clip_on=False)
    PLOTTED["a_qg"] = qout
    # win counts (GENCO beats the best interpolator and WLS), seed 0, from cells_v3.csv; stated in the text and caption
    cells = pd.read_csv(R / "2026-10-21-genco-voltage-wls/cells_v3.csv"); c3 = cells[cells.cond == "f40"]
    sets3 = [("carved pieces", c3[c3.study.isin(["control", "fvs"])]), ("locked 15", c3[c3.study == "locked15"]),
             ("seen whole grids", c3[c3.group == "oos_seen"]), ("unseen whole grids", c3[c3.group == "oos_unseen"])]
    PLOTTED["a_wls"] = [{"set": lab, "n": int(len(d)), "i1_i5": int(d.win5.sum()), "informed": int(d.win6.sum()), "informed_unc": int(d.win6unc.sum()),
                         "informed_unc_fb": int(d.win6unc_fb.sum()), "limit_blind": int(d.win6flat.sum()), "limit_blind_unc": int(d.win6flatunc.sum()),
                         "limit_blind_unc_fb": int(d.win6flatunc_fb.sum()), "limit_blind_fb": int(d.win6flat_fb.sum()),
                         "nc_inf_unc": float(d.frac_nc_Wunc.mean()), "nc_flat_unc": float(d.frac_nc_Wflatunc.mean())} for lab, d in sets3]
    wo = pd.read_csv(R / "2026-10-21-genco-voltage-wls/wls_only_v3.csv")
    PLOTTED["wls_only_v3"] = wo.to_dict(orient="records")
    foot = ("Bottom panel: GENCO error / WLS error on the unmeasured buses at 40% measured, mask seed 0, one marker per grid (above 1 = GENCO worse). "
            "WLS = weighted-least-squares state estimator. Hollow markers: the original iteration limit, where most WLS runs were stopped by the limit "
            "(at mask seed 0: 60% of informed solves; limit-blind 88% on carved pieces and 99% on seen-family whole grids). Filled markers: limit raised to 40/60 iterations, "
            "which still fails to converge on part of the scenarios. Informed WLS at 40/60, all scenarios run: not converged on 17% of locked-benchmark scenarios, "
            "11% and 30% of seen and unseen pieces, 49% of whole-grid scenarios. Limit-blind WLS at 40/60, on the scored scenarios: not converged on 35% of piece, "
            "25% of locked, 81% of seen whole-grid and 44% of unseen whole-grid scenarios. 10 to 30 scenarios per grid. "
            "The SimBench high-voltage grids, where WLS converges, sit at the top; values above 120 are clipped, and the numbers above the top "
            "give how many grids are clipped (largest ratio: informed 6,146, limit-blind 1,606). WLS counts checked in two audit rounds.")
    fig.text(0.02, 0.008, textwrap.fill(foot, 100), fontsize=8.5, color=INK2, va="bottom", ha="left", linespacing=1.15)
    draw_key(fig, 0.9880, [("seen family", BLUE, BLUE), ("unseen family", VERM, VERM), ("locked benchmark (15 grids)", GREEN, GREEN)])
    fig.savefig(OUT / "fig_a_voltage_inference_v35.png", dpi=DPI, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    fig_a()
    (HERE / "plotted_numbers_v35.json").write_text(json.dumps(PLOTTED, indent=1))
    print("done")
