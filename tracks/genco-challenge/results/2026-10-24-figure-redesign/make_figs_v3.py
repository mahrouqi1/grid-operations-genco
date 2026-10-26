"""GENCO post figures, v3 redesign. Reads ONLY saved result files; no model runs.
    python make_figs_v3.py  -> new/fig_a_voltage_inference.png, new/fig_b_transfer_12k.png, new/fig_c_unlearning.png,
                               compare/*_before_after.png, plotted_numbers.json (every number drawn; check_numbers.py reads it)
Palette: Okabe-Ito (colour-blind safe). Colour is never the only code: fill/hollow, marker shape and direct labels carry it too."""
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
OUT = HERE / "new"; CMP = HERE / "compare"
OUT.mkdir(exist_ok=True); CMP.mkdir(exist_ok=True)

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

    fig = plt.figure(figsize=(7.5, 12.4))
    ax = fig.add_axes([0.13, 0.655, 0.84, 0.30]); axn = fig.add_axes([0.13, 0.585, 0.84, 0.04], sharex=ax)
    axw = fig.add_axes([0.13, 0.165, 0.84, 0.335])

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

    # panel 3: GENCO / WLS error ratio per grid, log axis, informed vs limit-blind WLS
    QG = pd.read_csv(R / "2026-10-24-repair-root-audit/q_g_ratios_per_grid.csv")
    qsets = [("carved\npieces", QG[QG.study.isin(["control", "fvs"])]), ("locked 15", QG[QG.study == "locked15"]),
             ("seen\nwhole grids", QG[QG.group == "oos_seen"]), ("unseen\nwhole grids", QG[QG.group == "oos_unseen"])]
    YTOP = 120.0
    axw.set_yscale("log"); axw.set_ylim(0.06, YTOP); axw.set_xlim(-0.8, 12.3); axw.minorticks_off()
    axw.axhspan(0.06, 1, color="#eef3f7", zorder=0); axw.axhline(1, color=INK, lw=1.4, zorder=1)
    axw.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 100]); axw.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "30", "100"], fontsize=11)
    axw.tick_params(axis="y", length=0); axw.tick_params(axis="x", length=0, labelbottom=False)
    axw.grid(axis="y", color=GRID, lw=0.7, zorder=0); axw.set_axisbelow(True); clean(axw)
    axw.set_ylabel("GENCO error / WLS error", fontsize=11.5)
    variants = [("ratio_inf", "s", INK, INK, "informed WLS"), ("ratio_flat", "^", GREY, SURF, "limit-blind WLS")]
    qx0 = [0.6, 3.5, 6.4, 9.3]
    qout = []
    for (lab, d), xc in zip(qsets, qx0):
        rec = {"set": lab.replace("\n", " "), "n": int(len(d))}
        for vi, (col, mk, fc, ec, vname) in enumerate(variants):
            x = xc + (-0.62 if vi == 0 else 0.62)
            r = d[col].to_numpy(dtype=float)
            order = np.argsort(r); jit = np.zeros(len(r)); placed = []
            step = 0.12
            for i in order:
                k = 0
                while k < 9 and any(abs(np.log(r[i]) - np.log(r[j])) < 0.13 and abs(jit[j] - (step * ((k + 1) // 2) * (1 if k % 2 else -1))) < 0.11 for j in placed): k += 1
                jit[i] = float(np.clip(step * ((k + 1) // 2) * (1 if k % 2 else -1), -0.46, 0.46)); placed.append(i)
            yv = np.minimum(r, YTOP)
            axw.plot(x + jit, yv, mk, ms=7.5, mfc=fc, mec=INK if ec == SURF else ec, mew=1.4, ls="none", zorder=3, clip_on=False)
            med = float(np.median(r))
            axw.plot([x - 0.5, x + 0.5], [med, med], color=INK, lw=2.4, solid_capstyle="butt", zorder=4)
            axw.text(x, 0.0565, f"{med:.2g}" if med < 10 else f"{med:.0f}", ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK, clip_on=False)
            big = r > YTOP
            if big.any():
                axw.text(x + (0.55 if vi else -0.55), YTOP * 1.12, f"{int(big.sum())} more,\nup to {r[big].max():,.0f}", ha="left" if vi else "right", va="bottom",
                         fontsize=9.5, color=INK2, linespacing=0.95, clip_on=False)
            rec[vname] = {"median": med, "n": int(len(r)), "ratios": [float(v) for v in r], "n_off_scale": int(big.sum())}
        qout.append(rec)
        axw.text(xc, 0.0285, lab, ha="center", va="top", fontsize=11.5, color=INK, linespacing=0.95, clip_on=False)
    axw.text(12.2, 0.55, "GENCO\nbetter", fontsize=11, color=INK2, fontweight="bold", va="top", ha="right", linespacing=0.95)
    axw.text(12.2, 1.1, "WLS\nbetter", fontsize=11, color=INK2, fontweight="bold", va="bottom", ha="right", linespacing=0.95)
    # direct labels instead of a legend
    axw.plot(-0.3, 330, "s", ms=8, mfc=INK, mec=INK, clip_on=False)
    axw.text(-0.05, 330, "WLS with the true\ngenerator limits", fontsize=10, color=INK, va="center", ha="left", linespacing=0.95, clip_on=False)
    axw.plot(3.9, 330, "^", ms=8, mfc=GREY, mec=INK, mew=1.4, clip_on=False)
    axw.text(4.2, 330, "limit-blind WLS\n(every Pmax = median)", fontsize=10, color=INK, va="center", ha="left", linespacing=0.95, clip_on=False)
    axw.text(8.0, 330, "one marker per grid\nbar and number = median", fontsize=10, color=INK2, va="center", ha="left", style="italic", linespacing=0.95, clip_on=False)
    PLOTTED["a_qg"] = qout
    # win counts (GENCO beats WLS), kept from the earlier panel and moved to the caption
    wout = []
    for lab, d, has_inf in sets:
        n = len(d)
        wout.append({"set": lab, "n": n, "i1_i5": int(d.win5.sum()), "informed": int(d.win6.sum()) if has_inf else None, "limit_blind": int(d.win6flat.sum())})
    PLOTTED["a_wls"] = wout
    fig.text(0.02, 0.012, textwrap.fill("Bottom panel: GENCO error / WLS error on the unmeasured buses at 40% measured, one marker per grid (above 1 = GENCO worse). "
                                        "WLS = weighted-least-squares state estimator. Informed uses the true generator limits; "
                                        "limit-blind sets every Pmax to the median. 10 to 30 scenarios per grid. WLS was mostly iteration-capped "
                                        "(61% of informed solves hit the cap; limit-blind: 88% of carved-piece and 98% of seen-family whole-grid scenarios), "
                                        "so its error is that of a capped iterate. The six SimBench high-voltage grids, where WLS converges (ratios 94 to 5,016), sit at the top; values above 120 are clipped. "
                                        "WLS counts checked in two audit rounds.", 98),
             fontsize=9, color=INK2, va="bottom", ha="left", linespacing=1.15)
    draw_key(fig, 0.987, [("seen family", BLUE, BLUE), ("unseen family", VERM, VERM), ("locked benchmark (15 grids)", GREEN, GREEN)])
    fig.savefig(OUT / "fig_a_voltage_inference.png", dpi=DPI, facecolor=SURF); plt.close(fig)


# ------------------------------------------------------------------ (b) transfer 12k contrast
def fig_b():
    c = json.load(open(R / "2026-10-20-genco-transfer/contrast_12k.json"))
    RND, PFM = None, PURP    # random mask = hollow bar; arms that use the power-flow-task mask = purple
    groups = [("GENCO's original data only", [("C00", "random mask", RND), ("C01", "50/50 mix of both masks", PFM)], None),
              ("our data + original data", [("D", "random mask", RND), ("b", "half-then-mix (random so far)", RND)], "batches 2.34x larger"),
              ("our data + original, power-flow mask", [("a", "power-flow-task mask only", PFM), ("c", "power-flow curriculum", PFM),
                                                       ("m", "50/50 mix of both masks", PFM)], "batches 2.34x larger")]
    fig = plt.figure(figsize=(7.5, 5.7))
    ax = fig.add_axes([0.335, 0.12, 0.33, 0.74])
    from matplotlib.transforms import blended_transform_factory
    htr = blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0; yt = []; rows = []; heads = []
    for gname, arms, sub in groups:
        heads.append((y - 0.62, gname, sub));
        for k, lab, col in arms:
            rows.append((y, k, lab, col)); y += 1.0
        y += 0.75
    ymax = y - 0.75
    ax.set_ylim(ymax - 0.4, -1.55)
    ax.set_xlim(0, 150)
    out = []
    for yy, k, lab, col in rows:
        v = 100 * c[k]["change_vs_released"]
        if col is None: ax.barh(yy, v, height=0.68, facecolor=SURF, edgecolor=INK2, linewidth=2.0, zorder=3)
        else: ax.barh(yy, v, height=0.68, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(v + 2, yy, f"+{v:.1f}%", va="center", fontsize=12, fontweight="bold")
        ax.text(-3, yy, lab, ha="right", va="center", fontsize=11)
        out.append({"arm": k, "change_pct": v, "angle_better": int(c[k]["angle_better"]), "V_worse": int(c[k]["V_worse"])})
    for hy, gname, sub in heads:
        ax.text(0.02, hy, gname, ha="left", va="center", fontsize=11, color=INK2, fontweight="bold", transform=htr)
        if sub: ax.text(150, hy, sub, ha="right", va="center", fontsize=10, color=INK2, style="italic")
    ax.axvline(0, color=INK, lw=2.0, zorder=4)
    ax.text(0, -1.25, "released GENCO", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=11)
    ax.set_yticks([]); ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); clean(ax)
    ax.set_xlabel("% worse than released, branch angle\n(median over the 15 locked grids)", fontsize=10, linespacing=1.1)
    # per-grid strips (15 squares, 5 x 3 block)
    sx = [0.705, 0.86]
    fig.text(sx[0] + 0.065, 0.905, "|V| worse\nthan released", ha="center", va="bottom", fontsize=10, color=INK2, linespacing=0.95)
    fig.text(sx[1] + 0.065, 0.905, "angle better\nthan released", ha="center", va="bottom", fontsize=10, color=INK2, linespacing=0.95)
    for (yy, k, lab, col), o in zip(rows, out):
        for sxi, key, fillc in ((sx[0], "V_worse", INK), (sx[1], "angle_better", INK)):
            n = o[key]
            # axes-fraction -> figure coords for this row
            yfig = ax.transData.transform((0, yy))[1] / fig.bbox.height
            cell_w, cell_h = 0.0215, 0.0175
            for i in range(15):
                rr, cc = divmod(i, 5)
                rect = Rectangle((sxi + cc * cell_w * 1.02, yfig + 0.026 - rr * cell_h * 1.12 - cell_h * 0.5), cell_w * 0.88, cell_h * 0.88,
                                 transform=fig.transFigure, facecolor=fillc if i < n else "#ffffff", edgecolor=INK2 if i >= n else INK, lw=0.8)
                fig.add_artist(rect)
            fig.text(sxi + 5 * cell_w * 1.02 + 0.008, yfig, f"{n}", fontsize=11.5, fontweight="bold", va="center", ha="left")
    fig.text(0.705, 0.035, "one square = one of the\n15 locked grids", fontsize=10, color=INK2, va="bottom", linespacing=1.0)
    draw_key(fig, 0.975, [("uses the power-flow-task mask", PURP, PURP), ("random mask only", SURF, INK2)])
    fig.text(0.02, 0.925, "all runs at matched 12,000 updates", fontsize=10.5, color=INK2, va="center", style="italic")
    # group-batch note
    PLOTTED["b_rows"] = out
    fig.savefig(OUT / "fig_b_transfer_12k.png", dpi=DPI, facecolor=SURF); plt.close(fig)


# ------------------------------------------------------------------ (c) unlearning
def fig_c():
    Pj = json.load(open(R / "2026-10-20-genco-continued-pretraining/pooled.json"))
    pres = json.load(open(R / "2026-10-20-genco-continued-pretraining/preservation.json"))
    steps = {"step0": 0, "step25": 25, "step50": 50, "step100": 100, "step250": 250, "step500": 500, "step1000": 1000, "final": 5000}
    arms = (("fixed", SKY, "corrected labels"), ("published", CHAR, "original labels"))
    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_axes([0.14, 0.45, 0.70, 0.47]); ax2 = fig.add_axes([0.25, 0.09, 0.59, 0.20])
    ax.axhspan(0.5, 0.97, color="none")
    out = {}
    for arm, col, lab in arms:
        pts = sorted((steps[p["state"]], [p[s]["frac_lt_0.5"] for s in ("1000", "1001", "1002")]) for p in Pj if p["arm"] == arm and p["state"] in steps)
        x = [max(u, 10) for u, _ in pts]
        seeds = np.array([v for _, v in pts]); mean = seeds.mean(axis=1)
        for j in range(3): ax.plot(x, seeds[:, j], color=col, lw=1.0, alpha=0.35, zorder=2)
        ax.plot(x, mean, color=col, lw=3.0, marker="o", ms=8, mec=SURF, mew=1.6, zorder=4)
        out[arm] = {"updates": [u for u, _ in pts], "mean": [float(m) for m in mean], "seeds": seeds.tolist()}
    m_fix, m_pub = out["fixed"]["mean"], out["published"]["mean"]
    ax.text(7500, m_fix[-1], "corrected  " + f"{m_fix[-1]:.2f}", color=INK, fontsize=11.5, fontweight="bold", va="center", ha="left")
    ax.text(7500, m_pub[-1], "original  " + f"{m_pub[-1]:.2f}", color=INK, fontsize=11.5, fontweight="bold", va="center", ha="left")
    ax.annotate(f"released checkpoint {m_fix[0]:.2f}", xy=(10, m_fix[0]), xytext=(14, 0.78), fontsize=11, color=INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1), linespacing=0.95)
    ax.text(1000, 0.70, "thin lines: the 3 mask seeds", fontsize=10, color=INK2, ha="center", va="center")
    ax.set_xscale("log"); ax.set_xlim(8, 40000); ax.minorticks_off(); ax.set_ylim(0.0, 1.04)
    ax.set_xticks([10, 25, 50, 100, 250, 500, 1000, 5000]); ax.set_xticklabels(["0", "25", "50", "100", "250", "500", "1k", "final"], fontsize=11)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=11)
    ax.tick_params(length=0); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); clean(ax)
    ax.set_xlabel("continued-pretraining updates (log scale; final is about 5k)", fontsize=10.5)
    ax.set_ylabel("fraction of hidden collapsed buses\nstill predicted below 0.5 pu (axis from 0)", fontsize=11, linespacing=1.0)
    # preservation
    items = [("branch angle", "final/retrain_fixed", "final/retrain_published", "median_angle_ratio_to_released"),
             ("|V|", "final/retrain_fixed", "final/retrain_published", "median_V_ratio_to_released")]
    yy = 0.0; pout = []
    for lab, kf, kp, mkey in items:
        for k, col in ((kf, SKY), (kp, CHAR)):
            v = pres[k][mkey]
            ax2.barh(yy, v - 1, left=1, height=0.72, color=col, zorder=3, edgecolor=SURF, lw=1.2)
            ax2.text(v + 0.04, yy, f"{v:.2f}x", va="center", fontsize=12, fontweight="bold")
            pout.append({"metric": lab, "arm": k, "ratio": v}); yy += 0.85
        ax2.text(0.99, yy - 1.3, lab, ha="right", va="center", fontsize=11.5)
        yy += 0.55
    ax2.set_ylim(yy - 0.4, -1.1); ax2.set_xlim(1.0, 2.6)
    ax2.axvline(1.0, color=INK, lw=2.0, zorder=4)
    ax2.axvline(1.10, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax2.text(1.12, -0.7, "tolerance 1.10", fontsize=10, color=INK2, ha="left", va="center")
    ax2.text(0.985, -0.7, "released", fontsize=10, color=INK, ha="right", va="center", fontweight="bold")
    ax2.set_yticks([]); ax2.set_xticks([1.0, 1.5, 2.0, 2.5]); ax2.set_xticklabels(["1x", "1.5x", "2x", "2.5x"], fontsize=11); ax2.tick_params(length=0)
    ax2.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax2.set_axisbelow(True); clean(ax2)
    ax2.set_xlabel("held-out error vs released, median of 15 grids (higher is worse)", fontsize=10.5)
    draw_key(fig, 0.975, [("trained on corrected labels", SKY, SKY), ("trained on original labels", CHAR, CHAR)])
    fig.text(0.02, 0.345, "Held-out error after training, relative to the released model", fontsize=11.5, va="center", color=INK2)
    PLOTTED["c_curves"] = out; PLOTTED["c_preservation"] = pout; PLOTTED["c_ylim"] = list(ax.get_ylim())
    fig.savefig(OUT / "fig_c_unlearning.png", dpi=DPI, facecolor=SURF); plt.close(fig)


def compare(old, new, name):
    a = Image.open(old).convert("RGB"); b = Image.open(new).convert("RGB")
    h = 1500
    a = a.resize((int(a.width * h / a.height), h), Image.LANCZOS); b = b.resize((int(b.width * h / b.height), h), Image.LANCZOS)
    W = a.width + b.width + 90; canvas = Image.new("RGB", (W, h + 70), (252, 252, 251))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    canvas.paste(a, (30, 60)); canvas.paste(b, (60 + a.width, 60))
    d.text((30, 20), "BEFORE (current post figure)", fill=(0, 0, 0)); d.text((60 + a.width, 20), "AFTER (v3.1)", fill=(0, 0, 0))
    canvas.save(CMP / f"{name}_before_after.png")


if __name__ == "__main__":
    fig_a(); fig_b(); fig_c()
    (HERE / "plotted_numbers.json").write_text(json.dumps(PLOTTED, indent=1))
    old_a = R / "2026-10-21-genco-voltage-inference-summary-fig/summary_fig.png"
    old_b = R / "2026-10-18-genco-challenge-post/figures-v2/fig_transfer_12k.png"
    old_c = R / "2026-10-18-genco-challenge-post/figures-v2/fig_b_unlearning.png"
    compare(old_a, OUT / "fig_a_voltage_inference.png", "fig_a")
    compare(old_b, OUT / "fig_b_transfer_12k.png", "fig_b")
    compare(old_c, OUT / "fig_c_unlearning.png", "fig_c")
    print("done")
