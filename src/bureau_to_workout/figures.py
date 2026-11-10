"""
bureau_to_workout.figures
Builds every figure in the paper from paper_results/ into paper/figures/ (vector PDF + PNG preview).
One style for all figures: STIX (Times-compatible) type, Okabe-Ito colour-blind-safe palette,
the same colour for the same method everywhere, and layouts sized to the 6.5-inch text width.

Run: make figures
"""
from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

from .paths import PAPER, RESULTS as RES   # noqa: E402

FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titleweight": "normal", "axes.titlelocation": "left",
})

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C"}
INK, QUIET, EDGE = "#1A1A1A", "#5A5A5A", "#4D4D4D"
TEXT_W = 6.5   # inches


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=6)


# =============================================================================================
# Figure 1: the credit lifecycle seen by the linked data, with the three label mechanisms
# =============================================================================================

def fig_lifecycle():
    fig = plt.figure(figsize=(TEXT_W, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 48.5); ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def lane(y0, h, label, sub):
        ax.add_patch(FancyBboxPatch((5.0, y0), 94, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#F3F5F7", ec="none", zorder=0))
        t = ax.text(6.6, y0 + h - 1.6, label, fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left")
        bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        ax.text(bb.x1 + 1.2, y0 + h - 1.6, sub, fontsize=7.5, color=QUIET, va="top", ha="left")

    lane(26.5, 21.0, "Credit bureau (NCB)", "account-month histories across all member lenders")
    lane(2.0, 20.5, "Asset-management company (SAM)", "acquired NPLs, offers and workout outcomes")

    def box(cx, cy, w, h, l1, l2):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc="white", ec=EDGE, lw=0.8, zorder=2))
        ax.text(cx, cy + 1.25, l1, ha="center", va="center", fontsize=8, color=INK, zorder=3)
        ax.text(cx, cy - 1.75, l2, ha="center", va="center", fontsize=7, color=QUIET, zorder=3)

    def arrow(p0, p1, ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=EDGE, lw=0.9,
                                     linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))

    yt, yb, h, w = 33.5, 9.5, 7.0, 14.0
    top = [(15.0, "Application", "inquiry and decision"), (33.5, "Performing", "current"),
           (52.0, "Delinquent", "30 to 89 DPD"), (70.5, "Non-performing", "90+ DPD"),
           (89.0, "Transferred", "sold to the AMC")]
    for x, a, b in top:
        box(x, yt, w, h, a, b)
    for (x0, *_), (x1, *_) in zip(top[:-1], top[1:]):
        arrow((x0 + w / 2, yt), (x1 - w / 2, yt))

    wo = 18.5
    box(89.0, yb, w, h, "Acquisition", "loan tape and price")
    box(70.5, yb, wo, h, "Offer", "extend or write down")
    box(52.0, yb, w, h, "Resolution", "repaid, settled, legal")
    box(33.5, yb, w, h, "Re-entry", "new bureau accounts")
    arrow((89.0, yt - h / 2), (89.0, yb + h / 2))
    arrow((89.0 - w / 2, yb), (70.5 + wo / 2, yb))
    arrow((70.5 - wo / 2, yb), (52.0 + w / 2, yb))
    arrow((52.0 - w / 2, yb), (33.5 + w / 2, yb))
    # re-entry returns to the bureau: routed along the left margin, clear of every label
    ax.plot([33.5 - w / 2, 2.6, 2.6], [yb, yb, yt], color=EDGE, lw=0.9, ls=(0, (3, 2)), zorder=1,
            solid_capstyle="butt")
    arrow((2.6, yt), (15.0 - w / 2, yt), ls=(0, (3, 2)))

    def tag(x, y, sym, text, side="right"):
        ax.add_patch(plt.Circle((x, y), 1.5, fc=OI["red"], ec="none", zorder=4))
        ax.text(x, y - 0.05, sym, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if side == "right":
            ax.text(x + 2.3, y, text, ha="left", va="center", fontsize=7.5, color=OI["red"], zorder=5)
        else:
            ax.text(x - 2.3, y, text, ha="right", va="center", fontsize=7.5, color=OI["red"], zorder=5)

    tag(9.6, 41.4, "S", "selection: outcome seen only if approved")
    tag(46.6, 41.4, "M", "masking: deferral freezes reported DPD")
    tag(63.6, 16.9, "T", "treatment: the offer changes the outcome", side="left")
    save(fig, "fig1_lifecycle")



# =============================================================================================
# Helpers for multi-seed summaries
# =============================================================================================

def seed_stats(fname, key, col):
    d = pd.read_csv(RES / fname)
    g = d.groupby(key, sort=False)[col]
    return g.mean(), g.std()


def dotplot(ax, labels, means, sds, colors, ref=None, xlabel="", xlim=None, fmt="{:.2f}", show_values=True):
    y = np.arange(len(labels))[::-1]
    if ref is not None:
        ax.axvline(ref, color=QUIET, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for yi, m, sd, c in zip(y, means, sds, colors):
        ax.errorbar(m, yi, xerr=sd, fmt="o", color=c, ecolor=c, elinewidth=1.0, capsize=2.5, ms=4.5, zorder=3)
        if show_values:
            ax.text(m, yi + 0.28, fmt.format(m), ha="center", va="bottom", fontsize=7, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)


# =============================================================================================
# Figure 2 (E1): out-of-time benchmark
# =============================================================================================

E1_ORDER = ["WoE scorecard", "Monotone GBM", "SSL embedding only (linear probe)", "GBM + SSL embedding"]
E1_SHORT = ["WoE scorecard", "Monotone GBM", "SSL embedding only", "GBM + SSL embedding"]
E1_COL = [OI["orange"], OI["blue"], OI["purple"], OI["green"]]


def fig_e1():
    d = pd.read_csv(RES / "e1_all.csv")
    g = d.groupby("model", sort=False)
    m, sd = g.mean(numeric_only=True).loc[E1_ORDER], g.std(numeric_only=True).loc[E1_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.55), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.14})
    ax = axes[0]
    y = np.arange(4)[::-1]
    for yi, i in zip(y, range(4)):
        ax.errorbar(m["AUC"].iloc[i], yi + 0.14, xerr=sd["AUC"].iloc[i], fmt="o", color=E1_COL[i], ms=4.5,
                    capsize=2.5, elinewidth=1.0, zorder=3)
        ax.errorbar(m["AUC thin-file"].iloc[i], yi - 0.14, xerr=sd["AUC thin-file"].iloc[i], fmt="o", mfc="white",
                    color=E1_COL[i], ms=4.5, capsize=2.5, elinewidth=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(E1_SHORT); ax.set_ylim(-1.0, 3.6); ax.grid(axis="y", visible=False)
    ax.set_xlim(0.68, 0.84); ax.set_xticks([0.70, 0.74, 0.78, 0.82]); ax.set_xlabel("Out-of-time AUC")
    h1 = ax.errorbar([], [], fmt="o", color=INK, ms=4.5, label="all borrowers")
    h2 = ax.errorbar([], [], fmt="o", color=INK, mfc="white", ms=4.5, label="thin-file borrowers")
    ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_title("(a) Discrimination", loc="left", pad=20)

    ax = axes[1]
    ax.axvspan(0.10, 0.25, color="#F6EBDD", zorder=0, lw=0)
    ax.axvspan(0.25, 0.70, color="#F6DCD4", zorder=0, lw=0)
    for yi, i in zip(y, range(4)):
        ax.errorbar(m["PSI dev->OOT"].iloc[i], yi, xerr=sd["PSI dev->OOT"].iloc[i], fmt="o", color=E1_COL[i],
                    ms=4.5, capsize=2.5, elinewidth=1.0, zorder=3)
        ax.text(m["PSI dev->OOT"].iloc[i], yi + 0.25, f"{m['PSI dev->OOT'].iloc[i]:.2f}", ha="center", va="bottom",
                fontsize=7, color=E1_COL[i])
    ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.0, 3.6); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.70); ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xlabel("PSI, development vs. out-of-time")
    ax.text(0.175, -0.72, "moderate", ha="center", va="center", fontsize=6.8, color=QUIET)
    ax.text(0.475, -0.72, "significant", ha="center", va="center", fontsize=6.8, color=QUIET)
    ax.set_title("(b) Score stability", loc="left", pad=20)
    save(fig, "fig2_e1_benchmark")


# =============================================================================================
# Figure 3 (E2): selective labels
# =============================================================================================

E2_ORDER = ["Approved-only (AO)", "AO + IPW on observables", "Fuzzy augmentation", "Bureau-augmented (BA)",
            "BA + IPW for unobserved rejects", "Oracle (all applicants, true Y)"]
E2_SHORT = ["Approved-only", "Approved-only + IPW", "Fuzzy augmentation", "Bureau-augmented",
            "Bureau-augmented + IPW", "Oracle (all labels)"]
E2_COL = [OI["red"], OI["purple"], OI["orange"], OI["blue"], OI["sky"], OI["black"]]


def fig_e2():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.75), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.62})
    ax = axes[0]
    ax.plot([0, 0.40], [0, 0.40], color=QUIET, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for name, short, c in zip(E2_ORDER, E2_SHORT, E2_COL):
        if name in ("AO + IPW on observables", "BA + IPW for unobserved rejects"):
            continue
        r = pd.read_csv(RES / f"fig_e2_rel_{name}.csv")
        ls = (0, (3, 2)) if name.startswith("Oracle") else "-"
        ax.plot(r["pred"], r["actual"], marker="o", ms=3.2, color=c, lw=1.2, ls=ls, label=short, zorder=3)
    ax.set_xlim(0, 0.40); ax.set_ylim(0, 0.42)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Predicted default probability"); ax.set_ylabel("Observed default rate, $Y(1)$")
    ax.text(0.325, 0.333, "perfect calibration", rotation=45, rotation_mode="anchor", transform_rotates_text=True,
            fontsize=6.8, color=QUIET, ha="center", va="bottom")
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1, fontsize=7.2, labelspacing=0.3)
    ax.set_title("(a) Calibration on all applicants", loc="left", pad=6)

    ax = axes[1]
    mu, sd = seed_stats("e2_all.csv", "method", "PD/actual rejected")
    dotplot(ax, E2_SHORT, mu.loc[E2_ORDER].values, sd.loc[E2_ORDER].values, E2_COL, ref=1.0,
            xlabel="Predicted ÷ observed default rate", xlim=(0.3, 1.1))
    ax.set_title("(b) Bias on rejected applicants", loc="left", pad=6)
    save(fig, "fig3_e2_selective_labels")


# =============================================================================================
# Figure 4 (E3): label masking
# =============================================================================================

E3_ORDER = ["Naive: reported labels", "Exclude deferred windows", "Delayed labels (+6m after exit)",
            "Oracle: no-deferral counterfactual"]
E3_SHORT = ["Naive", "Exclude deferred", "Delayed labels", "Oracle"]
E3_COL = [OI["red"], OI["orange"], OI["blue"], OI["black"]]


def fig_e3():
    fig = plt.figure(figsize=(TEXT_W, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ts = pd.read_csv(RES / "fig_e3_timeseries.csv")
    ax.axvspan(23.5, 35.5, color="#EDEFF2", lw=0, zorder=0)
    ax.text(29.5, 0.8, "moratorium", ha="center", va="bottom", fontsize=7, color=QUIET)
    ax.plot(ts["t0"], 100 * ts["counterfactual"], color=OI["black"], ls=(0, (3, 2)), marker="o", ms=3,
            label="true (no deferral)")
    ax.plot(ts["t0"], 100 * ts["reported"], color=OI["red"], marker="o", ms=3, label="reported")
    ax.set_xlabel("Observation month $t_0$"); ax.set_ylabel("12-month default rate (%)")
    ax.set_ylim(0, 21); ax.set_xlim(4, 47)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(loc="upper left", borderaxespad=0.2)
    ax.set_title("(a) What the bureau reports", loc="left", pad=6)

    d = pd.read_csv(RES / "e3_all.csv")
    g = d.groupby("strategy", sort=False)
    m, sd = g.mean(numeric_only=True).loc[E3_ORDER], g.std(numeric_only=True).loc[E3_ORDER]
    ax2 = fig.add_subplot(gs[1])
    dotplot(ax2, E3_SHORT, m["OOT AUC"].values, sd["OOT AUC"].values, E3_COL, xlabel="Out-of-time AUC",
            xlim=(0.55, 0.88))
    ax2.set_title("(b) Discrimination", loc="left", pad=6)
    ax3 = fig.add_subplot(gs[2], sharey=ax2)
    dotplot(ax3, E3_SHORT, m["relative calibration, deferred (ideal 1)"].values,
            sd["relative calibration, deferred (ideal 1)"].values, E3_COL, ref=1.0,
            xlabel="Relative calibration, deferred", xlim=(0, 1.1))
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.tick_params(axis="y", length=0)
    ax3.set_title("(c) Deferred borrowers", loc="left", pad=6)
    fig.subplots_adjust(left=0.07, right=0.99)
    pos = ax2.get_position()
    ax2.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.03, pos.height])
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0 + 0.03, pos3.y0, pos3.width - 0.01, pos3.height])
    save(fig, "fig4_e3_masking")


# =============================================================================================
# Figure 5 (E4): groups
# =============================================================================================

def fig_e4():
    d = pd.read_csv(RES / "e4_all.csv")
    g = d.groupby("group", sort=False)
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    pretty = {"gender=F": "Gender: female", "gender=M": "Gender: male", "region=BKK": "Region: Bangkok",
              "region=Central": "Region: Central", "region=North": "Region: North",
              "region=Northeast": "Region: Northeast", "region=South": "Region: South",
              "occupation=informal": "Occupation: informal", "occupation=salaried": "Occupation: salaried",
              "occupation=self_employed": "Occupation: self-employed", "age_band=<30": "Age: under 30",
              "age_band=30-44": "Age: 30 to 44", "age_band=45-59": "Age: 45 to 59", "age_band=60+": "Age: 60+",
              "thin_file=False": "Thick file", "thin_file=True": "Thin file"}
    order = list(pretty)
    labels = [pretty[k] for k in order]
    fam = {"gender": OI["grey"], "region": OI["sky"], "occupation": OI["red"], "age_band": OI["purple"],
           "thin_file": OI["green"]}
    colors = [fam[k.split("=")[0]] for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.35), sharey=True, gridspec_kw={"wspace": 0.08})
    dotplot(axes[0], labels, m.loc[order, "relative calibration"].values, sd.loc[order, "relative calibration"].values,
            colors, ref=1.0, xlabel="Relative calibration (1 = same as portfolio)", xlim=(0.55, 1.45), show_values=False)
    axes[0].set_title("(a) Calibration by group", loc="left", pad=6)
    dotplot(axes[1], labels, 100 * m.loc[order, "approval among goods"].values,
            100 * sd.loc[order, "approval among goods"].values, colors, ref=None,
            xlabel="Approval rate among non-defaulters (%)", xlim=(55, 80), show_values=False)
    axes[1].axvline(100 * m.loc[order, "approval among goods"].mean(), color=QUIET, lw=0.8, ls=(0, (3, 2)))
    axes[1].set_title("(b) Approval at one cut-off (70% overall)", loc="left", pad=6)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[1].tick_params(axis="y", length=0)
    for ax in axes:
        for yv in (13.5, 8.5, 5.5, 1.5):
            ax.axhline(yv, color="#D9D9D9", lw=0.6, zorder=0)
    save(fig, "fig5_e4_groups")


# =============================================================================================
# Figure 6 (R2, R3, gain curves): causal targeting of offers
# =============================================================================================

def fig_resolution():
    r3 = pd.read_csv(RES / "r3_all.csv")
    r2 = pd.read_csv(RES / "r2_all.csv")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.75), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    style = {"Oracle": (OI["black"], (0, (3, 2))), "Causal (DR-learner)": (OI["blue"], "-"),
             "Predictive": (OI["red"], "-")}
    for name, (c, ls) in style.items():
        g = r3[r3.policy == name].groupby("budget (% of balance)")["recovery (% of balance)"]
        mu, sd = g.mean(), g.std()
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
        ax.plot(mu.index, mu.values, color=c, ls=ls, marker="o", ms=2.8)
    sq = r2[r2.policy == "Status quo (logged offers)"][["write-downs (% of balance)", "recovery (% of balance)"]].mean()
    ax.plot(sq.iloc[0], sq.iloc[1], marker="D", ms=5, color=OI["orange"], ls="none")
    ax.annotate("status quo", (sq.iloc[0], sq.iloc[1]), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.2, color=OI["orange"])
    ax.set_xlabel("Write-down budget (% of balance)"); ax.set_ylabel("Recovery value (% of balance)")
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(55.0, 58.5); ax.set_xticks([0, 5, 10, 15])
    ax.set_title("(a) Budget frontier", loc="left", pad=6)

    names = [("Oracle", OI["black"], (0, (3, 2))), ("DR-learner", OI["blue"], "-"), ("T-learner", OI["sky"], "-"),
             ("Predictive (low score first)", OI["red"], "-"), ("Random", OI["grey"], (0, (1, 1.5)))]
    for ax, arm, letter, title in [(axes[1], "Term extension", "b", "Gain: term extension"),
                                   (axes[2], "Extension + write-down", "c", "Gain: extension + write-down")]:
        cv = pd.read_csv(RES / f"fig_r_curve_{arm}.csv")
        for name, c, ls in names:
            ax.plot(100 * cv["fraction"], cv[name], color=c, ls=ls, lw=1.3)
        ax.axhline(0, color=QUIET, lw=0.6)
        ax.set_xlabel("Cases offered (%)")
        ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"({letter}) {title}", loc="left", pad=6)
    axes[1].set_ylabel("Cumulative gain (% of balance)")
    handles = [mpl.lines.Line2D([], [], color=OI["black"], ls=(0, (3, 2)), lw=1.3, label="Oracle"),
               mpl.lines.Line2D([], [], color=OI["blue"], lw=1.3, label="Causal, DR-learner"),
               mpl.lines.Line2D([], [], color=OI["sky"], lw=1.3, label="Causal, T-learner"),
               mpl.lines.Line2D([], [], color=OI["red"], lw=1.3, label="Predictive targeting"),
               mpl.lines.Line2D([], [], color=OI["grey"], ls=(0, (1, 1.5)), lw=1.3, label="Random order")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), handlelength=2.0,
               columnspacing=1.4)
    fig.subplots_adjust(bottom=0.27)
    save(fig, "fig6_resolution")


# =============================================================================================
# Figure 7 (R4): off-policy evaluation
# =============================================================================================

def fig_ope():
    d = pd.read_csv(RES / "r4_all.csv")
    lo, hi = 50.5, 61.5
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, est, c, title in [(axes[0], "DM", OI["orange"], "(a) Direct method"),
                              (axes[1], "IPS", OI["red"], "(b) Inverse propensity"),
                              (axes[2], "DR", OI["blue"], "(c) Doubly robust")]:
        ax.plot([lo, hi], [lo, hi], color=QUIET, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.scatter(d["true"], d[est], s=11, color=c, alpha=0.8, edgecolor="white", linewidth=0.3, zorder=3)
        mae = np.mean(np.abs(d[est] - d["true"]))
        ax.text(0.96, 0.05, f"mean abs. error\n{mae:.2f} pp", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.2, color=INK, linespacing=1.15)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xticks([52, 54, 56, 58, 60]); ax.set_yticks([52, 54, 56, 58, 60])
        ax.set_xlabel("True recovery (% of balance)")
        ax.set_title(title, loc="left", pad=6)
    axes[0].set_ylabel("Estimated recovery (% of balance)")
    save(fig, "fig7_ope")


FIGS = {"fig1": fig_lifecycle, "fig2": fig_e1, "fig3": fig_e2, "fig4": fig_e3, "fig5": fig_e4,
        "fig6": fig_resolution, "fig7": fig_ope}


def main(which=None):
    for k in which or list(FIGS):
        FIGS[k]()
        print("built", k)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
