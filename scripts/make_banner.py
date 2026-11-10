"""README banner -> docs/assets/banner.png (1680 x 624), in the style of the thainowcast banner.

The image on the left is real simulator output, not an illustration: reported arrears (the bureau's
delinquency state, 0-3) of 260 borrowers from the bureau simulator with a moratorium, one row per
borrower, one column per month. Deferred months are blue: there the bureau reports frozen arrears.

Run: make banner
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from bureau_to_workout import bureau  # noqa: E402
from bureau_to_workout.paths import ROOT  # noqa: E402

BG, AMBER, INK, MUTED = "#04070f", "#f5a623", "#e8e9ec", "#8f9bb3"
W, H, DPI = 1680, 624, 100


def main() -> None:
    p = bureau.simulate_bureau(bureau.SimConfig(n_borrowers=20_000, moratorium=True, seed=7))
    s, mor = p.s_obs, p.mor
    # borrowers who are current at the start and fall behind or are deferred later
    ever = np.flatnonzero((s[:, 0] == 0) & (((s > 0).sum(axis=1) >= 3) | mor.any(axis=1)))
    pick = np.random.default_rng(0).choice(ever, size=260, replace=False)
    rows, deferred = s[pick].astype(float), mor[pick]
    first = np.where((rows > 0) | deferred, 1, 0).argmax(axis=1)
    order = np.lexsort((-rows.sum(axis=1), first))
    rows, deferred = rows[order], deferred[order]
    rows[deferred] = 4  # deferred months: arrears reported as frozen

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.035, 0.09, 0.27, 0.80], facecolor=BG)
    cmap = ListedColormap(["#0b1222", "#6b420d", "#c27d14", AMBER, "#3b5b8c"])
    ax.imshow(rows, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    lo, hi = p.cfg.shock_start, p.cfg.shock_end
    ax.annotate("", xy=(lo - 0.5, -5), xytext=(hi + 0.5, -5), annotation_clip=False,
                arrowprops=dict(arrowstyle="|-|", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=3))
    ax.text((lo + hi) / 2, -9, "moratorium", color="#7fa3d6", ha="center", va="bottom", fontsize=10.5)
    ax.set_axis_off()

    fig.text(0.35, 0.60, "bureau-to-workout", color=AMBER, fontsize=58, fontweight="bold", va="baseline")
    fig.text(0.352, 0.41, "Credit scoring and causal debt resolution\nunder policy-shaped labels",
             color=INK, fontsize=19, va="baseline", linespacing=1.5)
    fig.text(0.352, 0.25, "selection at approval · masking by deferral · treatment in the workout",
             color=MUTED, fontsize=13.5, va="baseline")
    fig.text(0.985, 0.035,
             "Bureau simulator: reported arrears of 260 borrowers over 60 months (amber = 30 / 60 / 90+ DPD, "
             "blue = deferred, arrears frozen) · synthetic data",
             color="#7d879c", fontsize=9, ha="right")
    out = ROOT / "docs" / "assets" / "banner.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
