"""Supplementary Figure S3 — entropy-penalty (λ) sensitivity of the five subtypes.

Addresses the concern that the load-balancing weight λ=1.0 manufactures the fifth subtype. (a) Number
of populated archetypes (≥3%) and the minimum subtype fraction vs λ: without the penalty (λ=0) the
model collapses (4 populated, one empty, a 76% dominant blob); for every λ≥0.3 all five are populated
and balanced. (b) Each subtype's unfavorable-outcome rate across λ (subtypes matched to the canonical
λ=1 solution by clinical profile): the five profiles recur essentially unchanged for all λ≥0.3, so λ
controls the *balance* of the partition, not the existence of the five profiles. Reads aggregate
tables only.

Run: ``uv run python -m figures.supp.figS3_entropy_sensitivity``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import NARRATIVE_ORDER, subtype_color, subtype_name
from figures.panels import panel_label, save_figure
from figures.style import SINGLE_COL_MM, mm, set_style

FIG_DIR = FIGURES_DIR
SUMMARY = TABLES_DIR / "entropy_sensitivity_summary.csv"
PROFILES = TABLES_DIR / "entropy_sensitivity_profiles.csv"


def main() -> int:
    set_style()
    summ = pd.read_csv(SUMMARY)
    prof = pd.read_csv(PROFILES)
    lams = summ["lambda"].to_numpy()
    xpos = np.arange(len(lams))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(mm(2 * SINGLE_COL_MM), mm(62)))

    # --- a: populated archetypes (bars) + minimum subtype fraction (line) vs λ ---
    axA.bar(xpos, summ["n_populated_>=3%"], width=0.6, color="#BFBFBF", edgecolor="white",
            label="populated archetypes (≥3%)", zorder=2)
    for xi, npop in zip(xpos, summ["n_populated_>=3%"], strict=True):
        axA.text(xi, npop + 0.08, str(int(npop)), ha="center", va="bottom", fontsize=6)
    axA.set_ylim(0, 5.8)
    axA.set_ylabel("populated archetypes (≥3%)")
    axA.set_xticks(xpos)
    axA.set_xticklabels([f"{x:g}" for x in lams])
    axA.set_xlabel("entropy weight λ")
    ax2 = axA.twinx()
    ax2.plot(xpos, summ["min_frac%"], "-o", color="#2E3440", lw=1.2, ms=4, zorder=3,
             label="min subtype fraction")
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("min subtype fraction (%)", color="#2E3440")
    ax2.tick_params(axis="y", labelcolor="#2E3440")
    ax2.spines["top"].set_visible(False)
    axA.annotate("collapse:\none subtype empty", xy=(0, 0.6), xytext=(0.35, 8),
                 fontsize=5.5, color="#555555",
                 arrowprops=dict(arrowstyle="->", lw=0.5, color="#999999"))
    axA.set_title("λ controls balance: λ=0 collapses, λ≥0.3 keeps five", fontsize=6.5, pad=3)
    panel_label(axA, "a", dx=-0.16, dy=1.07)

    # --- b: each subtype's unfavorable rate vs λ (matched to canonical, coloured by subtype) ---
    for i in NARRATIVE_ORDER:
        canon = f"S{i}"
        sub = prof[prof["matches_canonical"] == canon].copy()
        sub["lambda"] = pd.Categorical(sub["lambda"], categories=lams, ordered=True)
        sub = sub.sort_values("lambda")
        # drop the empty (frac 0) points so a collapsed subtype doesn't plot a spurious 0
        sub = sub[sub["n"] > 0]
        xs = [list(lams).index(lv) for lv in sub["lambda"]]
        axB.plot(xs, sub["unfav%"], "-o", color=subtype_color(i), lw=1.2, ms=3.5,
                 label=subtype_name(i), zorder=3)
    axB.set_xticks(xpos)
    axB.set_xticklabels([f"{x:g}" for x in lams])
    axB.set_xlabel("entropy weight λ")
    axB.set_ylabel("unfavorable outcome (%)")
    # Fit every point, including the lambda = 0 values of the subtypes that collapse there.
    axB.set_ylim(0, max(42.0, 1.08 * float(prof.loc[prof["n"] > 0, "unfav%"].max())))
    axB.set_title("The five profiles recur for all λ≥0.3", fontsize=6.5, pad=3)
    axB.legend(loc="center right", fontsize=5.2, handlelength=1.2, labelspacing=0.25,
               borderpad=0.2)
    panel_label(axB, "b", dx=-0.16, dy=1.07)

    fig.subplots_adjust(wspace=0.42, left=0.09, right=0.93, bottom=0.16, top=0.88)
    pdf, png = save_figure(fig, "figS3_entropy_sensitivity", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
