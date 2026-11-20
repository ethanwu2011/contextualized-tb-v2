"""Supplementary Figure — per-drug context-dependence ranking.

How strongly each drug's effect is modulated by context, quantified as the total significant
drug-specific effect-size shift: ``magnitude = Σ |Δβ|`` summed over the context indicators whose
drug-specific interaction is significant (centred / gradient-removed Δβ; see scripts/14). Restricted
to commonly-prescribed drugs (prevalence ≥ 5%) so the per-drug estimates are stable; rifampicin and
linezolid lead. Reads aggregate tables only (the ranking table and the per-context bootstrap CIs).

Run: ``uv run python -m figures.supp.figS6_drug_ranking``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import SEQUENTIAL_CMAP
from figures.panels import panel_label, pval_annot, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
RANK = TABLES_DIR / "drug_heterogeneity_ranking.csv"
# Per-(context, drug) bootstrap Δβ CIs — used to propagate an uncertainty band onto the magnitude.
INTERACTION_CI = TABLES_DIR / "heterogeneity_unfavorable_interaction_ci.csv"
PREV_MIN = 0.05  # match scripts/14: figure restricted to prevalence >= 5% so estimates are stable.
N_BOOT = 60      # bootstrap resamples used upstream (scripts/14 N_BOOT).
TOPN = 12

# Human-readable display names for raw drug identifiers that leak from the pipeline.
DISPLAY = {"other_second_line": "Other 2nd-line", "Cotrimoxazol": "Co-trimoxazole"}


def _magnitude_ci_halfwidth() -> pd.Series:
    """Propagate per-context bootstrap CIs into a half-width on magnitude = Σ |Δβ|.

    Treating the significant context modifiers as independent, the standard error of their summed
    |Δβ| adds in quadrature; the per-context half-width is taken from the bootstrap CI on Δβ. This
    is a conservative uncertainty band on the ranked quantity (not a re-bootstrap of the sum).
    """
    ci = pd.read_csv(INTERACTION_CI)
    sig = ci[ci["significant"]].copy()
    sig["hw"] = (sig["ci_high"] - sig["ci_low"]) / 2.0
    return sig.groupby("drug")["hw"].apply(lambda h: float(np.sqrt((h**2).sum())))


def main() -> int:
    set_style()
    rank = pd.read_csv(RANK)
    # Restrict to commonly-prescribed drugs so estimates are stable (same filter as scripts/14).
    rank = rank[rank["prevalence"] >= PREV_MIN]
    rank = rank.sort_values("magnitude", ascending=False).head(TOPN).iloc[::-1].reset_index(drop=True)

    halfw = _magnitude_ci_halfwidth()
    rank["ci_hw"] = rank["drug"].map(halfw).fillna(0.0)
    labels = [DISPLAY.get(d, d) for d in rank["drug"]]

    fig = plt.figure(figsize=(mm(110), mm(72)))
    ax = fig.add_axes([0.22, 0.20, 0.62, 0.70])
    y = np.arange(len(rank))
    mags = rank["magnitude"].to_numpy()
    hw = rank["ci_hw"].to_numpy()

    # Single neutral fill — bar LENGTH already encodes magnitude, so colour must not duplicate it.
    bar_color = SEQUENTIAL_CMAP(0.72)
    ax.barh(y, mags, color=bar_color, edgecolor="white", lw=0.4, height=0.7, zorder=2)
    # Propagated 95%-style uncertainty whiskers (quadrature of per-context bootstrap CI half-widths).
    ax.errorbar(mags, y, xerr=hw, fmt="none", ecolor="#444444", elinewidth=0.7, capsize=2.0,
                zorder=3)

    # Label each bar with the quantity it ENCODES (magnitude), placed past the whisker.
    xmax = float((mags + hw).max())
    for yi, m, h in zip(y, mags, hw, strict=True):
        ax.text(m + h + 0.10, yi, f"{m:.1f}", va="center", ha="left", fontsize=5.5,
                color="#333333", zorder=4)

    # n_sig (count of significant context modifiers) is a DIFFERENT quantity — give it its own
    # right-hand column so it never reads as the bar's value.
    x_count = xmax * 1.17
    ax.text(x_count, len(rank) - 0.35, "n sig.\ncontexts", va="bottom", ha="center", fontsize=5.0,
            color="#888888", linespacing=0.9)
    for yi, (_, r) in zip(y, rank.iterrows(), strict=True):
        ax.text(x_count, yi, f"{int(r['n_sig'])}", va="center", ha="center", fontsize=5.0,
                color="#888888", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(r"context-dependence magnitude  ($\Sigma\,|\Delta\beta|$ over sig. contexts)",
                  labelpad=3)
    ax.set_xlim(0, x_count * 1.08)
    ax.set_xticks(np.arange(0, np.floor(xmax) + 1, 2))
    ax.set_title("Rifampicin and linezolid are most context-dependent",
                 fontsize=6.8, pad=10)
    pval_annot(ax, f"bars: prevalence ≥ {PREV_MIN:.0%}; whiskers: bootstrap CI (n={N_BOOT})",
               xy=(0.5, -0.17), fontsize=5.0)
    pdf, png = save_figure(fig, "figS6_drug_ranking", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
