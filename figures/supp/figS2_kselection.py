"""Supplementary Figure S2 — choosing K=5 archetypes.

Retrains the canonical PATCH model (entropy load-balancing) across K=4..8 with three seeds each and
reports how many archetypes stay populated (>=3% of patients) and the cross-seed reproducibility
(mean ARI). K=5 is selected because it retains populated archetypes while avoiding the
reproducibility drop observed at higher K.
Trains internally; emits only the figure.

Run: ``uv run python -m figures.supp.figS2_kselection``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from ctb.config import FIGURES_DIR
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from figures.panels import pval_annot, save_figure
from figures.style import SINGLE_COL_MM, mm, set_style

FIG_DIR = FIGURES_DIR
K_RANGE = range(4, 9)
N_SEEDS = 3
MIN_FRAC = 0.03
CFG = TrainConfig(epochs=1500, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
# Non-semantic emphasis colours: neutral dark accent for the ARI curve (NOT a modality/subtype
# colour) and a neutral slate to mark the selected K. Keeping these decorative avoids colliding
# with the locked subtype/modality palette ("a category is the same colour everywhere").
ARI_ACCENT = "#333333"     # dark neutral — the discriminating result line
SELECT_EDGE = "#222222"    # outline used to flag the selected K=5
BAR_GREY = "#D9D9D9"        # demoted, lightened populated-archetype bars


def main() -> int:
    set_style()
    data = load_primary_matrix()
    c, x = data["C"].astype(np.float64), data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)

    ks, populated, aris, ari_pairs = [], [], [], []
    for k in K_RANGE:
        subs = []
        for s in range(N_SEEDS):
            m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=k, simplex=True,
                                              seed=s)
            train(m, c, x, y, CFG, seed=s)
            subs.append(m.predict_loadings(c).argmax(1))
        pops = [int((np.bincount(s, minlength=k) / len(s) >= MIN_FRAC).sum()) for s in subs]
        pair = [adjusted_rand_score(subs[i], subs[j])
                for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
        ks.append(k)
        populated.append(float(np.mean(pops)))
        aris.append(float(np.mean(pair)))
        ari_pairs.append(np.asarray(pair, dtype=float))  # keep all seed-pair ARIs for dispersion

    fig = plt.figure(figsize=(mm(SINGLE_COL_MM), mm(72)))
    ax = fig.add_axes([0.14, 0.16, 0.70, 0.70])
    ks_arr = np.asarray(ks, dtype=float)

    # --- secondary (background) axis: populated archetypes, demoted to a thin reference -----------
    # The entropy load-balancing penalty keeps every archetype populated at all K (bars == K), so
    # this variable carries no discriminating information; lighten it so the eye lands on the ARI.
    ax_pop = ax.twinx()
    ax_pop.bar(ks_arr, populated, color=BAR_GREY, edgecolor="white", lw=0.5, width=0.62,
               zorder=1, label="populated archetypes (≥3%)")
    ax_pop.set_ylabel("populated archetypes (≥3% of patients)", color="#8A8A8A")
    ax_pop.set_ylim(0, max(ks) + 1.2)
    ax_pop.tick_params(axis="y", colors="#8A8A8A")
    for sp in ax_pop.spines.values():
        sp.set_visible(False)
    ax_pop.spines["right"].set_visible(True)
    ax_pop.spines["right"].set_color("#8A8A8A")

    # --- primary (foreground) axis: cross-seed ARI, the quantity that selects K ------------------
    # min–max whiskers across the N_SEEDS-choose-2 seed pairs, plus the jittered raw points.
    lo = np.array([p.min() for p in ari_pairs])
    hi = np.array([p.max() for p in ari_pairs])
    rng = np.random.default_rng(0)
    for k, pairs in zip(ks_arr, ari_pairs, strict=True):
        jx = k + rng.uniform(-0.07, 0.07, size=pairs.size)
        ax.scatter(jx, pairs, s=5, color=ARI_ACCENT, alpha=0.45, zorder=5, linewidths=0)
    (ari_line,) = ax.plot(ks_arr, aris, "-o", color=ARI_ACCENT, lw=1.6, ms=4.5, zorder=6,
                          label="cross-seed ARI (mean)")
    ax.errorbar(ks_arr, aris, yerr=[np.array(aris) - lo, hi - np.array(aris)], fmt="none",
                ecolor=ARI_ACCENT, elinewidth=0.7, capsize=2, capthick=0.7, zorder=6)
    ax.set_ylabel("cross-seed ARI")
    ax.set_ylim(0, 1)
    ax.set_zorder(ax_pop.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xlabel("number of archetypes K")
    ax.set_xticks(list(ks))
    ax.set_title("Cross-seed reproducibility (ARI) and populated archetypes vs K", fontsize=7,
                 pad=4)

    # Mark the selected K=5 with a non-semantic outline + label (no palette colour reuse).
    i5, i6 = ks.index(5), ks.index(6)
    ax.scatter([5], [aris[i5]], s=70, facecolors="none", edgecolors=SELECT_EDGE, lw=1.2, zorder=7)
    ax.annotate("K = 5 selected", xy=(5, aris[i5]), xytext=(4.05, 0.30), fontsize=6,
                color=SELECT_EDGE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=SELECT_EDGE))
    # Quantify the cliff: ARI value labels at K=5 and K=6.
    ax.annotate(f"{aris[i5]:.2f}", xy=(5, aris[i5]), xytext=(11, 7), textcoords="offset points",
                ha="left", fontsize=6, color=ARI_ACCENT)
    ax.annotate(f"{aris[i6]:.2f}", xy=(6, aris[i6]), xytext=(0, -11), textcoords="offset points",
                ha="center", fontsize=6, color=ARI_ACCENT)
    ax.annotate("reproducibility\ncliff", xy=(5.5, (aris[i5] + aris[i6]) / 2),
                xytext=(6.6, 0.72), fontsize=6, color=ARI_ACCENT, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=ARI_ACCENT))

    # Sample size on-panel (stats belong on the figure, not only the caption).
    pval_annot(ax, f"n = {N_SEEDS} seeds ({N_SEEDS * (N_SEEDS - 1) // 2} seed-pairs per K); "
               "whiskers = min–max", xy=(0.5, 0.045), fontsize=5.5)

    # Combined legend (both axes); without it the label= kwargs would render nothing. Placed in the
    # empty upper-right interior so it never overstrikes the title or the data.
    ax.legend(handles=[ari_line, ax_pop.patches[0]],
              labels=["cross-seed ARI (mean ± range)", "populated archetypes (≥3%)"],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6,
              handlelength=1.4, borderaxespad=0.3)

    pdf, png = save_figure(fig, "figS2_kselection", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
