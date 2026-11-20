"""Supplementary Figure S1 — subtype reproducibility across random seeds.

Retrains the canonical PATCH model (K=5, entropy load-balancing) under six seeds and shows (a) the
pairwise adjusted Rand index between seed solutions and (b) the per-subtype cross-seed consensus
(fraction of a subtype's patients that stay together after Hungarian label-matching). Trains
internally; emits only the figure.

Fitted on observed culture conversion by default, as published; ``--endpoint died_or_failure``
refits on the primary endpoint (see figures/supp/_endpoint.py).

Run: ``uv run python -m figures.supp.figS1_reproducibility``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ctb.config import FIGURES_DIR
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from figures.palette import NARRATIVE_ORDER, SEQUENTIAL_CMAP, subtype_color, subtype_name
from figures.supp._endpoint import figure_stem, load_matrix, parse_endpoint
from figures.panels import panel_label, pval_annot, save_figure, thin_colorbar
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
K = 5
N_SEEDS = 6
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)


def _match(ref: np.ndarray, other: np.ndarray) -> dict[int, int]:
    ov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ov[i, j] = np.logical_and(ref == i, other == j).sum()
    r, c = linear_sum_assignment(-ov)
    return {int(o): int(rr) for rr, o in zip(r, c, strict=True)}


def main(endpoint: str = "conversion") -> int:
    set_style()
    data = load_matrix(endpoint)
    c, x = data["C"].astype(np.float64), data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)

    subs = []
    for s in range(N_SEEDS):
        m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True,
                                          seed=s)
        train(m, c, x, y, CFG, seed=s)
        subs.append(m.predict_loadings(c).argmax(1))

    ari = np.eye(N_SEEDS)
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            ari[i, j] = ari[j, i] = adjusted_rand_score(subs[i], subs[j])

    # per-subtype consensus relative to seed 0 (Hungarian-matched)
    ref = subs[0]
    agree = np.zeros((K, N_SEEDS - 1))
    for jj, s in enumerate(range(1, N_SEEDS)):
        mp = _match(ref, subs[s])
        mapped = np.array([mp[v] for v in subs[s]])
        for k in range(K):
            sel = ref == k
            agree[k, jj] = (mapped[sel] == k).mean() if sel.sum() else np.nan
    consensus = np.nanmean(agree, axis=1)

    fig = plt.figure(figsize=(mm(180), mm(62)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.62)

    # --- Panel a: pairwise cross-seed ARI matrix ---------------------------------------------
    # The diagonal is self-comparison (always 1.0) and carries no information: mask it so it
    # neither steals the colormap nor draws the eye. Rescale the norm to the observed
    # off-diagonal range so the real spread (~0.4-0.8) is discriminable, not squeezed mid-scale.
    iu = np.triu_indices(N_SEEDS, 1)
    off = ari[iu]
    mu, sd = off.mean(), off.std(ddof=1)
    vmin, vmax = 0.35, 0.80
    ari_disp = ari.copy()
    np.fill_diagonal(ari_disp, np.nan)

    axA = fig.add_subplot(gs[0])
    norm = plt.Normalize(vmin, vmax)
    cmap = SEQUENTIAL_CMAP.copy()
    cmap.set_bad("#f0f0f0")
    axA.imshow(ari_disp, cmap=cmap, norm=norm)
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            if i == j:
                # self-comparison: neutral grey, no colour weight
                axA.text(j, i, "1.00", ha="center", va="center", fontsize=5.0, color="#999999")
                continue
            # text colour from the cell's rendered luminance (robust to any norm)
            r, g, b = cmap(norm(ari[i, j]))[:3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            axA.text(j, i, f"{ari[i, j]:.2f}", ha="center", va="center", fontsize=5.0,
                     color="white" if lum < 0.5 else "#222222")
    axA.set_xticks(range(N_SEEDS)); axA.set_yticks(range(N_SEEDS))
    axA.set_xticklabels([str(i) for i in range(N_SEEDS)], fontsize=5.5)
    axA.set_yticklabels([str(i) for i in range(N_SEEDS)], fontsize=5.5)
    axA.set_xlabel("random seed"); axA.set_ylabel("random seed")
    axA.set_title(f"Cross-seed agreement (ARI {mu:.2f} ± {sd:.2f}, n={N_SEEDS} seeds)",
                  fontsize=7, pad=3)
    thin_colorbar(fig, axA, cmap, norm, "ARI")
    panel_label(axA, "a", dx=-0.12, dy=1.07)

    # --- Panel b: per-subtype cross-seed consensus (relative to seed 0, Hungarian-matched) ----
    axB = fig.add_subplot(gs[1])
    order = NARRATIVE_ORDER  # narrative
    yb = np.arange(K)[::-1]
    cols = [subtype_color(k) for k in order]
    axB.barh(yb, [consensus[k] for k in order], color=cols, edgecolor="white", lw=0.5, height=0.72)
    # overlay the n=5 per-seed-pair raw points and a SD error bar so dispersion is visible
    for yi, k in zip(yb, order, strict=True):
        pts = agree[k][~np.isnan(agree[k])]
        axB.errorbar(consensus[k], yi, xerr=pts.std(ddof=1), fmt="none", ecolor="#333333",
                     lw=0.6, capsize=1.5, zorder=3)
        axB.scatter(pts, np.full(pts.size, yi), s=3, color="#333333", zorder=4, clip_on=False)
        axB.text(consensus[k] + pts.std(ddof=1) + 0.03, yi, f"{consensus[k]:.2f}", va="center",
                 fontsize=5.5)
    axB.axvline(0.7, ls="--", lw=0.5, color="#999999", zorder=0)
    axB.set_yticks(yb)
    axB.set_yticklabels([subtype_name(k) for k in order], fontsize=5.5)
    axB.set_xlim(0, 1.05)
    axB.set_xlabel("cross-seed consensus (fraction co-assigned vs seed 0)")
    axB.set_title("Per-subtype stability", fontsize=7, pad=3)
    pval_annot(axB, "n=5 seed pairs, whiskers show SD, dashed = 0.7 stable", xy=(0.5, 0.05),
               fontsize=5.5)
    panel_label(axB, "b", dx=-0.42, dy=1.07)

    pdf, png = save_figure(fig, figure_stem("figS1_reproducibility", endpoint), FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_endpoint(description=__doc__)))
