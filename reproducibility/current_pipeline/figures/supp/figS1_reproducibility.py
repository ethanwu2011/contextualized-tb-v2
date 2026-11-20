"""Diagnostic scaffold for died/failure hard-bin agreement across random seeds.

Retrains the canonical PATCH model (K=5, entropy load-balancing) under six seeds and shows (a) the
pairwise adjusted Rand index between seed solutions and (b) the per-bin cross-seed consensus
(fraction of seed-0 bin members that stay together after Hungarian label-matching). These hard
assignments are descriptive diagnostics, not validated biological subtypes. Trains internally;
emits only the figure.

Run: ``uv run python -m figures.supp.figS1_reproducibility``
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ctb.config import FIGURES_DIR
from ctb.data.model_matrices import DIED_FAILURE, load_endpoint_matrices
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from figures.palette import SEQUENTIAL_CMAP
from figures.panels import panel_label, pval_annot, save_figure, thin_colorbar
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
K = 5
REDUCED_RUN = os.environ.get("CTB_REDUCED_RUN") == "1"
WORKFLOW_MODE = os.environ.get("CTB_WORKFLOW_MODE", "untracked")
N_SEEDS = 3 if REDUCED_RUN else 6
EPOCHS = 25 if REDUCED_RUN else 2000
CFG = TrainConfig(
    epochs=EPOCHS,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)


def _match(ref: np.ndarray, other: np.ndarray) -> dict[int, int]:
    ov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ov[i, j] = np.logical_and(ref == i, other == j).sum()
    r, c = linear_sum_assignment(-ov)
    return {int(o): int(rr) for rr, o in zip(r, c, strict=True)}


def main() -> int:
    """Train seed replicates and render a non-claim-bearing hard-bin diagnostic."""
    set_style()
    data = load_endpoint_matrices(DIED_FAILURE)
    c, x = data.C, data.X
    y = data.y.astype(np.float64)

    subs = []
    for s in range(N_SEEDS):
        m = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=s
        )
        train(m, c, x, y, CFG, seed=s)
        subs.append(m.predict_loadings(c).argmax(1))

    ari = np.eye(N_SEEDS)
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            ari[i, j] = ari[j, i] = adjusted_rand_score(subs[i], subs[j])

    # Per-bin consensus relative to seed 0 (Hungarian-matched). The match is only a pragmatic
    # comparison device: it does not confer biological identity on any hard assignment.
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
    run_label = (
        "SYNTHETIC SMOKE TEST — NO SCIENTIFIC MEANING"
        if WORKFLOW_MODE == "synthetic"
        else "INTERMEDIATE DIAGNOSTIC — NOT CLAIM-BEARING"
    )
    fig.text(
        0.5,
        0.985,
        run_label,
        ha="center",
        va="top",
        fontsize=6,
        color="#A61B1B",
        fontweight="bold",
    )

    # --- Panel a: pairwise cross-seed ARI matrix ---------------------------------------------
    # The diagonal is self-comparison (always 1.0) and carries no information: mask it so it
    # neither steals the colormap nor draws the eye. Use the full defined ARI range so a weak or
    # negative smoke-test result cannot be visually inflated by a data-dependent narrow scale.
    iu = np.triu_indices(N_SEEDS, 1)
    off = ari[iu]
    mu, sd = off.mean(), off.std(ddof=1)
    vmin, vmax = -1.0, 1.0
    ari_disp = ari.copy()
    np.fill_diagonal(ari_disp, np.nan)

    ax_a = fig.add_subplot(gs[0])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f0f0f0")
    ax_a.imshow(ari_disp, cmap=cmap, norm=norm)
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            if i == j:
                # self-comparison: neutral grey, no colour weight
                ax_a.text(
                    j,
                    i,
                    "1.00",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="#999999",
                )
                continue
            # text colour from the cell's rendered luminance (robust to any norm)
            r, g, b = cmap(norm(ari[i, j]))[:3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax_a.text(
                j,
                i,
                f"{ari[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white" if lum < 0.5 else "#222222",
            )
    ax_a.set_xticks(range(N_SEEDS))
    ax_a.set_yticks(range(N_SEEDS))
    ax_a.set_xticklabels([str(i) for i in range(N_SEEDS)], fontsize=5.5)
    ax_a.set_yticklabels([str(i) for i in range(N_SEEDS)], fontsize=5.5)
    ax_a.set_xlabel("random seed")
    ax_a.set_ylabel("random seed")
    ax_a.set_title(
        f"Died/failure agreement (ARI {mu:.2f} ± {sd:.2f}, n={N_SEEDS} seeds)",
        fontsize=7,
        pad=3,
    )
    thin_colorbar(fig, ax_a, cmap, norm, "ARI")
    panel_label(ax_a, "a", dx=-0.12, dy=1.07)

    # --- Panel b: per-bin cross-seed consensus (relative to seed 0, Hungarian-matched) --------
    ax_b = fig.add_subplot(gs[1])
    order = list(range(K))
    yb = np.arange(K)[::-1]
    cols = [SEQUENTIAL_CMAP(value) for value in np.linspace(0.30, 0.82, K)]
    ax_b.barh(
        yb,
        [consensus[k] for k in order],
        color=cols,
        edgecolor="white",
        lw=0.5,
        height=0.72,
    )
    # Overlay each seed-vs-reference value and a SD error bar so dispersion is visible.
    for yi, k in zip(yb, order, strict=True):
        pts = agree[k][~np.isnan(agree[k])]
        ax_b.errorbar(
            consensus[k],
            yi,
            xerr=pts.std(ddof=1),
            fmt="none",
            ecolor="#333333",
            lw=0.6,
            capsize=1.5,
            zorder=3,
        )
        ax_b.scatter(
            pts,
            np.full(pts.size, yi),
            s=3,
            color="#333333",
            zorder=4,
            clip_on=False,
        )
        ax_b.text(
            consensus[k] + pts.std(ddof=1) + 0.03,
            yi,
            f"{consensus[k]:.2f}",
            va="center",
            fontsize=5.5,
        )
    ax_b.axvline(0.7, ls="--", lw=0.5, color="#999999", zorder=0)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([f"Hard bin {k + 1}" for k in order], fontsize=5.5)
    ax_b.set_xlim(0, 1.05)
    ax_b.set_xlabel("co-assigned fraction vs seed 0")
    ax_b.set_title("Per-bin diagnostic agreement", fontsize=7, pad=3)
    pval_annot(
        ax_b,
        f"n={N_SEEDS - 1} seed-vs-reference values; — = SD; dashed = 0.7",
        xy=(0.5, 0.05),
        fontsize=5.5,
    )
    panel_label(ax_b, "b", dx=-0.42, dy=1.07)

    pdf, png = save_figure(fig, "figS1_reproducibility", FIG_DIR)
    print(
        f"wrote {pdf.name}, {png.name} "
        f"(endpoint={data.endpoint}, source_n={data.source_n_rows}, assessed_n={data.n_rows}, "
        f"unassessable={data.n_unassessable}, events={data.n_events}, "
        f"outcome_sha256={data.outcome_sha256})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
