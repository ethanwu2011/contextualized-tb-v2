"""Diagnostic scaffold for died/failure hard-bin agreement and occupancy across K.

Retrains the canonical PATCH model (entropy load-balancing) across K=4..8 with three seeds each and
reports how many hard bins stay populated (>=3% of patients) and their cross-seed agreement (mean
hard-label ARI). The analysis does not select K: K=5 is marked only as the locked descriptive value
used elsewhere. A final K-selection analysis still requires prespecified soft-profile stability,
entropy, held-out performance, and fold-local preprocessing. Trains internally; emits only the
figure.

Run: ``uv run python -m figures.supp.figS2_kselection``
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from ctb.config import FIGURES_DIR
from ctb.data.model_matrices import DIED_FAILURE, load_endpoint_matrices
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from figures.panels import pval_annot, save_figure
from figures.style import SINGLE_COL_MM, mm, set_style

FIG_DIR = FIGURES_DIR
REDUCED_RUN = os.environ.get("CTB_REDUCED_RUN") == "1"
WORKFLOW_MODE = os.environ.get("CTB_WORKFLOW_MODE", "untracked")
K_RANGE = range(4, 7) if REDUCED_RUN else range(4, 9)
N_SEEDS = 2 if REDUCED_RUN else 3
MIN_FRAC = 0.03
EPOCHS = 25 if REDUCED_RUN else 1500
CFG = TrainConfig(
    epochs=EPOCHS,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)
# Non-semantic emphasis colours. K=5 is a locked descriptive value, not a selection result.
ARI_ACCENT = "#333333"
LOCKED_EDGE = "#222222"
BAR_GREY = "#D9D9D9"


def main() -> int:
    """Train a K sweep and render a diagnostic that explicitly does not select K."""
    set_style()
    data = load_endpoint_matrices(DIED_FAILURE)
    c, x = data.C, data.X
    y = data.y.astype(np.float64)

    ks, populated, aris, ari_pairs = [], [], [], []
    for k in K_RANGE:
        subs = []
        for s in range(N_SEEDS):
            m = LinearArchetypeContextualized(
                c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=s
            )
            train(m, c, x, y, CFG, seed=s)
            subs.append(m.predict_loadings(c).argmax(1))
        pops = [int((np.bincount(s, minlength=k) / len(s) >= MIN_FRAC).sum()) for s in subs]
        pair = [
            adjusted_rand_score(subs[i], subs[j])
            for i in range(N_SEEDS)
            for j in range(i + 1, N_SEEDS)
        ]
        ks.append(k)
        populated.append(float(np.mean(pops)))
        aris.append(float(np.mean(pair)))
        ari_pairs.append(np.asarray(pair, dtype=float))  # keep all seed-pair ARIs for dispersion

    fig = plt.figure(figsize=(mm(SINGLE_COL_MM), mm(72)))
    ax = fig.add_axes([0.14, 0.16, 0.70, 0.70])
    ks_arr = np.asarray(ks, dtype=float)
    run_label = (
        "SYNTHETIC SMOKE TEST — NO SCIENTIFIC MEANING"
        if WORKFLOW_MODE == "synthetic"
        else "INTERMEDIATE DIAGNOSTIC — DOES NOT SELECT K"
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

    # --- secondary (background) axis: populated hard bins -------------------------------------
    ax_pop = ax.twinx()
    ax_pop.bar(
        ks_arr,
        populated,
        color=BAR_GREY,
        edgecolor="white",
        lw=0.5,
        width=0.62,
        zorder=1,
        label="populated hard bins (≥3%)",
    )
    ax_pop.set_ylabel("populated hard bins (≥3% of patients)", color="#8A8A8A")
    ax_pop.set_ylim(0, max(ks) + 1.2)
    ax_pop.tick_params(axis="y", colors="#8A8A8A")
    for sp in ax_pop.spines.values():
        sp.set_visible(False)
    ax_pop.spines["right"].set_visible(True)
    ax_pop.spines["right"].set_color("#8A8A8A")

    # --- primary (foreground) axis: descriptive hard-label cross-seed ARI ----------------------
    # min–max whiskers across the N_SEEDS-choose-2 seed pairs, plus the jittered raw points.
    lo = np.array([p.min() for p in ari_pairs])
    hi = np.array([p.max() for p in ari_pairs])
    rng = np.random.default_rng(0)
    for k, pairs in zip(ks_arr, ari_pairs, strict=True):
        jx = k + rng.uniform(-0.07, 0.07, size=pairs.size)
        ax.scatter(jx, pairs, s=5, color=ARI_ACCENT, alpha=0.45, zorder=5, linewidths=0)
    (ari_line,) = ax.plot(
        ks_arr,
        aris,
        "-o",
        color=ARI_ACCENT,
        lw=1.6,
        ms=4.5,
        zorder=6,
        label="cross-seed ARI (mean)",
    )
    ax.errorbar(
        ks_arr,
        aris,
        yerr=[np.array(aris) - lo, hi - np.array(aris)],
        fmt="none",
        ecolor=ARI_ACCENT,
        elinewidth=0.7,
        capsize=2,
        capthick=0.7,
        zorder=6,
    )
    ax.set_ylabel("cross-seed ARI")
    ari_lower = max(-1.0, min(-0.05, float(lo.min()) - 0.05))
    ax.set_ylim(ari_lower, 1)
    ax.set_zorder(ax_pop.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xlabel("locked-profile count K")
    ax.set_xticks(list(ks))
    ax.set_title(
        "Died/failure hard-bin agreement and occupancy vs K",
        fontsize=7,
        pad=4,
    )

    # Mark the locked descriptive K=5 without treating it as a data-driven selection.
    i5 = ks.index(5)
    ax.scatter([5], [aris[i5]], s=70, facecolors="none", edgecolors=LOCKED_EDGE, lw=1.2, zorder=7)
    ax.annotate(
        "locked K=5\n(not selected here)",
        xy=(5, aris[i5]),
        xytext=(4.05, 0.30),
        fontsize=6,
        color=LOCKED_EDGE,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.6, color=LOCKED_EDGE),
    )
    ax.annotate(
        f"{aris[i5]:.2f}",
        xy=(5, aris[i5]),
        xytext=(11, 7),
        textcoords="offset points",
        ha="left",
        fontsize=6,
        color=ARI_ACCENT,
    )

    pval_annot(
        ax,
        f"Diagnostic only; n={N_SEEDS} seeds "
        f"({N_SEEDS * (N_SEEDS - 1) // 2} seed-pairs per K); whiskers=min–max",
        xy=(0.5, 0.045),
        fontsize=5.5,
    )

    # Combined legend (both axes); without it the label= kwargs would render nothing. Placed in the
    # empty upper-right interior so it never overstrikes the title or the data.
    ax.legend(
        handles=[ari_line, ax_pop.patches[0]],
        labels=["hard-label ARI (mean ± range)", "populated hard bins (≥3%)"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=6,
        handlelength=1.4,
        borderaxespad=0.3,
    )

    pdf, png = save_figure(fig, "figS2_kselection", FIG_DIR)
    print(
        f"wrote {pdf.name}, {png.name} "
        f"(endpoint={data.endpoint}, source_n={data.source_n_rows}, assessed_n={data.n_rows}, "
        f"unassessable={data.n_unassessable}, events={data.n_events}, "
        f"outcome_sha256={data.outcome_sha256})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
