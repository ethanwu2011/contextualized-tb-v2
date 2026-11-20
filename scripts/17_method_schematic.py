"""Figure 1 — schematic of the contextualized coefficient model.

Draws the conceptual diagram: patient context and the recorded regimen enter a contextualized
regression whose per-patient coefficient vector is factored through K shared archetypes, yielding the
decomposition ``logodds(Y|X,C) = Xβ(C) + μ(C)`` and, as read-offs, context-conditioned coefficients,
treatment-response subtypes, and the coefficient atlas. Schematic only (no data); a hand-polished
vector version may replace it for final submission.

Usage
-----
    uv run python scripts/17_method_schematic.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ctb.config import FIGURES_DIR, ensure_output_dirs
from ctb.viz.style import PALETTE, save_figure, set_style


def _box(ax, xy, w, h, text, fc, ec="#444444", fs=9, tc="#111111", weight="normal"):
    """Draw a rounded box with centered text."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.1, edgecolor=ec, facecolor=fc, mutation_aspect=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=weight, zorder=5)


def _arrow(ax, start, end, color="#666666"):
    """Draw a connecting arrow."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, lw=1.4,
            color=color, shrinkA=2, shrinkB=2, zorder=1,
        )
    )


def main() -> int:
    """Render and save the method schematic.

    Returns
    -------
    int
        Exit code.
    """
    set_style()
    ensure_output_dirs()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    blue, orange, green, red, purple = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]

    # --- Inputs ---
    _box(ax, (0.2, 3.2), 2.1, 1.2,
         "Patient context  C\n\nage · comorbidities\nresistance · DST\nimaging · region",
         fc="#EAF2FA", ec=blue, fs=8)
    _box(ax, (0.2, 1.0), 2.1, 1.2,
         "Recorded regimen  X\n\n26 anti-TB drugs\n(first recorded line)",
         fc="#FDF1E3", ec=orange, fs=8)

    # --- Contextualized model (middle) ---
    _box(ax, (3.0, 3.35), 1.9, 0.95,
         "Linear encoder\nZ(C) = softmax(M·C)\nK-simplex loadings", fc="#EAF7F1", ec=green, fs=8)
    _box(ax, (3.0, 2.15), 1.9, 0.95,
         "Archetypes  A\nK drug-response\nprofiles", fc="#F6ECF6", ec=purple, fs=8)
    _box(ax, (5.25, 2.75), 1.7, 1.0,
         "β(C) = Z(C)·A\nper-patient\ncontext-conditioned\ncoefficients", fc="#FFFFFF", ec="#333333", fs=8.5,
         weight="semibold")

    # --- Decomposition equation ---
    _box(ax, (3.0, 0.55), 3.95, 1.05, "", fc="#FAFAFA", ec="#999999")
    ax.text(4.97, 1.28, r"logodds(Y | X, C) = X·β(C) + μ(C)", ha="center", va="center",
            fontsize=11, weight="semibold", color="#111111")
    ax.text(4.97, 0.86,
            "homogeneous + context-conditioned coefficients + context effect",
            ha="center", va="center", fontsize=7.2, color="#777777")

    # --- Outputs ---
    _box(ax, (7.45, 3.55), 3.3, 0.95,
         "Per-patient coefficients β(Cᵢ)\n+ coefficient atlas", fc="#EAF2FA", ec=blue, fs=8.2)
    _box(ax, (7.45, 2.35), 3.3, 0.95,
         "5 treatment-response\nsubtypes (archetype mixture)", fc="#F6ECF6", ec=purple, fs=8.2)
    _box(ax, (7.45, 1.15), 3.3, 0.95,
         "Outcome  Y\ndied / failure", fc="#FDECEC", ec=red, fs=8.2)

    # --- Arrows ---
    _arrow(ax, (2.3, 3.8), (3.0, 3.8))   # C -> encoder
    _arrow(ax, (2.3, 1.6), (3.0, 1.05))  # X -> equation
    _arrow(ax, (4.9, 3.8), (5.4, 3.4))   # encoder -> beta
    _arrow(ax, (4.9, 2.6), (5.4, 3.05))  # archetypes -> beta
    _arrow(ax, (6.1, 3.0), (6.5, 1.6))   # beta -> equation
    _arrow(ax, (6.95, 1.1), (7.45, 1.6))  # equation -> outcome
    _arrow(ax, (6.1, 3.25), (7.45, 3.9))  # beta -> atlas
    _arrow(ax, (4.0, 2.15), (4.0, 1.6))   # archetypes -> equation (mu/beta)
    _arrow(ax, (6.1, 2.9), (7.45, 2.8))   # beta -> subtypes

    ax.text(5.5, 4.95, "Contextualized coefficient model", ha="center", va="center",
            fontsize=13, weight="semibold", color="#111111")
    written = save_figure(fig, FIGURES_DIR / "method_schematic")
    plt.rcParams["figure.constrained_layout.use"] = True
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
