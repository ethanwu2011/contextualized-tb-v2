"""Render the palette swatch sheet for approval. Run: ``uv run python -m figures._swatch``."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures.palette import (
    DIVERGING_CMAP,
    DIVERGING_POLES,
    MODALITY_COLORS,
    MODEL_COLORS,
    NARRATIVE_ORDER,
    RESISTANCE_COLORS,
    SEQUENTIAL_CMAP,
    subtype_color,
    subtype_name,
)
from figures.style import mm, set_style

# Subtypes in narrative order, labelled with their (now correct) canonical S-index.
SUBTYPE_SWATCH = {f"S{i}  {subtype_name(i)}": subtype_color(i) for i in NARRATIVE_ORDER}


def _row(ax: plt.Axes, mapping: dict[str, str], title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
    n = len(mapping)
    for i, (label, hexc) in enumerate(mapping.items()):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=hexc))
        ax.text(i + 0.46, -0.18, label, ha="right", va="top", rotation=30, fontsize=5.5)
        ax.text(i + 0.46, 0.5, hexc, ha="center", va="center", fontsize=5,
                color="white" if i >= n - 2 else "black")
    ax.set_xlim(0, max(n, 6))
    ax.set_ylim(-1.4, 1)
    ax.axis("off")


def main() -> int:
    set_style()
    fig = plt.figure(figsize=(mm(180), mm(170)))
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1.1, 1.1], hspace=1.4)

    _row(fig.add_subplot(gs[0]), SUBTYPE_SWATCH, "Treatment-response subtypes (cool=good, warm=failure)")
    _row(fig.add_subplot(gs[1]), RESISTANCE_COLORS, "Resistance classes (ordinal severity)")
    _row(fig.add_subplot(gs[2]), MODALITY_COLORS, "Context modalities")
    _row(fig.add_subplot(gs[3]), MODEL_COLORS, "Model comparison")

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    axd = fig.add_subplot(gs[4])
    axd.imshow(grad, aspect="auto", cmap=DIVERGING_CMAP, extent=[0, 1, 0, 1])
    axd.set_title("Diverging cmap — signed Δβ / interactions", loc="left", fontweight="bold", fontsize=8)
    axd.text(0, -0.35, DIVERGING_POLES[0], ha="left", va="top", fontsize=6)
    axd.text(1, -0.35, DIVERGING_POLES[1], ha="right", va="top", fontsize=6)
    axd.set_yticks([]); axd.set_xticks([])

    axs = fig.add_subplot(gs[5])
    axs.imshow(grad, aspect="auto", cmap=SEQUENTIAL_CMAP, extent=[0, 1, 0, 1])
    axs.set_title("Sequential cmap — magnitudes (importance, rates, counts)", loc="left",
                  fontweight="bold", fontsize=8)
    axs.text(0, -0.35, "low", ha="left", va="top", fontsize=6)
    axs.text(1, -0.35, "high", ha="right", va="top", fontsize=6)
    axs.set_yticks([]); axs.set_xticks([])

    out = Path(__file__).resolve().parent / "palette_swatch.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
