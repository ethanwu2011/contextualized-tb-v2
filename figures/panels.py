"""Reusable panel helpers so every PATCH figure is built from one consistent toolkit.

No figure should style panel letters, colorbars, significance marks, or scale bars by hand — use
these helpers so the whole set stays identical. All sizes are in points/axes-fraction to match the
Nature-Comms style in :mod:`figures.style`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure


def panel_label(ax: Axes, letter: str, *, dx: float = -0.06, dy: float = 1.06) -> None:
    """Place a bold lowercase panel letter top-left of ``ax`` (consistent offset everywhere)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="right")


def pval_annot(ax: Axes, text: str, *, xy: tuple[float, float] = (0.5, 0.97),
               fontsize: float = 6) -> None:
    """Print a stat string (P, test, n) inside a panel — stats are part of the figure, not caption."""
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize)


def stars(p: float) -> str:
    """Significance stars from a (corrected) p-value; '' if not significant."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def thin_colorbar(fig: Figure, ax: Axes, cmap: Colormap, norm: Normalize, label: str,
                  *, poles: tuple[str, str] | None = None) -> Colorbar:
    """Attach a thin colorbar to ``ax``; optionally name the two poles (for diverging maps)."""
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, aspect=30)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label(label, fontsize=6)
    if poles is not None:
        cb.ax.text(0.5, -0.02, poles[0], transform=cb.ax.transAxes, ha="center", va="top",
                   fontsize=5)
        cb.ax.text(0.5, 1.02, poles[1], transform=cb.ax.transAxes, ha="center", va="bottom",
                   fontsize=5)
    return cb


def scalebar(ax: Axes, length: float, label: str, *, loc: tuple[float, float] = (0.05, 0.05),
             lw: float = 1.5) -> None:
    """Draw a horizontal scale bar in axes-fraction coordinates with a label."""
    x0, y0 = loc
    ax.plot([x0, x0 + length], [y0, y0], transform=ax.transAxes, color="black", lw=lw,
            solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.02, label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5)


def despine(ax: Axes, *, left: bool = True, bottom: bool = True) -> None:
    """Turn off top/right spines (and optionally left/bottom) for a clean look."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def save_figure(fig: Figure, stem: str, outdir: Path) -> tuple[Path, Path]:
    """Write a figure as both vector PDF and 300-dpi PNG; return the two paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png
