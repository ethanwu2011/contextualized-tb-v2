"""Matplotlib style for the PATCH Nature-Communications figure set.

Import-and-go: ``from figures.style import set_style; set_style()``. Sets Nature-Comms-compliant
defaults — embedded fonts (Type 42, never Type 3), 5-7 pt sans-serif, thin axes, top/right spines
off, 300 dpi. Figure widths follow the journal grid (single column 88 mm, double column 180 mm);
use :func:`mm` / :data:`SINGLE_COL_MM` / :data:`DOUBLE_COL_MM` to size figures in millimetres.
"""

from __future__ import annotations

import matplotlib as mpl

# Nature Communications column widths (millimetres).
SINGLE_COL_MM: float = 88.0
DOUBLE_COL_MM: float = 180.0
MAX_HEIGHT_MM: float = 247.0

_RC = {
    # Embed real fonts; never Type 3 (journal requirement).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}


def mm(value: float) -> float:
    """Convert millimetres to inches (matplotlib figure units)."""
    return value / 25.4


def set_style() -> None:
    """Apply the PATCH figure style globally."""
    mpl.rcParams.update(_RC)
