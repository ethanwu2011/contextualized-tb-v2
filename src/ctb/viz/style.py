"""Publication-quality matplotlib styling.

Applies a consistent, colourblind-safe, sans-serif style sized for single-column figures. Call
:func:`set_style` once before generating figures. :func:`save_figure` writes both PNG and PDF at
300 dpi with vector text preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Colourblind-safe palette (Seaborn "colorblind"); explicit hex for reproducibility.
PALETTE: list[str] = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
]
# Stable colour assignment for the four baselines.
MODEL_COLORS: dict[str, str] = {
    "LR-no-context": "#949494",
    "LR-with-context": "#0173B2",
    "GBDT-no-context": "#DE8F05",
    "GBDT-with-context": "#029E73",
}


def set_style() -> None:
    """Apply the project-wide publication matplotlib style.

    Returns
    -------
    None
    """
    sns.set_theme(context="paper", style="white", palette=PALETTE)
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,  # editable/vector text in PDF
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlepad": 9,
            "axes.labelsize": 9.5,
            "axes.labelweight": "medium",
            "axes.labelpad": 5,
            "axes.edgecolor": "#444444",
            "axes.grid": False,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(fig: Figure, path: Path) -> list[Path]:
    """Save a figure as both PNG and PDF at 300 dpi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : pathlib.Path
        Output path; the suffix is replaced to produce ``.png`` and ``.pdf``.

    Returns
    -------
    list of pathlib.Path
        The written file paths.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        out = path.with_suffix(suffix)
        fig.savefig(out)
        written.append(out)
    logger.info("Saved figure %s (png+pdf)", path.stem)
    return written
