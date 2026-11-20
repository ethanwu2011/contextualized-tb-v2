"""Shared Plotly theme for the PATCH *parallel* (on-screen) figure set.

This is a NON-destructive alternative renderer: the matplotlib figures in ``figures/*.py`` remain the
Nature-Communications print/submission set. These Plotly versions mirror the same data and the
``figures.palette`` colour language, but are tuned for screen viewing (talks / web / preview) where
Plotly's defaults look cleaner. Outputs go to ``results/figures/plotly/`` as a high-res PNG (preview)
plus an interactive HTML.

DUA-safe: reads only the aggregate tables in ``results/tables/`` (no row-level patient data).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from ctb.config import RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS_DIR / "figures" / "plotly"
PRINT_OUT = OUT / "print"
TABLES = RESULTS_DIR / "tables"
# Mirror every render here too, so the figures are always trivially findable.
DESKTOP_MIRROR = Path.home() / "Desktop" / "PATCH_plotly_NEW"

# Print-hardening constants (Nature Communications).
PT_PER_PX = 0.75                 # Kaleido/Chromium: 1 px -> 0.75 pt (96-dpi layout convention)
PT_PER_MM = 72.0 / 25.4
DOUBLE_COL_MM = 180.0
SINGLE_COL_MM = 88.0
FONT_FLOOR_PT = 5.0              # Nature minimum final font size
FONT_KEYS = {"font", "tickfont", "labelfont", "rangefont", "textfont",
             "insidetextfont", "outsidetextfont"}

FONT = "Arial, Helvetica, sans-serif"
INK = "#1A1A1A"
GRID = "#ECECEC"
AXIS = "#9AA0A6"
MUTED = "#6B7177"

# Diverging colour scale matching palette.DIVERGING_CMAP
# (blue = more favorable recorded association / depleted; red = less favorable / enriched).
DIVERGING = [[0.0, "#2166AC"], [0.25, "#67A9CF"], [0.5, "#F7F7F7"], [0.75, "#EF8A62"], [1.0, "#B2182B"]]


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(font=dict(size=16, color=INK), x=0.012, xanchor="left", y=0.97, yanchor="top"),
        margin=dict(l=72, r=34, t=84, b=64),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1,
                   ticks="outside", tickcolor=AXIS, ticklen=4, tickfont=dict(size=11.5),
                   title=dict(font=dict(size=12.5, color=MUTED))),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS, linewidth=1,
                   ticks="outside", tickcolor=AXIS, ticklen=4, tickfont=dict(size=11.5),
                   title=dict(font=dict(size=12.5, color=MUTED))),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.65)", bordercolor=GRID,
                    borderwidth=0),
        colorway=["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"],
    )
    return t


pio.templates["patch"] = _template()


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """``#RRGGBB`` -> ``rgba(r,g,b,alpha)`` (for translucent fills/segments)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def panel_label(fig: go.Figure, letter: str, x: float, y: float) -> None:
    """Add a bold panel label (a, b, c) in paper coordinates."""
    fig.add_annotation(text=f"<b>{letter}</b>", xref="paper", yref="paper", x=x, y=y,
                       showarrow=False, font=dict(size=18, color=INK), xanchor="left", yanchor="bottom")


def style_subplot_titles(fig: go.Figure, size: float = 13.5) -> None:
    """Left-align and shrink make_subplots titles (they default to large, centred)."""
    for ann in fig.layout.annotations:
        if ann.text and ann.xref == "paper" and ann.yref == "paper" and ann.font.size is None:
            ann.font.size = size
            ann.font.color = INK


def _bump_fonts(node, parent_key: str, floor_px: float) -> None:
    """Recursively raise any font size below ``floor_px`` (so nothing prints below FONT_FLOOR_PT)."""
    if isinstance(node, dict):
        if parent_key in FONT_KEYS and isinstance(node.get("size"), (int, float)):
            node["size"] = max(node["size"], floor_px)
        for k, v in node.items():
            _bump_fonts(v, k, floor_px)
    elif isinstance(node, list):
        for item in node:
            _bump_fonts(item, parent_key, floor_px)


def save_print(fig: go.Figure, stem: str, width_mm: float = DOUBLE_COL_MM, dpi: int = 600) -> Path:
    """Export a Nature-Comms print figure: vector PDF sized exactly to ``width_mm`` with every font
    floored to FONT_FLOOR_PT, plus a high-dpi PNG at the same physical size. Returns the PDF path.

    Note: Plotly renders heatmaps, contours and parcoords as RASTER even inside a PDF — those panels
    are embedded at ``dpi`` rather than as vector paths (a Plotly limitation, not a sizing bug).
    """
    PRINT_OUT.mkdir(parents=True, exist_ok=True)
    width_px = float(fig.layout.width)
    target_pt = width_mm * PT_PER_MM
    scale = target_pt / (width_px * PT_PER_PX)            # vector PDF physical size = width_mm
    floor_px = FONT_FLOOR_PT / (PT_PER_PX * scale)        # px that renders at FONT_FLOOR_PT after scaling

    d = fig.to_dict()
    _bump_fonts(d.get("layout", {}), "layout", floor_px)
    for tr in d.get("data", []):
        _bump_fonts(tr, "__trace__", floor_px)
    pf = go.Figure(d)

    pdf = PRINT_OUT / f"{stem}.pdf"
    pf.write_image(pdf, format="pdf", scale=scale)
    png_scale = (width_mm / 25.4 * dpi) / width_px        # raster at `dpi` at the target physical size
    png = PRINT_OUT / f"{stem}_{dpi}dpi.png"
    pf.write_image(png, format="png", scale=png_scale)
    mirror = DESKTOP_MIRROR / "print"
    mirror.mkdir(parents=True, exist_ok=True)
    for p in (pdf, png):
        shutil.copy2(p, mirror / p.name)
    return pdf


def save(fig: go.Figure, stem: str, width: int, height: int, scale: int = 2) -> tuple[Path, Path]:
    """Apply the template, write a high-res PNG + interactive HTML, mirror to the Desktop, return paths.

    If ``CTB_PRINT`` is set in the environment, also emit a hardened Nature-Comms PDF via save_print
    (width from ``CTB_PRINT_MM``, default double-column 180 mm).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="patch", width=width, height=height)
    png, html = OUT / f"{stem}.png", OUT / f"{stem}.html"
    fig.write_image(png, scale=scale)
    try:                                        # vector sibling so the deck can embed SVG (Convert to Shape)
        fig.write_image(OUT / f"{stem}.svg")
    except Exception:
        pass
    fig.write_html(html, include_plotlyjs="cdn", full_html=True)
    try:                                        # Desktop mirror is best-effort (macOS TCC/lock can block)
        DESKTOP_MIRROR.mkdir(parents=True, exist_ok=True)
        for p in (png, html):
            shutil.copy2(p, DESKTOP_MIRROR / p.name)
    except OSError:
        pass
    if os.environ.get("CTB_PRINT"):
        save_print(fig, stem, width_mm=float(os.environ.get("CTB_PRINT_MM", DOUBLE_COL_MM)))
    return png, html
