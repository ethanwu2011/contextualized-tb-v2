"""Figure 3 (Plotly parallel version) — latent structure as density-contour small-multiples.

The one place a *contour* is honest: a continuous 2-D field. We embed the per-patient archetype loadings
(K=5 simplex) with UMAP, then show a filled density CONTOUR per subtype on a shared embedding (plus an
"all patients" reference) — so where each subtype concentrates, and how the cores overlap, reads at a
glance. DUA-safe by construction: the embedding is computed locally and only a 2-D histogram grid is ever
materialized — no per-patient coordinates are written to disk, to the HTML, or printed. (The published
matplotlib Fig 3 embeds the same loadings as a scatter; this is strictly more aggregate.)

Run: ``.venv/bin/python -m figures.fig03_umap_density_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from ctb.config import PROCESSED_DIR
from ctb.data.artifacts import load_primary_subtypes
from figures.palette import NARRATIVE_ORDER, subtype_color, subtype_name
from figures.plotly_style import INK, OUT, hex_to_rgba, save, style_subplot_titles

NPZ = PROCESSED_DIR / "multimodal_subtypes.npz"
NBINS = 80


def main() -> int:
    d = load_primary_subtypes(NPZ)
    loadings, subt = d["loadings"].astype(float), d["subtypes"].astype(int)

    import umap  # local import; embedding never leaves this process
    emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(loadings)
    x, y = emb[:, 0], emb[:, 1]
    xpad, ypad = 0.06 * np.ptp(x), 0.06 * np.ptp(y)
    xe = np.linspace(x.min() - xpad, x.max() + xpad, NBINS + 1)
    ye = np.linspace(y.min() - ypad, y.max() + ypad, NBINS + 1)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    def grid(mask=None):
        h, _, _ = np.histogram2d(x if mask is None else x[mask], y if mask is None else y[mask],
                                 bins=[xe, ye])
        return gaussian_filter(h, 1.4)

    def scale(c):
        return [[0.0, "rgba(255,255,255,0)"], [0.12, hex_to_rgba(c, 0.12)],
                [0.45, hex_to_rgba(c, 0.55)], [1.0, c]]

    panels = [("All patients", None, "#7A7F86")] + \
             [(subtype_name(i), i, subtype_color(i)) for i in NARRATIVE_ORDER]
    titles = [f"{nm} (n={int((subt == i).sum()) if i is not None else len(subt):,})" for nm, i, _ in panels]

    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.025, vertical_spacing=0.10, subplot_titles=titles)
    for k, (nm, i, c) in enumerate(panels):
        r, col = divmod(k, 3)
        H = grid(None if i is None else subt == i)
        fig.add_trace(go.Contour(z=H.T, x=xc, y=yc, showscale=False, ncontours=9,
                                 colorscale=scale(c), contours=dict(coloring="fill"),
                                 line=dict(width=0.4, color=hex_to_rgba(c, 0.5)),
                                 hoverinfo="skip"), row=r + 1, col=col + 1)

    fig.update_layout(
        width=1180, height=720, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=44, r=24, t=92, b=44),
        title=dict(text="Treatment-response structure — UMAP density of archetype loadings, by subtype "
                        "(filled contours; shared embedding)",
                   x=0.5, xanchor="center", y=0.975, font=dict(size=14.5, color=INK)))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="", linecolor="#E0E0E0")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="", linecolor="#E0E0E0")
    style_subplot_titles(fig, size=12)

    png, html = save(fig, "fig03_umap_density", width=1180, height=720)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
