"""Figure 4 (Plotly parallel version) — the context-conditioned coefficient atlas, hierarchically clustered.

Upgrades the flat context×drug heatmap into a dendrogram-clustered, significance-stippled atlas: the
106 contexts and 26 drugs are each clustered (Ward) so response *modules* — blocks of contexts and
drugs that behave alike — become adjacent and visible. Diverging blue→red on a symmetric scale about
0 (blue = more favorable recorded association, red = less favorable); a small dot marks the 853 cells
significant after correction, directing the eye to robust associations. Reads only the aggregate atlas
table (DUA-safe).

Run: ``.venv/bin/python -m figures.fig04_atlas_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

from figures.plotly_style import DIVERGING, INK, OUT, TABLES, save

ATLAS = TABLES / "heterogeneity_atlas_unfavorable_raw.csv"
GREY = "#9AA0A6"


def _ward(x):
    return sch.linkage(x, method="ward")


def main() -> int:
    a = pd.read_csv(ATLAS)
    piv = a.pivot_table(index="context", columns="drug", values="delta_beta")
    sig = a.pivot_table(index="context", columns="drug", values="significant").fillna(0).astype(bool)
    contexts, drugs = piv.index.tolist(), piv.columns.tolist()
    M = piv.to_numpy()

    # --- top dendrogram: cluster the 26 drugs (columns) -> shares heatmap x, sits on y2 ---
    fig = ff.create_dendrogram(M.T, orientation="bottom", labels=drugs,
                               linkagefun=_ward, colorscale=[GREY] * 8)
    for tr in fig["data"]:
        tr["yaxis"] = "y2"
        tr["line"]["width"] = 0.8

    # --- side dendrogram: cluster the 106 contexts (rows) -> shares heatmap y, sits on x2 ---
    side = ff.create_dendrogram(M, orientation="right", labels=contexts,
                                linkagefun=_ward, colorscale=[GREY] * 8)
    for tr in side["data"]:
        tr["xaxis"] = "x2"
        tr["line"]["width"] = 0.7
        fig.add_trace(tr)

    col_order = list(fig["layout"]["xaxis"]["ticktext"])   # drug leaf order
    row_order = list(side["layout"]["yaxis"]["ticktext"])  # context leaf order
    heat_x = list(fig["layout"]["xaxis"]["tickvals"])
    heat_y = list(side["layout"]["yaxis"]["tickvals"])

    Mr = piv.loc[row_order, col_order].to_numpy()
    Sr = sig.loc[row_order, col_order].to_numpy()
    vmax = float(np.nanmax(np.abs(M)))

    fig.add_trace(go.Heatmap(
        x=heat_x, y=heat_y, z=Mr, colorscale=DIVERGING, zmid=0, zmin=-vmax, zmax=vmax,
        xgap=0.4, ygap=0.4,
        colorbar=dict(title=dict(text="Δβ<br>less favorable →<br>← more favorable", font=dict(size=10)),
                      len=0.55, y=0.43, yanchor="middle", x=1.04, thickness=13,
                      tickfont=dict(size=9.5), outlinewidth=0),
        customdata=np.dstack([Mr]),
        hovertemplate="%{customdata[0]:.3f} Δβ<extra></extra>"))

    # significance stipple: a small dot on each of the 853 significant cells
    sx, sy = [], []
    for i, yv in enumerate(heat_y):
        for j, xv in enumerate(heat_x):
            if Sr[i, j]:
                sx.append(xv)
                sy.append(yv)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", hoverinfo="skip",
                             marker=dict(symbol="circle", size=2.2, color="rgba(20,20,20,0.55)")))

    # --- layout: heatmap [.13,1]x[0,.86]; side dendro x2 [0,.13]; top dendro y2 [.87,1] ---
    n_ctx, n_drug = len(row_order), len(col_order)
    fig.update_layout(
        width=1080, height=1500, showlegend=False, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=11, color=INK),
        margin=dict(l=10, r=150, t=86, b=64),
        title=dict(text="Context-conditioned coefficient atlas Δβ(C): clustered context × drug response modules",
                   x=0.06, xanchor="left", y=0.985, font=dict(size=16, color=INK)))
    fig.update_layout(xaxis=dict(domain=[0.13, 0.80], tickvals=heat_x, ticktext=col_order,
                                 tickangle=-90, tickfont=dict(size=9), showgrid=False, zeroline=False,
                                 showline=False, ticks="", mirror=False))
    fig.update_layout(xaxis2=dict(domain=[0.0, 0.125], showgrid=False, zeroline=False, showline=False,
                                  showticklabels=False, ticks=""))
    fig.update_layout(yaxis=dict(domain=[0.0, 0.86], tickvals=heat_y, ticktext=row_order, side="right",
                                 tickfont=dict(size=6.0), showgrid=False, zeroline=False, showline=False,
                                 ticks=""))
    fig.update_layout(yaxis2=dict(domain=[0.875, 1.0], showgrid=False, zeroline=False, showline=False,
                                  showticklabels=False, ticks=""))
    fig.add_annotation(text=f"{n_ctx} contexts × {n_drug} drugs · dot = significant (FDR) · Ward clustering",
                       xref="paper", yref="paper", x=0.06, y=1.005, showarrow=False,
                       font=dict(size=10.5, color=GREY), xanchor="left", yanchor="bottom")

    png, html = save(fig, "fig04_atlas", width=1080, height=1500)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
