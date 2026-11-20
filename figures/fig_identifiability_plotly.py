"""Supplement (Plotly) — the identifiability / positivity map.

Which subtype×drug effects the data can actually estimate. Heatmap of drug-receipt % per subtype;
a cell is *estimable* only when receipt sits away from 0/100 (enough treated and untreated patients).
Marked cells (○) are the 61/130 estimable ones; the rest are near-deterministic and PATCH declines to
over-interpret them. This is the honesty firewall, made visual. Aggregate table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_identifiability_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go

from figures.palette import NARRATIVE_ORDER, subtype_name
from figures.plotly_style import INK, MUTED, OUT, TABLES, save

NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]


def main() -> int:
    p = pd.read_csv(TABLES / "positivity_audit.csv")
    p["drug"] = p["drug"].str.replace("drug_", "", regex=False)
    rec = p.pivot(index="subtype", columns="drug", values="receipt_pct").reindex(NAMES)
    band = p.pivot(index="subtype", columns="drug", values="band").reindex(NAMES)
    drugs = rec.columns.tolist()
    n_est = int((band == "estimable").to_numpy().sum())

    fig = go.Figure(go.Heatmap(
        z=rec.to_numpy(), x=drugs, y=NAMES, colorscale="Cividis", zmin=0, zmax=100, xgap=1.4, ygap=1.4,
        colorbar=dict(title=dict(text="% receiving<br>the drug", font=dict(size=10)), len=0.9,
                      thickness=14, tickfont=dict(size=10), outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>%{z:.1f}% receive it<extra></extra>"))

    # mark the estimable cells with an open ring
    ex, ey = [], []
    for i, s in enumerate(NAMES):
        for d in drugs:
            if band.loc[s, d] == "estimable":
                ex.append(d)
                ey.append(s)
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(symbol="circle-open", size=11, color="white", line=dict(width=1.8, color="white"))))

    fig.update_layout(
        width=1320, height=420, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=180, r=40, t=78, b=86),
        title=dict(text=f"Identifiability map — only {n_est} of 130 subtype×drug cells are estimable "
                        "(○); the rest are near-deterministic",
                   x=0.5, xanchor="center", y=0.96, font=dict(size=15, color=INK)))
    fig.update_xaxes(tickangle=-55, tickfont=dict(size=10), showgrid=False, ticks="")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), showgrid=False, ticks="")
    fig.add_annotation(text="○ = estimable (drug receipt 5–95%) · effect is identifiable and reported; "
                            "blank = near-deterministic, walled off",
                       xref="paper", yref="paper", x=0.5, y=-0.30, showarrow=False,
                       font=dict(size=10.5, color=MUTED), xanchor="center")

    png, html = save(fig, "fig_identifiability", width=1320, height=420)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}  ({n_est} estimable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
