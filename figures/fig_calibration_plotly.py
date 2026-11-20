"""Supplement (Plotly) — calibration + precision-recall curves.

Left: reliability curve (predicted vs observed) for PATCH and the context-LR baseline against the
diagonal, Brier in the legend. Right: precision-recall curve, AUPRC in the legend. Reads the aggregate
curve table (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_calibration_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from figures.plotly_style import INK, OUT, TABLES, panel_label, save, style_subplot_titles

COLORS = {"PATCH": "#CC79A7", "context_LR": "#0072B2"}
LABEL = {"PATCH": "PATCH", "context_LR": "context LR"}


def main() -> int:
    c = pd.read_csv(TABLES / "calibration_auprc.csv")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Calibration (reliability)", "Precision–recall"))

    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", showlegend=False,
                             line=dict(color="#BBBBBB", width=1, dash="dash"),
                             hoverinfo="skip"), row=1, col=1)
    for model, g in c[c["kind"] == "calibration"].groupby("model"):
        g = g.sort_values("x")
        brier = float(g["brier"].iloc[0])
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines+markers", name=f"{LABEL[model]} (Brier {brier:.3f})",
                                 line=dict(color=COLORS[model], width=2.4), marker=dict(size=5),
                                 legendgroup=model), row=1, col=1)
    for model, g in c[c["kind"] == "pr"].groupby("model"):
        g = g.sort_values("x")
        auprc = float(g["auprc"].iloc[0])
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines", name=f"{LABEL[model]} (AUPRC {auprc:.3f})",
                                 line=dict(color=COLORS[model], width=2.4), legendgroup=model + "_pr"),
                      row=1, col=2)

    fig.update_xaxes(title_text="predicted probability", range=[0, 0.6], row=1, col=1)
    fig.update_yaxes(title_text="observed frequency", range=[0, 0.6], row=1, col=1)
    fig.update_xaxes(title_text="recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="precision", range=[0, 1], row=1, col=2)
    fig.update_layout(
        width=1080, height=480, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=66, r=30, t=80, b=58),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="center", x=0.5))
    style_subplot_titles(fig)
    panel_label(fig, "a", x=0.0, y=1.04)
    panel_label(fig, "b", x=0.56, y=1.04)

    png, html = save(fig, "fig_calibration", width=1080, height=480)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
