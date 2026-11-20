"""Supplement (Plotly) — leave-one-country-out generalization.

Held-out AUROC when each country is removed from training and predicted cold, as a lollipop sorted by
performance, with the null (0.5) reference and marker size ∝ held-out cohort size. Shows the model
generalizes to an unseen country rather than memorizing one. Aggregate table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_loco_country_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from figures.plotly_style import INK, MUTED, OUT, TABLES, save

DOT = "#0072B2"


def main() -> int:
    h = pd.read_csv(TABLES / "holdout_country_auroc.csv").sort_values("heldout_auroc").reset_index(drop=True)
    y = list(range(len(h)))
    sizes = 9 + 22 * (h["n"] / h["n"].max())  # marker area ∝ held-out n
    WARN = "#D55E00"
    colors = [WARN if a < 0.5 else DOT for a in h["heldout_auroc"]]

    fig = go.Figure()
    fig.add_vline(x=0.5, line=dict(color="#C0392B", width=1, dash="dash"))
    fig.add_annotation(x=0.5, y=len(h) - 0.3, text="null (0.5)", showarrow=False,
                       font=dict(size=9.5, color="#C0392B"), xanchor="center", yanchor="bottom")
    # stems from 0.5 to each AUROC
    for yi, auc in zip(y, h["heldout_auroc"], strict=True):
        fig.add_trace(go.Scatter(x=[0.5, auc], y=[yi, yi], mode="lines", showlegend=False,
                                 line=dict(color="#D5DBE1", width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=h["heldout_auroc"], y=y, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1.5), sizemode="diameter"),
        text=h["held_out_country"], customdata=np.stack([h["n"], h["unfavorable_pct"]], axis=-1),
        hovertemplate="%{text}<br>held-out AUROC %{x:.3f}<br>n=%{customdata[0]:,} · %{customdata[1]:.1f}% unfav<extra></extra>",
        showlegend=False))
    # flag the one out-of-distribution failure (below chance)
    fail = h[h["heldout_auroc"] < 0.5]
    for _, row in fail.iterrows():
        yi = h.index[h["held_out_country"] == row["held_out_country"]][0]
        fig.add_annotation(x=row["heldout_auroc"], y=yi - 0.5, text="below chance — distribution shift",
                           showarrow=False, font=dict(size=9, color=WARN), xanchor="left", yanchor="top")
    for yi, auc in zip(y, h["heldout_auroc"], strict=True):
        fig.add_annotation(x=auc + 0.006, y=yi, text=f"{auc:.3f}", showarrow=False, xanchor="left",
                           font=dict(size=10, color=INK))

    fig.update_layout(
        width=900, height=440, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=120, r=70, t=72, b=54),
        title=dict(text="Leave-one-country-out — held-out AUROC on an unseen country "
                        "(marker ∝ cohort size)", x=0.5, xanchor="center", y=0.95,
                   font=dict(size=14.5, color=INK)))
    fig.update_xaxes(range=[0.29, 0.86], title_text="held-out AUROC", showgrid=True, gridcolor="#ECECEC",
                     dtick=0.05)
    fig.update_yaxes(tickvals=y, ticktext=list(h["held_out_country"]), tickfont=dict(size=11),
                     showgrid=False, range=[-0.6, len(h) - 0.3])

    png, html = save(fig, "fig_loco_country", width=900, height=440)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
