"""Supplement (Plotly) — model-ladder forest.

AUROC with 95% CI for each rung of the model ladder, as a dot-and-whisker (forest) rather than bars:
bars imply a 0 baseline that AUROC does not have, and CI overlap is the whole story (context drives
accuracy; adding the regimen moves AUROC negligibly). Null reference at 0.5. Aggregate table (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_model_ladder_forest_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go

from figures.plotly_style import INK, MUTED, OUT, TABLES, save

PATCH_HUE = "#CC79A7"
DOT = "#0072B2"


def main() -> int:
    m = pd.read_csv(TABLES / "model_ladder_unfavorable.csv")
    m = m.iloc[::-1].reset_index(drop=True)  # first rung at top
    y = list(range(len(m)))
    is_patch = m["model"].str.contains("PATCH|archetype", case=False, regex=True)
    colors = [PATCH_HUE if p else DOT for p in is_patch]

    fig = go.Figure()
    fig.add_vline(x=0.5, line=dict(color="#C0392B", width=1, dash="dash"))
    fig.add_annotation(x=0.5, y=len(m) - 0.4, text="null (0.5)", showarrow=False,
                       font=dict(size=9.5, color="#C0392B"), xanchor="center", yanchor="bottom")
    fig.add_trace(go.Scatter(
        x=m["auc"], y=y, mode="markers",
        error_x=dict(type="data", symmetric=False, array=m["ci_high"] - m["auc"],
                     arrayminus=m["auc"] - m["ci_low"], color=MUTED, thickness=1.4, width=5),
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        hovertemplate="%{text}<br>AUROC %{x:.3f}<extra></extra>", text=m["model"], showlegend=False))
    # value labels to the right of each whisker
    for yi, (auc, hi, name) in enumerate(zip(m["auc"], m["ci_high"], m["model"], strict=True)):
        fig.add_annotation(x=hi + 0.004, y=yi, text=f"{auc:.3f}", showarrow=False, xanchor="left",
                           font=dict(size=10.5, color=INK))

    fig.update_layout(
        width=900, height=360, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=210, r=60, t=70, b=52),
        title=dict(text="Model ladder — context drives accuracy; the regimen adds little (AUROC ± 95% CI)",
                   x=0.5, xanchor="center", y=0.95, font=dict(size=14.5, color=INK)))
    fig.update_xaxes(range=[0.48, 0.84], title_text="AUROC (unfavorable outcome)", showgrid=True,
                     gridcolor="#ECECEC", dtick=0.05)
    fig.update_yaxes(tickvals=y, ticktext=list(m["model"]), tickfont=dict(size=11), showgrid=False,
                     range=[-0.6, len(m) - 0.2])

    png, html = save(fig, "fig_model_ladder_forest", width=900, height=360)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
