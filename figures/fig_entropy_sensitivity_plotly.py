"""Supplement (Plotly) — the max-entropy load-balancing penalty is (almost) free.

Left: archetype population share across the entropy weight λ — at λ=0 the solution collapses (one
archetype ≈76%, another empty); at λ≥0.3 all five are balanced and populated. Right: in-sample AUROC is
flat (even rises slightly) while the smallest archetype's share climbs from 0 to ~16%. So balance and
interpretability come at no discrimination cost. Aggregate tables only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_entropy_sensitivity_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from figures.palette import subtype_color, subtype_name
from figures.plotly_style import INK, OUT, TABLES, panel_label, save, style_subplot_titles

C_AUC, C_MIN = "#0072B2", "#D55E00"


def main() -> int:
    prof = pd.read_csv(TABLES / "entropy_sensitivity_profiles.csv")
    summ = pd.read_csv(TABLES / "entropy_sensitivity_summary.csv").sort_values("lambda")
    lambdas = sorted(prof["lambda"].unique())

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, specs=[[{}, {"secondary_y": True}]],
                        subplot_titles=("Archetype population share vs λ", "AUROC vs balance vs λ"))

    # --- (a) stacked population share by archetype slot ---
    for i in range(5):
        sub = prof[prof["subtype"] == f"S{i}"].set_index("lambda").reindex(lambdas)
        fig.add_trace(go.Bar(x=[str(l) for l in lambdas], y=sub["frac%"], name=subtype_name(i),
                             marker=dict(color=subtype_color(i), line=dict(color="white", width=1)),
                             hovertemplate=f"{subtype_name(i)}<br>λ=%{{x}} · %{{y:.1f}}%<extra></extra>"),
                      row=1, col=1)
    fig.update_xaxes(title_text="entropy weight λ", type="category", row=1, col=1)
    fig.update_yaxes(title_text="population share (%)", row=1, col=1)

    # --- (b) AUROC (left) + min archetype share (right) ---
    fig.add_trace(go.Scatter(x=summ["lambda"], y=summ["in_sample_auroc"], mode="lines+markers",
                             name="in-sample AUROC", line=dict(color=C_AUC, width=2.6),
                             marker=dict(size=7)), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=summ["lambda"], y=summ["min_frac%"], mode="lines+markers",
                             name="smallest archetype share", line=dict(color=C_MIN, width=2.6, dash="dot"),
                             marker=dict(size=7)), row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="entropy weight λ", row=1, col=2)
    fig.update_yaxes(title_text="in-sample AUROC", color=C_AUC, range=[0.5, 0.9], row=1, col=2,
                     secondary_y=False)
    fig.update_yaxes(title_text="smallest archetype share (%)", color=C_MIN, range=[0, 25], row=1, col=2,
                     secondary_y=True)

    fig.update_layout(
        width=1180, height=480, paper_bgcolor="white", plot_bgcolor="white", barmode="stack",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=64, r=64, t=80, b=116),
        legend=dict(orientation="h", yanchor="bottom", y=-0.36, xanchor="center", x=0.5))
    style_subplot_titles(fig)
    panel_label(fig, "a", x=0.0, y=1.05)
    panel_label(fig, "b", x=0.56, y=1.05)

    png, html = save(fig, "fig_entropy_sensitivity", width=1180, height=480)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
