"""Figure 6 (Plotly parallel version) — decision-analytic risk-targeting.

On-screen counterpart of ``figures.fig06_clinical_utility``. Same two panels, same data
(``results/tables/decision_curve*.csv``), rendered with Plotly for talks/web/preview.

Run: ``.venv/bin/python -m figures.fig06_clinical_utility_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from figures.plotly_style import (
    MUTED,
    OUT,
    TABLES,
    hex_to_rgba,
    panel_label,
    save,
    style_subplot_titles,
)

C_PATCH, C_RES, C_GREY, C_DARK = "#0072B2", "#882255", "#B8B8B8", "#333333"


def main() -> int:
    d = pd.read_csv(TABLES / "decision_curve.csv")
    yld = pd.read_csv(TABLES / "decision_curve_yield.csv").set_index("strategy")
    t = d["threshold"] * 100
    auc_res, auc_patch = float(d["auroc_resistance"].iloc[0]), float(d["auroc_patch"].iloc[0])
    prev = float(d["treat_all"].iloc[0])

    yield_subtitle = (
        f'True high-risk patients flagged, per 1,000'
        f'<br><span style="font-size:10.5px;color:{C_PATCH}">PATCH flags '
        f'{int(yld.loc["PATCH (multimodal)", "true_highrisk_per_1000"])} of '
        f'{int(yld.loc["Monitor all", "true_highrisk_per_1000"])} who die/fail vs '
        f'{int(yld.loc["Resistance class", "true_highrisk_per_1000"])} by resistance class '
        f'· AUROC {auc_patch:.2f} vs {auc_res:.2f}</span>')
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.11,
        subplot_titles=("Multimodal risk-targeting compared with resistance-class targeting", yield_subtitle))

    # ---- (a) decision curve -------------------------------------------------------------------
    fig.add_vrect(x0=10, x1=30, fillcolor=C_PATCH, opacity=0.06, line_width=0, row=1, col=1)
    fig.add_annotation(x=20, y=0, ax=0, ay=0, text="clinically<br>relevant band", showarrow=False,
                       font=dict(size=10, color=C_PATCH), xref="x", yref="y", yanchor="bottom",
                       row=1, col=1)
    curves = [
        ("PATCH (multimodal context)", d["PATCH_multimodal"], C_PATCH, 3.2, "solid"),
        ("resistance class only", d["resistance_class_only"], C_RES, 2.3, "solid"),
        ("intervene on all", d["treat_all"], C_GREY, 1.6, "dash"),
        ("intervene on none", d["treat_none"], C_DARK, 1.2, "dot"),
    ]
    for name, y, color, width, dash in curves:
        fig.add_trace(go.Scatter(x=t, y=y, name=name, mode="lines",
                                 line=dict(color=color, width=width, dash=dash),
                                 hovertemplate=f"{name}<br>threshold %{{x:.0f}}%<br>net benefit %{{y:.3f}}<extra></extra>"),
                      row=1, col=1)
    # reference lines: net-benefit zero, and the 20% threshold at which panel b's yield is computed
    fig.add_hline(y=0, line=dict(color="#B0B0B0", width=1, dash="dot"), row=1, col=1)
    fig.add_vline(x=20, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=1,
                  annotation_text="yield threshold (20%)", annotation_position="top right",
                  annotation_font=dict(size=8.5, color=MUTED))
    op = d.iloc[(d["threshold"] * 100 - 20).abs().argmin()]
    fig.add_trace(go.Scatter(
        x=[20, 20], y=[op["PATCH_multimodal"], op["resistance_class_only"]], mode="markers",
        marker=dict(size=7, color=[C_PATCH, C_RES], line=dict(color="white", width=1)),
        showlegend=False, hovertemplate="net benefit at 20%: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(range=[2, 50], title_text="threshold risk for intensified monitoring (%)", row=1, col=1)
    fig.update_yaxes(range=[-0.01, prev + 0.014], title_text="net benefit", row=1, col=1)

    # ---- (b) per-1,000 yield (stacked horizontal bars) ----------------------------------------
    # bottom -> top: Monitor all, Resistance class, PATCH  (PATCH on top, most prominent)
    order = ["Monitor all", "Resistance class", "PATCH (multimodal)"]
    labels = ["monitor all", "resistance class", "PATCH"]
    colors = [C_DARK, C_RES, C_PATCH]
    tp = [int(yld.loc[s, "true_highrisk_per_1000"]) for s in order]
    fp = [int(yld.loc[s, "overmonitored_per_1000"]) for s in order]

    fig.add_trace(go.Bar(
        y=labels, x=tp, orientation="h", name="true high-risk flagged (will die/fail)",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[str(v) for v in tp], textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=12.5, family="Arial"), legendgroup="solid",
        hovertemplate="%{y}: %{x} true high-risk per 1,000<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=fp, orientation="h", name="unnecessarily intensified",
        marker=dict(color=[hex_to_rgba(c, 0.26) for c in colors], line=dict(color="white", width=1)),
        legendgroup="light",
        hovertemplate="%{y}: %{x} over-monitored per 1,000<extra></extra>"), row=1, col=2)
    # total-flagged annotations at the end of each stacked bar
    for lab, a, b in zip(labels, tp, fp, strict=True):
        fig.add_annotation(x=a + b + 14, y=lab, text=f"{a + b} flagged", showarrow=False,
                           font=dict(size=10.5, color=MUTED), xanchor="left", xref="x2", yref="y2")
    fig.update_xaxes(range=[0, 1090], title_text="per 1,000 patients (monitoring threshold 20%)",
                     row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)

    fig.update_layout(
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
        margin=dict(l=64, r=40, t=104, b=96))
    style_subplot_titles(fig)
    panel_label(fig, "a", x=0.0, y=1.075)
    panel_label(fig, "b", x=0.605, y=1.075)

    png, html = save(fig, "fig06_clinical_utility", width=1180, height=520)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
