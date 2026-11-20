"""Figure 2 (Plotly parallel version) — the five treatment-response subtypes.

On-screen counterpart of ``figures.fig02a_subtypes``. Same three panels and data
(``multimodal_subtype_characterization.csv`` + ``multimodal_modality_attribution_ci.csv``):
(a) per-subtype unfavorable rate with Wilson 95% CI; (b) z-scored enrichment heatmap of defining
features; (c) modality attribution (total vs per-feature share). DUA-safe (aggregate tables only).

Run: ``.venv/bin/python -m figures.fig02_subtypes_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency

from figures.palette import MODALITY_COLORS, NARRATIVE_ORDER, subtype_color, subtype_name
from figures.plotly_style import (
    DIVERGING,
    INK,
    MUTED,
    OUT,
    TABLES,
    hex_to_rgba,
    panel_label,
    save,
    style_subplot_titles,
)

NARRATIVE = [f"S{i}" for i in NARRATIVE_ORDER]
NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
COLORS = [subtype_color(i) for i in NARRATIVE_ORDER]
FEATURES = [
    ("sensitive%", "Drug-sensitive"), ("MDR%", "MDR"), ("XDR%", "XDR"),
    ("HIV%", "HIV+"), ("diabetes%", "Diabetes"), ("smear+%", "Smear+"),
    ("Ukraine%", "Ukraine"), ("SSAfrica%", "SS-Africa"), ("age", "Age"),
]


def _wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half) * 100.0, (centre + half) * 100.0


def main() -> int:
    df = pd.read_csv(TABLES / "multimodal_subtype_characterization.csv").set_index("subtype").reindex(NARRATIVE)
    n = df["n"].to_numpy()
    unfav = df["unfavorable%"].to_numpy()
    raw = df[[c for c, _ in FEATURES]].to_numpy(float)
    z = (raw - raw.mean(0)) / raw.std(0, ddof=0)
    feat_labels = [lbl for _, lbl in FEATURES]

    unfav_k = np.round(n * unfav / 100.0).astype(int)
    ci_lo, ci_hi = _wilson_ci(unfav_k, n)
    chi2, _, _, _ = chi2_contingency(np.vstack([unfav_k, n - unfav_k]))

    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.31, 0.69], row_heights=[0.56, 0.44],
        vertical_spacing=0.155, horizontal_spacing=0.13,
        specs=[[{"type": "bar"}, {"type": "heatmap"}], [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=("",
                        "Defining clinical profile  ·  cell = % of subtype (age in years)",
                        "Modality attribution — no single modality dominates"))

    # ---- (a) unfavorable-rate bars ------------------------------------------------------------
    ticktext = [f'<span style="color:{c}">{nm}</span>' for c, nm in zip(COLORS, NAMES, strict=True)]
    fig.add_trace(go.Bar(
        y=NAMES, x=unfav, orientation="h", marker=dict(color=COLORS, line=dict(color="white", width=1)),
        error_x=dict(type="data", symmetric=False, array=ci_hi - unfav, arrayminus=unfav - ci_lo,
                     color="#555555", thickness=1.1, width=3),
        text=[f"{u:.1f}%" if u < 10 else f"{u:.0f}%" for u in unfav], textposition="outside",
        textfont=dict(size=11, color=INK), cliponaxis=False, showlegend=False,
        hovertemplate="%{y}<br>%{x:.1f}% unfavorable<extra></extra>"), row=1, col=1)
    fig.update_xaxes(range=[0, 46], title_text="unfavorable (%), Wilson 95% CI", row=1, col=1)
    fig.update_yaxes(autorange="reversed", tickvals=NAMES, ticktext=ticktext, automargin=True,
                     tickfont=dict(size=10.5), row=1, col=1)
    fig.add_annotation(text=f"χ² = {chi2:.0f}, df = 4, P < 0.001", xref="x domain", yref="y domain",
                       x=0.97, y=0.93, showarrow=False, font=dict(size=10, color=MUTED),
                       xanchor="right", yanchor="top", row=1, col=1)

    # ---- (b) z-scored enrichment heatmap ------------------------------------------------------
    tags = [f'<span style="color:{c}"><b>{t}</b></span>' for c, t in zip(COLORS, NARRATIVE, strict=True)]
    zmax = max(2.0, float(np.abs(z).max()))
    fig.add_trace(go.Heatmap(
        z=z, x=feat_labels, y=tags, colorscale=DIVERGING, zmid=0, zmin=-zmax, zmax=zmax,
        xgap=2, ygap=2, colorbar=dict(title=dict(text="z within<br>column", font=dict(size=10)),
                                      len=0.50, y=0.78, yanchor="middle", x=1.005, thickness=13,
                                      tickfont=dict(size=9.5), outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>z = %{z:.2f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), row=1, col=2)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10.5), row=1, col=2)
    # raw-value cell annotations (white on saturated cells, dark elsewhere)
    for i, tag in enumerate(tags):
        for j, lbl in enumerate(feat_labels):
            v = raw[i, j]
            txt = f"{v:.0f}y" if FEATURES[j][0] == "age" else f"{v:.0f}"
            fig.add_annotation(x=lbl, y=tag, text=txt, showarrow=False, xref="x2", yref="y2",
                               font=dict(size=9.5, color="white" if abs(z[i, j]) > 1.4 else "#333333"))

    # ---- (c) modality attribution -------------------------------------------------------------
    attr = pd.read_csv(TABLES / "multimodal_modality_attribution_ci.csv").sort_values(
        "total_share", ascending=False)
    mods = attr["modality"].tolist()
    mod_labels = [m.replace("_", " ") for m in mods]
    cols = [MODALITY_COLORS.get(m, "#BFBFBF") for m in mods]
    tot = attr["total_share"].to_numpy() * 100
    pf = attr["per_feature_share"].to_numpy() * 100
    fig.add_trace(go.Bar(
        x=mod_labels, y=tot, marker=dict(color=cols, line=dict(color="white", width=1)),
        error_y=dict(type="data", symmetric=False, array=attr["total_hi"] * 100 - tot,
                     arrayminus=tot - attr["total_lo"] * 100, color="#555555", thickness=1.0, width=3),
        name="total share", showlegend=False,
        hovertemplate="%{x}<br>total share %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(
        x=mod_labels, y=pf, marker=dict(color=[hex_to_rgba(c, 0.5) for c in cols],
                                        line=dict(color="white", width=1), pattern=dict(shape="/", size=4)),
        error_y=dict(type="data", symmetric=False, array=attr["perfeat_hi"] * 100 - pf,
                     arrayminus=pf - attr["perfeat_lo"] * 100, color="#555555", thickness=1.0, width=3),
        name="per-feature share", showlegend=False,
        hovertemplate="%{x}<br>per-feature share %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.update_xaxes(tickangle=-22, tickfont=dict(size=10.5), row=2, col=1)
    fig.update_yaxes(title_text="contribution (%)", range=[0, 40], row=2, col=1)
    # neutral solid-vs-pattern legend (colour already encodes modality) via dummy traces
    for nm, pat in (("total share", None), ("per-feature share", "/")):
        fig.add_trace(go.Bar(x=[None], y=[None], name=nm, marker=dict(
            color="#8A8A8A" if pat is None else hex_to_rgba("#8A8A8A", 0.6),
            pattern=dict(shape=pat) if pat else None), showlegend=True), row=2, col=1)
    # annotate the labs rank-flip
    i_labs = mods.index("labs")
    fig.add_annotation(x=mod_labels[i_labs], y=pf[i_labs] + 1.5,
                       text="largest rank flip:<br>last in aggregate,<br>highest per feature<br>(of named modalities)",
                       showarrow=True, arrowcolor="#AAAAAA", arrowwidth=1, ax=-58, ay=-34,
                       font=dict(size=9, color=MUTED), align="left", xref="x3", yref="y3")

    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="top", y=0.30, xanchor="right", x=0.62),
        margin=dict(l=70, r=78, t=64, b=70))
    style_subplot_titles(fig)
    panel_label(fig, "a", x=0.0, y=1.015)
    panel_label(fig, "b", x=0.345, y=1.015)
    panel_label(fig, "c", x=0.0, y=0.40)

    png, html = save(fig, "fig02_subtypes", width=1200, height=820)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
