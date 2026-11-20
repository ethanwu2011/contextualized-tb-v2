"""Figure 2 (Plotly parallel version, richer) — subtype fingerprints + modality attribution.

Replaces the grouped-bar wall with two chart types matched to the data:
(top) five small-multiple RADARS — each subtype's min-max-normalized 9-feature "fingerprint" as a
filled polygon, so the co-movement that defines a subtype (e.g. failure subtypes riding high on
MDR+HIV+smear simultaneously) reads as a shape; (bottom) a TREEMAP of the modality attribution where
tile area = total share and colour = per-feature share, exposing whether a high-share modality is one
concentrated signal or many thin ones. Aggregate tables only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig02_fingerprints_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from figures.palette import MODALITY_COLORS, NARRATIVE_ORDER, subtype_color, subtype_name
from figures.plotly_style import INK, MUTED, OUT, TABLES, hex_to_rgba, panel_label, save

NARRATIVE = [f"S{i}" for i in NARRATIVE_ORDER]
NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
COLORS = [subtype_color(i) for i in NARRATIVE_ORDER]
# radar axes (short labels) — ordered so clinically-related axes sit adjacent
FEATURES = [("sensitive%", "Drug-sens"), ("MDR%", "MDR"), ("XDR%", "XDR"), ("HIV%", "HIV+"),
            ("smear+%", "Smear+"), ("diabetes%", "Diabetes"), ("age", "Age"),
            ("Ukraine%", "Ukraine"), ("SSAfrica%", "SS-Africa")]


def main() -> int:
    df = pd.read_csv(TABLES / "multimodal_subtype_characterization.csv").set_index("subtype").reindex(NARRATIVE)
    unfav = df["unfavorable%"].to_numpy()
    raw = df[[c for c, _ in FEATURES]].to_numpy(float)
    theta = [lbl for _, lbl in FEATURES]
    # min-max normalize each feature across the 5 subtypes -> comparable fingerprint shapes
    lo, hi = raw.min(0), raw.max(0)
    norm = (raw - lo) / np.where(hi - lo == 0, 1, hi - lo)

    titles = [f"{nm}<br><span style='font-size:10px;color:{MUTED}'>{u:.1f}% unfavorable</span>"
              for nm, u in zip(NAMES, unfav, strict=True)]
    fig = make_subplots(
        rows=2, cols=5, row_heights=[0.42, 0.58], vertical_spacing=0.04,
        specs=[[{"type": "polar"}] * 5, [{"type": "domain", "colspan": 5}, None, None, None, None]],
        subplot_titles=titles + [""])

    th = theta + [theta[0]]  # close the polygon
    for k in range(5):
        r = list(norm[k]) + [norm[k][0]]
        raw_hover = list(raw[k]) + [raw[k][0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=th, fill="toself", name=NAMES[k],
            fillcolor=hex_to_rgba(COLORS[k], 0.30), line=dict(color=COLORS[k], width=2),
            customdata=raw_hover, hovertemplate="%{theta}: %{customdata:.1f}<extra></extra>",
            showlegend=False), row=1, col=k + 1)

    # polar cosmetics on each of the 5 polar axes
    for k in range(1, 6):
        pax = "polar" if k == 1 else f"polar{k}"
        fig.layout[pax].update(
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks="", gridcolor="#E6E6E6",
                            linewidth=0),
            angularaxis=dict(tickfont=dict(size=8.5, color="#555555"), gridcolor="#ECECEC",
                             linecolor="#D0D0D0", rotation=90, direction="clockwise"),
            bgcolor="white")

    # --- treemap of modality attribution -------------------------------------------------------
    attr = pd.read_csv(TABLES / "multimodal_modality_attribution_ci.csv").sort_values(
        "total_share", ascending=False)
    mods = attr["modality"].tolist()
    tot = (attr["total_share"] * 100).to_numpy()
    pf = (attr["per_feature_share"] * 100).to_numpy()
    nfeat = attr["n_features"].to_numpy()
    labels = ["attribution"] + [m.replace("_", " ") for m in mods]
    parents = [""] + ["attribution"] * len(mods)
    values = [float(tot.sum())] + list(tot)  # root must be >= sum of children for branchvalues='total'
    # colour each modality tile by per-feature share (sequential), root neutral
    colors = [0.0] + list(pf)
    text = [""] + [f"{m.replace('_', ' ')}<br>{t:.0f}% total · {p:.0f}%/feat<br>{int(n)} features"
                   for m, t, p, n in zip(mods, tot, pf, nfeat, strict=True)]
    fig.add_trace(go.Treemap(
        labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, colorscale="Cividis", cmin=0, cmax=float(pf.max()),
                    colorbar=dict(title=dict(text="% per<br>feature", font=dict(size=9.5)),
                                  len=0.40, y=0.20, yanchor="middle", x=1.005, thickness=12,
                                  tickfont=dict(size=9)), line=dict(color="white", width=2)),
        text=text, textinfo="text", textfont=dict(size=11, color="white"),
        hovertemplate="%{label}<br>total share %{value:.1f}%<extra></extra>",
        tiling=dict(pad=2)), row=2, col=1)

    fig.update_layout(
        width=1280, height=760, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=26, r=78, t=84, b=16), uniformtext=dict(minsize=8, mode="hide"))
    # style the polar subplot titles (smaller, coloured by subtype)
    for ann, c in zip(fig.layout.annotations, COLORS, strict=False):
        if ann.text and ann.text.split("<")[0] in NAMES:
            ann.font.size = 12
            ann.font.color = c
    panel_label(fig, "a", x=0.0, y=1.07)
    panel_label(fig, "b", x=0.0, y=0.555)
    fig.add_annotation(text="Subtype fingerprints (9 features, min–max normalized across subtypes)",
                       xref="paper", yref="paper", x=0.5, y=1.075, showarrow=False,
                       font=dict(size=13.5, color=INK), xanchor="center")
    fig.add_annotation(text="Modality attribution — area = total share, colour = per-feature share "
                            "(resistance: big but thin; labs/region: small but concentrated)",
                       xref="paper", yref="paper", x=0.5, y=0.565, showarrow=False,
                       font=dict(size=12.5, color=INK), xanchor="center")

    png, html = save(fig, "fig02_fingerprints", width=1280, height=860)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
