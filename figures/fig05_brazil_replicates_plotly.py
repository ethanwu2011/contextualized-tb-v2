"""Figure 5 (Plotly parallel version) — the context→subtype risk structure replicates on Brazil.

Dumbbell: each harmonized subtype's unfavorable-outcome rate in TB Portals vs in the Brazilian national
registry (n≈298,900) with the SAME model applied, so subtype index matches by construction. The risk
ordering holds across continents and the HIV-enriched subtype is the highest-risk group in both — the
positive external-validation story (not a transfer-AUC). Aggregate table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig05_brazil_replicates_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go

from figures.plotly_style import INK, MUTED, OUT, TABLES, hex_to_rgba, save

C_TBP, C_BR = "#4C566A", "#117733"  # slate = TB Portals; forest green = Brazil (another continent)


def main() -> int:
    p = pd.read_csv(TABLES / "brazil_subtype_profiles.csv")
    tbp = p[p["cohort"] == "TB_Portals"].set_index("subtype")
    bra = p[p["cohort"] == "Brazil_applied"].set_index("subtype")
    d = pd.DataFrame({"tbp": tbp["unfav%"], "bra": bra["unfav%"], "hiv": bra["hiv"]}).sort_values("tbp")
    order = d.index.tolist()
    def _label(code: object) -> str:
        # Harmonized seven-feature profiles are not the PATCH subtypes, so they get no subtype name.
        idx = int(str(code).removeprefix("S"))
        hiv_tag = "  ·  HIV-enriched" if d.loc[code, "hiv"] > 0.5 else ""
        return f"Profile {idx}{hiv_tag}"

    ylab = [_label(s) for s in order]
    y = list(range(len(order)))

    fig = go.Figure()
    # connector segments
    for yi, s in zip(y, order, strict=True):
        fig.add_trace(go.Scatter(x=[d.loc[s, "tbp"], d.loc[s, "bra"]], y=[yi, yi], mode="lines",
                                 line=dict(color="#C8CDD2", width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["tbp"], y=y, mode="markers+text", name="TB Portals",
                             marker=dict(size=15, color=C_TBP, line=dict(color="white", width=1.5)),
                             text=[f"{v:.0f}%" for v in d["tbp"]], textposition="middle left",
                             textfont=dict(size=10, color=C_TBP), cliponaxis=False,
                             hovertemplate="TB Portals %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["bra"], y=y, mode="markers+text", name="Brazil (n≈298,900)",
                             marker=dict(size=15, color=C_BR, line=dict(color="white", width=1.5)),
                             text=[f"{v:.0f}%" for v in d["bra"]], textposition="middle right",
                             textfont=dict(size=10, color=C_BR), cliponaxis=False,
                             hovertemplate="Brazil %{x:.1f}%<extra></extra>"))
    # highlight band behind the HIV-enriched (highest-risk) row
    hiv_y = [yi for yi, s in zip(y, order, strict=True) if d.loc[s, "hiv"] > 0.5]
    if hiv_y:
        fig.add_hrect(y0=hiv_y[0] - 0.45, y1=hiv_y[0] + 0.45, fillcolor=hex_to_rgba("#D55E00", 0.07),
                      line_width=0, layer="below")
        fig.add_annotation(x=max(d["tbp"].max(), d["bra"].max()) + 2.5, y=hiv_y[0],
                           text="highest-risk<br>on both continents", showarrow=False, xanchor="left",
                           font=dict(size=9.5, color="#D55E00"))

    fig.update_layout(
        width=920, height=440, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=170, r=150, t=80, b=112),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        title=dict(text="Context→subtype risk replicates on Brazil — same model, another continent",
                   x=0.5, xanchor="center", y=0.95, font=dict(size=15, color=INK)))
    fig.update_xaxes(title_text="unfavorable outcome (%)", range=[0, 36], showgrid=True, gridcolor="#ECECEC")
    fig.update_yaxes(tickvals=y, ticktext=ylab, tickfont=dict(size=11), showgrid=False,
                     range=[-0.6, len(order) - 0.4])
    fig.add_annotation(text="harmonized cross-cohort model · subtype index matched by applying the same map",
                       xref="paper", yref="paper", x=0.5, y=-0.46, showarrow=False,
                       font=dict(size=10, color=MUTED), xanchor="center")

    png, html = save(fig, "fig05_brazil_replicates", width=920, height=440)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
