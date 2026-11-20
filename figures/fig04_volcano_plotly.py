"""Figure 4 companion (Plotly) — volcano plot of the full context-conditioned coefficient atlas.

Signed effect (Δβ) vs significance (−log10 p) over all 2,756 context×drug estimates, with a marginal
histogram of the signed-coefficient distribution. Significant cells are coloured by sign (blue = more
favorable recorded association, red = less favorable); the marginal quantifies the signed association
distribution. Aggregate atlas table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig04_volcano_plotly``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.express as px

from figures.plotly_style import INK, OUT, TABLES, save

ATLAS = TABLES / "heterogeneity_atlas_unfavorable_raw.csv"


def main() -> int:
    a = pd.read_csv(ATLAS).copy()
    a["neglogp"] = -np.log10(a["pval"].clip(lower=1e-12))
    a["effect"] = np.where(~a["significant"], "not significant",
                           np.where(a["delta_beta"] < 0,
                                    "more favorable recorded association (significant)",
                                    "less favorable recorded association (significant)"))
    cmap = {"more favorable recorded association (significant)": "#2166AC",
            "less favorable recorded association (significant)": "#B2182B",
            "not significant": "#C9CDD2"}

    fig = px.scatter(
        a, x="delta_beta", y="neglogp", color="effect", marginal_x="histogram", render_mode="svg",
        color_discrete_map=cmap, opacity=0.7, hover_data={"context": True, "drug": True,
                                                           "delta_beta": ":.3f", "pval": ":.3g",
                                                           "neglogp": False, "effect": False},
        category_orders={"effect": list(cmap)})
    fig.update_traces(marker=dict(size=5, line=dict(width=0)), selector=dict(mode="markers"))
    fig.add_vline(x=0, line=dict(color="#888888", width=1, dash="dash"), row=1, col=1)
    sig_frac = float(a["significant"].mean())
    n_harm = int(((a["delta_beta"] > 0) & a["significant"]).sum())
    n_fav = int(((a["delta_beta"] < 0) & a["significant"]).sum())

    fig.update_layout(
        width=1080, height=620, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=70, r=40, t=86, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5, title_text=""),
        title=dict(text="Context-conditioned coefficient atlas: signed coefficient vs significance "
                        f"({n_fav} more favorable · {n_harm} less favorable, of {len(a)} estimates)",
                   x=0.06, xanchor="left", y=0.97, font=dict(size=15, color=INK)))
    fig.update_xaxes(title_text="Δβ  (negative = more favorable recorded association · positive = less favorable)",
                     zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="−log₁₀ p", zeroline=False, row=1, col=1)

    png, html = save(fig, "fig04_volcano", width=1080, height=620)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}  (sig frac {sig_frac:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
