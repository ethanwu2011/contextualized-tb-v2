"""Figure 2 alt (Plotly) — subtype fingerprints as parallel coordinates.

Companion to the radar version: each subtype is one polyline across the 9 per-axis-scaled fingerprint
features, coloured by unfavorable%. With only 5 lines there is no overplotting, and per-axis ranges fix
the age-in-years vs percent scale clash — the failure subtypes' co-movement on MDR+HIV+smear reads as a
single high-riding path. Aggregate table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig02_parcoords_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go

from figures.palette import NARRATIVE_ORDER, subtype_name
from figures.plotly_style import INK, OUT, TABLES, save

NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
NARRATIVE = [f"S{i}" for i in NARRATIVE_ORDER]
FEATURES = [("sensitive%", "Drug-sens"), ("MDR%", "MDR"), ("XDR%", "XDR"), ("HIV%", "HIV+"),
            ("smear+%", "Smear+"), ("diabetes%", "Diabetes"), ("age", "Age (yr)"),
            ("Ukraine%", "Ukraine"), ("SSAfrica%", "SS-Africa")]
# diverging cool->warm by unfavorable rate (good prognosis -> failure)
COOLWARM = [[0.0, "#009E73"], [0.25, "#56B4E9"], [0.5, "#F0E442"], [0.75, "#E69F00"], [1.0, "#D55E00"]]


def main() -> int:
    df = pd.read_csv(TABLES / "multimodal_subtype_characterization.csv").set_index("subtype").reindex(NARRATIVE)
    unfav = df["unfavorable%"].to_numpy()

    dims = [dict(label="subtype", values=list(range(5)), tickvals=list(range(5)), ticktext=NAMES,
                 range=[-0.3, 4.3])]
    for col, lbl in FEATURES:
        v = df[col].to_numpy(float)
        dims.append(dict(label=lbl, values=list(v), range=[float(v.min()), float(v.max())]))
    dims.append(dict(label="Unfav %", values=list(unfav), range=[0, float(unfav.max()) * 1.05]))

    fig = go.Figure(go.Parcoords(
        line=dict(color=unfav, colorscale=COOLWARM, cmin=0, cmax=float(unfav.max()),
                  colorbar=dict(title=dict(text="Unfav %", font=dict(size=10)), thickness=14,
                                len=0.85, tickfont=dict(size=10), outlinewidth=0)),
        dimensions=dims, labelfont=dict(size=11, color=INK), tickfont=dict(size=9.5, color="#555555"),
        labelangle=0))

    fig.update_layout(
        width=1280, height=520, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=110, r=90, t=84, b=40),
        title=dict(text="Subtype fingerprints — parallel coordinates across 9 features "
                        "(the two failure subtypes ride high together on MDR, HIV, smear+)",
                   x=0.5, xanchor="center", y=0.96, font=dict(size=14.5, color=INK)))

    png, html = save(fig, "fig02_parcoords", width=1280, height=520)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
