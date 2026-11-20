"""Supplement (Plotly) — hyperparameter-sweep parallel coordinates.

Each configuration of the sweep (K, entropy λ, L1, learning rate) is one polyline, coloured by held-out
AUROC, ending on the AUROC axis. Shows discrimination is robust across the grid and that the canonical
choice (K=5, λ=1.0) trades a few thousandths of AUROC for 5 balanced, interpretable archetypes. Aggregate
table only (DUA-safe).

Run: ``.venv/bin/python -m figures.fig_hparam_parcoords_plotly``
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.graph_objects as go

from figures.plotly_style import INK, OUT, TABLES, save


def main() -> int:
    h = pd.read_csv(TABLES / "hparam_sweep.csv")
    can = h[h["canonical"]].iloc[0]
    dims = [
        dict(label="K", values=h["K"], tickvals=sorted(h["K"].unique())),
        dict(label="entropy λ", values=h["entropy"], tickvals=sorted(h["entropy"].unique())),
        dict(label="L1", values=h["l1"], tickvals=sorted(h["l1"].unique())),
        dict(label="learning rate", values=h["lr"], tickvals=sorted(h["lr"].unique())),
        dict(label="# populated\narchetypes", values=h["populated"]),
        dict(label="OOF AUROC", values=h["oof_auroc"]),
    ]
    fig = go.Figure(go.Parcoords(
        line=dict(color=h["oof_auroc"], colorscale="Viridis",
                  cmin=float(h["oof_auroc"].min()), cmax=float(h["oof_auroc"].max()),
                  colorbar=dict(title=dict(text="OOF<br>AUROC", font=dict(size=10)), thickness=14,
                                len=0.85, tickfont=dict(size=10), outlinewidth=0)),
        dimensions=dims, labelfont=dict(size=11, color=INK), tickfont=dict(size=9.5, color="#555555")))

    fig.update_layout(
        width=1200, height=520, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
        margin=dict(l=70, r=90, t=90, b=44),
        title=dict(text=f"Hyperparameter sweep — AUROC robust across the grid "
                        f"({h['oof_auroc'].min():.3f}–{h['oof_auroc'].max():.3f}); "
                        f"canonical K={int(can['K'])}, λ={can['entropy']:g} → {can['populated']:.0f} "
                        f"balanced archetypes at AUROC {can['oof_auroc']:.3f}",
                   x=0.5, xanchor="center", y=0.95, font=dict(size=13.5, color=INK)))

    png, html = save(fig, "fig_hparam_parcoords", width=1200, height=520)
    print(f"wrote {png.relative_to(OUT.parents[2])} + {html.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
