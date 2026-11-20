"""Fig 3 headline (Plotly) — PATCH organizes the died/failure OUTCOME across embedding methods.

The point: raw features embed patients by their context inputs (resistance, region) but NOT by outcome;
PATCH's contextualized representation embeds them by outcome. Shown as a *local died/failure rate*
surface (smoothed % died/failure per region) on a SHARED colour scale, for three embedding methods
(UMAP, t-SNE, PCA) × two representations (PATCH loadings, raw 335 features). PATCH lights up with a steep
risk gradient under every method; raw stays flat. Each panel reports the outcome AUROC (how well position
predicts died/failure, 5-fold kNN).

DUA: embeddings + outcome computed locally; only rendered rate surfaces (aggregate 2-D grids) are emitted.

Run: ``.venv/bin/python -m figures.fig03_outcome_methods``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ctb.config import PROCESSED_DIR, RESULTS_DIR
from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes
from figures.plotly_style import INK

ASSETS = RESULTS_DIR / "figures" / "plotly" / "pptx_assets"
NPZ = PROCESSED_DIR / "multimodal_subtypes.npz"
NBINS = 95
RISK = [[0.0, "#FAF7F2"], [0.18, "#FDD49E"], [0.45, "#FC8D59"], [0.72, "#D7301F"], [1.0, "#7F0000"]]


def _reducer(name, seed=42):
    if name == "UMAP":
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3)
    if name == "t-SNE":
        return TSNE(n_components=2, random_state=seed, perplexity=40, init="pca")
    return PCA(n_components=2, random_state=seed)


def _grid(emb, outcome):
    x, y = emb[:, 0], emb[:, 1]
    xpad, ypad = 0.05 * np.ptp(x), 0.05 * np.ptp(y)
    xe = np.linspace(x.min() - xpad, x.max() + xpad, NBINS + 1)
    ye = np.linspace(y.min() - ypad, y.max() + ypad, NBINS + 1)
    cnt, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    ssum, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=outcome.astype(float))
    cs = gaussian_filter(cnt, 3.0)            # heavier smoothing -> smooth surface, not blocky bins
    ss = gaussian_filter(ssum, 3.0)
    rate = np.where(cs >= 0.6, ss / np.maximum(cs, 1e-6), np.nan)   # low threshold -> smooth boundary
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return xc, yc, rate


def _auroc(emb, outcome):
    return float(cross_val_score(KNeighborsClassifier(60), emb, outcome, cv=5, scoring="roc_auc").mean())


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sm = load_primary_subtypes(NPZ)
    cond = list(sm["cond_ids"])
    loadings = sm["loadings"].astype(np.float64)
    mm = load_primary_matrix()
    pos = {c: i for i, c in enumerate(list(mm["cond_ids"]))}
    idx = np.array([pos[c] for c in cond])
    C = StandardScaler().fit_transform(mm["C"].astype(np.float64)[idx])
    X = mm["X"].astype(np.float64)[idx]       # regimen (treatment) indicators, 26 drugs

    outcome = mm["y"].astype(int)[idx]

    reps = {"patch": loadings, "context": C, "treatment": X}
    methods = ["UMAP", "t-SNE", "PCA"]
    cells = {}
    print("computing 9 embeddings (UMAP/t-SNE/PCA × PATCH/context/treatment)...")
    for m in methods:
        for r, X in reps.items():
            emb = _reducer(m).fit_transform(X)
            xc, yc, rate = _grid(emb, outcome)
            cells[(r, m)] = (xc, yc, rate, _auroc(emb, outcome))
            print(f"  {r}-{m}: AUROC {cells[(r, m)][3]:.3f} · rate {np.nanmin(rate):.0%}–{np.nanmax(rate):.0%}")

    vmax = max(np.nanmax(c[2]) for c in cells.values())  # shared scale -> raw looks flat
    for (r, m), (xc, yc, rate, auc) in cells.items():
        fig = go.Figure(go.Heatmap(
            z=rate.T, x=xc, y=yc, colorscale=RISK, zmin=0, zmax=vmax, zsmooth="best",
            colorbar=dict(title=dict(text="died/<br>fail %", font=dict(size=9)), len=0.82, thickness=11,
                          tickformat=".0%", tickfont=dict(size=8.5), outlinewidth=0)))
        fig.update_layout(
            template="patch", width=560, height=540, paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK),
            title=dict(text=f"<span style='font-size:13px'>outcome AUROC = {auc:.2f}</span>", x=0.5,
                       xanchor="center"),
            margin=dict(l=8, r=8, t=34, b=8))
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
        fig.write_image(ASSETS / f"fig03m_{r}_{m.replace('-', '').lower()}.png", scale=3)
    print(f"wrote outcome-method panels to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
