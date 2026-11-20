"""Fig 3 UMAP comparison matrix (Plotly, PNAS-style) — contextualized vs non-contextualized.

Two UMAP embeddings — PATCH archetype loadings (contextualized) and raw 335-feature context
(non-contextualized) — each recoloured by subtype, resistance class, HIV, and region. Read DOWN a column
to see how a variable organizes each space; read ACROSS rows to see what contextualization adds. Each
panel reports the silhouette of THAT embedding by THAT variable's labels (high = visually separated).
The point the eye catches: resistance carves clean territories under PATCH, but HIV/region do not further
split the high-resistance cluster — and raw feature space separates by nothing.

DUA: embeddings computed locally; only rendered scatter PNGs are emitted (same exposure as the published
rasterized UMAPs). PNG only, no HTML / no coordinates written as data.

Run: ``.venv/bin/python -m figures.fig03_umap_panels``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from ctb.config import PROCESSED_DIR, RESULTS_DIR
from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes
from figures.palette import (
    COUNTRY_COLORS,
    NARRATIVE_ORDER,
    RESISTANCE_COLORS,
    STATUS_COLORS,
    subtype_color,
    subtype_name,
)
from figures.plotly_style import INK, hex_to_rgba

ASSETS = RESULTS_DIR / "figures" / "plotly" / "pptx_assets"
NPZ = PROCESSED_DIR / "multimodal_subtypes.npz"
META = PROCESSED_DIR / "unfavorable_context_meta.parquet"
BG = STATUS_COLORS["off"]
REGIONS = ("Ukraine", "Georgia", "Nigeria", "South Africa")


def _umap(X, seed=42):
    import umap
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3).fit_transform(X)


def _save(fig, name, w=520, h=560):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="patch", width=w, height=h, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Arial, Helvetica, sans-serif", size=12, color=INK))
    fig.write_image(ASSETS / f"{name}.png", scale=3)
    print(f"  {name}.png")


def _knn_acc(emb, labels):
    """How well 2-D position predicts the variable: 5-fold kNN balanced accuracy. Rewards gradients AND
    clusters (silhouette misses gradients), so it matches what the eye sees. Chance ≈ 1/n_classes."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    lab = pd.Series(labels).astype("object")
    valid = lab.notna().to_numpy()
    X = emb[valid]
    y = pd.factorize(lab[valid].to_numpy())[0]   # integer class codes for the classifier
    if len(y) < 100 or np.unique(y).size < 2:
        return float("nan")
    if len(y) > 5000:
        sel = np.random.RandomState(0).choice(len(y), 5000, replace=False)
        X, y = X[sel], y[sel]
    return float(cross_val_score(KNeighborsClassifier(30), X, y, cv=5,
                                 scoring="balanced_accuracy").mean())


def _panel(emb, layers, stat, name):
    fig = go.Figure()
    for color, mask, lbl, is_bg in layers:
        fig.add_trace(go.Scattergl(
            x=emb[mask, 0], y=emb[mask, 1], mode="markers", name=lbl,
            marker=dict(size=2.4 if is_bg else 4.2, color=hex_to_rgba(color, 0.28 if is_bg else 0.7),
                        line=dict(width=0)), showlegend=bool(lbl)))
    fig.update_layout(
        title=dict(text=f"<span style='color:#333;font-size:13px'>{stat}</span>", x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="top", y=-0.01, xanchor="center", x=0.5, font=dict(size=8.5),
                    itemsizing="constant"),
        margin=dict(l=8, r=8, t=34, b=58))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
    _save(fig, name)


def main() -> int:
    sm = load_primary_subtypes(NPZ)
    sub = sm["subtypes"].astype(int)
    cond_sub = list(sm["cond_ids"])
    mm = load_primary_matrix()
    pos = {c: i for i, c in enumerate(list(mm["cond_ids"]))}
    idx = np.array([pos[c] for c in cond_sub])
    Craw = StandardScaler().fit_transform(mm["C"].astype(np.float64)[idx])

    meta = pd.read_parquet(META).reindex(cond_sub)
    res = meta["type_of_resistance"].astype("string").to_numpy()
    hiv = (meta["comorb_hiv"] == 1).to_numpy()
    country = meta["country"].astype("string").to_numpy()
    region = np.where(np.isin(country, REGIONS), country, "other")

    outcome = mm["y"].astype(int)[idx].astype(bool)

    print("computing embeddings (contextualized + non-contextualized)...")
    emb_ctx = _umap(sm["loadings"].astype(np.float64))
    emb_raw = _umap(Craw)

    colorings = {
        "outcome": (outcome, [(BG, ~outcome, "survived / cured", True),
                              ("#B2182B", outcome, "died / failure", False)]),
        "subtype": (sub, [(subtype_color(k), sub == k, subtype_name(k), False)
                          for k in NARRATIVE_ORDER]),
        "resistance": (res, [(c, res == lvl, lvl, False) for lvl, c in RESISTANCE_COLORS.items()]),
        "hiv": (hiv, [(BG, ~hiv, "HIV−", True), (STATUS_COLORS["HIV+"], hiv, "HIV+", False)]),
        "region": (region, [(BG, region == "other", "other", True)]
                   + [(COUNTRY_COLORS[k], region == k, k, False) for k in REGIONS]),
    }
    print("rendering 2×4 comparison matrix:")
    for emb, tag in ((emb_ctx, "ctx"), (emb_raw, "raw")):
        for col, (labels, layers) in colorings.items():
            a = _knn_acc(emb, labels)
            nclass = pd.Series(labels).nunique()
            stat = f"position→variable: kNN acc {a:.0%} (chance {1/nclass:.0%})" if a == a else ""
            _panel(emb, layers, stat, f"fig03_{tag}_{col}")
    print(f"wrote matrix panels to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
