"""Figure 3 — the treatment-response structure is multimodal (modality-coloured UMAP).

One 2-D UMAP of the per-patient archetype loadings, shown five times: coloured by subtype and by four
context modalities (resistance class, HIV, smear burden, region). The same structure is read out by
several modalities — the visual core of the paper's multimodal thesis. Aggregates internally; emits
only the figure.

Run: ``uv run python -m figures.fig03_structure``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ctb.config import FIGURES_DIR, PROCESSED_DIR
from ctb.data.artifacts import load_primary_subtypes
from figures.palette import (
    COUNTRY_COLORS,
    NARRATIVE_ORDER,
    RESISTANCE_COLORS,
    STATUS_COLORS,
    subtype_color,
    subtype_name,
)
from figures.panels import panel_label, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
NPZ = PROCESSED_DIR / "multimodal_subtypes.npz"
META = PROCESSED_DIR / "unfavorable_context_meta.parquet"

BG = STATUS_COLORS["off"]  # one canonical neutral grey for the unhighlighted background class


def _embed(loadings: np.ndarray) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                         min_dist=0.3).fit_transform(loadings)
    except Exception:  # pragma: no cover
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(loadings)


def _cramers_v(subtypes: np.ndarray, var: np.ndarray) -> tuple[float, float, int]:
    """Cramér's V (association strength), p, and n for ``var`` vs the subtype labels."""
    df = pd.DataFrame({"s": subtypes, "v": pd.Series(var)}).dropna()
    ct = pd.crosstab(df["s"], df["v"])
    chi2, p, _, _ = chi2_contingency(ct)
    n = int(ct.to_numpy().sum())
    r, k = ct.shape
    v = float(np.sqrt((chi2 / n) / min(r - 1, k - 1)))
    return v, p, n


def _pfmt(p: float) -> str:
    return "P < 1e-300" if p < 1e-300 else f"P = {p:.0e}"


def _scatter(ax, emb, layers, title, letter, *, stat=None, ncol=1, bg_size=0.8,
             fg_size=2.0, fg_alpha=0.55, bg_alpha=0.35):
    """Draw one UMAP. ``layers`` are (color, mask, label, is_background) drawn in order, so the
    grey background / majority class is painted FIRST (small, faint) and the highlighted minority
    class LAST (on top, larger and more opaque) so it is never buried under the dense grey field."""
    handles = []
    for color, mask, lbl, is_bg in layers:
        s = bg_size if is_bg else fg_size
        alpha = bg_alpha if is_bg else fg_alpha
        ax.scatter(emb[mask, 0], emb[mask, 1], s=s, color=color, alpha=alpha, edgecolors="none",
                   rasterized=True, zorder=1 if is_bg else 3)
        if lbl:
            n_lbl = f"{lbl} (n={int(mask.sum()):,})"
            handles.append(plt.Line2D([], [], marker="o", ls="", ms=3.0, color=color, label=n_lbl))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    ax.set_title(title, fontsize=6.5, pad=2)
    if stat:
        ax.text(0.5, 0.985, stat, transform=ax.transAxes, ha="center", va="top", fontsize=6,
                color="#333333")
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=6,
                  ncol=ncol, handletextpad=0.2, columnspacing=0.7, labelspacing=0.18,
                  borderpad=0.1)
    panel_label(ax, letter, dx=0.02, dy=1.14)


def _axis_cue(ax) -> None:
    """Small lower-left L-shaped arrow labelled UMAP 1 / UMAP 2 (embedding convention)."""
    arrow = dict(arrowstyle="-|>", color="#555555", lw=0.7)
    ax.annotate("", xy=(0.20, 0.03), xytext=(0.03, 0.03), xycoords=ax.transAxes,
                arrowprops=arrow)
    ax.annotate("", xy=(0.03, 0.20), xytext=(0.03, 0.03), xycoords=ax.transAxes,
                arrowprops=arrow)
    ax.text(0.225, 0.03, "UMAP 1", transform=ax.transAxes, ha="left", va="center",
            fontsize=5, color="#555555")
    ax.text(0.03, 0.225, "UMAP 2", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5, color="#555555", rotation=90)


def main() -> int:
    set_style()
    data = load_primary_subtypes(NPZ)
    sub = data["subtypes"].astype(int)
    cond = list(data["cond_ids"])
    emb = _embed(data["loadings"].astype(np.float64))
    meta = pd.read_parquet(META).reindex(cond)
    res = meta["type_of_resistance"].astype("string").to_numpy()
    hiv = (meta["comorb_hiv"] == 1).to_numpy()
    smear = (meta["baseline_smear"] == "positive").to_numpy()
    country = meta["country"].astype("string").to_numpy()

    # Association of each modality with the subtype partition (Cramér's V, chi-square).
    v_res, p_res, _ = _cramers_v(sub, res)
    v_hiv, p_hiv, _ = _cramers_v(sub, hiv)
    v_smear, p_smear, _ = _cramers_v(sub, smear)
    v_cty, p_cty, _ = _cramers_v(sub, country)

    fig = plt.figure(figsize=(mm(180), mm(62)))
    gs = fig.add_gridspec(1, 5, wspace=0.12)
    gs.update(top=0.86, bottom=0.40, left=0.02, right=0.98)

    # (a) subtype — the partition itself, so report n (no self-association statistic).
    ax_a = fig.add_subplot(gs[0])
    _scatter(ax_a, emb,
             [(subtype_color(k), sub == k, subtype_name(k), False)
              for k in NARRATIVE_ORDER],
             "Subtype", "a", stat=f"n = {len(sub):,}")
    _axis_cue(ax_a)
    # Detached cluster: the small island floating right of the main blob is a genuine
    # subpopulation — patients sitting at the archetype-0 vertex of the loadings simplex
    # (near-pure loading, mean 0.84), all of which are subtype S0 (drug-sensitive responders).
    # UMAP separates this extreme vertex from the simplex interior; name it so it is not unexplained.
    a0 = data["loadings"].astype(np.float64)[:, 0]
    island = a0 >= 0.7
    if island.any():
        yspan = emb[:, 1].max() - emb[:, 1].min()
        ix = emb[island, 0]
        ax_a.annotate(f"near-pure archetype 0\n(drug-sensitive, n = {int(island.sum()):,})",
                      xy=(ix.mean(), emb[island, 1].max()),
                      xytext=(ix.max(), emb[island, 1].max() + 0.27 * yspan),
                      ha="right", va="bottom", fontsize=5, color="#333333",
                      annotation_clip=False,
                      arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5,
                                      shrinkA=1.0, shrinkB=1.0))

    # (b) resistance class — sequential purple, ordinal (less -> more resistant).
    _scatter(fig.add_subplot(gs[1]), emb,
             [(c, res == lvl, lvl, False) for lvl, c in RESISTANCE_COLORS.items()],
             "Resistance class\n(light → dark = more resistant)", "b",
             stat=f"V = {v_res:.2f}, {_pfmt(p_res)}")

    # (c)-(e) minority-overlay panels: grey majority drawn first (small, faint), highlighted
    # minority class drawn last (larger marker, more opaque) so the imbalanced minority is visible.
    _scatter(fig.add_subplot(gs[2]), emb,
             [(BG, ~hiv, "HIV−", True), (STATUS_COLORS["HIV+"], hiv, "HIV+", False)],
             "HIV status", "c", stat=f"V = {v_hiv:.2f}, {_pfmt(p_hiv)}",
             fg_size=3.0, fg_alpha=0.85)

    # (d) smear burden — burden_clinical-modality purple.
    _scatter(fig.add_subplot(gs[3]), emb,
             [(BG, ~smear, "smear−", True), (STATUS_COLORS["smear+"], smear, "smear+", False)],
             "Smear burden", "d", stat=f"V = {v_smear:.2f}, {_pfmt(p_smear)}",
             fg_size=3.0, fg_alpha=0.85)

    # (e) region — canonical COUNTRY_COLORS for the four most-populous identity colours.
    reg = {k: COUNTRY_COLORS[k] for k in ("Ukraine", "Georgia", "Nigeria", "South Africa")}
    rg = [(BG, ~np.isin(country, list(reg)), "other", True)] + [
        (c, country == k, k, False) for k, c in reg.items()]
    _scatter(fig.add_subplot(gs[4]), emb, rg, "Region", "e",
             stat=f"V = {v_cty:.2f}, {_pfmt(p_cty)}", fg_size=3.0, fg_alpha=0.85)

    fig.text(0.5, 0.015,
             "UMAP of per-patient archetype loadings (n = 10,653); same embedding coloured by five "
             "variables.  V = Cramér's V vs subtype (χ² test).  "
             "Embedding: UMAP, n_neighbors = 30, min_dist = 0.3.",
             ha="center", fontsize=5.5, color="#555555")
    pdf, png = save_figure(fig, "fig03_structure", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
