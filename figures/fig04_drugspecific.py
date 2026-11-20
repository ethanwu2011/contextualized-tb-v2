"""Figure 4 — treatment-response heterogeneity: from contextualized coefficients to robust archetypes.

The narrative of the figure runs top-to-bottom. (a) PATCH's *context-conditioned* output: the
per-patient coefficient map `β(C)`, read out as a context × drug atlas of gradient-removed coefficient
modification. (b) The *archetypal coarsening*: the same drug columns, now pooled into the five
treatment-response subtypes, where a label-matched bootstrap + BH-FDR leaves a single robust anchor
(rifampicin ↔ drug-sensitive). (c) The variance of the per-patient coefficient vectors is split into
a global responsiveness gradient and drug-specific modification. The columns of (a) and (b) are
aligned so the pooling is visible. Reads the aggregate tables only.

Run: ``uv run python -m figures.fig04_drugspecific``
"""

from __future__ import annotations

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import (
    DIVERGING_CMAP,
    NARRATIVE_ORDER,
    subtype_color,
    subtype_name,
)
from figures.panels import panel_label, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
TABLE = TABLES_DIR / "multimodal_subtype_drugspecific.csv"
ATLAS = TABLES_DIR / "heterogeneity_atlas_unfavorable_interaction.csv"
DECOMP = TABLES_DIR / "heterogeneity_decomposition.csv"
CHARACT = TABLES_DIR / "multimodal_subtype_characterization.csv"
NARRATIVE = [f"S{i}" for i in NARRATIVE_ORDER]  # responders first, then failure
NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
COLORS = [subtype_color(i) for i in NARRATIVE_ORDER]

ATLAS_MAX_ROWS = 34     # cap on contexts shown in the main-figure atlas for legibility (full set: Fig. S5)
ATLAS_N_BOOT = 40       # resamples behind the atlas significance test (percentile, BH-FDR)

# Clean display labels for the drug-token axis. Standard WHO TB-drug abbreviations are kept; the two
# non-WHO / programmatic tokens are normalised so no snake_case or misspelling reaches the figure.
DRUG_LABELS = {
    "Cotrimoxazol": "Cotrim",
    "other_second_line": "2nd-line",
    "ART": "ART",
}
DRUG_KEY = ("R rifampicin · H isoniazid · E ethambutol · Z pyrazinamide · S streptomycin · "
            "ART antiretroviral · Cotrim cotrimoxazole · 2nd-line other second-line")


def _clean_context(raw: str) -> str:
    """Human-readable context label from a raw schema token (no exposed feature names)."""
    s = raw.replace("_", " ")
    s = re.sub(r"\bbactec\b", "DST", s, flags=re.IGNORECASE)
    s = s.replace("injectible", "injectable")
    s = s.replace("comorb ", "comorbidity: ")
    s = s.replace("type of resistance", "resistance type")
    s = s.replace("case definition", "case def.")
    s = s.replace("lung localization", "localization")
    s = s.replace("shadow pattern", "imaging")
    s = s.replace("sub lineage", "lineage")
    s = s.replace(">med", " > median")
    return s.strip()


def main() -> int:
    set_style()
    ds = pd.read_csv(TABLE)
    eff = ds.pivot(index="subtype", columns="drug", values="drug_specific").reindex(NARRATIVE)
    sig = ds.pivot(index="subtype", columns="drug", values="significant").reindex(NARRATIVE)
    # order drugs by maximum absolute SUBTYPE coefficient (most structured drugs to the left); this same
    # order is imposed on the atlas above so the columns of (a) and (b) align.
    order = eff.abs().max(0).sort_values(ascending=False).index.tolist()
    eff, sig = eff[order], sig[order]
    Ms = eff.to_numpy(float)
    n_nominal = int((ds["pval"] < 0.05).sum())          # 23 of 130
    n_total = int(len(ds))                               # 130
    enrich = n_nominal / (n_total * 0.05)                # 3.5x over the 6.5 expected
    n_patients = int(pd.read_csv(CHARACT)["n"].sum())    # 10,653 across S0..S4

    # --- atlas (context-conditioned β(C)) reindexed onto the SAME drug order --------------------------
    at = pd.read_csv(ATLAS)
    a_eff = at.pivot(index="context", columns="drug", values="delta_beta").reindex(columns=order)
    a_sig = at.pivot(index="context", columns="drug", values="significant").reindex(columns=order).fillna(False)
    # keep every context with >=1 significant drug-modification; columns stay in `order` so a aligns with b
    has_sig = a_sig.any(axis=1)
    n_sig_contexts = int(has_sig.sum())
    keep = a_sig.index[has_sig]
    if len(keep) > ATLAS_MAX_ROWS:                        # cap for main-figure legibility (full set: Fig. S5)
        keep = (a_eff.abs() * a_sig).sum(1).loc[keep].sort_values(ascending=False).head(ATLAS_MAX_ROWS).index
    a_eff, a_sig = a_eff.loc[keep], a_sig.loc[keep]
    if len(a_eff) >= 3:                                   # cluster contexts (rows) by effect-modification profile
        ro = leaves_list(linkage(np.nan_to_num(a_eff.to_numpy(float)), method="ward"))
        a_eff, a_sig = a_eff.iloc[ro], a_sig.iloc[ro]
    Ma = a_eff.to_numpy(float)
    Sa = a_sig.to_numpy(bool)
    n_sig_atlas = int(Sa.sum())
    n_shown = len(a_eff)

    # ------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(mm(185), mm(205)))
    xL, xW = 0.345, 0.515                                # shared left / width so columns of a,b align
    drug_disp = [DRUG_LABELS.get(d, d) for d in order]

    # ===== (a) context-conditioned coefficient atlas =================================================
    ax_a = fig.add_axes([xL, 0.50, xW, 0.43])
    lim_a = float(np.nanpercentile(np.abs(Ma[Sa]), 98)) if Sa.any() else 0.1
    norm_a = TwoSlopeNorm(vmin=-lim_a, vcenter=0, vmax=lim_a)
    ax_a.imshow(np.where(Sa, np.nan, 0.0), cmap="Greys", vmin=0, vmax=1, alpha=0.16, aspect="auto",
                interpolation="nearest")
    im_a = ax_a.imshow(np.where(Sa, Ma, np.nan), cmap=DIVERGING_CMAP, norm=norm_a, aspect="auto",
                       interpolation="nearest")
    im_a.cmap.set_bad((1, 1, 1, 0))
    ax_a.set_xticks(range(len(order)))
    ax_a.set_xticklabels([])
    ax_a.set_yticks(range(len(a_eff)))
    ax_a.set_yticklabels([_clean_context(c) for c in a_eff.index], fontsize=4.6)
    ax_a.tick_params(length=2, width=0.4, pad=1)
    ax_a.set_ylabel("Interpretable context (clustered)", fontsize=6.5, labelpad=2)
    ax_a.set_title("Context-conditioned coefficient map  β(C):  context × drug\n"
                   f"{n_shown} clustered contexts · coloured = significant, grey = n.s. · full atlas Fig. S5",
                   fontsize=5.8, pad=4, loc="left", x=0.0, linespacing=1.3)
    cax_a = fig.add_axes([0.885, 0.60, 0.013, 0.27])
    cb_a = fig.colorbar(im_a, cax=cax_a, extend="both")
    cb_a.outline.set_linewidth(0.5)
    cb_a.ax.tick_params(width=0.5, length=2, labelsize=5)
    cb_a.set_label("drug-specific Δβ association", fontsize=5.5)
    panel_label(ax_a, "a", dx=-0.30, dy=1.05)

    # ===== (b) archetypal coarsening: subtype × drug ================================================
    ax_b = fig.add_axes([xL, 0.345, xW, 0.135])
    vlim = max(0.05, float(np.nanmax(np.abs(Ms))))
    norm_b = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    ax_b.imshow(Ms, cmap=DIVERGING_CMAP, norm=norm_b, aspect="auto")
    ax_b.set_xticks(range(len(order)))
    ax_b.set_xticklabels(drug_disp, rotation=45, ha="right", fontsize=5.5)
    ax_b.tick_params(axis="x", length=2, pad=1)
    ax_b.set_yticks(range(5))
    for i in range(5):
        for j in range(len(order)):
            if bool(sig.to_numpy()[i, j]):
                ax_b.text(j, i, "*", ha="center", va="center", fontsize=9, fontweight="bold",
                          color="white" if abs(norm_b(Ms[i, j]) - 0.5) > 0.3 else "#111111")
    ax_b.set_yticklabels(NAMES, fontsize=6)
    ax_b.tick_params(axis="y", length=0, pad=10)
    side = inset_axes(ax_b, width="100%", height="100%", bbox_to_anchor=(-0.022, 0.0, 0.014, 1.0),
                      bbox_transform=ax_b.transAxes, borderpad=0)
    side.imshow(np.arange(5).reshape(5, 1), cmap=ListedColormap(COLORS), aspect="auto")
    side.set_xticks([]); side.set_yticks([])
    for sp in side.spines.values():
        sp.set_visible(False)
    ax_b.set_ylabel("Treatment-response subtype", fontsize=6.5, labelpad=62)
    ax_b.set_title("Archetypal coarsening — same drugs, pooled into 5 subtypes   ·   * = FDR q < 0.05",
                   fontsize=6.3, pad=4, loc="left", x=0.0)
    cax_b = fig.add_axes([0.885, 0.355, 0.013, 0.115])
    cb_b = fig.colorbar(plt.cm.ScalarMappable(norm=norm_b, cmap=DIVERGING_CMAP), cax=cax_b)
    cb_b.outline.set_linewidth(0.5)
    cb_b.ax.tick_params(width=0.5, length=2, labelsize=5)
    cb_b.set_label("drug-specific Δβ", fontsize=5.5)
    cb_b.ax.text(0.5, -0.08, "▼ more favorable recorded association", transform=cb_b.ax.transAxes, ha="center", va="top",
                 fontsize=4.6)
    panel_label(ax_b, "b", dx=-0.30, dy=1.10)

    # ===== (c) variance decomposition of the per-patient coefficient vectors ========================
    dec = pd.read_csv(DECOMP).set_index("component")["variance_share"]
    grad, drug = float(dec["responsiveness_gradient"]), float(dec["drug_specific"])
    c_drug, c_grad = "#4C566A", "#9ECAE1"
    ax_c = fig.add_axes([0.255, 0.055, 0.16, 0.175])
    ax_c.pie([drug, grad], colors=[c_drug, c_grad], startangle=90, counterclock=False,
             wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.0))
    ax_c.text(0, 0, "variance\nof β(C)", ha="center", va="center", fontsize=5.5, color="#444444")
    ax_c.set_title(f"Coefficient heterogeneity is large\n(n = {n_patients:,} patients)", fontsize=6.2, pad=6)
    ax_c.legend(handles=[Patch(facecolor=c_drug, label=f"drug-specific  {drug*100:.0f}%"),
                         Patch(facecolor=c_grad, label=f"responsiveness gradient  {grad*100:.0f}%")],
                loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False,
                handlelength=1.1, handleheight=1.1, borderpad=0.2, labelspacing=0.7)
    panel_label(ax_c, "c", dx=-0.30, dy=1.18)

    # caption + drug key in the clean band along the bottom (two lines so neither runs to the edge)
    axt = fig.add_axes([0.05, 0.0, 0.91, 0.05])
    axt.axis("off")
    axt.text(0.0, 0.95,
             f"Of the 130 subtype × drug associations, {n_nominal} are nominally significant ({enrich:.1f}× the "
             f"{n_total*0.05:.1f} expected); rifampicin ↔ drug-sensitive responders is the sole FDR survivor "
             "(Δβ ≈ −0.43, q < 0.001).",
             transform=axt.transAxes, va="top", ha="left", fontsize=5.2, color="#555555")
    axt.text(0.0, 0.40, "Drug key:  " + DRUG_KEY,
             transform=axt.transAxes, va="top", ha="left", fontsize=5.0, color="#777777")

    pdf, png = save_figure(fig, "fig04_drugspecific", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
