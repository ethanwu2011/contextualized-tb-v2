"""Supplementary Figure S5 — full context-conditioned coefficient atlas (context × drug).

The complete drug-specific interaction map behind the Fig. 5 summary: for every interpretable context
indicator and drug, the gradient-removed coefficient modification Δβ, hierarchically ordered. Cells
that are NOT bootstrap-significant are greyed out so the significant cells carry the colour. Reads the
aggregate atlas table only.

Run: ``uv run python -m figures.supp.figS5_atlas_full``
"""

from __future__ import annotations

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import DIVERGING_CMAP, DIVERGING_POLES
from figures.panels import pval_annot, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
ATLAS = TABLES_DIR / "heterogeneity_atlas_unfavorable_interaction.csv"
TOP_CONTEXTS = 32

# Number of bootstrap resamples behind the significance test in the canonical atlas script.
# The 40-resample setting is part of the aggregate table provenance and is not a patient-level output.
N_BOOT = 40

# Standard / spelled-out drug names so no abbreviation is left undefined on the panel.
DRUG_LABELS = {
    "ART": "Antiretroviral therapy",
    "Am": "Amikacin",
    "Amx/Clv": "Amoxicillin/clavulanate",
    "Bdq": "Bedaquiline",
    "Cfz": "Clofazimine",
    "Cm": "Capreomycin",
    "Cotrimoxazol": "Cotrimoxazole",
    "Cs": "Cycloserine",
    "Dld": "Delamanid",
    "E": "Ethambutol",
    "Eto": "Ethionamide",
    "H": "Isoniazid",
    "Ipm/Cln": "Imipenem/cilastatin",
    "Km": "Kanamycin",
    "Lfx": "Levofloxacin",
    "Lzd": "Linezolid",
    "Mfx": "Moxifloxacin",
    "Ofx": "Ofloxacin",
    "Pa": "Pretomanid",
    "Pas": "Para-aminosalicylic acid",
    "Pto": "Protionamide",
    "R": "Rifampicin",
    "S": "Streptomycin",
    "Trd": "Terizidone",
    "Z": "Pyrazinamide",
    "other_second_line": "Other 2nd-line",
}


def _drug_label(code: str) -> str:
    """Full drug name; keep the code in parentheses for compactness where it is non-obvious."""
    name = DRUG_LABELS.get(code, code)
    if name == code or name.lower() == code.lower():
        return name
    return f"{name} ({code})"


def _clean_context(raw: str) -> str:
    """Human-readable context label from a raw schema token (no exposed feature names)."""
    s = raw.replace("_", " ")
    s = re.sub(r"\bbactec\b", "DST", s, flags=re.IGNORECASE)  # BACTEC = the DST assay system
    s = s.replace("injectible", "injectable")
    s = s.replace("p aminosalicylic acid", "para-aminosalicylic acid")
    s = s.replace("aminoglycosides injectable agents", "aminoglycoside injectables")
    s = s.replace("comorb ", "comorbidity: ")
    s = s.replace("type of resistance", "resistance type")
    s = s.replace("case definition", "case def.")
    s = s.replace("lung localization", "localization")
    s = s.replace("shadow pattern", "imaging")
    s = s.replace("sub lineage", "lineage")
    s = s.replace("baseline smear", "baseline smear")
    # DST results carry the convention =R / =S (resistant / susceptible by DST)
    return s.strip()


def _order(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    return leaves_list(linkage(np.nan_to_num(mat), method="ward"))


def main() -> int:
    set_style()
    df = pd.read_csv(ATLAS)
    eff = df.pivot(index="context", columns="drug", values="delta_beta")
    sig = df.pivot(index="context", columns="drug", values="significant").fillna(False)
    # keep the most active contexts (largest total absolute modification)
    keep = sig.index[sig.any(axis=1)]                    # all contexts with >=1 significant cell (true full atlas)
    eff, sig = eff.loc[keep], sig.loc[keep]
    ro = _order(eff.to_numpy())
    co = _order(eff.to_numpy().T)
    eff, sig = eff.iloc[ro, co], sig.iloc[ro, co]
    M = eff.to_numpy(float)
    S = sig.to_numpy(bool)
    n_sig = int(S.sum())

    fig = plt.figure(figsize=(mm(180), mm(235)))
    ax = fig.add_axes([0.32, 0.12, 0.54, 0.80])
    lim = np.nanpercentile(np.abs(M), 98)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    # Invert the significance encoding: colour ONLY the bootstrap-significant cells; grey out the
    # rest so the eye lands on the result instead of a near-uniform stipple of dots.
    M_sig = np.where(S, M, np.nan)
    bg = np.where(S, np.nan, 0.0)  # flat pale-grey layer behind non-significant cells
    ax.imshow(bg, cmap="Greys", vmin=0, vmax=1, alpha=0.18, aspect="auto",
              interpolation="nearest")
    im = ax.imshow(M_sig, cmap=DIVERGING_CMAP, norm=norm, aspect="auto",
                   interpolation="nearest")
    im.cmap.set_bad((1, 1, 1, 0))  # non-significant cells transparent -> show grey layer

    ax.set_xticks(range(eff.shape[1]))
    ax.set_xticklabels([_drug_label(c) for c in eff.columns], rotation=90, fontsize=5.0)
    ax.set_yticks(range(eff.shape[0]))
    ax.set_yticklabels([_clean_context(c) for c in eff.index], fontsize=4.0)
    ax.tick_params(length=2, width=0.4)

    ax.set_title(f"Full heterogeneity atlas — all {eff.shape[0]} contexts with a significant cell × "
                 f"{eff.shape[1]} drugs", fontsize=7, pad=4)
    # Stats provenance + scale honesty, on-panel (not only in the caption).
    pval_annot(
        ax,
        f"Coloured cells = bootstrap-significant ({n_sig}/{S.size}); white = n.s.  ·  "
        f"{N_BOOT}-resample percentile test, unadjusted.  Colour clipped at 98th pct of |Δβ|.",
        xy=(0.5, -0.34), fontsize=5.2,
    )
    pval_annot(ax, "DST results coded =R (resistant) / =S (susceptible)",
               xy=(0.5, -0.40), fontsize=5.2)

    # Dedicated colorbar axes, well clear of the rotated x-tick band (which extends below the
    # heatmap). extend='both' caps make the 98th-percentile clipping visually explicit.
    cax = fig.add_axes([0.90, 0.30, 0.018, 0.44])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label("drug-specific Δβ association", fontsize=6)
    # Two-line pole names placed beyond the extend caps, so the arrow tips never overprint them.
    def _two_line(pole: str) -> str:
        return pole.replace(" association", "\nassociation", 1)

    cb.ax.text(0.5, 1.08, _two_line(DIVERGING_POLES[1]), transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=5)
    cb.ax.text(0.5, -0.08, _two_line(DIVERGING_POLES[0]), transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=5)

    pdf, png = save_figure(fig, "figS5_atlas_full", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
