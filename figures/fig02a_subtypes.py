"""Figure 2a — the five treatment-response subtypes: outcome split + defining clinical profile.

Left: per-subtype unfavorable-outcome rate (the 3 cool responders vs 2 warm failure subtypes).
Right: z-scored enrichment heatmap of the defining features (resistance, HIV, burden, region, age)
with a subtype-colour sidebar. Reads only the aggregate characterization table (5 rows; DUA-safe).

Run: ``uv run python -m figures.fig02a_subtypes``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import (
    DIVERGING_CMAP,
    MODALITY_COLORS,
    NARRATIVE_ORDER,
    subtype_color,
    subtype_name,
)
from figures.panels import panel_label, pval_annot, save_figure, thin_colorbar
from figures.style import mm, set_style


def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score 95% CI for a binomial proportion (returned as percentages)."""
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half) * 100.0, (centre + half) * 100.0

FIG_DIR = FIGURES_DIR
TABLE = TABLES_DIR / "multimodal_subtype_characterization.csv"
ATTR = TABLES_DIR / "multimodal_modality_attribution.csv"

# Characterization-CSV row labels in narrative order (3 responders, then 2 failure subtypes).
NARRATIVE = [f"S{i}" for i in NARRATIVE_ORDER]
NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
COLORS = [subtype_color(i) for i in NARRATIVE_ORDER]
FEATURES = [
    ("sensitive%", "Drug-sensitive"), ("MDR%", "MDR"), ("XDR%", "XDR"),
    ("HIV%", "HIV+"), ("diabetes%", "Diabetes"), ("smear+%", "Smear+"),
    ("Ukraine%", "Ukraine"), ("SSAfrica%", "SS-Africa"), ("age", "Age"),
]


def main() -> int:
    set_style()
    df = pd.read_csv(TABLE).set_index("subtype").reindex(NARRATIVE)
    n = df["n"].to_numpy()
    unfav = df["unfavorable%"].to_numpy()
    raw = df[[c for c, _ in FEATURES]].to_numpy(float)
    z = (raw - raw.mean(0)) / raw.std(0, ddof=0)  # standardized enrichment across subtypes

    # Per-subtype unfavorable counts -> Wilson 95% CI and a chi-square across the 5 subtypes.
    unfav_k = np.round(n * unfav / 100.0).astype(int)
    ci_lo, ci_hi = _wilson_ci(unfav_k, n)
    contingency = np.vstack([unfav_k, n - unfav_k])
    chi2, p_chi, _, _ = chi2_contingency(contingency)

    fig = plt.figure(figsize=(mm(180), mm(100)))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.4], height_ratios=[1.45, 1.0],
                          wspace=0.05, hspace=0.55)

    # --- left: unfavorable-outcome rate bars (Wilson 95% CI error bars) ---
    axL = fig.add_subplot(gs[0, 0])
    y = np.arange(5)[::-1]
    axL.barh(y, unfav, color=COLORS, edgecolor="white", lw=0.5, height=0.72, zorder=2)
    xerr = np.vstack([unfav - ci_lo, ci_hi - unfav])
    axL.errorbar(unfav, y, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=0.6,
                 capsize=1.6, capthick=0.6, zorder=3)
    for yi, (u, hi, ni) in enumerate(zip(unfav, ci_hi, n, strict=True)):
        lab = f"{u:.1f}%" if u < 10 else f"{u:.0f}%"  # 1 dp for the near-tied responders
        axL.text(hi + 1, y[yi], lab, va="center", ha="left", fontsize=6)
        axL.text(0.5, y[yi] + 0.40, f"{NAMES[yi]}  (n={ni:,})", va="bottom", ha="left", fontsize=5.4)
    axL.set_xlim(0, 44)
    axL.set_ylim(-0.6, 4.9)
    axL.set_yticks([])
    axL.set_xlabel("unfavorable (%), 95% CI")
    axL.spines["left"].set_visible(False)
    pval_annot(axL, f"$\\chi^2$={chi2:.0f}, df=4, P<0.001", xy=(0.98, 0.10), fontsize=5.6)
    axL.texts[-1].set_ha("right")
    panel_label(axL, "a", dx=-0.02, dy=1.07)

    # --- right: z-scored enrichment heatmap ---
    axR = fig.add_subplot(gs[0, 1])
    norm = TwoSlopeNorm(vmin=min(-2, z.min()), vcenter=0, vmax=max(2, z.max()))
    axR.imshow(z, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")
    axR.set_xticks(range(len(FEATURES)))
    axR.set_xticklabels([lbl for _, lbl in FEATURES], rotation=35, ha="right", fontsize=6)
    axR.set_yticks(range(5))
    axR.set_yticklabels([])
    for i in range(5):  # raw values printed in each cell (the figure is self-contained)
        for j in range(len(FEATURES)):
            v = raw[i, j]
            txt = f"{v:.0f}" if FEATURES[j][0] != "age" else f"{v:.0f}y"
            axR.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                     color="white" if abs(z[i, j]) > 1.4 else "#222222")
    # subtype colour sidebar + short text row tags (sit fully left of the heatmap grid).
    # Tags (S0/S2/S4/S1/S3) make every row identifiable in grayscale / for colourblind readers;
    # full names with n are on panel a, so a compact tag here suffices without crowding panel a.
    for i in range(5):
        axR.add_patch(plt.Rectangle((-0.95, i - 0.5), 0.42, 1.0, color=COLORS[i],
                                    clip_on=False, zorder=5))
        axR.text(-1.02, i, NARRATIVE[i], ha="right", va="center", fontsize=5.5, clip_on=False)
    axR.set_xlim(-1.05, len(FEATURES) - 0.5)
    axR.set_xticks(range(len(FEATURES)))
    thin_colorbar(fig, axR, DIVERGING_CMAP, norm,
                  "z-scored within column (colour); cell = raw %",
                  poles=("depleted", "enriched"))
    panel_label(axR, "b", dx=-0.02, dy=1.07)
    axR.set_title("defining clinical profile  ·  cell = % of subtype (age in years)",
                  fontsize=6, pad=3, loc="left")

    # --- bottom (c): modality attribution — total vs per-feature share ---
    # Keep ALL modalities incl. 'other' (greyed) so both share series sum to 1.0 and the panel is
    # honest: 'other' (2 features) is the single LARGEST per-feature contributor, so dropping it
    # would hide the biggest per-feature category. Sorted by total share.
    axM = fig.add_subplot(gs[1, :])
    # Read the CI table (50-permutation 95% intervals) so the bars carry uncertainty.
    attr = pd.read_csv(ATTR.with_name("multimodal_modality_attribution_ci.csv")).sort_values(
        "total_share", ascending=False)
    mods = attr["modality"].tolist()
    x = np.arange(len(mods))
    w = 0.4
    cols = [MODALITY_COLORS.get(m, "#BFBFBF") for m in mods]  # 'other' -> neutral grey
    tot = attr["total_share"].to_numpy() * 100
    pf = attr["per_feature_share"].to_numpy() * 100
    tot_err = np.vstack([tot - attr["total_lo"] * 100, attr["total_hi"] * 100 - tot])
    pf_err = np.vstack([pf - attr["perfeat_lo"] * 100, attr["perfeat_hi"] * 100 - pf])
    axM.bar(x - w / 2, tot, w, color=cols, edgecolor="white", lw=0.5)
    axM.errorbar(x - w / 2, tot, yerr=tot_err, fmt="none", ecolor="#333333", elinewidth=0.5,
                 capsize=1.2, zorder=4)
    axM.bar(x + w / 2, pf, w, color=cols, edgecolor="white", lw=0.5, alpha=0.45, hatch="///")
    axM.errorbar(x + w / 2, pf, yerr=pf_err, fmt="none", ecolor="#333333", elinewidth=0.5,
                 capsize=1.2, zorder=4)
    axM.set_xticks(x)
    axM.set_xticklabels([m.replace("_", " ") for m in mods], rotation=25, ha="right", fontsize=6)
    axM.set_ylabel("contribution (%)", fontsize=6.5)
    axM.set_title("No single modality dominates — resistance leads in aggregate but is near-lowest "
                  "per feature; sparse modalities lead per feature",
                  fontsize=6.5, pad=3, loc="left")
    # Neutral solid-vs-hatched proxies: colour already encodes modality, so the legend must convey
    # only the PATTERN (which series), not a colour. Placed in interior whitespace, clear of bars.
    handles = [Patch(facecolor="#888888", edgecolor="white", label="total share"),
               Patch(facecolor="#888888", edgecolor="white", alpha=0.45, hatch="///",
                     label="per-feature share")]
    axM.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, 1.0), fontsize=6,
               handlelength=1.4, ncol=2)
    # Annotate the largest rank flip: labs is LAST in aggregate but the highest per feature among
    # the named modalities (the catch-all 'other' is higher per feature but is not a single modality).
    i_labs = mods.index("labs")
    axM.annotate("largest rank flip:\nlast in aggregate,\nhighest per feature\n"
                 "(of named modalities)",
                 xy=(i_labs + w / 2, attr["per_feature_share"].iloc[i_labs] * 100 + 0.5),
                 xytext=(i_labs - 2.6, 22), fontsize=5.4, color="#555555", ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.5, color="#999999"))
    axM.set_ylim(0, 40)
    panel_label(axM, "c", dx=-0.05, dy=1.10)

    pdf, png = save_figure(fig, "fig02_subtypes", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
