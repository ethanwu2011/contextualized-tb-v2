"""Figure 2, the five treatment-response subtypes.

(a) Per-subtype unfavorable-outcome rate (the 3 cool responders vs 2 warm failure subtypes), with
Wilson 95% CIs and a chi-square test across subtypes. (b) z-scored enrichment heatmap of the defining
features (resistance, HIV, burden, region, age) on the same rows as (a). (c) Modality attribution,
total and per-feature share, with Table 2's point estimates and permutation intervals.
Reads only aggregate tables (5 subtype rows; 9 modality rows; DUA-safe).

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
from figures.panels import save_figure, thin_colorbar
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

    fig = plt.figure(figsize=(mm(180), mm(104)))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.1], height_ratios=[1.3, 1.0],
                          left=0.135, right=0.93, top=0.95, bottom=0.12, wspace=0.04, hspace=0.62)

    # --- a: unfavorable-outcome rate. Rows line up with the heatmap rows in b, so each subtype is
    # named once, on the left, and both panels read across as one table. ---
    rows = np.arange(5)
    axL = fig.add_subplot(gs[0, 0])
    axL.barh(rows, unfav, color=COLORS, edgecolor="white", lw=0.5, height=0.62, zorder=2)
    xerr = np.vstack([unfav - ci_lo, ci_hi - unfav])
    axL.errorbar(unfav, rows, xerr=xerr, fmt="none", ecolor="#222222", elinewidth=0.6,
                 capsize=1.6, capthick=0.6, zorder=3)
    for r, (u, hi) in enumerate(zip(unfav, ci_hi, strict=True)):
        lab = f"{u:.1f}%" if u < 10 else f"{u:.0f}%"  # 1 dp for the near-tied responders
        axL.text(hi + 1.0, r, lab, va="center", ha="left", fontsize=6)
    axL.set_yticks(rows)
    axL.set_yticklabels([f"{name}\nn = {ni:,}" for name, ni in zip(NAMES, n, strict=True)],
                        fontsize=6, linespacing=1.15)
    axL.tick_params(axis="y", length=0, pad=3)
    axL.set_ylim(4.5, -0.5)
    axL.set_xlim(0, 45)
    axL.set_xticks([0, 10, 20, 30, 40])
    axL.set_xlabel("Died or treatment failure (%)")
    axL.spines["left"].set_visible(False)
    p_txt = "P < 0.001" if p_chi < 0.001 else f"P = {p_chi:.3f}"
    axL.text(0.98, 0.97, f"$\\chi^2$ = {chi2:,.0f}, 4 df\n{p_txt}", transform=axL.transAxes,
             ha="right", va="top", fontsize=5.8, color="#000000")

    # --- b: z-scored enrichment heatmap on the same rows ---
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    norm = TwoSlopeNorm(vmin=min(-2, z.min()), vcenter=0, vmax=max(2, z.max()))
    axR.imshow(z, cmap=DIVERGING_CMAP, norm=norm, aspect="auto",
               extent=(-0.5, len(FEATURES) - 0.5, 4.5, -0.5))
    axR.set_xticks(range(len(FEATURES)))
    axR.set_xticklabels([lbl for _, lbl in FEATURES], rotation=35, ha="right", fontsize=6)
    axR.tick_params(axis="y", left=False, labelleft=False)
    axR.set_ylim(4.5, -0.5)
    for i in range(5):  # raw values printed in each cell (the figure is self-contained)
        for j in range(len(FEATURES)):
            v = raw[i, j]
            txt = f"{v:.0f}" if FEATURES[j][0] != "age" else f"{v:.0f}y"
            axR.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                     color="white" if abs(z[i, j]) > 1.4 else "#000000")
    for side in ("top", "right", "left", "bottom"):
        axR.spines[side].set_visible(False)
    thin_colorbar(fig, axR, DIVERGING_CMAP, norm, "z-score within column",
                  poles=("depleted", "enriched"))
    axR.set_title("% of subtype (age in years)", fontsize=6, pad=3, loc="left")

    # --- c: modality attribution, total vs per-feature share ---
    # Bar heights are the point estimates of Table 2 (multimodal_modality_attribution.csv); the error
    # bars are the 2.5th to 97.5th percentiles over 50 permutation rounds (..._ci.csv). Keeps ALL
    # modalities incl. 'other' (greyed) so both share series sum to 1.0.
    axM = fig.add_subplot(gs[1, :])
    point = pd.read_csv(ATTR).set_index("modality")
    ci = pd.read_csv(ATTR.with_name("multimodal_modality_attribution_ci.csv")).set_index("modality")
    if set(point.index) != set(ci.index):
        raise ValueError("attribution point and interval tables list different modalities")
    point = point.sort_values("total_share", ascending=False)
    ci = ci.reindex(point.index)
    mods = point.index.tolist()
    x = np.arange(len(mods))
    w = 0.4
    cols = [MODALITY_COLORS.get(m, "#BFBFBF") for m in mods]  # 'other' -> neutral grey
    tot = point["total_share"].to_numpy() * 100
    pf = point["per_feature_share"].to_numpy() * 100
    tot_err = np.clip(np.vstack([tot - ci["total_lo"] * 100, ci["total_hi"] * 100 - tot]), 0, None)
    pf_err = np.clip(np.vstack([pf - ci["perfeat_lo"] * 100, ci["perfeat_hi"] * 100 - pf]), 0, None)
    axM.bar(x - w / 2, tot, w, color=cols, edgecolor="white", lw=0.5)
    axM.errorbar(x - w / 2, tot, yerr=tot_err, fmt="none", ecolor="#222222", elinewidth=0.5,
                 capsize=1.2, zorder=4)
    axM.bar(x + w / 2, pf, w, color=cols, edgecolor="white", lw=0.5, alpha=0.45, hatch="///")
    axM.errorbar(x + w / 2, pf, yerr=pf_err, fmt="none", ecolor="#222222", elinewidth=0.5,
                 capsize=1.2, zorder=4)
    axM.set_xticks(x)
    axM.set_xticklabels([m.replace("_", " ") for m in mods], rotation=25, ha="right", fontsize=6)
    axM.set_ylabel("Contribution (%)", fontsize=6.5)
    # Neutral solid-vs-hatched proxies: colour already encodes modality, so the legend conveys only
    # which series is which.
    handles = [Patch(facecolor="#888888", edgecolor="white", label="Total share"),
               Patch(facecolor="#888888", edgecolor="white", alpha=0.45, hatch="///",
                     label="Per-feature share")]
    axM.legend(handles=handles, loc="upper center", fontsize=6, handlelength=1.4, ncol=2,
               frameon=False)
    axM.set_ylim(0, 40)

    # Panel letters on one left margin for a and c, and at the heatmap edge for b.
    for ax, letter, xf in ((axL, "a", 0.012), (axR, "b", None), (axM, "c", 0.012)):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.026 if xf is None else xf, pos.y1 + 0.035, letter, fontsize=9,
                 fontweight="bold", va="top", ha="left")

    pdf, png = save_figure(fig, "fig02_subtypes", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
