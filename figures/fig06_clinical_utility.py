"""Figure 6 — prognostic risk-targeting (decision-curve analysis).

The decision-analytic view of the "treatment failure is multimodal" result, as a
deliverable. (a) Net benefit (Vickers & Elkin) versus the threshold risk at which a programme would
intensify monitoring: multimodal PATCH risk has higher net benefit than resistance-class
stratification across the examined band. (b) Translated to patients, at a representative 20%
monitoring threshold and per 1,000
culture-positive patients, how many true high-risk patients each strategy flags versus how many it
intensifies unnecessarily. Reads the two aggregate tables from ``scripts/37_decision_curve.py``.

Run: ``uv run python -m figures.fig06_clinical_utility``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.panels import panel_label, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
CURVE = TABLES_DIR / "decision_curve.csv"
YIELD = TABLES_DIR / "decision_curve_yield.csv"

C_PATCH, C_RES, C_GREY, C_DARK = "#0072B2", "#882255", "#B8B8B8", "#333333"


def main() -> int:
    set_style()
    d = pd.read_csv(CURVE)
    yld = pd.read_csv(YIELD)
    t = d["threshold"].to_numpy() * 100
    auc_res = float(d["auroc_resistance"].iloc[0])
    auc_patch = float(d["auroc_patch"].iloc[0])
    prev = float(d["treat_all"].iloc[0]) if d["treat_all"].iloc[0] > 0 else 0.16

    fig = plt.figure(figsize=(mm(183), mm(76)))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42, left=0.08, right=0.97,
                  top=0.86, bottom=0.18)

    # --- (a) decision curve --------------------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.axvspan(10, 30, color=C_PATCH, alpha=0.05, lw=0)
    ax.plot(t, d["PATCH_multimodal"], color=C_PATCH, lw=2.0, label="PATCH (multimodal context)", zorder=5)
    ax.plot(t, d["resistance_class_only"], color=C_RES, lw=1.6, label="resistance class only (standard)")
    ax.plot(t, d["treat_all"], color=C_GREY, lw=1.0, ls="--", label="intervene on all")
    ax.plot(t, d["treat_none"], color=C_DARK, lw=0.8, ls=":", label="intervene on none")
    ax.set_xlim(2, 50)
    ax.set_ylim(-0.01, prev + 0.012)
    ax.set_xlabel("threshold risk for intensified monitoring (%)", fontsize=7)
    ax.set_ylabel("net benefit", fontsize=7)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right")
    ax.text(20, -0.007, "clinically relevant band", ha="center", va="bottom", fontsize=5.2, color=C_PATCH)
    ax.set_title("Multimodal risk-targeting compared with resistance-class targeting", fontsize=7.2, pad=4)
    panel_label(ax, "a", dx=-0.10, dy=1.06)

    # --- (b) per-1,000 yield at the operating threshold --------------------------------------------
    axb = fig.add_subplot(gs[1])
    order = ["Monitor all", "Resistance class", "PATCH (multimodal)"]
    yld = yld.set_index("strategy").reindex(order)
    cols = [C_DARK, C_RES, C_PATCH]
    yy = np.arange(len(order))[::-1]
    tp = yld["true_highrisk_per_1000"].to_numpy()
    fp = yld["overmonitored_per_1000"].to_numpy()
    for yi, c, t_i, f_i in zip(yy, cols, tp, fp, strict=True):
        axb.barh(yi, t_i, color=c, edgecolor="white", lw=0.5, height=0.62, zorder=3)
        axb.barh(yi, f_i, left=t_i, color=c, alpha=0.22, edgecolor="white", lw=0.5, height=0.62, zorder=2)
        axb.text(t_i / 2, yi, f"{int(t_i)}", ha="center", va="center", fontsize=6.0, color="white",
                 fontweight="bold")
        axb.text(t_i + f_i + 12, yi, f"{int(t_i + f_i)} flagged", ha="left", va="center", fontsize=5.4,
                 color="#555555")
    axb.set_yticks(yy)
    axb.set_yticklabels(["monitor\nall", "resistance\nclass", "PATCH"], fontsize=6.0, linespacing=0.9)
    axb.set_xlim(0, 1080)
    axb.set_xlabel("per 1,000 patients (monitoring threshold 20%)", fontsize=7)
    axb.set_title("True high-risk patients flagged", fontsize=7.2, pad=4)
    axb.legend(handles=[Patch(facecolor=C_DARK, label="true high-risk flagged (will die/fail)"),
                        Patch(facecolor=C_DARK, alpha=0.22, label="unnecessarily intensified")],
               fontsize=5.4, frameon=False, loc="lower right", borderpad=0.2)
    n_true = int(yld.loc["Monitor all", "true_highrisk_per_1000"])  # = prevalence x 1,000
    axb.text(0.5, 1.12, f"PATCH flags {int(tp[2])} of {n_true} who die/fail vs {int(tp[1])} by resistance "
             f"class  (AUROC {auc_patch:.2f} vs {auc_res:.2f})", transform=axb.transAxes, fontsize=5.4,
             color=C_PATCH, va="bottom", ha="center")
    panel_label(axb, "b", dx=-0.30, dy=1.20)

    pdf, png = save_figure(fig, "fig06_clinical_utility", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
