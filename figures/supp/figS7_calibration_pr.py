"""Supplementary Figure S7 — calibration and precision-recall for the unfavorable-outcome model.

Reports metrics beyond AUROC at 16.3% prevalence. (a) Out-of-fold
reliability (calibration) curve for the contextualized PATCH model and the concatenated-context
logistic baseline, with Brier scores. (b) Precision-recall curves with AUPRC and the prevalence
baseline. Both models are well-calibrated and PATCH matches the baseline, consistent with the AUROC
result. Reads the aggregate curve table only.

Run: ``uv run python -m figures.supp.figS7_calibration_pr``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import MODEL_COLORS
from figures.panels import panel_label, save_figure
from figures.style import SINGLE_COL_MM, mm, set_style

FIG_DIR = FIGURES_DIR
TABLE = TABLES_DIR / "calibration_auprc.csv"
PREVALENCE = 0.163
COLORS = {"PATCH": MODEL_COLORS["PATCH"], "context_LR": MODEL_COLORS["LR-with-context"]}
LABELS = {"PATCH": "PATCH (contextualized)", "context_LR": "context-concatenation LR"}


def main() -> int:
    set_style()
    df = pd.read_csv(TABLE)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(mm(2 * SINGLE_COL_MM), mm(64)))

    # --- a: calibration / reliability ---
    axA.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#999999", zorder=1, label="perfect")
    for model, sub in df[df["kind"] == "calibration"].groupby("model"):
        brier = sub["brier"].iloc[0]
        axA.plot(sub["x"], sub["y"], "-o", ms=3, lw=1.1, color=COLORS[model], zorder=3,
                 label=f"{LABELS[model]} (Brier {brier:.3f})")
    axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    axA.set_xlabel("mean predicted probability")
    axA.set_ylabel("observed fraction unfavorable")
    axA.set_title("Calibration (out-of-fold, quantile bins)", fontsize=6.5, pad=3)
    axA.legend(loc="upper left", fontsize=5.4, handlelength=1.4)
    panel_label(axA, "a", dx=-0.16, dy=1.07)

    # --- b: precision-recall ---
    axB.axhline(PREVALENCE, ls=":", lw=0.6, color="#999999", zorder=1,
                label=f"prevalence ({PREVALENCE:.2f})")
    for model, sub in df[df["kind"] == "pr"].groupby("model"):
        ap = sub["auprc"].iloc[0]
        axB.plot(sub["x"], sub["y"], "-", lw=1.2, color=COLORS[model], zorder=3,
                 label=f"{LABELS[model]} (AUPRC {ap:.3f})")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_xlabel("recall")
    axB.set_ylabel("precision")
    axB.set_title("Precision-recall (out-of-fold)", fontsize=6.5, pad=3)
    axB.legend(loc="upper right", fontsize=5.4, handlelength=1.4)
    panel_label(axB, "b", dx=-0.16, dy=1.07)

    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.97, bottom=0.16, top=0.88)
    pdf, png = save_figure(fig, "figS7_calibration_pr", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
