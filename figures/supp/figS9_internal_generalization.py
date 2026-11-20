"""Supplementary Figure S9 — internal generalization: temporal persistence and cross-population transfer.

(a) Temporal hold-out: subtype risk stratification recurs in future (2021+) patients (Spearman rho of
    train-vs-test unfavorable rates reported on-panel).
(b) Leave-one-country-out transfer with 95% CIs.

The predictive-performance comparison (model ladder, nested-CV tuning, accuracy↔interpretability
frontier) is reported in Supplementary Fig. S8 — PATCH is optimized for interpretation and matches a
standard classifier on discrimination, so the main figures lead with structure and generalization.
Reads aggregate tables only.

Run: ``uv run python -m figures.figS9_internal_generalization``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import NARRATIVE_ORDER, POWERED, UNDERPOWERED, subtype_color, subtype_name
from figures.panels import panel_label, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
TRAIN = TABLES_DIR / "temporal_subtypes_train.csv"
TEST = TABLES_DIR / "temporal_subtypes_test.csv"
CROSS = TABLES_DIR / "cross_population_auroc.csv"
TEMPORAL = TABLES_DIR / "temporal_validation.csv"
TEMPORAL_LEAKFREE = TABLES_DIR / "temporal_validation_leakfree.csv"
CEILING, MIN_EVENTS = 0.79, 30


def main() -> int:
    set_style()
    fig = plt.figure(figsize=(mm(150), mm(74)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.55)

    # --- a: temporal persistence ---
    axB = fig.add_subplot(gs[0])
    tr = pd.read_csv(TRAIN).set_index("subtype")
    te = pd.read_csv(TEST).set_index("subtype")
    order = NARRATIVE_ORDER
    x = np.arange(5)
    w = 0.38
    tick_cols = []
    for j, s in enumerate(order):
        col = subtype_color(s)
        tick_cols.append(col)
        axB.bar(x[j] - w / 2, tr.loc[f"S{s}", "unfav%"], w, color=col, edgecolor="white", lw=0.4)
        axB.bar(x[j] + w / 2, te.loc[f"S{s}", "unfav%"], w, color=col, edgecolor="white", lw=0.4,
                alpha=0.5, hatch="///")
    labels = []
    for s in order:
        words = subtype_name(s).split(" ")
        mid = (len(words) + 1) // 2
        labels.append(" ".join(words[:mid]) + "\n" + " ".join(words[mid:]))
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=6.0,
                        linespacing=0.9)
    for tick, col in zip(axB.get_xticklabels(), tick_cols, strict=True):
        tick.set_color(col)
    axB.set_ylabel("unfavorable (%)", fontsize=7.0)
    axB.set_ylim(0, 50)
    axB.set_title("Risk stratification recurs out-of-time", fontsize=7.0, pad=14)
    split_handles = [
        Patch(facecolor="#888888", edgecolor="white", label="train ≤2020"),
        Patch(facecolor="#888888", edgecolor="white", alpha=0.5, hatch="///", label="test 2021+"),
    ]
    axB.legend(handles=split_handles, loc="lower center", fontsize=6.0, handletextpad=0.4,
               ncol=2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0)
    tr_rates = [tr.loc[f"S{s}", "unfav%"] for s in order]
    te_rates = [te.loc[f"S{s}", "unfav%"] for s in order]
    rho, _ = stats.spearmanr(tr_rates, te_rates)
    temporal = pd.read_csv(TEMPORAL).iloc[0]
    leakfree = pd.read_csv(TEMPORAL_LEAKFREE).iloc[0]
    temporal_auroc = float(temporal["auroc_contextualized"])
    leakfree_auroc = float(leakfree["leakfree_auroc_contextualized"])
    temporal_ntest = int(temporal["n_test"])
    temporal_nevents = int(round(float((te["n"] * te["unfav%"] / 100).sum())))
    axB.text(
        0.03, 0.97,
        f"Spearman $\\rho$={rho:.2f} (train vs test rank)\n"
        f"out-of-time AUROC {temporal_auroc:.3f} (leak-free {leakfree_auroc:.3f})\n"
        f"n={temporal_ntest:,} test, about {temporal_nevents} events",
        transform=axB.transAxes, ha="left", va="top", fontsize=6.0, linespacing=1.1,
    )
    panel_label(axB, "a", dx=-0.26, dy=1.10)

    # --- b: cross-population ---
    axC = fig.add_subplot(gs[1])
    df = pd.read_csv(CROSS).sort_values("auroc").reset_index(drop=True)
    yc = np.arange(len(df))
    powered = df["n_events"] >= MIN_EVENTS
    colors = [POWERED if p else UNDERPOWERED for p in powered]
    axC.axvspan(0.0, 0.5, color="#F4F4F4", zorder=0)
    axC.axvline(0.5, color="#999999", lw=0.6, ls=":", zorder=1)
    axC.axvline(CEILING, color="#444444", lw=0.7, ls="--", zorder=1)
    for yi, (_, r) in zip(yc, df.iterrows(), strict=True):
        axC.plot([r.ci_low, r.ci_high], [yi, yi], color=colors[yi], lw=1.4, solid_capstyle="round",
                 zorder=3)
        axC.scatter(r.auroc, yi, s=20, color=colors[yi], zorder=4, edgecolors="white", lw=0.5)
    axC.set_yticks(yc)
    axC.set_yticklabels(df["country"], fontsize=6.0)
    axC.set_xlim(0.1, 1.0)
    axC.set_xlabel("held-out-country AUROC (95% CI)")
    axC.set_ylim(-1.4, len(df) - 0.3)
    # Reference lines explained via a single legend (no floating rotated labels to collide with the
    # markers). Placed upper-left, the region the high-AUROC countries vacate.
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=POWERED, edgecolor="none", label=f"≥{MIN_EVENTS} events"),
        Patch(facecolor=UNDERPOWERED, edgecolor="none", label="under-powered (<30)"),
        Line2D([0], [0], color="#999999", lw=0.8, ls=":", label="chance (0.5)"),
        Line2D([0], [0], color="#444444", lw=0.8, ls="--", label=f"pooled internal ({CEILING:.2f})"),
    ]
    axC.legend(handles=handles, loc="upper left", fontsize=5.6, handlelength=1.5, labelspacing=0.3,
               borderpad=0.3, handletextpad=0.5, bbox_to_anchor=(-0.02, 1.0))
    axC.set_title("Cross-population transfer", fontsize=7.0, pad=3)
    panel_label(axC, "b", dx=-0.26, dy=1.10)

    pdf, png = save_figure(fig, "figS9_internal_generalization", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
