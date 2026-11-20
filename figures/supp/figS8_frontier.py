"""Supplementary Figure S8 — model performance: matched baseline and the accuracy↔interpretability frontier.

PATCH is optimized for interpretation, not maximum discrimination; this supplement carries the
predictive-performance comparison so the main figures can lead with the decomposition and its
generalization. (a) Out-of-fold model ladder: the regimen alone predicts weakly, patient context
carries the signal, and the contextualized archetype model matches the concatenated-context baseline
(n.s.). (b) Hyperparameter-sweep frontier: out-of-fold AUROC vs. the number of populated
(interpretable) subtypes — configurations reach/exceed the baseline only by relaxing the load-balancing
penalty and collapsing the subtypes; the canonical operating point spends ~0.005 AUROC, within
bootstrap noise, to keep five balanced subtypes. Reads aggregate tables only.

Run: ``uv run python -m figures.supp.figS8_frontier``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import subtype_color
from figures.panels import panel_label, pval_annot, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
LADDER = TABLES_DIR / "model_ladder_unfavorable.csv"
SWEEP = TABLES_DIR / "hparam_sweep.csv"
BASELINE = 0.7933  # concatenated-context LR (default), from scripts/33


def main() -> int:
    set_style()
    fig = plt.figure(figsize=(mm(180), mm(72)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.34)

    # --- a: model ladder (regimen -> context -> +regimen -> PATCH) ---
    axA = fig.add_subplot(gs[0])
    lad = pd.read_csv(LADDER)
    n = len(lad)
    yy = np.arange(n)[::-1].astype(float)
    # context-only and PATCH share the prognostic-ceiling colour; regimen and black-box concat are greys
    rung_c = (["#B0B0B0", "#0072B2", "#6C7787", "#0072B2"] if n == 4
              else ["#B0B0B0", "#6C7787", "#2E3440"])
    for k in range(n):
        r = lad.iloc[k]
        axA.plot([r.ci_low, r.ci_high], [yy[k], yy[k]], color=rung_c[k], lw=1.6, solid_capstyle="round")
        axA.scatter(r.auc, yy[k], s=26, color=rung_c[k], zorder=4, edgecolors="white", lw=0.5)
    ceil = lad.iloc[1].auc                                   # context-only = prognostic ceiling
    axA.axvline(ceil, ls="--", lw=0.6, color="#0072B2", alpha=0.55, zorder=0)
    a_reg, a_ctx = lad.iloc[0].auc, lad.iloc[1].auc          # regimen -> context jump (the real signal)
    axA.annotate("", xy=(a_ctx, yy[1]), xytext=(a_reg, yy[0]),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7,
                                 connectionstyle="arc3,rad=-0.25"))
    axA.text((a_reg + a_ctx) / 2 - 0.015, (yy[0] + yy[1]) / 2 + 0.18, f"+{a_ctx - a_reg:.2f}",
             ha="center", va="center", fontsize=6.0, color="#333333")
    d_edge = lad.iloc[2].auc - lad.iloc[1].auc              # black box's marginal regimen edge
    axA.text(lad.iloc[2].ci_high + 0.004, yy[2], f"+{d_edge:.3f} (regimen)", va="center", ha="left",
             fontsize=5.2, color="#666666")
    axA.text(lad.iloc[3].ci_high + 0.004, yy[3], "= ceiling (n.s.)", va="center", ha="left",
             fontsize=5.2, color="#0072B2")
    axA.set_yticks(yy)
    axA.set_yticklabels(["regimen\nonly", "context only\n(prognosis)", "+ regimen\n(black box)",
                         "archetypes\n(PATCH)"], fontsize=5.6, linespacing=0.9)
    axA.set_xlim(0.60, 0.92)
    axA.set_ylim(-0.6, n - 0.4)
    axA.set_xticks([0.6, 0.7, 0.8, 0.9])
    axA.set_xlabel("out-of-fold AUROC (95% CI)")
    axA.set_title("Context carries the signal; PATCH sits at the prognostic ceiling", fontsize=6.2, pad=3)
    pval_annot(axA, "5-fold OOF · paired bootstrap", xy=(0.5, 0.10), fontsize=6.0)
    panel_label(axA, "a", dx=-0.42, dy=1.08)

    # --- b: accuracy <-> interpretability frontier ---
    axB = fig.add_subplot(gs[1])
    df = pd.read_csv(SWEEP)
    axB.axhspan(BASELINE - 0.012, BASELINE + 0.012, color="#E8E8E8", zorder=0)
    axB.axhline(BASELINE, color="#555555", lw=0.8, ls="--", zorder=1)
    axB.text(5.35, BASELINE + 0.0013, "baseline (±boot. noise)", fontsize=5.4, color="#555555",
             ha="right", va="bottom")
    for ent, col, lab in [(1.0, subtype_color(0), "penalty on (balanced)"),
                          (0.0, "#B0B0B0", "penalty off (collapses)")]:
        d = df[df["entropy"] == ent]
        axB.scatter(d["populated"], d["oof_auroc"], s=22, color=col, edgecolors="white",
                    linewidths=0.4, label=lab, zorder=3)
    canon = df[df["canonical"]]
    if len(canon):
        cx, cy = float(canon["populated"].iloc[0]), float(canon["oof_auroc"].iloc[0])
        axB.scatter([cx], [cy], s=90, marker="*", color=subtype_color(0), edgecolors="black",
                    linewidths=0.6, zorder=5)
        axB.annotate("PATCH operating point\n(5 balanced subtypes)", xy=(cx, cy),
                     xytext=(cx - 1.7, cy - 0.006), fontsize=5.4, color="#222222", ha="left",
                     arrowprops=dict(arrowstyle="->", lw=0.5, color="#999999"))
    best = df.sort_values("oof_auroc", ascending=False).iloc[0]
    axB.annotate("best AUROC\n(penalty off, ~4 subtypes)", xy=(best["populated"], best["oof_auroc"]),
                 xytext=(best["populated"] + 0.25, best["oof_auroc"] + 0.004), fontsize=5.4,
                 color="#666666", ha="left", arrowprops=dict(arrowstyle="->", lw=0.5, color="#999999"))
    axB.set_xlabel("populated (interpretable) subtypes")
    axB.set_ylabel("out-of-fold AUROC (died / failure)")
    axB.set_xlim(2.8, 5.5)
    axB.set_title("Accuracy ↔ interpretability frontier (≈0.79 ceiling is model-independent)",
                  fontsize=6.6, pad=3)
    # Legend lower-RIGHT (the operating-point annotation occupies the lower-left); the right margin
    # past 5 populated subtypes is empty, so it clears both the points and the annotation.
    axB.legend(loc="lower right", fontsize=5.6, handlelength=1.2)
    panel_label(axB, "b", dx=-0.16, dy=1.08)

    pdf, png = save_figure(fig, "figS8_frontier", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
