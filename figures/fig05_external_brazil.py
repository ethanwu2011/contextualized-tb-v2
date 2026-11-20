"""Figure 5 — external validation on the Brazil (SINAN) registry.

(a) Risk transfer: the harmonized PATCH/context model fit on TB Portals retains its discrimination
when applied to 298,900 independent Brazilian patients (held-out AUROC vs the within-Brazil ceiling).
(b) Prognostic-structure replication: the TB-Portals-learned context->subtype map stratifies Brazilian
died/failure risk across a ~10x gradient, and the HIV-enriched subtype is the highest-risk group in
both cohorts. Reads the aggregate validation tables only.

Run: ``uv run python -m figures.fig05_external_brazil``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd

from ctb.config import FIGURES_DIR, TABLES_DIR
from figures.palette import POWERED, STATUS_COLORS, subtype_color, subtype_name
from figures.panels import panel_label, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
SUMMARY = TABLES_DIR / "brazil_external_validation.csv"
PROFILES = TABLES_DIR / "brazil_subtype_profiles.csv"


def main() -> int:
    set_style()
    s = pd.read_csv(SUMMARY).iloc[0]
    prof = pd.read_csv(PROFILES)
    tbp = prof[prof["cohort"] == "TB_Portals"].set_index("subtype")
    bz = prof[prof["cohort"] == "Brazil_applied"].set_index("subtype")
    order = tbp["unfav%"].sort_values().index.tolist()  # subtypes by TB Portals risk

    fig = plt.figure(figsize=(mm(180), mm(64)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.34)

    # --- a: risk transfer AUROC ---
    axA = fig.add_subplot(gs[0])
    bars = [("TB Portals\n(in-sample)", s["auc_tbp_insample"], "#B0B0B0"),
            ("→ Brazil\n(PATCH)", s["auc_transfer_patch"], POWERED),
            ("→ Brazil\n(context LR)", s["auc_transfer_lr"], "#6C7787"),
            ("Brazil\n(CV ceiling)", s["auc_brazil_ceiling"], "#2E3440")]
    y = range(len(bars))[::-1]
    for yi, (lab, val, col) in zip(y, bars, strict=True):
        axA.barh(yi, val, color=col, edgecolor="white", height=0.7)
        axA.text(val + 0.004, yi, f"{val:.3f}", va="center", fontsize=6)
    axA.set_yticks(list(y))
    axA.set_yticklabels([b[0] for b in bars], fontsize=5.6)
    axA.set_xlim(0.5, 0.82)
    axA.axvline(0.5, color="#999999", lw=0.6, ls=":")
    axA.set_xlabel("AUROC (died / failure)")
    axA.set_title(f"Risk transfers to Brazil (n={int(s['n_brazil']):,})", fontsize=6.6, pad=3)
    panel_label(axA, "a", dx=-0.55, dy=1.08)

    # --- b: per-subtype died/failure, TB Portals vs Brazil (same Z(C) map) ---
    axB = fig.add_subplot(gs[1])
    import numpy as np
    x = np.arange(len(order))
    w = 0.38
    cols = [subtype_color(k) for k in order]
    axB.bar(x - w / 2, [tbp.loc[k, "unfav%"] for k in order], w, color=cols, edgecolor="white",
            lw=0.4, label="TB Portals")
    axB.bar(x + w / 2, [bz.loc[k, "unfav%"] for k in order], w, color=cols, edgecolor="white",
            lw=0.4, alpha=0.5, hatch="///", label="Brazil (applied)")
    # mark the HIV-enriched subtype (highest hiv loading in TB Portals)
    hiv_k = tbp["hiv"].idxmax()
    if hiv_k in order:
        xi = order.index(hiv_k)
        axB.annotate("HIV-enriched subtype\n~30% in both", xy=(xi, max(tbp.loc[hiv_k, "unfav%"],
                     bz.loc[hiv_k, "unfav%"]) + 1), xytext=(xi - 1.7, 34), fontsize=5.4,
                     color=STATUS_COLORS["HIV+"], ha="left",
                     arrowprops=dict(arrowstyle="->", lw=0.5, color="#999999"))
    axB.set_xticks(x)
    def _label(code: object) -> str:
        idx = int(str(code).removeprefix("S"))
        # Keep the stable index for cross-cohort matching and the full canonical name for reading.
        return f"S{idx}\n{subtype_name(idx)}"

    axB.set_xticklabels([_label(k) for k in order], fontsize=5.4)
    axB.set_xlabel("harmonized subtype (ordered by TB Portals risk)")
    axB.set_ylabel("died / failure (%)")
    axB.set_ylim(0, 40)
    axB.set_title("The prognostic structure replicates (risk 3→30%)", fontsize=6.6, pad=3)
    # neutral solid/hatch legend
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor="#888888", label="TB Portals"),
                        Patch(facecolor="#888888", alpha=0.5, hatch="///", label="Brazil (applied)")],
               loc="upper left", fontsize=5.6, handlelength=1.3)
    panel_label(axB, "b", dx=-0.12, dy=1.08)

    pdf, png = save_figure(fig, "fig05_external_brazil", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
