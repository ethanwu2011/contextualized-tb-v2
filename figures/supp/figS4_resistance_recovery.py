"""Supplementary Figure S4 — rifampicin receipt differs across treatment-response subtypes.

The subtypes are derived from the contextualized model loadings, and resistance variables are part of
the context supplied to that model. The figure therefore reports a descriptive alignment between
subtype assignment and rifampicin receipt, not an unsupervised recovery claim. Aggregates internally
and emits only the figure.

Run: ``uv run python -m figures.supp.figS4_resistance_recovery``
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

from ctb.config import FIGURES_DIR, PROCESSED_DIR
from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes
from figures.palette import NARRATIVE_ORDER, subtype_color, subtype_name
from figures.panels import pval_annot, save_figure
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
SUBNPZ = PROCESSED_DIR / "multimodal_subtypes.npz"
ORDER = NARRATIVE_ORDER  # canonical labels in narrative order (responders first)


def main() -> int:
    set_style()
    mat = load_primary_matrix()
    X = mat["X"].astype(float)
    drugs = [str(d) for d in mat["drugs"]]
    cond_mat = list(mat["cond_ids"])
    sd = load_primary_subtypes(SUBNPZ)
    sub = sd["subtypes"].astype(int)
    cond_sub = list(sd["cond_ids"])

    # align subtypes to the X row order
    pos = {c: i for i, c in enumerate(cond_sub)}
    sub_aligned = np.array([sub[pos[c]] for c in cond_mat])
    assert "drug_R" in drugs, f"drug_R missing from {drugs[:10]}..."
    rif = X[:, drugs.index("drug_R")]

    # per-subtype counts (rifampicin is a binary 0/1 receipt indicator)
    num = np.array([rif[sub_aligned == k].sum() for k in range(5)])
    den = np.array([(sub_aligned == k).sum() for k in range(5)])
    pct = 100 * num / den
    # Wilson 95% CI on each proportion (proportion CI, not SEM)
    ci_lo, ci_hi = proportion_confint(num, den, method="wilson")
    err_lo = pct - 100 * ci_lo
    err_hi = 100 * ci_hi - pct
    # omnibus chi-square: does receipt differ across the 5 subtypes?
    chi2, p_omni, _, _ = chi2_contingency(np.vstack([num, den - num]))
    p_str = "p < 1e-300" if p_omni == 0 else f"p = {p_omni:.1e}"

    fig = plt.figure(figsize=(mm(110), mm(68)))
    ax = fig.add_axes([0.13, 0.40, 0.83, 0.45])
    x = np.arange(5)
    cols = [subtype_color(k) for k in ORDER]
    vals = [pct[k] for k in ORDER]
    yerr = np.array([[err_lo[k] for k in ORDER], [err_hi[k] for k in ORDER]])
    ax.bar(x, vals, color=cols, edgecolor="white", lw=0.5, width=0.72, zorder=2)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.7,
                capsize=2, capthick=0.7, zorder=3)
    for xi, k in zip(x, ORDER, strict=True):
        ax.text(xi, pct[k] + err_hi[k] + 2.0, f"{pct[k]:.1f}%", ha="center", va="bottom",
                fontsize=6)
        ax.text(xi, -4.0, f"n={den[k]}", ha="center", va="top", fontsize=5.5, color="#555555")

    ax.set_xticks(x)
    # Keep the full canonical labels while wrapping them so the five categories remain legible
    # at the single-column figure width.
    wrapped_labels = {
        0: "Drug-sensitive\nresponders",
        2: "Mixed-spectrum\nresistance\nresponders",
        4: "MDR-predominant\nresponders",
        1: "High-burden /\nregional failure",
        3: "HIV-enriched severe-\nresistance failure",
    }
    ax.set_xticklabels([wrapped_labels[k] for k in ORDER], rotation=0, ha="center", fontsize=5.2)
    ax.tick_params(axis="x", pad=16)
    ax.set_xlabel("treatment-response subtype  (cool = responders · warm = failures)", labelpad=20)
    ax.set_ylabel("received rifampicin (% of subtype)")
    ax.set_ylim(0, 100)

    # responder vs failure grouping brackets (three responders, then two failures).
    # Drawn in the upper-interior whitespace so they never reach the title / stat line.
    n_resp = sum(k in (0, 2, 4) for k in ORDER)

    def _bracket(x0: float, x1: float, label: str, y: float) -> None:
        ax.plot([x0, x0, x1, x1], [y, y + 2.5, y + 2.5, y], color="#888888", lw=0.6)
        ax.text((x0 + x1) / 2, y + 3.0, label, ha="center", va="bottom", fontsize=5.5,
                color="#555555")

    # responders span all three responder bars (incl. the tall drug-sensitive one) just below 100;
    # failures sit lower over the two short failure bars.
    _bracket(-0.36, n_resp - 1 + 0.36, "responders", 95.0)
    _bracket(n_resp - 0.36, 4 + 0.36, "failures", 45.0)

    ax.set_title("Rifampicin receipt differs across treatment-response subtypes",
                 fontsize=7, pad=4)
    # stat string placed in the right-side interior whitespace (above the short bars), not the title
    ax.text(0.62, 0.93, f"chi-square across subtypes\n{p_str}  ·  n = {den.sum():,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6)
    fig.text(0.13, 0.06, "Subtypes use model loadings and recorded context. Rifampicin receipt "
             "is evaluated descriptively.", fontsize=6.0, color="#555555")

    pdf, png = save_figure(fig, "figS4_resistance_recovery", FIG_DIR)
    print(f"wrote {pdf.name}, {png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
