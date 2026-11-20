"""Leave-one-country-out cross-population generalization (canonical multimodal K=5 model).

Promotes cross-site transfer to a headline result: train the contextualized model on all-but-one
country, evaluate held-out-country AUROC with event counts and bootstrap 95% CIs. Countries with
adequate events transfer well; low-event countries are reported transparently (wide CI) rather than
hidden. Georgia (large n, low 7% event rate, distinct mix) is the strong-transfer case.

Usage
-----
    uv run python scripts/26_cross_population.py
"""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctb.config import FIGURES_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import apply_context_encoder, build_context_matrix, fit_context_encoder
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.style import PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)

K = 5
MIN_TEST_N = 200
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
SEED = 0


def _auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """AUROC with bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> int:
    """Run leave-one-country-out transfer and render the generalization figure."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    cond_ids = list(data["cond_ids"])
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, spec = build_context_matrix(pc, bx, cond_ids)
    craw = craw.reindex(pd.Index(cond_ids, name="condition_id")).reset_index(drop=True)
    country = craw["country"].astype("string").to_numpy()
    y = data["y"].astype(np.float64)

    counts = pd.Series(country).value_counts()
    test_countries = [ct for ct, n in counts.items() if n >= MIN_TEST_N and str(ct) != "<NA>"]
    rows = []
    for ct in test_countries:
        te = country == ct
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue
        encoder = fit_context_encoder(craw[tr], spec)
        c_tr = apply_context_encoder(craw[tr], spec, encoder).to_numpy(dtype=np.float64)
        c_te = apply_context_encoder(craw[te], spec, encoder).to_numpy(dtype=np.float64)
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED
        )
        train(m, c_tr, x[tr], y[tr], CFG, seed=SEED)
        p = m.predict_proba(c_te, x[te])
        auc, lo, hi = _auc_ci(y[te], p)
        rows.append({"country": str(ct), "n": int(te.sum()), "n_events": int(y[te].sum()),
                     "event_rate": round(100 * y[te].mean(), 1),
                     "auroc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        logger.info("%-12s n=%4d events=%3d rate=%4.1f%% AUROC=%.3f [%.3f,%.3f]",
                    ct, int(te.sum()), int(y[te].sum()), 100 * y[te].mean(), auc, lo, hi)
    tab = pd.DataFrame(rows).sort_values("auroc")
    tab.to_csv(TABLES_DIR / "cross_population_auroc.csv", index=False)

    # figure: AUROC with CI per country, well-powered vs under-powered (<30 events) distinguished
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(tab) + 1.2))
    ypos = np.arange(len(tab))
    powered = tab["n_events"] >= 30
    colors = [PALETTE[0] if pw else "#BBBBBB" for pw in powered]
    ax.errorbar(tab["auroc"], ypos,
                xerr=[tab["auroc"] - tab["ci_low"], tab["ci_high"] - tab["auroc"]],
                fmt="none", ecolor="#999999", elinewidth=1.3, capsize=3, zorder=1)
    ax.scatter(tab["auroc"], ypos, c=colors, s=45, zorder=2)
    ax.axvline(0.5, color="#cccccc", ls=":", lw=1, label="chance")
    ax.axvline(0.79, color=PALETTE[2], ls="--", lw=1, label="in-cohort CV (0.79)")
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.text(r["ci_high"] + 0.01, i, f"{r['auroc']:.2f}  (n={r['n']}, {int(r['n_events'])} ev)",
                va="center", fontsize=7, color="#444444")
    ax.set_yticks(ypos)
    ax.set_yticklabels(tab["country"])
    powered_count = int(powered.sum())
    ax.set(
        xlim=(0.1, 1.05),
        xlabel="Held-out-country AUROC (95% CI)",
        title=f"Cross-population transfer ({powered_count}/{len(tab)} powered)",
    )
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save_figure(fig, FIGURES_DIR / "cross_population_auroc")
    powered_aucs = tab.loc[powered, "auroc"]
    logger.info("Well-powered (>=30 events) countries: %d, AUROC range %.2f-%.2f, median %.2f",
                int(powered.sum()), powered_aucs.min(), powered_aucs.max(), powered_aucs.median())
    return 0


if __name__ == "__main__":
    sys.exit(main())
