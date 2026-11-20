"""Additional treatment-response heterogeneity analyses.

Builds on the heterogeneity atlas (script 10) to answer the focused questions for the paper's
heterogeneous-treatment-effect section, on the **unfavorable treatment response (died OR failure)**
outcome:

1. **Per-drug heterogeneity** — for each drug, how context-dependent are its coefficients? (count of
   significant drug-specific context associations + total magnitude) → ranked figure.
2. **Top context associations per key drug** — forest plots (Δβ with bootstrap 95% CI) of the strongest
   context modifiers for the most heterogeneous drugs.
3. **Gradient vs drug-specific decomposition** — how much of the heterogeneity is a global
   responsiveness gradient (a context shifts *all* drugs together) vs drug-specific variation.

All from a single label-free joint-model bootstrap (:func:`atlas_bootstrap`).

Usage
-----
    uv run python scripts/14_heterogeneity_deepening.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_masks, build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import atlas_bootstrap, scan_from_atlas_boot
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_drug_heterogeneity_ranking,
    figure_effect_modifier_forests,
)

logger = logging.getLogger(__name__)

N_BOOT = 60
N_KEY_DRUGS = 6  # drugs to show effect-modifier forests for


def main() -> int:
    """Run the per-drug heterogeneity, effect-modifier, and gradient-decomposition analyses.

    Returns
    -------
    int
        Process exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = list(data["cond_ids"])

    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    c_raw, spec = build_context_matrix(pc, bx, cond_ids)
    c_raw = c_raw.reindex(pd.Index(cond_ids, name="condition_id"))
    masks = build_context_masks(c_raw, spec)

    y = data["y"].astype(np.float64)
    logger.info("Outcome = died OR failure: %.1f%% positive", 100 * y.mean())

    boot, names = atlas_bootstrap(c, x, y, masks, n_archetypes=5, n_boot=N_BOOT, epochs=2000)
    raw = scan_from_atlas_boot(boot, names, drugs, center_drugs=False)
    interaction = scan_from_atlas_boot(boot, names, drugs, center_drugs=True)
    raw.to_csv(TABLES_DIR / "heterogeneity_unfavorable_raw_ci.csv", index=False)
    interaction.to_csv(TABLES_DIR / "heterogeneity_unfavorable_interaction_ci.csv", index=False)

    # 1) Per-drug heterogeneity: significant *drug-specific* modifiers + total magnitude.
    prevalence = pd.Series(x.mean(axis=0), index=drugs)
    sig = interaction[interaction["significant"]]
    summary = (
        sig.assign(absd=sig["delta_beta"].abs())
        .groupby("drug")
        .agg(magnitude=("absd", "sum"), n_sig=("absd", "size"))
        .reindex(drugs)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "drug"})
    )
    summary["prevalence"] = summary["drug"].map(prevalence)
    summary.to_csv(TABLES_DIR / "drug_heterogeneity_ranking.csv", index=False)
    # Figure/forests restricted to commonly-used drugs (>=5%) so estimates are stable.
    common = summary[summary["prevalence"] >= 0.05]
    figure_drug_heterogeneity_ranking(common)
    key_drugs = common.sort_values("magnitude", ascending=False)["drug"].head(N_KEY_DRUGS).tolist()
    logger.info("Most context-dependent drugs (prevalence>=5%%): %s", key_drugs)

    # 2) Effect-modifier forests for the key drugs (raw Δβ = actual direction, with CIs).
    forest = raw[raw["drug"].isin(key_drugs) & raw["significant"]][
        ["drug", "context", "delta_beta", "ci_low", "ci_high"]
    ]
    forest.to_csv(TABLES_DIR / "effect_modifiers_key_drugs.csv", index=False)
    figure_effect_modifier_forests(forest, key_drugs)

    # 3) Gradient vs drug-specific decomposition (variance share, per outcome).
    est = np.nanmean(boot, axis=0)  # (n_context, n_drug) mean Δβ
    est = est[~np.isnan(est).any(axis=1)]  # drop contexts empty in every resample
    gradient = est.mean(axis=1, keepdims=True)  # across-drug mean per context = responsiveness
    specific = est - gradient  # residual = drug-specific modification (zero row-mean)
    # Orthogonal variance decomposition of the full (context x drug) Δβ matrix.
    var_grad = float((np.broadcast_to(gradient, est.shape) ** 2).sum())
    var_spec = float((specific**2).sum())
    frac_grad = var_grad / (var_grad + var_spec)
    decomp = pd.DataFrame(
        [{"component": "responsiveness_gradient", "variance_share": frac_grad},
         {"component": "drug_specific", "variance_share": 1 - frac_grad}]
    )
    decomp.to_csv(TABLES_DIR / "heterogeneity_decomposition.csv", index=False)
    logger.info(
        "Heterogeneity decomposition: %.0f%% global responsiveness gradient, %.0f%% drug-specific",
        100 * frac_grad, 100 * (1 - frac_grad),
    )
    logger.info("Heterogeneous-TE deepening complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
