"""Subtype × treatment-effect interactions — the headline result of the contextualized model.

The contextualized model gives every patient a context-conditioned drug coefficient vector
``beta(C_i)`` and a soft archetype membership. Hard-assigning patients to subtypes, this script
quantifies which recorded drug coefficients differ by subtype, with uncertainty, on the
unfavorable-treatment-response outcome (died OR
failure):

* **Per-subtype effect** ``beta_{s,d}`` — mean treatment effect of drug ``d`` within subtype ``s``
  (negative = protective against an unfavorable outcome).
* **Interaction** ``Δ_{s,d} = beta_{s,d} − beta_population_d`` — how a subtype's response *differs*
  from the cohort average (this is the effect-modification the model unlocks).

Uncertainty is a model-refit bootstrap: each resample refits the K-archetype model, re-derives
per-patient effects, hard-assigns subtypes, and **matches** them back to the reference subtypes by
maximum patient overlap (Hungarian assignment) to defeat archetype label-switching. A cell is
flagged when its bootstrap 95% CI excludes zero.

**Caveat.** Effects are observational (treatment-by-indication): a subtype "benefiting" from a drug
partly reflects that the drug was *given* to that subtype. These are response *associations*, not
trial effects.

Usage
-----
    uv run python scripts/13_subtype_treatment_effects.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    select_n_archetypes,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_subtype_interaction

logger = logging.getLogger(__name__)

MIN_SUBTYPE_FRAC = 0.03
N_BOOT = 100
EPOCHS = 2000  # must match the selection epochs (the 5-way split is epoch-sensitive)
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
BASE_SEED = 42


def _match_to_reference(
    ref_sub: NDArray[np.int_], boot_sub: NDArray[np.int_], k: int
) -> dict[int, int]:
    """Map bootstrap subtype labels to reference labels by maximum patient overlap (Hungarian)."""
    overlap = np.zeros((k, k))
    for i in range(k):
        ri = ref_sub == i
        for j in range(k):
            overlap[i, j] = int(np.logical_and(ri, boot_sub == j).sum())
    ref_idx, boot_idx = linear_sum_assignment(-overlap)
    return {int(b): int(r) for r, b in zip(ref_idx, boot_idx, strict=True)}


def main() -> int:
    """Estimate per-subtype drug effects and interactions with a label-matched bootstrap.

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
    y = data["y"].astype(np.float64)
    logger.info("Outcome = died OR failure: %.1f%% positive", 100 * y.mean())

    # Reference model (rule-selected K) -> reference subtype assignment.
    k, seed = select_n_archetypes(
        c, x, y, drug_names=drugs, k_range=range(8, 3, -1),
        min_subtype_frac=MIN_SUBTYPE_FRAC, epochs=EPOCHS,
    )
    ref = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=seed
    )
    train(ref, c, x, y, CFG, seed=seed)
    ref_sub = ref.predict_loadings(c).argmax(axis=1)
    ref_sizes = np.bincount(ref_sub, minlength=k).tolist()
    logger.info("Reference K=%d seed=%d sizes=%s", k, seed, ref_sizes)

    n_drug = x.shape[1]
    eff_boot = np.full((N_BOOT, k, n_drug), np.nan)  # per-subtype mean beta
    int_boot = np.full((N_BOOT, k, n_drug), np.nan)  # interaction: beta_subtype - beta_population
    rng = np.random.default_rng(BASE_SEED)
    n = c.shape[0]
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=BASE_SEED + b
        )
        train(m, c[idx], x[idx], y[idx], CFG, seed=BASE_SEED + b)
        beta, _ = m.predict_beta(c)  # personalized effects on the fixed population
        beta_pop = beta.mean(axis=0)
        boot_sub = m.predict_loadings(c).argmax(axis=1)
        mapping = _match_to_reference(ref_sub, boot_sub, k)
        for boot_label, ref_label in mapping.items():
            sel = boot_sub == boot_label
            if sel.sum() == 0:
                continue
            mean_beta = beta[sel].mean(axis=0)
            eff_boot[b, ref_label, :] = mean_beta
            int_boot[b, ref_label, :] = mean_beta - beta_pop
        logger.info("Subtype-effect bootstrap %d/%d complete", b + 1, N_BOOT)

    def _summary(arr: NDArray[np.float64], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        est = np.nanmean(arr, axis=0)
        lo = np.nanpercentile(arr, 2.5, axis=0)
        hi = np.nanpercentile(arr, 97.5, axis=0)
        sig = (lo > 0) | (hi < 0)
        idx = pd.Index([f"S{s}" for s in range(k)], name="subtype")
        eff_df = pd.DataFrame(est, index=idx, columns=drugs)
        sig_df = pd.DataFrame(sig, index=idx, columns=drugs)
        long = []
        for s in range(k):
            for j, dn in enumerate(drugs):
                long.append(
                    {
                        "subtype": f"S{s}", "drug": dn, label: est[s, j],
                        "ci_low": lo[s, j], "ci_high": hi[s, j], "significant": bool(sig[s, j]),
                    }
                )
        return eff_df, sig_df, pd.DataFrame(long)  # type: ignore[return-value]

    eff_df, eff_sig, eff_long = _summary(eff_boot, "beta")
    int_df, int_sig, int_long = _summary(int_boot, "interaction")
    eff_long.to_csv(TABLES_DIR / "subtype_treatment_effects.csv", index=False)
    int_long.to_csv(TABLES_DIR / "subtype_treatment_interactions.csv", index=False)

    figure_subtype_interaction(
        eff_df, eff_sig, name="subtype_treatment_effects",
        title="Drug treatment effect by patient subtype (died/failure)",
        cbar_label="β subtype (log-odds unfavorable)",
    )
    figure_subtype_interaction(
        int_df, int_sig, name="subtype_treatment_interactions",
        title="Subtype × drug interaction vs. cohort average (died/failure)",
        cbar_label="β subtype − β population (log-odds)",
    )
    n_int = int(int_sig.to_numpy().sum())
    logger.info("Subtype treatment-effect analysis complete: %d significant interactions.", n_int)
    return 0


if __name__ == "__main__":
    sys.exit(main())
