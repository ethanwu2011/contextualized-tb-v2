"""Archetype-count sweep (K = 4..8) for the contextualized model.

With the context matrix now 335-dimensional (vs 14 originally), more than 5 archetypes may be
warranted. This sweep evaluates, for each K:

* **Performance** — 5-fold cross-validated AUROC (with a bootstrap CI).
* **Patient distribution** — hard-assigning each patient to their top-loading archetype, how the
  cohort splits across subtypes (are new archetypes balanced subtypes, or tiny slivers?).
* **Archetype profiles** — the canonical per-drug treatment-effect signature of each archetype,
  saved for interpretation at every K.

Interpretability — not raw AUROC — is the deciding criterion. The outcome is the **unfavorable
treatment response** (died OR failure), the higher-power mortality-family endpoint.

Usage
-----
    uv run python scripts/11_archetype_sweep.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctb.config import SEED, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import FeatureSpec, apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_archetype_distribution_sweep,
    figure_archetype_k_sweep,
)

logger = logging.getLogger(__name__)

# The prespecified model-selection grid reported in the manuscript and supplementary figure.
K_RANGE = range(4, 9)
N_SPLITS = 5
MIN_SUBTYPE_FRAC = 0.02  # "active" archetype = top subtype for >=2% of patients
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)


def _oof_auc(
    raw_context: pd.DataFrame,
    spec: FeatureSpec,
    x: NDArray[np.float64],
    y: NDArray[np.int_],
    k: int,
) -> NDArray[np.float64]:
    """Out-of-fold predicted probabilities for a K-archetype contextualized model."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        fold_seed = SEED + fold
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=fold_seed
        )
        train(m, c_tr, x[tr], y[tr], CFG, seed=fold_seed)
        oof[te] = m.predict_proba(c_te, x[te])
    return oof


def _auc_ci(
    y: NDArray[np.int_], s: NDArray[np.float64], n_boot: int = 500
) -> tuple[float, float, float]:
    """AUROC with a bootstrap 95% CI (resampling predictions, no model refit)."""
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> int:
    """Run the archetype-count sweep and write tables + figures.

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
    y = data["y"].astype(int)
    raw_context, spec = load_primary_raw_context(list(data["cond_ids"]))
    logger.info("Outcome = died OR failure: %.1f%% positive (n=%d)", 100 * y.mean(), int(y.sum()))

    sweep_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    arch_rows: list[dict[str, object]] = []
    for k in K_RANGE:
        oof = _oof_auc(raw_context, spec, x, y, k)
        auc, lo, hi = _auc_ci(y, oof)

        model = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=SEED
        )
        train(model, c, x, y, CFG, seed=SEED)
        loadings = model.predict_loadings(c)
        assign = loadings.argmax(axis=1)
        fracs = np.bincount(assign, minlength=k) / len(assign)
        sorted_fracs = np.sort(fracs)[::-1]
        n_active = int((fracs >= MIN_SUBTYPE_FRAC).sum())
        # population-level usage entropy (how evenly patients spread across archetypes)
        p = fracs[fracs > 0]
        entropy = float(-(p * np.log(p)).sum() / np.log(k))  # normalized 0..1
        sweep_rows.append(
            {
                "k": k,
                "auc": auc,
                "ci_low": lo,
                "ci_high": hi,
                "n_active": n_active,
                "usage_entropy": entropy,
                "largest_frac": float(sorted_fracs[0]),
                "smallest_active_frac": float(fracs[fracs >= MIN_SUBTYPE_FRAC].min()),
            }
        )
        for slot, fr in enumerate(sorted_fracs):
            dist_rows.append({"k": k, "slot": slot, "frac": float(fr)})
        profiles = model.normalized_archetypes()
        for a in range(k):
            for j, dn in enumerate(drugs):
                arch_rows.append(
                    {"k": k, "archetype": a, "drug": dn, "weight": float(profiles[a, j])}
                )
        logger.info(
            "K=%2d | AUC=%.3f [%.3f-%.3f] | active=%d/%d | entropy=%.2f | largest=%.0f%%",
            k, auc, lo, hi, n_active, k, entropy, 100 * sorted_fracs[0],
        )

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(TABLES_DIR / "archetype_sweep.csv", index=False)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(TABLES_DIR / "archetype_sweep_distribution.csv", index=False)
    pd.DataFrame(arch_rows).to_csv(TABLES_DIR / "archetype_sweep_profiles.csv", index=False)

    figure_archetype_k_sweep(sweep, name="archetype_sweep_auc")
    figure_archetype_distribution_sweep(dist, name="archetype_sweep_distribution")
    logger.info("Archetype sweep complete for K=%d..%d.", K_RANGE.start, K_RANGE.stop - 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
