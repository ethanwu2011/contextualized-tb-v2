"""Per-subtype drug-specific treatment effects on the canonical multimodal K=5 model.

Same label-matched bootstrap as the no-resistance analysis, but on the full multimodal context (the
canonical model from script 22), so the drug-effect figures are consistent with the headline
subtypes. Reports the gradient-removed (drug-specific) interactions with bootstrap CIs.

Usage
-----
    uv run python scripts/25_multimodal_drug_effects.py
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
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_subtype_interaction

logger = logging.getLogger(__name__)

K = 5
N_BOOT = 100
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
SEED = 0


def _match(ref: NDArray[np.int_], other: NDArray[np.int_], k: int) -> dict[int, int]:
    """Map ``other`` labels to ``ref`` by max patient overlap."""
    overlap = np.zeros((k, k))
    for i in range(k):
        ri = ref == i
        for j in range(k):
            overlap[i, j] = int(np.logical_and(ri, other == j).sum())
    r_idx, o_idx = linear_sum_assignment(-overlap)
    return {int(o): int(r) for r, o in zip(r_idx, o_idx, strict=True)}


def main() -> int:
    """Bootstrap drug-specific subtype effects on the multimodal model."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    y = data["y"].astype(np.float64)

    ref = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED
    )
    train(ref, c, x, y, CFG, seed=SEED)
    ref_sub = ref.predict_loadings(c).argmax(axis=1)

    nd = x.shape[1]
    spec_ = np.full((N_BOOT, K, nd), np.nan)
    rng = np.random.default_rng(SEED)
    n = c.shape[0]
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED + b
        )
        train(m, c[idx], x[idx], y[idx], CFG, seed=SEED + b)
        beta, _ = m.predict_beta(c)
        beta_pop = beta.mean(axis=0)
        sub = m.predict_loadings(c).argmax(axis=1)
        for bl, rl in _match(ref_sub, sub, K).items():
            sel = sub == bl
            if sel.sum() == 0:
                continue
            inter = beta[sel].mean(axis=0) - beta_pop
            spec_[b, rl, :] = inter - inter.mean()
        logger.info("bootstrap %d/%d", b + 1, N_BOOT)

    from scipy.stats import false_discovery_control, norm
    est = np.nanmean(spec_, axis=0)
    se = np.nanstd(spec_, axis=0, ddof=1)  # bootstrap standard error per (subtype, drug)
    lo = np.nanpercentile(spec_, 2.5, axis=0)
    hi = np.nanpercentile(spec_, 97.5, axis=0)
    rows = []
    for s in range(K):
        for j, dn in enumerate(drugs):
            mean = float(est[s, j])
            sd = float(se[s, j])
            # two-sided z-test on the bootstrap distribution (effect / bootstrap SE)
            pval = float(2 * norm.sf(abs(mean / sd))) if (np.isfinite(sd) and sd > 0) else 1.0
            rows.append({"subtype": f"S{s}", "drug": dn, "drug_specific": mean,
                         "ci_low": float(lo[s, j]), "ci_high": float(hi[s, j]),
                         "se": sd, "pval": pval})
    ds = pd.DataFrame(rows)
    ds["p_fdr"] = false_discovery_control(ds["pval"].to_numpy(), method="bh")
    ds["significant"] = ds["p_fdr"] < 0.05  # FDR-corrected (130 tests)
    ds.to_csv(TABLES_DIR / "multimodal_subtype_drugspecific.csv", index=False)
    n_nominal = int((ds["pval"] < 0.05).sum())
    logger.info("Collective enrichment: %d of %d nominally significant (expected %.1f at 5%%)",
                n_nominal, len(ds), 0.05 * len(ds))
    keep = ds[ds.significant]["drug"].unique().tolist() or ds["drug"].unique().tolist()
    dsk = ds[ds.drug.isin(keep)]
    figure_subtype_interaction(
        dsk.pivot(index="subtype", columns="drug", values="drug_specific"),
        dsk.pivot(index="subtype", columns="drug", values="significant"),
        name="multimodal_subtype_drugspecific",
        title="Drug-specific subtype effects (FDR-significant, z-test; gradient removed)",
        cbar_label="drug-specific Δβ association (negative = more favorable recorded association)")
    logger.info("FDR-significant (z-test): %d of %d", int(ds.significant.sum()), len(ds))
    logger.info("Top FDR-significant:\n%s", ds[ds.significant].reindex(
        ds[ds.significant]["drug_specific"].abs().sort_values().index).tail(12)[
        ["subtype", "drug", "drug_specific", "p_fdr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
