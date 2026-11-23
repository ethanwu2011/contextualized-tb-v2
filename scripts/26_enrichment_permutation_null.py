"""Correlation-preserving null for the drug-specific enrichment count.

The headline enrichment ("23 nominally-significant subtype x drug effects versus 6.5 expected,
a ~3.5-fold excess") in script 25 compares the observed count to a binomial/INDEPENDENCE
expectation (0.05 * 130 = 6.5). But the 130 subtype x drug cells are not independent: every cell
is read off the same K archetype signatures A through the same loadings z(C), so the cell-level
test statistics are positively correlated, which inflates the *variance* of the number of nominal
hits above the binomial value (SD sqrt(130*.05*.95) ~ 2.48). A count of 23 is therefore not
necessarily beyond the null DISTRIBUTION of the count once dependence is accounted for.

This script replaces the independence expectation with a ROTATION null that preserves the
dependence structure. The observed per-cell statistic is z_sj = D_sj / SE_sj, where D_sj is the
gradient-removed drug-specific interaction of subtype s and drug j (bootstrap mean) and SE_sj its
bootstrap SE; the count is T = #{|z_sj| > 1.96}. Under the null of no genuine subtype x drug
modification the true means are zero but the estimator's cross-cell CORRELATION R persists (it is
structural, induced by the shared A and z(C)). We estimate R from the standardized bootstrap
residuals (Ledoit-Wolf shrinkage for stability) and Monte-Carlo draw z ~ N(0, R), counting
exceedances each draw. This yields the null distribution of the count with the correct mean (6.5)
AND the correlation-inflated variance, against which the observed 23 is tested.

We also report a subtype-label permutation as an even more conservative bound (permuting patient
labels collapses each subtype toward the population mean, so its interactions --- and its count ---
go to ~0).

DUA-safe: reads the fitted context/regimen matrices; caches the aggregate bootstrap tensor and
writes only aggregate counts + a histogram.

Usage
-----
    .venv/bin/python scripts/26_enrichment_permutation_null.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from ctb.config import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
N_BOOT = 100
N_PERM = 2000
Z_THRESH = float(norm.isf(0.025))  # 1.9599...
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
SEED = 0


def _match(ref: NDArray[np.int_], other: NDArray[np.int_], k: int) -> dict[int, int]:
    overlap = np.zeros((k, k))
    for i in range(k):
        ri = ref == i
        for j in range(k):
            overlap[i, j] = int(np.logical_and(ri, other == j).sum())
    r_idx, o_idx = linear_sum_assignment(-overlap)
    return {int(o): int(r) for r, o in zip(r_idx, o_idx, strict=True)}


def drug_specific(beta: NDArray[np.float64], lab: NDArray[np.int_], k: int) -> NDArray[np.float64]:
    """Gradient-removed per-subtype drug-specific interaction (K x d)."""
    pop = beta.mean(axis=0)
    out = np.zeros((k, beta.shape[1]))
    for s in range(k):
        sel = lab == s
        if sel.sum() == 0:
            continue
        inter = beta[sel].mean(axis=0) - pop
        out[s] = inter - inter.mean()
    return out


def main() -> int:
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)

    nd = x.shape[1]
    ncells = K * nd
    cache = TABLES_DIR / "_enrichment_bootstrap_cache.npz"

    # --- reference fit + bootstrap SE (reproduces script 25's estimator); cached --------
    if cache.exists():
        z = np.load(cache)
        spec_, ref_beta, ref_sub = z["spec_"], z["ref_beta"], z["ref_sub"]
        logger.info("loaded cached bootstrap tensor %s", spec_.shape)
    else:
        ref = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED)
        train(ref, c, x, y, CFG, seed=SEED)
        ref_sub = ref.predict_loadings(c).argmax(axis=1)
        ref_beta, _ = ref.predict_beta(c)
        spec_ = np.full((N_BOOT, K, nd), np.nan)
        rng = np.random.default_rng(SEED)
        n = c.shape[0]
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED + b)
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
            if (b + 1) % 20 == 0:
                logger.info("bootstrap %d/%d", b + 1, N_BOOT)
        np.savez(cache, spec_=spec_, ref_beta=ref_beta, ref_sub=ref_sub)

    est_boot = np.nanmean(spec_, axis=0)   # bootstrap-mean estimate (the manuscript's 23 uses this)
    se = np.nanstd(spec_, axis=0, ddof=1)  # bootstrap SE
    valid = np.isfinite(se) & (se > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_boot = np.where(valid, est_boot / se, 0.0)
    n_obs = int((np.abs(z_boot) > Z_THRESH).sum())
    n_valid = int(valid.sum())
    indep_exp = 0.05 * n_valid
    binom_sd = float(np.sqrt(n_valid * 0.05 * 0.95))

    # --- rotation null: z ~ N(0, R), R = shrunk correlation of standardized residuals ---
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(valid[None], (spec_ - est_boot) / se, 0.0).reshape(N_BOOT, -1)
    cols = valid.reshape(-1)
    Rz = resid[:, cols]                                   # N_BOOT x n_valid, cols ~ mean0/sd1
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(Rz).covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr = 0.5 * (corr + corr.T)
    L = np.linalg.cholesky(corr + 1e-6 * np.eye(n_valid))
    mc_rng = np.random.default_rng(12345)
    M = 20000
    draws = (mc_rng.standard_normal((M, n_valid)) @ L.T)  # ~ N(0, corr)
    rot_counts = (np.abs(draws) > Z_THRESH).sum(axis=1)
    rot_mean = float(rot_counts.mean())
    rot_sd = float(rot_counts.std(ddof=1))
    rot_p95 = float(np.percentile(rot_counts, 95))
    rot_p99 = float(np.percentile(rot_counts, 99))
    rot_max = int(rot_counts.max())
    n_ge = int((rot_counts >= n_obs).sum())
    emp_p = n_ge / M if n_ge > 0 else float("nan")   # nan flagged as < 1/M below

    # --- secondary: subtype-label permutation (collapses interactions -> ~0) ------------
    perm_rng = np.random.default_rng(7)
    lab_counts = np.empty(N_PERM, dtype=int)
    for r in range(N_PERM):
        lab = perm_rng.permutation(ref_sub)
        est_p = drug_specific(ref_beta, lab, K)
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = np.where(valid, est_p / se, 0.0)
        lab_counts[r] = int((np.abs(zp) > Z_THRESH).sum())

    out = pd.DataFrame([{
        "n_cells": n_valid,
        "observed_nominal": n_obs,
        "independence_expected": indep_exp,
        "independence_sd": binom_sd,
        "fold_excess_vs_independence": n_obs / indep_exp,
        "rotation_null_mean": rot_mean,
        "rotation_null_sd": rot_sd,
        "rotation_null_p95": rot_p95,
        "rotation_null_p99": rot_p99,
        "rotation_null_max": rot_max,
        "variance_inflation_vs_binomial": (rot_sd / binom_sd) ** 2,
        "z_score_vs_rotation": (n_obs - rot_mean) / rot_sd if rot_sd > 0 else np.nan,
        "emp_p_rotation": emp_p if not np.isnan(emp_p) else 1.0 / M,
        "emp_p_rotation_is_upper_bound": bool(np.isnan(emp_p)),
        "labelperm_null_mean": float(lab_counts.mean()),
        "labelperm_null_max": int(lab_counts.max()),
        "n_mc": M,
        "n_perm": N_PERM,
        "n_boot": N_BOOT,
    }])
    out.to_csv(TABLES_DIR / "enrichment_permutation_null.csv", index=False)
    pd.DataFrame({"rotation_count": rot_counts}).to_csv(
        TABLES_DIR / "enrichment_rotation_null_dist.csv", index=False)

    # --- histogram figure (figS11) -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        bins = np.arange(-0.5, max(rot_max, n_obs) + 1.5, 1)
        ax.hist(rot_counts, bins=bins, color="#BDBDBD", edgecolor="white",
                label=f"rotation null (mean {rot_mean:.1f}, SD {rot_sd:.1f})")
        ax.axvline(indep_exp, color="#666666", ls="--", lw=1.5,
                   label=f"independence expectation ({indep_exp:.1f}, SD {binom_sd:.1f})")
        ax.axvline(n_obs, color="#0072B2", lw=2.2, label=f"observed ({n_obs})")  # PATCH_COLOR
        ax.set_xlabel("number of nominally-significant subtype $\\times$ drug cells")
        ax.set_ylabel("Monte-Carlo draws")
        ax.set_title("Enrichment vs a correlation-preserving rotation null")
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"figS11_enrichment_null.{ext}", dpi=200)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure skipped: %s", exc)

    logger.info(
        "ENRICHMENT NULL | observed=%d of %d | independence exp=%.1f (SD %.2f) | "
        "rotation null mean=%.2f SD=%.2f p95=%.0f p99=%.0f max=%d | var-inflation=%.1fx | "
        "z=%.1f | emp_p=%s | label-perm mean=%.2f max=%d",
        n_obs, n_valid, indep_exp, binom_sd, rot_mean, rot_sd, rot_p95, rot_p99, rot_max,
        (rot_sd / binom_sd) ** 2, (n_obs - rot_mean) / rot_sd,
        (f"{emp_p:.5f}" if not np.isnan(emp_p) else f"<{1.0/M:.1e}"),
        float(lab_counts.mean()), int(lab_counts.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
