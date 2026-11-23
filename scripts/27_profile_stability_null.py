"""Matched null for the cross-seed profile-stability claim.

The manuscript reports that, although the hard subtype partition is only moderately stable
(cross-seed ARI 0.52-0.58), the five archetype PROFILES recur across random initialisations once
matched by Hungarian assignment. The question is whether that recurrence is genuine or a best-case
artifact of optimal matching: Hungarian assignment always finds *some* pairing, so profile agreement
under matching needs a null.

This script refits K=5 on six seeds, forms each seed's five subtype drug-effect profiles
(mean beta(c) over the patients assigned to each subtype), Hungarian-matches every seed to the
reference (seed 0) by profile cosine similarity, and contrasts the matched agreement against two
nulls:
  (a) RANDOM CORRESPONDENCE -- the agreement expected if a reference profile were paired with a
      randomly chosen (non-matched) profile from another seed; and
  (b) the full permutation distribution of the total match similarity over all 5! = 120 possible
      bijections, to check whether the Hungarian pairing is the UNIQUE optimum (a proxy for the
      correspondence being identified rather than interchangeable).
It also reports how tightly matched subtype failure rates reproduce versus the random-correspondence
baseline.

DUA-safe: reads fitted matrices; writes only aggregate agreement statistics + a figure.

Usage
-----
    .venv/bin/python scripts/27_profile_stability_null.py
"""

from __future__ import annotations

import itertools
import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ctb.config import FIGURES_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
N_SEEDS = 6
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)


def _profiles(beta: NDArray[np.float64], sub: NDArray[np.int_], y: NDArray[np.float64], k: int):
    """Per-subtype drug-effect profile (mean beta), failure rate, and size."""
    prof = np.full((k, beta.shape[1]), np.nan)
    fail = np.full(k, np.nan)
    size = np.zeros(k, dtype=int)
    for s in range(k):
        sel = sub == s
        size[s] = int(sel.sum())
        if size[s]:
            prof[s] = beta[sel].mean(axis=0)
            fail[s] = float(y[sel].mean())
    return prof, fail, size


def _cos(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> int:
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)

    profs, fails = [], []
    for seed in range(N_SEEDS):
        m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=seed)
        train(m, c, x, y, CFG, seed=seed)
        beta, _ = m.predict_beta(c)
        sub = m.predict_loadings(c).argmax(axis=1)
        p, f, sz = _profiles(beta, sub, y, K)
        profs.append(p)
        fails.append(f)
        logger.info("seed %d sizes=%s", seed, sz.tolist())

    ref_p, ref_f = profs[0], fails[0]
    perms = list(itertools.permutations(range(K)))

    matched_cos, matched_dfail = [], []       # Hungarian-matched
    random_cos, random_dfail = [], []         # off-diagonal (random correspondence)
    unique_optimum = 0
    n_pairs = 0
    for s in range(1, N_SEEDS):
        P, F = profs[s], fails[s]
        # cosine similarity matrix (ref rows x seed-s cols)
        S = np.array([[_cos(ref_p[i], P[j]) for j in range(K)] for i in range(K)])
        r_idx, c_idx = linear_sum_assignment(-S)          # Hungarian: maximise total cosine
        mapping = {int(i): int(j) for i, j in zip(r_idx, c_idx, strict=True)}
        for i in range(K):
            j = mapping[i]
            matched_cos.append(S[i, j])
            matched_dfail.append(abs(ref_f[i] - F[j]))
            for jj in range(K):
                if jj != j:
                    random_cos.append(S[i, jj])
                    random_dfail.append(abs(ref_f[i] - F[jj]))
        # is the Hungarian pairing the unique best among all 120 bijections?
        totals = np.array([sum(S[i, p[i]] for i in range(K)) for p in perms])
        best = totals.max()
        unique_optimum += int((totals >= best - 1e-9).sum() == 1)
        n_pairs += 1

    matched_cos = np.array(matched_cos); random_cos = np.array(random_cos)
    matched_dfail = np.array(matched_dfail); random_dfail = np.array(random_dfail)

    out = pd.DataFrame([{
        "n_seeds": N_SEEDS,
        "matched_cos_mean": float(matched_cos.mean()),
        "matched_cos_min": float(matched_cos.min()),
        "random_cos_mean": float(random_cos.mean()),
        "correspondence_gap": float(matched_cos.mean() - random_cos.mean()),
        "matched_dfail_mean": float(matched_dfail.mean()),
        "matched_dfail_max": float(matched_dfail.max()),
        "random_dfail_mean": float(random_dfail.mean()),
        "pairs_with_unique_hungarian_optimum": unique_optimum,
        "n_pairs": n_pairs,
    }])
    out.to_csv(TABLES_DIR / "profile_stability_null.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
        ax[0].hist(random_cos, bins=20, color="#BDBDBD", alpha=0.9, label="random correspondence")
        ax[0].hist(matched_cos, bins=20, color="#0072B2", alpha=0.9, label="Hungarian-matched")  # PATCH_COLOR
        ax[0].set_xlabel("cross-seed profile cosine similarity"); ax[0].set_ylabel("count")
        ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Profile correspondence vs null")
        ax[1].hist(random_dfail, bins=20, color="#BDBDBD", alpha=0.9, label="random")
        ax[1].hist(matched_dfail, bins=20, color="#0072B2", alpha=0.9, label="matched")
        ax[1].set_xlabel("|Δ subtype failure rate| across seeds"); ax[1].set_ylabel("count")
        ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Failure-rate reproduction")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"figS12_profile_stability_null.{ext}", dpi=200)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure skipped: %s", exc)

    logger.info(
        "PROFILE STABILITY NULL | matched cos mean=%.3f (min %.3f) vs random %.3f (gap %.3f) | "
        "matched dfail mean=%.3f max=%.3f vs random %.3f | unique Hungarian optimum in %d/%d pairs",
        matched_cos.mean(), matched_cos.min(), random_cos.mean(),
        matched_cos.mean() - random_cos.mean(), matched_dfail.mean(), matched_dfail.max(),
        random_dfail.mean(), unique_optimum, n_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
