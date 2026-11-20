"""Subtype stability via bootstrap consensus.

Refits the K=5 contextualized model on many patient bootstrap resamples, matches each solution's
archetype labels back to the reference (Hungarian assignment on patient overlap), and measures how
consistently patients land in the same subtype. Reports an overall agreement (adjusted Rand index)
and a per-subtype retention/consensus score — the formal "are the subtypes reproducible" evidence.

Usage
-----
    uv run python scripts/18_subtype_stability.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ctb.config import FIGURES_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    select_n_archetypes,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.style import PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt  # noqa: E402

N_BOOT = 50
EPOCHS = 2000
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
BASE_SEED = 100


def _match(ref: NDArray[np.int_], other: NDArray[np.int_], k: int) -> dict[int, int]:
    """Map ``other`` labels to ``ref`` labels by maximum patient overlap."""
    overlap = np.zeros((k, k))
    for i in range(k):
        ri = ref == i
        for j in range(k):
            overlap[i, j] = int(np.logical_and(ri, other == j).sum())
    r_idx, o_idx = linear_sum_assignment(-overlap)
    return {int(o): int(r) for r, o in zip(r_idx, o_idx, strict=True)}


def main() -> int:
    """Compute subtype stability and write the table + figure."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    y = data["y"].astype(np.float64)

    k, seed = select_n_archetypes(
        c, x, y, drug_names=drugs, k_range=range(8, 3, -1), min_subtype_frac=0.03, epochs=EPOCHS
    )
    ref_m = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=seed
    )
    train(ref_m, c, x, y, CFG, seed=seed)
    ref = ref_m.predict_loadings(c).argmax(axis=1)
    logger.info("Reference K=%d seed=%d sizes=%s", k, seed, np.bincount(ref, minlength=k).tolist())

    n = c.shape[0]
    rng = np.random.default_rng(BASE_SEED)
    same = np.zeros(n)  # fraction of bootstraps where patient stays in its reference subtype
    aris = []
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=BASE_SEED + b
        )
        train(m, c[idx], x[idx], y[idx], CFG, seed=BASE_SEED + b)
        sub = m.predict_loadings(c).argmax(axis=1)
        mapped = np.array([_match(ref, sub, k).get(int(s_), -1) for s_ in sub])
        same += (mapped == ref).astype(float)
        aris.append(adjusted_rand_score(ref, sub))
        logger.info("Stability bootstrap %d/%d (ARI=%.3f)", b + 1, N_BOOT, aris[-1])
    same /= N_BOOT

    rows = []
    for s in range(k):
        msk = ref == s
        rows.append({"subtype": f"S{s}", "n": int(msk.sum()), "consensus": float(same[msk].mean())})
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES_DIR / "subtype_stability.csv", index=False)
    mean_ari = float(np.mean(aris))
    logger.info("Mean ARI vs reference = %.3f; per-subtype consensus:\n%s",
                mean_ari, tab.to_string(index=False))

    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    t = tab.sort_values("consensus")
    ax.barh(t["subtype"], t["consensus"], color=PALETTE[2])
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r["consensus"], i, f"  {r['consensus']:.2f} (n={int(r['n'])})",
                va="center", fontsize=7)
    ax.set(xlim=(0, 1), xlabel="Bootstrap consensus (same-subtype fraction)", ylabel="Subtype",
           title=f"Subtype stability (mean ARI vs reference = {mean_ari:.2f})")
    save_figure(fig, FIGURES_DIR / "subtype_stability")
    return 0


if __name__ == "__main__":
    sys.exit(main())
