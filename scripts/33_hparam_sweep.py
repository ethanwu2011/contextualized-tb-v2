"""Hyperparameter sweep: how much discrimination can the archetype model actually buy?

Maps the accuracy <-> interpretability frontier of the contextualized archetype model on the
primary died/failure outcome. For a grid over K, the entropy load-balancing weight, the archetype
L1, and the learning rate, computes 5-fold out-of-fold AUROC and the number of populated
archetypes (>=3%), and compares to the concatenated-context logistic baseline.

The point is to report the trade-off directly. The sweep checks whether the canonical config was
under-tuned and whether a looser, less-interpretable config can reach or exceed the baseline.

Usage
-----
    uv run python scripts/33_hparam_sweep.py
"""

from __future__ import annotations

import logging
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctb.config import TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

SEED = 0
N_SPLITS = 5
# Focused grid. The canonical config is K=5, entropy=1.0, l1=5e-4, and lr=0.01.
KS = [5, 6, 8, 10]
ENTROPY = [0.0, 1.0]
L1 = [0.0, 5e-4]
LR = [0.01, 0.02]


def _oof(
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    ent: float,
    l1: float,
    lr: float,
) -> tuple[float, float]:
    """Return (5-fold OOF AUROC, mean populated-archetype count) for one config."""
    cfg = TrainConfig(epochs=1500, lr=lr, weight_decay=1e-3, archetype_l1=l1, entropy_weight=ent)
    oof = np.zeros(len(y))
    pops = []
    for fold, (tr, te, c_tr, c_te) in enumerate(splits):
        fold_seed = SEED + fold
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=fold_seed
        )
        train(m, c_tr, x[tr], y[tr], cfg, seed=fold_seed)
        oof[te] = m.predict_proba(c_te, x[te])
        sub = m.predict_loadings(c_tr).argmax(1)
        pops.append(int((np.bincount(sub, minlength=k) / len(sub) >= 0.03).sum()))
    return roc_auc_score(y, oof), float(np.mean(pops))


def main() -> int:
    """Run the sweep and write the accuracy-vs-interpretability frontier table."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    y = data["y"].astype(float)
    raw_context, spec = load_primary_raw_context(list(data["cond_ids"]))
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    split_cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in split_cv.split(x, y):
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        splits.append((tr, te, c_tr, c_te))

    # Baseline: concatenated-context logistic regression (the bar PATCH matches).
    oof_lr = np.zeros(len(y))
    for tr, te, c_tr, c_te in splits:
        lr_m = LogisticRegression(max_iter=2000).fit(np.hstack([x[tr], c_tr]), y[tr])
        oof_lr[te] = lr_m.predict_proba(np.hstack([x[te], c_te]))[:, 1]
    base = roc_auc_score(y, oof_lr)
    logger.info("Concatenated-context LR baseline OOF AUROC = %.4f", base)

    rows = []
    for k, ent, l1, lr in product(KS, ENTROPY, L1, LR):
        auc, pop = _oof(splits, x, y, k, ent, l1, lr)
        canonical = (k == 5 and ent == 1.0 and l1 == 5e-4 and lr == 0.01)
        rows.append({"K": k, "entropy": ent, "l1": l1, "lr": lr, "oof_auroc": round(auc, 4),
                     "populated": round(pop, 1), "vs_baseline": round(auc - base, 4),
                     "canonical": canonical})
        logger.info("K=%2d ent=%.1f l1=%.0e lr=%.2f -> AUROC %.4f (pop %.1f) %s",
                    k, ent, l1, lr, auc, pop, "<= CANONICAL" if canonical else "")

    df = pd.DataFrame(rows).sort_values("oof_auroc", ascending=False).reset_index(drop=True)
    df.attrs["baseline"] = base
    df.to_csv(TABLES_DIR / "hparam_sweep.csv", index=False)
    best = df.iloc[0]
    canon = df[df["canonical"]].iloc[0]
    logger.info("=== FRONTIER ===")
    logger.info("baseline (concat LR): %.4f", base)
    logger.info(
        "best config: K=%d ent=%.1f l1=%.0e lr=%.2f -> AUROC %.4f (pop %.1f), %+.4f vs baseline",
        best["K"],
        best["entropy"],
        best["l1"],
        best["lr"],
        best["oof_auroc"],
        best["populated"],
        best["vs_baseline"],
    )
    logger.info("canonical (K=5): AUROC %.4f (pop %.1f), %+.4f vs baseline",
                canon["oof_auroc"], canon["populated"], canon["vs_baseline"])
    logger.info("# configs beating baseline: %d / %d", int((df["vs_baseline"] > 0).sum()), len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
