"""Calibration and AUPRC for the unfavorable-outcome model.

The main figures report AUROC only; at 16.3% prevalence a clinical-map paper should also show
calibration and precision-recall. This computes 5-fold out-of-fold predictions for the
contextualized PATCH model and the concatenated-context logistic baseline on the died/failure
outcome, and writes calibration bins, PR points, AUPRC, and Brier score. Writes aggregate curves
only with no per-patient rows.

Usage
-----
    uv run python scripts/30_calibration_auprc.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from ctb.config import TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
SEED = 0
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)


def main() -> int:
    """Compute OOF calibration + PR curves for PATCH and the context-LR baseline."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    raw_context, spec = load_primary_raw_context(list(data["cond_ids"]))

    oof_patch = np.zeros(len(y))
    oof_lr = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(x, y)):
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        fold_seed = SEED + fold
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=fold_seed
        )
        train(m, c_tr, x[tr], y[tr].astype(np.float64), CFG, seed=fold_seed)
        oof_patch[te] = m.predict_proba(c_te, x[te])
        lr = LogisticRegression(max_iter=2000).fit(np.hstack([x[tr], c_tr]), y[tr])
        oof_lr[te] = lr.predict_proba(np.hstack([x[te], c_te]))[:, 1]
        logger.info("fold %d done", fold)

    rows = []
    for name, p in [("PATCH", oof_patch), ("context_LR", oof_lr)]:
        ap = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        prec, rec, _ = precision_recall_curve(y, p)
        # store calibration bins
        for mp, fp in zip(mean_pred, frac_pos, strict=True):
            rows.append({"model": name, "kind": "calibration", "x": round(float(mp), 4),
                         "y": round(float(fp), 4), "auprc": round(float(ap), 4),
                         "brier": round(float(brier), 4)})
        # store a downsampled PR curve (<=100 points)
        idx = np.linspace(0, len(rec) - 1, min(100, len(rec))).astype(int)
        for ri, pi in zip(rec[idx], prec[idx], strict=True):
            rows.append({"model": name, "kind": "pr", "x": round(float(ri), 4),
                         "y": round(float(pi), 4), "auprc": round(float(ap), 4),
                         "brier": round(float(brier), 4)})
        logger.info("%s: AUPRC=%.3f  Brier=%.3f  (prevalence %.3f)", name, ap, brier, y.mean())

    pd.DataFrame(rows).to_csv(TABLES_DIR / "calibration_auprc.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
