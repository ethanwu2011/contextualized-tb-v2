"""Leak-free temporal validation: fit ALL preprocessing on the training period only.

Motivation: the canonical context matrix `C` is median-imputed, standardized, and
one-hot encoded once on the FULL cohort (scripts/05b) and then sliced by train/test masks in the
temporal (24) and cross-country (26) experiments — so imputation medians, scaling statistics, and
the categorical vocabulary see the held-out data, in exactly the experiments that argue portability.

This script repeats the temporal hold-out (train registered ≤2020, test 2021+) with preprocessing
fit on the TRAINING period only (imputer, scaler, and one-hot categories learned on train; test
aligned to the train column space), and compares the leak-free held-out-time AUROC to the original
(leaky) number. A small gap means the leak was immaterial; a large gap means the portability claim
must use these corrected numbers. Writes an aggregate table only.

Usage
-----
    uv run python scripts/28_temporal_leakfree.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ctb.config import TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import apply_context_encoder, build_context_matrix, fit_context_encoder
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
SEED = 0


def main() -> int:
    """Run the leak-free temporal hold-out and compare to the leaky baseline."""
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
    y = data["y"].astype(np.float64)

    yr = pc.set_index("condition_id")["registration_date"].dt.year
    year = yr.reindex(cond_ids).to_numpy()
    tr, te = year <= 2020, year >= 2021
    if not tr.any() or not te.any():
        raise ValueError(
            "Temporal split is empty. Check registration_date parsing and the declared year cut."
        )
    logger.info("Temporal split: train=%d, test=%d", int(tr.sum()), int(te.sum()))

    # Leak-free: encoder fit on TRAIN rows of the raw context only, then applied to both.
    enc = fit_context_encoder(craw[tr], spec)
    c_tr = apply_context_encoder(craw[tr], spec, enc).to_numpy(dtype=np.float64)
    c_te = apply_context_encoder(craw[te], spec, enc).to_numpy(dtype=np.float64)
    logger.info("Leak-free encoded context: train %s, test %s", c_tr.shape, c_te.shape)

    m = LinearArchetypeContextualized(c_tr.shape[1], x.shape[1], n_archetypes=K, simplex=True,
                                      seed=SEED)
    train(m, c_tr, x[tr], y[tr], CFG, seed=SEED)
    auc_ctx = roc_auc_score(y[te], m.predict_proba(c_te, x[te]))
    lr = LogisticRegression(max_iter=2000).fit(np.hstack([x[tr], c_tr]), y[tr])
    auc_lr = roc_auc_score(y[te], lr.predict_proba(np.hstack([x[te], c_te]))[:, 1])

    # Original (leaky) numbers for side-by-side, if available.
    leaky = {}
    leaky_path = TABLES_DIR / "temporal_validation.csv"
    if leaky_path.exists():
        prev = pd.read_csv(leaky_path).iloc[0]
        leaky = {"leaky_auroc_contextualized": round(float(prev["auroc_contextualized"]), 3),
                 "leaky_auroc_context_lr": round(float(prev["auroc_context_lr"]), 3)}

    out = {"leakfree_auroc_contextualized": round(float(auc_ctx), 3),
           "leakfree_auroc_context_lr": round(float(auc_lr), 3),
           "n_train": int(tr.sum()), "n_test": int(te.sum()), **leaky}
    pd.DataFrame([out]).to_csv(TABLES_DIR / "temporal_validation_leakfree.csv", index=False)
    logger.info("Leak-free held-out-time AUROC: contextualized=%.3f | context-LR=%.3f  (%s)",
                auc_ctx, auc_lr, leaky)
    return 0


if __name__ == "__main__":
    sys.exit(main())
