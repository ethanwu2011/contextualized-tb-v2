"""Temporal (out-of-time) validation: train on patients registered <=2020, test on 2021+.

This evaluates held-out discrimination for the died-or-failure endpoint, checks whether the
multimodal subtypes replicate when the trained model is applied to the unseen future cohort, and
records the resulting train and test subtype profiles.

Usage
-----
    uv run python scripts/24_temporal_validation.py
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
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
SEED = 0


def _profile(craw: pd.DataFrame, y: np.ndarray, sub: np.ndarray, k: int) -> pd.DataFrame:
    """Subtype profile across modalities."""
    rows = []
    for s in range(k):
        m = sub == s
        if m.sum() == 0:
            continue
        rows.append({
            "subtype": f"S{s}", "n": int(m.sum()), "unfav%": round(100 * y[m].mean(), 1),
            "MDR%": round(100 * (craw["type_of_resistance"][m] == "MDR non XDR").mean(), 0),
            "XDR%": round(100 * (craw["type_of_resistance"][m] == "XDR").mean(), 0),
            "sens%": round(100 * (craw["type_of_resistance"][m] == "Sensitive").mean(), 0),
            "HIV%": round(100 * (craw["comorb_hiv"][m] == 1).mean(), 0),
            "smear+%": round(100 * (craw["baseline_smear"][m] == "positive").mean(), 0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    """Run the temporal hold-out validation."""
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
    train_mask = year <= 2020
    test_mask = year >= 2021
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "Temporal split is empty. Check registration_date parsing and the declared year cut."
        )
    encoder = fit_context_encoder(craw[train_mask], spec)
    c = apply_context_encoder(craw, spec, encoder).to_numpy(dtype=np.float64)
    logger.info("Temporal split: train(<=2020)=%d, test(2021+)=%d, unfav train=%.1f%% test=%.1f%%",
                int(train_mask.sum()), int(test_mask.sum()),
                100 * y[train_mask].mean(), 100 * y[test_mask].mean())

    # (1) held-out-time AUROC: contextualized vs context-concatenation baseline
    m = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED
    )
    train(m, c[train_mask], x[train_mask], y[train_mask], CFG, seed=SEED)
    auc_ctx = roc_auc_score(y[test_mask], m.predict_proba(c[test_mask], x[test_mask]))
    lr = LogisticRegression(max_iter=2000).fit(
        np.hstack([x, c])[train_mask], y[train_mask])
    auc_lr = roc_auc_score(
        y[test_mask], lr.predict_proba(np.hstack([x, c])[test_mask])[:, 1])
    logger.info("Held-out-time AUROC: contextualized=%.3f | context-concat LR=%.3f",
                auc_ctx, auc_lr)

    # (2) subtype replication: profile train vs applied-to-test subtypes
    sub_tr = m.predict_loadings(c[train_mask]).argmax(axis=1)
    sub_te = m.predict_loadings(c[test_mask]).argmax(axis=1)
    p_tr = _profile(craw[train_mask].reset_index(drop=True), y[train_mask], sub_tr, K)
    p_te = _profile(craw[test_mask].reset_index(drop=True), y[test_mask], sub_te, K)
    p_tr.to_csv(TABLES_DIR / "temporal_subtypes_train.csv", index=False)
    p_te.to_csv(TABLES_DIR / "temporal_subtypes_test.csv", index=False)
    logger.info("TRAIN (<=2020) subtypes:\n%s", p_tr.to_string(index=False))
    logger.info("TEST (2021+) subtypes (same model applied):\n%s", p_te.to_string(index=False))
    pd.DataFrame([{"auroc_contextualized": auc_ctx, "auroc_context_lr": auc_lr,
                   "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}]).to_csv(
        TABLES_DIR / "temporal_validation.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
