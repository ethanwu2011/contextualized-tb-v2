"""Fair, nested-CV head-to-head: tuned contextualized PATCH vs tuned context-LR baseline.

The model ladder compared PATCH to a *default* logistic regression. This is the nested,
tuned version: both models' hyperparameters are tuned on INNER folds and evaluated on held-out OUTER folds,
so neither is advantaged. We report each model's out-of-fold AUROC with a bootstrap 95% CI and a
paired-bootstrap comparison, plus the interpretability (populated-archetype count) of the PATCH config
the nested procedure selected.

The result is reported as obtained, whichever model it favours.

Usage
-----
    uv run python scripts/34_fair_comparison.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctb.config import TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import apply_context_encoder, build_context_matrix, fit_context_encoder
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

SEED = 0
OUTER, INNER = 5, 3
# Small honest PATCH grid for the inner tuning loop (canonical + looser, higher-capacity options).
PATCH_GRID = [
    {"k": 5, "ent": 1.0, "l1": 5e-4},   # canonical (5 interpretable subtypes)
    {"k": 8, "ent": 1.0, "l1": 5e-4},
    {"k": 8, "ent": 0.0, "l1": 0.0},    # loosest / highest-capacity
    {"k": 10, "ent": 0.0, "l1": 0.0},
]


def _fit_patch(c, x, y, cfg_d, seed=SEED):
    cfg = TrainConfig(epochs=1500, lr=0.01, weight_decay=1e-3, archetype_l1=cfg_d["l1"],
                      entropy_weight=cfg_d["ent"])
    m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=cfg_d["k"], simplex=True,
                                      seed=seed)
    train(m, c, x, y, cfg, seed=seed)
    return m


def _boot_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> int:
    """Run the nested-CV fair comparison and write the result table."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    cond = list(data["cond_ids"])
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, spec = build_context_matrix(pc, bx, cond)
    craw = craw.reindex(pd.Index(cond, name="condition_id")).reset_index(drop=True)
    y = data["y"].astype(float)
    oof_lr = np.zeros(len(y))
    oof_pa = np.zeros(len(y))
    chosen = []
    outer = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(x, y)):
        outer_encoder = fit_context_encoder(craw.iloc[tr], spec)
        c_tr = apply_context_encoder(craw.iloc[tr], spec, outer_encoder).to_numpy(dtype=np.float64)
        c_te = apply_context_encoder(craw.iloc[te], spec, outer_encoder).to_numpy(dtype=np.float64)
        # --- LR: tune C on inner folds with fold-fitted context encoders ---
        inner = StratifiedKFold(INNER, shuffle=True, random_state=SEED)
        cs = np.logspace(-3, 1, 9)
        lr_scores = np.zeros(len(cs))
        for ci, c_value in enumerate(cs):
            iaucs = []
            for itr, ite in inner.split(x[tr], y[tr]):
                inner_encoder = fit_context_encoder(craw.iloc[tr[itr]], spec)
                c_itr = apply_context_encoder(
                    craw.iloc[tr[itr]], spec, inner_encoder
                ).to_numpy(dtype=np.float64)
                c_ite = apply_context_encoder(
                    craw.iloc[tr[ite]], spec, inner_encoder
                ).to_numpy(dtype=np.float64)
                lr_inner = LogisticRegression(C=float(c_value), max_iter=2000).fit(
                    np.hstack([x[tr[itr]], c_itr]), y[tr[itr]]
                )
                iaucs.append(
                    roc_auc_score(
                        y[tr[ite]], lr_inner.predict_proba(np.hstack([x[tr[ite]], c_ite]))[:, 1]
                    )
                )
            lr_scores[ci] = float(np.mean(iaucs))
        best_c = float(cs[int(lr_scores.argmax())])
        lr = LogisticRegression(C=best_c, max_iter=2000).fit(np.hstack([x[tr], c_tr]), y[tr])
        oof_lr[te] = lr.predict_proba(np.hstack([x[te], c_te]))[:, 1]
        # --- PATCH: pick config by inner-fold AUROC, refit on outer-train ---
        scores = np.zeros(len(PATCH_GRID))
        for gi, cfg_d in enumerate(PATCH_GRID):
            iaucs = []
            for itr, ite in inner.split(x[tr], y[tr]):
                inner_encoder = fit_context_encoder(craw.iloc[tr[itr]], spec)
                c_itr = apply_context_encoder(
                    craw.iloc[tr[itr]], spec, inner_encoder
                ).to_numpy(dtype=np.float64)
                c_ite = apply_context_encoder(
                    craw.iloc[tr[ite]], spec, inner_encoder
                ).to_numpy(dtype=np.float64)
                m = _fit_patch(c_itr, x[tr[itr]], y[tr[itr]], cfg_d, seed=SEED + fold + gi)
                iaucs.append(
                    roc_auc_score(
                        y[tr[ite]], m.predict_proba(c_ite, x[tr[ite]])
                    )
                )
            scores[gi] = float(np.mean(iaucs))
        best = PATCH_GRID[int(scores.argmax())]
        mbest = _fit_patch(c_tr, x[tr], y[tr], best, seed=SEED + fold)
        oof_pa[te] = mbest.predict_proba(c_te, x[te])
        sub = mbest.predict_loadings(c_tr).argmax(1)
        pop = int((np.bincount(sub, minlength=best["k"]) / len(sub) >= 0.03).sum())
        chosen.append({**best, "populated": pop})
        logger.info("fold %d: LR C=%.3g | PATCH picked K=%d ent=%.1f (pop %d)",
                    fold, best_c, best["k"], best["ent"], pop)

    auc_lr, auc_pa = roc_auc_score(y, oof_lr), roc_auc_score(y, oof_pa)
    lo_lr, hi_lr = _boot_ci(y, oof_lr)
    lo_pa, hi_pa = _boot_ci(y, oof_pa)
    # paired bootstrap: P(PATCH >= LR)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    wins = 0
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        wins += roc_auc_score(y[b], oof_pa[b]) >= roc_auc_score(y[b], oof_lr[b])
    p_ge = wins / 2000

    out = {
        "auroc_patch_tuned": round(auc_pa, 4), "patch_ci": f"{lo_pa:.3f}-{hi_pa:.3f}",
        "auroc_lr_tuned": round(auc_lr, 4), "lr_ci": f"{lo_lr:.3f}-{hi_lr:.3f}",
        "delta_patch_minus_lr": round(auc_pa - auc_lr, 4),
        "P(PATCH>=LR)_paired_boot": round(p_ge, 3),
        "patch_config_modal_K": int(pd.Series([c2["k"] for c2 in chosen]).mode().iloc[0]),
        "patch_populated_mean": round(float(np.mean([c2["populated"] for c2 in chosen])), 1),
    }
    pd.DataFrame([out]).to_csv(TABLES_DIR / "fair_comparison.csv", index=False)
    logger.info("=== FAIR (nested-CV) COMPARISON ===")
    logger.info("PATCH (tuned): %.4f [%.3f-%.3f]", auc_pa, lo_pa, hi_pa)
    logger.info("LR    (tuned): %.4f [%.3f-%.3f]", auc_lr, lo_lr, hi_lr)
    logger.info("delta (PATCH-LR): %+.4f | P(PATCH>=LR) paired-boot = %.3f", auc_pa - auc_lr, p_ge)
    logger.info("nested procedure selected PATCH K(modal)=%d, mean populated=%.1f",
                out["patch_config_modal_K"], out["patch_populated_mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
