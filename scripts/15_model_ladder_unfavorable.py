"""Model ladder on the primary outcome (unfavorable treatment response = died OR failure).

The "value of context" figure: out-of-fold AUROC climbing from a population model -> + context ->
+ contextualized archetypes, with bootstrap 95% CIs and paired-bootstrap significance vs the
previous rung. Establishes that contextualization captures structure a one-size-fits-all model
misses on the died/failure endpoint.

Usage
-----
    uv run python scripts/15_model_ladder_unfavorable.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctb.config import METRICS_DIR, SEED, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import FeatureSpec, apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_model_ladder

logger = logging.getLogger(__name__)


def _oof_lr(
    feats: NDArray[np.float64], y: NDArray[np.int_], cv: StratifiedKFold
) -> NDArray[np.float64]:
    """Out-of-fold logistic-regression probabilities."""
    oof = np.zeros(len(y))
    for tr, te in cv.split(feats, y):
        lr = LogisticRegression(max_iter=2000).fit(feats[tr], y[tr])
        oof[te] = lr.predict_proba(feats[te])[:, 1]
    return oof


def _oof_context_models(
    x: NDArray[np.float64],
    raw_context: pd.DataFrame,
    spec: FeatureSpec,
    y: NDArray[np.int_],
    cv: StratifiedKFold,
    k: int,
    entropy: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return context, concatenated, and PATCH OOF predictions with fold-fitted encoders."""
    oof_context = np.zeros(len(y))
    oof_concat = np.zeros(len(y))
    oof_patch = np.zeros(len(y))
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        fold_seed = SEED + fold
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        context_lr = LogisticRegression(max_iter=2000).fit(c_tr, y[tr])
        oof_context[te] = context_lr.predict_proba(c_te)[:, 1]
        concat_lr = LogisticRegression(max_iter=2000).fit(
            np.hstack([x[tr], c_tr]), y[tr]
        )
        oof_concat[te] = concat_lr.predict_proba(np.hstack([x[te], c_te]))[:, 1]
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=fold_seed
        )
        cfg = TrainConfig(
            epochs=2000,
            lr=0.01,
            weight_decay=1e-3,
            archetype_l1=5e-4,
            entropy_weight=entropy,
        )
        train(m, c_tr, x[tr], y[tr], cfg, seed=fold_seed)
        oof_patch[te] = m.predict_proba(c_te, x[te])
    return oof_context, oof_concat, oof_patch


def _auc_ci(
    y: NDArray[np.int_], s: NDArray[np.float64], n_boot: int = 1000
) -> tuple[float, float, float]:
    """AUROC with percentile bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _paired_p(
    y: NDArray[np.int_], sa: NDArray[np.float64], sb: NDArray[np.float64], n_boot: int = 2000
) -> float:
    """Two-sided paired bootstrap p-value for AUROC(sa) - AUROC(sb)."""
    rng = np.random.default_rng(0)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    diffs_arr = np.asarray(diffs)
    return float(2 * min((diffs_arr <= 0).mean(), (diffs_arr >= 0).mean()))


def main() -> int:
    """Generate the model-ladder figure for the died/failure outcome.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    cond_ids = list(data["cond_ids"])
    y = data["y"].astype(int)
    logger.info("Outcome = died OR failure: %.1f%% positive", 100 * y.mean())

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    raw_context, spec = load_primary_raw_context(cond_ids)
    context_oof, concat_oof, patch_oof = _oof_context_models(x, raw_context, spec, y, cv, k=5)
    preds = {
        "Population (treatments only)": _oof_lr(x, y, cv),
        "Context only (prognosis)": context_oof,
        "+ Context (concatenated)": concat_oof,
        "+ Contextualized archetypes": patch_oof,
    }
    rows = []
    prev: str | None = None
    for nm in preds:
        auc, lo, hi = _auc_ci(y, preds[nm])
        p = _paired_p(y, preds[nm], preds[prev]) if prev else np.nan
        rows.append({"model": nm, "auc": auc, "ci_low": lo, "ci_high": hi, "p_vs_prev": p})
        prev = nm
    ladder = pd.DataFrame(rows)
    ladder.to_csv(TABLES_DIR / "model_ladder_unfavorable.csv", index=False)
    figure_model_ladder(ladder, name="model_ladder_unfavorable")
    (METRICS_DIR / "model_ladder_unfavorable.json").write_text(
        ladder.set_index("model")[["auc", "p_vs_prev"]].round(4).to_json(indent=2)
    )
    logger.info("Model ladder (died/failure):\n%s", ladder.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
