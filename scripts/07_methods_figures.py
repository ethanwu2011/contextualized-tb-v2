"""Methods-headline figures: the model ladder (AUC with p-values) + CI-annotated heterogeneity.

Produces the two figures that carry the paper's *methods* thesis:

* **Model ladder** — out-of-fold AUROC climbing from a population model -> +context -> +archetypes,
  with bootstrap 95% CIs and paired-bootstrap significance vs the previous rung. This *is* the
  headline claim: contextualization finds structure population models miss.
* **Heterogeneous effects with bootstrap significance** — the β(C)-by-context heatmap with ``*``
  marking effects whose 95% CI excludes zero.

Usage
-----
    uv run python scripts/07_methods_figures.py
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

from ctb.config import METRICS_DIR, PROCESSED_DIR, SEED, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import FeatureSpec, apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    bootstrap_context_effects,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_archetype_k_sweep,
    figure_heterogeneous_effects_with_ci,
    figure_model_ladder,
)

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


def _oof_ctx(
    x: NDArray[np.float64],
    raw_context: pd.DataFrame,
    spec: FeatureSpec,
    y: NDArray[np.int_],
    cv: StratifiedKFold,
    k: int,
) -> NDArray[np.float64]:
    """Out-of-fold contextualized-model probabilities (K archetypes, simplex)."""
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(cv.split(x, y)):
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        fold_seed = SEED + fold
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=fold_seed
        )
        # Tuned config (lower lr, longer training) — the K-sweep winner.
        train(
            m,
            c_tr,
            x[tr],
            y[tr],
            TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                        entropy_weight=1.0),
            seed=fold_seed,
        )
        oof[te] = m.predict_proba(c_te, x[te])
    return oof


def _oof_context_lr(
    x: NDArray[np.float64],
    raw_context: pd.DataFrame,
    spec: FeatureSpec,
    y: NDArray[np.int_],
    cv: StratifiedKFold,
    include_treatments: bool,
) -> NDArray[np.float64]:
    """Return OOF logistic predictions with a fold-fitted context encoder."""
    oof = np.zeros(len(y))
    for tr, te in cv.split(x, y):
        encoder = fit_context_encoder(raw_context.iloc[tr], spec)
        c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(np.float64)
        c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(np.float64)
        train_features = np.hstack([x[tr], c_tr]) if include_treatments else c_tr
        test_features = np.hstack([x[te], c_te]) if include_treatments else c_te
        lr = LogisticRegression(max_iter=2000).fit(train_features, y[tr])
        oof[te] = lr.predict_proba(test_features)[:, 1]
    return oof


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
    """Generate the methods-headline figures.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x, y = data["X"], data["y"]
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = list(data["cond_ids"])
    raw_context, spec = load_primary_raw_context(cond_ids)
    meta = pd.read_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet").loc[cond_ids]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # --- model ladder ---
    logger.info("Computing model-ladder OOF predictions...")
    preds = {
        "Population (treatments only)": _oof_lr(x, y, cv),
        "+ Context (concatenated)": _oof_context_lr(x, raw_context, spec, y, cv, True),
        "+ Contextualized archetypes": _oof_ctx(x, raw_context, spec, y, cv, k=5),
    }
    names = list(preds)
    rows = []
    prev: str | None = None
    for nm in names:
        auc, lo, hi = _auc_ci(y, preds[nm])
        p = _paired_p(y, preds[nm], preds[prev]) if prev else np.nan
        rows.append({"model": nm, "auc": auc, "ci_low": lo, "ci_high": hi, "p_vs_prev": p})
        prev = nm
    ladder = pd.DataFrame(rows)
    ladder.to_csv(TABLES_DIR / "model_ladder.csv", index=False)
    figure_model_ladder(ladder)
    logger.info("Model ladder:\n%s", ladder.round(4).to_string(index=False))

    # --- archetype-K sweep (value-of-archetypes curve, K=3..7) ---
    logger.info("Computing archetype-K sweep...")
    k_rows = []
    for k in range(3, 8):
        s = _oof_ctx(x, raw_context, spec, y, cv, k=k)
        auc, lo, hi = _auc_ci(y, s)
        k_rows.append({"k": k, "auc": auc, "ci_low": lo, "ci_high": hi})
    k_sweep = pd.DataFrame(k_rows)
    k_sweep.to_csv(TABLES_DIR / "archetype_k_sweep.csv", index=False)
    figure_archetype_k_sweep(k_sweep)
    logger.info("K sweep:\n%s", k_sweep.round(4).to_string(index=False))

    # --- heterogeneous effects with bootstrap significance ---
    logger.info("Bootstrapping heterogeneous-effect significance...")
    masks = {
        "Sensitive": (meta["type_of_resistance"] == "Sensitive").to_numpy(),
        "MDR": (meta["type_of_resistance"] == "MDR non XDR").to_numpy(),
        "XDR": (meta["type_of_resistance"] == "XDR").to_numpy(),
        "HIV+": (meta["comorb_hiv"] == 1).to_numpy(),
        "smear+": (meta["baseline_smear"] == "positive").to_numpy(),
    }
    full_encoder = fit_context_encoder(raw_context, spec)
    c_full = apply_context_encoder(raw_context, spec, full_encoder).to_numpy(np.float64)
    sig = bootstrap_context_effects(
        c_full,
        x.astype(np.float64),
        y.astype(np.float64),
        masks,
        drugs,
        n_archetypes=5,
        n_boot=25,
        epochs=600,
    )
    sig.to_csv(TABLES_DIR / "heterogeneous_effects_significance.csv", index=False)
    figure_heterogeneous_effects_with_ci(sig)

    n_sig = int(sig["significant"].sum())
    (METRICS_DIR / "methods_summary.json").write_text(
        pd.Series(
            {
                "ladder_auc": ladder.set_index("model")["auc"].round(4).to_dict(),
                "ladder_p_vs_prev": ladder.set_index("model")["p_vs_prev"].round(4).to_dict(),
                "n_significant_heterogeneous_effects": n_sig,
            }
        ).to_json(indent=2)
    )
    logger.info("Methods figures written. %d significant heterogeneous effects.", n_sig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
