"""Phase 2 (model-first): fit the contextualized classifier and compare to the baseline floor.

For each outcome (mortality and treatment intensification), this:

1. Reuses the four baselines (LR/GBDT × with/without context) to establish the concatenation floor.
2. Fits the **contextualized** classifier (deep context encoder → per-patient logistic regression)
   with the same 5-fold stratified CV and seed.
3. Reports AUROC, AUPRC, Brier, and class-specific accuracy at the Youden threshold (the ML4H Table
   1 view), saving a comparison table and out-of-fold predictions.

This script intentionally stops **before** generating the treatment-effect heatmaps, so the
modelling result can be reviewed first.

Usage
-----
    uv run python scripts/04_train_contextualized.py
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import METRICS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.context import FeatureSpec
from ctb.models.baselines import run_all_baselines
from ctb.models.contextualized import ContextualizedConfig, run_contextualized_cv
from ctb.models.evaluation import compute_metrics
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

OUTCOMES = ["mortality", "treatment_intensification"]
# Tuned: K=8 archetypes (ties unconstrained MLP on AUROC, beats floor, yields patient subtypes).
CONFIG = ContextualizedConfig(
    encoder_type="mlp",
    n_bootstraps=5,
    max_epochs=150,
    learning_rate=1e-3,
    num_archetypes=8,
    es_patience=10,
)


def _summary_row(
    model: str, outcome: str, y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, object]:
    """Compute a compact metric row for one (model, outcome).

    Parameters
    ----------
    model : str
        Model name.
    outcome : str
        Outcome name.
    y_true, y_score : numpy.ndarray
        Labels and predicted probabilities.

    Returns
    -------
    dict
        Row with AUROC/AUPRC (+CIs), Brier, and class-specific accuracy.
    """
    m = compute_metrics(y_true, y_score)
    return {
        "outcome": outcome,
        "model": model,
        "auroc": m["auroc"],
        "auroc_lo": m["auroc_ci"][0],
        "auroc_hi": m["auroc_ci"][1],
        "auprc": m["auprc"],
        "brier": m["brier"],
        "acc_positive": m["sensitivity"],
        "acc_negative": m["specificity"],
        "prevalence": m["prevalence"],
    }


def main() -> int:
    """Fit baselines + contextualized model for each outcome and save comparison metrics.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    analysis = pd.read_parquet(PROCESSED_DIR / "analysis_frame.parquet")
    spec_payload = json.loads((PROCESSED_DIR / "feature_spec.json").read_text())
    treatment_cols = spec_payload["treatment_cols"]
    spec = FeatureSpec(numeric=spec_payload["numeric"], categorical=spec_payload["categorical"])

    X = analysis[treatment_cols]
    C = analysis[spec.all_columns]

    rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []

    for outcome in OUTCOMES:
        y = analysis[outcome].astype(int)
        logger.info("=== Outcome: %s (prevalence %.3f) ===", outcome, y.mean())

        # 1. Baseline floor (concatenation).
        baselines = run_all_baselines(X, C, spec, y)
        for bname, res in baselines.items():
            rows.append(_summary_row(bname, outcome, res.y_true, res.y_score))
            oof_frames.append(
                pd.DataFrame(
                    {
                        "outcome": outcome,
                        "model": bname,
                        "y_true": res.y_true,
                        "y_score": res.y_score,
                    }
                )
            )

        # 2. Contextualized model.
        yt, ys = run_contextualized_cv(X, C, spec, y, name=f"Contextualized[{outcome}]", cfg=CONFIG)
        rows.append(_summary_row("Contextualized", outcome, yt, ys))
        oof_frames.append(
            pd.DataFrame(
                {"outcome": outcome, "model": "Contextualized", "y_true": yt, "y_score": ys}
            )
        )

        ctx = rows[-1]
        floor = next(
            r for r in rows if r["outcome"] == outcome and r["model"] == "GBDT-with-context"
        )
        logger.info(
            "%s | Contextualized AUROC=%.3f vs GBDT+ctx floor %.3f (Δ=%+.3f)",
            outcome,
            ctx["auroc"],
            floor["auroc"],
            ctx["auroc"] - floor["auroc"],
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "contextualized_comparison.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(
        PROCESSED_DIR / "oof_predictions_contextualized.parquet"
    )
    for outcome in OUTCOMES:
        sub = summary[summary["outcome"] == outcome]
        (METRICS_DIR / f"contextualized_{outcome}.json").write_text(
            sub.to_json(orient="records", indent=2)
        )

    logger.info("=== COMPARISON (AUROC) ===")
    pivot = summary.pivot(index="model", columns="outcome", values="auroc")
    logger.info("\n%s", pivot.round(3).to_string())
    logger.info("Saved comparison to %s", TABLES_DIR / "contextualized_comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
