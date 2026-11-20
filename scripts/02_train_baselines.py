"""Train the four baseline models with 5-fold CV and save metrics + out-of-fold predictions.

Reads ``data/processed/analysis_frame.parquet`` and ``feature_spec.json`` (written by
``01_build_dataset.py``). Writes:

* ``results/metrics/baseline_<name>.json`` — full metric suite per model.
* ``results/tables/baseline_metrics_summary.csv`` — compact comparison table.
* ``data/processed/oof_predictions.parquet`` — out-of-fold y_true/y_score per model (for figures).

Usage
-----
    uv run python scripts/02_train_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from ctb.config import METRICS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.context import FeatureSpec
from ctb.models.baselines import run_all_baselines
from ctb.models.evaluation import compute_metrics
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


def main() -> int:
    """Train all baselines and persist metrics.

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
    outcome_col = spec_payload["outcome_col"]
    spec = FeatureSpec(numeric=spec_payload["numeric"], categorical=spec_payload["categorical"])

    X = analysis[treatment_cols]
    C = analysis[spec.all_columns]
    y = analysis[outcome_col].astype(int)

    results = run_all_baselines(X, C, spec, y)

    summary_rows = []
    oof_frames = []
    for name, res in results.items():
        metrics = compute_metrics(res.y_true, res.y_score)
        metrics["model"] = name
        (METRICS_DIR / f"baseline_{name}.json").write_text(json.dumps(metrics, indent=2))
        summary_rows.append(
            {
                "model": name,
                "auroc": metrics["auroc"],
                "auroc_lo": metrics["auroc_ci"][0],
                "auroc_hi": metrics["auroc_ci"][1],
                "auprc": metrics["auprc"],
                "auprc_lo": metrics["auprc_ci"][0],
                "auprc_hi": metrics["auprc_ci"][1],
                "brier": metrics["brier"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
        oof_frames.append(
            pd.DataFrame({"model": name, "y_true": res.y_true, "y_score": res.y_score})
        )
        logger.info(
            "%-20s AUROC=%.3f [%.3f-%.3f]  AUPRC=%.3f  Brier=%.3f",
            name,
            metrics["auroc"],
            metrics["auroc_ci"][0],
            metrics["auroc_ci"][1],
            metrics["auprc"],
            metrics["brier"],
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "baseline_metrics_summary.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(PROCESSED_DIR / "oof_predictions.parquet")
    logger.info("Saved metrics for %d models to %s", len(results), METRICS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
