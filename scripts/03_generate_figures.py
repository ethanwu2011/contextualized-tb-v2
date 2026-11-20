"""Regenerate all Phase 1 figures from saved processed data and results.

Reads the processed artifacts and out-of-fold predictions and writes the five figures (plus a
calibration plot) to ``results/figures/`` as 300-dpi PNG+PDF.

Usage
-----
    uv run python scripts/03_generate_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.clean import CohortStep
from ctb.data.context import FeatureSpec
from ctb.models.baselines import fit_full_gbdt_with_context
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.figures import (
    figure_calibration,
    figure_cohort_flow,
    figure_model_performance,
    figure_outcome_prevalence,
    figure_shap_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class _Result:
    """Lightweight container matching the ``BaselineResult`` interface for figures."""

    name: str
    y_true: NDArray[np.float64]
    y_score: NDArray[np.float64]


def main() -> int:
    """Generate all figures.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    # Figure 1 — cohort flow.
    steps_payload = json.loads((PROCESSED_DIR / "cohort_steps.json").read_text())
    steps = [CohortStep(**s) for s in steps_payload]
    figure_cohort_flow(steps)

    # Figure 2 — outcome prevalence + UpSet.
    prevalence = pd.read_csv(TABLES_DIR / "outcome_prevalence.csv")
    outcomes = pd.read_parquet(PROCESSED_DIR / "outcomes.parquet")
    figure_outcome_prevalence(prevalence, outcomes)

    # Figure 4 + calibration — model performance.
    oof = pd.read_parquet(PROCESSED_DIR / "oof_predictions.parquet")
    results = {
        name: _Result(
            name=name,
            y_true=grp["y_true"].to_numpy(np.float64),
            y_score=grp["y_score"].to_numpy(np.float64),
        )
        for name, grp in oof.groupby("model", sort=False)
    }
    figure_model_performance(results)
    figure_calibration(results)

    # Figure 5 — SHAP for GBDT-with-context (refit on full cohort).
    analysis = pd.read_parquet(PROCESSED_DIR / "analysis_frame.parquet")
    spec_payload = json.loads((PROCESSED_DIR / "feature_spec.json").read_text())
    spec = FeatureSpec(numeric=spec_payload["numeric"], categorical=spec_payload["categorical"])
    X = analysis[spec_payload["treatment_cols"]]
    C = analysis[spec.all_columns]
    y = analysis[spec_payload["outcome_col"]].astype(int)
    try:
        pipe, transformed = fit_full_gbdt_with_context(X, C, spec, y)
        figure_shap_summary(pipe, transformed)
    except Exception as exc:  # pragma: no cover - plotting convenience
        logger.warning("SHAP figure skipped: %s", exc)

    logger.info("All figures generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
