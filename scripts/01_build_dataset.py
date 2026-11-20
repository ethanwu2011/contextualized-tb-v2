"""End-to-end ETL: raw TB Portals tables -> analysis-ready processed artifacts.

Produces (under ``data/processed/`` and ``results/``):

* ``data/processed/analysis_frame.parquet`` — treatments (X) + context (C) + outcome columns.
* ``data/processed/outcomes.parquet``        — full per-condition outcome component frame.
* ``data/processed/feature_spec.json``       — treatment/numeric/categorical column groups.
* ``data/processed/cohort_steps.json``       — CONSORT inclusion steps.
* ``results/tables/outcome_prevalence.csv``  — component + composite prevalence.
* ``results/tables/cohort_flow.csv``         — CONSORT table.
* ``results/tables/context_missingness.csv`` — per-feature missingness.
* ``results/tables/table1.csv``              — Table 1 by outcome status (tableone).

This script prints only aggregate summaries; no row-level data is emitted.

Usage
-----
    uv run python scripts/01_build_dataset.py
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.clean import build_cohort, clean_patient_cases, clean_regimens
from ctb.data.context import build_context_matrix, context_missingness_table
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.data.outcomes import (
    derive_mortality,
    derive_treatment_intensification,
    outcome_prevalence_table,
)
from ctb.data.treatments import build_treatment_matrix
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

OUTCOME_COL = "treatment_intensification"


def main() -> int:
    """Run the full ETL pipeline.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    pc = clean_patient_cases(load_patient_cases())
    reg = clean_regimens(load_regimens())
    bx = load_biochemistry()

    cohort = build_cohort(pc, reg)
    ids = cohort.condition_ids

    outcomes_full = derive_treatment_intensification(pc, reg)
    outcomes = outcomes_full.reindex(ids)
    mortality = derive_mortality(reg).reindex(ids)["mortality"].fillna(False).astype(int)

    prevalence = outcome_prevalence_table(outcomes)
    X = build_treatment_matrix(reg, ids)
    C, spec = build_context_matrix(pc, bx, ids)
    y = outcomes[OUTCOME_COL].astype(int)

    # Align everything on the cohort index.
    X = X.reindex(ids)
    C = C.reindex(ids)
    analysis = X.join(C)
    analysis[OUTCOME_COL] = y.to_numpy()
    analysis["treatment_intensification_with_unfavorable"] = outcomes[
        "treatment_intensification_with_unfavorable"
    ].to_numpy()
    analysis["mortality"] = mortality.to_numpy()

    # --- Persist processed artifacts ---
    analysis.to_parquet(PROCESSED_DIR / "analysis_frame.parquet")
    outcomes.to_parquet(PROCESSED_DIR / "outcomes.parquet")
    spec_payload = {
        "treatment_cols": list(X.columns),
        "numeric": spec.numeric,
        "categorical": spec.categorical,
        "outcome_col": OUTCOME_COL,
    }
    (PROCESSED_DIR / "feature_spec.json").write_text(json.dumps(spec_payload, indent=2))
    (PROCESSED_DIR / "cohort_steps.json").write_text(
        json.dumps([s.__dict__ for s in cohort.steps], indent=2)
    )

    # --- Result tables ---
    prevalence.to_csv(TABLES_DIR / "outcome_prevalence.csv", index=False)
    pd.DataFrame([s.__dict__ for s in cohort.steps]).to_csv(
        TABLES_DIR / "cohort_flow.csv", index=False
    )
    context_missingness_table(C, spec).to_csv(TABLES_DIR / "context_missingness.csv", index=False)
    _write_table1(C, y)

    # --- Aggregate summary (safe to print) ---
    logger.info("=== BUILD SUMMARY ===")
    logger.info("Cohort conditions: %d", len(ids))
    logger.info("Outcome prevalence (%s): %.1f%%", OUTCOME_COL, 100 * y.mean())
    logger.info("Treatment matrix: %d drug columns", X.shape[1])
    logger.info(
        "Context matrix: %d features (%d numeric, %d categorical)",
        C.shape[1],
        len(spec.numeric),
        len(spec.categorical),
    )
    logger.info("Artifacts written to %s and %s", PROCESSED_DIR, TABLES_DIR)
    return 0


def _write_table1(context: pd.DataFrame, y: pd.Series[int]) -> None:
    """Write a Table 1 (cohort characteristics by outcome) using ``tableone``.

    Parameters
    ----------
    context : pandas.DataFrame
        Context matrix.
    y : pandas.Series
        Binary outcome.

    Returns
    -------
    None
    """
    try:
        from tableone import TableOne

        df = context.copy()
        df["treatment_intensification"] = y.to_numpy()
        # Use a readable subset for Table 1.
        display_num = [c for c in ["age_of_onset", "bmi", "lab_alt", "lab_ast"] if c in df.columns]
        display_cat = [
            c
            for c in [
                "gender",
                "type_of_resistance",
                "case_definition",
                "lung_localization",
                "comorb_hiv",
                "comorb_diabetes",
            ]
            if c in df.columns
        ]
        for c in display_cat:
            df[c] = df[c].astype("object")
        t1 = TableOne(
            df,
            columns=display_num + display_cat,
            categorical=display_cat,
            groupby="treatment_intensification",
            pval=True,
        )
        t1.to_csv(TABLES_DIR / "table1.csv")
        logger.info(
            "Wrote Table 1 (%d numeric, %d categorical vars)", len(display_num), len(display_cat)
        )
    except Exception as exc:  # pragma: no cover - reporting convenience
        logger.warning("Table 1 generation skipped: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
