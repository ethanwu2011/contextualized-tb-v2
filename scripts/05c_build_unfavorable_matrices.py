"""Build the named died-or-failure modelling matrices used by the PATCH manuscript.

This is deliberately separate from ``05b_build_conversion_matrices.py``. The latter is retained
for the culture-conversion analysis and must not be used as the outcome source for the PATCH
manuscript. This script writes a matrix with an explicit endpoint name so that figures cannot
silently inherit the older outcome.

Only aggregate metadata is written to the results directory. The matrix itself is a local,
patient-level intermediate and is excluded from the public repository by the release allowlist.

Usage
-----
    uv run python scripts/05c_build_unfavorable_matrices.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ctb.config import METRICS_DIR, PROCESSED_DIR, SEED, ensure_output_dirs
from ctb.data.clean import build_cohort, clean_patient_cases, clean_regimens
from ctb.data.context import (
    FeatureSpec,
    apply_context_encoder,
    build_context_matrix,
    fit_context_encoder,
)
from ctb.data.conversion import derive_conversion
from ctb.data.load import load_biochemistry, load_dst, load_patient_cases, load_regimens
from ctb.data.treatments import build_treatment_matrix
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

MATRIX_PATH = PROCESSED_DIR / "unfavorable_model_matrices.npz"
MANIFEST_PATH = METRICS_DIR / "unfavorable_model_matrices.json"
ENDPOINT_NAME = "died_or_failure"
ENDPOINT_CODES = frozenset({"died", "failure"})

_META_CONTEXT_COLS = [
    "type_of_resistance",
    "baseline_smear",
    "age_of_onset",
    "country",
    "comorb_hiv",
    "comorb_anemia",
    "comorb_diabetes",
    "comorb_covid",
    "comorb_psychiatric",
    "comorb_renal",
    "comorb_hepatic",
]


def encode_context(context: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Encode numeric and categorical context with fit-on-cohort transforms."""
    encoder = fit_context_encoder(context, spec)
    return apply_context_encoder(context, spec, encoder)


def _died_or_failure(regimens: pd.DataFrame) -> pd.Series:
    """Return the primary manuscript endpoint without including loss to follow-up."""
    grouped = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    return grouped.agg(
        lambda values: bool(values.dropna().astype(str).isin(ENDPOINT_CODES).any())
    ).rename(ENDPOINT_NAME)


def _file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a local release artifact."""
    return sha256(path.read_bytes()).hexdigest()


def _repository_state() -> tuple[str | None, bool | None]:
    """Return the current Git commit and whether the checkout has uncommitted changes."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit_result.stdout.strip() or None, bool(status_result.stdout.strip())


def _critical_source_hashes(root: Path) -> dict[str, str]:
    """Hash the source files that define the primary endpoint and model preprocessing."""
    paths = (
        "scripts/05c_build_unfavorable_matrices.py",
        "src/ctb/config.py",
        "src/ctb/data/clean.py",
        "src/ctb/data/context.py",
        "src/ctb/data/load.py",
        "src/ctb/data/treatments.py",
        "src/ctb/models/archetype_contextualized.py",
        "uv.lock",
    )
    return {path: _file_sha256(root / path) for path in paths if (root / path).exists()}


def main() -> int:
    """Build and persist the endpoint-explicit matrices."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    patient_cases = clean_patient_cases(load_patient_cases())
    regimens = clean_regimens(load_regimens())
    biochemistry = load_biochemistry()
    dst = load_dst()

    conversion = derive_conversion(dst)
    # The current manuscript-era matrix retains any episode with an interpretable positive culture
    # in the observed record. ``first_pos_day`` is retained in the manifest so that this rule is
    # not mislabeled as a strict day-zero baseline window. A future confirmatory release must set
    # and enforce a prespecified time-zero cutoff before rebuilding the full analysis.
    observed_record_positive = set(conversion[conversion["first_pos_day"].notna()].index)
    cohort = build_cohort(patient_cases, regimens)
    condition_ids = [cid for cid in cohort.condition_ids if cid in observed_record_positive]
    index: pd.Index[Any] = pd.Index(condition_ids, name="condition_id")
    first_positive_day = conversion["first_pos_day"].reindex(index)

    treatments = build_treatment_matrix(regimens, condition_ids).reindex(index)
    context_raw, spec = build_context_matrix(patient_cases, biochemistry, condition_ids)
    context_raw = context_raw.reindex(index)
    context = encode_context(context_raw, spec)
    outcome = _died_or_failure(regimens).reindex(index).fillna(False).astype(int)

    repository_root = Path(__file__).resolve().parents[1]
    repository_commit, repository_dirty = _repository_state()
    if os.environ.get("CTB_REQUIRE_CLEAN_RELEASE") == "1" and repository_dirty:
        raise RuntimeError(
            "Refusing a final release build from a dirty Git checkout. "
            "Commit the reviewed source or unset CTB_REQUIRE_CLEAN_RELEASE for local analysis."
        )

    # Keep the raw fields needed by the interpretation figures next to the endpoint matrix. This
    # is a local patient-level artifact and is excluded from the public source-only export.
    meta = pd.DataFrame(index=index)
    meta[ENDPOINT_NAME] = outcome
    for col in _META_CONTEXT_COLS:
        if col in context_raw.columns:
            meta[col] = context_raw[col]
    meta.to_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet")

    np.savez(
        MATRIX_PATH,
        X=treatments.to_numpy(dtype=np.float64),
        C=context.to_numpy(dtype=np.float64),
        y=outcome.to_numpy(dtype=np.int64),
        drugs=np.asarray(list(treatments.columns), dtype=object),
        cond_ids=np.asarray(condition_ids, dtype=object),
        context_names=np.asarray(list(context.columns), dtype=object),
        outcome_name=np.asarray(ENDPOINT_NAME),
        outcome_codes=np.asarray(sorted(ENDPOINT_CODES), dtype=object),
    )
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    manifest = {
        "source_release": "TB Portals published clinical data August 2023",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "repository_commit": repository_commit,
        "repository_dirty": repository_dirty,
        "critical_source_sha256": _critical_source_hashes(repository_root),
        "dependency_lock_sha256": _file_sha256(lock_path) if lock_path.exists() else None,
        "seed": SEED,
        "endpoint": ENDPOINT_NAME,
        "outcome_codes": sorted(ENDPOINT_CODES),
        "n_conditions": int(len(index)),
        "n_positive": int(outcome.sum()),
        "n_drugs": int(treatments.shape[1]),
        "n_context_features": int(context.shape[1]),
        "drug_names": list(treatments.columns),
        "context_names": list(context.columns),
        "drug_names_sha256": sha256("\n".join(treatments.columns).encode("utf-8")).hexdigest(),
        "context_names_sha256": sha256("\n".join(context.columns).encode("utf-8")).hexdigest(),
        "preprocessing": {
            "numeric": "median imputation followed by standardization",
            "categorical": "one-hot encoding with an explicit missing category",
            "matrix_fit_scope": "full analysis cohort",
            "holdout_fit_scope": "training split only in holdout scripts",
        },
        "eligibility": {
            "culture_positive_rule": (
                "at least one interpretable positive culture in the observed record"
            ),
            "regimen_rule": (
                "at least one parsed drug on any recorded regimen line for eligibility; "
                "X uses the first regimen line in record order"
            ),
            "resistance_rule": "defined resistance class",
            "baseline_window_days": None,
            "first_positive_day_summary": {
                "day_zero": int((first_positive_day == 0).sum()),
                "after_day_zero": int((first_positive_day > 0).sum()),
                "after_day_seven": int((first_positive_day > 7).sum()),
                "after_day_thirty": int((first_positive_day > 30).sum()),
                "after_day_sixty": int((first_positive_day > 60).sum()),
            },
        },
        "split_definitions": {
            "primary": "all eligible observed-record culture-positive episodes",
            "temporal": "registration year <= 2020 for training and >= 2021 for testing",
            "country": "one country held out with the encoder fitted on all other countries",
        },
        "baseline_culture_positive_filter": False,
        "loss_to_follow_up_in_primary_endpoint": False,
        "metadata_artifact": "data/processed/unfavorable_context_meta.parquet",
        "artifact_sha256": {
            "matrix": _file_sha256(MATRIX_PATH),
            "metadata": _file_sha256(PROCESSED_DIR / "unfavorable_context_meta.parquet"),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote %s matrices with n=%d, positive=%.3f, drugs=%d, context=%d",
        ENDPOINT_NAME,
        len(index),
        float(outcome.mean()) if len(outcome) else 0.0,
        treatments.shape[1],
        context.shape[1],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
