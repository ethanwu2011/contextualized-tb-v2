"""Load analysis artifacts with an explicit endpoint contract.

The repository contains older culture-conversion artifacts for a separate analysis. Manuscript
analyses must use the endpoint-explicit died-or-failure matrix so an old file cannot be mistaken for
the current primary outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR

PRIMARY_ENDPOINT = "died_or_failure"
PRIMARY_MATRIX = PROCESSED_DIR / "unfavorable_model_matrices.npz"
PRIMARY_SUBTYPES = PROCESSED_DIR / "multimodal_subtypes.npz"
REQUIRED_MATRIX_KEYS = frozenset(
    {"X", "C", "y", "drugs", "cond_ids", "context_names", "outcome_name"}
)
REQUIRED_SUBTYPE_KEYS = frozenset({"subtypes", "loadings", "cond_ids", "outcome_name"})


def load_primary_raw_context(
    condition_ids: Sequence[object],
) -> tuple[pd.DataFrame, Any]:
    """Build the unencoded primary context for fold-specific preprocessing.

    The persisted primary matrix is already encoded for the full analysis cohort and is suitable
    for descriptive fits. Out-of-fold and holdout analyses must instead fit imputation, scaling,
    and categorical vocabularies on their training rows. This helper reconstructs the raw context
    using the approved local data directory so callers can apply that rule explicitly.

    Parameters
    ----------
    condition_ids : sequence
        Matrix row identifiers in the desired order.

    Returns
    -------
    tuple
        Raw context rows aligned to ``condition_ids`` and the corresponding ``FeatureSpec``.
    """
    from ctb.data.clean import clean_patient_cases
    from ctb.data.context import build_context_matrix
    from ctb.data.load import load_biochemistry, load_patient_cases

    ids = list(condition_ids)
    raw, spec = build_context_matrix(
        clean_patient_cases(load_patient_cases()), load_biochemistry(), ids
    )
    aligned = raw.reindex(pd.Index(ids, name="condition_id"))
    return aligned.reset_index(drop=True), spec


def load_primary_matrix(path: Path | None = None) -> Any:
    """Load the manuscript matrix and reject missing or mismatched endpoint metadata.

    Parameters
    ----------
    path : pathlib.Path or None
        Optional path used by tests. The default is the local processed artifact.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        Open NumPy archive. Callers that retain the object should close it when finished.

    Raises
    ------
    FileNotFoundError
        If the endpoint-explicit matrix has not been built.
    ValueError
        If the archive is missing required fields or names another endpoint.
    """
    matrix_path = path or PRIMARY_MATRIX
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {PRIMARY_ENDPOINT} matrix at {matrix_path}. "
            "Run scripts/05c_build_unfavorable_matrices.py first."
        )
    data = np.load(matrix_path, allow_pickle=True)
    try:
        missing = REQUIRED_MATRIX_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"Primary matrix is missing required fields: {sorted(missing)}")
        endpoint = str(np.asarray(data["outcome_name"]).item())
        if endpoint != PRIMARY_ENDPOINT:
            raise ValueError(
                f"Expected endpoint {PRIMARY_ENDPOINT!r}, found {endpoint!r} in {matrix_path}"
            )
        x = np.asarray(data["X"])
        context = np.asarray(data["C"])
        outcome = np.asarray(data["y"]).reshape(-1)
        condition_ids = np.asarray(data["cond_ids"]).reshape(-1)
        drugs = np.asarray(data["drugs"]).reshape(-1)
        context_names = np.asarray(data["context_names"]).reshape(-1)
        if x.ndim != 2 or context.ndim != 2:
            raise ValueError("Primary matrix predictors must be two-dimensional")
        n = x.shape[0]
        if context.shape[0] != n or outcome.shape[0] != n or condition_ids.shape[0] != n:
            raise ValueError("Primary matrix row-aligned arrays have inconsistent lengths")
        if x.shape[1] != drugs.shape[0] or context.shape[1] != context_names.shape[0]:
            raise ValueError("Primary matrix names do not match predictor dimensions")
        if len(set(map(str, condition_ids.tolist()))) != n:
            raise ValueError("Primary matrix condition identifiers must be unique")
        if not np.isin(outcome.astype(int), [0, 1]).all():
            raise ValueError("Primary matrix outcome must be binary")
    except Exception:
        data.close()
        raise
    return data


def load_primary_subtypes(
    path: Path | None = None,
    expected_condition_ids: Sequence[object] | None = None,
) -> Any:
    """Load subtype assignments linked to the endpoint-explicit primary matrix.

    The subtype archive is an intermediate local artifact. It is accepted only when it carries the
    same endpoint label and row-aligned condition identifiers as the primary analysis contract. If
    ``expected_condition_ids`` is supplied, the subtype rows must match that sequence exactly. When
    no sequence is supplied and the default primary matrix exists, its condition-id sequence is used
    automatically. This prevents a stale subtype archive from being paired silently with a new
    primary matrix.
    """
    subtype_path = path or PRIMARY_SUBTYPES
    if not subtype_path.exists():
        raise FileNotFoundError(
            f"Missing subtype artifact at {subtype_path}. Run scripts/22_multimodal_subtypes.py."
        )
    data = np.load(subtype_path, allow_pickle=True)
    try:
        missing = REQUIRED_SUBTYPE_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"Subtype archive is missing required fields: {sorted(missing)}")
        endpoint = str(np.asarray(data["outcome_name"]).item())
        if endpoint != PRIMARY_ENDPOINT:
            raise ValueError(
                f"Expected endpoint {PRIMARY_ENDPOINT!r}, found {endpoint!r} in {subtype_path}"
            )
        loadings = np.asarray(data["loadings"])
        subtypes = np.asarray(data["subtypes"]).reshape(-1)
        condition_ids = np.asarray(data["cond_ids"]).reshape(-1)
        if loadings.ndim != 2 or loadings.shape[0] != subtypes.shape[0]:
            raise ValueError("Subtype loadings and assignments are not row aligned")
        if condition_ids.shape[0] != subtypes.shape[0]:
            raise ValueError("Subtype condition identifiers are not row aligned")
        expected = expected_condition_ids
        if expected is None and PRIMARY_MATRIX.exists():
            with load_primary_matrix(PRIMARY_MATRIX) as primary:
                expected = np.asarray(primary["cond_ids"]).reshape(-1).tolist()
        if expected is not None:
            expected_array = np.asarray(list(expected)).reshape(-1)
            if expected_array.shape != condition_ids.shape or not np.array_equal(
                expected_array.astype(str), condition_ids.astype(str)
            ):
                raise ValueError(
                    "Subtype condition identifiers do not match the primary matrix row order"
                )
        if len(set(map(str, condition_ids.tolist()))) != condition_ids.shape[0]:
            raise ValueError("Subtype condition identifiers must be unique")
        if not np.isfinite(loadings.astype(float)).all():
            raise ValueError("Subtype loadings contain non-finite values")
        if not np.issubdtype(subtypes.dtype, np.integer) or (subtypes < 0).any():
            raise ValueError("Subtype assignments must be non-negative integers")
    except Exception:
        data.close()
        raise
    return data
