"""Versioned, endpoint-aware model-matrix artifacts.

Historically, the PATCH analysis reused ``conversion_model_matrices.npz`` as a convenient container
for ``X`` and ``C`` and then sometimes replaced its culture-conversion ``y`` in memory.  That
convention is unsafe for figure code: reading the stored ``y`` can silently analyze culture
conversion while a caption claims the died/failure endpoint.

This module makes the endpoint part of the artifact contract.  Loaders require the caller to name
the expected endpoint and verify stored cohort/outcome fingerprints before returning any arrays.
Legacy NPZ files without this metadata fail closed and must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ctb.config import N_CV_FOLDS, PROCESSED_DIR
from ctb.data.conversion import derive_conversion
from ctb.data.outcomes import derive_patient_case_died_failure

CULTURE_CONVERSION: Final[str] = "culture_conversion"
DIED_FAILURE: Final[str] = "died_failure"
ARTIFACT_SCHEMA_VERSION: Final[int] = 6

ENDPOINT_ARTIFACTS: Final[dict[str, str]] = {
    CULTURE_CONVERSION: "conversion_model_matrices.npz",
    DIED_FAILURE: "died_failure_model_matrices.npz",
}
ENDPOINT_DEFINITION_IDS: Final[dict[str, str]] = {
    CULTURE_CONVERSION: "sustained_culture_conversion_v1",
    DIED_FAILURE: "patient_cases_final_died_failure_v1",
}
COHORT_DEFINITION_IDS: Final[dict[str, str]] = {
    CULTURE_CONVERSION: "consort_baseline_positive_v1",
    DIED_FAILURE: "consort_recorded_final_outcome_v1",
}
ENDPOINT_DEFINITIONS: Final[dict[str, str]] = {
    CULTURE_CONVERSION: "sustained culture conversion ever observed",
    DIED_FAILURE: (
        "recorded final Patient Cases outcome: Died or Failure is an event; Cured or Completed is "
        "an observed non-event; loss to follow-up, still on treatment, palliative care, unknown, "
        "missing, and unclassified outcomes are excluded"
    ),
}
COHORT_DEFINITIONS: Final[dict[str, str]] = {
    CULTURE_CONVERSION: (
        "CONSORT conditions with a first observed positive culture at or before day 0; conversion "
        "follow-up otherwise uses the archived ever-observed rule"
    ),
    DIED_FAILURE: (
        "all CONSORT conditions with a valid selected initial regimen, a defined resistance class, "
        "and an assessable recorded final Patient Cases died/failure outcome; culture measurements "
        "are not an eligibility requirement"
    ),
}

FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int64]
StringVector = NDArray[np.str_]


@dataclass(frozen=True)
class ModelMatrices:
    """Validated modelling matrices and their endpoint provenance."""

    X: FloatMatrix
    C: FloatMatrix
    y: IntVector
    drugs: StringVector
    cond_ids: StringVector
    patient_ids: StringVector
    source_cond_ids: StringVector
    context_names: StringVector
    endpoint: str
    endpoint_definition_id: str
    endpoint_definition: str
    cohort_definition_id: str
    cohort_definition: str
    source_n_rows: int
    n_unassessable: int
    cohort_sha256: str
    patient_sha256: str
    source_cohort_sha256: str
    outcome_sha256: str
    design_sha256: str
    provenance_sha256: str

    @property
    def n_rows(self) -> int:
        """Number of analysis rows."""
        return int(self.y.shape[0])

    @property
    def n_events(self) -> int:
        """Number of positive endpoint events."""
        return int(self.y.sum())

    @property
    def n_non_events(self) -> int:
        """Number of observed endpoint non-events."""
        return self.n_rows - self.n_events


def _hash_strings(values: NDArray[np.str_]) -> str:
    """Hash a sequence of identifiers without delimiter ambiguity."""
    digest = sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def cohort_fingerprint(cond_ids: NDArray[np.str_]) -> str:
    """Return a stable SHA-256 fingerprint of ordered condition identifiers."""
    return _hash_strings(np.asarray(cond_ids, dtype=str))


def patient_group_fingerprint(
    cond_ids: NDArray[np.str_],
    patient_ids: NDArray[np.str_],
) -> str:
    """Hash the ordered condition-to-patient mapping used for grouped validation."""
    condition_values = np.asarray(cond_ids, dtype=str)
    patient_values = np.asarray(patient_ids, dtype=str)
    if condition_values.ndim != 1 or patient_values.ndim != 1:
        raise ValueError("Condition and patient identifiers must be 1-D")
    if len(condition_values) != len(patient_values):
        raise ValueError("Condition and patient identifiers must have the same length")
    digest = sha256()
    for condition_id, patient_id in zip(condition_values, patient_values, strict=True):
        for value in (condition_id, patient_id):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def outcome_fingerprint(
    endpoint: str, cond_ids: NDArray[np.str_], y: NDArray[np.integer[Any]]
) -> str:
    """Return a stable endpoint + ordered-cohort + outcome SHA-256 fingerprint."""
    digest = sha256()
    digest.update(endpoint.encode("utf-8"))
    digest.update(bytes.fromhex(cohort_fingerprint(np.asarray(cond_ids, dtype=str))))
    digest.update(np.asarray(y, dtype=np.int64).tobytes(order="C"))
    return digest.hexdigest()


def design_fingerprint(
    X: NDArray[np.floating[Any]],
    C: NDArray[np.floating[Any]],
    drugs: NDArray[np.str_],
    context_names: NDArray[np.str_],
) -> str:
    """Hash ordered design values and column labels in a platform-stable encoding."""
    digest = sha256()
    for array in (np.asarray(X, dtype="<f8"), np.asarray(C, dtype="<f8")):
        digest.update(len(array.shape).to_bytes(8, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    digest.update(bytes.fromhex(_hash_strings(np.asarray(drugs, dtype=str))))
    digest.update(bytes.fromhex(_hash_strings(np.asarray(context_names, dtype=str))))
    return digest.hexdigest()


def provenance_fingerprint(
    *,
    endpoint: str,
    endpoint_definition_id: str,
    endpoint_definition: str,
    cohort_definition_id: str,
    cohort_definition: str,
    source_cond_ids: StringVector,
    cond_ids: StringVector,
    patient_ids: StringVector,
    y: IntVector,
    X: FloatMatrix,
    C: FloatMatrix,
    drugs: StringVector,
    context_names: StringVector,
) -> str:
    """Bind definitions, denominators, cohort membership, outcome, and design into one hash."""
    source_hash = cohort_fingerprint(source_cond_ids)
    cohort_hash = cohort_fingerprint(cond_ids)
    patient_hash = patient_group_fingerprint(cond_ids, patient_ids)
    outcome_hash = outcome_fingerprint(endpoint, cond_ids, y)
    design_hash = design_fingerprint(X, C, drugs, context_names)
    counts = (
        len(source_cond_ids),
        len(source_cond_ids) - len(cond_ids),
        len(cond_ids),
        int(y.sum()),
        len(y) - int(y.sum()),
    )

    digest = sha256()
    for value in (
        str(ARTIFACT_SCHEMA_VERSION),
        endpoint,
        endpoint_definition_id,
        endpoint_definition,
        cohort_definition_id,
        cohort_definition,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for count in counts:
        digest.update(int(count).to_bytes(8, "big", signed=False))
    for fingerprint in (source_hash, cohort_hash, patient_hash, outcome_hash, design_hash):
        digest.update(bytes.fromhex(fingerprint))
    return digest.hexdigest()


def _write_model_matrices(
    path: Path,
    *,
    X: NDArray[np.floating[Any]],
    C: NDArray[np.floating[Any]],
    y: NDArray[Any],
    drugs: NDArray[Any],
    cond_ids: NDArray[Any],
    patient_ids: NDArray[Any],
    source_cond_ids: NDArray[Any],
    context_names: NDArray[Any],
    endpoint: str,
) -> None:
    """Write a self-identifying model-matrix artifact.

    Parameters are keyword-only so the endpoint cannot be confused with another array.  The writer
    validates the same shape/binary invariants as the loader before persisting the artifact.
    """
    if endpoint not in ENDPOINT_ARTIFACTS:
        raise ValueError(
            f"Unsupported endpoint {endpoint!r}; expected one of {sorted(ENDPOINT_ARTIFACTS)}"
        )
    endpoint_definition_id = ENDPOINT_DEFINITION_IDS[endpoint]
    endpoint_definition = ENDPOINT_DEFINITIONS[endpoint]
    cohort_definition_id = COHORT_DEFINITION_IDS[endpoint]
    cohort_definition = COHORT_DEFINITIONS[endpoint]

    x_arr = _coerce_numeric_matrix(X, field="X", binary=True)
    c_arr = _coerce_numeric_matrix(C, field="C", binary=False)
    y_arr = _coerce_binary_vector(y, field="y")
    drug_arr = _coerce_string_vector(drugs, field="drugs")
    id_arr = _coerce_identifier_vector(cond_ids, field="cond_ids")
    patient_id_arr = _coerce_identifier_vector(patient_ids, field="patient_ids")
    source_id_arr = _coerce_identifier_vector(source_cond_ids, field="source_cond_ids")
    name_arr = _coerce_string_vector(context_names, field="context_names")
    _validate_arrays(x_arr, c_arr, y_arr, drug_arr, id_arr, patient_id_arr, name_arr)
    _validate_source_cohort(source_id_arr, id_arr)
    source_n_rows = len(source_id_arr)
    n_unassessable = source_n_rows - len(id_arr)
    cohort_hash = cohort_fingerprint(id_arr)
    patient_hash = patient_group_fingerprint(id_arr, patient_id_arr)
    source_cohort_hash = cohort_fingerprint(source_id_arr)
    outcome_hash = outcome_fingerprint(endpoint, id_arr, y_arr)
    design_hash = design_fingerprint(x_arr, c_arr, drug_arr, name_arr)
    provenance_hash = provenance_fingerprint(
        endpoint=endpoint,
        endpoint_definition_id=endpoint_definition_id,
        endpoint_definition=endpoint_definition,
        cohort_definition_id=cohort_definition_id,
        cohort_definition=cohort_definition,
        source_cond_ids=source_id_arr,
        cond_ids=id_arr,
        patient_ids=patient_id_arr,
        y=y_arr,
        X=x_arr,
        C=c_arr,
        drugs=drug_arr,
        context_names=name_arr,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=x_arr,
        C=c_arr,
        y=y_arr,
        drugs=drug_arr,
        cond_ids=id_arr,
        patient_ids=patient_id_arr,
        source_cond_ids=source_id_arr,
        context_names=name_arr,
        artifact_schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
        endpoint=np.asarray(endpoint),
        endpoint_definition_id=np.asarray(endpoint_definition_id),
        endpoint_definition=np.asarray(endpoint_definition),
        cohort_definition_id=np.asarray(cohort_definition_id),
        cohort_definition=np.asarray(cohort_definition),
        source_n_rows=np.asarray(source_n_rows, dtype=np.int64),
        n_unassessable=np.asarray(n_unassessable, dtype=np.int64),
        n_rows=np.asarray(len(y_arr), dtype=np.int64),
        n_events=np.asarray(y_arr.sum(), dtype=np.int64),
        n_non_events=np.asarray(len(y_arr) - y_arr.sum(), dtype=np.int64),
        cohort_sha256=np.asarray(cohort_hash),
        patient_sha256=np.asarray(patient_hash),
        source_cohort_sha256=np.asarray(source_cohort_hash),
        outcome_sha256=np.asarray(outcome_hash),
        design_sha256=np.asarray(design_hash),
        provenance_sha256=np.asarray(provenance_hash),
    )


def write_culture_conversion_matrices(
    path: Path,
    *,
    X: NDArray[np.floating[Any]],
    C: NDArray[np.floating[Any]],
    dst: pd.DataFrame,
    drugs: NDArray[Any],
    cond_ids: NDArray[Any],
    patient_ids: NDArray[Any],
    source_cond_ids: NDArray[Any],
    context_names: NDArray[Any],
) -> None:
    """Derive culture-conversion labels from the raw semantic table, then write the artifact."""
    ordered_ids = np.asarray(cond_ids).tolist()
    labels = derive_conversion(dst)["event"].reindex(ordered_ids)
    if labels.isna().any():
        raise ValueError(
            "Culture-conversion artifact includes identifiers without an assessable derived event"
        )
    _write_model_matrices(
        path,
        X=X,
        C=C,
        y=labels.astype(int).to_numpy(dtype=np.int64),
        drugs=drugs,
        cond_ids=cond_ids,
        patient_ids=patient_ids,
        source_cond_ids=source_cond_ids,
        context_names=context_names,
        endpoint=CULTURE_CONVERSION,
    )


def write_died_failure_matrices(
    path: Path,
    *,
    X: NDArray[np.floating[Any]],
    C: NDArray[np.floating[Any]],
    patient_cases: pd.DataFrame,
    drugs: NDArray[Any],
    cond_ids: NDArray[Any],
    patient_ids: NDArray[Any],
    source_cond_ids: NDArray[Any],
    context_names: NDArray[Any],
) -> None:
    """Derive final Patient Cases died/failure labels, then write the primary artifact."""
    ordered_ids = np.asarray(cond_ids).tolist()
    labels = derive_patient_case_died_failure(patient_cases)["died_failure"].reindex(ordered_ids)
    if labels.isna().any():
        raise ValueError(
            "Died/failure artifact includes identifiers without an assessable derived label"
        )
    patient_lookup = patient_cases.set_index("condition_id")["patient_id"].reindex(ordered_ids)
    missing_patient_links = int(
        (patient_lookup.isna() | patient_lookup.astype("string").str.strip().eq("")).sum()
    )
    if missing_patient_links:
        raise ValueError(
            "Died/failure artifact has incomplete condition-to-patient linkage: "
            f"missing_links={missing_patient_links}"
        )
    expected_patient_ids = _coerce_identifier_vector(
        patient_lookup.to_numpy(),
        field="patient_cases.patient_id",
    )
    declared_patient_ids = _coerce_identifier_vector(patient_ids, field="patient_ids")
    if not np.array_equal(declared_patient_ids, expected_patient_ids):
        raise ValueError(
            "Died/failure artifact patient_ids disagree with the Patient Cases linkage"
        )
    _write_model_matrices(
        path,
        X=X,
        C=C,
        y=labels.astype(int).to_numpy(dtype=np.int64),
        drugs=drugs,
        cond_ids=cond_ids,
        patient_ids=patient_ids,
        source_cond_ids=source_cond_ids,
        context_names=context_names,
        endpoint=DIED_FAILURE,
    )


def load_endpoint_matrices(
    expected_endpoint: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
) -> ModelMatrices:
    """Load the canonical artifact for ``expected_endpoint`` and verify its provenance."""
    try:
        filename = ENDPOINT_ARTIFACTS[expected_endpoint]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported endpoint {expected_endpoint!r}; expected one of "
            f"{sorted(ENDPOINT_ARTIFACTS)}"
        ) from exc
    return load_model_matrices(processed_dir / filename, expected_endpoint=expected_endpoint)


def load_model_matrices(path: Path, *, expected_endpoint: str) -> ModelMatrices:
    """Load and validate one endpoint-specific NPZ artifact.

    Raises
    ------
    FileNotFoundError
        If the endpoint artifact has not been built.
    ValueError
        If metadata are absent, hashes/shapes disagree, or the stored endpoint differs from the
        caller's declared endpoint.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {expected_endpoint!r} model artifact: {path}. "
            "Run scripts/05b_build_conversion_matrices.py to build endpoint-specific artifacts."
        )

    required = {
        "X",
        "C",
        "y",
        "drugs",
        "cond_ids",
        "patient_ids",
        "source_cond_ids",
        "context_names",
        "artifact_schema_version",
        "endpoint",
        "endpoint_definition_id",
        "endpoint_definition",
        "cohort_definition_id",
        "cohort_definition",
        "source_n_rows",
        "n_unassessable",
        "n_rows",
        "n_events",
        "n_non_events",
        "cohort_sha256",
        "patient_sha256",
        "source_cohort_sha256",
        "outcome_sha256",
        "design_sha256",
        "provenance_sha256",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"{path} is an unversioned or incomplete model artifact; missing {missing}. "
                "Rebuild it before generating endpoint-labelled results."
            )

        schema_version = _scalar_int(data["artifact_schema_version"], "artifact_schema_version")
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses model-artifact schema {schema_version}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        endpoint = _scalar_text(data["endpoint"], "endpoint")
        if endpoint != expected_endpoint:
            raise ValueError(
                f"Endpoint provenance mismatch for {path}: artifact declares {endpoint!r}, "
                f"but the caller requires {expected_endpoint!r}"
            )

        endpoint_definition_id = _scalar_text(
            data["endpoint_definition_id"], "endpoint_definition_id"
        )
        endpoint_definition = _scalar_text(data["endpoint_definition"], "endpoint_definition")
        cohort_definition_id = _scalar_text(data["cohort_definition_id"], "cohort_definition_id")
        cohort_definition = _scalar_text(data["cohort_definition"], "cohort_definition")
        if (
            endpoint_definition_id != ENDPOINT_DEFINITION_IDS[endpoint]
            or endpoint_definition != ENDPOINT_DEFINITIONS[endpoint]
            or cohort_definition_id != COHORT_DEFINITION_IDS[endpoint]
            or cohort_definition != COHORT_DEFINITIONS[endpoint]
        ):
            raise ValueError(
                f"Endpoint contract mismatch for {path}; rebuild with the canonical "
                f"{endpoint!r} producer"
            )

        X = _coerce_numeric_matrix(data["X"], field="X", binary=True)
        C = _coerce_numeric_matrix(data["C"], field="C", binary=False)
        y = _coerce_binary_vector(data["y"], field="y")
        drugs = _coerce_string_vector(data["drugs"], field="drugs")
        cond_ids = _coerce_identifier_vector(data["cond_ids"], field="cond_ids")
        patient_ids = _coerce_identifier_vector(data["patient_ids"], field="patient_ids")
        source_cond_ids = _coerce_identifier_vector(
            data["source_cond_ids"], field="source_cond_ids"
        )
        context_names = _coerce_string_vector(data["context_names"], field="context_names")
        source_n_rows = _scalar_int(data["source_n_rows"], "source_n_rows")
        n_unassessable = _scalar_int(data["n_unassessable"], "n_unassessable")
        stored_rows = _scalar_int(data["n_rows"], "n_rows")
        stored_events = _scalar_int(data["n_events"], "n_events")
        stored_non_events = _scalar_int(data["n_non_events"], "n_non_events")
        stored_cohort_hash = _scalar_text(data["cohort_sha256"], "cohort_sha256")
        stored_patient_hash = _scalar_text(data["patient_sha256"], "patient_sha256")
        stored_source_cohort_hash = _scalar_text(
            data["source_cohort_sha256"], "source_cohort_sha256"
        )
        stored_outcome_hash = _scalar_text(data["outcome_sha256"], "outcome_sha256")
        stored_design_hash = _scalar_text(data["design_sha256"], "design_sha256")
        stored_provenance_hash = _scalar_text(data["provenance_sha256"], "provenance_sha256")

    _validate_arrays(X, C, y, drugs, cond_ids, patient_ids, context_names)
    _validate_source_cohort(source_cond_ids, cond_ids)
    if (
        stored_rows != len(y)
        or stored_events != int(y.sum())
        or stored_non_events != len(y) - int(y.sum())
    ):
        raise ValueError(
            f"Count metadata mismatch for {path}: stored n/events/non-events="
            f"{stored_rows}/{stored_events}/{stored_non_events}, arrays contain "
            f"{len(y)}/{int(y.sum())}/{len(y) - int(y.sum())}"
        )
    if (
        source_n_rows != len(source_cond_ids)
        or n_unassessable != len(source_cond_ids) - stored_rows
    ):
        raise ValueError(
            f"Assessability metadata mismatch for {path}: source={source_n_rows}, "
            f"assessed={stored_rows}, unassessable={n_unassessable}"
        )

    actual_cohort_hash = cohort_fingerprint(cond_ids)
    actual_patient_hash = patient_group_fingerprint(cond_ids, patient_ids)
    actual_source_cohort_hash = cohort_fingerprint(source_cond_ids)
    actual_outcome_hash = outcome_fingerprint(endpoint, cond_ids, y)
    actual_design_hash = design_fingerprint(X, C, drugs, context_names)
    actual_provenance_hash = provenance_fingerprint(
        endpoint=endpoint,
        endpoint_definition_id=endpoint_definition_id,
        endpoint_definition=endpoint_definition,
        cohort_definition_id=cohort_definition_id,
        cohort_definition=cohort_definition,
        source_cond_ids=source_cond_ids,
        cond_ids=cond_ids,
        patient_ids=patient_ids,
        y=y,
        X=X,
        C=C,
        drugs=drugs,
        context_names=context_names,
    )
    if stored_cohort_hash != actual_cohort_hash:
        raise ValueError(f"Cohort fingerprint mismatch for {path}")
    if stored_patient_hash != actual_patient_hash:
        raise ValueError(f"Patient-group fingerprint mismatch for {path}")
    if stored_source_cohort_hash != actual_source_cohort_hash:
        raise ValueError(f"Source-cohort fingerprint mismatch for {path}")
    if stored_outcome_hash != actual_outcome_hash:
        raise ValueError(f"Outcome fingerprint mismatch for {path}")
    if stored_design_hash != actual_design_hash:
        raise ValueError(f"Design fingerprint mismatch for {path}")
    if stored_provenance_hash != actual_provenance_hash:
        raise ValueError(f"Provenance fingerprint mismatch for {path}")

    return ModelMatrices(
        X=X,
        C=C,
        y=y,
        drugs=drugs,
        cond_ids=cond_ids,
        patient_ids=patient_ids,
        source_cond_ids=source_cond_ids,
        context_names=context_names,
        endpoint=endpoint,
        endpoint_definition_id=endpoint_definition_id,
        endpoint_definition=endpoint_definition,
        cohort_definition_id=cohort_definition_id,
        cohort_definition=cohort_definition,
        source_n_rows=source_n_rows,
        n_unassessable=n_unassessable,
        cohort_sha256=actual_cohort_hash,
        patient_sha256=actual_patient_hash,
        source_cohort_sha256=actual_source_cohort_hash,
        outcome_sha256=actual_outcome_hash,
        design_sha256=actual_design_hash,
        provenance_sha256=actual_provenance_hash,
    )


def _scalar_text(value: NDArray[np.generic], field: str) -> str:
    """Extract one text scalar from an NPZ metadata array."""
    if value.shape != ():
        raise ValueError(f"Metadata field {field!r} must be scalar")
    item = value.item()
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Metadata field {field!r} must be a non-empty text scalar")
    return item


def _scalar_int(value: NDArray[np.generic], field: str) -> int:
    """Extract one integer scalar from an NPZ metadata array."""
    if value.shape != ():
        raise ValueError(f"Metadata field {field!r} must be scalar")
    item = value.item()
    if isinstance(item, bool) or not isinstance(item, Integral):
        raise ValueError(f"Metadata field {field!r} must be an integer scalar")
    return int(item)


def _coerce_numeric_matrix(
    values: NDArray[Any],
    *,
    field: str,
    binary: bool,
) -> FloatMatrix:
    """Validate a raw numeric matrix before converting it to float64."""
    raw = _unmasked_array(values, field=field)
    if raw.ndim != 2:
        raise ValueError(f"Model artifact {field} must be 2-D")
    if raw.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(
            f"Model artifact {field} must have a Boolean, integer, or floating dtype; "
            f"found {raw.dtype}"
        )
    converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"Model artifact {field} must contain only finite values")
    if binary and not np.isin(converted, [0.0, 1.0]).all():
        raise ValueError(f"Model artifact {field} must contain only binary treatment indicators")
    return converted


def _coerce_binary_vector(values: NDArray[Any], *, field: str) -> IntVector:
    """Validate exact binary integer labels before converting them to int64."""
    raw = _unmasked_array(values, field=field)
    if raw.ndim != 1:
        raise ValueError(f"Model artifact {field} must be 1-D")
    if not len(raw):
        raise ValueError(f"Model artifact {field} must be non-empty")
    if raw.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(
            f"Model artifact {field} must have a Boolean or integer dtype; found {raw.dtype}"
        )
    if not np.isin(raw, [0, 1]).all():
        raise ValueError(f"Model artifact {field} must contain only binary values 0 and 1")
    converted = np.asarray(raw, dtype=np.int64)
    class_counts = np.bincount(converted, minlength=2)
    if bool((class_counts < N_CV_FOLDS).any()):
        raise ValueError(
            f"Model artifact {field} requires at least {N_CV_FOLDS} observations in each class; "
            f"found class counts {class_counts.tolist()}"
        )
    return converted


def _coerce_string_vector(values: NDArray[Any], *, field: str) -> StringVector:
    """Validate identifiers or labels before converting them to NumPy Unicode."""
    raw = _unmasked_array(values, field=field)
    if raw.ndim != 1:
        raise ValueError(f"Model artifact {field} must be 1-D")
    items = raw.tolist()
    if any(not isinstance(value, str | np.str_) for value in items):
        raise ValueError(f"Model artifact {field} must contain text values only")
    converted = np.asarray(items, dtype=str)
    if any(not value.strip() for value in converted):
        raise ValueError(f"Model artifact {field} must contain non-empty text values")
    return converted


def _coerce_identifier_vector(values: NDArray[Any], *, field: str) -> StringVector:
    """Validate condition identifiers before an explicit integer/text canonicalization."""
    raw = _unmasked_array(values, field=field)
    if raw.ndim != 1:
        raise ValueError(f"Model artifact {field} must be 1-D")
    items = raw.tolist()
    valid = all(
        (isinstance(value, str | np.str_) and bool(str(value).strip()))
        or (isinstance(value, Integral) and not isinstance(value, bool))
        for value in items
    )
    if not valid:
        raise ValueError(
            f"Model artifact {field} must contain only non-empty text or integer identifiers"
        )
    return np.asarray([str(value) for value in items], dtype=str)


def _unmasked_array(values: NDArray[Any], *, field: str) -> NDArray[Any]:
    """Convert an array-like only after rejecting masked missing values."""
    if isinstance(values, np.ma.MaskedArray):
        mask = np.asarray(values.mask)
        n_masked = int(mask.sum())
        if n_masked:
            raise ValueError(
                f"Model artifact {field} contains {n_masked} masked values; "
                "missingness must not be discarded"
            )
        return np.asarray(values.data)
    return np.asarray(values)


def _validate_source_cohort(source_cond_ids: StringVector, cond_ids: StringVector) -> None:
    """Require assessed identifiers to be an ordered subset of the source-candidate cohort."""
    if len(set(source_cond_ids.tolist())) != len(source_cond_ids):
        raise ValueError("Model artifact source_cond_ids must be unique")
    source_positions = {
        condition_id: position for position, condition_id in enumerate(source_cond_ids)
    }
    missing_count = sum(condition_id not in source_positions for condition_id in cond_ids)
    if missing_count:
        raise ValueError(
            f"Model artifact has {missing_count} assessed identifiers absent from source_cond_ids"
        )
    assessed_positions = [source_positions[condition_id] for condition_id in cond_ids]
    if assessed_positions != sorted(assessed_positions):
        raise ValueError(
            "Model artifact cond_ids must preserve source_cond_ids order after "
            "assessability filtering"
        )


def _validate_arrays(
    X: FloatMatrix,
    C: FloatMatrix,
    y: IntVector,
    drugs: StringVector,
    cond_ids: StringVector,
    patient_ids: StringVector,
    context_names: StringVector,
) -> None:
    """Validate row, column, finiteness, treatment, identifier, and outcome invariants."""
    if (
        X.ndim != 2
        or C.ndim != 2
        or y.ndim != 1
        or drugs.ndim != 1
        or cond_ids.ndim != 1
        or patient_ids.ndim != 1
        or context_names.ndim != 1
    ):
        raise ValueError("X and C must be 2-D; labels, y, cond_ids, and patient_ids must be 1-D")
    row_counts = {
        X.shape[0],
        C.shape[0],
        y.shape[0],
        cond_ids.shape[0],
        patient_ids.shape[0],
    }
    if len(row_counts) != 1:
        raise ValueError(
            "Model artifact row mismatch: "
            f"X={X.shape[0]}, C={C.shape[0]}, y={y.shape[0]}, "
            f"cond_ids={cond_ids.shape[0]}, patient_ids={patient_ids.shape[0]}"
        )
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Model artifact y must contain only binary values 0 and 1")
    if not np.isfinite(X).all() or not np.isfinite(C).all():
        raise ValueError("Model artifact X and C must contain only finite values")
    if not np.isin(X, [0.0, 1.0]).all():
        raise ValueError("Model artifact X must contain only binary treatment indicators")
    if len(drugs) != X.shape[1] or len(context_names) != C.shape[1]:
        raise ValueError(
            "Model artifact column-label mismatch: "
            f"X/drugs={X.shape[1]}/{len(drugs)}, "
            f"C/context_names={C.shape[1]}/{len(context_names)}"
        )
    if len(set(drugs.tolist())) != len(drugs):
        raise ValueError("Model artifact drugs must be unique")
    if len(set(context_names.tolist())) != len(context_names):
        raise ValueError("Model artifact context_names must be unique")
    if len(set(cond_ids.tolist())) != len(cond_ids):
        raise ValueError("Model artifact cond_ids must be unique")
    if any(not value for value in cond_ids):
        raise ValueError("Model artifact cond_ids must be non-empty")
    if any(not value for value in patient_ids):
        raise ValueError("Model artifact patient_ids must be non-empty")
