"""Tests for endpoint-specific model artifacts and fail-closed figure provenance."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ctb.data.model_matrices import (
    ARTIFACT_SCHEMA_VERSION,
    COHORT_DEFINITION_IDS,
    COHORT_DEFINITIONS,
    CULTURE_CONVERSION,
    DIED_FAILURE,
    ENDPOINT_ARTIFACTS,
    ENDPOINT_DEFINITION_IDS,
    ENDPOINT_DEFINITIONS,
    _write_model_matrices,
    load_endpoint_matrices,
    load_model_matrices,
    write_died_failure_matrices,
)


def _write(
    path: Path,
    endpoint: str,
    y: list[Any] | np.ndarray[Any, Any],
    *,
    cond_ids: list[object] | None = None,
    source_n_rows: int | None = None,
    n_unassessable: int = 0,
) -> None:
    raw_y = y if np.ma.isMaskedArray(y) else np.asarray(y)
    n = raw_y.shape[0]
    assessed_ids = cond_ids if cond_ids is not None else [f"condition-{i}" for i in range(n)]
    declared_source_n = n + n_unassessable if source_n_rows is None else source_n_rows
    source_ids = [
        *assessed_ids,
        *[f"excluded-condition-{i}" for i in range(declared_source_n - n)],
    ]
    _write_model_matrices(
        path,
        X=(np.arange(n * 2) % 2).astype(float).reshape(n, 2),
        C=np.arange(n * 3, dtype=float).reshape(n, 3),
        y=raw_y,
        drugs=np.asarray(["drug_a", "drug_b"], dtype=object),
        cond_ids=np.asarray(assessed_ids, dtype=object),
        patient_ids=np.asarray([f"patient-{i // 2}" for i in range(n)], dtype=object),
        source_cond_ids=np.asarray(source_ids, dtype=object),
        context_names=np.asarray(["c0", "c1", "c2"], dtype=object),
        endpoint=endpoint,
    )


def _balanced_y() -> list[int]:
    return [0, 1] * 5


def test_endpoint_loader_returns_validated_counts_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(
        path,
        DIED_FAILURE,
        _balanced_y(),
        source_n_rows=12,
        n_unassessable=2,
    )

    data = load_endpoint_matrices(DIED_FAILURE, processed_dir=tmp_path)

    assert data.endpoint == DIED_FAILURE
    assert data.n_rows == 10
    assert data.n_events == 5
    assert data.n_non_events == 5
    assert data.source_n_rows == 12
    assert data.n_unassessable == 2
    assert data.endpoint_definition_id == ENDPOINT_DEFINITION_IDS[DIED_FAILURE]
    assert data.endpoint_definition == ENDPOINT_DEFINITIONS[DIED_FAILURE]
    assert data.cohort_definition_id == COHORT_DEFINITION_IDS[DIED_FAILURE]
    assert data.cohort_definition == COHORT_DEFINITIONS[DIED_FAILURE]
    assert len(data.source_cond_ids) == 12
    assert len(data.patient_ids) == 10
    assert len(set(data.patient_ids.tolist())) == 5
    assert len(data.cohort_sha256) == 64
    assert len(data.patient_sha256) == 64
    assert len(data.source_cohort_sha256) == 64
    assert len(data.outcome_sha256) == 64
    assert len(data.provenance_sha256) == 64


def test_died_failure_figure_loader_rejects_conversion_artifact(tmp_path: Path) -> None:
    """A culture-conversion y can never satisfy a died/failure figure request."""
    path = tmp_path / "wrong_endpoint.npz"
    _write(path, CULTURE_CONVERSION, _balanced_y())

    with pytest.raises(ValueError, match="Endpoint provenance mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_unversioned_legacy_npz_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, X=np.zeros((2, 1)), C=np.zeros((2, 1)), y=np.asarray([0, 1]))

    with pytest.raises(ValueError, match="unversioned or incomplete"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_outcome_tampering_breaks_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["y"] = 1 - payload["y"]
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Outcome fingerprint mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_design_tampering_breaks_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["C"] = payload["C"].copy()
    payload["C"][0, 0] += 1
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Design fingerprint mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_patient_group_tampering_breaks_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["patient_ids"] = payload["patient_ids"].copy()
    payload["patient_ids"][0] = "different-patient"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Patient-group fingerprint mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_source_cohort_tampering_breaks_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y(), source_n_rows=12, n_unassessable=2)
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["source_cond_ids"] = payload["source_cond_ids"].copy()
    payload["source_cond_ids"][-1] = "different-excluded-condition"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Source-cohort fingerprint mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_definition_tampering_breaks_canonical_endpoint_contract(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["cohort_definition"] = np.asarray("tampered cohort definition")
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Endpoint contract mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_definition_id_tampering_breaks_canonical_endpoint_contract(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["endpoint_definition_id"] = np.asarray("unreviewed_endpoint_v999")
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Endpoint contract mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_writer_rejects_nonbinary_outcome(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="binary"):
        _write(tmp_path / "bad.npz", DIED_FAILURE, [0, 1] * 4 + [0, 2])


@pytest.mark.parametrize(
    "bad_y",
    [
        np.asarray([0.0, 1.0] * 5),
        np.asarray([0.9, 1.9] * 5),
        np.asarray(["0", "1"] * 5),
        np.asarray([0, 1, None, 0, 1, 0, 1, 0, 1, 0], dtype=object),
        np.asarray([0, 1, np.inf, 0, 1, 0, 1, 0, 1, 0]),
        np.asarray([0 + 0j, 1 + 0j] * 5),
    ],
)
def test_writer_rejects_noninteger_or_object_labels(
    tmp_path: Path,
    bad_y: np.ndarray[Any, Any],
) -> None:
    with pytest.raises(ValueError, match="Boolean or integer dtype"):
        _write(tmp_path / "bad.npz", DIED_FAILURE, bad_y)


def test_writer_rejects_masked_labels_before_mask_is_discarded(tmp_path: Path) -> None:
    masked = np.ma.array(_balanced_y(), mask=[True, *([False] * 9)])

    with pytest.raises(ValueError, match="masked values"):
        _write(tmp_path / "masked.npz", DIED_FAILURE, masked)


def test_writer_rejects_empty_or_multidimensional_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _write(tmp_path / "empty.npz", DIED_FAILURE, [])
    with pytest.raises(ValueError, match="must be 1-D"):
        _write(
            tmp_path / "matrix.npz",
            DIED_FAILURE,
            np.asarray([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
        )


def test_writer_rejects_too_few_observations_per_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 5 observations in each class"):
        _write(tmp_path / "underpowered.npz", DIED_FAILURE, [0] * 5 + [1] * 4)


def test_writer_rejects_missing_and_duplicate_condition_ids(tmp_path: Path) -> None:
    ids = [f"condition-{i}" for i in range(10)]
    missing = list(ids)
    missing[2] = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="non-empty text or integer"):
        _write(tmp_path / "missing-id.npz", DIED_FAILURE, _balanced_y(), cond_ids=missing)

    duplicate = list(ids)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="must be unique"):
        _write(tmp_path / "duplicate-id.npz", DIED_FAILURE, _balanced_y(), cond_ids=duplicate)


def test_writer_rejects_missing_or_misaligned_patient_ids(tmp_path: Path) -> None:
    common = {
        "X": (np.arange(20) % 2).astype(float).reshape(10, 2),
        "C": np.arange(30, dtype=float).reshape(10, 3),
        "y": np.asarray(_balanced_y()),
        "drugs": np.asarray(["drug_a", "drug_b"]),
        "cond_ids": np.asarray([f"condition-{i}" for i in range(10)]),
        "source_cond_ids": np.asarray([f"condition-{i}" for i in range(10)]),
        "context_names": np.asarray(["c0", "c1", "c2"]),
        "endpoint": DIED_FAILURE,
    }
    with pytest.raises(ValueError, match="non-empty text or integer"):
        _write_model_matrices(
            tmp_path / "missing-patient.npz",
            patient_ids=np.asarray(["patient"] * 9 + [None], dtype=object),
            **common,
        )
    with pytest.raises(ValueError, match="row mismatch"):
        _write_model_matrices(
            tmp_path / "misaligned-patient.npz",
            patient_ids=np.asarray(["patient"] * 9),
            **common,
        )


def test_fractional_label_tampering_is_rejected_before_coercion(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    # These values truncate to the original 0/1 labels. A coercion-first loader would therefore
    # accept the old count and fingerprint metadata.
    payload["y"] = payload["y"].astype(float) + 0.9
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Boolean or integer dtype"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_condition_id_reordering_breaks_ordered_cohort_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["cond_ids"] = payload["cond_ids"][::-1]
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="preserve source_cond_ids order"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_schema_and_assessability_metadata_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]
    _write(path, DIED_FAILURE, _balanced_y())
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    assert int(payload["artifact_schema_version"].item()) == ARTIFACT_SCHEMA_VERSION
    payload["source_n_rows"] = np.asarray(12, dtype=np.int64)
    payload["n_unassessable"] = np.asarray(1, dtype=np.int64)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Assessability metadata mismatch"):
        load_model_matrices(path, expected_endpoint=DIED_FAILURE)


def test_writer_rejects_assessed_ids_absent_from_source_cohort(tmp_path: Path) -> None:
    y = np.asarray(_balanced_y())
    with pytest.raises(ValueError, match="absent from source_cond_ids"):
        _write_model_matrices(
            tmp_path / "bad-source.npz",
            X=(np.arange(20) % 2).astype(float).reshape(10, 2),
            C=np.arange(30, dtype=float).reshape(10, 3),
            y=y,
            drugs=np.asarray(["drug_a", "drug_b"], dtype=object),
            cond_ids=np.asarray([f"condition-{i}" for i in range(10)], dtype=object),
            patient_ids=np.asarray([f"patient-{i}" for i in range(10)], dtype=object),
            source_cond_ids=np.asarray(
                ["unrelated", *[f"condition-{i}" for i in range(9)]], dtype=object
            ),
            context_names=np.asarray(["c0", "c1", "c2"], dtype=object),
            endpoint=DIED_FAILURE,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("drugs", "drugs must be unique"),
        ("context_names", "context_names must be unique"),
    ],
)
def test_writer_rejects_duplicate_feature_labels(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    y = np.asarray(_balanced_y())
    drugs = np.asarray(["drug_a", "drug_a"] if field == "drugs" else ["drug_a", "drug_b"])
    context_names = np.asarray(
        ["c0", "c0", "c2"] if field == "context_names" else ["c0", "c1", "c2"]
    )

    with pytest.raises(ValueError, match=message):
        _write_model_matrices(
            tmp_path / "duplicate-label.npz",
            X=(np.arange(20) % 2).astype(float).reshape(10, 2),
            C=np.arange(30, dtype=float).reshape(10, 3),
            y=y,
            drugs=drugs,
            cond_ids=np.asarray([f"condition-{i}" for i in range(10)]),
            patient_ids=np.asarray([f"patient-{i}" for i in range(10)]),
            source_cond_ids=np.asarray([f"condition-{i}" for i in range(10)]),
            context_names=context_names,
            endpoint=DIED_FAILURE,
        )


def test_public_died_failure_writer_derives_labels_and_does_not_accept_y(tmp_path: Path) -> None:
    assert "y" not in inspect.signature(write_died_failure_matrices).parameters
    cond_ids = np.asarray([f"condition-{i}" for i in range(10)])
    patient_cases = pd.DataFrame(
        {
            "condition_id": cond_ids,
            "patient_id": [f"patient-{i // 2}" for i in range(10)],
            "outcome": ["Died"] * 5 + ["Cured"] * 5,
        }
    )
    patient_ids = np.asarray([f"patient-{i // 2}" for i in range(10)])
    path = tmp_path / ENDPOINT_ARTIFACTS[DIED_FAILURE]

    write_died_failure_matrices(
        path,
        X=(np.arange(20) % 2).astype(float).reshape(10, 2),
        C=np.arange(30, dtype=float).reshape(10, 3),
        patient_cases=patient_cases,
        drugs=np.asarray(["drug_a", "drug_b"]),
        cond_ids=cond_ids,
        patient_ids=patient_ids,
        source_cond_ids=cond_ids,
        context_names=np.asarray(["c0", "c1", "c2"]),
    )

    loaded = load_model_matrices(path, expected_endpoint=DIED_FAILURE)
    assert loaded.y.tolist() == [1] * 5 + [0] * 5
    assert loaded.patient_ids.tolist() == patient_ids.tolist()


def test_public_died_failure_writer_rejects_patient_linkage_disagreement(tmp_path: Path) -> None:
    cond_ids = np.asarray([f"condition-{i}" for i in range(10)])
    patient_cases = pd.DataFrame(
        {
            "condition_id": cond_ids,
            "patient_id": [f"patient-{i}" for i in range(10)],
            "outcome": ["Died"] * 5 + ["Cured"] * 5,
        }
    )
    common = {
        "X": (np.arange(20) % 2).astype(float).reshape(10, 2),
        "C": np.arange(30, dtype=float).reshape(10, 3),
        "patient_cases": patient_cases,
        "drugs": np.asarray(["drug_a", "drug_b"]),
        "cond_ids": cond_ids,
        "source_cond_ids": cond_ids,
        "context_names": np.asarray(["c0", "c1", "c2"]),
    }

    with pytest.raises(ValueError, match="disagree with the Patient Cases linkage"):
        write_died_failure_matrices(
            tmp_path / "wrong-patient.npz",
            patient_ids=np.asarray(["wrong", *[f"patient-{i}" for i in range(1, 10)]]),
            **common,
        )

    patient_cases.loc[0, "patient_id"] = None
    with pytest.raises(ValueError, match="missing_links=1"):
        write_died_failure_matrices(
            tmp_path / "missing-patient.npz",
            patient_ids=np.asarray([f"patient-{i}" for i in range(10)]),
            **common,
        )
