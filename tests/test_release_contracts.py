"""Tests for the row-free release contract and date parsing guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes
from ctb.data.load import _parse_registration_date, load_table


def test_registration_years_are_not_parsed_as_nanoseconds() -> None:
    """Integer calendar years remain the intended calendar years."""
    values = pd.Series([2020, 2021, 2022])
    parsed = _parse_registration_date(values)
    assert parsed.dt.year.tolist() == [2020, 2021, 2022]


def test_registration_iso_dates_are_supported() -> None:
    """ISO registration dates retain their calendar year."""
    values = pd.Series(["2020-03-01", "2021-07-15"])
    parsed = _parse_registration_date(values)
    assert parsed.dt.year.tolist() == [2020, 2021]


def test_dst_day_offsets_remain_numeric(tmp_path: Path) -> None:
    """Culture day offsets are not silently converted to Unix-epoch timestamps."""
    raw = tmp_path / "TB_Portals_DST_August_2023.csv"
    pd.DataFrame(
        {
            "condition_id": ["c1", "c1"],
            "specimen_collection_date": [0, 60],
            "test_date": [0, 60],
            "cultureresults": ["Positive", "Negative"],
        }
    ).to_csv(raw, index=False)
    loaded = load_table("dst", tmp_path)
    assert loaded["specimen_collection_date"].tolist() == [0, 60]
    assert loaded["test_date"].tolist() == [0, 60]
    assert pd.api.types.is_numeric_dtype(loaded["specimen_collection_date"])


def test_public_manifest_names_endpoint_and_smoke_command() -> None:
    """The public manifest must point to the endpoint-explicit, synthetic-only path."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["primary_endpoint"]["name"] == "died_or_failure"
    assert manifest["primary_endpoint"]["loss_to_follow_up_included"] is False
    assert manifest["synthetic_command"] == "make public-smoke"
    assert any(
        item.endswith("unfavorable_model_matrices.npz")
        for item in manifest["local_patient_level_artifacts"]
    )
    assert any(
        item.endswith("unfavorable_context_meta.parquet")
        for item in manifest["local_patient_level_artifacts"]
    )


def test_primary_matrix_loader_rejects_legacy_endpoint(tmp_path: Path) -> None:
    """The manuscript loader must reject a conversion matrix even if its shape matches."""
    path = tmp_path / "matrix.npz"
    np.savez(
        path,
        X=np.zeros((2, 1)),
        C=np.zeros((2, 1)),
        y=np.zeros(2),
        drugs=np.asarray(["drug_R"], dtype=object),
        cond_ids=np.asarray(["a", "b"], dtype=object),
        context_names=np.asarray(["age"], dtype=object),
        outcome_name=np.asarray("conversion"),
    )
    with pytest.raises(ValueError, match="Expected endpoint"):
        load_primary_matrix(path)


def test_subtype_loader_rejects_mismatched_primary_rows(tmp_path: Path) -> None:
    """Subtype labels cannot be silently paired with a different row order."""
    path = tmp_path / "subtypes.npz"
    np.savez(
        path,
        subtypes=np.asarray([0, 1]),
        loadings=np.zeros((2, 2)),
        cond_ids=np.asarray(["a", "b"], dtype=object),
        outcome_name=np.asarray("died_or_failure"),
    )
    with pytest.raises(ValueError, match="do not match"):
        load_primary_subtypes(path, expected_condition_ids=["b", "a"])
