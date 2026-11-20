"""Unit tests for culture-conversion endpoint derivation (synthetic, schema-matching)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctb.data import conversion as cv


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Negative", "negative"),
        ("Positive", "positive"),
        ("1 to 19", "positive"),
        ("20 to 100", "positive"),
        ("100 to 200", "positive"),
        ("More than 200", "positive"),
        ("MOTT", "unknown"),
        ("Nonspecific microflora", "unknown"),
        ("Unknown result", "unknown"),
        ("Study in progress", "unknown"),
        (np.nan, "unknown"),
        (42, "unknown"),
    ],
)
def test_classify_culture(value: object, expected: str) -> None:
    assert cv.classify_culture(value) == expected


def _dst(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["condition_id", "specimen_collection_date", "cultureresults"]
    )


def test_simple_conversion() -> None:
    # positive at day 0, negative at day 60 (sustained) -> event=1, ttc=60, day60=1
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Negative"},
        ]
    )
    out = cv.derive_conversion(dst)
    r = out.loc[1]
    assert r["event"] == 1
    assert r["time_to_event"] == 60.0
    assert r["conversion_day"] == 60.0
    assert r["day60_status"] == 1  # negative in [45,75] window


def test_never_converts_is_censored() -> None:
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 90, "cultureresults": "100 to 200"},
        ]
    )
    out = cv.derive_conversion(dst)
    r = out.loc[1]
    assert r["event"] == 0
    assert r["time_to_event"] == 90.0  # censored at last culture
    assert np.isnan(r["conversion_day"])


def test_relapse_breaks_sustained_conversion() -> None:
    # pos -> neg -> pos again: not sustained, so censored (event=0)
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 50, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 100, "cultureresults": "Positive"},
        ]
    )
    out_sustained = cv.derive_conversion(dst, cv.ConversionConfig(require_sustained=True))
    assert out_sustained.loc[1, "event"] == 0  # relapse breaks it
    out_first = cv.derive_conversion(dst, cv.ConversionConfig(require_sustained=False))
    assert out_first.loc[1, "event"] == 1  # first-negative definition counts it


def test_later_negative_sequence_can_establish_sustained_conversion() -> None:
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 30, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 90, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 120, "cultureresults": "Negative"},
        ]
    )

    out = cv.derive_conversion(dst, cv.ConversionConfig(require_sustained=True))

    assert out.loc[1, "event"] == 1
    assert out.loc[1, "conversion_day"] == 90
    assert out.loc[1, "time_to_event"] == 90


def test_day60_window_selection() -> None:
    # cultures at 40 (out), 50 (in, negative), 80 (out): day60 picks day-50 negative
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 40, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 50, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 80, "cultureresults": "Positive"},
        ]
    )
    out = cv.derive_conversion(dst)
    assert out.loc[1, "day60_status"] == 1


def test_no_window_culture_is_nan() -> None:
    # cultures only at 0 and 120 -> nothing in [45,75] -> day60_status NaN
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 120, "cultureresults": "Negative"},
        ]
    )
    out = cv.derive_conversion(dst)
    assert np.isnan(out.loc[1, "day60_status"])
    assert out.loc[1, "event"] == 1  # but time-to-event still defined


def test_never_positive_not_assessable() -> None:
    # only negative cultures: not in conversion-estimable population
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Negative"},
        ]
    )
    out = cv.derive_conversion(dst)
    assert np.isnan(out.loc[1, "event"])
    assert np.isnan(out.loc[1, "first_pos_day"])


def test_unknown_cultures_dropped() -> None:
    # MOTT / unknown rows ignored; conversion computed on interpretable ones only
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 30, "cultureresults": "MOTT"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Negative"},
        ]
    )
    out = cv.derive_conversion(dst)
    assert out.loc[1, "n_cultures"] == 2  # MOTT dropped
    assert out.loc[1, "event"] == 1


def test_fractional_nonfinite_and_boolean_days_are_excluded_and_counted() -> None:
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 1.5, "cultureresults": "Negative"},
            {
                "condition_id": 1,
                "specimen_collection_date": float("inf"),
                "cultureresults": "Negative",
            },
            {"condition_id": 1, "specimen_collection_date": True, "cultureresults": "Negative"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Negative"},
        ]
    )

    out = cv.derive_conversion(dst)

    assert out.loc[1, "n_cultures"] == 2
    assert out.loc[1, "conversion_day"] == 60
    assert out.attrs["date_report"]["invalid_day_rows"] == 3


def test_all_invalid_culture_days_return_an_empty_audited_result() -> None:
    dst = _dst(
        [
            {
                "condition_id": 1,
                "specimen_collection_date": 1.5,
                "cultureresults": "Positive",
            }
        ]
    )

    out = cv.derive_conversion(dst)

    assert out.empty
    assert out.index.name == "condition_id"
    assert out.attrs["date_report"]["invalid_day_rows"] == 1


def test_conflicting_same_day_cultures_quarantine_condition_independent_of_row_order() -> None:
    dst = _dst(
        [
            {"condition_id": 1, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Positive"},
            {"condition_id": 1, "specimen_collection_date": 60, "cultureresults": "Negative"},
            {"condition_id": 2, "specimen_collection_date": 0, "cultureresults": "Positive"},
            {"condition_id": 2, "specimen_collection_date": 60, "cultureresults": "Negative"},
        ]
    )

    expected = cv.derive_conversion(dst)
    reversed_result = cv.derive_conversion(dst.iloc[::-1].reset_index(drop=True))

    pd.testing.assert_frame_equal(reversed_result, expected)
    assert expected.index.tolist() == [2]
    report = expected.attrs["date_report"]
    assert report["conflicting_same_day_groups"] == 1
    assert report["conditions_quarantined_for_same_day_conflict"] == 1
