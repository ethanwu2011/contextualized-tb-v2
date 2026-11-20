"""Tests for time-zero parsing and baseline-window safeguards."""

from __future__ import annotations

import pandas as pd
import pytest

from ctb.data.clean import clean_patient_cases
from ctb.data.temporal import (
    baseline_event_mask,
    extract_calendar_year,
    parse_temporal_coordinate,
    patient_disjoint_temporal_masks,
)


def test_extract_calendar_year_supports_integer_and_iso_like_values() -> None:
    values = pd.Series(
        [
            2019,
            "2020",
            "2021-06-15",
            pd.Timestamp("2022-12-31"),
        ],
        index=["integer", "numeric_string", "iso_date", "timestamp"],
        name="registration_date",
    )

    years = extract_calendar_year(values)

    assert years.to_dict() == {
        "integer": 2019,
        "numeric_string": 2020,
        "iso_date": 2021,
        "timestamp": 2022,
    }
    assert str(years.dtype) == "Int64"
    assert years.name == values.name


def test_extract_calendar_year_supports_datetime64_series_used_by_loader() -> None:
    values = pd.Series(pd.to_datetime(["2019-01-01", "2020-02-02", "2021-03-03", "2022-04-04"]))

    years = extract_calendar_year(values)

    assert years.tolist() == [2019, 2020, 2021, 2022]


def test_integer_registration_year_survives_cleaning_and_extracts_correctly() -> None:
    """Exercise the audited integer-year representation through the cleaner."""
    patient_cases = pd.DataFrame(
        {
            "condition_id": [1, 2, 3, 4],
            "registration_date": [2019, 2020, 2021, 2022],
        }
    )

    cleaned = clean_patient_cases(patient_cases)
    years = extract_calendar_year(cleaned.set_index("condition_id")["registration_date"])

    assert years.tolist() == [2019, 2020, 2021, 2022]


def test_temporal_split_excludes_cross_period_patients() -> None:
    index = pd.Index([1, 2, 3, 4, 5], name="condition_id")
    years = pd.Series([2020, 2021, 2019, 2022, 2022], index=index, dtype="Int64")
    patients = pd.Series(["cross", "cross", "train", "test", None], index=index)

    train, test, report = patient_disjoint_temporal_masks(years, patients)

    assert train.tolist() == [False, False, True, False, False]
    assert test.tolist() == [False, False, False, True, False]
    assert report.cross_period_patients == 1
    assert report.cross_period_records_excluded == 2
    assert report.missing_patient_records == 1


def test_extract_calendar_year_fails_closed_for_invalid_or_missing_values() -> None:
    values = pd.Series([19, 2020.5, "not-a-date", "", None, pd.NaT])

    years = extract_calendar_year(values)

    assert years.isna().all()


def test_baseline_event_mask_excludes_post_index_and_unalignable_facts() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 1, 1, 2, 3, 1],
            "observed_at": [
                "2020-01-05",  # eligible
                "2020-01-10",  # index day: eligible
                "2020-01-11",  # post-index
                "2019-01-01",  # outside 30-day lookback
                "2020-01-01",  # no index date
                "invalid",  # no event date
            ],
            "fact": ["a", "b", "c", "d", "e", "f"],
        }
    )
    index_dates = pd.Series(
        pd.to_datetime(["2020-01-10", "2020-01-10"]),
        index=pd.Index([1, 2], name="condition_id"),
    )

    mask, report = baseline_event_mask(
        events,
        index_dates,
        event_date_col="observed_at",
        lookback_days=30,
    )

    assert mask.tolist() == [True, True, False, False, False, False]
    assert report.total_events == 6
    assert report.eligible_events == 2
    assert report.post_index_events == 1
    assert report.before_lookback_events == 1
    assert report.missing_index_date == 1
    assert report.missing_event_date == 1


def test_baseline_event_mask_uses_integer_values_as_day_offsets() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 1, 1, 1, 1],
            "observed_at": [-31, -30, -1, 0, 1],
        }
    )
    index_dates = pd.Series([0], index=pd.Index([1], name="condition_id"))

    mask, report = baseline_event_mask(
        events,
        index_dates,
        event_date_col="observed_at",
        lookback_days=30,
    )

    assert mask.tolist() == [False, True, True, True, False]
    assert report.eligible_events == 3
    assert report.post_index_events == 1
    assert report.before_lookback_events == 1


def test_fractional_numeric_temporal_tokens_fail_closed() -> None:
    values = pd.Series([1.5, float("inf"), float("-inf"), True], name="observed_at")

    parsed, coordinate = parse_temporal_coordinate(values, field="observed_at")

    assert parsed.isna().all()
    assert coordinate is None


def test_baseline_event_mask_excludes_fractional_day_offsets() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 1],
            "observed_at": [-1.5, 0],
        }
    )
    index_dates = pd.Series([0], index=pd.Index([1], name="condition_id"))

    mask, report = baseline_event_mask(events, index_dates, event_date_col="observed_at")

    assert mask.tolist() == [False, True]
    assert report.eligible_events == 1
    assert report.missing_event_date == 1


def test_calendar_events_with_entirely_missing_index_fail_closed() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 2],
            "observed_at": ["2020-01-01", "2020-01-02"],
        }
    )
    index_dates = pd.Series(
        [None, None],
        index=pd.Index([1, 2], name="condition_id"),
    )

    mask, report = baseline_event_mask(events, index_dates, event_date_col="observed_at")

    assert not mask.any()
    assert report.missing_index_date == 2
    assert report.missing_event_date == 0


def test_all_missing_temporal_coordinates_with_lookback_fail_closed() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 1],
            "observed_at": [None, ""],
        }
    )
    index_dates = pd.Series([None], index=pd.Index([1], name="condition_id"))

    mask, report = baseline_event_mask(
        events,
        index_dates,
        event_date_col="observed_at",
        lookback_days=30,
    )

    assert not mask.any()
    assert report.missing_event_date == 2
    assert report.before_lookback_events == 0


@pytest.mark.parametrize(
    "lookback_days",
    [1.5, float("inf"), float("nan"), True, "30"],
)
def test_baseline_event_mask_rejects_noninteger_lookback(lookback_days: object) -> None:
    events = pd.DataFrame({"condition_id": [1], "observed_at": [0]})
    index_dates = pd.Series([0], index=pd.Index([1], name="condition_id"))

    with pytest.raises(ValueError, match="non-negative integer"):
        baseline_event_mask(
            events,
            index_dates,
            event_date_col="observed_at",
            lookback_days=lookback_days,  # type: ignore[arg-type]
        )


def test_baseline_event_mask_rejects_mixed_temporal_coordinates() -> None:
    events = pd.DataFrame(
        {
            "condition_id": [1, 1],
            "observed_at": [0, "2020-01-01"],
        }
    )
    index_dates = pd.Series([0], index=pd.Index([1], name="condition_id"))

    with pytest.raises(ValueError, match="mixes numeric day offsets and calendar dates"):
        baseline_event_mask(events, index_dates, event_date_col="observed_at")
