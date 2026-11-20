"""Time-zero utilities for temporal splits and baseline event windows.

The helpers in this module operate only on caller-supplied pandas objects.  They deliberately fail
closed: values that cannot be interpreted as a full calendar year or events that cannot be aligned
to a valid index date are not silently assigned to a temporal period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineWindowReport:
    """Aggregate accounting for a baseline-window eligibility mask.

    Counts are mutually exclusive and sum to :attr:`total_events`.  No condition identifiers or
    row-level values are retained in the report.
    """

    total_events: int
    eligible_events: int
    post_index_events: int
    before_lookback_events: int
    missing_event_date: int
    missing_index_date: int


@dataclass(frozen=True)
class TemporalSplitReport:
    """Aggregate accounting for a patient-disjoint calendar split."""

    train_records: int
    test_records: int
    cross_period_patients: int
    cross_period_records_excluded: int
    missing_year_records: int
    missing_patient_records: int


def parse_temporal_coordinate(
    values: pd.Series[Any],
    *,
    field: str,
) -> tuple[pd.Series[Any], str | None]:
    """Parse an all-numeric day-offset or all-calendar-date coordinate, never a mixture."""
    object_values = values.astype(object)
    date_object = object_values.map(
        lambda value: isinstance(value, date | datetime | pd.Timestamp | np.datetime64)
    )
    boolean_token = object_values.map(lambda value: isinstance(value, bool | np.bool_))
    numeric_raw = pd.to_numeric(object_values.mask(date_object), errors="coerce")
    numeric_token = numeric_raw.notna()
    numeric = numeric_raw.mask(~np.isfinite(numeric_raw) | boolean_token)
    nonintegral = numeric.notna() & numeric.mod(1).ne(0)
    numeric = numeric.mask(nonintegral)

    # Mask every token that pandas recognized as numeric, including fractional and non-finite
    # values rejected above. Otherwise ``to_datetime`` can silently reinterpret a value such as
    # 1.5 as 1.5 nanoseconds after the Unix epoch.
    date_candidates = object_values.mask(numeric_token & ~date_object)
    parsed_dates = pd.to_datetime(date_candidates, errors="coerce", format="mixed")
    has_numeric = bool(numeric.notna().any())
    has_dates = bool(parsed_dates.notna().any())
    if has_numeric and has_dates:
        raise ValueError(f"{field} mixes numeric day offsets and calendar dates")
    if has_numeric:
        return numeric.astype("Float64"), "day_offset"
    if has_dates:
        return parsed_dates, "calendar_date"
    return pd.Series(pd.NA, index=values.index, dtype="Float64", name=values.name), None


def extract_calendar_year(values: pd.Series[Any]) -> pd.Series[Any]:
    """Extract four-digit calendar years from integer years or ISO-like dates.

    Integer-like values in the inclusive range 1000--9999 are treated as already-normalized
    calendar years.  All other non-numeric values are parsed as dates.  Two-digit years, fractional
    numeric values, invalid dates, and missing values become ``pd.NA`` rather than being guessed.

    Parameters
    ----------
    values : pandas.Series
        Integer years, date-like values, or a mixture of both.

    Returns
    -------
    pandas.Series
        Nullable ``Int64`` years with the same index and name as ``values``.
    """
    out = pd.Series(pd.NA, index=values.index, dtype="Int64", name=values.name)
    date_object = values.map(
        lambda value: isinstance(value, date | datetime | pd.Timestamp | np.datetime64)
    )
    numeric = pd.to_numeric(values.mask(date_object), errors="coerce")
    integer_year = numeric.notna() & numeric.mod(1).eq(0) & numeric.between(1000, 9999)
    out.loc[integer_year] = numeric.loc[integer_year].astype("int64")

    # Do not reinterpret invalid numeric values (for example 19 or 2020.5) as nanosecond dates.
    date_like = (numeric.isna() | date_object) & values.notna()
    if date_like.any():
        parsed = pd.to_datetime(values.loc[date_like], errors="coerce", format="mixed")
        out.loc[date_like] = parsed.dt.year.astype("Int64")
    return out


def patient_disjoint_temporal_masks(
    years: pd.Series[Any],
    patient_ids: pd.Series[Any],
    *,
    train_through: int = 2020,
    test_from: int = 2021,
) -> tuple[pd.Series[bool], pd.Series[bool], TemporalSplitReport]:
    """Build train/test masks while excluding patients represented in both periods.

    Conditions without a valid year or patient identifier fail closed. If a patient has at least
    one condition on each side of the temporal boundary, every one of that patient's otherwise
    eligible conditions is excluded rather than leaking a patient across train and test.
    """
    if not years.index.is_unique or not patient_ids.index.is_unique:
        raise ValueError("years and patient_ids must contain at most one row per condition")
    if train_through >= test_from:
        raise ValueError("train_through must be earlier than test_from")

    patients = patient_ids.reindex(years.index)
    valid_patient = patients.notna() & patients.astype("string").str.strip().ne("")
    train_candidate = years.le(train_through).fillna(False)
    test_candidate = years.ge(test_from).fillna(False)

    train_patients = set(patients.loc[train_candidate & valid_patient])
    test_patients = set(patients.loc[test_candidate & valid_patient])
    cross_period_patients = train_patients.intersection(test_patients)
    crosses = patients.isin(cross_period_patients)

    train = (train_candidate & valid_patient & ~crosses).astype(bool)
    test = (test_candidate & valid_patient & ~crosses).astype(bool)
    remaining_overlap = set(patients.loc[train]).intersection(set(patients.loc[test]))
    if remaining_overlap:
        raise AssertionError("Patient-disjoint temporal split invariant failed")

    report = TemporalSplitReport(
        train_records=int(train.sum()),
        test_records=int(test.sum()),
        cross_period_patients=len(cross_period_patients),
        cross_period_records_excluded=int(((train_candidate | test_candidate) & crosses).sum()),
        missing_year_records=int(years.isna().sum()),
        missing_patient_records=int((~valid_patient).sum()),
    )
    train.name = "train"
    test.name = "test"
    return train, test, report


def baseline_event_mask(
    events: pd.DataFrame,
    index_dates: pd.Series[Any],
    *,
    event_date_col: str,
    id_col: str = "condition_id",
    lookback_days: int | None = None,
) -> tuple[pd.Series[bool], BaselineWindowReport]:
    """Identify events observed on or before each condition's index date.

    When ``lookback_days`` is supplied, eligible events must also fall within that many days before
    index.  Events with an invalid event date, an unavailable index date, or a post-index date are
    excluded.  The returned aggregate report makes each exclusion category explicit.

    Parameters
    ----------
    events : pandas.DataFrame
        Event-level table containing ``id_col`` and ``event_date_col``.
    index_dates : pandas.Series
        One index value per condition, indexed by ``id_col`` and expressed in the same day-offset
        or calendar-date coordinate as the events.
    event_date_col : str
        Name of the event time column. Values must use one coherent coordinate: integral day
        offsets or calendar dates.
    id_col : str, default ``"condition_id"``
        Name of the condition identifier column.
    lookback_days : int or None, default None
        Optional maximum number of days before index.  ``None`` permits any pre-index history.

    Returns
    -------
    tuple of (pandas.Series, BaselineWindowReport)
        Boolean eligibility mask aligned to ``events.index`` and aggregate exclusion accounting.

    Raises
    ------
    KeyError
        If a required event column is absent.
    ValueError
        If index values are duplicated, ``lookback_days`` is negative, or temporal coordinates are
        mixed/incompatible.
    """
    missing = {id_col, event_date_col} - set(events.columns)
    if missing:
        raise KeyError(f"Missing required event columns: {sorted(missing)}")
    if not index_dates.index.is_unique:
        raise ValueError("index_dates must contain at most one value per condition")
    if lookback_days is not None and (
        isinstance(lookback_days, bool | np.bool_)
        or not isinstance(lookback_days, Integral)
        or lookback_days < 0
    ):
        raise ValueError("lookback_days must be a non-negative integer")

    event_dates, event_coordinate = parse_temporal_coordinate(
        events[event_date_col],
        field=event_date_col,
    )
    parsed_index, index_coordinate = parse_temporal_coordinate(
        index_dates,
        field="index_dates",
    )
    if (
        event_coordinate is not None
        and index_coordinate is not None
        and event_coordinate != index_coordinate
    ):
        raise ValueError(
            "Baseline event and index coordinates are incompatible: "
            f"{event_coordinate} versus {index_coordinate}"
        )
    coordinate = event_coordinate or index_coordinate
    # An entirely missing side has no inferable coordinate and is represented as nullable Float64.
    # Once the other side establishes a calendar coordinate, replace that sentinel with a
    # type-compatible all-NaT series before pandas comparisons. Otherwise an all-missing index (or
    # event side) can raise instead of failing closed.
    if coordinate == "calendar_date":
        if event_coordinate is None:
            event_dates = pd.Series(
                pd.NaT,
                index=events.index,
                dtype="datetime64[ns]",
                name=event_dates.name,
            )
        if index_coordinate is None:
            parsed_index = pd.Series(
                pd.NaT,
                index=index_dates.index,
                dtype="datetime64[ns]",
                name=parsed_index.name,
            )
    aligned_index = events[id_col].map(parsed_index)

    missing_event = event_dates.isna()
    missing_index = event_dates.notna() & aligned_index.isna()
    comparable = event_dates.notna() & aligned_index.notna()
    post_index = comparable & event_dates.gt(aligned_index)
    eligible = comparable & event_dates.le(aligned_index)

    before_lookback = pd.Series(False, index=events.index)
    if lookback_days is not None and bool(comparable.any()):
        if coordinate is None:
            raise AssertionError("Comparable temporal values require a known coordinate")
        earliest = (
            aligned_index - lookback_days
            if coordinate == "day_offset"
            else aligned_index - pd.to_timedelta(lookback_days, unit="D")
        )
        before_lookback = eligible & event_dates.lt(earliest)
        eligible &= ~before_lookback

    eligible = eligible.astype(bool)
    eligible.name = "baseline_eligible"
    report = BaselineWindowReport(
        total_events=len(events),
        eligible_events=int(eligible.sum()),
        post_index_events=int(post_index.sum()),
        before_lookback_events=int(before_lookback.sum()),
        missing_event_date=int(missing_event.sum()),
        missing_index_date=int(missing_index.sum()),
    )
    return eligible, report
