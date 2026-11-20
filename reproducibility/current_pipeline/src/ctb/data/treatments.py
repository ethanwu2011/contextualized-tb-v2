"""Construction of the treatment matrix ``X`` (binary drug indicators for the initial regimen).

The initial regimen is the earliest non-empty regimen according to validated start fields.
``period_start`` and ``activities_period_start`` are ordered lexicographically rather than
coalesced: they are two measurements in the same day-offset/calendar coordinate, not interchangeable
fallback copies. Exact ties must contain the same drug set or selection fails closed. Drugs
administered to fewer than a prevalence threshold of the cohort are collapsed into a single
``drug_other_second_line`` indicator to control dimensionality, per the analysis plan.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ctb.data.outcomes import parse_drug_set

logger = logging.getLogger(__name__)

OTHER_DRUG_COL = "drug_other_second_line"
REGIMEN_START_COLUMNS = ("period_start", "activities_period_start")


def _normalise_drug_set(value: object) -> frozenset[str]:
    """Return a normalized immutable drug set from a parsed or raw value."""
    if isinstance(value, set | frozenset | list | tuple):
        return frozenset(str(drug).strip() for drug in value if str(drug).strip())
    return parse_drug_set(value)


def _parse_start_column(
    values: pd.Series[Any],
    column: str,
) -> tuple[pd.Series[Any], str | None, pd.Series[bool]]:
    """Parse one regimen start field and identify malformed nonmissing tokens."""
    object_values = values.astype(object)
    date_object = object_values.map(
        lambda value: isinstance(value, date | datetime | pd.Timestamp | np.datetime64)
    )
    boolean_token = object_values.map(lambda value: isinstance(value, bool | np.bool_))
    blank_token = object_values.map(lambda value: isinstance(value, str) and not value.strip())
    missing_token = object_values.isna() | blank_token
    numeric_raw = pd.to_numeric(object_values.mask(date_object), errors="coerce")
    numeric_token = numeric_raw.notna()
    numeric = numeric_raw.mask(~np.isfinite(numeric_raw) | boolean_token)
    nonintegral = numeric.notna() & numeric.mod(1).ne(0)
    numeric = numeric.mask(nonintegral)

    # Invalid numeric tokens must not fall through to ``to_datetime``, which would reinterpret
    # fractional values as epoch nanoseconds rather than rejecting them as invalid day offsets.
    date_candidates = object_values.mask(numeric_token & ~date_object)
    parsed_dates = pd.to_datetime(date_candidates, errors="coerce", format="mixed")
    invalid_token = (~missing_token & numeric.isna() & parsed_dates.isna()).astype(bool)
    has_numeric = bool(numeric.notna().any())
    has_dates = bool(parsed_dates.notna().any())
    if has_numeric and has_dates:
        raise ValueError(
            f"Regimen start column {column!r} mixes numeric day offsets and calendar dates"
        )
    if has_numeric:
        return numeric.astype("Float64"), "day_offset", invalid_token
    if has_dates:
        return parsed_dates, "calendar_date", invalid_token
    return pd.Series(pd.NA, index=values.index, dtype="Float64"), None, invalid_token


def select_initial_regimens(
    regimens: pd.DataFrame,
    *,
    start_columns: Sequence[str] = REGIMEN_START_COLUMNS,
) -> pd.DataFrame:
    """Select each condition's earliest validated non-empty regimen.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with ``condition_id`` and ``regimen_drug`` (or ``drug_set``).
    start_columns : sequence of str
        Start fields in lexicographic priority order. The default sorts first by ``period_start``
        and then by ``activities_period_start``; a missing primary value sorts after a valid one.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with ``drug_set`` and ``initial_start`` columns. Conditions
        without any non-empty regimen are retained with an empty drug set and no start. The
        aggregate (non-row-level) selection counts are available in ``DataFrame.attrs`` under
        ``"selection_report"``.

    Notes
    -----
    Empty drug rows are not regimens and are ignored. Exact ties are accepted only when their drug
    sets agree; conflicting tied sets fail closed instead of being unioned into a regimen that may
    never have been recorded. Any undated or malformed non-empty regimen row aborts selection for
    the complete input rather than silently excluding conditions according to chronology
    completeness.

    Raises
    ------
    KeyError
        If required identifier/drug columns or every configured start column are absent.
    ValueError
        If start columns mix temporal coordinate systems, any non-empty row has unsafe chronology,
        or an exact start-field tie contains conflicting drug sets.
    """
    df = regimens.copy()
    if "condition_id" not in df.columns:
        raise KeyError("Initial-regimen selection requires a 'condition_id' column")
    if "drug_set" in df.columns:
        df["_drug_set"] = df["drug_set"].map(_normalise_drug_set)
    elif "regimen_drug" in df.columns:
        df["_drug_set"] = df["regimen_drug"].map(parse_drug_set)
    else:
        raise KeyError("Initial-regimen selection requires 'drug_set' or 'regimen_drug'")

    available_starts = [column for column in start_columns if column in df.columns]
    if not available_starts:
        raise KeyError(
            "Initial-regimen selection requires at least one start column; expected one of "
            f"{list(start_columns)}"
        )
    parsed_columns: dict[str, pd.Series[Any]] = {}
    invalid_columns: dict[str, pd.Series[bool]] = {}
    coordinate_kinds: set[str] = set()
    for column in available_starts:
        parsed, coordinate_kind, invalid = _parse_start_column(df[column], column)
        parsed_columns[column] = parsed
        invalid_columns[column] = invalid
        if coordinate_kind is not None:
            coordinate_kinds.add(coordinate_kind)
    if len(coordinate_kinds) > 1:
        raise ValueError(
            "Regimen start columns use incompatible temporal coordinates: "
            f"{sorted(coordinate_kinds)}"
        )
    coordinate_kind = next(iter(coordinate_kinds), "unknown")
    parsed_starts = pd.DataFrame(parsed_columns, index=df.index)
    invalid_starts = pd.DataFrame(invalid_columns, index=df.index)
    for position, column in enumerate(available_starts):
        df[f"_start_{position}"] = parsed_starts[column]

    conflicting_start = parsed_starts.notna().sum(axis=1).gt(1) & parsed_starts.nunique(axis=1).gt(
        1
    )
    zero_drug = df["_drug_set"].map(len).eq(0)
    has_any_start = parsed_starts.notna().any(axis=1)
    undated_nonempty = ~zero_drug & ~has_any_start
    invalid_nonempty = ~zero_drug & invalid_starts.any(axis=1)
    unsafe_chronology = undated_nonempty | invalid_nonempty
    if unsafe_chronology.any():
        unsafe_rows = int(unsafe_chronology.sum())
        unsafe_conditions = int(df.loc[unsafe_chronology, "condition_id"].nunique(dropna=True))
        logger.error(
            "Initial-regimen selection aborted: %d non-empty rows across %d conditions have "
            "missing or malformed chronology",
            unsafe_rows,
            unsafe_conditions,
        )
        raise ValueError(
            "Initial-regimen selection cannot proceed: "
            f"{unsafe_rows} non-empty rows across {unsafe_conditions} conditions have missing or "
            "malformed chronology"
        )
    candidate = df.loc[~zero_drug & has_any_start].copy()

    condition_index = df.groupby("condition_id", sort=True).size().index
    selected = pd.DataFrame(
        {
            "drug_set": [frozenset() for _ in range(len(condition_index))],
            "initial_start": pd.Series([None] * len(condition_index), index=condition_index),
        },
        index=condition_index,
    )
    tied_condition_count = 0
    conflicting_tie_count = 0
    skipped_leading_zero_condition_count = 0
    if not candidate.empty:
        sort_columns = [f"_start_{position}" for position in range(len(available_starts))]
        candidate = candidate.sort_values(
            ["condition_id", *sort_columns],
            kind="mergesort",
            na_position="last",
        )
        selected_rows: list[pd.Series[Any]] = []
        for _, group in candidate.groupby("condition_id", sort=True):
            first = group.iloc[0]
            tied = pd.Series(True, index=group.index)
            for column in sort_columns:
                tied &= group[column].eq(first[column]) | (
                    group[column].isna() & pd.isna(first[column])
                )
            tied_group = group.loc[tied]
            if len(tied_group) > 1:
                tied_condition_count += 1
                if tied_group["_drug_set"].nunique() > 1:
                    conflicting_tie_count += 1
            selected_rows.append(first)

        if conflicting_tie_count:
            raise ValueError(
                "Initial-regimen selection found "
                f"{conflicting_tie_count} exact start-field tie conditions with conflicting "
                "drug sets; a prespecified clinical tie strategy is required"
            )

        selected_condition_ids = {row["condition_id"] for row in selected_rows}
        earliest_dated_rows = (
            df.loc[has_any_start]
            .sort_values(
                ["condition_id", *sort_columns],
                kind="mergesort",
                na_position="last",
            )
            .groupby("condition_id", sort=True)
            .head(1)
        )
        skipped_leading_zero_condition_count = int(
            (
                earliest_dated_rows["condition_id"].isin(selected_condition_ids)
                & earliest_dated_rows["_drug_set"].map(len).eq(0)
            ).sum()
        )

        for row in selected_rows:
            condition_id = row["condition_id"]
            selected.at[condition_id, "drug_set"] = row["_drug_set"]
            selected.at[condition_id, "initial_start"] = next(
                (row[column] for column in sort_columns if pd.notna(row[column])),
                None,
            )

    conditions_without_initial = int(selected["drug_set"].map(len).eq(0).sum())
    report = {
        "total_rows": len(df),
        "zero_drug_rows": int(zero_drug.sum()),
        "undated_nonempty_rows": int(undated_nonempty.sum()),
        "invalid_start_rows": int(invalid_nonempty.sum()),
        "conflicting_start_rows": int(conflicting_start.sum()),
        "same_start_tie_conditions": tied_condition_count,
        "conflicting_tie_conditions": conflicting_tie_count,
        "conditions_skipping_leading_zero_drug_row": skipped_leading_zero_condition_count,
        "conditions_without_initial_regimen": conditions_without_initial,
        "temporal_coordinate": coordinate_kind,
    }
    selected.attrs["selection_report"] = report
    if report["zero_drug_rows"]:
        logger.warning(
            "Initial-regimen selection ignored %d zero-drug rows", report["zero_drug_rows"]
        )
    if skipped_leading_zero_condition_count:
        logger.warning(
            "Initial-regimen selection advanced past a chronologically first zero-drug row for "
            "%d conditions",
            skipped_leading_zero_condition_count,
        )
    if report["conflicting_start_rows"]:
        logger.warning(
            "Initial-regimen selection found %d rows with conflicting start fields; "
            "used start-column priority %s",
            report["conflicting_start_rows"],
            available_starts,
        )
    if conditions_without_initial:
        logger.warning(
            "Initial-regimen selection found %d conditions without a non-empty regimen",
            conditions_without_initial,
        )
    logger.info(
        "Selected initial regimens for %d conditions (%d exact identical-set ties)",
        len(selected) - conditions_without_initial,
        tied_condition_count,
    )
    return selected


def _initial_drug_sets(regimens: pd.DataFrame) -> pd.Series[Any]:
    """Return validated initial-regimen drug sets indexed by condition."""
    return select_initial_regimens(regimens)["drug_set"]


def build_treatment_matrix(
    regimens: pd.DataFrame,
    condition_ids: Sequence[object],
    min_prevalence: float = 0.01,
) -> pd.DataFrame:
    """Build a binary treatment matrix for the given conditions.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table.
    condition_ids : sequence
        Conditions to include (defines the matrix index and the prevalence denominator).
    min_prevalence : float, default 0.01
        Drugs used by a smaller fraction of the cohort are merged into
        :data:`OTHER_DRUG_COL`.

    Returns
    -------
    pandas.DataFrame
        Binary (0/1 int) matrix indexed by ``condition_id``; one column per retained drug
        (prefixed ``drug_``) plus :data:`OTHER_DRUG_COL`.
    """
    idx: pd.Index[Any] = pd.Index(condition_ids, name="condition_id")
    initial = _initial_drug_sets(regimens).reindex(idx)
    initial = initial.map(lambda s: s if isinstance(s, frozenset) else frozenset())
    n_without_initial = int(initial.map(len).eq(0).sum())
    if n_without_initial:
        logger.warning(
            "Treatment matrix contains %d requested conditions without a validated initial "
            "regimen; their drug indicators are all zero",
            n_without_initial,
        )

    all_drugs = sorted({d for s in initial for d in s})
    sets: list[frozenset[str]] = list(initial)
    raw = pd.DataFrame(
        {drug: [int(drug in s) for s in sets] for drug in all_drugs},
        index=idx,
    )

    prevalence = raw.mean(axis=0)
    common = [d for d in all_drugs if prevalence[d] >= min_prevalence]
    rare = [d for d in all_drugs if prevalence[d] < min_prevalence]

    out = raw[common].copy()
    out.columns = pd.Index([f"drug_{c}" for c in common])
    out[OTHER_DRUG_COL] = (raw[rare].sum(axis=1) > 0).astype(int) if rare else 0
    out = out.astype(int)
    logger.info(
        "Treatment matrix: %d conditions x %d drug cols (%d common, %d rare->other)",
        out.shape[0],
        out.shape[1],
        len(common),
        len(rare),
    )
    return out
