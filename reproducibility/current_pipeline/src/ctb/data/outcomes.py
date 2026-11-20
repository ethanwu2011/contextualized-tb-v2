"""Derivation of the *treatment-intensification* outcome from the TB Portals regimen timeline.

The composite outcome flags episodes whose initial regimen proved insufficient. See
``docs/OUTCOME_DEFINITION.md`` for the full specification and the rationale behind each design
decision. The archived regimen table does not provide sequentially dated phases, so row-order-based
drug-set growth is disabled; only an explicit status-reason signal is executable.

Primary composite (locked decisions):

``treatment_intensification = escalation OR extension OR re_initiation``

where

* **escalation** — any line carries ``activities_statusreason_cd`` in
  :data:`ctb.config.ESCALATION_STATUSREASONS`. Drug-set growth is unassessable without a validated
  sequential regimen chronology and is retained as a nullable audit column only.
* **extension** — total treatment duration (``period_span``) exceeds
  :data:`ctb.config.EXTENSION_THRESHOLD` × the class-expected duration
  (:data:`ctb.config.RESISTANCE_EXPECTED_DAYS`).
* **re_initiation** — when episode starts and prior ends share one day-resolved coordinate, a
  subsequent episode begins within :data:`ctb.config.REINITIATION_WINDOW_DAYS` days of the prior
  episode's end. The audited year-only registration field is not day-resolved, so this component is
  unassessable rather than guessed for that schema.

``unfavorable_outcome`` (WHO died/failure/LTFU) is computed and returned for validation but is
**excluded** from the primary composite, to keep the outcome distinct from mortality.

All functions are pure with respect to their inputs (no raw files are read or written here) so that
they can be unit-tested against small synthetic frames.
"""

from __future__ import annotations

import logging
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from ctb.config import (
    DIED_FAILURE_NONEVENT_OUTCOMES,
    DIED_FAILURE_OUTCOMES,
    ESCALATION_STATUSREASONS,
    EXTENSION_THRESHOLD,
    REINITIATION_WINDOW_DAYS,
    RESISTANCE_EXPECTED_DAYS,
    UNFAVORABLE_OUTCOMES,
)
from ctb.data.temporal import parse_temporal_coordinate

logger = logging.getLogger(__name__)


def require_modelable_binary_outcome(
    values: pd.Series[Any],
    *,
    outcome_name: str,
    min_class_count: int = 1,
) -> pd.Series[int]:
    """Validate a binary outcome before any model or figure is fit.

    Missing labels are not silently dropped or filled because doing so changes the estimand and can
    convert an unassessable endpoint into an apparently valid negative class. Non-binary values and
    single-class cohorts are likewise rejected before any downstream artifacts are written.

    Parameters
    ----------
    values : pandas.Series
        Candidate outcome values.
    outcome_name : str
        Human-readable column name included in aggregate-only error messages.
    min_class_count : int, default 1
        Minimum required observations in each class. Five-fold callers should pass ``5``.

    Returns
    -------
    pandas.Series of int
        The validated 0/1 outcome, preserving the input index.

    Raises
    ------
    ValueError
        If any value is missing or invalid, or if both binary classes are not present.
    """
    if (
        isinstance(min_class_count, bool)
        or not isinstance(min_class_count, Integral)
        or min_class_count < 1
    ):
        raise ValueError("min_class_count must be a positive non-Boolean integer")

    is_supported_dtype = bool(
        pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_integer_dtype(values.dtype)
        or pd.api.types.is_float_dtype(values.dtype)
    )
    present = values.notna()
    missing = int((~present).sum())
    if not is_supported_dtype:
        invalid = int(present.sum())
        raise ValueError(
            f"Outcome {outcome_name!r} is not modelable: total={len(values)}, assessable=0, "
            f"missing={missing}, invalid={invalid}, classes=[], dtype={values.dtype}. "
            "Labels must use a Boolean, integer, or exact binary floating dtype; strings, bytes, "
            "objects, timedeltas, and complex values are not coerced."
        )

    numeric = values.astype("Float64")
    invalid_mask = present & ~numeric.isin([0.0, 1.0])
    assessable_mask = present & ~invalid_mask

    invalid = int(invalid_mask.sum())
    assessable = int(assessable_mask.sum())
    classes = sorted({int(value) for value in numeric.loc[assessable_mask].tolist()})
    class_counts = {
        binary_class: int((numeric.loc[assessable_mask] == binary_class).sum())
        for binary_class in (0, 1)
    }

    if missing or invalid or classes != [0, 1] or min(class_counts.values()) < min_class_count:
        raise ValueError(
            f"Outcome {outcome_name!r} is not modelable: total={len(values)}, "
            f"assessable={assessable}, missing={missing}, invalid={invalid}, classes={classes}. "
            f"class_counts={class_counts}, required_per_class={min_class_count}. "
            "Do not fill or drop unassessable labels; use a prespecified endpoint and cohort "
            "containing both classes."
        )
    return values.astype(int)


def parse_drug_set(value: object) -> frozenset[str]:
    """Parse a TB Portals ``regimen_drug`` set-string into a set of drug codes.

    The source encodes a regimen line as a brace-wrapped, comma-separated set, e.g.
    ``"{Bdq,Cfz,Cs,Lfx,Lzd}"``. Whitespace is stripped; an empty or missing value yields an empty
    set.

    Parameters
    ----------
    value : object
        The raw cell value (typically ``str`` or ``float`` NaN).

    Returns
    -------
    frozenset of str
        The set of drug codes; empty if the value is missing or ``"{}"``.
    """
    if not isinstance(value, str):
        return frozenset()
    inner = value.strip().strip("{}").strip()
    if not inner:
        return frozenset()
    return frozenset(tok.strip() for tok in inner.split(",") if tok.strip())


def derive_escalation(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive the assessable escalation signal per condition.

    A condition is flagged only when any line's ``activities_statusreason_cd`` explicitly indicates
    a change consistent with escalation. Drug-set growth is not derived because the audited regimen
    rows do not encode a validated sequence; treating source-row order as time makes the endpoint
    change when identical input rows are shuffled.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with at least ``condition_id``, ``regimen_drug``, and
        ``activities_statusreason_cd`` columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with boolean columns ``escalation`` and
        ``escalation_statusreason`` plus nullable audit columns ``escalation_drugset`` and
        ``escalation_key_drug``. Aggregate assessability metadata is stored in
        ``DataFrame.attrs["escalation_report"]``.
    """
    df = regimens.copy()
    sr = df.get("activities_statusreason_cd", pd.Series(index=df.index, dtype=object))

    records: list[dict[str, object]] = []
    for cid, grp in df.groupby("condition_id", sort=True):
        reasons = sr.loc[grp.index].dropna().astype(str)
        sr_flag = bool(reasons.isin(ESCALATION_STATUSREASONS).any())
        records.append(
            {
                "condition_id": cid,
                "escalation_drugset": pd.NA,
                "escalation_key_drug": pd.NA,
                "escalation_statusreason": sr_flag,
                "escalation": sr_flag,
            }
        )
    if records:
        out = pd.DataFrame.from_records(records).set_index("condition_id")
    else:
        out = pd.DataFrame(
            columns=[
                "escalation_drugset",
                "escalation_key_drug",
                "escalation_statusreason",
                "escalation",
            ],
            index=pd.Index([], name="condition_id"),
        )
    for col in ("escalation_drugset", "escalation_key_drug"):
        out[col] = out[col].astype("boolean")
    for col in ("escalation_statusreason", "escalation"):
        out[col] = out[col].astype(bool)
    out.attrs["escalation_report"] = {
        "drugset_growth_assessable": False,
        "reason": "no validated sequential chronology for regimen rows",
        "statusreason_flagged_conditions": int(out["escalation_statusreason"].sum()),
    }
    logger.warning(
        "Drug-set escalation disabled for %d conditions because regimen rows lack validated "
        "sequential chronology; explicit status reasons remain assessable",
        len(out),
    )
    logger.info(
        "Escalation derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["escalation"].mean() if len(out) else 0.0,
    )
    return out


def _episode_duration_days(regimens: pd.DataFrame) -> pd.Series[float]:
    """Compute treatment duration (days) per condition from ``period_span``.

    Uses the maximum ``period_span`` across an episode's regimen lines as the episode duration,
    since lines share the episode start date and ``period_span`` carries per-line length.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with ``condition_id`` and ``period_span``.

    Returns
    -------
    pandas.Series
        Duration in days indexed by ``condition_id`` (NaN where ``period_span`` is missing).
    """
    return regimens.groupby("condition_id", sort=True)["period_span"].max()


def derive_extension(regimens: pd.DataFrame, patient_cases: pd.DataFrame) -> pd.DataFrame:
    """Derive the treatment-extension component per condition.

    A condition is flagged when its duration exceeds ``EXTENSION_THRESHOLD`` times the expected
    duration for its resistance class. Conditions whose resistance class has no defined expected
    duration are not flagged (``extension = False``, ``expected_days = NaN``).

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``period_span``).
    patient_cases : pandas.DataFrame
        Patient Cases table (``condition_id``, ``type_of_resistance``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with columns ``duration_days`` (float),
        ``expected_days`` (float), and ``extension`` (bool).
    """
    duration = _episode_duration_days(regimens).rename("duration_days")
    res = (
        patient_cases[["condition_id", "type_of_resistance"]]
        .drop_duplicates("condition_id")
        .set_index("condition_id")["type_of_resistance"]
    )
    expected = res.map(RESISTANCE_EXPECTED_DAYS).rename("expected_days")

    out = pd.concat([duration, expected], axis=1)
    out["expected_days"] = out["expected_days"].astype("float64")
    out["duration_days"] = out["duration_days"].astype("float64")
    threshold = EXTENSION_THRESHOLD * out["expected_days"]
    out["extension"] = (out["duration_days"] > threshold).fillna(False)
    logger.info(
        "Extension derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["extension"].mean() if len(out) else 0.0,
    )
    return out


def derive_reinitiation(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    window_days: int = REINITIATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Derive the treatment re-initiation component per condition.

    When starts and ends share one day-resolved coordinate, patients' episodes are ordered by
    ``registration_date``. A non-first episode is flagged when it begins within ``window_days``
    days after the prior episode's latest ``period_end``. Concurrent/overlapping episodes (negative
    gap) and gaps beyond the window are not flagged. A year-only start or incompatible coordinate
    disables the complete derivation and records that decision in ``DataFrame.attrs``.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Patient Cases table (``patient_id``, ``condition_id``, ``registration_date``).
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``period_end``).
    window_days : int, default :data:`ctb.config.REINITIATION_WINDOW_DAYS`
        Maximum gap (days) between prior end and new start to count as re-initiation.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a nullable boolean ``re_initiation`` column. Incompatible
        coordinates produce ``pd.NA`` rather than false negatives.
    """
    if (
        isinstance(window_days, bool | np.bool_)
        or not isinstance(window_days, Integral)
        or window_days < 0
    ):
        raise ValueError("window_days must be a non-negative integer")

    pc = (
        patient_cases[["patient_id", "condition_id", "registration_date"]]
        .drop_duplicates("condition_id")
        .copy()
    )
    idx = pc.set_index("condition_id").index

    # The audited release stores ``registration_date`` as a four-digit calendar year. A year has
    # insufficient resolution for a day-gap calculation and must never be converted to epoch
    # nanoseconds. If any year-only token is present, the complete derivation is marked
    # unassessable rather than mixing it with regimen day offsets or guessing January 1.
    registration_raw = pc["registration_date"].astype(object)
    registration_numeric = pd.to_numeric(registration_raw, errors="coerce")
    registration_boolean = registration_raw.map(lambda value: isinstance(value, bool | np.bool_))
    year_only = (
        registration_numeric.notna()
        & np.isfinite(registration_numeric)
        & registration_numeric.mod(1).eq(0)
        & registration_numeric.between(1000, 9999)
        & ~registration_boolean
    )
    if year_only.any():
        registration_coordinate = "calendar_year"
        parsed_registration = pd.Series(pd.NA, index=pc.index, dtype="Float64")
    else:
        parsed_registration, registration_coordinate = parse_temporal_coordinate(
            pc["registration_date"],
            field="registration_date",
        )

    parsed_end, end_coordinate = parse_temporal_coordinate(
        regimens["period_end"],
        field="period_end",
    )
    coordinates_compatible = (
        registration_coordinate is not None
        and end_coordinate is not None
        and registration_coordinate == end_coordinate
        and registration_coordinate != "calendar_year"
    )
    if not coordinates_compatible:
        out = pd.DataFrame(
            {"re_initiation": pd.Series(pd.NA, index=idx, dtype="boolean")},
            index=idx,
        )
        out.attrs["reinitiation_report"] = {
            "coordinate_compatible": False,
            "registration_coordinate": registration_coordinate,
            "period_end_coordinate": end_coordinate,
            "assessable_transitions": 0,
            "unassessable_transitions": 0,
            "unassessable_conditions": len(idx),
            "flagged_conditions": 0,
        }
        logger.warning(
            "Re-initiation disabled: registration coordinate %r is incompatible with period-end "
            "coordinate %r; no day gaps were guessed",
            registration_coordinate,
            end_coordinate,
        )
        return out

    regimen_times = regimens[["condition_id"]].copy()
    regimen_times["_period_end"] = parsed_end
    end_by_cond = regimen_times.groupby("condition_id", sort=True)["_period_end"].max()
    pc["_registration"] = parsed_registration
    pc["prior_end"] = pc["condition_id"].map(end_by_cond)

    flagged: set[object] = set()
    unassessable: set[object] = set(
        pc.loc[
            pc["patient_id"].isna() | pc["patient_id"].astype("string").str.strip().eq(""),
            "condition_id",
        ]
    )
    assessable_transitions = 0
    unassessable_transitions = 0
    for _, grp in pc.groupby("patient_id", sort=True):
        if len(grp) < 2:
            continue
        if grp["_registration"].isna().any() or grp["_registration"].duplicated().any():
            unassessable.update(grp["condition_id"])
            unassessable_transitions += len(grp) - 1
            continue
        grp = grp.sort_values("_registration", kind="mergesort")
        prior_end = grp["prior_end"].to_numpy()
        starts = grp["_registration"].to_numpy()
        cids = grp["condition_id"].to_numpy()
        for i in range(1, len(grp)):
            if pd.isna(prior_end[i - 1]) or pd.isna(starts[i]):
                unassessable_transitions += 1
                unassessable.add(cids[i])
                continue
            assessable_transitions += 1
            gap = (
                (pd.Timestamp(starts[i]) - pd.Timestamp(prior_end[i - 1])).days
                if registration_coordinate == "calendar_date"
                else float(starts[i]) - float(prior_end[i - 1])
            )
            if 0 <= gap <= window_days:
                flagged.add(cids[i])

    reinitiation = pd.Series(False, index=idx, dtype="boolean", name="re_initiation")
    reinitiation.loc[reinitiation.index.isin(unassessable)] = pd.NA
    reinitiation.loc[reinitiation.index.isin(flagged)] = True
    out = reinitiation.to_frame()
    out.attrs["reinitiation_report"] = {
        "coordinate_compatible": True,
        "registration_coordinate": registration_coordinate,
        "period_end_coordinate": end_coordinate,
        "assessable_transitions": assessable_transitions,
        "unassessable_transitions": unassessable_transitions,
        "unassessable_conditions": len(unassessable),
        "flagged_conditions": len(flagged),
    }
    reinitiation_mean = out["re_initiation"].mean()
    logger.info(
        "Re-initiation derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * float(reinitiation_mean) if pd.notna(reinitiation_mean) else float("nan"),
    )
    return out


def derive_patient_case_died_failure(patient_cases: pd.DataFrame) -> pd.DataFrame:
    """Derive the recorded final died/failure endpoint from Patient Cases.

    The audited Patient Cases release contains one final ``outcome`` value per condition. ``Died``
    and ``Failure`` are events. ``Cured`` and ``Completed`` are observed non-events. Every other
    value remains unassessable. Condition identifiers must be present and unique because duplicate
    rows would make the meaning of a single final outcome ambiguous.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Patient Cases table with ``condition_id`` and final ``outcome`` columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with one nullable boolean ``died_failure`` column. Aggregate
        assessability counts are stored in ``DataFrame.attrs["died_failure_report"]``.

    Raises
    ------
    KeyError
        If a required source column is absent.
    ValueError
        If condition identifiers are missing or duplicated.
    """
    required = {"condition_id", "outcome"}
    missing_columns = sorted(required.difference(patient_cases.columns))
    if missing_columns:
        raise KeyError(f"Patient Cases died/failure derivation requires columns {missing_columns}")

    missing_condition_ids = int(patient_cases["condition_id"].isna().sum())
    duplicated_condition_rows = patient_cases["condition_id"].duplicated(keep=False)
    duplicated_conditions = int(
        patient_cases.loc[duplicated_condition_rows, "condition_id"].nunique(dropna=True)
    )
    if missing_condition_ids or duplicated_conditions:
        raise ValueError(
            "Patient Cases final-outcome derivation requires one row per condition: "
            f"missing_condition_ids={missing_condition_ids}, "
            f"duplicated_conditions={duplicated_conditions}"
        )

    normalized = (
        patient_cases["outcome"].astype("string").str.strip().str.casefold().replace("", pd.NA)
    )
    labels = pd.Series(pd.NA, index=patient_cases.index, dtype="boolean")
    labels.loc[normalized.isin(DIED_FAILURE_OUTCOMES)] = True
    labels.loc[normalized.isin(DIED_FAILURE_NONEVENT_OUTCOMES)] = False

    out = pd.DataFrame(
        {"died_failure": labels.to_numpy()},
        index=pd.Index(patient_cases["condition_id"], name="condition_id"),
    )
    out["died_failure"] = out["died_failure"].astype("boolean")
    n_events = int(out["died_failure"].sum(skipna=True))
    n_assessable = int(out["died_failure"].notna().sum())
    n_unassessable = int(out["died_failure"].isna().sum())
    out.attrs["died_failure_report"] = {
        "source_table": "patient_cases",
        "source_column": "outcome",
        "n_conditions": len(out),
        "n_assessable": n_assessable,
        "n_events": n_events,
        "n_observed_nonevents": n_assessable - n_events,
        "n_unassessable": n_unassessable,
        "event_values": ["died", "failure"],
        "observed_nonevent_values": ["completed", "cured"],
    }
    logger.info(
        "Patient Cases final died/failure endpoint derived for %d conditions: assessable=%d, "
        "events=%d, unassessable=%d (%.1f%% positive among assessable)",
        len(out),
        n_assessable,
        n_events,
        n_unassessable,
        100 * n_events / n_assessable if n_assessable else float("nan"),
    )
    return out


def derive_died_failure(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive a regimen-history died/failure label for source concordance checks.

    A condition is an event only if every nonempty recorded state belongs to
    :data:`ctb.config.DIED_FAILURE_OUTCOMES` (``died`` or ``failure``). It is an observed non-event
    only if every nonempty recorded state belongs to
    :data:`ctb.config.DIED_FAILURE_NONEVENT_OUTCOMES` (``cured`` or ``completed``). Loss to
    follow-up, missing, unknown, ongoing, palliative-care, and mixed event/non-event or
    event/unknown states remain unassessable rather than being given event precedence or treated as
    negatives. This is retained as an independent concordance source. The primary PATCH endpoint is
    derived from the Patient Cases final ``outcome`` field by
    :func:`derive_patient_case_died_failure`.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``outcome_cd``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with one nullable boolean ``died_failure`` column. Aggregate
        assessability counts are stored in ``DataFrame.attrs["died_failure_report"]``.
    """

    def classify(values: pd.Series[Any]) -> object:
        observed = values.dropna().astype(str).str.strip()
        observed = observed[observed.ne("")]
        if len(observed) and observed.isin(DIED_FAILURE_OUTCOMES).all():
            return True
        if len(observed) and observed.isin(DIED_FAILURE_NONEVENT_OUTCOMES).all():
            return False
        return pd.NA

    grp = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    flag = grp.agg(classify).astype("boolean")
    out = flag.rename("died_failure").to_frame()
    n_events = int(out["died_failure"].sum(skipna=True))
    n_assessable = int(out["died_failure"].notna().sum())
    n_unassessable = int(out["died_failure"].isna().sum())
    out.attrs["died_failure_report"] = {
        "n_conditions": len(out),
        "n_assessable": n_assessable,
        "n_events": n_events,
        "n_observed_nonevents": n_assessable - n_events,
        "n_unassessable": n_unassessable,
        "event_codes": sorted(DIED_FAILURE_OUTCOMES),
        "observed_nonevent_codes": sorted(DIED_FAILURE_NONEVENT_OUTCOMES),
    }
    logger.info(
        "Died/failure endpoint derived for %d conditions: assessable=%d, events=%d, "
        "unassessable=%d (%.1f%% positive among assessable)",
        len(out),
        n_assessable,
        n_events,
        n_unassessable,
        100 * n_events / n_assessable if n_assessable else float("nan"),
    )
    return out


def derive_unfavorable(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive the WHO unfavorable-outcome flag per condition (validation column).

    A condition is flagged if any regimen line records an ``outcome_cd`` in
    :data:`ctb.config.UNFAVORABLE_OUTCOMES` (died/failure/LTFU).

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``outcome_cd``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a single boolean column ``unfavorable_outcome``.
    """
    grp = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    flag = grp.agg(lambda s: bool(s.dropna().astype(str).isin(UNFAVORABLE_OUTCOMES).any()))
    out = flag.rename("unfavorable_outcome").to_frame()
    logger.info(
        "Unfavorable outcome derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["unfavorable_outcome"].mean() if len(out) else 0.0,
    )
    return out


def derive_mortality(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive an all-cause mortality flag per condition.

    A condition is flagged if any regimen line records ``outcome_cd == "died"``. This reproduces the
    outcome used in the ML4H 2024 contextualized-TB analysis (X = recorded treatments,
    Y = mortality). It is an alternative observed endpoint that is separate from the archived
    regimen-derived intensification label. It does not identify a treatment effect.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``outcome_cd``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a single boolean column ``mortality``.
    """
    grp = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    flag = grp.agg(lambda s: bool(s.dropna().astype(str).eq("died").any()))
    out = flag.rename("mortality").to_frame()
    logger.info(
        "Mortality derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["mortality"].mean() if len(out) else 0.0,
    )
    return out


def derive_treatment_intensification(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    window_days: int = REINITIATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Assemble the full per-condition treatment-intensification outcome frame.

    Combines the escalation, extension, re-initiation, and unfavorable components. The primary
    composite ``treatment_intensification`` is the logical OR of escalation, extension, and
    re-initiation (unfavorable is excluded). A secondary
    ``treatment_intensification_with_unfavorable`` column additionally ORs in the unfavorable flag.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Patient Cases table.
    regimens : pandas.DataFrame
        Regimens table.
    window_days : int, default :data:`ctb.config.REINITIATION_WINDOW_DAYS`
        Re-initiation window passed to :func:`derive_reinitiation`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with all component and composite columns. Every condition in
        ``patient_cases`` is represented. Unassessable components remain nullable and propagate
        through composites under three-valued boolean logic.
    """
    base_idx = pd.Index(patient_cases["condition_id"].drop_duplicates(), name="condition_id")

    esc = derive_escalation(regimens)
    ext = derive_extension(regimens, patient_cases)
    rei = derive_reinitiation(patient_cases, regimens, window_days=window_days)
    unf = derive_unfavorable(regimens)

    out = pd.DataFrame(index=base_idx)
    out = out.join(esc).join(ext).join(rei).join(unf)

    bool_cols = [
        "escalation_statusreason",
        "escalation",
        "extension",
        "unfavorable_outcome",
    ]
    for col in bool_cols:
        out[col] = out[col].where(out[col].notna(), other=False).astype(bool)
    for col in ("escalation_drugset", "escalation_key_drug", "re_initiation"):
        out[col] = out[col].astype("boolean")

    out["treatment_intensification"] = out["escalation"] | out["extension"] | out["re_initiation"]
    out["treatment_intensification_with_unfavorable"] = (
        out["treatment_intensification"] | out["unfavorable_outcome"]
    )
    out.attrs["escalation_report"] = esc.attrs.get("escalation_report", {})
    out.attrs["reinitiation_report"] = rei.attrs.get("reinitiation_report", {})
    primary_mean = out["treatment_intensification"].mean()
    secondary_mean = out["treatment_intensification_with_unfavorable"].mean()
    logger.info(
        "Composite treatment_intensification: %.1f%% positive (primary), %.1f%% (with unfavorable)",
        100 * float(primary_mean) if pd.notna(primary_mean) else float("nan"),
        100 * float(secondary_mean) if pd.notna(secondary_mean) else float("nan"),
    )
    return out


def outcome_prevalence_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarise prevalence of each component and composite.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        Output of :func:`derive_treatment_intensification`.

    Returns
    -------
    pandas.DataFrame
        Columns ``component``, ``n_positive``, ``n_total``, ``prevalence`` (one row per flag).
    """
    cols = [
        "escalation",
        "escalation_drugset",
        "escalation_key_drug",
        "escalation_statusreason",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
        "treatment_intensification",
        "treatment_intensification_with_unfavorable",
    ]
    rows = [
        {
            "component": c,
            "n_positive": int(outcomes[c].sum(skipna=True)),
            "n_total": int(outcomes[c].notna().sum()),
            "prevalence": (
                float(outcomes[c].mean()) if bool(outcomes[c].notna().any()) else float("nan")
            ),
        }
        for c in cols
        if c in outcomes.columns
    ]
    return pd.DataFrame(rows)
