"""Cleaning, type coercion, and cohort construction for the TB Portals clinical tables.

This module never mutates raw files; it returns cleaned copies. Cohort inclusion is implemented as
an explicit, logged sequence of steps so that a CONSORT-style flow diagram can be reconstructed
exactly (see :func:`build_cohort`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ctb.config import RESISTANCE_EXPECTED_DAYS
from ctb.data.outcomes import parse_drug_set
from ctb.data.treatments import select_initial_regimens

logger = logging.getLogger(__name__)

# Numeric columns in patient_cases that must be coerced from object dtype.
_PC_NUMERIC = [
    "age_of_onset",
    "bmi",
    "number_of_children",
    "number_of_daily_contacts",
    "timika_score",
]
# Upper bound for plausible TB treatment duration (days). Values above are treated as data errors
# and set to NaN (≈ 6.8 years; the WHO longest XDR regimens rarely exceed ~24 months).
MAX_PLAUSIBLE_DURATION_DAYS = 2500


@dataclass
class CohortStep:
    """A single inclusion/exclusion step for the CONSORT flow.

    Attributes
    ----------
    label : str
        Human-readable description of the step.
    n_conditions : int
        Number of conditions remaining after the step.
    n_excluded : int
        Number of conditions removed by this step.
    """

    label: str
    n_conditions: int
    n_excluded: int


@dataclass
class Cohort:
    """Result of cohort construction.

    Attributes
    ----------
    condition_ids : list
        Condition IDs included in the final analysis cohort.
    steps : list of CohortStep
        Ordered inclusion steps (for the CONSORT diagram).
    """

    condition_ids: list[object]
    steps: list[CohortStep] = field(default_factory=list)


def clean_patient_cases(patient_cases: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and tidy the Patient Cases table.

    Numeric columns are coerced to floats (non-numeric tokens become NaN). Categorical columns are
    left as-is so that sentinel categories such as ``"Not Reported"`` can be handled explicitly by
    the context builder. The raw input is not modified.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Raw Patient Cases table.

    Returns
    -------
    pandas.DataFrame
        Cleaned copy.
    """
    df = patient_cases.copy()
    for col in _PC_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Preserve registration_date exactly as loaded. The audited release represents it as a
    # four-digit calendar year; generic datetime conversion would reinterpret 2020 as 2,020
    # nanoseconds after the Unix epoch. Temporal consumers parse it explicitly.
    logger.info("Cleaned patient_cases: %d rows", len(df))
    return df


def clean_regimens(regimens: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and winsorise implausible durations in the Regimens table.

    ``period_span`` values above :data:`MAX_PLAUSIBLE_DURATION_DAYS` or below 0 are set to NaN.
    A parsed ``drug_set`` column (``frozenset``) is added for convenience.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Raw Regimens table.

    Returns
    -------
    pandas.DataFrame
        Cleaned copy.
    """
    df = regimens.copy()
    df["period_span"] = pd.to_numeric(df["period_span"], errors="coerce")
    bad = (df["period_span"] < 0) | (df["period_span"] > MAX_PLAUSIBLE_DURATION_DAYS)
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning("Set %d implausible period_span values to NaN", n_bad)
        df.loc[bad, "period_span"] = np.nan
    df["drug_set"] = df["regimen_drug"].map(parse_drug_set)
    logger.info("Cleaned regimens: %d rows", len(df))
    return df


def build_cohort(patient_cases: pd.DataFrame, regimens: pd.DataFrame) -> Cohort:
    """Apply inclusion criteria and record CONSORT steps.

    Inclusion criteria (in order):

    1. All conditions in the Patient Cases table.
    2. Condition has an initial regimen with ≥1 drug (initial treatment is known).
    3. Resistance class is one of the defined classes in
       :data:`ctb.config.RESISTANCE_EXPECTED_DAYS` (excludes ``Negative``/unknown).

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Cleaned Patient Cases table.
    regimens : pandas.DataFrame
        Cleaned Regimens table. Must contain ``drug_set`` or ``regimen_drug`` and at least one of
        ``period_start`` or ``activities_period_start``; undated initial exposure is not guessed.

    Returns
    -------
    Cohort
        Final condition IDs and the ordered inclusion steps.
    """
    steps: list[CohortStep] = []
    all_ids = pd.Index(patient_cases["condition_id"].drop_duplicates())
    n0 = len(all_ids)
    steps.append(CohortStep("All TB Portals conditions", n0, 0))

    # Step 2: validated, start-ordered initial regimen with >=1 drug. This uses the same
    # deterministic selection as the treatment matrix instead of accepting a drug found only on a
    # later line.
    initial = select_initial_regimens(regimens)
    has_drug = initial["drug_set"].map(len).gt(0)
    ids_with_regimen = set(has_drug[has_drug].index)
    keep2 = [c for c in all_ids if c in ids_with_regimen]
    steps.append(
        CohortStep("With a valid-start initial regimen (\u22651 drug)", len(keep2), n0 - len(keep2))
    )

    # Step 3: defined resistance class.
    res = (
        patient_cases[["condition_id", "type_of_resistance"]]
        .drop_duplicates("condition_id")
        .set_index("condition_id")["type_of_resistance"]
    )
    defined = set(RESISTANCE_EXPECTED_DAYS)
    keep3 = [c for c in keep2 if res.get(c) in defined]
    steps.append(CohortStep("With a defined resistance class", len(keep3), len(keep2) - len(keep3)))

    for s in steps:
        logger.info("CONSORT | %-40s n=%d (excluded %d)", s.label, s.n_conditions, s.n_excluded)
    return Cohort(condition_ids=keep3, steps=steps)
