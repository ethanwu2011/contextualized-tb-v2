"""Construction of the treatment matrix ``X`` (binary drug indicators for the initial regimen).

The initial regimen is taken as the first regimen line (in record order) for each condition, since
regimen lines are not distinctly timestamped. Drugs administered to fewer than a prevalence
threshold of the cohort are collapsed into a single ``drug_other_second_line`` indicator to control
dimensionality, per the analysis plan.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ctb.data.outcomes import parse_drug_set

logger = logging.getLogger(__name__)

OTHER_DRUG_COL = "drug_other_second_line"


def _initial_drug_sets(regimens: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """Return the initial-regimen drug set for each condition.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with ``condition_id`` and ``regimen_drug`` (or ``drug_set``).

    Returns
    -------
    pandas.Series
        Index ``condition_id``; values are ``frozenset`` of drug codes from the first line.
    """
    df = regimens.copy()
    if "drug_set" not in df.columns:
        df["drug_set"] = df["regimen_drug"].map(parse_drug_set)
    return df.groupby("condition_id", sort=True)["drug_set"].first()


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
