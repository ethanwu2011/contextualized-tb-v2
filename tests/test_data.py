"""Unit tests for :mod:`ctb.data.clean` and :mod:`ctb.data.treatments` (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctb.data.clean import (
    MAX_PLAUSIBLE_DURATION_DAYS,
    build_cohort,
    clean_patient_cases,
    clean_regimens,
)
from ctb.data.treatments import OTHER_DRUG_COL, build_treatment_matrix


def _pc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": [1, 2, 3, 4],
            "patient_id": ["a", "b", "c", "d"],
            "age_of_onset": ["30", "45", "x", "50"],  # 'x' -> NaN after coercion
            "bmi": ["20.1", "22", "", "18"],
            "type_of_resistance": ["Sensitive", "MDR non XDR", "Negative", "XDR"],
            "registration_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            "number_of_children": [0, 1, 2, 3],
            "number_of_daily_contacts": [1, 2, 3, 4],
            "timika_score": [5, 6, 7, 8],
        }
    )


def _reg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": [1, 2, 3, 4],
            "patient_id": ["a", "b", "c", "d"],
            "regimen_drug": ["{E,H,R,Z}", "{Bdq,Lzd}", "{}", "{Bdq,Cfz,Lzd}"],
            "period_span": [180, 540, 200, 99999],  # last is implausible
            "period_start": pd.to_datetime(["2020-01-01"] * 4),
            "period_end": pd.to_datetime(["2020-07-01"] * 4),
            "outcome_cd": ["cured", "cured", "cured", "cured"],
            "activities_statusreason_cd": [np.nan, np.nan, np.nan, np.nan],
        }
    )


def test_clean_patient_cases_coercion() -> None:
    out = clean_patient_cases(_pc())
    assert out["age_of_onset"].dtype.kind == "f"
    assert pd.isna(out.loc[2, "age_of_onset"])  # 'x' -> NaN
    assert pd.isna(out.loc[2, "bmi"])  # '' -> NaN


def test_clean_regimens_winsorises_duration() -> None:
    out = clean_regimens(_reg())
    assert pd.isna(out.loc[3, "period_span"])  # 99999 > max -> NaN
    assert out.loc[0, "period_span"] == 180
    assert "drug_set" in out.columns
    assert out.loc[1, "drug_set"] == frozenset({"Bdq", "Lzd"})
    assert MAX_PLAUSIBLE_DURATION_DAYS > 540  # sanity: real regimens not clipped


def test_build_cohort_excludes_empty_regimen_and_undefined_class() -> None:
    pc = clean_patient_cases(_pc())
    reg = clean_regimens(_reg())
    cohort = build_cohort(pc, reg)
    # condition 3 excluded (empty regimen AND Negative class)
    assert 3 not in cohort.condition_ids
    assert set(cohort.condition_ids) == {1, 2, 4}
    # CONSORT steps are monotonically non-increasing
    counts = [s.n_conditions for s in cohort.steps]
    assert counts == sorted(counts, reverse=True)


def test_build_treatment_matrix_binary_and_other_bucket() -> None:
    reg = clean_regimens(_reg())
    X = build_treatment_matrix(reg, [1, 2, 4], min_prevalence=0.5)
    assert X.shape[0] == 3
    assert OTHER_DRUG_COL in X.columns
    assert set(np.unique(X.to_numpy())) <= {0, 1}
    # Bdq appears in 2/3 conditions (>=0.5) -> retained as its own column
    assert "drug_Bdq" in X.columns
    assert X.loc[2, "drug_Bdq"] == 1


def test_build_treatment_matrix_rare_drug_collapsed() -> None:
    reg = clean_regimens(_reg())
    # With high threshold, drugs in only 1/3 conditions go to the 'other' bucket.
    X = build_treatment_matrix(reg, [1, 2, 4], min_prevalence=0.9)
    # E,H,R,Z (cond 1 only) are rare -> collapsed
    assert "drug_E" not in X.columns
    assert X.loc[1, OTHER_DRUG_COL] == 1
