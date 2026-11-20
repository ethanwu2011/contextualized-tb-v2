"""Unit tests for :mod:`ctb.data.context` (synthetic, schema-matching data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctb.data.context import (
    FeatureSpec,
    _baseline_labs,
    build_context_matrix,
    context_missingness_table,
    parse_comorbidity_flags,
)


def test_parse_comorbidity_flags_tokens() -> None:
    s = pd.Series(["HIV", "Hepatitis C, HIV", "Diabetes", np.nan, "Anemia, Others"])
    flags = parse_comorbidity_flags(s)
    assert flags.loc[0, "comorb_hiv"] == 1
    assert flags.loc[1, "comorb_hiv"] == 1
    assert flags.loc[1, "comorb_hepatitis_c"] == 1
    assert flags.loc[2, "comorb_diabetes"] == 1
    assert flags.loc[3, "comorb_missing"] == 1
    assert flags.loc[4, "comorb_anemia"] == 1
    assert flags.loc[0, "comorb_missing"] == 0


def _pc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": [1, 2, 3],
            "patient_id": ["a", "b", "c"],
            "age_of_onset": [30, 45, np.nan],
            "bmi": [20.1, np.nan, 25.0],
            "gender": ["Male", "Female", "Male"],
            "education": ["Higher (university)", "Not Reported", "No education"],
            "employment": ["Employed", "Unemployed", "Retired"],
            "country": ["Ukraine", "Georgia", "Nigeria"],
            "comorbidity": ["HIV", np.nan, "Diabetes, Anemia"],
            "type_of_resistance": ["Sensitive", "MDR non XDR", "XDR"],
            "case_definition": ["New", "Relapse", "New"],
            "lung_localization": ["Pulmonary", "Pulmonary", "Extrapulmonary"],
            "lungcavity_size": ["No cavities", "More than 25mm", "Not Reported"],
            "culture": ["Positive", "Negative", "Not Reported"],
            "microscopy": ["10 to 99 in 100 (1+)", "Negative", "Not Reported"],
            "timika_score": [10, np.nan, 5],
            "number_of_children": [0, 2, 1],
            "number_of_daily_contacts": [3, 5, np.nan],
        }
    )


def _bx() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": [1, 1, 2],
            "test_date": ["2020-01-01", "2020-02-01", "2020-01-15"],
            "Alanine Aminotransferase (u/l)": [30.0, 40.0, np.nan],
            "Aspartate Aminotransferase (u/l)": [25.0, np.nan, 35.0],
            "Creatinine (umol/l)": [70.0, np.nan, 80.0],
            "Glucose (mmol/l)": [5.0, np.nan, 6.0],
            "Albumin (g/l)": [40.0, np.nan, np.nan],
            "Hemoglobin (g/l)": [120.0, np.nan, np.nan],
        }
    )


def test_build_context_matrix_shape_and_spec() -> None:
    C, spec = build_context_matrix(_pc(), _bx(), [1, 2, 3])
    assert isinstance(spec, FeatureSpec)
    assert C.shape[0] == 3
    assert set(C.columns) == set(spec.all_columns)
    # numeric and categorical groups are disjoint and present
    assert "age_of_onset" in spec.numeric
    assert "gender" in spec.categorical
    assert not set(spec.numeric) & set(spec.categorical)


def test_build_context_can_preserve_degenerate_columns_for_fold_local_pruning() -> None:
    patient_cases = _pc().assign(
        totalcavernum=0.0,
        dissemination="Not Reported",
    )

    historical, historical_spec = build_context_matrix(patient_cases, _bx(), [1, 2, 3])
    raw, raw_spec = build_context_matrix(
        patient_cases,
        _bx(),
        [1, 2, 3],
        preserve_degenerate_columns=True,
    )

    assert "totalcavernum" not in historical
    assert "dissemination" not in historical
    assert "totalcavernum" not in historical_spec.numeric
    assert "dissemination" not in historical_spec.categorical
    assert "totalcavernum" in raw
    assert "dissemination" in raw
    assert "totalcavernum" in raw_spec.numeric
    assert "dissemination" in raw_spec.categorical
    # Explicit exclusions remain prespecified even in the raw fold-local path.
    assert "lab_hemoglobin" not in raw


def test_build_context_earliest_lab_value() -> None:
    labs = _baseline_labs(_bx(), pd.Index([1, 2, 3], name="condition_id"))
    # condition 1 ALT earliest (2020-01-01) = 30.0
    assert labs.loc[1, "lab_alt"] == 30.0
    # condition 3 has no biochem rows -> NaN
    assert pd.isna(labs.loc[3, "lab_alt"])


def test_baseline_labs_exclude_post_index_measurements() -> None:
    bx = pd.DataFrame(
        {
            "condition_id": [1, 1, 2],
            "test_date": ["2020-01-01", "2020-02-01", "2020-01-05"],
            "Alanine Aminotransferase (u/l)": [30.0, 999.0, 40.0],
        }
    )
    index_dates = pd.Series(
        pd.to_datetime(["2020-01-15", "2020-01-01"]),
        index=pd.Index([1, 2], name="condition_id"),
    )

    labs = _baseline_labs(
        bx,
        pd.Index([1, 2], name="condition_id"),
        index_dates=index_dates,
    )

    assert labs.loc[1, "lab_alt"] == 30.0
    assert pd.isna(labs.loc[2, "lab_alt"])


def test_build_context_supports_backward_compatible_baseline_window() -> None:
    bx = pd.DataFrame(
        {
            "condition_id": [1, 1, 2],
            "test_date": ["2020-01-01", "2020-02-01", "2019-12-20"],
            "Alanine Aminotransferase (u/l)": [30.0, 999.0, 40.0],
        }
    )
    index_dates = pd.Series(
        pd.to_datetime(["2020-01-15", "2020-01-15"]),
        index=pd.Index([1, 2], name="condition_id"),
    )

    context, _ = build_context_matrix(
        _pc(),
        bx,
        [1, 2],
        baseline_index_dates=index_dates,
        baseline_lookback_days=30,
    )

    assert context.loc[1, "lab_alt"] == 30.0
    assert context.loc[2, "lab_alt"] == 40.0


def test_build_context_derived_categoricals() -> None:
    C, _ = build_context_matrix(_pc(), _bx(), [1, 2, 3])
    assert C.loc[1, "cavitation"] == "no"
    assert C.loc[2, "cavitation"] == "yes"
    assert C.loc[3, "cavitation"] == "unknown"
    assert C.loc[1, "baseline_culture"] == "positive"
    assert C.loc[2, "baseline_culture"] == "negative"
    assert C.loc[1, "comorb_hiv"] == 1


def test_missingness_table() -> None:
    C, spec = build_context_matrix(_pc(), _bx(), [1, 2, 3])
    tab = context_missingness_table(C, spec)
    assert {"feature", "kind", "missing_rate"} <= set(tab.columns)
    assert tab["missing_rate"].between(0, 1).all()
    assert (tab["missing_rate"].diff().dropna() <= 0).all()  # sorted desc
