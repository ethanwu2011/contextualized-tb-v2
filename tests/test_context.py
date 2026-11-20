"""Unit tests for :mod:`ctb.data.context` (synthetic, schema-matching data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctb.data.context import (
    FeatureSpec,
    apply_context_encoder,
    build_context_matrix,
    context_missingness_table,
    fit_context_encoder,
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


def test_build_context_earliest_lab_value() -> None:
    C, _ = build_context_matrix(_pc(), _bx(), [1, 2, 3])
    # condition 1 ALT earliest (2020-01-01) = 30.0
    assert C.loc[1, "lab_alt"] == 30.0
    # condition 3 has no biochem rows -> NaN
    assert pd.isna(C.loc[3, "lab_alt"])


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


def test_context_encoder_uses_training_schema_for_held_out_rows() -> None:
    """Held-out categories cannot change the fitted numeric or one-hot columns."""
    spec = FeatureSpec(numeric=["age"], categorical=["country"])
    train = pd.DataFrame({"age": [20.0, np.nan], "country": ["A", "A"]})
    test = pd.DataFrame({"age": [40.0], "country": ["unseen"]})
    encoder = fit_context_encoder(train, spec)
    train_encoded = apply_context_encoder(train, spec, encoder)
    test_encoded = apply_context_encoder(test, spec, encoder)
    assert train_encoded.columns.tolist() == test_encoded.columns.tolist()
    assert test_encoded.shape == (1, train_encoded.shape[1])
    assert "country_unseen" not in test_encoded.columns
    assert test_encoded.iloc[0]["age"] > train_encoded["age"].max()
