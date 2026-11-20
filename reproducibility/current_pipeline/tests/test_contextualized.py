"""Fast unit tests for contextualized-model preprocessing (no neural-net fit).

The model fit itself is exercised by ``scripts/04_train_contextualized.py``; here we only verify the
leakage-safe dense preprocessing contract that the context encoder depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctb.data.context import FeatureSpec
from ctb.models.contextualized import make_context_preprocessor


def _context() -> tuple[pd.DataFrame, FeatureSpec]:
    df = pd.DataFrame(
        {
            "age_of_onset": [30.0, 45.0, np.nan, 50.0, 22.0, 60.0],
            "bmi": [20.1, np.nan, 25.0, 18.0, 21.0, np.nan],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "type_of_resistance": [
                "Sensitive",
                "MDR non XDR",
                "XDR",
                "Sensitive",
                "XDR",
                "Mono DR",
            ],
        }
    )
    spec = FeatureSpec(
        numeric=["age_of_onset", "bmi"], categorical=["gender", "type_of_resistance"]
    )
    return df, spec


def test_preprocessor_outputs_dense_no_nan() -> None:
    df, spec = _context()
    pre = make_context_preprocessor(spec)
    out = pre.fit_transform(df)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "f"
    assert not np.isnan(out).any()  # encoder requires NaN-free input
    assert out.shape[0] == len(df)


def test_preprocessor_has_named_features() -> None:
    df, spec = _context()
    pre = make_context_preprocessor(spec)
    pre.fit(df)
    names = list(pre.get_feature_names_out())
    assert "age_of_onset" in names  # numeric passes name through
    assert any(n.startswith("gender") for n in names)  # one-hot expands categoricals


def test_preprocessor_is_leakage_safe() -> None:
    # Transforming a held-out row must not require refitting (imputation stats from train only).
    df, spec = _context()
    pre = make_context_preprocessor(spec)
    pre.fit(df.iloc[:4])
    out_test = pre.transform(df.iloc[4:])
    assert not np.isnan(out_test).any()
    assert out_test.shape[0] == 2
