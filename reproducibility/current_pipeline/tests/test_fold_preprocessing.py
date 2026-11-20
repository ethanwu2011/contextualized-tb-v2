from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ctb.data.treatments import OTHER_DRUG_COL
from ctb.models.fold_preprocessing import (
    MISSING_CATEGORY,
    FoldMatrices,
    FoldPreprocessor,
    FoldTransformer,
)


def _training_context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10.0, 20.0, np.nan, 40.0],
            "all_missing_numeric": [np.nan, np.nan, np.nan, np.nan],
            "constant_numeric": [7.0, 7.0, np.nan, 7.0],
            "country": ["Peru", "India", None, "India"],
            "all_missing_category": [None, None, None, None],
            "constant_category": ["adult", "adult", "adult", "adult"],
        },
        index=pd.Index(["a", "b", "c", "d"], name="condition_id"),
    )


def _preprocessor(*, threshold: float = 0.5) -> FoldPreprocessor:
    return FoldPreprocessor(
        numeric_columns=["age", "all_missing_numeric", "constant_numeric"],
        categorical_columns=["country", "all_missing_category", "constant_category"],
        min_treatment_prevalence=threshold,
    )


def test_fit_transform_learns_numeric_statistics_and_drops_training_constants() -> None:
    context = _training_context()
    treatments = [{"H", "R"}, {"H"}, {"H", "Z"}, {"H", "R"}]

    pre = _preprocessor()
    result = pre.fit_transform(context, treatments)

    assert pre.retained_numeric_columns_ == ("age",)
    assert pre.dropped_numeric_columns_ == ("all_missing_numeric", "constant_numeric")
    assert pre.numeric_medians_ == {"age": 20.0}
    assert pre.numeric_means_ == {"age": 22.5}
    assert pre.numeric_scales_["age"] == pytest.approx(np.std([10.0, 20.0, 20.0, 40.0]))
    assert pre.retained_categorical_columns_ == ("country",)
    assert pre.dropped_categorical_columns_ == (
        "all_missing_category",
        "constant_category",
    )
    assert result.C.shape == (4, 4)
    assert result.C.dtype == np.float64
    assert np.isfinite(result.C).all()
    assert result.row_index.equals(context.index)


def test_heldout_numeric_values_use_training_median_mean_and_scale_only() -> None:
    train = _training_context()
    pre = _preprocessor().fit(train, [{"H"}, {"H"}, {"H"}, {"H"}])
    heldout = pd.DataFrame(
        {
            "age": [np.nan, 1_000.0],
            # These vary in heldout data but stay dropped because train decided the schema.
            "all_missing_numeric": [1.0, 2.0],
            "constant_numeric": [7.0, 99.0],
            "country": ["India", "new-country"],
            "all_missing_category": ["now-present", "also-present"],
            "constant_category": ["child", "adult"],
        },
        index=["test-1", "test-2"],
    )

    result = pre.transform(heldout, [{"H"}, {"unseen-drug"}])

    expected_missing = (20.0 - pre.numeric_means_["age"]) / pre.numeric_scales_["age"]
    expected_outlier = (1_000.0 - pre.numeric_means_["age"]) / pre.numeric_scales_["age"]
    assert result.C[:, 0] == pytest.approx([expected_missing, expected_outlier])
    assert result.context_feature_names == pre.context_feature_names_
    assert "all_missing_numeric" not in result.context_feature_names
    assert "constant_numeric" not in result.context_feature_names


def test_categorical_vocabulary_is_train_only_deterministic_and_ignores_unknowns() -> None:
    train = _training_context()
    pre = _preprocessor().fit(train, [set(), set(), set(), set()])
    heldout = train.iloc[:3].copy()
    heldout["country"] = ["unseen", None, "Peru"]
    result = pre.transform(heldout, [set(), set(), set()])

    assert pre.categorical_vocabulary_["country"] == (
        MISSING_CATEGORY,
        "India",
        "Peru",
    )
    country_names = tuple(
        name for name in result.context_feature_names if name.startswith("country")
    )
    assert country_names == (
        "country__%3CMISSING%3E",
        "country__India",
        "country__Peru",
    )
    country_block = result.C[:, 1:]
    np.testing.assert_array_equal(
        country_block,
        np.asarray(
            [
                [0.0, 0.0, 0.0],  # held-out level is ignored
                [1.0, 0.0, 0.0],  # missing was a training level
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_treatment_vocabulary_is_train_only_with_fixed_other_bucket() -> None:
    context = _training_context()
    train_treatments = [
        {"H", "R", "Bdq"},
        {"H", "R"},
        {"H", "Z"},
        {"H", "R"},
    ]
    pre = _preprocessor(threshold=0.5).fit(context, train_treatments)

    assert pre.treatment_vocabulary_ == ("H", "R")
    assert pre.rare_treatments_ == ("Bdq", "Z")
    assert pre.treatment_feature_names_ == ("drug_H", "drug_R", OTHER_DRUG_COL)

    heldout = context.iloc[:4].copy()
    result = pre.transform(
        heldout,
        [
            {"H"},
            {"R", "Bdq"},  # training-rare -> other
            {"new-drug"},  # never observed in training -> other
            {"H", "R", "new-drug"},
        ],
    )
    np.testing.assert_array_equal(
        result.X,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    assert set(np.unique(result.X)) <= {0.0, 1.0}


def test_other_bucket_is_present_even_when_no_treatment_is_rare() -> None:
    context = _training_context()
    pre = _preprocessor(threshold=0.0).fit(
        context,
        [{"H"}, {"R"}, {"Z"}, {"Bdq"}],
    )

    assert pre.rare_treatments_ == ()
    assert pre.treatment_feature_names_[-1] == OTHER_DRUG_COL
    train_result = pre.transform(context, [{"H"}, {"R"}, {"Z"}, {"Bdq"}])
    assert not train_result.X[:, -1].any()
    heldout_result = pre.transform(context.iloc[:1], [{"new"}])
    assert heldout_result.X[0, -1] == 1.0


def test_names_and_vocabularies_do_not_depend_on_training_row_order() -> None:
    context = _training_context()
    treatments = pd.Series(
        [{"H", "R", "Bdq"}, {"H", "R"}, {"H", "Z"}, {"H", "R"}],
        index=context.index,
    )
    shuffled_context = context.iloc[[2, 0, 3, 1]]
    shuffled_treatments = treatments.reindex(shuffled_context.index)

    original = _preprocessor().fit(context, treatments)
    shuffled = _preprocessor().fit(shuffled_context, shuffled_treatments)

    assert original.context_feature_names_ == shuffled.context_feature_names_
    assert original.treatment_feature_names_ == shuffled.treatment_feature_names_
    assert original.categorical_vocabulary_ == shuffled.categorical_vocabulary_
    assert original.treatment_vocabulary_ == shuffled.treatment_vocabulary_
    assert original.numeric_medians_ == shuffled.numeric_medians_
    assert original.numeric_means_ == shuffled.numeric_means_
    assert original.numeric_scales_ == pytest.approx(shuffled.numeric_scales_)


def test_all_context_columns_may_be_dropped_without_breaking_shape_contract() -> None:
    context = pd.DataFrame(
        {"constant": [1.0, np.nan], "all_missing": [None, None]},
        index=["a", "b"],
    )
    pre = FoldPreprocessor(["constant"], ["all_missing"], min_treatment_prevalence=0.5)

    result = pre.fit_transform(context, [set(), set()])

    assert result.C.shape == (2, 0)
    assert result.context_feature_names == ()
    assert result.X.shape == (2, 1)
    assert result.treatment_feature_names == (OTHER_DRUG_COL,)


def test_empty_heldout_partition_preserves_fitted_column_shapes() -> None:
    context = _training_context()
    pre = _preprocessor().fit(context, [{"H"}, {"H"}, {"H"}, {"H"}])
    empty = context.iloc[:0]

    result = pre.transform(empty, [])

    assert result.C.shape == (0, len(pre.context_feature_names_))
    assert result.X.shape == (0, len(pre.treatment_feature_names_))


def test_transform_requires_fit_and_complete_schema() -> None:
    context = _training_context()
    pre = _preprocessor()

    with pytest.raises(NotFittedError, match="fitted"):
        pre.transform(context, [set()] * len(context))

    pre.fit(context, [set()] * len(context))
    with pytest.raises(KeyError, match="constant_category"):
        pre.transform(context.drop(columns="constant_category"), [set()] * len(context))


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, np.inf, np.nan])
def test_prevalence_threshold_must_be_finite_probability(bad_threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _preprocessor(threshold=bad_threshold)


def test_column_schema_rejects_duplicates_and_type_overlap() -> None:
    with pytest.raises(ValueError, match="unique"):
        FoldPreprocessor(["age", "age"], [])
    with pytest.raises(ValueError, match="disjoint"):
        FoldPreprocessor(["age"], ["age"])

    context = _training_context()
    context.columns = [
        "age",
        "all_missing_numeric",
        "constant_numeric",
        "country",
        "constant_category",
        "constant_category",
    ]
    with pytest.raises(ValueError, match="columns must be unique"):
        _preprocessor().fit(context, [set()] * len(context))


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("age", "not-a-number", "non-numeric"),
        ("age", np.inf, "non-finite"),
        ("country", np.inf, "non-finite"),
        ("country", MISSING_CATEGORY, "reserved level"),
    ],
)
def test_raw_values_fail_closed(column: str, bad_value: object, message: str) -> None:
    context = _training_context()
    if column == "age" and isinstance(bad_value, str):
        context[column] = context[column].astype(object)
    context.loc["a", column] = bad_value

    with pytest.raises((TypeError, ValueError), match=message):
        _preprocessor().fit(context, [set()] * len(context))


@pytest.mark.parametrize(
    ("treatments", "error", "message"),
    [
        ([{"H"}], ValueError, "same number of rows"),
        ([{"H"}, {"H"}, {"H"}, None], TypeError, "set-like"),
        ([{"H"}, {"H"}, {"H"}, {" H"}], ValueError, "outer whitespace"),
        ([{"H"}, {"H"}, {"H"}, {1}], TypeError, "non-string"),
        ([{"H"}, {"H"}, {"H"}, {""}], ValueError, "empty treatment"),
    ],
)
def test_malformed_treatment_rows_fail_closed(
    treatments: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _preprocessor().fit(_training_context(), treatments)


def test_indexed_treatments_must_be_exactly_aligned() -> None:
    context = _training_context()
    treatments = pd.Series([{"H"}] * len(context), index=context.index[::-1])

    with pytest.raises(ValueError, match="index must exactly match"):
        _preprocessor().fit(context, treatments)


def test_fold_matrices_validates_shape_finiteness_and_binary_values() -> None:
    index = pd.Index(["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        FoldMatrices(
            C=np.asarray([[0.0], [np.inf]], dtype=np.float64),
            X=np.asarray([[0.0], [1.0]], dtype=np.float64),
            context_feature_names=("age",),
            treatment_feature_names=(OTHER_DRUG_COL,),
            row_index=index,
        )
    with pytest.raises(ValueError, match="binary"):
        FoldMatrices(
            C=np.asarray([[0.0], [1.0]], dtype=np.float64),
            X=np.asarray([[0.0], [0.5]], dtype=np.float64),
            context_feature_names=("age",),
            treatment_feature_names=(OTHER_DRUG_COL,),
            row_index=index,
        )
    with pytest.raises(ValueError, match="expected"):
        FoldMatrices(
            C=np.asarray([[0.0], [1.0]], dtype=np.float64),
            X=np.asarray([[0.0], [1.0]], dtype=np.float64),
            context_feature_names=("age", "extra"),
            treatment_feature_names=(OTHER_DRUG_COL,),
            row_index=index,
        )


def test_fold_transformer_alias_is_stable() -> None:
    assert FoldTransformer is FoldPreprocessor
