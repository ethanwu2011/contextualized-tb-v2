"""Leakage-safe preprocessing for one model-development fold.

The transformer in this module consumes raw context columns and one set-valued treatment record
per row.  Every learned quantity--numeric imputation and scaling statistics, categorical levels,
and the treatment vocabulary--is estimated only by :meth:`FoldPreprocessor.fit`.  A fitted
transformer can therefore be applied unchanged to validation, temporal-holdout, or external rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any, cast
from urllib.parse import quote

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ctb.data.treatments import OTHER_DRUG_COL

FloatMatrix = NDArray[np.float64]

MISSING_CATEGORY = "<MISSING>"
TREATMENT_PREFIX = "drug_"


def _validate_feature_names(names: tuple[str, ...], *, field: str) -> None:
    """Require nonempty, unique feature names."""
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{field} must contain only non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must be unique")


def _as_column_tuple(columns: Sequence[str], *, field: str) -> tuple[str, ...]:
    """Validate and freeze a requested raw-column sequence."""
    if isinstance(columns, str | bytes):
        raise TypeError(f"{field} must be a sequence of column names, not a string")
    names = tuple(columns)
    _validate_feature_names(names, field=field)
    return names


def _is_missing_scalar(value: object, *, field: str) -> bool:
    """Return whether a scalar is missing and reject container-valued cells."""
    marker = pd.isna(cast(Any, value))
    if isinstance(marker, bool | np.bool_):
        return bool(marker)
    raise TypeError(f"{field} must contain scalar values")


def _category_text(value: object, *, column: str) -> str:
    """Convert one nonmissing scalar category to a stable text representation."""
    if isinstance(value, str | np.str_):
        text = str(value)
    elif isinstance(value, bool | np.bool_):
        text = "True" if bool(value) else "False"
    elif isinstance(value, Real):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Categorical column {column!r} contains a non-finite number")
        text = format(number, ".17g")
    elif isinstance(value, date | datetime | pd.Timestamp | np.datetime64):
        text = pd.Timestamp(value).isoformat()
    else:
        raise TypeError(
            f"Categorical column {column!r} must contain strings, finite numbers, dates, "
            "or missing values"
        )
    if text == MISSING_CATEGORY:
        raise ValueError(
            f"Categorical column {column!r} contains reserved level {MISSING_CATEGORY!r}"
        )
    return text


def _categorical_values(series: pd.Series[Any], *, column: str) -> list[str]:
    """Return canonical category strings, replacing missing cells explicitly."""
    values: list[str] = []
    for value in series.tolist():
        if _is_missing_scalar(value, field=f"Categorical column {column!r}"):
            values.append(MISSING_CATEGORY)
        else:
            values.append(_category_text(value, column=column))
    return values


def _numeric_frame(context: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce declared numeric columns without treating malformed tokens as missing."""
    out = pd.DataFrame(index=context.index)
    for column in columns:
        raw = context[column]
        raw_missing = pd.Series(
            [
                _is_missing_scalar(value, field=f"Numeric column {column!r}")
                for value in raw.tolist()
            ],
            index=raw.index,
            dtype=bool,
        )
        numeric = pd.to_numeric(raw, errors="coerce")
        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError(f"Numeric column {column!r} contains complex values")
        invalid = ~raw_missing & numeric.isna()
        if bool(invalid.any()):
            raise ValueError(
                f"Numeric column {column!r} contains {int(invalid.sum())} non-numeric values"
            )
        array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        if bool(np.isinf(array).any()):
            raise ValueError(f"Numeric column {column!r} contains non-finite values")
        out[column] = array
    return out


def _validate_context(
    context: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    allow_empty_rows: bool,
) -> None:
    """Validate the raw context frame and its required schema."""
    if not isinstance(context, pd.DataFrame):
        raise TypeError("context must be a pandas DataFrame")
    if not allow_empty_rows and len(context) == 0:
        raise ValueError("context must contain at least one training row")
    if not context.columns.is_unique:
        raise ValueError("context columns must be unique")
    missing = [column for column in required_columns if column not in context.columns]
    if missing:
        raise KeyError(f"context is missing required columns: {missing}")


def _normalise_treatment_rows(
    treatments: Sequence[object] | pd.Series[Any],
    *,
    expected_rows: int,
    context_index: pd.Index[Any],
) -> list[frozenset[str]]:
    """Validate and normalize one set-valued treatment record per context row."""
    raw_rows: list[object]
    if isinstance(treatments, pd.Series):
        if not treatments.index.equals(context_index):
            raise ValueError("treatment Series index must exactly match the context index")
        raw_rows = cast(list[object], treatments.tolist())
    else:
        if isinstance(treatments, str | bytes):
            raise TypeError("treatments must contain one set-valued record per row")
        if isinstance(treatments, np.ndarray) and treatments.ndim != 1:
            raise ValueError("treatments array must be one-dimensional")
        try:
            raw_rows = list(treatments)
        except TypeError as exc:
            raise TypeError("treatments must be a finite sequence") from exc

    if len(raw_rows) != expected_rows:
        raise ValueError(
            "context and treatments must have the same number of rows; "
            f"got {expected_rows} and {len(raw_rows)}"
        )

    normalized: list[frozenset[str]] = []
    allowed_containers = (set, frozenset, list, tuple, np.ndarray)
    for row_number, raw_set in enumerate(raw_rows):
        if not isinstance(raw_set, allowed_containers) or isinstance(raw_set, str | bytes):
            raise TypeError(
                f"Treatment row {row_number} must be a set-like collection of drug names"
            )
        if isinstance(raw_set, np.ndarray) and raw_set.ndim != 1:
            raise ValueError(f"Treatment row {row_number} must be one-dimensional")
        tokens: set[str] = set()
        for token in raw_set:
            if not isinstance(token, str | np.str_):
                raise TypeError(f"Treatment row {row_number} contains a non-string treatment name")
            name = str(token)
            if not name or not name.strip():
                raise ValueError(f"Treatment row {row_number} contains an empty treatment name")
            if name != name.strip():
                raise ValueError(
                    f"Treatment row {row_number} contains a treatment name with outer whitespace"
                )
            tokens.add(name)
        normalized.append(frozenset(tokens))
    return normalized


def _categorical_feature_name(column: str, category: str) -> str:
    """Return an unambiguous deterministic name for an encoded categorical level."""
    return f"{column}__{quote(category, safe='-_.~')}"


def _treatment_feature_name(treatment: str) -> str:
    """Return a deterministic feature name for a retained treatment."""
    return f"{TREATMENT_PREFIX}{quote(treatment, safe='-_.~')}"


def _validate_matrix(
    matrix: FloatMatrix,
    *,
    field: str,
    n_rows: int,
    n_columns: int,
    binary: bool,
) -> None:
    """Validate a transformed dense matrix without silently repairing it."""
    if matrix.ndim != 2:
        raise ValueError(f"{field} must be a two-dimensional matrix")
    if matrix.shape != (n_rows, n_columns):
        raise ValueError(f"{field} has shape {matrix.shape}; expected {(n_rows, n_columns)}")
    if matrix.dtype != np.dtype(np.float64):
        raise TypeError(f"{field} must have float64 dtype")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{field} must contain only finite values")
    if binary and not bool(np.isin(matrix, (0.0, 1.0)).all()):
        raise ValueError(f"{field} must be binary (0/1)")


@dataclass(frozen=True)
class FoldMatrices:
    """Aligned model matrices produced by a fitted fold preprocessor.

    Attributes
    ----------
    C : numpy.ndarray
        Dense, finite context matrix.
    X : numpy.ndarray
        Dense, finite binary treatment matrix.
    context_feature_names : tuple of str
        Names aligned to columns of ``C``.
    treatment_feature_names : tuple of str
        Names aligned to columns of ``X``.
    row_index : pandas.Index
        Copy of the source context index.
    """

    C: FloatMatrix
    X: FloatMatrix
    context_feature_names: tuple[str, ...]
    treatment_feature_names: tuple[str, ...]
    row_index: pd.Index[Any]

    def __post_init__(self) -> None:
        """Validate the complete output contract."""
        _validate_feature_names(self.context_feature_names, field="context_feature_names")
        _validate_feature_names(self.treatment_feature_names, field="treatment_feature_names")
        overlap = set(self.context_feature_names) & set(self.treatment_feature_names)
        if overlap:
            raise ValueError(f"Context and treatment feature names overlap: {sorted(overlap)}")
        if len(self.row_index) != self.C.shape[0]:
            raise ValueError("row_index length must equal the matrix row count")
        _validate_matrix(
            self.C,
            field="C",
            n_rows=len(self.row_index),
            n_columns=len(self.context_feature_names),
            binary=False,
        )
        _validate_matrix(
            self.X,
            field="X",
            n_rows=len(self.row_index),
            n_columns=len(self.treatment_feature_names),
            binary=True,
        )

    @property
    def n_rows(self) -> int:
        """Return the number of transformed observations."""
        return len(self.row_index)


class FoldPreprocessor:
    """Fit raw context and treatment encoders on one training fold only.

    Parameters
    ----------
    numeric_columns : sequence of str
        Raw numeric context columns. Retained columns are median-imputed and standardized.
    categorical_columns : sequence of str
        Raw categorical context columns. Missing values become an explicit training vocabulary
        level and retained columns are one-hot encoded.
    min_treatment_prevalence : float, default 0.01
        Minimum fraction of training rows containing a treatment for it to receive its own column.
        Less prevalent and previously unseen treatments activate the fixed
        :data:`ctb.data.treatments.OTHER_DRUG_COL` bucket.
    """

    retained_numeric_columns_: tuple[str, ...]
    retained_categorical_columns_: tuple[str, ...]
    dropped_numeric_columns_: tuple[str, ...]
    dropped_categorical_columns_: tuple[str, ...]
    numeric_medians_: dict[str, float]
    numeric_means_: dict[str, float]
    numeric_scales_: dict[str, float]
    categorical_vocabulary_: dict[str, tuple[str, ...]]
    treatment_vocabulary_: tuple[str, ...]
    rare_treatments_: tuple[str, ...]
    context_feature_names_: tuple[str, ...]
    treatment_feature_names_: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    min_treatment_prevalence: float

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        *,
        min_treatment_prevalence: float = 0.01,
    ) -> None:
        self.numeric_columns = _as_column_tuple(numeric_columns, field="numeric_columns")
        self.categorical_columns = _as_column_tuple(
            categorical_columns, field="categorical_columns"
        )
        overlap = set(self.numeric_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(
                "numeric_columns and categorical_columns must be disjoint; "
                f"overlap={sorted(overlap)}"
            )
        if isinstance(min_treatment_prevalence, bool) or not isinstance(
            min_treatment_prevalence, Real
        ):
            raise TypeError("min_treatment_prevalence must be a real number")
        prevalence = float(min_treatment_prevalence)
        if not np.isfinite(prevalence) or not 0.0 <= prevalence <= 1.0:
            raise ValueError("min_treatment_prevalence must be finite and between 0 and 1")
        self.min_treatment_prevalence = prevalence

    @property
    def raw_columns(self) -> tuple[str, ...]:
        """Return the declared raw context schema in deterministic order."""
        return (*self.numeric_columns, *self.categorical_columns)

    def _check_is_fitted(self) -> None:
        """Raise a standard error if no training fold has been fitted."""
        if not hasattr(self, "context_feature_names_"):
            raise NotFittedError("FoldPreprocessor must be fitted before transform")

    def fit(
        self,
        context: pd.DataFrame,
        treatments: Sequence[object] | pd.Series[Any],
    ) -> FoldPreprocessor:
        """Learn all preprocessing state from training rows.

        All-missing and constant raw columns are removed according to the training frame only.
        The method does not inspect any future frame that may later be passed to
        :meth:`transform`.

        Parameters
        ----------
        context : pandas.DataFrame
            Raw context rows for the training fold.
        treatments : sequence or pandas.Series
            One set-like collection of treatment-name strings per training row.

        Returns
        -------
        FoldPreprocessor
            This fitted instance.
        """
        _validate_context(context, self.raw_columns, allow_empty_rows=False)
        numeric = _numeric_frame(context, self.numeric_columns)
        categorical = {
            column: _categorical_values(context[column], column=column)
            for column in self.categorical_columns
        }
        treatment_rows = _normalise_treatment_rows(
            treatments,
            expected_rows=len(context),
            context_index=context.index,
        )

        retained_numeric: list[str] = []
        dropped_numeric: list[str] = []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in self.numeric_columns:
            observed = numeric[column].dropna()
            if observed.empty or observed.nunique(dropna=True) <= 1:
                dropped_numeric.append(column)
                continue
            median = float(observed.median())
            imputed = numeric[column].fillna(median).to_numpy(dtype=np.float64)
            mean = float(imputed.mean())
            scale = float(imputed.std(ddof=0))
            if not np.isfinite(median) or not np.isfinite(mean) or not np.isfinite(scale):
                raise ValueError(f"Numeric training statistics for {column!r} are non-finite")
            if scale <= 0.0:
                raise ValueError(f"Numeric training scale for {column!r} is not positive")
            retained_numeric.append(column)
            medians[column] = median
            means[column] = mean
            scales[column] = scale

        retained_categorical: list[str] = []
        dropped_categorical: list[str] = []
        categorical_vocabulary: dict[str, tuple[str, ...]] = {}
        for column in self.categorical_columns:
            values = categorical[column]
            vocabulary = tuple(sorted(set(values)))
            if vocabulary == (MISSING_CATEGORY,) or len(vocabulary) <= 1:
                dropped_categorical.append(column)
                continue
            retained_categorical.append(column)
            categorical_vocabulary[column] = vocabulary

        counts = Counter(token for row in treatment_rows for token in row)
        common_treatments = tuple(
            sorted(
                token
                for token, count in counts.items()
                if count / len(treatment_rows) >= self.min_treatment_prevalence
            )
        )
        rare_treatments = tuple(sorted(set(counts) - set(common_treatments)))

        context_names = tuple(retained_numeric)
        context_names += tuple(
            _categorical_feature_name(column, category)
            for column in retained_categorical
            for category in categorical_vocabulary[column]
        )
        treatment_names = tuple(_treatment_feature_name(token) for token in common_treatments)
        treatment_names += (OTHER_DRUG_COL,)
        _validate_feature_names(context_names, field="context_feature_names")
        _validate_feature_names(treatment_names, field="treatment_feature_names")
        overlap = set(context_names) & set(treatment_names)
        if overlap:
            raise ValueError(f"Context and treatment feature names overlap: {sorted(overlap)}")

        self.retained_numeric_columns_ = tuple(retained_numeric)
        self.dropped_numeric_columns_ = tuple(dropped_numeric)
        self.numeric_medians_ = medians
        self.numeric_means_ = means
        self.numeric_scales_ = scales
        self.retained_categorical_columns_ = tuple(retained_categorical)
        self.dropped_categorical_columns_ = tuple(dropped_categorical)
        self.categorical_vocabulary_ = categorical_vocabulary
        self.treatment_vocabulary_ = common_treatments
        self.rare_treatments_ = rare_treatments
        self.context_feature_names_ = context_names
        self.treatment_feature_names_ = treatment_names
        return self

    def transform(
        self,
        context: pd.DataFrame,
        treatments: Sequence[object] | pd.Series[Any],
    ) -> FoldMatrices:
        """Apply training-fold preprocessing state to aligned rows.

        Unknown categorical levels are ignored (all retained levels for that raw column remain
        zero). Unknown or training-rare treatments activate the fixed other/rare bucket.

        Parameters
        ----------
        context : pandas.DataFrame
            Raw context rows with the fitted input schema.
        treatments : sequence or pandas.Series
            One set-like collection of treatment-name strings per context row.

        Returns
        -------
        FoldMatrices
            Finite context matrix, binary treatment matrix, names, and row index.
        """
        self._check_is_fitted()
        _validate_context(context, self.raw_columns, allow_empty_rows=True)
        numeric = _numeric_frame(context, self.numeric_columns)
        categorical = {
            column: _categorical_values(context[column], column=column)
            for column in self.categorical_columns
        }
        treatment_rows = _normalise_treatment_rows(
            treatments,
            expected_rows=len(context),
            context_index=context.index,
        )
        n_rows = len(context)

        context_blocks: list[FloatMatrix] = []
        if self.retained_numeric_columns_:
            numeric_columns: list[FloatMatrix] = []
            for column in self.retained_numeric_columns_:
                numeric_values = (
                    numeric[column].fillna(self.numeric_medians_[column]).to_numpy(dtype=np.float64)
                )
                standardized = (
                    numeric_values - self.numeric_means_[column]
                ) / self.numeric_scales_[column]
                numeric_columns.append(standardized.reshape(-1, 1))
            context_blocks.append(np.hstack(numeric_columns))

        for column in self.retained_categorical_columns_:
            category_values = categorical[column]
            vocabulary = self.categorical_vocabulary_[column]
            encoded = np.zeros((n_rows, len(vocabulary)), dtype=np.float64)
            positions = {category: position for position, category in enumerate(vocabulary)}
            for row_number, category in enumerate(category_values):
                position = positions.get(category)
                if position is not None:
                    encoded[row_number, position] = 1.0
            context_blocks.append(encoded)

        C = (
            np.hstack(context_blocks).astype(np.float64, copy=False)
            if context_blocks
            else np.empty((n_rows, 0), dtype=np.float64)
        )

        treatment_positions = {
            treatment: position for position, treatment in enumerate(self.treatment_vocabulary_)
        }
        other_position = len(self.treatment_vocabulary_)
        X = np.zeros((n_rows, len(self.treatment_feature_names_)), dtype=np.float64)
        for row_number, treatment_set in enumerate(treatment_rows):
            has_other = False
            for treatment in treatment_set:
                position = treatment_positions.get(treatment)
                if position is None:
                    has_other = True
                else:
                    X[row_number, position] = 1.0
            if has_other:
                X[row_number, other_position] = 1.0

        return FoldMatrices(
            C=C,
            X=X,
            context_feature_names=self.context_feature_names_,
            treatment_feature_names=self.treatment_feature_names_,
            row_index=context.index.copy(),
        )

    def fit_transform(
        self,
        context: pd.DataFrame,
        treatments: Sequence[object] | pd.Series[Any],
    ) -> FoldMatrices:
        """Fit on and transform the same training rows."""
        return self.fit(context, treatments).transform(context, treatments)


# A concise alias is useful in fold-runner code while preserving the explicit primary class name.
FoldTransformer = FoldPreprocessor
