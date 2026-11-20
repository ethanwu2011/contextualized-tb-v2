"""Baseline models: logistic regression and gradient-boosted trees, with and without context.

Four baselines establish a performance floor for the treatment-intensification outcome:

==================  ============================  =================
Name                Estimator                     Features
==================  ============================  =================
LR-no-context       Logistic regression           treatments (X)
LR-with-context     Logistic regression           [X, C]
GBDT-no-context     LightGBM                       treatments (X)
GBDT-with-context   LightGBM                       [X, C]
==================  ============================  =================

Each is evaluated with stratified 5-fold cross-validation producing out-of-fold probabilities.
Imputation and encoding live inside the per-fold :class:`~sklearn.pipeline.Pipeline`, so they are
fit on training folds only and cannot leak into validation folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  (registers IterativeImputer)
)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctb.config import N_CV_FOLDS, SEED
from ctb.data.context import FeatureSpec

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Out-of-fold predictions for one baseline.

    Attributes
    ----------
    name : str
        Model identifier.
    y_true : numpy.ndarray
        Ground-truth labels (cohort order).
    y_score : numpy.ndarray
        Out-of-fold predicted positive-class probabilities.
    """

    name: str
    y_true: NDArray[np.float64]
    y_score: NDArray[np.float64]


def _make_preprocessor(
    treatment_cols: list[str],
    spec: FeatureSpec,
    use_context: bool,
    scale: bool,
) -> ColumnTransformer:
    """Build a leakage-safe column transformer.

    Parameters
    ----------
    treatment_cols : list of str
        Binary treatment indicator columns (passed through unchanged).
    spec : FeatureSpec
        Context feature specification (numeric vs categorical).
    use_context : bool
        Whether to include context features.
    scale : bool
        Whether to standardise numeric features (True for logistic regression).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformer mapping the input frame to a model-ready matrix.
    """
    transformers: list[tuple[str, object, list[str]]] = [("treat", "passthrough", treatment_cols)]
    if use_context:
        num_steps: list[tuple[str, object]] = [
            ("impute", IterativeImputer(random_state=SEED, max_iter=10, sample_posterior=False))
        ]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(num_steps)
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
            ]
        )
        if spec.numeric:
            transformers.append(("num", num_pipe, spec.numeric))
        if spec.categorical:
            transformers.append(("cat", cat_pipe, spec.categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _make_estimator(model_type: str) -> object:
    """Instantiate a seeded estimator.

    Parameters
    ----------
    model_type : str
        Either ``"lr"`` (logistic regression) or ``"gbdt"`` (LightGBM).

    Returns
    -------
    object
        A scikit-learn-compatible classifier.

    Raises
    ------
    ValueError
        If ``model_type`` is unrecognised.
    """
    if model_type == "lr":
        return LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
    if model_type == "gbdt":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model_type {model_type!r}")


def run_baseline(
    features: pd.DataFrame,
    y: pd.Series[int],
    treatment_cols: list[str],
    spec: FeatureSpec,
    model_type: str,
    use_context: bool,
    name: str,
    n_folds: int = N_CV_FOLDS,
) -> BaselineResult:
    """Run stratified K-fold CV for one baseline and return out-of-fold probabilities.

    Parameters
    ----------
    features : pandas.DataFrame
        Concatenated ``[treatments, context]`` feature frame (NaNs allowed in context).
    y : pandas.Series
        Binary outcome aligned to ``features``.
    treatment_cols : list of str
        Treatment indicator column names.
    spec : FeatureSpec
        Context feature specification.
    model_type : str
        ``"lr"`` or ``"gbdt"``.
    use_context : bool
        Whether to include context features.
    name : str
        Model identifier (stored on the result).
    n_folds : int, default :data:`ctb.config.N_CV_FOLDS`
        Number of stratified CV folds.

    Returns
    -------
    BaselineResult
        Out-of-fold predictions.
    """
    pre = _make_preprocessor(treatment_cols, spec, use_context, scale=(model_type == "lr"))
    pipe = Pipeline([("pre", pre), ("clf", _make_estimator(model_type))])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_arr = y.to_numpy().astype(np.float64)
    proba = cross_val_predict(pipe, features, y_arr, cv=cv, method="predict_proba", n_jobs=1)
    y_score = proba[:, 1].astype(np.float64)
    logger.info("Baseline %-20s done (n=%d, prevalence=%.3f)", name, len(y_arr), y_arr.mean())
    return BaselineResult(name=name, y_true=y_arr, y_score=y_score)


def run_all_baselines(
    treatments: pd.DataFrame,
    context: pd.DataFrame,
    spec: FeatureSpec,
    y: pd.Series[int],
    n_folds: int = N_CV_FOLDS,
) -> dict[str, BaselineResult]:
    """Run all four baselines on a common cohort.

    Parameters
    ----------
    treatments : pandas.DataFrame
        Treatment matrix ``X`` (binary), indexed by condition.
    context : pandas.DataFrame
        Context matrix ``C``, indexed by condition (aligned to ``treatments``).
    spec : FeatureSpec
        Context feature specification.
    y : pandas.Series
        Binary outcome aligned to the cohort index.
    n_folds : int, default :data:`ctb.config.N_CV_FOLDS`
        Number of CV folds.

    Returns
    -------
    dict of str to BaselineResult
        Keyed by model name.
    """
    treatment_cols = list(treatments.columns)
    features = treatments.join(context, how="left")
    y = y.reindex(features.index)

    configs = [
        ("LR-no-context", "lr", False),
        ("LR-with-context", "lr", True),
        ("GBDT-no-context", "gbdt", False),
        ("GBDT-with-context", "gbdt", True),
    ]
    results: dict[str, BaselineResult] = {}
    for name, mtype, use_ctx in configs:
        results[name] = run_baseline(
            features, y, treatment_cols, spec, mtype, use_ctx, name, n_folds
        )
    return results


def fit_full_gbdt_with_context(
    treatments: pd.DataFrame,
    context: pd.DataFrame,
    spec: FeatureSpec,
    y: pd.Series[int],
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit the GBDT-with-context model on the full cohort (for SHAP feature importance).

    Parameters
    ----------
    treatments : pandas.DataFrame
        Treatment matrix.
    context : pandas.DataFrame
        Context matrix.
    spec : FeatureSpec
        Context feature specification.
    y : pandas.Series
        Binary outcome.

    Returns
    -------
    tuple of (sklearn.pipeline.Pipeline, pandas.DataFrame)
        The fitted pipeline and the transformed (model-ready) feature frame with names.
    """
    treatment_cols = list(treatments.columns)
    features = treatments.join(context, how="left")
    y = y.reindex(features.index)
    pre = _make_preprocessor(treatment_cols, spec, use_context=True, scale=False)
    pipe = Pipeline([("pre", pre), ("clf", _make_estimator("gbdt"))])
    pipe.fit(features, y.to_numpy().astype(np.float64))
    transformed = pipe.named_steps["pre"].transform(features)
    names = pipe.named_steps["pre"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=names, index=features.index)
    return pipe, transformed_df
