"""Contextualized logistic regression — the Phase 2 treatment-effect model.

Wraps :class:`contextualized.easy.ContextualizedClassifier` (Lengerich, Ellington, Xing et al.) to
fit the decomposition

``logodds(Y | X, C) = Xβ + Xβ(C) + µ(C)``

where ``X`` are treatment indicators and ``C`` is patient context. A deep **context encoder**
``Φ(C)`` generates per-patient parameters ``(β(C), µ(C))``; the per-patient model itself is a plain
logistic regression on ``X``. This is the model whose ``β(C)`` drives the homogeneous/heterogeneous
treatment-effect heatmaps.

Unlike the tree/linear baselines, the encoder requires **dense, numeric, NaN-free** inputs, so
context is imputed + one-hot encoded + scaled here (fit on the training fold only, to avoid
leakage). Treatment indicators are already binary and dense. Feature names are preserved so the
downstream effect plots are interpretable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from contextualized.easy import ContextualizedClassifier
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctb.config import N_CV_FOLDS, SEED
from ctb.data.context import FeatureSpec
from ctb.models.evaluation import FloatArray

logger = logging.getLogger(__name__)

# Quieten the very verbose PyTorch-Lightning trainer logs during fitting.
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


@dataclass
class ContextualizedConfig:
    """Hyperparameters for the contextualized classifier.

    Attributes
    ----------
    encoder_type : str
        Context-encoder architecture (``"mlp"`` or ``"ngam"`` for a neural additive model).
    n_bootstraps : int
        Number of bootstrap models (for stability and effect-significance estimation).
    max_epochs : int
        Maximum training epochs per bootstrap (with early stopping, an upper bound).
    learning_rate : float
        Optimiser learning rate.
    num_archetypes : int
        If > 0, use archetype-based modelling (sample models as a convex mix of ``k`` archetypes),
        which reduces the encoder's output dimension and can improve estimation. 0 disables it.
    encoder_width : int
        Hidden width of the context encoder.
    encoder_layers : int
        Number of hidden layers in the context encoder.
    es_patience : int
        Early-stopping patience (epochs without validation improvement). Requires ``val_split > 0``.
    val_split : float
        Fraction of training data held out for early-stopping validation.
    """

    # Defaults reflect the tuned configuration: K=8 treatment-effect archetypes match the
    # unconstrained MLP on AUROC (and beat the concatenation floor) while yielding interpretable
    # patient subtypes via the per-patient archetype weights Z = f(C).
    encoder_type: str = "mlp"
    n_bootstraps: int = 5
    max_epochs: int = 150
    learning_rate: float = 1e-3
    num_archetypes: int = 8
    encoder_width: int = 25
    encoder_layers: int = 2
    es_patience: int = 10
    val_split: float = 0.2


@dataclass
class ContextualizedFit:
    """A fitted contextualized model plus the named, dense inputs used to train it.

    Attributes
    ----------
    model : ContextualizedClassifier
        The fitted classifier.
    context : numpy.ndarray
        Dense context matrix ``C`` (n_samples × n_context_features).
    treatments : numpy.ndarray
        Dense treatment matrix ``X`` (n_samples × n_treatments).
    context_names : list of str
        Column names of ``context`` (for heatmap axes).
    treatment_names : list of str
        Column names of ``treatments`` (for heatmap axes).
    """

    model: ContextualizedClassifier
    context: NDArray[np.float64]
    treatments: NDArray[np.float64]
    context_names: list[str]
    treatment_names: list[str]


def make_context_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build a dense, leakage-safe context preprocessor for the encoder.

    Numeric features are iteratively imputed and standardised; categorical features are constant-
    imputed (explicit ``"missing"`` category) and one-hot encoded (dense). Fit on training data
    only.

    Parameters
    ----------
    spec : FeatureSpec
        Context feature specification (numeric vs categorical column groups).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformer producing a dense numeric context matrix with retrievable feature names.
    """
    num_pipe = Pipeline(
        [
            ("impute", IterativeImputer(random_state=SEED, max_iter=10, sample_posterior=False)),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
            ),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = []
    if spec.numeric:
        transformers.append(("num", num_pipe, spec.numeric))
    if spec.categorical:
        transformers.append(("cat", cat_pipe, spec.categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _proba_positive(model: ContextualizedClassifier, C: FloatArray, X: FloatArray) -> FloatArray:
    """Return positive-class probabilities, normalising the package's output shape.

    Parameters
    ----------
    model : ContextualizedClassifier
        Fitted classifier.
    C, X : numpy.ndarray
        Dense context and treatment matrices.

    Returns
    -------
    numpy.ndarray
        1-D array of positive-class probabilities.
    """
    proba = np.asarray(model.predict_proba(C, X))
    if proba.ndim == 3:  # (n, n_outcomes=1, n_classes=2)
        return proba[:, 0, 1].astype(np.float64)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1].astype(np.float64)
    return proba.ravel().astype(np.float64)


def _fit_one(
    C_train: FloatArray,
    X_train: FloatArray,
    y_train: FloatArray,
    cfg: ContextualizedConfig,
) -> ContextualizedClassifier:
    """Fit a single contextualized classifier on dense arrays.

    Parameters
    ----------
    C_train, X_train : numpy.ndarray
        Dense context and treatment training matrices.
    y_train : numpy.ndarray
        Binary labels.
    cfg : ContextualizedConfig
        Hyperparameters.

    Returns
    -------
    ContextualizedClassifier
        The fitted model.
    """
    model_kwargs: dict[str, object] = {
        "encoder_type": cfg.encoder_type,
        "n_bootstraps": cfg.n_bootstraps,
        "encoder_kwargs": {"width": cfg.encoder_width, "layers": cfg.encoder_layers},
    }
    if cfg.num_archetypes > 0:
        model_kwargs["num_archetypes"] = cfg.num_archetypes
    model = ContextualizedClassifier(**model_kwargs)
    model.fit(
        C_train.astype(np.float32),
        X_train.astype(np.float32),
        y_train.reshape(-1, 1).astype(np.float32),
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        es_patience=cfg.es_patience,
        val_split=cfg.val_split,
    )
    return model


def run_contextualized_cv(
    treatments: pd.DataFrame,
    context: pd.DataFrame,
    spec: FeatureSpec,
    y: pd.Series[int],
    name: str = "Contextualized",
    cfg: ContextualizedConfig | None = None,
    n_folds: int = N_CV_FOLDS,
) -> tuple[FloatArray, FloatArray]:
    """Stratified K-fold CV for the contextualized classifier; returns out-of-fold predictions.

    Context preprocessing is fit on each training fold only (no leakage). Treatments pass through
    unchanged.

    Parameters
    ----------
    treatments : pandas.DataFrame
        Binary treatment matrix ``X``.
    context : pandas.DataFrame
        Raw (NaN-containing) context matrix ``C``.
    spec : FeatureSpec
        Context feature specification.
    y : pandas.Series
        Binary outcome aligned to the matrices.
    name : str, default "Contextualized"
        Label for logging.
    cfg : ContextualizedConfig or None, optional
        Hyperparameters (defaults used if ``None``).
    n_folds : int, default :data:`ctb.config.N_CV_FOLDS`
        Number of CV folds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y_true, y_score)`` out-of-fold, in cohort order.
    """
    cfg = cfg or ContextualizedConfig()
    pl.seed_everything(SEED, workers=True)
    X_all = treatments.to_numpy(np.float64)
    y_all = y.to_numpy().astype(np.float64)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof = np.zeros(len(y_all), dtype=np.float64)
    for fold, (tr, te) in enumerate(cv.split(X_all, y_all), start=1):
        pre = make_context_preprocessor(spec)
        C_tr = pre.fit_transform(context.iloc[tr]).astype(np.float64)
        C_te = pre.transform(context.iloc[te]).astype(np.float64)
        model = _fit_one(C_tr, X_all[tr], y_all[tr], cfg)
        oof[te] = _proba_positive(model, C_te, X_all[te])
        logger.info("%s fold %d/%d fit (n_train=%d)", name, fold, n_folds, len(tr))
    return y_all, oof


def fit_full_contextualized(
    treatments: pd.DataFrame,
    context: pd.DataFrame,
    spec: FeatureSpec,
    y: pd.Series[int],
    cfg: ContextualizedConfig | None = None,
) -> ContextualizedFit:
    """Fit the contextualized model on the full cohort and return named dense inputs.

    Used to extract per-patient treatment effects ``β(C)`` and context effects ``µ(C)`` for the
    homogeneous/heterogeneous effect heatmaps.

    Parameters
    ----------
    treatments : pandas.DataFrame
        Binary treatment matrix ``X``.
    context : pandas.DataFrame
        Raw context matrix ``C``.
    spec : FeatureSpec
        Context feature specification.
    y : pandas.Series
        Binary outcome.
    cfg : ContextualizedConfig or None, optional
        Hyperparameters.

    Returns
    -------
    ContextualizedFit
        Fitted model plus the dense, named context and treatment matrices.
    """
    cfg = cfg or ContextualizedConfig()
    pl.seed_everything(SEED, workers=True)
    pre = make_context_preprocessor(spec)
    C_dense = pre.fit_transform(context).astype(np.float64)
    context_names = list(pre.get_feature_names_out())
    X = treatments.to_numpy(np.float64)
    treatment_names = list(treatments.columns)
    model = _fit_one(C_dense, X, y.to_numpy().astype(np.float64), cfg)
    logger.info(
        "Full contextualized fit: %d samples, %d context dims, %d treatments",
        C_dense.shape[0],
        C_dense.shape[1],
        X.shape[1],
    )
    return ContextualizedFit(
        model=model,
        context=C_dense,
        treatments=X,
        context_names=context_names,
        treatment_names=treatment_names,
    )
