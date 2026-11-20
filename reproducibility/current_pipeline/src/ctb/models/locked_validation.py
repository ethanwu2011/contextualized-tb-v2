"""Governed patient-grouped nested validation for additive LR and residual PATCH.

The public result from this module contains aggregate metrics and fold counts only. Row-level
outcomes, predictions, condition identifiers, and patient identifiers are written only to an
explicit DUA-private directory under a caller-supplied secure output root.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from ctb.config import N_BOOTSTRAP, SEED
from ctb.data.clean import build_cohort
from ctb.data.context import build_context_matrix
from ctb.data.outcomes import (
    derive_patient_case_died_failure,
    require_modelable_binary_outcome,
)
from ctb.data.treatments import select_initial_regimens
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.residual_archetype import (
    ResidualArchetypeLogistic,
    ResidualTrainConfig,
    train_residual,
)

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

METRIC_NAMES = (
    "auroc",
    "auprc",
    "log_loss",
    "brier",
    "calibration_intercept",
    "calibration_slope",
)


@dataclass(frozen=True)
class PatchCandidate:
    """One prespecified residual-PATCH candidate.

    ``n_archetypes=None`` denotes the exact additive-logistic null. Non-null candidates freeze the
    fitted additive component and estimate only the residual interaction block.
    """

    name: str
    n_archetypes: int | None = None
    loading_l2: float = 0.0
    deviation_l2: float = 0.0
    residual_scale: float = 0.0
    epochs: int = 0
    learning_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate a finite candidate specification."""
        if not self.name or not self.name.strip():
            raise ValueError("Patch candidate name must be nonblank")
        if self.n_archetypes is None:
            if any(
                value != 0
                for value in (
                    self.loading_l2,
                    self.deviation_l2,
                    self.residual_scale,
                    self.epochs,
                    self.learning_rate,
                )
            ):
                raise ValueError("The null PATCH candidate cannot have training parameters")
            return
        if self.n_archetypes < 2:
            raise ValueError("A non-null PATCH candidate needs at least two archetypes")
        if self.epochs < 1:
            raise ValueError("A non-null PATCH candidate needs at least one epoch")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("PATCH learning rate must be finite and positive")
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError("PATCH residual scale must be finite and positive")
        for field, value in (
            ("loading_l2", self.loading_l2),
            ("deviation_l2", self.deviation_l2),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")

    @property
    def is_null(self) -> bool:
        """Return whether this is the exact additive-logistic candidate."""
        return self.n_archetypes is None


DEFAULT_PATCH_CANDIDATES = (
    PatchCandidate(name="null"),
    PatchCandidate(
        name="residual_k3_strong_shrinkage",
        n_archetypes=3,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.5,
        epochs=300,
        learning_rate=0.01,
    ),
    PatchCandidate(
        name="residual_k3_moderate_shrinkage",
        n_archetypes=3,
        loading_l2=1e-4,
        deviation_l2=1e-3,
        residual_scale=1.0,
        epochs=300,
        learning_rate=0.01,
    ),
    PatchCandidate(
        name="residual_k5_strong_shrinkage",
        n_archetypes=5,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.5,
        epochs=300,
        learning_rate=0.01,
    ),
    PatchCandidate(
        name="residual_k5_moderate_shrinkage",
        n_archetypes=5,
        loading_l2=1e-4,
        deviation_l2=1e-3,
        residual_scale=1.0,
        epochs=300,
        learning_rate=0.01,
    ),
)


@dataclass(frozen=True)
class LockedValidationConfig:
    """Finite, deterministic nested-validation settings."""

    outer_splits: int = 5
    inner_splits: int = 3
    lr_c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    patch_candidates: tuple[PatchCandidate, ...] = DEFAULT_PATCH_CANDIDATES
    min_treatment_prevalence: float = 0.01
    tie_tolerance: float = 1e-4
    logistic_max_iter: int = 2000
    bootstrap_resamples: int = N_BOOTSTRAP
    seed: int = SEED

    def __post_init__(self) -> None:
        """Validate the complete analysis configuration."""
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("Outer and inner split counts must both be at least two")
        if not self.lr_c_values or len(set(self.lr_c_values)) != len(self.lr_c_values):
            raise ValueError("lr_c_values must be a nonempty unique tuple")
        if any(not np.isfinite(value) or value <= 0 for value in self.lr_c_values):
            raise ValueError("Every logistic C value must be finite and positive")
        names = [candidate.name for candidate in self.patch_candidates]
        if not names or len(set(names)) != len(names):
            raise ValueError("PATCH candidates must have unique names")
        if sum(candidate.is_null for candidate in self.patch_candidates) != 1:
            raise ValueError("PATCH candidates must contain exactly one null")
        if not np.isfinite(self.min_treatment_prevalence) or not (
            0 <= self.min_treatment_prevalence <= 1
        ):
            raise ValueError("min_treatment_prevalence must be between zero and one")
        if not np.isfinite(self.tie_tolerance) or self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be finite and nonnegative")
        if self.logistic_max_iter < 1:
            raise ValueError("logistic_max_iter must be positive")
        if self.bootstrap_resamples < 10:
            raise ValueError("bootstrap_resamples must be at least ten")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True)
class RawValidationData:
    """Aligned raw inputs for fold-local validation."""

    context: pd.DataFrame
    treatments: pd.Series[Any]
    outcome: pd.Series[Any]
    patient_groups: pd.Series[Any]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Fail closed on alignment, identifiers, and binary endpoint validity."""
        if not isinstance(self.context, pd.DataFrame) or self.context.empty:
            raise ValueError("context must be a nonempty DataFrame")
        if not self.context.index.is_unique or self.context.index.hasnans:
            raise ValueError("context must have unique, nonmissing condition identifiers")
        for name, values in (
            ("treatments", self.treatments),
            ("outcome", self.outcome),
            ("patient_groups", self.patient_groups),
        ):
            if not isinstance(values, pd.Series):
                raise TypeError(f"{name} must be a pandas Series")
            if not values.index.equals(self.context.index):
                raise ValueError(f"{name} index must exactly match context index")
        all_columns = (*self.numeric_columns, *self.categorical_columns)
        if len(set(all_columns)) != len(all_columns):
            raise ValueError("Numeric and categorical raw column specifications must be disjoint")
        missing_columns = [column for column in all_columns if column not in self.context.columns]
        if missing_columns:
            raise KeyError(f"context is missing declared columns: {missing_columns}")
        if not all_columns:
            raise ValueError("At least one raw context column is required")

        outcome = np.asarray(self.outcome)
        if outcome.ndim != 1 or len(outcome) != len(self.context):
            raise ValueError("outcome must be one-dimensional and aligned")
        numeric_outcome = pd.to_numeric(self.outcome, errors="coerce")
        if numeric_outcome.isna().any() or not numeric_outcome.isin([0, 1]).all():
            raise ValueError("outcome must contain only assessed binary zero/one labels")
        if set(numeric_outcome.astype(int).unique()) != {0, 1}:
            raise ValueError("outcome must contain both classes")

        groups = self.patient_groups.astype("string")
        stripped = groups.str.strip()
        if stripped.isna().any() or stripped.eq("").any():
            raise ValueError("Every condition needs one nonblank patient group")


@dataclass(frozen=True)
class LockedValidationResult:
    """Private predictions plus a release-safe aggregate report."""

    condition_ids: NDArray[Any]
    patient_groups: NDArray[Any]
    y_true: IntArray
    additive_probability: FloatArray
    patch_probability: FloatArray
    outer_fold: IntArray
    aggregate_report: dict[str, Any]


@dataclass(frozen=True)
class _PreparedFold:
    """One inner split after train-only preprocessing."""

    train: FoldMatrices
    validation: FoldMatrices
    y_train: IntArray
    y_validation: IntArray


def resolve_patient_groups(
    patient_cases: pd.DataFrame,
    condition_ids: pd.Index[Any],
) -> pd.Series[str]:
    """Resolve one nonblank patient identifier per condition without exposing identifiers."""
    required = {"condition_id", "patient_id"}
    missing_columns = sorted(required.difference(patient_cases.columns))
    if missing_columns:
        raise KeyError(f"Patient grouping requires columns {missing_columns}")
    requested = pd.Index(condition_ids, name="condition_id")
    if not requested.is_unique or requested.hasnans:
        raise ValueError("Requested condition identifiers must be unique and nonmissing")
    requested_set = set(requested.tolist())
    links = patient_cases.loc[
        patient_cases["condition_id"].isin(requested_set),
        ["condition_id", "patient_id"],
    ].copy()
    links["_patient_group"] = links["patient_id"].astype("string").str.strip().replace("", pd.NA)
    grouped = links.groupby("condition_id", sort=False)["_patient_group"]
    distinct = grouped.nunique(dropna=True)
    ambiguous = int(distinct.gt(1).sum())
    resolved = grouped.first().reindex(requested)
    missing = int(resolved.isna().sum())
    if missing or ambiguous:
        raise ValueError(
            "Condition-to-patient linkage is not modelable: "
            f"requested_conditions={len(requested)}, missing_links={missing}, "
            f"ambiguous_links={ambiguous}"
        )
    return pd.Series(
        resolved.astype(str).to_numpy(),
        index=requested,
        dtype="string",
        name="patient_group",
    )


def build_schema6_raw_validation_data(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    biochemistry: pd.DataFrame,
    config: LockedValidationConfig,
) -> RawValidationData:
    """Build the assessed schema-6 cohort with regimen-start laboratory time zero."""
    cohort = build_cohort(patient_cases, regimens)
    direct_frame = derive_patient_case_died_failure(patient_cases)
    direct = direct_frame["died_failure"].reindex(cohort.condition_ids)
    assessed_ids = [
        condition_id
        for condition_id, assessable in zip(
            cohort.condition_ids,
            direct.notna().to_numpy(dtype=bool),
            strict=True,
        )
        if assessable
    ]
    assessed_index: pd.Index[Any] = pd.Index(assessed_ids, name="condition_id")
    outcome = require_modelable_binary_outcome(
        direct.reindex(assessed_index),
        outcome_name="recorded final Patient Cases died/failure",
        min_class_count=config.outer_splits,
    )

    initial = select_initial_regimens(regimens).reindex(assessed_index)
    missing_start = int(initial["initial_start"].isna().sum())

    def drug_set_size(value: object) -> int:
        if not isinstance(value, set | frozenset | list | tuple):
            raise ValueError("Initial regimen rows must contain set-like drug collections")
        return len(value)

    empty_regimen = int(initial["drug_set"].map(drug_set_size).eq(0).sum())
    if missing_start or empty_regimen:
        raise ValueError(
            "Final assessed cohort lacks validated initial treatment time zero: "
            f"missing_start={missing_start}, empty_initial_regimen={empty_regimen}"
        )
    context, feature_spec = build_context_matrix(
        patient_cases,
        biochemistry,
        assessed_ids,
        baseline_index_dates=initial["initial_start"],
        preserve_degenerate_columns=True,
    )
    context = context.reindex(assessed_index)
    treatments = pd.Series(
        initial["drug_set"].to_numpy(),
        index=assessed_index,
        name="initial_regimen",
    )
    patient_groups = resolve_patient_groups(patient_cases, assessed_index)
    logger.info(
        "Locked cohort built: conditions=%d, patients=%d, events=%d, " "raw_context_features=%d",
        len(assessed_index),
        patient_groups.nunique(),
        int(outcome.sum()),
        context.shape[1],
    )
    return RawValidationData(
        context=context,
        treatments=treatments,
        outcome=outcome,
        patient_groups=patient_groups,
        numeric_columns=tuple(feature_spec.numeric),
        categorical_columns=tuple(feature_spec.categorical),
    )


def _seed(base: int, *coordinates: int) -> int:
    """Derive a deterministic nonnegative seed without Python hash state."""
    value = int(base)
    for position, coordinate in enumerate(coordinates, start=1):
        value = (value + position * 104729 * (coordinate + 1)) % (2**31 - 1)
    return value


def make_grouped_splits(
    outcome: IntArray,
    groups: NDArray[Any],
    *,
    n_splits: int,
    seed: int,
    label: str,
) -> tuple[tuple[IntArray, IntArray], ...]:
    """Create and audit deterministic stratified patient-group splits."""
    y = np.asarray(outcome, dtype=np.int64)
    patient_groups = np.asarray(groups).astype(str)
    if y.ndim != 1 or patient_groups.ndim != 1 or len(y) != len(patient_groups):
        raise ValueError(f"{label} outcome and groups must be aligned one-dimensional arrays")
    if len(np.unique(patient_groups)) < n_splits:
        raise ValueError(f"{label} needs at least {n_splits} distinct patient groups")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw_splits = splitter.split(np.zeros((len(y), 1)), y, patient_groups)
    validation_coverage = np.zeros(len(y), dtype=np.int64)
    checked: list[tuple[IntArray, IntArray]] = []
    for fold_number, (train, validation) in enumerate(raw_splits):
        train_idx = np.asarray(train, dtype=np.int64)
        validation_idx = np.asarray(validation, dtype=np.int64)
        overlap = np.intersect1d(
            np.unique(patient_groups[train_idx]),
            np.unique(patient_groups[validation_idx]),
        )
        if len(overlap):
            raise ValueError(f"{label} fold {fold_number} has patient-group overlap")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[validation_idx])) != 2:
            raise ValueError(f"{label} fold {fold_number} does not contain both outcome classes")
        validation_coverage[validation_idx] += 1
        checked.append((train_idx, validation_idx))
    if not np.all(validation_coverage == 1):
        raise ValueError(f"{label} validation partitions do not cover every row exactly once")
    return tuple(checked)


def _fit_additive(
    matrices: FoldMatrices,
    outcome: IntArray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    """Fit one penalized additive context-plus-regimen logistic model."""
    design = np.hstack([matrices.C, matrices.X])
    if design.shape[1] == 0:
        raise ValueError("Additive logistic regression has no retained training features")
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(design, outcome)
    return model


def _additive_probability(model: LogisticRegression, matrices: FoldMatrices) -> FloatArray:
    """Return validated positive-class probabilities from an additive model."""
    design = np.hstack([matrices.C, matrices.X])
    probability = np.asarray(model.predict_proba(design)[:, 1], dtype=np.float64)
    return _validate_probability(probability, n_rows=matrices.n_rows, model_name="additive LR")


def _validate_probability(
    probability: FloatArray,
    *,
    n_rows: int,
    model_name: str,
) -> FloatArray:
    """Fail closed on malformed predicted probabilities."""
    values = np.asarray(probability, dtype=np.float64)
    if values.shape != (n_rows,):
        raise ValueError(f"{model_name} returned probability shape {values.shape}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{model_name} returned non-finite or out-of-range probabilities")
    return values


def _fit_residual(
    baseline: LogisticRegression,
    matrices: FoldMatrices,
    outcome: IntArray,
    candidate: PatchCandidate,
    *,
    seed: int,
) -> ResidualArchetypeLogistic:
    """Fit a residual head around a frozen same-fold additive baseline."""
    if candidate.is_null or candidate.n_archetypes is None:
        raise ValueError("The exact null does not require residual fitting")
    if matrices.C.shape[1] < 1:
        raise ValueError("Residual PATCH needs at least one context feature retained in training")
    coefficients = np.asarray(baseline.coef_[0], dtype=np.float64)
    context_dim = matrices.C.shape[1]
    model = ResidualArchetypeLogistic(
        context_dim=context_dim,
        x_dim=matrices.X.shape[1],
        n_archetypes=candidate.n_archetypes,
        seed=seed,
    )
    model.initialize_from_logistic(
        context_coef=coefficients[:context_dim],
        treatment_coef=coefficients[context_dim:],
        intercept=float(baseline.intercept_[0]),
        freeze=True,
    )
    train_residual(
        model,
        matrices.C,
        matrices.X,
        outcome.astype(np.float64),
        ResidualTrainConfig(
            epochs=candidate.epochs,
            lr=candidate.learning_rate,
            baseline_l2=0.0,
            loading_l2=candidate.loading_l2,
            deviation_l2=candidate.deviation_l2,
            deviation_l1=0.0,
            entropy_weight=0.0,
        ),
        seed=seed,
    )
    model.set_residual_scale(candidate.residual_scale)
    return model


def _prepare_inner_folds(
    data: RawValidationData,
    outer_train: IntArray,
    inner_splits: tuple[tuple[IntArray, IntArray], ...],
    *,
    min_treatment_prevalence: float,
) -> tuple[_PreparedFold, ...]:
    """Fit each preprocessor only on its corresponding inner-training rows."""
    outer_context = data.context.iloc[outer_train]
    outer_treatments = data.treatments.iloc[outer_train]
    outer_y = np.asarray(data.outcome.iloc[outer_train], dtype=np.int64)
    prepared: list[_PreparedFold] = []
    for inner_train, inner_validation in inner_splits:
        preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=min_treatment_prevalence,
        )
        train = preprocessor.fit_transform(
            outer_context.iloc[inner_train],
            outer_treatments.iloc[inner_train],
        )
        validation = preprocessor.transform(
            outer_context.iloc[inner_validation],
            outer_treatments.iloc[inner_validation],
        )
        prepared.append(
            _PreparedFold(
                train=train,
                validation=validation,
                y_train=outer_y[inner_train],
                y_validation=outer_y[inner_validation],
            )
        )
    return tuple(prepared)


def _select_additive_c(
    prepared: tuple[_PreparedFold, ...],
    config: LockedValidationConfig,
    *,
    outer_fold: int,
) -> tuple[float, dict[str, float]]:
    """Tune additive regularization on grouped inner-fold log loss."""
    mean_scores: dict[str, float] = {}
    for c_position, c_value in enumerate(config.lr_c_values):
        fold_scores: list[float] = []
        for inner_fold, fold in enumerate(prepared):
            model = _fit_additive(
                fold.train,
                fold.y_train,
                c_value=c_value,
                max_iter=config.logistic_max_iter,
                seed=_seed(config.seed, outer_fold, c_position, inner_fold),
            )
            probability = _additive_probability(model, fold.validation)
            fold_scores.append(float(log_loss(fold.y_validation, probability, labels=[0, 1])))
        mean_scores[format(c_value, ".12g")] = float(np.mean(fold_scores))

    best_loss = min(mean_scores.values())
    eligible = [
        value
        for value in sorted(config.lr_c_values)
        if mean_scores[format(value, ".12g")] <= best_loss + config.tie_tolerance
    ]
    return eligible[0], mean_scores


def _select_patch_candidate(
    prepared: tuple[_PreparedFold, ...],
    selected_c: float,
    selected_c_score: float,
    config: LockedValidationConfig,
    *,
    outer_fold: int,
) -> tuple[PatchCandidate, dict[str, float]]:
    """Tune the residual extension and prefer its exact null whenever performance ties."""
    null = next(candidate for candidate in config.patch_candidates if candidate.is_null)
    mean_scores: dict[str, float] = {null.name: selected_c_score}
    baselines = [
        _fit_additive(
            fold.train,
            fold.y_train,
            c_value=selected_c,
            max_iter=config.logistic_max_iter,
            seed=_seed(config.seed, outer_fold, 700, inner_fold),
        )
        for inner_fold, fold in enumerate(prepared)
    ]

    for candidate_position, candidate in enumerate(config.patch_candidates):
        if candidate.is_null:
            continue
        fold_scores: list[float] = []
        for inner_fold, (fold, baseline) in enumerate(zip(prepared, baselines, strict=True)):
            seed = _seed(config.seed, outer_fold, 1000 + candidate_position, inner_fold)
            model = _fit_residual(
                baseline,
                fold.train,
                fold.y_train,
                candidate,
                seed=seed,
            )
            probability = _validate_probability(
                model.predict_proba(fold.validation.C, fold.validation.X),
                n_rows=fold.validation.n_rows,
                model_name=candidate.name,
            )
            fold_scores.append(float(log_loss(fold.y_validation, probability, labels=[0, 1])))
        mean_scores[candidate.name] = float(np.mean(fold_scores))

    nonnull = [candidate for candidate in config.patch_candidates if not candidate.is_null]
    if not nonnull:
        return null, mean_scores
    best_nonnull = min(
        nonnull,
        key=lambda candidate: (
            mean_scores[candidate.name],
            candidate.n_archetypes if candidate.n_archetypes is not None else 0,
            candidate.name,
        ),
    )
    improvement = mean_scores[null.name] - mean_scores[best_nonnull.name]
    selected = best_nonnull if improvement > config.tie_tolerance else null
    return selected, mean_scores


def _calibration_coefficients(outcome: IntArray, probability: FloatArray) -> tuple[float, float]:
    """Estimate calibration intercept and slope from the probability logit."""
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    probability_logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    calibrator.fit(probability_logit, outcome)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def _metric_vector(outcome: IntArray, probability: FloatArray) -> dict[str, float]:
    """Compute all prespecified discrimination, accuracy, and calibration metrics."""
    calibration_intercept, calibration_slope = _calibration_coefficients(outcome, probability)
    return {
        "auroc": float(roc_auc_score(outcome, probability)),
        "auprc": float(average_precision_score(outcome, probability)),
        "log_loss": float(log_loss(outcome, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(outcome, probability)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def patient_clustered_paired_metrics(
    outcome: IntArray,
    additive_probability: FloatArray,
    patch_probability: FloatArray,
    patient_groups: NDArray[Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute patient-clustered model intervals and paired PATCH-minus-LR intervals."""
    y = np.asarray(outcome, dtype=np.int64)
    additive = _validate_probability(
        additive_probability,
        n_rows=len(y),
        model_name="additive LR",
    )
    patch = _validate_probability(patch_probability, n_rows=len(y), model_name="PATCH")
    groups = np.asarray(patient_groups).astype(str)
    if len(groups) != len(y):
        raise ValueError("patient_groups must align with outcome")
    if len(np.unique(y)) != 2:
        raise ValueError("Clustered metrics require both outcome classes")
    unique_groups = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == group) for group in unique_groups]

    point_additive = _metric_vector(y, additive)
    point_patch = _metric_vector(y, patch)
    draws_additive: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    draws_patch: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    draws_difference: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    valid_resamples = 0
    for _ in range(n_resamples):
        sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        sampled_rows = np.concatenate([rows_by_group[position] for position in sampled_groups])
        sampled_y = y[sampled_rows]
        if len(np.unique(sampled_y)) != 2:
            continue
        try:
            additive_metrics = _metric_vector(sampled_y, additive[sampled_rows])
            patch_metrics = _metric_vector(sampled_y, patch[sampled_rows])
        except ValueError:
            continue
        valid_resamples += 1
        for name in METRIC_NAMES:
            draws_additive[name].append(additive_metrics[name])
            draws_patch[name].append(patch_metrics[name])
            draws_difference[name].append(patch_metrics[name] - additive_metrics[name])

    required_valid = max(5, int(np.ceil(0.5 * n_resamples)))
    if valid_resamples < required_valid:
        raise ValueError(
            "Too few valid patient-clustered resamples: "
            f"valid={valid_resamples}, required={required_valid}"
        )

    def summarize(
        point: dict[str, float],
        draws: dict[str, list[float]],
    ) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "estimate": point[name],
                "ci95": [
                    float(np.percentile(draws[name], 2.5)),
                    float(np.percentile(draws[name], 97.5)),
                ],
            }
            for name in METRIC_NAMES
        }

    difference_point = {name: point_patch[name] - point_additive[name] for name in METRIC_NAMES}
    return {
        "additive_lr": summarize(point_additive, draws_additive),
        "residual_patch": summarize(point_patch, draws_patch),
        "paired_difference_patch_minus_additive": summarize(
            difference_point,
            draws_difference,
        ),
        "bootstrap": {
            "unit": "patient",
            "paired": True,
            "requested_resamples": n_resamples,
            "valid_resamples": valid_resamples,
            "seed": seed,
        },
    }


def run_locked_validation(
    data: RawValidationData,
    config: LockedValidationConfig | None = None,
) -> LockedValidationResult:
    """Run nested, patient-grouped, fold-local validation for both models."""
    config = config or LockedValidationConfig()
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    outer_splits = make_grouped_splits(
        outcome,
        groups,
        n_splits=config.outer_splits,
        seed=config.seed,
        label="outer validation",
    )

    additive_oof = np.full(len(outcome), np.nan, dtype=np.float64)
    patch_oof = np.full(len(outcome), np.nan, dtype=np.float64)
    outer_assignment = np.full(len(outcome), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
        logger.info(
            "Locked validation outer fold %d/%d: train=%d, test=%d",
            outer_fold + 1,
            config.outer_splits,
            len(outer_train),
            len(outer_test),
        )
        inner_splits = make_grouped_splits(
            outcome[outer_train],
            groups[outer_train],
            n_splits=config.inner_splits,
            seed=_seed(config.seed, outer_fold, 17),
            label=f"inner validation for outer fold {outer_fold + 1}",
        )
        prepared = _prepare_inner_folds(
            data,
            outer_train,
            inner_splits,
            min_treatment_prevalence=config.min_treatment_prevalence,
        )
        selected_c, lr_scores = _select_additive_c(
            prepared,
            config,
            outer_fold=outer_fold,
        )
        selected_c_score = lr_scores[format(selected_c, ".12g")]
        selected_patch, patch_scores = _select_patch_candidate(
            prepared,
            selected_c,
            selected_c_score,
            config,
            outer_fold=outer_fold,
        )

        outer_preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=config.min_treatment_prevalence,
        )
        outer_train_matrices = outer_preprocessor.fit_transform(
            data.context.iloc[outer_train],
            data.treatments.iloc[outer_train],
        )
        outer_test_matrices = outer_preprocessor.transform(
            data.context.iloc[outer_test],
            data.treatments.iloc[outer_test],
        )
        baseline = _fit_additive(
            outer_train_matrices,
            outcome[outer_train],
            c_value=selected_c,
            max_iter=config.logistic_max_iter,
            seed=_seed(config.seed, outer_fold, 29),
        )
        additive_probability = _additive_probability(baseline, outer_test_matrices)
        if selected_patch.is_null:
            patch_probability = additive_probability.copy()
        else:
            patch_model = _fit_residual(
                baseline,
                outer_train_matrices,
                outcome[outer_train],
                selected_patch,
                seed=_seed(config.seed, outer_fold, 31),
            )
            patch_probability = _validate_probability(
                patch_model.predict_proba(outer_test_matrices.C, outer_test_matrices.X),
                n_rows=len(outer_test),
                model_name=selected_patch.name,
            )

        additive_oof[outer_test] = additive_probability
        patch_oof[outer_test] = patch_probability
        outer_assignment[outer_test] = outer_fold
        fold_summaries.append(
            {
                "outer_fold": outer_fold + 1,
                "train_conditions": len(outer_train),
                "test_conditions": len(outer_test),
                "train_patients": int(len(np.unique(groups[outer_train]))),
                "test_patients": int(len(np.unique(groups[outer_test]))),
                "train_events": int(outcome[outer_train].sum()),
                "test_events": int(outcome[outer_test].sum()),
                "retained_context_features": outer_train_matrices.C.shape[1],
                "retained_treatment_features": outer_train_matrices.X.shape[1],
                "selected_lr_c": selected_c,
                "selected_patch_candidate": selected_patch.name,
                "inner_lr_mean_log_loss": lr_scores,
                "inner_patch_mean_log_loss": patch_scores,
            }
        )

    if np.isnan(additive_oof).any() or np.isnan(patch_oof).any() or np.any(outer_assignment < 0):
        raise RuntimeError("Outer-fold predictions are incomplete")
    metrics = patient_clustered_paired_metrics(
        outcome,
        additive_oof,
        patch_oof,
        groups,
        n_resamples=config.bootstrap_resamples,
        seed=_seed(config.seed, 97),
    )
    null_name = next(candidate.name for candidate in config.patch_candidates if candidate.is_null)
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "analysis_id": "locked_patient_grouped_residual_patch",
        "endpoint": "recorded_final_patient_cases_died_or_failure",
        "interpretation": "observed-data prediction and treatment-associated heterogeneity",
        "claim_status": "aggregate validation result requiring governance and author review",
        "cohort": {
            "assessed_conditions": len(outcome),
            "events": int(outcome.sum()),
            "observed_non_events": int(len(outcome) - outcome.sum()),
            "patients": int(len(np.unique(groups))),
        },
        "validation": {
            "outer_splitter": "StratifiedGroupKFold",
            "outer_splits": config.outer_splits,
            "inner_splitter": "StratifiedGroupKFold",
            "inner_splits": config.inner_splits,
            "grouping_unit": "patient",
            "preprocessing_scope": "fit on each inner or outer training partition only",
            "selection_metric": "mean inner-fold log loss",
            "patch_null_tie_preference": True,
            "residual_scale_strategy": "train at unit scale, then shrink validation predictions",
            "tie_tolerance": config.tie_tolerance,
            "seed": config.seed,
        },
        "candidate_grid": {
            "lr_c_values": list(config.lr_c_values),
            "patch": [
                {
                    "name": candidate.name,
                    "is_null": candidate.is_null,
                    "n_archetypes": candidate.n_archetypes,
                    "loading_l2": candidate.loading_l2,
                    "deviation_l2": candidate.deviation_l2,
                    "residual_scale": candidate.residual_scale,
                    "epochs": candidate.epochs,
                    "learning_rate": candidate.learning_rate,
                }
                for candidate in config.patch_candidates
            ],
        },
        "outer_folds": fold_summaries,
        "patch_null_selected_folds": int(
            sum(summary["selected_patch_candidate"] == null_name for summary in fold_summaries)
        ),
        "metrics": metrics,
        "governance": {
            "public_report_contains_row_level_data": False,
            "public_report_contains_identifiers": False,
            "private_predictions_required_for_audit": True,
        },
    }
    _assert_release_safe_report(report)
    return LockedValidationResult(
        condition_ids=data.context.index.to_numpy(copy=True),
        patient_groups=groups.copy(),
        y_true=outcome,
        additive_probability=additive_oof,
        patch_probability=patch_oof,
        outer_fold=outer_assignment,
        aggregate_report=report,
    )


_FORBIDDEN_PUBLIC_KEYS = {
    "condition_id",
    "patient_id",
    "row_index",
    "outcome",
    "y_true",
    "y_score",
    "prediction",
    "predictions",
    "additive_probability",
    "patch_probability",
}


def _assert_release_safe_report(value: Any, *, path: str = "report") -> None:
    """Reject identifier or row-level fields from the aggregate report tree."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"Release report contains forbidden field {path}.{key}")
            _assert_release_safe_report(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for position, item in enumerate(value):
            _assert_release_safe_report(item, path=f"{path}[{position}]")


def prepare_dua_output_root(output_root: Path, project_root: Path) -> Path:
    """Create an empty, private output root outside the source repository."""
    root = output_root.expanduser().resolve()
    project = project_root.expanduser().resolve()
    if root == project or root.is_relative_to(project) or project.is_relative_to(root):
        raise ValueError(
            "DUA output root must be outside and must not contain the source repository"
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("DUA output root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    private_dir = root / "dua_private"
    aggregate_dir = root / "aggregate_only"
    private_dir.mkdir(mode=0o700)
    aggregate_dir.mkdir(mode=0o700)
    os.chmod(private_dir, 0o700)
    os.chmod(aggregate_dir, 0o700)
    return root


def _write_performance_figure(result: LockedValidationResult, pdf: Path, png: Path) -> None:
    """Write aggregate ROC, precision-recall, and calibration panels."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    y = result.y_true
    metric_report = result.aggregate_report["metrics"]
    additive_metrics = metric_report["additive_lr"]
    patch_metrics = metric_report["residual_patch"]
    difference_metrics = metric_report["paired_difference_patch_minus_additive"]
    predictions = (
        ("Additive LR", result.additive_probability, "#4C78A8", additive_metrics),
        ("Residual PATCH", result.patch_probability, "#E45756", patch_metrics),
    )
    figure, raw_axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes = np.asarray(raw_axes, dtype=object).reshape(-1)
    prevalence = float(y.mean())
    for name, probability, color, model_metrics in predictions:
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        mean_probability, fraction_positive = calibration_curve(
            y,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            linewidth=1.8,
            label=f"{name}  {model_metrics['auroc']['estimate']:.4f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=1.8,
            label=f"{name}  {model_metrics['auprc']['estimate']:.4f}",
        )
        axes[2].plot(
            mean_probability,
            fraction_positive,
            marker="o",
            markersize=3.5,
            color=color,
            linewidth=1.5,
            label=f"{name}  slope {model_metrics['calibration_slope']['estimate']:.3f}",
        )
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[1].axhline(prevalence, color="#999999", linestyle="--", linewidth=1)
    axes[2].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set(title="Discrimination", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision and recall", xlabel="Recall", ylabel="Precision")
    axes[2].set(
        title="Calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed fraction",
    )
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Locked patient-grouped out-of-fold validation", fontsize=11)
    auroc_difference = difference_metrics["auroc"]
    loss_difference = difference_metrics["log_loss"]
    figure.text(
        0.5,
        0.015,
        (
            f"Paired ΔAUROC {auroc_difference['estimate']:.5f} "
            f"[{auroc_difference['ci95'][0]:.5f}, {auroc_difference['ci95'][1]:.5f}]    "
            f"Paired Δ log loss {loss_difference['estimate']:.5f} "
            f"[{loss_difference['ci95'][0]:.5f}, {loss_difference['ci95'][1]:.5f}]"
        ),
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_validation_outputs(
    result: LockedValidationResult,
    output_root: Path,
) -> dict[str, Path]:
    """Persist DUA-private predictions and release-safe aggregate artifacts."""
    root = output_root.expanduser().resolve()
    private_dir = root / "dua_private"
    aggregate_dir = root / "aggregate_only"
    if not private_dir.is_dir() or not aggregate_dir.is_dir():
        raise FileNotFoundError("Output root must first be created by prepare_dua_output_root")

    private_predictions = private_dir / "oof_predictions_DUA_PRIVATE.npz"
    private_readme = private_dir / "README_DUA_PRIVATE.txt"
    aggregate_json = aggregate_dir / "locked_nested_validation.json"
    aggregate_pdf = aggregate_dir / "locked_nested_validation_performance.pdf"
    aggregate_png = aggregate_dir / "locked_nested_validation_performance.png"
    targets = (
        private_predictions,
        private_readme,
        aggregate_json,
        aggregate_pdf,
        aggregate_png,
    )
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Validation output targets already exist: {existing}")

    _assert_release_safe_report(result.aggregate_report)
    serialized = json.dumps(
        result.aggregate_report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    aggregate_json.write_text(serialized + "\n", encoding="utf-8")
    np.savez_compressed(
        private_predictions,
        condition_id=np.asarray(result.condition_ids, dtype=str),
        patient_id=np.asarray(result.patient_groups, dtype=str),
        outcome=result.y_true,
        outer_fold=result.outer_fold,
        additive_probability=result.additive_probability,
        patch_probability=result.patch_probability,
    )
    private_readme.write_text(
        "DUA-PRIVATE\n"
        "This directory contains row-level TB Portals identifiers, outcomes, fold assignments, "
        "and predictions. Do not copy it into the repository or release it publicly.\n",
        encoding="utf-8",
    )
    _write_performance_figure(result, aggregate_pdf, aggregate_png)
    for path in targets:
        os.chmod(path, 0o600)
    return {
        "private_predictions": private_predictions,
        "private_readme": private_readme,
        "aggregate_json": aggregate_json,
        "aggregate_pdf": aggregate_pdf,
        "aggregate_png": aggregate_png,
    }
