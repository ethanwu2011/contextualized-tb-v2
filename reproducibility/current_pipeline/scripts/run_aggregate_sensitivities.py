"""Run disclosure-safe PATCH sensitivity and support analyses without editing submission artifacts.

This is an analysis-only driver. It loads DUA-governed TB Portals rows locally, keeps all row-level
objects in memory, and writes one aggregate JSON report. Counts below SMALL_CELL_MIN are suppressed
in the serialized report. No identifiers, outcomes, treatment vectors, predictions, or loadings are
written.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from ctb.config import (
    DIED_FAILURE_NONEVENT_OUTCOMES,
    DIED_FAILURE_OUTCOMES,
    RESISTANCE_EXPECTED_DAYS,
    UNFAVORABLE_OUTCOMES,
)
from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.context import build_context_matrix
from ctb.data.conversion import classify_culture
from ctb.data.load import (
    load_biochemistry,
    load_dst,
    load_patient_cases,
    load_regimens,
)
from ctb.data.treatments import select_initial_regimens
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    RawValidationData,
    _calibration_coefficients,
    _seed,
    make_grouped_splits,
    resolve_patient_groups,
)

SMALL_CELL_MIN = 30
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ModelMode = Literal["context_only", "regimen_only", "additive"]


@dataclass(frozen=True)
class ModelRun:
    """Private in-memory predictions plus release-safe aggregate summaries."""

    probability: FloatArray
    outer_fold: IntArray
    selected_c: tuple[float, ...]
    retained_context: tuple[int, ...]
    retained_regimen: tuple[int, ...]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    return args


def _safe_count(value: int, *, threshold: int = SMALL_CELL_MIN) -> int | dict[str, Any]:
    count = int(value)
    if count < 0:
        raise ValueError("Counts must be nonnegative")
    if count < threshold:
        return {"suppressed": True, "threshold": threshold}
    return count


def _finite(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _endpoint_labels(patient_cases: pd.DataFrame, endpoint: str) -> pd.Series[Any]:
    normalized = (
        patient_cases["outcome"].astype("string").str.strip().str.casefold().replace("", pd.NA)
    )
    labels = pd.Series(pd.NA, index=patient_cases.index, dtype="boolean")
    if endpoint == "died_failure":
        event_values = DIED_FAILURE_OUTCOMES
    elif endpoint == "who_unfavorable":
        # Regimen histories encode the state as ``ltfu`` while Patient Cases spells it out.
        event_values = frozenset({*UNFAVORABLE_OUTCOMES, "lost to follow up"})
    else:
        raise ValueError(f"Unknown endpoint {endpoint!r}")
    labels.loc[normalized.isin(event_values)] = True
    labels.loc[normalized.isin(DIED_FAILURE_NONEVENT_OUTCOMES)] = False
    return pd.Series(
        labels.to_numpy(),
        index=pd.Index(patient_cases["condition_id"], name="condition_id"),
        dtype="boolean",
        name=endpoint,
    )


def _build_raw_data(
    patient_cases: pd.DataFrame,
    biochemistry: pd.DataFrame,
    initial: pd.DataFrame,
    *,
    endpoint: str,
    restrict_to: set[Any] | None = None,
) -> RawValidationData:
    labels = _endpoint_labels(patient_cases, endpoint)
    resistance = (
        patient_cases[["condition_id", "type_of_resistance"]]
        .drop_duplicates("condition_id")
        .set_index("condition_id")["type_of_resistance"]
    )
    defined = set(RESISTANCE_EXPECTED_DAYS)
    eligible: list[Any] = []
    for condition_id in patient_cases["condition_id"].drop_duplicates().tolist():
        if restrict_to is not None and condition_id not in restrict_to:
            continue
        if condition_id not in initial.index:
            continue
        drug_set = initial.at[condition_id, "drug_set"]
        if not isinstance(drug_set, set | frozenset | list | tuple) or len(drug_set) == 0:
            continue
        if resistance.get(condition_id) not in defined:
            continue
        if pd.isna(labels.get(condition_id)):
            continue
        eligible.append(condition_id)
    assessed = pd.Index(eligible, name="condition_id")
    if not assessed.is_unique or len(assessed) < 2:
        raise ValueError("Sensitivity cohort is empty or contains duplicate conditions")
    outcome = labels.reindex(assessed).astype(int)
    if outcome.nunique() != 2:
        raise ValueError("Sensitivity endpoint must contain both outcome classes")
    starts = initial["initial_start"].reindex(assessed)
    if starts.isna().any():
        raise ValueError("Sensitivity cohort contains missing initial-regimen time zero")
    context, feature_spec = build_context_matrix(
        patient_cases,
        biochemistry,
        assessed.tolist(),
        baseline_index_dates=starts,
        preserve_degenerate_columns=True,
    )
    treatments = pd.Series(
        initial["drug_set"].reindex(assessed).to_numpy(),
        index=assessed,
        name="initial_regimen",
    )
    groups = resolve_patient_groups(patient_cases, assessed)
    return RawValidationData(
        context=context.reindex(assessed),
        treatments=treatments,
        outcome=outcome,
        patient_groups=groups,
        numeric_columns=tuple(feature_spec.numeric),
        categorical_columns=tuple(feature_spec.categorical),
    )


def _design(matrices: FoldMatrices, mode: ModelMode) -> FloatArray:
    if mode == "context_only":
        design = matrices.C
    elif mode == "regimen_only":
        design = matrices.X
    elif mode == "additive":
        design = np.hstack([matrices.C, matrices.X])
    else:
        raise ValueError(f"Unknown model mode {mode!r}")
    if design.shape[1] == 0:
        raise ValueError(f"{mode} has no retained features")
    return np.asarray(design, dtype=np.float64)


def _fit_logistic(
    train: FoldMatrices,
    outcome: IntArray,
    mode: ModelMode,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(_design(train, mode), outcome)
    return model


def _probability(model: LogisticRegression, matrices: FoldMatrices, mode: ModelMode) -> FloatArray:
    probability = np.asarray(model.predict_proba(_design(matrices, mode))[:, 1], dtype=np.float64)
    if probability.shape != (matrices.n_rows,) or not np.isfinite(probability).all():
        raise ValueError(f"Invalid {mode} probability vector")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"Out-of-range {mode} probabilities")
    return probability


def _run_grouped_logistic(
    data: RawValidationData,
    mode: ModelMode,
    *,
    seed: int,
    c_grid: tuple[float, ...] = (0.1,),
    outer_splits: int = 5,
    inner_splits: int = 3,
    max_iter: int = 2000,
) -> ModelRun:
    y = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    outer = make_grouped_splits(
        y,
        groups,
        n_splits=outer_splits,
        seed=seed,
        label=f"{mode} outer validation",
    )
    oof = np.full(len(y), np.nan, dtype=np.float64)
    assignments = np.full(len(y), -1, dtype=np.int64)
    selected_values: list[float] = []
    retained_context: list[int] = []
    retained_regimen: list[int] = []
    for outer_index, (train_index, test_index) in enumerate(outer):
        if len(c_grid) == 1:
            selected_c = c_grid[0]
        else:
            inner = make_grouped_splits(
                y[train_index],
                groups[train_index],
                n_splits=inner_splits,
                seed=_seed(seed, outer_index, 17),
                label=f"{mode} inner validation fold {outer_index + 1}",
            )
            outer_context = data.context.iloc[train_index]
            outer_treatments = data.treatments.iloc[train_index]
            prepared_inner: list[tuple[FoldMatrices, FoldMatrices, IntArray, IntArray]] = []
            for inner_train, inner_valid in inner:
                preprocessor = FoldPreprocessor(
                    data.numeric_columns,
                    data.categorical_columns,
                    min_treatment_prevalence=0.01,
                )
                train_matrices = preprocessor.fit_transform(
                    outer_context.iloc[inner_train],
                    outer_treatments.iloc[inner_train],
                )
                valid_matrices = preprocessor.transform(
                    outer_context.iloc[inner_valid],
                    outer_treatments.iloc[inner_valid],
                )
                prepared_inner.append(
                    (
                        train_matrices,
                        valid_matrices,
                        y[train_index][inner_train],
                        y[train_index][inner_valid],
                    )
                )
            score_by_c: dict[float, float] = {}
            for c_position, c_value in enumerate(c_grid):
                losses: list[float] = []
                for inner_index, (
                    train_matrices,
                    valid_matrices,
                    inner_y_train,
                    inner_y_valid,
                ) in enumerate(prepared_inner):
                    model = _fit_logistic(
                        train_matrices,
                        inner_y_train,
                        mode,
                        c_value=c_value,
                        max_iter=max_iter,
                        seed=_seed(seed, outer_index, c_position, inner_index),
                    )
                    prediction = _probability(model, valid_matrices, mode)
                    losses.append(float(log_loss(inner_y_valid, prediction)))
                score_by_c[c_value] = float(np.mean(losses))
            selected_c = min(c_grid, key=lambda value: (score_by_c[value], value))
        preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=0.01,
        )
        train_matrices = preprocessor.fit_transform(
            data.context.iloc[train_index],
            data.treatments.iloc[train_index],
        )
        test_matrices = preprocessor.transform(
            data.context.iloc[test_index],
            data.treatments.iloc[test_index],
        )
        model = _fit_logistic(
            train_matrices,
            y[train_index],
            mode,
            c_value=selected_c,
            max_iter=max_iter,
            seed=_seed(seed, outer_index, 29),
        )
        oof[test_index] = _probability(model, test_matrices, mode)
        assignments[test_index] = outer_index
        selected_values.append(float(selected_c))
        retained_context.append(train_matrices.C.shape[1])
        retained_regimen.append(train_matrices.X.shape[1])
    if np.isnan(oof).any() or np.any(assignments < 0):
        raise RuntimeError(f"Incomplete {mode} out-of-fold predictions")
    return ModelRun(
        probability=oof,
        outer_fold=assignments,
        selected_c=tuple(selected_values),
        retained_context=tuple(retained_context),
        retained_regimen=tuple(retained_regimen),
    )


def _point_metrics(outcome: IntArray, probability: FloatArray) -> dict[str, float]:
    intercept, slope = _calibration_coefficients(outcome, probability)
    return {
        "auroc": _finite(roc_auc_score(outcome, probability), field="auroc"),
        "auprc": _finite(average_precision_score(outcome, probability), field="auprc"),
        "log_loss": _finite(log_loss(outcome, probability), field="log_loss"),
        "brier": _finite(brier_score_loss(outcome, probability), field="brier"),
        "calibration_intercept": _finite(intercept, field="calibration_intercept"),
        "calibration_slope": _finite(slope, field="calibration_slope"),
        "observed_to_expected": _finite(
            float(np.sum(outcome)) / float(np.sum(probability)),
            field="observed_to_expected",
        ),
    }


def _calibration_bins(
    outcome: IntArray, probability: FloatArray, n_bins: int = 10
) -> list[dict[str, Any]]:
    order = np.argsort(probability, kind="mergesort")
    bins: list[dict[str, Any]] = []
    for position, indices in enumerate(np.array_split(order, n_bins), start=1):
        if len(indices) < SMALL_CELL_MIN:
            raise ValueError("Calibration bin violates the disclosure threshold")
        bins.append(
            {
                "bin": position,
                "n": int(len(indices)),
                "mean_predicted": float(np.mean(probability[indices])),
                "observed_rate": float(np.mean(outcome[indices])),
            }
        )
    return bins


def _cluster_bootstrap_metrics(
    outcome: IntArray,
    probabilities: dict[str, FloatArray],
    patient_groups: NDArray[Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.asarray(patient_groups).astype(str)
    unique_groups = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    metric_names = ("auroc", "auprc", "log_loss", "brier")
    points = {
        name: {metric: _point_metrics(outcome, prob)[metric] for metric in metric_names}
        for name, prob in probabilities.items()
    }
    draws: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metric_names} for name in probabilities
    }
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(n_resamples):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        rows = np.concatenate([rows_by_group[index] for index in sampled])
        y = outcome[rows]
        if len(np.unique(y)) != 2:
            continue
        valid += 1
        for name, probability in probabilities.items():
            p = probability[rows]
            values = {
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "log_loss": float(log_loss(y, p)),
                "brier": float(brier_score_loss(y, p)),
            }
            for metric, value in values.items():
                draws[name][metric].append(value)
    if valid < max(100, n_resamples // 2):
        raise RuntimeError("Too few valid patient-cluster bootstrap resamples")
    summary: dict[str, Any] = {
        "resampling_unit": "patient",
        "requested_resamples": n_resamples,
        "valid_resamples": valid,
        "models": {},
    }
    for name in probabilities:
        summary["models"][name] = {
            metric: {
                "estimate": points[name][metric],
                "ci95": [
                    float(np.percentile(draws[name][metric], 2.5)),
                    float(np.percentile(draws[name][metric], 97.5)),
                ],
            }
            for metric in metric_names
        }
    comparator = "additive"
    if comparator in probabilities:
        summary["paired_difference_vs_additive"] = {}
        for name in probabilities:
            if name == comparator:
                continue
            summary["paired_difference_vs_additive"][name] = {}
            for metric in metric_names:
                difference = np.asarray(draws[name][metric]) - np.asarray(draws[comparator][metric])
                summary["paired_difference_vs_additive"][name][metric] = {
                    "estimate": points[name][metric] - points[comparator][metric],
                    "ci95": [
                        float(np.percentile(difference, 2.5)),
                        float(np.percentile(difference, 97.5)),
                    ],
                }
    return summary


def _cohort_summary(data: RawValidationData) -> dict[str, Any]:
    y = np.asarray(data.outcome, dtype=np.int64)
    return {
        "conditions": int(len(y)),
        "patients": int(data.patient_groups.nunique()),
        "events": int(y.sum()),
        "observed_non_events": int(len(y) - y.sum()),
        "event_rate": float(np.mean(y)),
        "raw_context_columns": int(data.context.shape[1]),
    }


def _model_summary(data: RawValidationData, run: ModelRun) -> dict[str, Any]:
    y = np.asarray(data.outcome, dtype=np.int64)
    return {
        "metrics": _point_metrics(y, run.probability),
        "calibration_bins": _calibration_bins(y, run.probability),
        "selected_c_by_outer_fold": list(run.selected_c),
        "retained_context_features_by_outer_fold": list(run.retained_context),
        "retained_regimen_features_by_outer_fold": list(run.retained_regimen),
    }


def _initial_policy_report(
    default: pd.DataFrame,
    period: pd.DataFrame,
    activities: pd.DataFrame,
) -> tuple[dict[str, Any], set[Any]]:
    def public_selection_report(frame: pd.DataFrame) -> dict[str, Any]:
        raw = frame.attrs.get("selection_report", {})
        return {
            key: _safe_count(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else value
            for key, value in raw.items()
        }

    common = default.index.intersection(period.index).intersection(activities.index)
    default_nonempty = default.loc[common, "drug_set"].map(len).gt(0)
    period_nonempty = period.loc[common, "drug_set"].map(len).gt(0)
    activities_nonempty = activities.loc[common, "drug_set"].map(len).gt(0)
    eligible = default_nonempty & period_nonempty & activities_nonempty
    common_eligible = common[eligible]
    same_drugs = pd.Series(
        [
            period.at[condition_id, "drug_set"] == activities.at[condition_id, "drug_set"]
            for condition_id in common_eligible
        ],
        index=common_eligible,
        dtype=bool,
    )
    period_start = pd.to_numeric(period.loc[common_eligible, "initial_start"], errors="coerce")
    activities_start = pd.to_numeric(
        activities.loc[common_eligible, "initial_start"], errors="coerce"
    )
    same_starts = period_start.eq(activities_start) | (
        period_start.isna() & activities_start.isna()
    )
    invariant = same_drugs & same_starts
    invariant_ids = set(common_eligible[invariant].tolist())
    report = {
        "policy_definitions": {
            "default": "lexicographic period_start then activities_period_start",
            "period_only": "unique earliest nonempty drug set under period_start; ambiguous earliest ties excluded",
            "activities_only": "unique earliest nonempty drug set under activities_period_start; ambiguous earliest ties excluded",
            "strict_invariant": "period-only and activities-only select the same drug set and start",
        },
        "period_only_selection": public_selection_report(period),
        "activities_only_selection": public_selection_report(activities),
        "valid_nonempty_default": int(default["drug_set"].map(len).gt(0).sum()),
        "valid_nonempty_period_only": int(period["drug_set"].map(len).gt(0).sum()),
        "valid_nonempty_activities_only": int(activities["drug_set"].map(len).gt(0).sum()),
        "common_valid_nonempty": int(len(common_eligible)),
        "same_selected_drug_set": int(same_drugs.sum()),
        "same_selected_start": int(same_starts.sum()),
        "strict_policy_invariant": int(invariant.sum()),
        "different_selected_drug_set": _safe_count(int((~same_drugs).sum())),
        "different_selected_start": _safe_count(int((~same_starts).sum())),
        "small_cell_suppression_threshold": SMALL_CELL_MIN,
    }
    return report, invariant_ids


def _select_single_field_unambiguous(
    regimens: pd.DataFrame,
    *,
    start_column: str,
) -> pd.DataFrame:
    """Select a unique earliest drug set under one start field and exclude ambiguous ties.

    The production selector correctly fails closed when a single field creates an earliest-start
    tie containing different drug sets. For sensitivity analysis, this helper preserves that
    caution by leaving those conditions without an initial regimen rather than inventing a tie
    breaker. It is intentionally local to this analysis-only driver.
    """
    if start_column not in regimens.columns:
        raise KeyError(f"Missing regimen start field {start_column!r}")
    frame = regimens[["condition_id", "drug_set", start_column]].copy()
    frame["_start"] = pd.to_numeric(frame[start_column], errors="coerce")
    nonempty = frame["drug_set"].map(len).gt(0)
    valid = frame["_start"].notna() & np.isfinite(frame["_start"]) & frame["_start"].mod(1).eq(0)
    invalid_nonempty = nonempty & ~valid
    if invalid_nonempty.any():
        raise ValueError(
            f"{start_column} contains {int(invalid_nonempty.sum())} unsafe nonempty regimen rows"
        )
    conditions = pd.Index(frame["condition_id"].drop_duplicates(), name="condition_id")
    selected = pd.DataFrame(
        {
            "drug_set": [frozenset() for _ in range(len(conditions))],
            "initial_start": pd.Series(
                [pd.NA] * len(conditions), index=conditions, dtype="Float64"
            ),
        },
        index=conditions,
    )
    ambiguous_ties = 0
    identical_ties = 0
    candidate = frame.loc[nonempty & valid].copy()
    for condition_id, group in candidate.groupby("condition_id", sort=True):
        minimum = group["_start"].min()
        earliest = group.loc[group["_start"].eq(minimum)]
        distinct_sets = list(dict.fromkeys(earliest["drug_set"].tolist()))
        if len(distinct_sets) > 1:
            ambiguous_ties += 1
            continue
        if len(earliest) > 1:
            identical_ties += 1
        selected.at[condition_id, "drug_set"] = distinct_sets[0]
        selected.at[condition_id, "initial_start"] = float(minimum)
    selected.attrs["selection_report"] = {
        "start_column": start_column,
        "ambiguous_earliest_tie_conditions_excluded": ambiguous_ties,
        "identical_earliest_tie_conditions_retained": identical_ties,
        "conditions_without_unambiguous_initial_regimen": int(
            selected["drug_set"].map(len).eq(0).sum()
        ),
    }
    return selected


def _strict_baseline_positive_ids(dst: pd.DataFrame) -> set[Any]:
    """Return conditions with an interpretable positive culture on or before day zero.

    Conditions with positive and negative culture statuses on the same day are excluded in full,
    matching the production conversion quarantine without constructing row-level endpoint output.
    """
    required = {"condition_id", "specimen_collection_date", "cultureresults"}
    missing = sorted(required.difference(dst.columns))
    if missing:
        raise KeyError(f"Culture eligibility requires columns {missing}")
    frame = dst[list(required)].copy()
    raw_day = frame["specimen_collection_date"]
    day = pd.to_numeric(raw_day, errors="coerce")
    boolean = raw_day.astype(object).map(lambda value: isinstance(value, bool | np.bool_))
    valid_day = day.notna() & np.isfinite(day) & day.mod(1).eq(0) & ~boolean
    frame["_day"] = day.where(valid_day)
    frame["_status"] = frame["cultureresults"].map(classify_culture)
    interpretable = frame.loc[frame["_day"].notna() & frame["_status"].ne("unknown")].copy()
    status_count = interpretable.groupby(["condition_id", "_day"])["_status"].transform("nunique")
    conflict_ids = set(interpretable.loc[status_count.gt(1), "condition_id"].tolist())
    eligible = interpretable.loc[
        ~interpretable["condition_id"].isin(conflict_ids)
        & interpretable["_status"].eq("positive")
        & interpretable["_day"].le(0)
    ]
    return set(eligible["condition_id"].tolist())


def _full_descriptive_matrices(data: RawValidationData) -> FoldMatrices:
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=0.01,
    )
    return preprocessor.fit_transform(data.context, data.treatments)


def _treatment_design_report(data: RawValidationData) -> dict[str, Any]:
    matrices = _full_descriptive_matrices(data)
    x = matrices.X
    y = np.asarray(data.outcome, dtype=np.int64)
    names = matrices.treatment_feature_names
    if x.shape[1] == 0:
        raise ValueError("No treatment features were retained")
    centered = x - x.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=1)
    nonconstant = scale > 0
    standardized = centered[:, nonconstant] / scale[nonconstant]
    singular = np.linalg.svd(standardized, compute_uv=False)
    tolerance = np.finfo(float).eps * max(standardized.shape) * singular[0]
    positive = singular[singular > tolerance]
    rank = int(len(positive))
    condition_number = float(positive[0] / positive[-1]) if len(positive) else float("inf")

    component_rows: list[dict[str, Any]] = []
    for column, name in enumerate(names):
        treated = x[:, column].astype(bool)
        n_treated = int(treated.sum())
        n_control = int((~treated).sum())
        row: dict[str, Any] = {
            "component": name,
            "treated_n": _safe_count(n_treated),
            "control_n": _safe_count(n_control),
            "marginal_prevalence": float(treated.mean()),
            "support_eligible": n_treated >= SMALL_CELL_MIN and n_control >= SMALL_CELL_MIN,
        }
        if row["support_eligible"]:
            row["treated_events"] = _safe_count(int(y[treated].sum()))
            row["control_events"] = _safe_count(int(y[~treated].sum()))
        component_rows.append(row)

    supported_pairs: list[dict[str, Any]] = []
    for left in range(x.shape[1]):
        for right in range(left + 1, x.shape[1]):
            joint_counts = [
                int(np.sum((x[:, left] == left_value) & (x[:, right] == right_value)))
                for left_value in (0.0, 1.0)
                for right_value in (0.0, 1.0)
            ]
            if min(joint_counts) < SMALL_CELL_MIN:
                continue
            correlation = float(np.corrcoef(x[:, left], x[:, right])[0, 1])
            if abs(correlation) >= 0.80:
                supported_pairs.append(
                    {
                        "component_a": names[left],
                        "component_b": names[right],
                        "phi": correlation,
                        "minimum_joint_cell": min(joint_counts),
                    }
                )
    supported_pairs.sort(key=lambda row: abs(row["phi"]), reverse=True)

    vector_counts = pd.Series([tuple(row.astype(int).tolist()) for row in x]).value_counts()
    return {
        "scope": "full-cohort descriptive exposure matrix; no causal identification claim",
        "conditions": int(len(x)),
        "component_count": int(x.shape[1]),
        "standardized_rank": rank,
        "standardized_condition_number": condition_number,
        "smallest_nonzero_singular_value": float(positive[-1]),
        "largest_singular_value": float(positive[0]),
        "unique_observed_regimen_vectors": int(len(vector_counts)),
        "vectors_with_at_least_30_conditions": int((vector_counts >= SMALL_CELL_MIN).sum()),
        "share_in_ten_most_common_vectors": float(vector_counts.iloc[:10].sum() / len(x)),
        "components": component_rows,
        "high_correlation_pairs_with_all_joint_cells_at_least_30": supported_pairs,
    }


def _conditional_support_report(
    data: RawValidationData,
    *,
    seed: int,
    outer_splits: int = 5,
) -> dict[str, Any]:
    y_outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    splits = make_grouped_splits(
        y_outcome,
        groups,
        n_splits=outer_splits,
        seed=seed,
        label="conditional treatment support",
    )
    global_names: set[str] | None = None
    fold_data: list[tuple[IntArray, IntArray, FoldMatrices, FoldMatrices]] = []
    for train_index, test_index in splits:
        preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=0.01,
        )
        train = preprocessor.fit_transform(
            data.context.iloc[train_index],
            data.treatments.iloc[train_index],
        )
        test = preprocessor.transform(
            data.context.iloc[test_index],
            data.treatments.iloc[test_index],
        )
        names = set(train.treatment_feature_names)
        global_names = names if global_names is None else global_names.intersection(names)
        fold_data.append((train_index, test_index, train, test))
    if not global_names:
        raise ValueError("No treatment component is retained in every support fold")

    results: list[dict[str, Any]] = []
    for component in sorted(global_names):
        propensity = np.full(len(y_outcome), np.nan, dtype=np.float64)
        observed = np.full(len(y_outcome), -1, dtype=np.int64)
        for fold_number, (_train_index, test_index, train, test) in enumerate(fold_data):
            position = train.treatment_feature_names.index(component)
            train_receipt = train.X[:, position].astype(np.int64)
            test_receipt = test.X[:, position].astype(np.int64)
            if len(np.unique(train_receipt)) != 2:
                raise ValueError(f"Support fold lacks both classes for {component}")
            train_other = np.delete(train.X, position, axis=1)
            test_other = np.delete(test.X, position, axis=1)
            train_design = np.hstack([train.C, train_other])
            test_design = np.hstack([test.C, test_other])
            model = LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=_seed(seed, fold_number, position),
            )
            model.fit(train_design, train_receipt)
            propensity[test_index] = model.predict_proba(test_design)[:, 1]
            observed[test_index] = test_receipt
        if np.isnan(propensity).any() or np.any(observed < 0):
            raise RuntimeError(f"Incomplete support predictions for {component}")
        treated = observed == 1
        control = ~treated
        n_treated = int(treated.sum())
        n_control = int(control.sum())
        if n_treated < SMALL_CELL_MIN or n_control < SMALL_CELL_MIN:
            continue
        clipped = np.clip(propensity, 1e-6, 1 - 1e-6)
        treated_weights = 1.0 / clipped[treated]
        control_weights = 1.0 / (1.0 - clipped[control])
        treated_ess = float(np.square(treated_weights.sum()) / np.square(treated_weights).sum())
        control_ess = float(np.square(control_weights.sum()) / np.square(control_weights).sum())
        results.append(
            {
                "component": component,
                "treated_n": n_treated,
                "control_n": n_control,
                "selection_auroc": float(roc_auc_score(observed, propensity)),
                "fraction_propensity_0.10_to_0.90": float(
                    np.mean((propensity >= 0.10) & (propensity <= 0.90))
                ),
                "treated_ate_weight_ess": treated_ess,
                "control_ate_weight_ess": control_ess,
                "treated_events": _safe_count(int(y_outcome[treated].sum())),
                "control_events": _safe_count(int(y_outcome[control].sum())),
                "passes_descriptive_support_gate": bool(
                    treated_ess >= 50
                    and control_ess >= 50
                    and np.mean((propensity >= 0.10) & (propensity <= 0.90)) >= 0.50
                ),
            }
        )
    results.sort(key=lambda row: row["selection_auroc"], reverse=True)
    selection_values = np.asarray([row["selection_auroc"] for row in results], dtype=float)
    overlap_values = np.asarray(
        [row["fraction_propensity_0.10_to_0.90"] for row in results], dtype=float
    )
    return {
        "model": "five-fold patient-grouped fold-local logistic receipt model using context and other regimen components",
        "interpretation": "diagnostic of conditional component support, not a causal propensity model",
        "components_assessed": int(len(results)),
        "components_passing_descriptive_support_gate": int(
            sum(row["passes_descriptive_support_gate"] for row in results)
        ),
        "median_selection_auroc": float(np.median(selection_values)),
        "maximum_selection_auroc": float(np.max(selection_values)),
        "median_overlap_fraction": float(np.median(overlap_values)),
        "components": results,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    forbidden = {
        "condition_id",
        "patient_id",
        "patient_ids",
        "condition_ids",
        "prediction",
        "predictions",
        "probability",
        "probabilities",
        "outcome",
        "outcomes",
        "y_true",
        "row_index",
    }

    def audit(value: Any, location: str = "report") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    raise ValueError(f"Forbidden row-level field at {location}.{key}")
                audit(item, f"{location}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                audit(item, f"{location}[{index}]")

    audit(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    dst = load_dst(raw_dir)

    default_initial = select_initial_regimens(regimens)
    period_initial = _select_single_field_unambiguous(
        regimens,
        start_column="period_start",
    )
    activities_initial = _select_single_field_unambiguous(
        regimens,
        start_column="activities_period_start",
    )
    policy_report, invariant_ids = _initial_policy_report(
        default_initial,
        period_initial,
        activities_initial,
    )

    baseline_positive_ids = _strict_baseline_positive_ids(dst)

    base = _build_raw_data(
        patient_cases,
        biochemistry,
        default_initial,
        endpoint="died_failure",
    )
    period = _build_raw_data(
        patient_cases,
        biochemistry,
        period_initial,
        endpoint="died_failure",
    )
    activities = _build_raw_data(
        patient_cases,
        biochemistry,
        activities_initial,
        endpoint="died_failure",
    )
    invariant = _build_raw_data(
        patient_cases,
        biochemistry,
        default_initial,
        endpoint="died_failure",
        restrict_to=invariant_ids,
    )
    baseline_culture = _build_raw_data(
        patient_cases,
        biochemistry,
        default_initial,
        endpoint="died_failure",
        restrict_to=baseline_positive_ids,
    )
    who_unfavorable = _build_raw_data(
        patient_cases,
        biochemistry,
        default_initial,
        endpoint="who_unfavorable",
    )

    seed = LockedValidationConfig().seed
    base_runs = {
        mode: _run_grouped_logistic(base, mode, seed=seed)
        for mode in ("context_only", "regimen_only", "additive")
    }
    sensitivity_data = {
        "period_only": period,
        "activities_only": activities,
        "strict_policy_invariant": invariant,
        "strict_baseline_culture_positive": baseline_culture,
        "who_unfavorable": who_unfavorable,
    }
    sensitivity_runs = {
        name: _run_grouped_logistic(data, "additive", seed=seed)
        for name, data in sensitivity_data.items()
    }
    y_base = np.asarray(base.outcome, dtype=np.int64)
    base_probabilities = {name: run.probability for name, run in base_runs.items()}
    bootstrap = _cluster_bootstrap_metrics(
        y_base,
        base_probabilities,
        base.patient_groups.to_numpy(dtype=str),
        n_resamples=args.bootstrap_resamples,
        seed=_seed(seed, 5001),
    )

    report = {
        "report_schema_version": 1,
        "analysis_id": "patch_nonnegotiable_aggregate_sensitivities",
        "analysis_status": "analysis-only; manuscript and submission artifacts untouched",
        "interpretation": "prognostic and treatment-associated sensitivity analyses; no causal effect claim",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "governance": {
            "dua_acknowledged": bool(args.dua_acknowledge),
            "contains_identifiers": False,
            "contains_row_level_data": False,
            "small_cell_minimum": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
        "model_validation_contract": {
            "outer_splitter": "five-fold StratifiedGroupKFold",
            "grouping_unit": "patient",
            "preprocessing_scope": "fit on each outer training fold only",
            "logistic_c": 0.1,
            "logistic_c_basis": "locked value selected in every outer fold of the reference nested validation; not retuned per sensitivity",
        },
        "initial_regimen_policy": policy_report,
        "cohorts": {
            "primary_died_failure": _cohort_summary(base),
            **{name: _cohort_summary(data) for name, data in sensitivity_data.items()},
        },
        "primary_ablation": {name: _model_summary(base, run) for name, run in base_runs.items()},
        "primary_patient_clustered_uncertainty": bootstrap,
        "additive_sensitivities": {
            name: _model_summary(sensitivity_data[name], run)
            for name, run in sensitivity_runs.items()
        },
        "treatment_design": _treatment_design_report(base),
        "conditional_component_support": _conditional_support_report(base, seed=seed),
        "limitations_requiring_human_adjudication": [
            "period_start and activities_period_start disagree for many source rows; code can quantify policy sensitivity but cannot decide which field is clinically authoritative",
            "most Patient Cases context fields have no event timestamp, so their pre-treatment status cannot be proven from this export",
            "component-support diagnostics describe observed prescribing support and do not identify causal drug effects",
            "Brazil cannot be rerun because the access-controlled harmonized source file is not present locally",
        ],
    }
    _write_report(output, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "aggregate_report": str(output),
                "contains_row_level_data": False,
                "submission_artifacts_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
