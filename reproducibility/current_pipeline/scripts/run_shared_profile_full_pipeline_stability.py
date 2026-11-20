"""Audit end-to-end sampling stability of the published shared-loading PATCH K=5 model.

The published clinical profiles use one simplex loading map for both the regimen coefficients and
the prognostic intercept. This analysis refits that exact architecture on the endpoint-locked
cohort. Every patient-cluster bootstrap repeats preprocessing and model fitting. Patient-level
objects remain in memory. Only disclosure-safe aggregate summaries are written.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    RawValidationData,
    _seed,
    build_schema6_raw_validation_data,
)
from ctb.utils.reproducibility import set_global_seed
from run_aggregate_sensitivities import SMALL_CELL_MIN, _safe_count, _write_report

K = 5
REFERENCE_SEED = 0
OPTIMIZATION_SEEDS = (0, 1, 2, 3, 4, 5)
CANONICAL_OTHER = "drug_other_second_line"
PUBLISHED_CONFIG = TrainConfig(
    epochs=2000,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)


@dataclass(frozen=True)
class SharedFit:
    """Private evaluation of one shared-loading fit on the locked reference rows."""

    seed: int
    treatment_names: tuple[str, ...]
    canonical_beta: np.ndarray
    intercept_archetypes: np.ndarray
    loadings: np.ndarray
    canonical_beta_surface: np.ndarray
    prognostic_intercept: np.ndarray
    fitted_probability: np.ndarray


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20)
    args = parser.parse_args()
    if args.resamples < 20:
        parser.error("--resamples must be at least 20")
    return args


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        raise ValueError("Cosine similarity is unidentified for a zero vector")
    return float(np.dot(a, b) / denominator)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        raise ValueError("Pearson correlation is unidentified for a constant vector")
    return float(np.corrcoef(a, b)[0, 1])


def _summary(values: list[float] | np.ndarray) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Summary values must be a finite nonempty vector")
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _canonicalize_beta(
    beta: np.ndarray,
    current_names: tuple[str, ...],
    canonical_names: tuple[str, ...],
) -> np.ndarray:
    if CANONICAL_OTHER not in current_names or CANONICAL_OTHER not in canonical_names:
        raise ValueError("Every regimen vocabulary must contain the other-treatment bucket")
    if beta.ndim != 2 or beta.shape[1] != len(current_names):
        raise ValueError("Archetype coefficients and regimen vocabulary are misaligned")
    positions = {name: position for position, name in enumerate(current_names)}
    other_position = positions[CANONICAL_OTHER]
    return np.asarray(
        np.column_stack([beta[:, positions.get(name, other_position)] for name in canonical_names]),
        dtype=np.float64,
    )


def _fit_shared(
    training: FoldMatrices,
    evaluation: FoldMatrices,
    outcome: np.ndarray,
    *,
    canonical_names: tuple[str, ...],
    seed: int,
) -> SharedFit:
    if training.context_feature_names != evaluation.context_feature_names:
        raise ValueError("Training and evaluation context matrices are misaligned")
    if training.treatment_feature_names != evaluation.treatment_feature_names:
        raise ValueError("Training and evaluation regimen matrices are misaligned")
    model = LinearArchetypeContextualized(
        training.C.shape[1],
        training.X.shape[1],
        n_archetypes=K,
        simplex=True,
        seed=seed,
    )
    model = train(
        model,
        training.C,
        training.X,
        outcome.astype(np.float64),
        PUBLISHED_CONFIG,
        seed=seed,
    )
    raw = model.archetypes.detach().cpu().numpy().astype(np.float64)
    canonical_beta = _canonicalize_beta(
        raw[:, : training.X.shape[1]],
        training.treatment_feature_names,
        canonical_names,
    )
    intercepts = np.asarray(raw[:, training.X.shape[1]], dtype=np.float64)
    loadings = np.asarray(model.predict_loadings(evaluation.C), dtype=np.float64)
    beta_surface = np.asarray(loadings @ canonical_beta, dtype=np.float64)
    prognostic_intercept = np.asarray(loadings @ intercepts, dtype=np.float64)
    fitted_probability = np.asarray(
        model.predict_proba(evaluation.C, evaluation.X), dtype=np.float64
    )
    arrays = (
        canonical_beta,
        intercepts,
        loadings,
        beta_surface,
        prognostic_intercept,
        fitted_probability,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Shared fit contains a non-finite value")
    if not np.allclose(loadings.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Shared fit loadings do not sum to one")
    return SharedFit(
        seed=seed,
        treatment_names=canonical_names,
        canonical_beta=canonical_beta,
        intercept_archetypes=intercepts,
        loadings=loadings,
        canonical_beta_surface=beta_surface,
        prognostic_intercept=prognostic_intercept,
        fitted_probability=fitted_probability,
    )


def _alignment_order(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError("Profile alignment requires equal two-dimensional coefficient matrices")
    similarities = np.asarray(
        [[_cosine(reference[row], current[column]) for column in range(K)] for row in range(K)],
        dtype=np.float64,
    )
    reference_rows, current_columns = linear_sum_assignment(-similarities)
    return np.asarray(current_columns[np.argsort(reference_rows)], dtype=np.int64)


def _align(reference: SharedFit, current: SharedFit) -> SharedFit:
    order = _alignment_order(reference.canonical_beta, current.canonical_beta)
    beta = current.canonical_beta[order]
    intercepts = current.intercept_archetypes[order]
    loadings = current.loadings[:, order]
    return SharedFit(
        seed=current.seed,
        treatment_names=current.treatment_names,
        canonical_beta=beta,
        intercept_archetypes=intercepts,
        loadings=loadings,
        canonical_beta_surface=np.asarray(loadings @ beta, dtype=np.float64),
        prognostic_intercept=np.asarray(loadings @ intercepts, dtype=np.float64),
        fitted_probability=current.fitted_probability,
    )


def _soft_event_rates(loadings: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    numerator = np.sum(loadings * outcome[:, None], axis=0)
    denominator = np.sum(loadings, axis=0)
    return np.asarray(numerator / denominator, dtype=np.float64)


def _pair_metrics(
    reference: SharedFit, current: SharedFit, outcome: np.ndarray
) -> dict[str, float]:
    reference_rates = _soft_event_rates(reference.loadings, outcome)
    current_rates = _soft_event_rates(current.loadings, outcome)
    rank = float(spearmanr(reference_rates, current_rates).statistic)
    return {
        "canonical_beta_surface_cosine": _cosine(
            reference.canonical_beta_surface, current.canonical_beta_surface
        ),
        "prognostic_intercept_pearson": _pearson(
            reference.prognostic_intercept, current.prognostic_intercept
        ),
        "fitted_probability_pearson": _pearson(
            reference.fitted_probability, current.fitted_probability
        ),
        "hard_profile_ari_secondary": float(
            adjusted_rand_score(reference.loadings.argmax(axis=1), current.loadings.argmax(axis=1))
        ),
        "mean_matched_soft_loading_pearson": float(
            np.mean(
                [
                    _pearson(reference.loadings[:, profile], current.loadings[:, profile])
                    for profile in range(K)
                ]
            )
        ),
        "soft_event_rate_rank_spearman": rank,
        "maximum_absolute_soft_event_rate_change": float(
            np.max(np.abs(reference_rates - current_rates))
        ),
    }


def _patient_bootstrap_rows(groups: np.ndarray, *, resamples: int, seed: int) -> list[np.ndarray]:
    normalized = np.asarray(groups).astype(str)
    unique = np.unique(normalized)
    grouped_rows = [np.flatnonzero(normalized == group) for group in unique]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [grouped_rows[position] for position in rng.integers(0, len(unique), len(unique))]
        ).astype(np.int64)
        for _ in range(resamples)
    ]


def _reference_profile_summary(reference: SharedFit, outcome: np.ndarray) -> list[dict[str, Any]]:
    rates = _soft_event_rates(reference.loadings, outcome)
    hard = reference.loadings.argmax(axis=1)
    order = np.argsort(rates)
    rows: list[dict[str, Any]] = []
    for risk_rank, profile in enumerate(order, start=1):
        selected = hard == profile
        hard_count = int(selected.sum())
        rows.append(
            {
                "risk_rank": risk_rank,
                "fitted_profile": int(profile + 1),
                "soft_effective_count": float(reference.loadings[:, profile].sum()),
                "soft_weighted_event_rate": float(rates[profile]),
                "hard_count_secondary": _safe_count(hard_count),
                "hard_event_rate_secondary": (
                    float(outcome[selected].mean()) if hard_count >= SMALL_CELL_MIN else None
                ),
            }
        )
    return rows


def _optimization_seed_stability(
    reference: SharedFit,
    fits: list[SharedFit],
    outcome: np.ndarray,
) -> dict[str, Any]:
    aligned = [reference if fit.seed == reference.seed else _align(reference, fit) for fit in fits]
    rows = [_pair_metrics(left, right, outcome) for left, right in combinations(aligned, 2)]
    return {
        "seeds": list(OPTIMIZATION_SEEDS),
        "seed_pairs": len(rows),
        "global": {key: _summary([row[key] for row in rows]) for key in rows[0]},
    }


def _run_bootstrap(
    data: RawValidationData,
    reference: SharedFit,
    reference_matrices: FoldMatrices,
    *,
    resamples: int,
    config: LockedValidationConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    samples = _patient_bootstrap_rows(
        groups,
        resamples=resamples,
        seed=_seed(config.seed, 5, 8801),
    )
    rows: list[dict[str, float]] = []
    coefficient_draws: list[np.ndarray] = []
    profile_cosines: list[list[float]] = []
    loading_correlations: list[list[float]] = []
    occupancies: list[list[float]] = []
    changed_context_vocabulary = 0
    changed_treatment_vocabulary = 0
    for replicate, sample in enumerate(samples, start=1):
        preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=config.min_treatment_prevalence,
        )
        training = preprocessor.fit_transform(
            data.context.iloc[sample], data.treatments.iloc[sample]
        )
        evaluation = preprocessor.transform(data.context, data.treatments)
        changed_context_vocabulary += int(
            evaluation.context_feature_names != reference_matrices.context_feature_names
        )
        changed_treatment_vocabulary += int(
            evaluation.treatment_feature_names != reference_matrices.treatment_feature_names
        )
        fitted = _fit_shared(
            training,
            evaluation,
            outcome[sample],
            canonical_names=reference.treatment_names,
            seed=REFERENCE_SEED,
        )
        aligned = _align(reference, fitted)
        rows.append(_pair_metrics(reference, aligned, outcome))
        coefficient_draws.append(aligned.canonical_beta)
        profile_cosines.append(
            [
                _cosine(reference.canonical_beta[profile], aligned.canonical_beta[profile])
                for profile in range(K)
            ]
        )
        loading_correlations.append(
            [
                _pearson(reference.loadings[:, profile], aligned.loadings[:, profile])
                for profile in range(K)
            ]
        )
        occupancies.append([float(aligned.loadings[:, profile].sum()) for profile in range(K)])
        logging.getLogger(__name__).warning(
            "Shared K=5 full-pipeline bootstrap %d/%d complete", replicate, resamples
        )
    profile_cosine_array = np.asarray(profile_cosines, dtype=np.float64)
    loading_array = np.asarray(loading_correlations, dtype=np.float64)
    occupancy_array = np.asarray(occupancies, dtype=np.float64)
    return (
        {
            "resamples": resamples,
            "resampling_unit": "patient",
            "repeated_steps": [
                "numeric imputation and scaling",
                "categorical vocabulary and encoding",
                "one-percent regimen vocabulary",
                "shared-loading K=5 model fit",
                "profile alignment on canonical regimen coefficients",
            ],
            "optimization_seed_scope": (
                "published deterministic seed 0 fixed across resamples; optimization-seed "
                "sensitivity reported separately"
            ),
            "changed_context_vocabulary_resamples": changed_context_vocabulary,
            "changed_regimen_vocabulary_resamples": changed_treatment_vocabulary,
            "global": {key: _summary([row[key] for row in rows]) for key in rows[0]},
            "profiles": [
                {
                    "profile": profile + 1,
                    "canonical_beta_cosine_to_reference": _summary(
                        profile_cosine_array[:, profile]
                    ),
                    "soft_loading_pearson_to_reference": _summary(loading_array[:, profile]),
                    "soft_effective_occupancy": _summary(occupancy_array[:, profile]),
                }
                for profile in range(K)
            ],
        },
        np.stack(coefficient_draws),
    )


def _regimen_label(feature: str) -> str:
    if feature == CANONICAL_OTHER:
        return "Other second-line"
    return unquote(feature.removeprefix("drug_"))


def _cell_uncertainty(
    reference: SharedFit,
    draws: np.ndarray,
    matrices: FoldMatrices,
) -> dict[str, Any]:
    if draws.shape[1:] != reference.canonical_beta.shape:
        raise ValueError("Bootstrap coefficient draws are misaligned")
    rows: list[dict[str, Any]] = []
    supported = 0
    interval_excludes_zero = 0
    sign_consistent = 0
    for profile in range(K):
        weights = reference.loadings[:, profile]
        for component, feature in enumerate(reference.treatment_names):
            receipt = matrices.X[:, component]
            treated_support = float(np.dot(weights, receipt))
            untreated_support = float(np.dot(weights, 1.0 - receipt))
            is_supported = min(treated_support, untreated_support) >= SMALL_CELL_MIN
            values = draws[:, profile, component]
            reference_value = float(reference.canonical_beta[profile, component])
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            excludes_zero = bool(lower > 0 or upper < 0)
            sign_fraction = float(np.mean(np.sign(values) == np.sign(reference_value)))
            consistent = bool(sign_fraction >= 0.8)
            if is_supported:
                supported += 1
                interval_excludes_zero += int(excludes_zero)
                sign_consistent += int(consistent)
            rows.append(
                {
                    "profile": profile + 1,
                    "regimen_feature": feature,
                    "regimen_label": _regimen_label(feature),
                    "support_status": "supported" if is_supported else "suppressed",
                    "soft_treated_effective_count": (
                        round(treated_support, 1)
                        if is_supported
                        else _safe_count(math.floor(treated_support))
                    ),
                    "soft_untreated_effective_count": (
                        round(untreated_support, 1)
                        if is_supported
                        else _safe_count(math.floor(untreated_support))
                    ),
                    "reference_shared_logit_coefficient": (
                        reference_value if is_supported else None
                    ),
                    "bootstrap_median": float(np.median(values)) if is_supported else None,
                    "bootstrap_ci95": [lower, upper] if is_supported else None,
                    "bootstrap_sign_agreement_fraction": (sign_fraction if is_supported else None),
                    "interval_excludes_zero": excludes_zero if is_supported else None,
                    "sign_agreement_at_least_0.80": consistent if is_supported else None,
                }
            )
    total = K * len(reference.treatment_names)
    return {
        "cells_total": total,
        "cells_supported_by_soft_marginal_gate": supported,
        "cells_suppressed_by_soft_marginal_gate": total - supported,
        "supported_cells_with_interval_excluding_zero": interval_excludes_zero,
        "supported_cells_with_sign_agreement_at_least_0.80": sign_consistent,
        "support_gate": (
            "at least 30 soft effective exposed and 30 soft effective unexposed observations "
            "within the reference profile"
        ),
        "conditional_support_context": (
            "No retained regimen component passed the separate prespecified conditional-support "
            "gate. These intervals quantify sampling and fitting stability, not treatment-effect "
            "identification."
        ),
        "cells": rows,
    }


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    logging.getLogger("ctb").setLevel(logging.ERROR)
    config = LockedValidationConfig()
    set_global_seed(REFERENCE_SEED)
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)

    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=config.min_treatment_prevalence,
    )
    matrices = preprocessor.fit_transform(data.context, data.treatments)
    fits = [
        _fit_shared(
            matrices,
            matrices,
            outcome,
            canonical_names=matrices.treatment_feature_names,
            seed=seed,
        )
        for seed in OPTIMIZATION_SEEDS
    ]
    reference = fits[OPTIMIZATION_SEEDS.index(REFERENCE_SEED)]
    bootstrap, draws = _run_bootstrap(
        data,
        reference,
        matrices,
        resamples=args.resamples,
        config=config,
    )
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "analysis_id": "shared_patch_k5_full_pipeline_profile_stability",
        "analysis_role": "analysis-only remediation; no submission artifact was modified",
        "estimand": (
            "descriptive shared prognostic and context-varying recorded regimen-outcome "
            "association representation"
        ),
        "cohort": {
            "assessed_conditions": len(outcome),
            "patients": int(len(np.unique(groups))),
            "events": int(outcome.sum()),
            "observed_non_events": int(len(outcome) - outcome.sum()),
            "retained_context_features": matrices.C.shape[1],
            "retained_regimen_features": matrices.X.shape[1],
        },
        "published_shared_fit": {
            "k": K,
            "epochs": PUBLISHED_CONFIG.epochs,
            "learning_rate": PUBLISHED_CONFIG.lr,
            "weight_decay": PUBLISHED_CONFIG.weight_decay,
            "archetype_l1": PUBLISHED_CONFIG.archetype_l1,
            "entropy_weight": PUBLISHED_CONFIG.entropy_weight,
            "reference_seed": REFERENCE_SEED,
            "alignment": "Hungarian matching on canonical regimen-coefficient cosine",
            "regimen_prevalence_threshold": config.min_treatment_prevalence,
        },
        "reference_profiles_ordered_by_soft_event_rate": _reference_profile_summary(
            reference, outcome
        ),
        "optimization_seed_stability": _optimization_seed_stability(reference, fits, outcome),
        "full_pipeline_patient_cluster_bootstrap": bootstrap,
        "profile_by_regimen_uncertainty": _cell_uncertainty(reference, draws, matrices),
        "interpretive_rules": [
            "The shared loading controls both prognosis and regimen coefficients.",
            "Hard assignments are secondary summaries of continuous loadings.",
            "Bootstrap intervals include patient sampling, preprocessing, and shared-model fitting under the published fixed seed.",
            "The 130 coefficients are recorded regimen-outcome associations, not causal treatment effects.",
            "A stable coefficient is not an identified treatment effect.",
        ],
        "governance": {
            "row_level_objects_serialized": False,
            "identifiers_serialized": False,
            "small_cell_threshold": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    _write_report(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
