"""Evaluate nearby K values for the corrected shared-loading PATCH architecture.

All preprocessing is fold-local for predictive evaluation. Full-cohort seed fits are used only for
descriptive profile stability. Only disclosure-safe aggregate summaries are written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    _additive_probability,
    _fit_additive,
    _seed,
    build_schema6_raw_validation_data,
    make_grouped_splits,
)
from run_aggregate_sensitivities import (
    _cluster_bootstrap_metrics,
    _point_metrics,
    _write_report,
)

K_VALUES = (3, 4, 5, 6, 7, 8)
FULL_SEEDS = (0, 1, 2)
FULL_CONFIG = TrainConfig(
    epochs=2000,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)
OOF_CONFIG = TrainConfig(
    epochs=500,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)


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


def _fit_model(
    matrices: FoldMatrices,
    outcome: np.ndarray,
    *,
    k: int,
    seed: int,
    config: TrainConfig,
) -> LinearArchetypeContextualized:
    model = LinearArchetypeContextualized(
        matrices.C.shape[1],
        matrices.X.shape[1],
        n_archetypes=k,
        simplex=True,
        seed=seed,
    )
    return train(
        model,
        matrices.C,
        matrices.X,
        outcome.astype(np.float64),
        config,
        seed=seed,
    )


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


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": [float(array.min()), float(array.max())],
    }


def _alignment(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    k = reference.shape[0]
    similarities = np.asarray(
        [[_cosine(reference[row], current[column]) for column in range(k)] for row in range(k)],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(-similarities)
    return np.asarray(columns[np.argsort(rows)], dtype=np.int64)


def _full_seed_stability(
    matrices: FoldMatrices,
    outcome: np.ndarray,
    *,
    k: int,
) -> dict[str, Any]:
    fits: list[dict[str, Any]] = []
    for seed in FULL_SEEDS:
        model = _fit_model(matrices, outcome, k=k, seed=seed, config=FULL_CONFIG)
        loadings = np.asarray(model.predict_loadings(matrices.C), dtype=np.float64)
        beta, _ = model.predict_beta(matrices.C)
        raw = model.archetypes.detach().cpu().numpy().astype(np.float64)
        coefficients = raw[:, : matrices.X.shape[1]]
        soft_rates = np.sum(loadings * outcome[:, None], axis=0) / np.sum(loadings, axis=0)
        hard = loadings.argmax(axis=1)
        hard_counts = np.bincount(hard, minlength=k)
        fits.append(
            {
                "seed": seed,
                "loadings": loadings,
                "beta_surface": np.asarray(beta, dtype=np.float64),
                "coefficients": coefficients,
                "soft_rates": np.asarray(soft_rates, dtype=np.float64),
                "hard": hard,
                "hard_counts": hard_counts,
            }
        )
    pair_rows: list[dict[str, float]] = []
    for left, right in combinations(fits, 2):
        order = _alignment(left["coefficients"], right["coefficients"])
        aligned_loadings = right["loadings"][:, order]
        aligned_rates = right["soft_rates"][order]
        pair_rows.append(
            {
                "hard_profile_ari": float(adjusted_rand_score(left["hard"], right["hard"])),
                "beta_surface_cosine": _cosine(left["beta_surface"], right["beta_surface"]),
                "mean_matched_loading_pearson": float(
                    np.mean(
                        [
                            _pearson(
                                left["loadings"][:, profile],
                                aligned_loadings[:, profile],
                            )
                            for profile in range(k)
                        ]
                    )
                ),
                "maximum_absolute_soft_event_rate_change": float(
                    np.max(np.abs(left["soft_rates"] - aligned_rates))
                ),
            }
        )
    min_hard_fraction = [float(np.min(fit["hard_counts"]) / len(outcome)) for fit in fits]
    event_rate_range = [
        float(np.max(fit["soft_rates"]) - np.min(fit["soft_rates"])) for fit in fits
    ]
    profile_distances: list[float] = []
    for fit in fits:
        coefficients = fit["coefficients"]
        for left, right in combinations(range(k), 2):
            profile_distances.append(1.0 - _cosine(coefficients[left], coefficients[right]))
    return {
        "seeds": list(FULL_SEEDS),
        "seed_pairs": len(pair_rows),
        "minimum_hard_profile_fraction": _summary(min_hard_fraction),
        "soft_event_rate_range": _summary(event_rate_range),
        "pairwise_coefficient_cosine_distance": _summary(profile_distances),
        "pair_metrics": {key: _summary([row[key] for row in pair_rows]) for key in pair_rows[0]},
    }


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    logging.getLogger("ctb").setLevel(logging.ERROR)
    config = LockedValidationConfig()
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    splits = make_grouped_splits(
        outcome,
        groups,
        n_splits=config.outer_splits,
        seed=config.seed,
        label="shared K sensitivity",
    )
    predictions = {
        "additive": np.full(len(outcome), np.nan, dtype=np.float64),
        **{f"shared_k{k}": np.full(len(outcome), np.nan, dtype=np.float64) for k in K_VALUES},
    }
    for fold_number, (train_index, test_index) in enumerate(splits):
        preprocessor = FoldPreprocessor(
            data.numeric_columns,
            data.categorical_columns,
            min_treatment_prevalence=config.min_treatment_prevalence,
        )
        training = preprocessor.fit_transform(
            data.context.iloc[train_index], data.treatments.iloc[train_index]
        )
        evaluation = preprocessor.transform(
            data.context.iloc[test_index], data.treatments.iloc[test_index]
        )
        additive = _fit_additive(
            training,
            outcome[train_index],
            c_value=0.1,
            max_iter=config.logistic_max_iter,
            seed=_seed(config.seed, fold_number, 1),
        )
        predictions["additive"][test_index] = _additive_probability(additive, evaluation)
        for position, k in enumerate(K_VALUES, start=2):
            model = _fit_model(
                training,
                outcome[train_index],
                k=k,
                seed=_seed(config.seed, fold_number, position),
                config=OOF_CONFIG,
            )
            predictions[f"shared_k{k}"][test_index] = model.predict_proba(
                evaluation.C, evaluation.X
            )
    if any(np.isnan(values).any() for values in predictions.values()):
        raise RuntimeError("Shared K sensitivity produced incomplete out-of-fold predictions")
    uncertainty = _cluster_bootstrap_metrics(
        outcome,
        predictions,
        groups,
        n_resamples=args.bootstrap_resamples,
        seed=_seed(config.seed, 5501),
    )
    full_preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=config.min_treatment_prevalence,
    )
    full_matrices = full_preprocessor.fit_transform(data.context, data.treatments)
    report = {
        "report_schema_version": 1,
        "analysis_id": "shared_patch_corrected_k_sensitivity",
        "analysis_role": "analysis-only remediation; no submission artifact was modified",
        "cohort": {
            "assessed_conditions": len(outcome),
            "patients": int(len(np.unique(groups))),
            "events": int(outcome.sum()),
            "observed_non_events": int(len(outcome) - outcome.sum()),
        },
        "candidate_k": list(K_VALUES),
        "oof_configuration": {
            "outer_folds": config.outer_splits,
            "patient_grouped": True,
            "preprocessing": "fit within each training fold",
            "shared_epochs": OOF_CONFIG.epochs,
            "candidate_selection": "none; all K values are forced diagnostics",
        },
        "oof_metrics": {
            name: _point_metrics(outcome, values) for name, values in predictions.items()
        },
        "patient_clustered_uncertainty": uncertainty,
        "full_fit_seed_stability": {
            f"k{k}": _full_seed_stability(full_matrices, outcome, k=k) for k in K_VALUES
        },
        "interpretive_rules": [
            "Predictive performance is not used to select K after inspecting the profile labels.",
            "Full-cohort seed stability is descriptive and not an out-of-sample performance estimate.",
            "Hard assignments are secondary summaries of continuous shared loadings.",
            "K=5 may be retained only as a descriptive granularity choice if nearby K values predict similarly.",
        ],
        "governance": {
            "row_level_objects_serialized": False,
            "identifiers_serialized": False,
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
