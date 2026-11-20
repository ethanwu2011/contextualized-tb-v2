"""Audit prognosis, treatment-associated structure, and profile composition in PATCH.

The script evaluates the original shared-loading K=5 architecture, a risk-only K=5 archetype
model with constant treatment coefficients, the identified residual K=5 treatment-heterogeneity
model, and targeted context ablations. Row-level inputs and fitted quantities remain in memory.
Only disclosure-safe aggregate summaries are serialized.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from torch import nn
from torch.optim.adam import Adam

from ctb.config import RESISTANCE_EXPECTED_DAYS
from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    train,
)
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    PatchCandidate,
    RawValidationData,
    _additive_probability,
    _fit_additive,
    _fit_residual,
    _seed,
    build_schema6_raw_validation_data,
    make_grouped_splits,
)
from run_aggregate_sensitivities import (
    SMALL_CELL_MIN,
    _cluster_bootstrap_metrics,
    _point_metrics,
    _safe_count,
    _write_report,
)

HEADLINE_CONFIG = TrainConfig(
    epochs=2000,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)
OOF_ARCHETYPE_CONFIG = TrainConfig(
    epochs=500,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)
K = 5
RESIDUAL_CANDIDATE = PatchCandidate(
    name="residual_k5_strong_shrinkage",
    n_archetypes=K,
    loading_l2=1e-3,
    deviation_l2=1e-2,
    residual_scale=0.5,
    epochs=300,
    learning_rate=0.01,
)
RESIDUAL_RAW_FIELDS = ("number_of_children", "number_of_daily_contacts")


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


class RiskOnlyArchetypeLogistic(nn.Module):
    """Context archetypes for prognosis plus one constant regimen coefficient vector."""

    def __init__(self, context_dim: int, x_dim: int, n_archetypes: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_archetypes = n_archetypes
        self.loading_map = nn.Linear(context_dim, n_archetypes)
        self.risk_archetypes = nn.Parameter(torch.randn(n_archetypes) * 0.1)
        self.treatment_main = nn.Parameter(torch.zeros(x_dim))

    def _loadings(self, context: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.loading_map(context), dim=1)

    def forward(self, context: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        risk = self._loadings(context) @ self.risk_archetypes
        return risk + treatment @ self.treatment_main

    @torch.no_grad()
    def predict_loadings(self, context: np.ndarray) -> np.ndarray:
        values = self._loadings(torch.tensor(context, dtype=torch.float32))
        return values.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_proba(self, context: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        logit = self.forward(
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(treatment, dtype=torch.float32),
        )
        return torch.sigmoid(logit).cpu().numpy().astype(np.float64)


def _train_risk_only(
    model: RiskOnlyArchetypeLogistic,
    matrices: FoldMatrices,
    outcome: np.ndarray,
    config: TrainConfig,
    *,
    seed: int,
) -> RiskOnlyArchetypeLogistic:
    torch.manual_seed(seed)
    context = torch.tensor(matrices.C, dtype=torch.float32)
    treatment = torch.tensor(matrices.X, dtype=torch.float32)
    y = torch.tensor(outcome, dtype=torch.float32)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(context, treatment)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        if config.archetype_l1 > 0:
            loss = loss + config.archetype_l1 * model.risk_archetypes.abs().mean()
        if config.entropy_weight > 0:
            mean_loading = model._loadings(context).mean(dim=0)
            entropy = -(mean_loading * torch.log(mean_loading + 1e-9)).sum()
            loss = loss - config.entropy_weight * entropy
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    return model


def _drop_context(
    data: RawValidationData,
    predicate: Callable[[str], bool],
) -> RawValidationData:
    retained_numeric = tuple(name for name in data.numeric_columns if not predicate(name))
    retained_categorical = tuple(name for name in data.categorical_columns if not predicate(name))
    retained = [*retained_numeric, *retained_categorical]
    if not retained:
        raise ValueError("Context ablation removed every feature")
    return RawValidationData(
        context=data.context.loc[:, retained].copy(),
        treatments=data.treatments,
        outcome=data.outcome,
        patient_groups=data.patient_groups,
        numeric_columns=retained_numeric,
        categorical_columns=retained_categorical,
    )


def _preprocess_full(data: RawValidationData, config: LockedValidationConfig) -> FoldMatrices:
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=config.min_treatment_prevalence,
    )
    return preprocessor.fit_transform(data.context, data.treatments)


def _fit_shared(
    matrices: FoldMatrices,
    outcome: np.ndarray,
    config: TrainConfig,
    *,
    seed: int,
) -> LinearArchetypeContextualized:
    model = LinearArchetypeContextualized(
        matrices.C.shape[1],
        matrices.X.shape[1],
        n_archetypes=K,
        simplex=True,
        seed=seed,
    )
    return train(model, matrices.C, matrices.X, outcome.astype(np.float64), config, seed=seed)


def _fit_risk(
    matrices: FoldMatrices,
    outcome: np.ndarray,
    config: TrainConfig,
    *,
    seed: int,
) -> RiskOnlyArchetypeLogistic:
    model = RiskOnlyArchetypeLogistic(
        matrices.C.shape[1],
        matrices.X.shape[1],
        K,
        seed=seed,
    )
    return _train_risk_only(model, matrices, outcome, config, seed=seed)


@dataclass(frozen=True)
class FullFit:
    name: str
    loadings: np.ndarray
    probability: np.ndarray
    beta_surface: np.ndarray | None


def _full_fits(
    datasets: dict[str, RawValidationData],
    config: LockedValidationConfig,
) -> tuple[dict[str, FullFit], dict[str, FoldMatrices]]:
    matrices = {name: _preprocess_full(data, config) for name, data in datasets.items()}
    y = np.asarray(datasets["full"].outcome, dtype=np.int64)
    seed = config.seed

    shared_full_model = _fit_shared(matrices["full"], y, HEADLINE_CONFIG, seed=seed)
    risk_full_model = _fit_risk(matrices["full"], y, HEADLINE_CONFIG, seed=seed)
    additive = _fit_additive(
        matrices["full"],
        y,
        c_value=0.1,
        max_iter=config.logistic_max_iter,
        seed=seed,
    )
    residual_full_model = _fit_residual(
        additive,
        matrices["full"],
        y,
        RESIDUAL_CANDIDATE,
        seed=seed,
    )
    shared_no_resistance = _fit_shared(
        matrices["without_resistance_dst"],
        y,
        HEADLINE_CONFIG,
        seed=seed,
    )
    shared_no_residual = _fit_shared(
        matrices["without_residual_fields"],
        y,
        HEADLINE_CONFIG,
        seed=seed,
    )

    shared_beta, _ = shared_full_model.predict_beta(matrices["full"].C)
    residual_beta, _ = residual_full_model.predict_beta(matrices["full"].C)
    no_resistance_beta, _ = shared_no_resistance.predict_beta(matrices["without_resistance_dst"].C)
    no_residual_beta, _ = shared_no_residual.predict_beta(matrices["without_residual_fields"].C)
    fits = {
        "shared_full": FullFit(
            "shared_full",
            shared_full_model.predict_loadings(matrices["full"].C),
            shared_full_model.predict_proba(matrices["full"].C, matrices["full"].X),
            shared_beta,
        ),
        "risk_only_full": FullFit(
            "risk_only_full",
            risk_full_model.predict_loadings(matrices["full"].C),
            risk_full_model.predict_proba(matrices["full"].C, matrices["full"].X),
            None,
        ),
        "residual_treatment_full": FullFit(
            "residual_treatment_full",
            residual_full_model.predict_loadings(matrices["full"].C),
            residual_full_model.predict_proba(matrices["full"].C, matrices["full"].X),
            residual_beta,
        ),
        "shared_without_resistance_dst": FullFit(
            "shared_without_resistance_dst",
            shared_no_resistance.predict_loadings(matrices["without_resistance_dst"].C),
            shared_no_resistance.predict_proba(
                matrices["without_resistance_dst"].C,
                matrices["without_resistance_dst"].X,
            ),
            no_resistance_beta,
        ),
        "shared_without_residual_fields": FullFit(
            "shared_without_residual_fields",
            shared_no_residual.predict_loadings(matrices["without_residual_fields"].C),
            shared_no_residual.predict_proba(
                matrices["without_residual_fields"].C,
                matrices["without_residual_fields"].X,
            ),
            no_residual_beta,
        ),
    }
    return fits, matrices


def _fit_oof_models(
    datasets: dict[str, RawValidationData],
    config: LockedValidationConfig,
) -> dict[str, np.ndarray]:
    full = datasets["full"]
    y = np.asarray(full.outcome, dtype=np.int64)
    groups = full.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    splits = make_grouped_splits(
        y,
        groups,
        n_splits=config.outer_splits,
        seed=config.seed,
        label="architecture ablation",
    )
    names = (
        "additive",
        "shared_k5",
        "risk_only_k5",
        "residual_k5",
        "residual_k5_without_resistance_dst",
        "residual_k5_without_residual_fields",
    )
    values = {name: np.full(len(y), np.nan, dtype=np.float64) for name in names}
    for fold_number, (train_index, test_index) in enumerate(splits):
        fold_matrices: dict[str, tuple[FoldMatrices, FoldMatrices]] = {}
        for dataset_name, data in datasets.items():
            preprocessor = FoldPreprocessor(
                data.numeric_columns,
                data.categorical_columns,
                min_treatment_prevalence=config.min_treatment_prevalence,
            )
            train_matrix = preprocessor.fit_transform(
                data.context.iloc[train_index], data.treatments.iloc[train_index]
            )
            test_matrix = preprocessor.transform(
                data.context.iloc[test_index], data.treatments.iloc[test_index]
            )
            fold_matrices[dataset_name] = (train_matrix, test_matrix)

        train_full, test_full = fold_matrices["full"]
        baseline = _fit_additive(
            train_full,
            y[train_index],
            c_value=0.1,
            max_iter=config.logistic_max_iter,
            seed=_seed(config.seed, fold_number, 1),
        )
        values["additive"][test_index] = _additive_probability(baseline, test_full)

        shared = _fit_shared(
            train_full,
            y[train_index],
            OOF_ARCHETYPE_CONFIG,
            seed=_seed(config.seed, fold_number, 2),
        )
        values["shared_k5"][test_index] = shared.predict_proba(test_full.C, test_full.X)
        risk = _fit_risk(
            train_full,
            y[train_index],
            OOF_ARCHETYPE_CONFIG,
            seed=_seed(config.seed, fold_number, 3),
        )
        values["risk_only_k5"][test_index] = risk.predict_proba(test_full.C, test_full.X)
        residual = _fit_residual(
            baseline,
            train_full,
            y[train_index],
            RESIDUAL_CANDIDATE,
            seed=_seed(config.seed, fold_number, 4),
        )
        values["residual_k5"][test_index] = residual.predict_proba(test_full.C, test_full.X)

        for position, (dataset_name, output_name) in enumerate(
            (
                ("without_resistance_dst", "residual_k5_without_resistance_dst"),
                ("without_residual_fields", "residual_k5_without_residual_fields"),
            ),
            start=5,
        ):
            train_matrix, test_matrix = fold_matrices[dataset_name]
            ablated_baseline = _fit_additive(
                train_matrix,
                y[train_index],
                c_value=0.1,
                max_iter=config.logistic_max_iter,
                seed=_seed(config.seed, fold_number, position, 1),
            )
            ablated_residual = _fit_residual(
                ablated_baseline,
                train_matrix,
                y[train_index],
                RESIDUAL_CANDIDATE,
                seed=_seed(config.seed, fold_number, position, 2),
            )
            values[output_name][test_index] = ablated_residual.predict_proba(
                test_matrix.C, test_matrix.X
            )
    if any(np.isnan(probability).any() for probability in values.values()):
        raise RuntimeError("Architecture ablation produced incomplete out-of-fold values")
    return values


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    return float(np.dot(left_vector, right_vector) / denominator) if denominator > 1e-12 else 0.0


def _align_loadings(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, list[float]]:
    similarity = np.asarray(
        [
            [_correlation(reference[:, row], current[:, column]) for column in range(K)]
            for row in range(K)
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(-similarity)
    order = columns[np.argsort(rows)]
    aligned = current[:, order]
    matched = [_correlation(reference[:, row], aligned[:, row]) for row in range(K)]
    return aligned, matched


def _fit_comparison(reference: FullFit, current: FullFit) -> dict[str, Any]:
    aligned, matched = _align_loadings(reference.loadings, current.loadings)
    result: dict[str, Any] = {
        "matched_loading_correlations": matched,
        "mean_matched_loading_correlation": float(np.mean(matched)),
        "minimum_matched_loading_correlation": float(np.min(matched)),
        "hard_label_ari": float(
            adjusted_rand_score(reference.loadings.argmax(axis=1), aligned.argmax(axis=1))
        ),
        "fitted_value_correlation": _correlation(reference.probability, current.probability),
    }
    if reference.beta_surface is not None and current.beta_surface is not None:
        result["coefficient_surface_cosine"] = _cosine(reference.beta_surface, current.beta_surface)
    return result


def _soft_nmi(loadings: np.ndarray, categories: pd.Series[Any]) -> float:
    category = categories.astype("string").fillna("Missing").to_numpy(dtype=str)
    levels, encoded = np.unique(category, return_inverse=True)
    if len(levels) < 2 or loadings.shape[1] < 2:
        return 0.0
    one_hot = np.eye(len(levels), dtype=np.float64)[encoded]
    joint = loadings.T @ one_hot
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    joint = joint / total
    profile_mass = joint.sum(axis=1, keepdims=True)
    category_mass = joint.sum(axis=0, keepdims=True)
    expected = profile_mass @ category_mass
    valid = (joint > 0) & (expected > 0)
    mutual_information = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    profile_entropy = float(
        -np.sum(profile_mass[profile_mass > 0] * np.log(profile_mass[profile_mass > 0]))
    )
    category_entropy = float(
        -np.sum(category_mass[category_mass > 0] * np.log(category_mass[category_mass > 0]))
    )
    denominator = math.sqrt(profile_entropy * category_entropy)
    return mutual_information / denominator if denominator > 0 else 0.0


def _cluster_rows(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    rows = [np.flatnonzero(inverse == position) for position in range(len(unique))]
    return unique, rows


def _nmi_asymmetry(
    sorted_loadings: np.ndarray,
    attributes: dict[str, pd.Series[Any]],
    groups: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    _, grouped_rows = _cluster_rows(groups)
    rng = np.random.default_rng(seed)
    low = sorted_loadings[:, :3]
    high = sorted_loadings[:, 3:]
    result: dict[str, Any] = {}
    for name, attribute in attributes.items():
        low_point = _soft_nmi(low, attribute)
        high_point = _soft_nmi(high, attribute)
        low_draws: list[float] = []
        high_draws: list[float] = []
        difference_draws: list[float] = []
        for _ in range(n_resamples):
            sampled = rng.integers(0, len(grouped_rows), size=len(grouped_rows))
            rows = np.concatenate([grouped_rows[position] for position in sampled])
            sampled_attribute = attribute.iloc[rows].reset_index(drop=True)
            low_value = _soft_nmi(low[rows], sampled_attribute)
            high_value = _soft_nmi(high[rows], sampled_attribute)
            low_draws.append(low_value)
            high_draws.append(high_value)
            difference_draws.append(low_value - high_value)
        result[name] = {
            "lower_risk_profiles_nmi": {
                "estimate": low_point,
                "ci95": [
                    float(np.percentile(low_draws, 2.5)),
                    float(np.percentile(low_draws, 97.5)),
                ],
            },
            "higher_risk_profiles_nmi": {
                "estimate": high_point,
                "ci95": [
                    float(np.percentile(high_draws, 2.5)),
                    float(np.percentile(high_draws, 97.5)),
                ],
            },
            "lower_minus_higher": {
                "estimate": low_point - high_point,
                "ci95": [
                    float(np.percentile(difference_draws, 2.5)),
                    float(np.percentile(difference_draws, 97.5)),
                ],
            },
        }
    return result


def _weighted_cell(weight: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    effective = float(np.sum(weight[mask]))
    total = float(np.sum(weight))
    if effective < SMALL_CELL_MIN:
        return {"suppressed": True, "threshold": SMALL_CELL_MIN}
    return {
        "soft_effective_count": effective,
        "percent": 100.0 * effective / total,
    }


def _profile_composition(
    loadings: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    event_rates = np.sum(loadings * y[:, None], axis=0) / np.sum(loadings, axis=0)
    order = np.argsort(event_rates)
    sorted_loadings = loadings[:, order]
    hard = sorted_loadings.argmax(axis=1)
    resistance = metadata["type_of_resistance"].astype("string")
    outcome_state = metadata["recorded_outcome"].astype("string")
    rows: list[dict[str, Any]] = []
    for profile in range(K):
        weight = sorted_loadings[:, profile]
        hard_mask = hard == profile
        hard_count = int(hard_mask.sum())
        row: dict[str, Any] = {
            "profile": profile + 1,
            "risk_order": "lower" if profile < 3 else "higher",
            "soft_effective_count": float(weight.sum()),
            "soft_weighted_event_rate": float(np.sum(weight * y) / np.sum(weight)),
            "hard_count_secondary": _safe_count(hard_count),
            "hard_event_rate_secondary": (
                float(y[hard_mask].mean())
                if hard_count >= SMALL_CELL_MIN
                else {"suppressed": True, "threshold": SMALL_CELL_MIN}
            ),
            "resistance_composition": {
                level: _weighted_cell(weight, resistance.eq(level).to_numpy())
                for level in RESISTANCE_EXPECTED_DAYS
            },
            "hiv_positive": _weighted_cell(
                weight, metadata["comorb_hiv"].astype(float).eq(1).to_numpy()
            ),
            "ukraine": _weighted_cell(weight, metadata["country"].eq("Ukraine").to_numpy()),
            "smear_positive": _weighted_cell(
                weight, metadata["baseline_smear"].eq("positive").to_numpy()
            ),
            "cavitation_present": _weighted_cell(
                weight, metadata["cavitation"].eq("yes").to_numpy()
            ),
            "recorded_death": _weighted_cell(weight, outcome_state.eq("died").to_numpy()),
            "recorded_failure": _weighted_cell(weight, outcome_state.eq("failure").to_numpy()),
        }
        rows.append(row)
    return rows, sorted_loadings


def _event_rate_spread(loadings: np.ndarray, y: np.ndarray) -> dict[str, float]:
    rates = np.sum(loadings * y[:, None], axis=0) / np.sum(loadings, axis=0)
    return {
        "minimum_soft_weighted_rate": float(np.min(rates)),
        "maximum_soft_weighted_rate": float(np.max(rates)),
        "range": float(np.max(rates) - np.min(rates)),
    }


def _normalize_recorded_outcome(values: pd.Series[Any]) -> pd.Series[str]:
    return values.astype("string").str.strip().str.casefold().replace("", "unknown").astype(str)


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

    config = LockedValidationConfig()
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    full = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    without_resistance = _drop_context(
        full,
        lambda name: name == "type_of_resistance" or name.startswith("bactec_"),
    )
    without_residual = _drop_context(full, lambda name: name in RESIDUAL_RAW_FIELDS)
    datasets = {
        "full": full,
        "without_resistance_dst": without_resistance,
        "without_residual_fields": without_residual,
    }
    y = np.asarray(full.outcome, dtype=np.int64)
    groups = full.patient_groups.astype("string").str.strip().to_numpy(dtype=str)

    full_fits, matrices = _full_fits(datasets, config)
    oof = _fit_oof_models(datasets, config)
    validation = _cluster_bootstrap_metrics(
        y,
        oof,
        groups,
        n_resamples=args.bootstrap_resamples,
        seed=_seed(config.seed, 8801),
    )

    indexed_cases = patient_cases.set_index("condition_id").reindex(full.context.index)
    metadata = pd.DataFrame(index=full.context.index)
    metadata["type_of_resistance"] = indexed_cases["type_of_resistance"].astype("string")
    metadata["country"] = full.context["country"].astype("string")
    metadata["comorb_hiv"] = pd.to_numeric(full.context["comorb_hiv"], errors="coerce").fillna(0)
    metadata["baseline_smear"] = full.context["baseline_smear"].astype("string")
    metadata["cavitation"] = full.context["cavitation"].astype("string")
    metadata["recorded_outcome"] = _normalize_recorded_outcome(indexed_cases["outcome"])

    profile_rows, sorted_loadings = _profile_composition(
        full_fits["shared_full"].loadings,
        y,
        metadata,
    )
    attributes = {
        "resistance_class": metadata["type_of_resistance"],
        "country": metadata["country"],
        "hiv": metadata["comorb_hiv"].astype("string"),
        "smear": metadata["baseline_smear"],
        "cavitation": metadata["cavitation"],
    }
    asymmetry = _nmi_asymmetry(
        sorted_loadings,
        attributes,
        groups,
        n_resamples=args.bootstrap_resamples,
        seed=_seed(config.seed, 9901),
    )

    reference = full_fits["shared_full"]
    comparisons = {
        name: _fit_comparison(reference, fit)
        for name, fit in full_fits.items()
        if name != "shared_full"
    }
    report = {
        "report_schema_version": 1,
        "analysis_id": "patch_architecture_composition_ablations",
        "analysis_status": "analysis-only; manuscript and public artifacts untouched",
        "endpoint": "recorded final Patient Cases died or failure versus cured or completed",
        "cohort": {
            "conditions": int(len(y)),
            "patients": int(len(np.unique(groups))),
            "events": int(y.sum()),
            "observed_non_events": int(len(y) - y.sum()),
            "full_raw_context_columns": int(full.context.shape[1]),
            "full_encoded_context_columns": int(matrices["full"].C.shape[1]),
            "regimen_components": int(matrices["full"].X.shape[1]),
        },
        "architectures": {
            "shared_k5": "one simplex loading controls both baseline-risk intercept and regimen coefficients",
            "risk_only_k5": "simplex loading controls risk archetypes; regimen coefficients are constant",
            "residual_k5": "linear context and constant regimen parent plus simplex loading only for residual regimen heterogeneity",
        },
        "full_fit_configuration": {
            "k": K,
            "epochs": HEADLINE_CONFIG.epochs,
            "learning_rate": HEADLINE_CONFIG.lr,
            "weight_decay": HEADLINE_CONFIG.weight_decay,
            "archetype_l1": HEADLINE_CONFIG.archetype_l1,
            "entropy_weight": HEADLINE_CONFIG.entropy_weight,
            "seed": config.seed,
        },
        "oof_configuration": {
            "outer_folds": config.outer_splits,
            "patient_grouped": True,
            "preprocessing": "fit within each training fold",
            "shared_and_risk_only_epochs": OOF_ARCHETYPE_CONFIG.epochs,
            "residual_epochs": RESIDUAL_CANDIDATE.epochs,
        },
        "oof_validation": {
            "metrics": {name: _point_metrics(y, values) for name, values in oof.items()},
            "patient_clustered_uncertainty": validation,
        },
        "headline_shared_profile_composition": profile_rows,
        "composition_asymmetry": {
            "definition": "soft normalized mutual information between profile identity and each attribute, evaluated separately among the three lower-risk and two higher-risk profiles",
            "patient_clustered_uncertainty": asymmetry,
        },
        "full_fit_comparisons_to_shared_k5": comparisons,
        "event_rate_separation": {
            name: _event_rate_spread(fit.loadings, y) for name, fit in full_fits.items()
        },
        "resistance_dst_ablation": {
            "removed_raw_fields": [
                "type_of_resistance",
                "all Patient Cases bactec per-drug fields",
            ],
            "remaining_raw_context_columns": int(without_resistance.context.shape[1]),
            "comparison_to_full_shared_k5": comparisons["shared_without_resistance_dst"],
        },
        "residual_field_audit": {
            "raw_fields": list(RESIDUAL_RAW_FIELDS),
            "source": "Patient Cases",
            "classification": "social and household-contact variables; neither is a site, registration-date, organization, or enrollment identifier",
            "timing": "field-specific timestamps unavailable",
            "remaining_raw_context_columns_after_removal": int(without_residual.context.shape[1]),
            "comparison_to_full_shared_k5": comparisons["shared_without_residual_fields"],
        },
        "governance": {
            "dua_acknowledged": bool(args.dua_acknowledge),
            "contains_identifiers": False,
            "contains_row_level_data": False,
            "small_cell_minimum": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
        "limitations": [
            "The full-fit profile comparisons are descriptive and use a full-cohort encoder.",
            "Profile event-rate separation is expected in an outcome-supervised model and is not itself a discovery.",
            "The composition NMI analysis is conditional on the fitted shared K=5 representation.",
            "Regimen coefficients remain treatment-outcome associations rather than causal effects.",
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
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
