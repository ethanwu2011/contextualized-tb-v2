"""Finite synthetic benchmark for residual PATCH.

The benchmark evaluates prediction and recovery of a known context-varying
regimen-coefficient surface. It does not represent treatment assignment as randomized and does
not support a causal interpretation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import matplotlib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ctb.models.residual_archetype import (
    ResidualArchetypeLogistic,
    ResidualTrainConfig,
    train_residual,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ScenarioName = Literal["additive_null", "low_rank", "nonlinear_misspecified"]

SCENARIOS: Final[tuple[ScenarioName, ...]] = (
    "additive_null",
    "low_rank",
    "nonlinear_misspecified",
)
MODEL_ORDER: Final[tuple[str, ...]] = (
    "additive_logistic_null",
    "residual_patch",
    "unrestricted_interaction_logistic",
)
SCENARIO_LABELS: Final[dict[ScenarioName, str]] = {
    "additive_null": "Additive null",
    "low_rank": "Low-rank surface",
    "nonlinear_misspecified": "Nonlinear misspecification",
}
MODEL_LABELS: Final[dict[str, str]] = {
    "additive_logistic_null": "Additive logistic null",
    "residual_patch": "Residual PATCH",
    "unrestricted_interaction_logistic": "Full interaction logistic",
}
MODEL_COLORS: Final[dict[str, str]] = {
    "additive_logistic_null": "#0072B2",
    "residual_patch": "#D55E00",
    "unrestricted_interaction_logistic": "#009E73",
}
MODEL_MARKERS: Final[dict[str, str]] = {
    "additive_logistic_null": "o",
    "residual_patch": "s",
    "unrestricted_interaction_logistic": "^",
}

_CONTEXT_MAIN: Final[FloatArray] = np.array([0.45, -0.35, 0.25, 0.15, -0.20])
_REGIMEN_MAIN: Final[FloatArray] = np.array([-0.45, 0.30, -0.25, 0.20])
_ASSIGNMENT_OFFSET: Final[FloatArray] = np.array([-0.25, -0.05, 0.15, -0.35])
_ASSIGNMENT_MAP: Final[FloatArray] = np.array(
    [
        [0.45, -0.25, 0.20, 0.15],
        [-0.30, 0.35, 0.10, -0.20],
        [0.15, 0.10, -0.40, 0.30],
        [0.10, -0.20, 0.25, 0.35],
        [-0.25, 0.15, 0.20, -0.10],
    ]
)
_LOW_RANK_LOADING_MAP: Final[FloatArray] = np.array(
    [
        [1.20, -0.90, -0.30],
        [-0.70, 0.20, 0.50],
        [0.10, 0.85, -0.95],
        [0.35, -0.15, -0.20],
        [-0.20, 0.40, -0.20],
    ]
)
_LOW_RANK_LOADING_BIAS: Final[FloatArray] = np.array([0.15, -0.10, -0.05])
_LOW_RANK_DEVIATIONS: Final[FloatArray] = np.array(
    [
        [1.35, -1.00, 0.80, -0.70],
        [-1.05, 0.70, -0.80, 0.90],
        [-0.30, 0.30, 0.00, -0.20],
    ]
)


@dataclass(frozen=True)
class SimulationBatch:
    """One synthetic sample with its known coefficient surface."""

    context: FloatArray
    regimen: FloatArray
    outcome: IntArray
    true_regimen_coefficients: FloatArray

    def __post_init__(self) -> None:
        """Validate aligned finite arrays and binary values."""
        n_rows = self.context.shape[0]
        if self.context.ndim != 2 or self.regimen.ndim != 2:
            raise ValueError("context and regimen must both be two-dimensional")
        if self.outcome.shape != (n_rows,):
            raise ValueError("outcome must be one-dimensional and aligned")
        if self.regimen.shape[0] != n_rows:
            raise ValueError("context and regimen row counts must match")
        expected_surface_shape = (n_rows, self.regimen.shape[1])
        if self.true_regimen_coefficients.shape != expected_surface_shape:
            raise ValueError(
                "true_regimen_coefficients must have shape "
                f"{expected_surface_shape}, got {self.true_regimen_coefficients.shape}"
            )
        for name, values in (
            ("context", self.context),
            ("regimen", self.regimen),
            ("true_regimen_coefficients", self.true_regimen_coefficients),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        if not np.isin(self.regimen, [0.0, 1.0]).all():
            raise ValueError("regimen must contain only binary indicators")
        if not np.isin(self.outcome, [0, 1]).all():
            raise ValueError("outcome must contain only binary indicators")


@dataclass(frozen=True)
class SimulationBenchmarkConfig:
    """Prespecified finite benchmark settings."""

    sample_sizes: tuple[int, ...] = (400, 1000)
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    n_validation: int = 500
    n_test: int = 2500
    logistic_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    residual_scale_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    residual_archetypes: int = 3
    residual_epochs: int = 250
    residual_learning_rate: float = 0.03
    loading_l2: float = 1e-4
    deviation_l2: float = 1e-3
    validation_tie_tolerance: float = 5e-4
    logistic_max_iter: int = 3000

    def __post_init__(self) -> None:
        """Reject unbounded or malformed benchmark configurations."""
        if not self.sample_sizes or len(set(self.sample_sizes)) != len(self.sample_sizes):
            raise ValueError("sample_sizes must be nonempty and unique")
        if any(value < 80 for value in self.sample_sizes):
            raise ValueError("Every training sample size must be at least 80")
        if tuple(sorted(self.sample_sizes)) != self.sample_sizes:
            raise ValueError("sample_sizes must be strictly increasing")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if self.n_validation < 80 or self.n_test < 100:
            raise ValueError("n_validation must be at least 80 and n_test at least 100")
        if not self.logistic_c_grid or any(
            not np.isfinite(value) or value <= 0 for value in self.logistic_c_grid
        ):
            raise ValueError("logistic_c_grid must contain finite positive values")
        if not self.residual_scale_grid or any(
            not np.isfinite(value) or value < 0 or value > 1 for value in self.residual_scale_grid
        ):
            raise ValueError("residual_scale_grid must contain values between zero and one")
        if tuple(sorted(set(self.residual_scale_grid))) != self.residual_scale_grid:
            raise ValueError("residual_scale_grid must be strictly increasing")
        if self.residual_scale_grid[0] != 0:
            raise ValueError("residual_scale_grid must include the exact additive null at zero")
        if self.residual_archetypes < 2:
            raise ValueError("residual_archetypes must be at least two")
        if self.residual_epochs < 1 or self.logistic_max_iter < 1:
            raise ValueError("Training iteration limits must be positive")
        if not np.isfinite(self.residual_learning_rate) or self.residual_learning_rate <= 0:
            raise ValueError("residual_learning_rate must be finite and positive")
        for name, value in (
            ("loading_l2", self.loading_l2),
            ("deviation_l2", self.deviation_l2),
            ("validation_tie_tolerance", self.validation_tie_tolerance),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BenchmarkArtifactPaths:
    """Paths written by :func:`write_benchmark_artifacts`."""

    replicate_csv: Path
    summary_json: Path
    figure_png: Path
    figure_pdf: Path


@dataclass(frozen=True)
class _LogisticSelection:
    """One validation-selected logistic estimator."""

    c_value: float
    validation_log_loss: float
    model: LogisticRegression


@dataclass(frozen=True)
class _PatchSelection:
    """One residual model with a validation-selected scale."""

    scale: float
    validation_log_loss: float
    model: ResidualArchetypeLogistic
    grid_log_loss: dict[str, float]


def _seed(base: int, *coordinates: int) -> int:
    """Derive a stable seed without process-dependent hashing."""
    value = int(base)
    for position, coordinate in enumerate(coordinates, start=1):
        value = (value + position * 104729 * (coordinate + 1)) % (2**31 - 1)
    return value


def true_regimen_coefficient_surface(
    scenario: ScenarioName,
    context: FloatArray,
) -> FloatArray:
    """Return the known predictive regimen-coefficient surface."""
    c = np.asarray(context, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != len(_CONTEXT_MAIN):
        raise ValueError(f"context must have shape (n, {len(_CONTEXT_MAIN)}), got {c.shape}")
    if not np.isfinite(c).all():
        raise ValueError("context contains non-finite values")
    main = np.broadcast_to(_REGIMEN_MAIN, (len(c), len(_REGIMEN_MAIN))).copy()
    if scenario == "additive_null":
        return main
    if scenario == "low_rank":
        logits = c @ _LOW_RANK_LOADING_MAP + _LOW_RANK_LOADING_BIAS
        loadings = softmax(logits, axis=1)
        return np.asarray(main + loadings @ _LOW_RANK_DEVIATIONS, dtype=np.float64)
    if scenario == "nonlinear_misspecified":
        deviations = np.column_stack(
            [
                0.85 * np.sin(1.25 * c[:, 0]) + 0.35 * (np.square(c[:, 1]) - 1.0),
                -0.90 * np.tanh(c[:, 0] * c[:, 2]),
                0.75 * np.tanh(1.8 * c[:, 3]),
                0.65 * np.sin(c[:, 1] * c[:, 4]),
            ]
        )
        return np.asarray(main + deviations, dtype=np.float64)
    raise ValueError(f"Unknown scenario {scenario!r}")


def simulate_batch(
    scenario: ScenarioName,
    n_rows: int,
    seed: int,
) -> SimulationBatch:
    """Generate one context-dependent regimen sample and binary outcome."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}")
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n_rows, len(_CONTEXT_MAIN))).astype(np.float64)
    assignment_logit = _ASSIGNMENT_OFFSET + context @ _ASSIGNMENT_MAP
    assignment_probability = np.clip(expit(assignment_logit), 0.08, 0.92)
    regimen = rng.binomial(1, assignment_probability).astype(np.float64)
    true_beta = true_regimen_coefficient_surface(scenario, context)
    outcome_logit = -0.55 + context @ _CONTEXT_MAIN + np.sum(regimen * true_beta, axis=1)
    outcome = rng.binomial(1, expit(outcome_logit)).astype(np.int64)
    return SimulationBatch(
        context=context,
        regimen=regimen,
        outcome=outcome,
        true_regimen_coefficients=true_beta,
    )


def _additive_design(context: FloatArray, regimen: FloatArray) -> FloatArray:
    """Return context and regimen main effects."""
    return np.hstack([context, regimen])


def _interaction_design(context: FloatArray, regimen: FloatArray) -> FloatArray:
    """Return main effects plus every context-by-regimen product."""
    interactions = np.einsum("ni,nj->nij", context, regimen).reshape(len(context), -1)
    return np.hstack([context, regimen, interactions])


def _fit_logistic(
    design: FloatArray,
    outcome: IntArray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    """Fit one deterministic L2-penalized logistic estimator."""
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(design, outcome)
    return model


def _select_logistic(
    train_design: FloatArray,
    train_outcome: IntArray,
    validation_design: FloatArray,
    validation_outcome: IntArray,
    *,
    c_grid: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> _LogisticSelection:
    """Select L2 strength by validation log loss with deterministic tie handling."""
    candidates: list[_LogisticSelection] = []
    for index, c_value in enumerate(c_grid):
        model = _fit_logistic(
            train_design,
            train_outcome,
            c_value=c_value,
            max_iter=max_iter,
            seed=_seed(seed, index),
        )
        probability = np.asarray(model.predict_proba(validation_design)[:, 1], dtype=np.float64)
        candidates.append(
            _LogisticSelection(
                c_value=c_value,
                validation_log_loss=float(log_loss(validation_outcome, probability)),
                model=model,
            )
        )
    return min(candidates, key=lambda item: (item.validation_log_loss, item.c_value))


def fit_residual_from_additive(
    additive: LogisticRegression,
    context: FloatArray,
    regimen: FloatArray,
    outcome: IntArray,
    config: SimulationBenchmarkConfig,
    *,
    seed: int,
) -> ResidualArchetypeLogistic:
    """Fit only the residual PATCH block around a frozen additive baseline."""
    context_dim = context.shape[1]
    coefficients = np.asarray(additive.coef_[0], dtype=np.float64)
    expected_width = context_dim + regimen.shape[1]
    if coefficients.shape != (expected_width,):
        raise ValueError(
            "Additive coefficient vector must have width "
            f"{expected_width}, got {coefficients.shape}"
        )
    model = ResidualArchetypeLogistic(
        context_dim=context_dim,
        x_dim=regimen.shape[1],
        n_archetypes=config.residual_archetypes,
        seed=seed,
    )
    model.initialize_from_logistic(
        context_coef=coefficients[:context_dim],
        treatment_coef=coefficients[context_dim:],
        intercept=float(additive.intercept_[0]),
        freeze=True,
    )
    train_residual(
        model,
        context,
        regimen,
        outcome.astype(np.float64),
        ResidualTrainConfig(
            epochs=config.residual_epochs,
            lr=config.residual_learning_rate,
            baseline_l2=0.0,
            loading_l2=config.loading_l2,
            deviation_l2=config.deviation_l2,
            deviation_l1=0.0,
            entropy_weight=0.0,
        ),
        seed=seed,
    )
    return model


def select_residual_scale(
    model: ResidualArchetypeLogistic,
    context: FloatArray,
    regimen: FloatArray,
    outcome: IntArray,
    scale_grid: tuple[float, ...],
    *,
    tie_tolerance: float,
) -> tuple[float, float, dict[str, float]]:
    """Select the smallest residual scale within a prespecified loss tolerance."""
    if not scale_grid or scale_grid[0] != 0:
        raise ValueError("scale_grid must begin with the exact additive null at zero")
    if tuple(sorted(set(scale_grid))) != scale_grid:
        raise ValueError("scale_grid must be strictly increasing")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    losses: dict[float, float] = {}
    for scale in scale_grid:
        model.set_residual_scale(scale)
        probability = model.predict_proba(context, regimen)
        losses[scale] = float(log_loss(outcome, probability))
    best_loss = min(losses.values())
    eligible = [
        scale
        for scale in scale_grid
        if losses[scale] <= best_loss + tie_tolerance + np.finfo(np.float64).eps
    ]
    selected_scale = min(eligible)
    model.set_residual_scale(selected_scale)
    return (
        selected_scale,
        losses[selected_scale],
        {f"{scale:.2f}": losses[scale] for scale in scale_grid},
    )


def _select_patch(
    additive: LogisticRegression,
    train: SimulationBatch,
    validation: SimulationBatch,
    config: SimulationBenchmarkConfig,
    *,
    seed: int,
) -> _PatchSelection:
    """Train one residual head and select its post-fit scale."""
    model = fit_residual_from_additive(
        additive,
        train.context,
        train.regimen,
        train.outcome,
        config,
        seed=seed,
    )
    scale, validation_loss, grid_losses = select_residual_scale(
        model,
        validation.context,
        validation.regimen,
        validation.outcome,
        config.residual_scale_grid,
        tie_tolerance=config.validation_tie_tolerance,
    )
    return _PatchSelection(
        scale=scale,
        validation_log_loss=validation_loss,
        model=model,
        grid_log_loss=grid_losses,
    )


def _coefficient_rmse(estimated: FloatArray, truth: FloatArray) -> float:
    """Return root mean squared coefficient-surface error."""
    if estimated.shape != truth.shape:
        raise ValueError(
            f"Coefficient surfaces differ in shape: {estimated.shape} and {truth.shape}"
        )
    return float(np.sqrt(np.mean(np.square(estimated - truth))))


def _metric_row(
    *,
    scenario: ScenarioName,
    n_train: int,
    seed: int,
    model_name: str,
    selected_c: float,
    selected_scale: float | None,
    validation_log_loss: float,
    test: SimulationBatch,
    probability: FloatArray,
    estimated_beta: FloatArray,
    n_parameters: int,
) -> dict[str, int | float | str | None]:
    """Build one aggregate replicate row."""
    return {
        "scenario": scenario,
        "n_train": n_train,
        "seed": seed,
        "model": model_name,
        "selected_c": selected_c,
        "selected_residual_scale": selected_scale,
        "validation_log_loss": validation_log_loss,
        "test_n": len(test.outcome),
        "test_event_rate": float(np.mean(test.outcome)),
        "test_log_loss": float(log_loss(test.outcome, probability)),
        "test_auroc": float(roc_auc_score(test.outcome, probability)),
        "coefficient_rmse": _coefficient_rmse(
            estimated_beta,
            test.true_regimen_coefficients,
        ),
        "n_parameters": n_parameters,
    }


def _evaluate_replicate(
    scenario: ScenarioName,
    n_train: int,
    seed: int,
    train: SimulationBatch,
    validation: SimulationBatch,
    test: SimulationBatch,
    config: SimulationBenchmarkConfig,
    *,
    model_seed: int,
) -> tuple[list[dict[str, int | float | str | None]], dict[str, float]]:
    """Fit all models and return aggregate held-out metrics."""
    additive_train_design = _additive_design(train.context, train.regimen)
    additive_validation_design = _additive_design(validation.context, validation.regimen)
    additive_selection = _select_logistic(
        additive_train_design,
        train.outcome,
        additive_validation_design,
        validation.outcome,
        c_grid=config.logistic_c_grid,
        max_iter=config.logistic_max_iter,
        seed=_seed(model_seed, 0),
    )
    interaction_selection = _select_logistic(
        _interaction_design(train.context, train.regimen),
        train.outcome,
        _interaction_design(validation.context, validation.regimen),
        validation.outcome,
        c_grid=config.logistic_c_grid,
        max_iter=config.logistic_max_iter,
        seed=_seed(model_seed, 1),
    )
    patch_selection = _select_patch(
        additive_selection.model,
        train,
        validation,
        config,
        seed=_seed(model_seed, 2),
    )

    combined_context = np.vstack([train.context, validation.context])
    combined_regimen = np.vstack([train.regimen, validation.regimen])
    combined_outcome = np.concatenate([train.outcome, validation.outcome])
    final_additive = _fit_logistic(
        _additive_design(combined_context, combined_regimen),
        combined_outcome,
        c_value=additive_selection.c_value,
        max_iter=config.logistic_max_iter,
        seed=_seed(model_seed, 3),
    )
    final_interaction = _fit_logistic(
        _interaction_design(combined_context, combined_regimen),
        combined_outcome,
        c_value=interaction_selection.c_value,
        max_iter=config.logistic_max_iter,
        seed=_seed(model_seed, 4),
    )
    final_patch = fit_residual_from_additive(
        final_additive,
        combined_context,
        combined_regimen,
        combined_outcome,
        config,
        seed=_seed(model_seed, 5),
    )
    final_patch.set_residual_scale(patch_selection.scale)

    context_dim = test.context.shape[1]
    regimen_dim = test.regimen.shape[1]
    additive_probability = np.asarray(
        final_additive.predict_proba(_additive_design(test.context, test.regimen))[:, 1],
        dtype=np.float64,
    )
    additive_coefficients = np.broadcast_to(
        final_additive.coef_[0, context_dim:],
        (len(test.outcome), regimen_dim),
    ).copy()
    patch_probability = final_patch.predict_proba(test.context, test.regimen)
    patch_coefficients, _ = final_patch.predict_beta(test.context)
    interaction_probability = np.asarray(
        final_interaction.predict_proba(_interaction_design(test.context, test.regimen))[:, 1],
        dtype=np.float64,
    )
    interaction_vector = np.asarray(final_interaction.coef_[0], dtype=np.float64)
    interaction_main = interaction_vector[context_dim : context_dim + regimen_dim]
    interaction_surface = interaction_vector[context_dim + regimen_dim :].reshape(
        context_dim,
        regimen_dim,
    )
    interaction_coefficients = interaction_main + test.context @ interaction_surface

    additive_parameters = context_dim + regimen_dim + 1
    interaction_parameters = context_dim + regimen_dim + context_dim * regimen_dim + 1
    rows = [
        _metric_row(
            scenario=scenario,
            n_train=n_train,
            seed=seed,
            model_name="additive_logistic_null",
            selected_c=additive_selection.c_value,
            selected_scale=None,
            validation_log_loss=additive_selection.validation_log_loss,
            test=test,
            probability=additive_probability,
            estimated_beta=additive_coefficients,
            n_parameters=additive_parameters,
        ),
        _metric_row(
            scenario=scenario,
            n_train=n_train,
            seed=seed,
            model_name="residual_patch",
            selected_c=additive_selection.c_value,
            selected_scale=patch_selection.scale,
            validation_log_loss=patch_selection.validation_log_loss,
            test=test,
            probability=patch_probability,
            estimated_beta=patch_coefficients,
            n_parameters=final_patch.parameter_count,
        ),
        _metric_row(
            scenario=scenario,
            n_train=n_train,
            seed=seed,
            model_name="unrestricted_interaction_logistic",
            selected_c=interaction_selection.c_value,
            selected_scale=None,
            validation_log_loss=interaction_selection.validation_log_loss,
            test=test,
            probability=interaction_probability,
            estimated_beta=interaction_coefficients,
            n_parameters=interaction_parameters,
        ),
    ]
    return rows, patch_selection.grid_log_loss


def run_simulation_benchmark(
    config: SimulationBenchmarkConfig | None = None,
) -> tuple[pd.DataFrame, list[dict[str, int | str | dict[str, float]]]]:
    """Run the complete paired benchmark and return aggregate replicate results."""
    config = config or SimulationBenchmarkConfig()
    torch.set_num_threads(1)
    rows: list[dict[str, int | float | str | None]] = []
    scale_diagnostics: list[dict[str, int | str | dict[str, float]]] = []
    maximum_train_size = max(config.sample_sizes)
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed_index, seed in enumerate(config.seeds):
            train_pool = simulate_batch(
                scenario,
                maximum_train_size,
                _seed(seed, scenario_index, 0),
            )
            validation = simulate_batch(
                scenario,
                config.n_validation,
                _seed(seed, scenario_index, 1),
            )
            test = simulate_batch(
                scenario,
                config.n_test,
                _seed(seed, scenario_index, 2),
            )
            for size_index, n_train in enumerate(config.sample_sizes):
                train = SimulationBatch(
                    context=train_pool.context[:n_train].copy(),
                    regimen=train_pool.regimen[:n_train].copy(),
                    outcome=train_pool.outcome[:n_train].copy(),
                    true_regimen_coefficients=train_pool.true_regimen_coefficients[:n_train].copy(),
                )
                replicate_rows, grid_losses = _evaluate_replicate(
                    scenario,
                    n_train,
                    seed,
                    train,
                    validation,
                    test,
                    config,
                    model_seed=_seed(seed, scenario_index, seed_index, size_index),
                )
                rows.extend(replicate_rows)
                scale_diagnostics.append(
                    {
                        "scenario": scenario,
                        "n_train": n_train,
                        "seed": seed,
                        "validation_log_loss_by_scale": grid_losses,
                    }
                )
    result = pd.DataFrame(rows)
    expected_rows = len(SCENARIOS) * len(config.sample_sizes) * len(config.seeds) * len(MODEL_ORDER)
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} benchmark rows, produced {len(result)}")
    result["scenario"] = pd.Categorical(result["scenario"], categories=SCENARIOS, ordered=True)
    result["model"] = pd.Categorical(result["model"], categories=MODEL_ORDER, ordered=True)
    result = result.sort_values(["scenario", "n_train", "seed", "model"]).reset_index(drop=True)
    result["scenario"] = result["scenario"].astype("string")
    result["model"] = result["model"].astype("string")
    return result, scale_diagnostics


def summarize_benchmark(
    results: pd.DataFrame,
    scale_diagnostics: list[dict[str, int | str | dict[str, float]]],
    config: SimulationBenchmarkConfig,
) -> dict[str, object]:
    """Create a JSON-serializable aggregate benchmark summary."""
    metric_names = ("test_log_loss", "test_auroc", "coefficient_rmse")
    grouped = results.groupby(["scenario", "n_train", "model"], sort=False, observed=True)
    summaries: list[dict[str, int | float | str]] = []
    for (scenario, n_train, model_name), frame in grouped:
        record: dict[str, int | float | str] = {
            "scenario": str(scenario),
            "n_train": int(n_train),
            "model": str(model_name),
            "replicates": int(len(frame)),
        }
        for metric in metric_names:
            values = frame[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_se"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        summaries.append(record)
    patch_rows = results.loc[results["model"].eq("residual_patch")].copy()
    null_rows = patch_rows.loc[patch_rows["scenario"].eq("additive_null")]
    selected_zero = int(null_rows["selected_residual_scale"].eq(0.0).sum())
    return {
        "artifact_schema_version": 1,
        "benchmark": "residual PATCH synthetic benchmark",
        "interpretation": (
            "Predictive and treatment-associated coefficient recovery only. "
            "The simulation does not establish causal treatment effects."
        ),
        "configuration": asdict(config),
        "scenario_definitions": {
            "additive_null": "Constant regimen coefficients with additive context prognosis.",
            "low_rank": (
                "Three soft archetypes with sum-zero regimen deviations and linear loading logits."
            ),
            "nonlinear_misspecified": (
                "Smooth nonlinear coefficient surface outside both fitted interaction classes."
            ),
        },
        "model_definitions": {
            "additive_logistic_null": "L2 logistic context and regimen main effects.",
            "residual_patch": (
                "Frozen selected additive baseline plus a K=3 "
                "contrast-constrained residual surface."
            ),
            "unrestricted_interaction_logistic": (
                "L2 logistic main effects plus every context-by-regimen product."
            ),
        },
        "replicate_rows": int(len(results)),
        "group_summary": summaries,
        "null_scale_selection": {
            "eligible_replicates": int(len(null_rows)),
            "selected_zero": selected_zero,
            "selected_zero_rate": float(selected_zero / len(null_rows)),
            "selection_rule": (
                "Smallest scale within the prespecified validation log-loss tolerance "
                "of the grid minimum."
            ),
        },
        "residual_scale_validation_diagnostics": scale_diagnostics,
    }


def _plot_benchmark(results: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    """Write a color-safe three-metric aggregate benchmark figure."""
    metrics = (
        ("test_log_loss", "Held-out log loss", "Lower is better"),
        ("test_auroc", "Held-out AUROC", "Higher is better"),
        ("coefficient_rmse", "Coefficient surface RMSE", "Lower is better"),
    )
    combinations = [
        (scenario, sample_size)
        for scenario in SCENARIOS
        for sample_size in sorted(results["n_train"].unique())
    ]
    x_positions = np.arange(len(combinations), dtype=np.float64)
    fig, raw_axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True)
    axes = np.asarray(raw_axes, dtype=object).reshape(-1)
    for axis, (metric, title, direction) in zip(axes, metrics, strict=True):
        for model_name in MODEL_ORDER:
            means: list[float] = []
            intervals: list[float] = []
            for scenario, sample_size in combinations:
                values = results.loc[
                    results["scenario"].eq(scenario)
                    & results["n_train"].eq(sample_size)
                    & results["model"].eq(model_name),
                    metric,
                ].to_numpy(dtype=np.float64)
                means.append(float(np.mean(values)))
                if len(values) > 1:
                    intervals.append(float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values))))
                else:
                    intervals.append(0.0)
            axis.errorbar(
                x_positions,
                means,
                yerr=intervals,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=1.6,
                markersize=5.5,
                capsize=2.5,
                label=MODEL_LABELS[model_name],
            )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=20)
        axis.text(
            0.0,
            1.005,
            direction,
            transform=axis.transAxes,
            fontsize=8,
            color="#4D4D4D",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        for boundary in range(1, len(SCENARIOS)):
            axis.axvline(
                boundary * len(configured_sample_sizes(results)) - 0.5,
                color="#BDBDBD",
                linewidth=0.8,
                linestyle="--",
            )
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    labels = [
        f"{SCENARIO_LABELS[scenario]}\nn={sample_size}" for scenario, sample_size in combinations
    ]
    axes[0].set_xticks(x_positions, labels)
    axes[1].set_xticks(x_positions, labels)
    axes[2].set_xticks(x_positions, labels)
    null_patch = results.loc[
        results["scenario"].eq("additive_null") & results["model"].eq("residual_patch")
    ]
    selected_zero = int(null_patch["selected_residual_scale"].eq(0.0).sum())
    fig.suptitle(
        "Residual PATCH synthetic benchmark",
        x=0.025,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.01,
        (
            f"Points show means and 95% normal intervals across {results['seed'].nunique()} seeds. "
            f"Residual scale zero was selected in {selected_zero} of {len(null_patch)} null "
            "replicates."
        ),
        fontsize=8.5,
        color="#3F3F3F",
    )
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.67, 0.987),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.995, 0.88))
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configured_sample_sizes(results: pd.DataFrame) -> tuple[int, ...]:
    """Return sorted sample sizes represented in an aggregate result frame."""
    return tuple(sorted(int(value) for value in results["n_train"].unique()))


def write_benchmark_artifacts(
    results: pd.DataFrame,
    scale_diagnostics: list[dict[str, int | str | dict[str, float]]],
    config: SimulationBenchmarkConfig,
    output_root: Path,
) -> BenchmarkArtifactPaths:
    """Write deterministic aggregate tables, metadata, and a color-safe figure."""
    root = Path(output_root)
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    for directory in (tables_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    paths = BenchmarkArtifactPaths(
        replicate_csv=tables_dir / "residual_simulation_benchmark.csv",
        summary_json=metrics_dir / "residual_simulation_benchmark.json",
        figure_png=figures_dir / "residual_simulation_benchmark.png",
        figure_pdf=figures_dir / "residual_simulation_benchmark.pdf",
    )
    results.to_csv(paths.replicate_csv, index=False, float_format="%.8f")
    summary = summarize_benchmark(results, scale_diagnostics, config)
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_benchmark(results, paths.figure_png, paths.figure_pdf)
    return paths
