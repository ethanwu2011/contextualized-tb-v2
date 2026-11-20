"""Aggregate-only stability analysis for the locked residual PATCH profiles."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    PatchCandidate,
    RawValidationData,
    _assert_release_safe_report,
    _fit_additive,
    _fit_residual,
    _seed,
    prepare_dua_output_root,
)

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

OKABE_ITO = ("#0072B2", "#E69F00", "#009E73")


@dataclass(frozen=True)
class ProfileStabilityConfig:
    """Frozen full-cohort profile and perturbation settings."""

    seeds: tuple[int, ...] = (42, 137, 314, 2718, 8675, 104729)
    lr_c: float = 0.1
    candidate: PatchCandidate = PatchCandidate(
        name="locked_residual_k3_strong_shrinkage",
        n_archetypes=3,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.5,
        epochs=300,
        learning_rate=0.01,
    )
    min_treatment_prevalence: float = 0.01
    logistic_max_iter: int = 2000
    small_cell_min: int = 30
    bootstrap_resamples: int = 12
    bootstrap_seed: int = 20260730

    def __post_init__(self) -> None:
        """Enforce the locked analysis and finite perturbation budget."""
        if len(self.seeds) < 6 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Profile analysis requires at least six unique deterministic seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Profile seeds must be nonnegative")
        if not np.isfinite(self.lr_c) or self.lr_c <= 0:
            raise ValueError("lr_c must be finite and positive")
        if self.candidate.is_null or self.candidate.n_archetypes != 3:
            raise ValueError("The locked profile candidate must be non-null with K=3")
        if not np.isclose(self.candidate.residual_scale, 0.5):
            raise ValueError("The locked profile candidate must use residual scale 0.5")
        if not np.isclose(self.candidate.loading_l2, 1e-3) or not np.isclose(
            self.candidate.deviation_l2,
            1e-2,
        ):
            raise ValueError("The locked profile candidate must use strong shrinkage")
        if not 0 <= self.min_treatment_prevalence <= 1:
            raise ValueError("min_treatment_prevalence must be between zero and one")
        if self.logistic_max_iter < 1:
            raise ValueError("logistic_max_iter must be positive")
        if self.small_cell_min < 1:
            raise ValueError("small_cell_min must be positive")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples must be nonnegative")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be nonnegative")

    def validation_config(self) -> LockedValidationConfig:
        """Return the matching cohort and preprocessing settings."""
        return LockedValidationConfig(
            outer_splits=5,
            inner_splits=3,
            lr_c_values=(self.lr_c,),
            patch_candidates=(PatchCandidate(name="null"), self.candidate),
            min_treatment_prevalence=self.min_treatment_prevalence,
            logistic_max_iter=self.logistic_max_iter,
            bootstrap_resamples=10,
            seed=self.seeds[0],
        )


@dataclass(frozen=True)
class ProfileStabilityResult:
    """Private aligned loadings plus release-safe aggregate tables."""

    condition_ids: NDArray[Any]
    patient_groups: NDArray[Any]
    y_true: IntArray
    seeds: IntArray
    aligned_soft_loadings: FloatArray
    residual_coefficient_surfaces: FloatArray
    observed_regimen_residual_logit_contributions: FloatArray
    consensus_soft_loadings: FloatArray
    consensus_hard_profile: IntArray
    aggregate_report: dict[str, Any]
    aggregate_tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class _SeedProfile:
    """One fitted seed's profile parameters and prediction-invariant outputs."""

    seed: int
    total_coefficients: FloatArray
    deviation_coefficients: FloatArray
    soft_loadings: FloatArray
    residual_coefficient_surface: FloatArray
    observed_regimen_residual_logit_contribution: FloatArray


def _cosine(left: FloatArray, right: FloatArray) -> float:
    """Return cosine similarity, failing on an unidentified zero vector."""
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("Cannot align a zero-norm residual coefficient profile")
    return float(np.dot(left, right) / (left_norm * right_norm))


def align_profile_to_reference(
    reference_deviations: FloatArray,
    current: _SeedProfile,
) -> _SeedProfile:
    """Align one seed to a reference using Hungarian matching on residual coefficients."""
    reference = np.asarray(reference_deviations, dtype=np.float64)
    deviations = np.asarray(current.deviation_coefficients, dtype=np.float64)
    if reference.ndim != 2 or deviations.shape != reference.shape:
        raise ValueError("Reference and current deviation profiles must have equal 2D shapes")
    n_profiles = reference.shape[0]
    similarity = np.asarray(
        [
            [_cosine(reference[row], deviations[column]) for column in range(n_profiles)]
            for row in range(n_profiles)
        ],
        dtype=np.float64,
    )
    reference_rows, current_columns = linear_sum_assignment(-similarity)
    if not np.array_equal(reference_rows, np.arange(n_profiles)):
        order = np.argsort(reference_rows)
        current_columns = current_columns[order]
    return _SeedProfile(
        seed=current.seed,
        total_coefficients=current.total_coefficients[current_columns],
        deviation_coefficients=current.deviation_coefficients[current_columns],
        soft_loadings=current.soft_loadings[:, current_columns],
        residual_coefficient_surface=current.residual_coefficient_surface,
        observed_regimen_residual_logit_contribution=(
            current.observed_regimen_residual_logit_contribution
        ),
    )


def _fit_seed_profile(
    matrices: FoldMatrices,
    outcome: IntArray,
    config: ProfileStabilityConfig,
    *,
    seed: int,
    evaluation_matrices: FoldMatrices | None = None,
) -> _SeedProfile:
    """Fit the frozen model and evaluate its residual surface on a common matrix."""
    evaluation = evaluation_matrices or matrices
    if evaluation.context_feature_names != matrices.context_feature_names:
        raise ValueError("Training and evaluation context feature names must match")
    if evaluation.treatment_feature_names != matrices.treatment_feature_names:
        raise ValueError("Training and evaluation treatment feature names must match")
    baseline = _fit_additive(
        matrices,
        outcome,
        c_value=config.lr_c,
        max_iter=config.logistic_max_iter,
        seed=seed,
    )
    model = _fit_residual(
        baseline,
        matrices,
        outcome,
        config.candidate,
        seed=seed,
    )
    total = model.prototype_coefficients()
    deviations = total - total.mean(axis=0, keepdims=True)
    loadings = model.predict_loadings(evaluation.C)
    residual_surface = np.asarray(loadings @ deviations, dtype=np.float64)
    observed_contribution = np.asarray(
        np.sum(evaluation.X * residual_surface, axis=1),
        dtype=np.float64,
    )
    if not np.allclose(loadings.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Fitted soft loadings do not sum to one")
    if not np.isfinite(residual_surface).all() or not np.isfinite(observed_contribution).all():
        raise ValueError("Fitted prediction-invariant residual outputs must be finite")
    return _SeedProfile(
        seed=seed,
        total_coefficients=np.asarray(total, dtype=np.float64),
        deviation_coefficients=np.asarray(deviations, dtype=np.float64),
        soft_loadings=np.asarray(loadings, dtype=np.float64),
        residual_coefficient_surface=residual_surface,
        observed_regimen_residual_logit_contribution=observed_contribution,
    )


def _align_seed_profiles(profiles: list[_SeedProfile]) -> list[_SeedProfile]:
    """Use a fixed first-seed reference, then refine once against the aligned consensus."""
    if len(profiles) < 2:
        raise ValueError("At least two seed profiles are required for alignment")
    first_pass = [
        profiles[0],
        *[
            align_profile_to_reference(profiles[0].deviation_coefficients, profile)
            for profile in profiles[1:]
        ],
    ]
    consensus = np.mean(
        np.stack([profile.deviation_coefficients for profile in first_pass]),
        axis=0,
    )
    return [align_profile_to_reference(consensus, profile) for profile in profiles]


def _pearson(left: FloatArray, right: FloatArray) -> float:
    """Return a finite Pearson correlation for two nonconstant loading vectors."""
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        raise ValueError("Soft-loading stability is undefined for a constant profile loading")
    return float(np.corrcoef(left, right)[0, 1])


def _jensen_shannon_rows(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return row-wise Jensen-Shannon divergence for two simplex matrices."""
    epsilon = 1e-12
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(
        left * np.log((left + epsilon) / (midpoint + epsilon)),
        axis=1,
    )
    divergence += 0.5 * np.sum(
        right * np.log((right + epsilon) / (midpoint + epsilon)),
        axis=1,
    )
    return np.asarray(divergence, dtype=np.float64)


def _normalized_rmse(left: FloatArray, right: FloatArray) -> float:
    """Return RMSE divided by the symmetric pooled root-mean-square magnitude."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise ValueError("Normalized RMSE requires equal nonempty arrays")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("Normalized RMSE inputs must be finite")
    rmse = float(np.sqrt(np.mean(np.square(left_array - right_array))))
    pooled_rms = float(
        np.sqrt(0.5 * (np.mean(np.square(left_array)) + np.mean(np.square(right_array))))
    )
    if pooled_rms <= 1e-12:
        if rmse <= 1e-12:
            return 0.0
        raise ValueError("Normalized RMSE is undefined when pooled magnitude is zero")
    return rmse / pooled_rms


def _prediction_invariant_array_metrics(
    left_surface: FloatArray,
    right_surface: FloatArray,
    left_observed: FloatArray,
    right_observed: FloatArray,
) -> dict[str, float]:
    """Compare invariant residual surfaces and observed-regimen contributions."""
    left_surface_vector = np.asarray(left_surface, dtype=np.float64).reshape(-1)
    right_surface_vector = np.asarray(right_surface, dtype=np.float64).reshape(-1)
    left_observed_vector = np.asarray(left_observed, dtype=np.float64).reshape(-1)
    right_observed_vector = np.asarray(right_observed, dtype=np.float64).reshape(-1)
    return {
        "residual_coefficient_surface_cosine": _cosine(
            left_surface_vector,
            right_surface_vector,
        ),
        "residual_coefficient_surface_pearson": _pearson(
            left_surface_vector,
            right_surface_vector,
        ),
        "residual_coefficient_surface_normalized_rmse": _normalized_rmse(
            left_surface_vector,
            right_surface_vector,
        ),
        "observed_regimen_residual_logit_contribution_cosine": _cosine(
            left_observed_vector,
            right_observed_vector,
        ),
        "observed_regimen_residual_logit_contribution_pearson": _pearson(
            left_observed_vector,
            right_observed_vector,
        ),
        "observed_regimen_residual_logit_contribution_normalized_rmse": _normalized_rmse(
            left_observed_vector,
            right_observed_vector,
        ),
    }


def _prediction_invariant_metrics(
    left: _SeedProfile,
    right: _SeedProfile,
) -> dict[str, float]:
    """Compare prediction-invariant outputs from two fitted seeds."""
    return _prediction_invariant_array_metrics(
        left.residual_coefficient_surface,
        right.residual_coefficient_surface,
        left.observed_regimen_residual_logit_contribution,
        right.observed_regimen_residual_logit_contribution,
    )


def _seed_pair_tables(
    profiles: list[_SeedProfile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aggregate profile-wise and global seed-pair stability tables."""
    profile_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for left, right in combinations(profiles, 2):
        left_labels = left.soft_loadings.argmax(axis=1)
        right_labels = right.soft_loadings.argmax(axis=1)
        absolute_difference = np.abs(left.soft_loadings - right.soft_loadings)
        js = _jensen_shannon_rows(left.soft_loadings, right.soft_loadings)
        ari = float(adjusted_rand_score(left_labels, right_labels))
        global_rows.append(
            {
                "aggregation_unit": "seed_pair",
                "seed_a": left.seed,
                "seed_b": right.seed,
                "soft_loading_mae": float(absolute_difference.mean()),
                "soft_loading_mae_p95": float(np.percentile(absolute_difference, 95)),
                "soft_loading_js_mean": float(js.mean()),
                "soft_loading_js_p95": float(np.percentile(js, 95)),
                "hard_label_ari_secondary": ari,
                **_prediction_invariant_metrics(left, right),
            }
        )
        for profile in range(left.deviation_coefficients.shape[0]):
            profile_rows.append(
                {
                    "aggregation_unit": "seed_pair_profile",
                    "seed_a": left.seed,
                    "seed_b": right.seed,
                    "profile": profile + 1,
                    "deviation_coefficient_cosine": _cosine(
                        left.deviation_coefficients[profile],
                        right.deviation_coefficients[profile],
                    ),
                    "total_coefficient_cosine": _cosine(
                        left.total_coefficients[profile],
                        right.total_coefficients[profile],
                    ),
                    "soft_loading_pearson": _pearson(
                        left.soft_loadings[:, profile],
                        right.soft_loadings[:, profile],
                    ),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(global_rows)


def _profile_summary(
    consensus_loadings: FloatArray,
    outcome: IntArray,
    *,
    small_cell_min: int,
) -> pd.DataFrame:
    """Summarize soft occupancy and outcome association with hard bins secondary."""
    hard = consensus_loadings.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for profile in range(consensus_loadings.shape[1]):
        weights = consensus_loadings[:, profile]
        effective_count = float(weights.sum())
        hard_mask = hard == profile
        hard_count = int(hard_mask.sum())
        hard_events = int(outcome[hard_mask].sum())
        hard_nonevents = hard_count - hard_events
        suppress_hard_rate = min(hard_events, hard_nonevents) < small_cell_min
        rows.append(
            {
                "aggregation_unit": "profile",
                "profile": profile + 1,
                "soft_effective_count": effective_count,
                "soft_mean_loading": float(weights.mean()),
                "soft_median_loading": float(np.median(weights)),
                "soft_loading_p90": float(np.percentile(weights, 90)),
                "soft_weighted_event_rate": float(np.dot(weights, outcome) / effective_count),
                "hard_count_secondary": (hard_count if hard_count >= small_cell_min else pd.NA),
                "hard_event_rate_secondary": (
                    float(hard_events / hard_count)
                    if hard_count >= small_cell_min and not suppress_hard_rate
                    else np.nan
                ),
                "hard_rate_suppressed": bool(hard_count < small_cell_min or suppress_hard_rate),
            }
        )
    return pd.DataFrame(rows)


def _regimen_label(feature_name: str) -> str:
    """Return a human-readable label for one deterministic treatment feature name."""
    if feature_name == "drug_other_second_line":
        return "Other second-line"
    return unquote(feature_name.removeprefix("drug_"))


def _regimen_support_table(
    matrices: FoldMatrices,
    consensus_loadings: FloatArray,
    *,
    small_cell_min: int,
) -> pd.DataFrame:
    """Build support-audited soft receipt fractions by consensus profile."""
    hard = consensus_loadings.argmax(axis=1)
    n_rows = matrices.n_rows
    rows: list[dict[str, Any]] = []
    for profile in range(consensus_loadings.shape[1]):
        hard_mask = hard == profile
        hard_count = int(hard_mask.sum())
        weights = consensus_loadings[:, profile]
        for treatment_position, treatment in enumerate(matrices.treatment_feature_names):
            receipt = matrices.X[:, treatment_position] == 1
            cohort_receipt = int(receipt.sum())
            hard_receipt = int((receipt & hard_mask).sum())
            hard_nonreceipt = hard_count - hard_receipt
            unsafe = (
                min(cohort_receipt, n_rows - cohort_receipt) < small_cell_min
                or hard_count < small_cell_min
                or min(hard_receipt, hard_nonreceipt) < small_cell_min
            )
            rows.append(
                {
                    "aggregation_unit": "profile_regimen",
                    "profile": profile + 1,
                    "regimen_feature": treatment,
                    "regimen_label": _regimen_label(treatment),
                    "cohort_receipt_count": (
                        cohort_receipt
                        if min(cohort_receipt, n_rows - cohort_receipt) >= small_cell_min
                        else pd.NA
                    ),
                    "hard_profile_count": (hard_count if hard_count >= small_cell_min else pd.NA),
                    "hard_profile_receipt_count": hard_receipt if not unsafe else pd.NA,
                    "soft_weighted_receipt_fraction": (
                        float(np.dot(weights, receipt.astype(np.float64)) / weights.sum())
                        if not unsafe
                        else np.nan
                    ),
                    "small_cell_suppressed": unsafe,
                    "small_cell_threshold": small_cell_min,
                }
            )
    return pd.DataFrame(rows)


def _coefficient_table(
    profiles: list[_SeedProfile],
    regimen_support: pd.DataFrame,
    treatment_names: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize aligned total and residual regimen coefficients across seeds."""
    total = np.stack([profile.total_coefficients for profile in profiles])
    deviation = np.stack([profile.deviation_coefficients for profile in profiles])
    suppression = {
        (int(str(row.profile)), str(row.regimen_feature)): bool(row.small_cell_suppressed)
        for row in regimen_support.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for profile in range(total.shape[1]):
        for treatment_position, treatment in enumerate(treatment_names):
            suppressed = suppression[(profile + 1, treatment)]
            rows.append(
                {
                    "aggregation_unit": "seed_profile_regimen",
                    "profile": profile + 1,
                    "regimen_feature": treatment,
                    "regimen_label": _regimen_label(treatment),
                    "mean_deviation_coefficient": (
                        float(deviation[:, profile, treatment_position].mean())
                        if not suppressed
                        else np.nan
                    ),
                    "sd_deviation_coefficient": (
                        float(deviation[:, profile, treatment_position].std(ddof=1))
                        if not suppressed
                        else np.nan
                    ),
                    "mean_total_coefficient": (
                        float(total[:, profile, treatment_position].mean())
                        if not suppressed
                        else np.nan
                    ),
                    "small_cell_suppressed": suppressed,
                }
            )
    return pd.DataFrame(rows)


def _patient_cluster_bootstrap_rows(
    patient_groups: NDArray[Any],
    *,
    n_resamples: int,
    seed: int,
) -> list[IntArray]:
    """Return patient-cluster bootstrap row indices without exposing group values."""
    groups = np.asarray(patient_groups).astype(str)
    unique_groups = np.unique(groups)
    rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [rows[position] for position in rng.integers(0, len(rows), len(rows))]
        ).astype(np.int64)
        for _ in range(n_resamples)
    ]


def _bootstrap_coefficient_table(
    matrices: FoldMatrices,
    outcome: IntArray,
    patient_groups: NDArray[Any],
    reference_deviations: FloatArray,
    reference_residual_surface: FloatArray,
    reference_observed_contribution: FloatArray,
    config: ProfileStabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit perturbations and compare profiles plus prediction-invariant outputs."""
    profile_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    sampled_rows = _patient_cluster_bootstrap_rows(
        patient_groups,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    for replicate, sample in enumerate(sampled_rows, start=1):
        sampled_matrices = FoldMatrices(
            C=np.asarray(matrices.C[sample], dtype=np.float64),
            X=np.asarray(matrices.X[sample], dtype=np.float64),
            context_feature_names=matrices.context_feature_names,
            treatment_feature_names=matrices.treatment_feature_names,
            row_index=pd.RangeIndex(len(sample)),
        )
        sampled_outcome = outcome[sample]
        if len(np.unique(sampled_outcome)) != 2:
            raise ValueError("A patient-cluster bootstrap perturbation lost one outcome class")
        fitted = _fit_seed_profile(
            sampled_matrices,
            sampled_outcome,
            config,
            seed=_seed(config.bootstrap_seed, replicate),
            evaluation_matrices=matrices,
        )
        aligned = align_profile_to_reference(reference_deviations, fitted)
        for profile in range(reference_deviations.shape[0]):
            profile_rows.append(
                {
                    "aggregation_unit": "patient_cluster_bootstrap_profile",
                    "bootstrap_replicate": replicate,
                    "profile": profile + 1,
                    "deviation_coefficient_cosine_to_consensus": _cosine(
                        reference_deviations[profile],
                        aligned.deviation_coefficients[profile],
                    ),
                }
            )
        invariant_rows.append(
            {
                "aggregation_unit": "patient_cluster_bootstrap_prediction_invariant",
                "bootstrap_replicate": replicate,
                **_prediction_invariant_array_metrics(
                    reference_residual_surface,
                    aligned.residual_coefficient_surface,
                    reference_observed_contribution,
                    aligned.observed_regimen_residual_logit_contribution,
                ),
            }
        )
        logger.info(
            "Profile stability patient-cluster perturbation %d/%d complete",
            replicate,
            config.bootstrap_resamples,
        )
    return pd.DataFrame(profile_rows), pd.DataFrame(invariant_rows)


def _summary_interval(values: pd.Series[Any]) -> dict[str, Any]:
    """Return finite median and central 95 percent range."""
    array = values.to_numpy(dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "ci95": [
            float(np.percentile(array, 2.5)),
            float(np.percentile(array, 97.5)),
        ],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_profile_stability(
    data: RawValidationData,
    config: ProfileStabilityConfig | None = None,
) -> ProfileStabilityResult:
    """Fit, align, and summarize the frozen residual profile model."""
    config = config or ProfileStabilityConfig()
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=config.min_treatment_prevalence,
    )
    matrices = preprocessor.fit_transform(data.context, data.treatments)
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    profiles: list[_SeedProfile] = []
    for position, seed in enumerate(config.seeds, start=1):
        profiles.append(_fit_seed_profile(matrices, outcome, config, seed=seed))
        logger.info("Full-cohort residual profile seed %d/%d complete", position, len(config.seeds))
    aligned = _align_seed_profiles(profiles)
    aligned_loadings = np.stack([profile.soft_loadings for profile in aligned])
    residual_surfaces = np.stack([profile.residual_coefficient_surface for profile in aligned])
    observed_contributions = np.stack(
        [profile.observed_regimen_residual_logit_contribution for profile in aligned]
    )
    consensus_loadings = np.asarray(aligned_loadings.mean(axis=0), dtype=np.float64)
    consensus_deviations = np.asarray(
        np.mean(
            np.stack([profile.deviation_coefficients for profile in aligned]),
            axis=0,
        ),
        dtype=np.float64,
    )
    consensus_residual_surface = np.asarray(
        residual_surfaces.mean(axis=0),
        dtype=np.float64,
    )
    consensus_observed_contribution = np.asarray(
        observed_contributions.mean(axis=0),
        dtype=np.float64,
    )
    consensus_hard = consensus_loadings.argmax(axis=1).astype(np.int64)

    seed_pair_profile, seed_pair_global = _seed_pair_tables(aligned)
    profile_summary = _profile_summary(
        consensus_loadings,
        outcome,
        small_cell_min=config.small_cell_min,
    )
    regimen_support = _regimen_support_table(
        matrices,
        consensus_loadings,
        small_cell_min=config.small_cell_min,
    )
    coefficients = _coefficient_table(
        aligned,
        regimen_support,
        matrices.treatment_feature_names,
    )
    bootstrap, bootstrap_invariant = _bootstrap_coefficient_table(
        matrices,
        outcome,
        groups,
        consensus_deviations,
        consensus_residual_surface,
        consensus_observed_contribution,
        config,
    )

    profile_stability: list[dict[str, Any]] = []
    for profile in range(config.candidate.n_archetypes or 0):
        profile_rows = seed_pair_profile.loc[seed_pair_profile["profile"].eq(profile + 1)]
        bootstrap_rows = bootstrap.loc[bootstrap["profile"].eq(profile + 1)]
        profile_stability.append(
            {
                "profile": profile + 1,
                "deviation_coefficient_cosine": _summary_interval(
                    profile_rows["deviation_coefficient_cosine"]
                ),
                "total_coefficient_cosine": _summary_interval(
                    profile_rows["total_coefficient_cosine"]
                ),
                "soft_loading_pearson": _summary_interval(profile_rows["soft_loading_pearson"]),
                "bootstrap_deviation_cosine_to_consensus": (
                    _summary_interval(bootstrap_rows["deviation_coefficient_cosine_to_consensus"])
                    if len(bootstrap_rows)
                    else None
                ),
            }
        )
    hard_ari = _summary_interval(seed_pair_global["hard_label_ari_secondary"])
    soft_mae = _summary_interval(seed_pair_global["soft_loading_mae"])
    soft_js = _summary_interval(seed_pair_global["soft_loading_js_mean"])
    invariant_columns = (
        "residual_coefficient_surface_cosine",
        "residual_coefficient_surface_pearson",
        "residual_coefficient_surface_normalized_rmse",
        "observed_regimen_residual_logit_contribution_cosine",
        "observed_regimen_residual_logit_contribution_pearson",
        "observed_regimen_residual_logit_contribution_normalized_rmse",
    )
    seed_invariant_summary = {
        column: _summary_interval(seed_pair_global[column]) for column in invariant_columns
    }
    bootstrap_invariant_summary = (
        {column: _summary_interval(bootstrap_invariant[column]) for column in invariant_columns}
        if len(bootstrap_invariant)
        else None
    )
    entropy = -np.sum(
        consensus_loadings * np.log(consensus_loadings + 1e-12),
        axis=1,
    )
    report: dict[str, Any] = {
        "report_schema_version": 2,
        "analysis_id": "locked_residual_patch_profile_stability",
        "endpoint": "recorded_final_patient_cases_died_or_failure",
        "interpretation": "descriptive treatment-associated residual profiles, not causal effects",
        "locked_model": {
            "lr_c": config.lr_c,
            "n_archetypes": config.candidate.n_archetypes,
            "loading_l2": config.candidate.loading_l2,
            "deviation_l2": config.candidate.deviation_l2,
            "residual_scale": config.candidate.residual_scale,
            "epochs": config.candidate.epochs,
            "selection_basis": "K=3 strong shrinkage selected in four of five locked outer folds",
            "profile_result_tuning": False,
            "alignment_basis": "residual regimen coefficient cosine with Hungarian matching",
        },
        "cohort": {
            "assessed_conditions": len(outcome),
            "patients": int(len(np.unique(groups))),
            "events": int(outcome.sum()),
            "observed_non_events": int(len(outcome) - outcome.sum()),
            "retained_context_features": matrices.C.shape[1],
            "retained_regimen_features": matrices.X.shape[1],
        },
        "seed_analysis": {
            "seeds": list(config.seeds),
            "n_seed_pairs": len(seed_pair_global),
            "profile_stability": profile_stability,
            "soft_loading_mae": soft_mae,
            "soft_loading_js_mean": soft_js,
            "hard_label_ari_secondary": hard_ari,
            "prediction_invariant_stability_primary": {
                "residual_surface_definition": (
                    "prediction-scale z(C) @ D evaluated for every assessed condition and "
                    "retained regimen feature"
                ),
                "observed_regimen_contribution_definition": (
                    "row-wise sum of observed regimen indicators times the residual "
                    "coefficient surface"
                ),
                "normalized_rmse_definition": (
                    "RMSE divided by the square root of the mean of the two compared "
                    "mean-square magnitudes"
                ),
                "pairwise_metrics": seed_invariant_summary,
            },
            "consensus_loading_entropy": {
                "mean": float(entropy.mean()),
                "median": float(np.median(entropy)),
                "maximum": float(entropy.max()),
            },
        },
        "patient_cluster_bootstrap": {
            "resamples": config.bootstrap_resamples,
            "unit": "patient",
            "preprocessing_scope": (
                "fixed full-cohort encoder for conditional perturbation stability"
            ),
            "seed": config.bootstrap_seed,
            "prediction_invariant_stability_to_seed_consensus": (bootstrap_invariant_summary),
        },
        "small_cell_suppression": {
            "threshold": config.small_cell_min,
            "suppressed_profile_regimen_cells": int(regimen_support["small_cell_suppressed"].sum()),
            "total_profile_regimen_cells": len(regimen_support),
        },
        "profile_summary": profile_summary.replace({np.nan: None}).to_dict(orient="records"),
        "governance": {
            "public_outputs_contain_row_level_data": False,
            "public_outputs_contain_identifiers": False,
            "private_loadings_required_for_audit": True,
        },
        "limitations": [
            "Profile labels are identified only up to permutation and are aligned post fit.",
            "Bootstrap perturbations hold the full-cohort encoder fixed and do not include "
            "preprocessing uncertainty.",
            "Outcome and regimen summaries are descriptive associations.",
            "Fixed-epoch optimization does not establish a unique global optimum.",
            "Prediction-invariant stability is evaluated on the same full cohort and therefore "
            "does not establish transportability.",
            "Observed-regimen residual logit contributions reflect recorded treatment support "
            "and are not causal treatment effects.",
        ],
    }
    _assert_release_safe_report(report)
    tables = {
        "profile_summary": profile_summary,
        "regimen_support": regimen_support,
        "coefficient_profiles": coefficients,
        "seed_pair_profile_stability": seed_pair_profile,
        "seed_pair_global_stability": seed_pair_global,
        "bootstrap_coefficient_stability": bootstrap,
        "bootstrap_prediction_invariant_stability": bootstrap_invariant,
    }
    for name, table in tables.items():
        _assert_public_table_safe(table, name=name)
    return ProfileStabilityResult(
        condition_ids=data.context.index.to_numpy(copy=True),
        patient_groups=groups.copy(),
        y_true=outcome,
        seeds=np.asarray(config.seeds, dtype=np.int64),
        aligned_soft_loadings=aligned_loadings,
        residual_coefficient_surfaces=residual_surfaces,
        observed_regimen_residual_logit_contributions=observed_contributions,
        consensus_soft_loadings=consensus_loadings,
        consensus_hard_profile=consensus_hard,
        aggregate_report=report,
        aggregate_tables=tables,
    )


_FORBIDDEN_PUBLIC_TABLE_COLUMNS = {
    "condition_id",
    "patient_id",
    "outcome",
    "soft_loadings",
    "hard_label",
    "row_index",
}


def _assert_public_table_safe(table: pd.DataFrame, *, name: str) -> None:
    """Reject identifier or row-level columns from an aggregate table."""
    forbidden = sorted(set(table.columns).intersection(_FORBIDDEN_PUBLIC_TABLE_COLUMNS))
    if forbidden:
        raise ValueError(f"Public aggregate table {name!r} contains forbidden columns {forbidden}")
    if "aggregation_unit" not in table.columns:
        raise ValueError(f"Public aggregate table {name!r} lacks an aggregation_unit")


def _write_profile_figure(
    result: ProfileStabilityResult,
    pdf: Path,
    png: Path,
) -> None:
    """Write a color-safe aggregate profile and stability figure."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    tables = result.aggregate_tables
    coefficients = tables["coefficient_profiles"]
    profile_summary = tables["profile_summary"]
    seed_pair = tables["seed_pair_profile_stability"]
    seed_pair_global = tables["seed_pair_global_stability"]
    bootstrap = tables["bootstrap_coefficient_stability"]
    bootstrap_invariant = tables["bootstrap_prediction_invariant_stability"]

    available = coefficients.loc[~coefficients["small_cell_suppressed"]].copy()
    available["magnitude"] = available["mean_deviation_coefficient"].abs()
    top_treatments = (
        available.groupby("regimen_label")["magnitude"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    heatmap = (
        available.loc[available["regimen_label"].isin(top_treatments)]
        .pivot(index="profile", columns="regimen_label", values="mean_deviation_coefficient")
        .reindex(index=[1, 2, 3], columns=top_treatments)
    )

    figure, raw_axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    axes = np.asarray(raw_axes, dtype=object).reshape(-1)
    heatmap_values = heatmap.to_numpy(dtype=float)
    coefficient_limit = float(np.nanmax(np.abs(heatmap_values)))
    image = axes[0].imshow(
        heatmap_values,
        aspect="auto",
        cmap="PuOr_r",
        vmin=-coefficient_limit,
        vmax=coefficient_limit,
    )
    axes[0].set(
        title="Mean residual regimen coefficients",
        xlabel="Regimen feature",
        ylabel="Aligned profile",
        yticks=np.arange(3),
        yticklabels=["1", "2", "3"],
        xticks=np.arange(len(top_treatments)),
        xticklabels=top_treatments,
    )
    axes[0].tick_params(axis="x", rotation=55, labelsize=7)
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    profile_positions = np.arange(3)
    effective_count = profile_summary["soft_effective_count"].to_numpy(dtype=float)
    event_rate = profile_summary["soft_weighted_event_rate"].to_numpy(dtype=float)
    axes[1].bar(profile_positions, effective_count, color=OKABE_ITO, alpha=0.85)
    axes[1].set(
        title="Soft profile occupancy and event rate",
        xlabel="Aligned profile",
        ylabel="Soft effective count",
        xticks=profile_positions,
        xticklabels=["1", "2", "3"],
    )
    event_axis = axes[1].twinx()
    event_axis.plot(
        profile_positions,
        event_rate,
        color="#000000",
        marker="o",
        linewidth=1.5,
    )
    event_axis.set_ylabel("Soft weighted event rate")
    event_axis.set_ylim(0, max(0.25, float(event_rate.max()) * 1.1))

    coefficient_data = [
        seed_pair.loc[seed_pair["profile"].eq(profile), "deviation_coefficient_cosine"]
        for profile in (1, 2, 3)
    ]
    loading_data = [
        seed_pair.loc[seed_pair["profile"].eq(profile), "soft_loading_pearson"]
        for profile in (1, 2, 3)
    ]
    for profile, color in enumerate(OKABE_ITO):
        axes[2].scatter(
            np.full(len(coefficient_data[profile]), profile + 0.85),
            coefficient_data[profile],
            color=color,
            marker="o",
            s=22,
            alpha=0.7,
        )
        axes[2].scatter(
            np.full(len(loading_data[profile]), profile + 1.15),
            loading_data[profile],
            color=color,
            marker="x",
            s=28,
            alpha=0.8,
        )
    axes[2].set(
        title="Seed-pair stability",
        xlabel="Aligned profile",
        ylabel="Similarity or correlation",
        xticks=[1, 2, 3],
        xlim=(0.5, 3.5),
        ylim=(-0.05, 1.05),
    )
    axes[2].text(
        0.02,
        0.04,
        "circles  coefficient cosine\ncrosses  soft-loading Pearson",
        transform=axes[2].transAxes,
        fontsize=8,
    )

    if len(bootstrap):
        bootstrap_data = [
            bootstrap.loc[
                bootstrap["profile"].eq(profile),
                "deviation_coefficient_cosine_to_consensus",
            ]
            for profile in (1, 2, 3)
        ]
        box = axes[3].boxplot(
            bootstrap_data,
            tick_labels=["1", "2", "3"],
            patch_artist=True,
            widths=0.55,
        )
        for patch, color in zip(box["boxes"], OKABE_ITO, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[3].set(
            title="Patient-cluster perturbation stability",
            xlabel="Aligned profile",
            ylabel="Coefficient cosine to consensus",
            ylim=(-0.05, 1.05),
        )
    else:
        axes[3].axis("off")

    invariant_tick_labels = [
        "Surface\ncosine",
        "Observed\ncosine",
        "Surface\nNRMSE",
        "Observed\nNRMSE",
    ]
    invariant_colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[1])
    seed_invariant_data = [
        seed_pair_global["residual_coefficient_surface_cosine"],
        seed_pair_global["observed_regimen_residual_logit_contribution_cosine"],
        seed_pair_global["residual_coefficient_surface_normalized_rmse"],
        seed_pair_global["observed_regimen_residual_logit_contribution_normalized_rmse"],
    ]
    seed_box = axes[4].boxplot(
        seed_invariant_data,
        tick_labels=invariant_tick_labels,
        patch_artist=True,
        widths=0.55,
    )
    for patch, color in zip(seed_box["boxes"], invariant_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[4].set(
        title="Seed-pair prediction-invariant stability",
        ylabel="Higher cosine or lower NRMSE is better",
    )

    if len(bootstrap_invariant):
        bootstrap_invariant_data = [
            bootstrap_invariant["residual_coefficient_surface_cosine"],
            bootstrap_invariant["observed_regimen_residual_logit_contribution_cosine"],
            bootstrap_invariant["residual_coefficient_surface_normalized_rmse"],
            bootstrap_invariant["observed_regimen_residual_logit_contribution_normalized_rmse"],
        ]
        bootstrap_box = axes[5].boxplot(
            bootstrap_invariant_data,
            tick_labels=invariant_tick_labels,
            patch_artist=True,
            widths=0.55,
        )
        for patch, color in zip(bootstrap_box["boxes"], invariant_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[5].set(
            title="Bootstrap prediction-invariant stability",
            ylabel="Higher cosine or lower NRMSE is better",
        )
    else:
        axes[5].axis("off")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Locked residual PATCH profiles", fontsize=13)
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_profile_outputs(
    result: ProfileStabilityResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write private loadings and aggregate-only profile artifacts."""
    root = output_root.expanduser().resolve()
    private_dir = root / "dua_private"
    aggregate_dir = root / "aggregate_only"
    if not private_dir.is_dir() or not aggregate_dir.is_dir():
        raise FileNotFoundError("Output root must first be created by prepare_dua_output_root")
    private_npz = private_dir / "profile_loadings_DUA_PRIVATE.npz"
    private_readme = private_dir / "README_DUA_PRIVATE.txt"
    report_json = aggregate_dir / "residual_profile_stability.json"
    figure_pdf = aggregate_dir / "residual_profile_stability.pdf"
    figure_png = aggregate_dir / "residual_profile_stability.png"
    table_paths = {name: aggregate_dir / f"{name}.csv" for name in result.aggregate_tables}
    targets = [
        private_npz,
        private_readme,
        report_json,
        figure_pdf,
        figure_png,
        *table_paths.values(),
    ]
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Profile output targets already exist: {existing}")

    _assert_release_safe_report(result.aggregate_report)
    report_json.write_text(
        json.dumps(
            result.aggregate_report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, table in result.aggregate_tables.items():
        _assert_public_table_safe(table, name=name)
        table.to_csv(table_paths[name], index=False)
    np.savez_compressed(
        private_npz,
        condition_id=np.asarray(result.condition_ids, dtype=str),
        patient_id=np.asarray(result.patient_groups, dtype=str),
        outcome=result.y_true,
        seeds=result.seeds,
        aligned_soft_loadings=result.aligned_soft_loadings,
        residual_coefficient_surfaces=result.residual_coefficient_surfaces,
        observed_regimen_residual_logit_contributions=(
            result.observed_regimen_residual_logit_contributions
        ),
        consensus_soft_loadings=result.consensus_soft_loadings,
        consensus_hard_profile=result.consensus_hard_profile,
    )
    private_readme.write_text(
        "DUA-PRIVATE\n"
        "This directory contains row-level TB Portals identifiers, outcomes, soft profile "
        "loadings, residual coefficient surfaces, observed-regimen residual logit "
        "contributions, and hard profile diagnostics. Do not release it publicly.\n",
        encoding="utf-8",
    )
    _write_profile_figure(result, figure_pdf, figure_png)
    for path in targets:
        os.chmod(path, 0o600)
    return {
        "private_loadings": private_npz,
        "private_readme": private_readme,
        "aggregate_report": report_json,
        "aggregate_figure_pdf": figure_pdf,
        "aggregate_figure_png": figure_png,
        **{f"aggregate_table_{name}": path for name, path in table_paths.items()},
    }


__all__ = [
    "ProfileStabilityConfig",
    "ProfileStabilityResult",
    "align_profile_to_reference",
    "prepare_dua_output_root",
    "run_profile_stability",
    "write_profile_outputs",
]
