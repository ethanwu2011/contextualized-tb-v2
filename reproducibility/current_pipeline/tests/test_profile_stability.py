from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ctb.models.profile_stability as ps
from ctb.models.locked_validation import PatchCandidate, RawValidationData, prepare_dua_output_root
from ctb.models.profile_stability import (
    ProfileStabilityConfig,
    align_profile_to_reference,
    run_profile_stability,
    write_profile_outputs,
)


def _synthetic_data(n_patients: int = 36) -> RawValidationData:
    rng = np.random.default_rng(71)
    n_rows = 2 * n_patients
    index = pd.Index([f"c{row:03d}" for row in range(n_rows)], name="condition_id")
    groups = np.repeat([f"p{group:03d}" for group in range(n_patients)], 2)
    group_outcome = np.asarray([group % 2 for group in range(n_patients)], dtype=int)
    outcome = np.repeat(group_outcome, 2)
    context = pd.DataFrame(
        {
            "age": 35 + rng.normal(size=n_rows) + outcome,
            "marker": rng.normal(size=n_rows) + 0.5 * outcome,
            "country": np.where(np.arange(n_rows) % 3, "A", "B"),
        },
        index=index,
    )
    treatment_values = [
        {"H", "R"} if group % 3 else {"H", "Z"} for group in np.repeat(np.arange(n_patients), 2)
    ]
    return RawValidationData(
        context=context,
        treatments=pd.Series(treatment_values, index=index),
        outcome=pd.Series(outcome, index=index),
        patient_groups=pd.Series(groups, index=index),
        numeric_columns=("age", "marker"),
        categorical_columns=("country",),
    )


def _test_config() -> ProfileStabilityConfig:
    return ProfileStabilityConfig(
        seeds=(2, 3, 5, 7, 11, 13),
        candidate=PatchCandidate(
            name="locked_test",
            n_archetypes=3,
            loading_l2=1e-3,
            deviation_l2=1e-2,
            residual_scale=0.5,
            epochs=3,
            learning_rate=0.01,
        ),
        small_cell_min=2,
        bootstrap_resamples=2,
        bootstrap_seed=29,
    )


@pytest.fixture(scope="module")
def profile_result() -> ps.ProfileStabilityResult:
    return run_profile_stability(_synthetic_data(), _test_config())


def test_hungarian_alignment_restores_permuted_profiles_and_loadings() -> None:
    reference = np.asarray(
        [
            [2.0, -1.0, 0.0, 0.5],
            [-1.0, 2.0, 0.5, 0.0],
            [-1.0, -1.0, -0.5, -0.5],
        ],
        dtype=np.float64,
    )
    base = np.asarray([0.2, -0.1, 0.3, 0.4])
    loadings = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    permutation = np.asarray([2, 0, 1])
    current = ps._SeedProfile(
        seed=9,
        total_coefficients=(reference + base)[permutation],
        deviation_coefficients=reference[permutation],
        soft_loadings=loadings[:, permutation],
        residual_coefficient_surface=loadings @ reference,
        observed_regimen_residual_logit_contribution=(loadings @ reference).sum(axis=1),
    )

    aligned = align_profile_to_reference(reference, current)

    np.testing.assert_allclose(aligned.deviation_coefficients, reference)
    np.testing.assert_allclose(aligned.total_coefficients, reference + base)
    np.testing.assert_allclose(aligned.soft_loadings, loadings)
    np.testing.assert_allclose(
        aligned.residual_coefficient_surface,
        loadings @ reference,
    )


def test_prediction_invariant_metrics_ignore_profile_permutation() -> None:
    deviations = np.asarray(
        [[2.0, -1.0], [-1.0, 2.0], [-1.0, -1.0]],
        dtype=np.float64,
    )
    loadings = np.asarray(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]],
        dtype=np.float64,
    )
    surface = loadings @ deviations
    observed = np.sum(surface * np.asarray([[1, 1], [1, 0], [0, 1]]), axis=1)
    permutation = np.asarray([2, 0, 1])
    first = ps._SeedProfile(
        seed=1,
        total_coefficients=deviations,
        deviation_coefficients=deviations,
        soft_loadings=loadings,
        residual_coefficient_surface=surface,
        observed_regimen_residual_logit_contribution=observed,
    )
    second = ps._SeedProfile(
        seed=2,
        total_coefficients=deviations[permutation],
        deviation_coefficients=deviations[permutation],
        soft_loadings=loadings[:, permutation],
        residual_coefficient_surface=loadings[:, permutation] @ deviations[permutation],
        observed_regimen_residual_logit_contribution=observed,
    )

    metrics = ps._prediction_invariant_metrics(first, second)

    assert metrics["residual_coefficient_surface_cosine"] == pytest.approx(1.0)
    assert metrics["residual_coefficient_surface_pearson"] == pytest.approx(1.0)
    assert metrics["residual_coefficient_surface_normalized_rmse"] == pytest.approx(0.0)
    assert metrics["observed_regimen_residual_logit_contribution_cosine"] == pytest.approx(1.0)
    assert metrics["observed_regimen_residual_logit_contribution_pearson"] == pytest.approx(1.0)
    assert metrics["observed_regimen_residual_logit_contribution_normalized_rmse"] == pytest.approx(
        0.0
    )


def test_profile_analysis_uses_six_seeds_and_keeps_soft_loadings_primary(
    profile_result: ps.ProfileStabilityResult,
) -> None:
    assert profile_result.aligned_soft_loadings.shape == (6, 72, 3)
    assert profile_result.consensus_soft_loadings.shape == (72, 3)
    np.testing.assert_allclose(
        profile_result.consensus_soft_loadings.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    report = profile_result.aggregate_report
    assert report["locked_model"]["profile_result_tuning"] is False
    assert report["locked_model"]["alignment_basis"].startswith("residual regimen")
    assert report["seed_analysis"]["n_seed_pairs"] == 15
    assert report["patient_cluster_bootstrap"]["resamples"] == 2
    assert len(report["profile_summary"]) == 3
    assert "hard_label_ari_secondary" in report["seed_analysis"]
    invariant = report["seed_analysis"]["prediction_invariant_stability_primary"]
    assert "pairwise_metrics" in invariant
    assert "bootstrap_prediction_invariant_stability" in profile_result.aggregate_tables


def test_profile_public_tables_are_aggregate_and_small_cells_are_suppressed(
    profile_result: ps.ProfileStabilityResult,
) -> None:
    for name, table in profile_result.aggregate_tables.items():
        assert "aggregation_unit" in table
        assert not {"condition_id", "patient_id", "outcome"}.intersection(table.columns)
        ps._assert_public_table_safe(table, name=name)
    support = profile_result.aggregate_tables["regimen_support"]
    assert support["small_cell_suppressed"].any()
    suppressed = support["small_cell_suppressed"]
    assert support.loc[suppressed, "soft_weighted_receipt_fraction"].isna().all()


def test_profile_outputs_keep_rows_private_and_public_files_aggregate(
    profile_result: ps.ProfileStabilityResult,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output_root = prepare_dua_output_root(tmp_path / "profile-output", project)

    paths = write_profile_outputs(profile_result, output_root)

    public_text = paths["aggregate_report"].read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert public["governance"]["public_outputs_contain_identifiers"] is False
    assert "condition_id" not in public_text
    assert "patient_id" not in public_text
    assert paths["aggregate_figure_pdf"].stat().st_size > 0
    assert paths["aggregate_figure_png"].stat().st_size > 0
    private = np.load(paths["private_loadings"])
    assert {
        "condition_id",
        "patient_id",
        "outcome",
        "aligned_soft_loadings",
        "residual_coefficient_surfaces",
        "observed_regimen_residual_logit_contributions",
    } <= set(private.files)
    assert paths["private_loadings"].stat().st_mode & 0o777 == 0o600


def test_profile_config_rejects_fewer_than_six_seeds() -> None:
    with pytest.raises(ValueError, match="at least six"):
        ProfileStabilityConfig(seeds=(1, 2, 3, 4, 5))


def test_public_table_guard_rejects_row_level_columns() -> None:
    unsafe = pd.DataFrame(
        {
            "aggregation_unit": ["row"],
            "condition_id": ["secret"],
        }
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        ps._assert_public_table_safe(unsafe, name="unsafe")
