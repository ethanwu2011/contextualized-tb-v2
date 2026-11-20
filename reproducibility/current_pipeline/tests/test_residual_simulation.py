from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from ctb.models.residual_archetype import ResidualArchetypeLogistic
from ctb.models.residual_simulation import (
    MODEL_ORDER,
    SCENARIOS,
    SimulationBenchmarkConfig,
    fit_residual_from_additive,
    run_simulation_benchmark,
    select_residual_scale,
    simulate_batch,
    true_regimen_coefficient_surface,
    write_benchmark_artifacts,
)


def test_simulation_is_deterministic_and_surfaces_are_prespecified() -> None:
    first = simulate_batch("low_rank", 160, seed=17)
    second = simulate_batch("low_rank", 160, seed=17)

    np.testing.assert_array_equal(first.context, second.context)
    np.testing.assert_array_equal(first.regimen, second.regimen)
    np.testing.assert_array_equal(first.outcome, second.outcome)
    np.testing.assert_array_equal(
        first.true_regimen_coefficients,
        second.true_regimen_coefficients,
    )

    null_surface = true_regimen_coefficient_surface("additive_null", first.context)
    np.testing.assert_allclose(
        null_surface,
        np.broadcast_to(null_surface[0], null_surface.shape),
        rtol=0,
        atol=0,
    )
    assert np.std(first.true_regimen_coefficients[:, 0]) > 0.1


def test_scale_grid_selects_exact_additive_null_on_a_tie() -> None:
    batch = simulate_batch("additive_null", 180, seed=23)
    additive = LogisticRegression(C=1.0, max_iter=2000).fit(
        np.hstack([batch.context, batch.regimen]),
        batch.outcome,
    )
    model = ResidualArchetypeLogistic(5, 4, n_archetypes=3, seed=4)
    model.initialize_from_logistic(
        additive.coef_[0, :5],
        additive.coef_[0, 5:],
        float(additive.intercept_[0]),
        freeze=True,
    )
    with torch.no_grad():
        model.deviation_coordinates.zero_()

    selected, _, losses = select_residual_scale(
        model,
        batch.context,
        batch.regimen,
        batch.outcome,
        (0.0, 0.5, 1.0),
        tie_tolerance=0.0,
    )

    assert selected == 0.0
    assert len(set(losses.values())) == 1


def test_fitted_scale_zero_matches_additive_baseline() -> None:
    config = replace(
        SimulationBenchmarkConfig(),
        sample_sizes=(100,),
        seeds=(5,),
        n_validation=80,
        n_test=120,
        residual_epochs=5,
    )
    batch = simulate_batch("low_rank", 100, seed=13)
    additive = LogisticRegression(C=1.0, max_iter=2000).fit(
        np.hstack([batch.context, batch.regimen]),
        batch.outcome,
    )
    patch = fit_residual_from_additive(
        additive,
        batch.context,
        batch.regimen,
        batch.outcome,
        config,
        seed=8,
    )
    patch.set_residual_scale(0.0)
    expected = additive.predict_proba(np.hstack([batch.context, batch.regimen]))[:, 1]

    np.testing.assert_allclose(
        patch.predict_proba(batch.context, batch.regimen),
        expected,
        rtol=1e-6,
        atol=2e-7,
    )


def test_small_benchmark_has_complete_aggregate_schema() -> None:
    config = replace(
        SimulationBenchmarkConfig(),
        sample_sizes=(100,),
        seeds=(7,),
        n_validation=80,
        n_test=120,
        logistic_c_grid=(0.5,),
        residual_scale_grid=(0.0, 1.0),
        residual_epochs=5,
    )

    results, diagnostics = run_simulation_benchmark(config)

    assert len(results) == len(SCENARIOS) * len(MODEL_ORDER)
    assert set(results["scenario"]) == set(SCENARIOS)
    assert set(results["model"]) == set(MODEL_ORDER)
    assert len(diagnostics) == len(SCENARIOS)
    for column in ("test_log_loss", "test_auroc", "coefficient_rmse"):
        assert np.isfinite(results[column]).all()
    patch = results.loc[results["model"].eq("residual_patch")]
    assert patch["selected_residual_scale"].isin(config.residual_scale_grid).all()
    assert results["test_auroc"].between(0, 1).all()


def test_artifact_writer_emits_csv_json_and_color_safe_figure(tmp_path: Path) -> None:
    config = replace(
        SimulationBenchmarkConfig(),
        sample_sizes=(100,),
        seeds=(7,),
        n_validation=80,
        n_test=120,
        logistic_c_grid=(0.5,),
        residual_scale_grid=(0.0, 1.0),
        residual_epochs=3,
    )
    results, diagnostics = run_simulation_benchmark(config)

    paths = write_benchmark_artifacts(results, diagnostics, config, tmp_path)

    for path in (
        paths.replicate_csv,
        paths.summary_json,
        paths.figure_png,
        paths.figure_pdf,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0
    reloaded = pd.read_csv(paths.replicate_csv)
    assert len(reloaded) == len(results)
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["artifact_schema_version"] == 1
    assert summary["replicate_rows"] == len(results)
    assert "causal treatment effects" in summary["interpretation"]
