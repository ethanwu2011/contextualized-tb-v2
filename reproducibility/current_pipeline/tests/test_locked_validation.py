from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ctb.models.locked_validation as lv
from ctb.models.fold_preprocessing import FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    PatchCandidate,
    RawValidationData,
    make_grouped_splits,
    patient_clustered_paired_metrics,
    prepare_dua_output_root,
    resolve_patient_groups,
    run_locked_validation,
    write_validation_outputs,
)


def _synthetic_data(n_patients: int = 30) -> RawValidationData:
    rng = np.random.default_rng(17)
    n_rows = 2 * n_patients
    index = pd.Index([f"c{row:03d}" for row in range(n_rows)], name="condition_id")
    patient = np.repeat([f"p{group:03d}" for group in range(n_patients)], 2)
    patient_outcome = np.asarray([group % 2 for group in range(n_patients)], dtype=int)
    outcome = np.repeat(patient_outcome, 2)
    context = pd.DataFrame(
        {
            "age": 30 + rng.normal(size=n_rows) + 3 * outcome,
            "marker": rng.normal(size=n_rows) + 0.7 * outcome,
            "constant_raw": 1.0,
            "country": np.where(np.arange(n_rows) % 3 == 0, "A", "B"),
        },
        index=index,
    )
    context.loc[index[::11], "marker"] = np.nan
    treatment_values = [
        {"H", "R"} if group % 3 else {"H", "Z"} for group in np.repeat(np.arange(n_patients), 2)
    ]
    return RawValidationData(
        context=context,
        treatments=pd.Series(treatment_values, index=index, name="initial_regimen"),
        outcome=pd.Series(outcome, index=index, name="died_failure"),
        patient_groups=pd.Series(patient, index=index, name="patient_group"),
        numeric_columns=("age", "marker", "constant_raw"),
        categorical_columns=("country",),
    )


def _null_config() -> LockedValidationConfig:
    return LockedValidationConfig(
        outer_splits=3,
        inner_splits=2,
        lr_c_values=(0.1, 1.0),
        patch_candidates=(PatchCandidate(name="null"),),
        bootstrap_resamples=20,
        seed=23,
    )


def test_grouped_splits_are_deterministic_disjoint_and_complete() -> None:
    data = _synthetic_data()
    outcome = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.to_numpy(dtype=str)

    first = make_grouped_splits(outcome, groups, n_splits=3, seed=5, label="test")
    second = make_grouped_splits(outcome, groups, n_splits=3, seed=5, label="test")

    assert len(first) == 3
    coverage = np.zeros(len(outcome), dtype=int)
    for (train, test), (train_again, test_again) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        assert set(groups[train]).isdisjoint(set(groups[test]))
        coverage[test] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(outcome), dtype=int))


def test_patient_group_resolution_allows_repeated_patients_but_rejects_ambiguity() -> None:
    patient_cases = pd.DataFrame(
        {
            "condition_id": ["c1", "c2", "c3"],
            "patient_id": ["p1", "p1", "p2"],
        }
    )
    conditions = pd.Index(["c1", "c2", "c3"], name="condition_id")

    resolved = resolve_patient_groups(patient_cases, conditions)

    assert resolved.tolist() == ["p1", "p1", "p2"]
    ambiguous = pd.concat(
        [
            patient_cases,
            pd.DataFrame({"condition_id": ["c1"], "patient_id": ["different"]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="ambiguous_links=1"):
        resolve_patient_groups(ambiguous, conditions)


def test_validation_preprocessing_never_sees_a_complete_or_split_patient_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_data()
    observed_fit_indices: list[set[str]] = []
    original_fit = FoldPreprocessor.fit

    def recording_fit(
        self: FoldPreprocessor,
        context: pd.DataFrame,
        treatments: object,
    ) -> FoldPreprocessor:
        observed_fit_indices.append(set(context.index.astype(str)))
        return original_fit(self, context, treatments)  # type: ignore[arg-type]

    monkeypatch.setattr(FoldPreprocessor, "fit", recording_fit)
    result = run_locked_validation(data, _null_config())

    assert observed_fit_indices
    assert max(map(len, observed_fit_indices)) < len(data.context)
    patient_rows = {
        patient: set(data.context.index[data.patient_groups.eq(patient)].astype(str))
        for patient in data.patient_groups.unique()
    }
    for fitted in observed_fit_indices:
        for rows in patient_rows.values():
            assert not fitted.intersection(rows) or rows <= fitted
    np.testing.assert_allclose(result.additive_probability, result.patch_probability, atol=0.0)
    assert set(result.outer_fold) == {0, 1, 2}
    for patient in data.patient_groups.unique():
        patient_rows_mask = data.patient_groups.eq(patient).to_numpy()
        assert len(set(result.outer_fold[patient_rows_mask])) == 1


def test_nonnull_candidate_is_scored_inside_grouped_inner_validation() -> None:
    data = _synthetic_data(n_patients=30)
    nonnull = PatchCandidate(
        name="test_residual",
        n_archetypes=2,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.5,
        epochs=3,
        learning_rate=0.01,
    )
    config = LockedValidationConfig(
        outer_splits=3,
        inner_splits=2,
        lr_c_values=(0.5,),
        patch_candidates=(PatchCandidate(name="null"), nonnull),
        bootstrap_resamples=10,
        seed=11,
    )

    result = run_locked_validation(data, config)

    assert len(result.aggregate_report["outer_folds"]) == 3
    for fold in result.aggregate_report["outer_folds"]:
        assert set(fold["inner_patch_mean_log_loss"]) == {"null", "test_residual"}
        assert np.isfinite(fold["inner_patch_mean_log_loss"]["test_residual"])


def test_residual_scale_is_applied_only_after_unit_scale_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_data(n_patients=12)
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=0.01,
    )
    matrices = preprocessor.fit_transform(data.context, data.treatments)
    y = np.asarray(data.outcome, dtype=np.int64)
    baseline = lv._fit_additive(
        matrices,
        y,
        c_value=1.0,
        max_iter=500,
        seed=4,
    )
    scale_seen_during_training: list[float] = []

    def record_unit_scale(
        model: lv.ResidualArchetypeLogistic,
        *args: object,
        **kwargs: object,
    ) -> lv.ResidualArchetypeLogistic:
        scale_seen_during_training.append(float(model.residual_scale.item()))
        return model

    monkeypatch.setattr(lv, "train_residual", record_unit_scale)
    candidate = PatchCandidate(
        name="post_fit_shrinkage",
        n_archetypes=2,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.25,
        epochs=1,
        learning_rate=0.01,
    )

    model = lv._fit_residual(baseline, matrices, y, candidate, seed=4)

    assert scale_seen_during_training == [1.0]
    assert float(model.residual_scale.item()) == pytest.approx(0.25)


def test_patient_clustered_paired_intervals_are_reproducible() -> None:
    data = _synthetic_data(n_patients=24)
    y = np.asarray(data.outcome, dtype=np.int64)
    additive = np.clip(0.2 + 0.55 * y + np.linspace(-0.05, 0.05, len(y)), 0.01, 0.99)
    patch = np.clip(additive + np.where(y == 1, 0.01, -0.01), 0.01, 0.99)
    groups = data.patient_groups.to_numpy(dtype=str)

    first = patient_clustered_paired_metrics(
        y,
        additive,
        patch,
        groups,
        n_resamples=20,
        seed=3,
    )
    second = patient_clustered_paired_metrics(
        y,
        additive,
        patch,
        groups,
        n_resamples=20,
        seed=3,
    )

    assert first == second
    assert first["bootstrap"]["unit"] == "patient"
    assert first["bootstrap"]["paired"] is True
    assert set(first["paired_difference_patch_minus_additive"]) == set(lv.METRIC_NAMES)


def test_outputs_separate_private_rows_from_release_safe_aggregates(tmp_path: Path) -> None:
    result = run_locked_validation(_synthetic_data(), _null_config())
    project = tmp_path / "project"
    project.mkdir()
    output_root = prepare_dua_output_root(tmp_path / "secure-output", project)

    paths = write_validation_outputs(result, output_root)

    public_text = paths["aggregate_json"].read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert public["governance"]["public_report_contains_identifiers"] is False
    assert "condition_id" not in public_text
    assert "patient_id" not in public_text
    assert "additive_probability" not in public_text
    assert paths["aggregate_pdf"].stat().st_size > 0
    assert paths["aggregate_png"].stat().st_size > 0
    private = np.load(paths["private_predictions"])
    assert set(private.files) == {
        "condition_id",
        "patient_id",
        "outcome",
        "outer_fold",
        "additive_probability",
        "patch_probability",
    }
    assert paths["private_readme"].read_text(encoding="utf-8").startswith("DUA-PRIVATE")
    assert paths["private_predictions"].stat().st_mode & 0o777 == 0o600


def test_secure_output_root_rejects_repository_local_and_nonempty_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError, match="outside"):
        prepare_dua_output_root(project / "results", project)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="absent or empty"):
        prepare_dua_output_root(occupied, project)


def test_public_report_guard_rejects_row_level_fields() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        lv._assert_release_safe_report({"condition_id": ["secret"]})
