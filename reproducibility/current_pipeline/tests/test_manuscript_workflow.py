"""Tests for the tracked, bounded manuscript workflow declaration and dry run."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scripts.run_manuscript as manuscript_runner
from scripts.run_manuscript import parse_args

from ctb.data.model_matrices import (
    ARTIFACT_SCHEMA_VERSION,
    COHORT_DEFINITION_IDS,
    DIED_FAILURE,
    ENDPOINT_DEFINITION_IDS,
)
from ctb.data.outcomes import derive_died_failure, derive_patient_case_died_failure

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config" / "manuscript_workflow.json"
endpoint_builder = import_module("scripts.05b_build_conversion_matrices")
synthetic_generator = import_module("scripts.00b_make_synthetic_data")
build_culture_eligibility_report = endpoint_builder.build_culture_eligibility_report
build_died_failure_assessability_report = endpoint_builder.build_died_failure_assessability_report
ALLOWED_SCRIPT_ENTRYPOINTS = {
    "00_download_data.py",
    "00b_make_synthetic_data.py",
    "05b_build_conversion_matrices.py",
    "run_manuscript.py",
    # Reviewed shared-primary aggregate builder and pristine-snapshot verifier.
    "build_shared_primary_aggregates.py",
    "verify_snapshot.py",
    # Governed DUA-local analysis drivers. These emit only disclosure-safe aggregate reports to
    # the public release path; any row-level working output is required to remain outside it.
    "decompress_sinan_dbc.py",
    "build_aggregate_checksums.py",
    "build_public_file_manifest.py",
    "prepare_audit_output_root.py",
    "run_aggregate_sensitivities.py",
    "run_architecture_composition_ablations.py",
    "run_brazil_source_rebuild_audit.py",
    "run_episode_endpoint_audit.py",
    "run_expanded_simulation.py",
    "run_fixed_k_comparison.py",
    "run_fixed_k_seed_stability.py",
    "run_full_pipeline_profile_stability.py",
    "run_internal_decision_curve.py",
    "run_internal_transport.py",
    "run_k5_oof_seed_robustness.py",
    "run_locked_residual_analysis.py",
    "run_nonlinear_baseline.py",
    "run_shared_k_sensitivity.py",
    "run_shared_profile_full_pipeline_stability.py",
    "run_shared_profile_reference_composition.py",
}
ALLOWED_FIGURE_ENTRYPOINTS = {
    "_swatch.py",
    "release/render_aggregate_figures.py",
    "release/render_shared_primary_submission.py",
    "release/ew_aggregate_figure_templates.py",
    "supp/figS1_reproducibility.py",
    "supp/figS2_kselection.py",
}


def _current_source_env() -> dict[str, str]:
    env = os.environ.copy()
    current_source = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        current_source if not existing else os.pathsep.join((current_source, existing))
    )
    return env


def test_panel_manifest_maps_s1_s2_once_to_died_failure() -> None:
    spec = json.loads(SPEC.read_text())
    panels = spec["panels"]

    assert not spec["submission_complete"]
    assert spec["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert spec["cohort_report_schema_version"] == 2
    assert [panel["panel"] for panel in panels] == ["S1", "S2"]
    assert all(panel["endpoint"] == "died_failure" for panel in panels)
    assert all(panel["release_status"].startswith("diagnostic_scaffold") for panel in panels)
    assert len({panel["stage"] for panel in panels}) == len(panels)
    assert all((ROOT / panel["producer"]).is_file() for panel in panels)
    assert spec["known_release_gaps"]


def test_synthetic_tables_match_final_outcome_and_patient_group_contract() -> None:
    rng = np.random.default_rng(42)
    patient_cases = synthetic_generator.make_patient_cases(50, rng)
    regimens = synthetic_generator.make_regimens(patient_cases, rng)

    assert patient_cases["condition_id"].is_unique
    assert patient_cases["patient_id"].notna().all()
    assert patient_cases["patient_id"].astype("string").str.strip().ne("").all()
    assert patient_cases.groupby("condition_id")["patient_id"].nunique(dropna=True).eq(1).all()
    assert "outcome" in patient_cases
    assert pd.api.types.is_numeric_dtype(regimens["period_start"])
    assert pd.api.types.is_numeric_dtype(regimens["activities_period_start"])

    direct = derive_patient_case_died_failure(patient_cases)["died_failure"]
    history = derive_died_failure(regimens)["died_failure"].reindex(direct.index)
    pd.testing.assert_series_equal(direct, history, check_names=False)


def test_culture_eligibility_reasons_are_mutually_exclusive_and_complete() -> None:
    conversion = pd.DataFrame(
        {
            "first_pos_day": [float("nan"), 2.0, 0.0],
        },
        index=[5, 6, 7],
    )
    private_sets = {
        "has_any_culture_row": frozenset({2, 3, 4, 5, 6, 7}),
        "has_interpretable_culture_result": frozenset({3, 4, 5, 6, 7}),
        "has_valid_interpretable_dated_culture": frozenset({4, 5, 6, 7}),
        "quarantined_same_day_conflict": frozenset({4}),
    }

    report = build_culture_eligibility_report(
        consort_ids=[1, 2, 3, 4, 5, 6, 7],
        conversion=conversion,
        private_condition_sets=private_sets,
    )

    assert report["consort_conditions"] == 7
    assert report["excluded_before_baseline_eligibility"] == 6
    for reason in (
        "no_culture_row",
        "no_interpretable_culture_result",
        "no_valid_interpretable_culture_timing",
        "quarantined_same_day_conflict",
        "never_positive",
        "first_positive_after_day_zero",
        "baseline_eligible_first_positive_at_or_before_day_zero",
    ):
        assert report[reason] == 1


def test_died_failure_assessability_reasons_are_mutually_exclusive() -> None:
    patient_cases = pd.DataFrame(
        {
            "condition_id": list(range(1, 9)),
            "outcome": [
                "Died",
                "Cured",
                None,
                "Lost to follow up",
                "Still on treatment",
                "Palliative Care",
                "Unknown",
                "Not evaluated",
            ],
        }
    )

    report = build_died_failure_assessability_report(
        patient_cases=patient_cases,
        source_ids=list(range(1, 9)),
    )

    assert report["source_conditions"] == 8
    assert report["assessed_conditions"] == 2
    assert report["unassessable_conditions"] == 6
    for reason in (
        "event_recorded_died_failure",
        "observed_nonevent_recorded_cured_completed",
        "missing_or_blank_outcome",
        "lost_to_follow_up",
        "still_on_treatment",
        "palliative_care",
        "unknown",
        "other_unclassified_outcome",
    ):
        assert report[reason] == 1


def test_dua_mode_automatically_requires_clean_checkout() -> None:
    assert parse_args(["--mode", "dua"]).require_clean


def test_nonreduced_child_environment_removes_inherited_reduced_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CTB_REDUCED_RUN", "1")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/source")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/home")

    env = manuscript_runner._stage_environment(
        output_root=tmp_path / "out",
        raw_dir=tmp_path / "raw",
        mode="dua",
        reduced=False,
        synthetic_stage=False,
        seed=42,
    )

    assert "CTB_REDUCED_RUN" not in env
    assert "PYTHONHOME" not in env
    assert env["PYTHONPATH"] == str(ROOT / "src")
    assert env["PYTHONHASHSEED"] == "42"
    assert env["CTB_RAW_DIR"] == str(tmp_path / "raw")
    assert env["CTB_GOVERNED_RUN"] == "1"


def test_synthetic_mode_rejects_overlap_with_governed_raw_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "governed"
    monkeypatch.setenv("CTB_RAW_DIR", str(protected))
    args = parse_args(
        [
            "--mode",
            "synthetic",
            "--output-root",
            str(protected),
            "--synthetic-dir",
            str(protected / "synthetic"),
        ]
    )

    with pytest.raises(SystemExit, match="overlaps"):
        manuscript_runner.run_workflow(args)


def test_dua_mode_rejects_missing_git_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_value(*args: str) -> str | None:
        return "tracked" if args and args[0] == "ls-files" else None

    monkeypatch.setattr(manuscript_runner, "_git_value", fake_git_value)
    monkeypatch.setenv("CTB_RAW_DIR", str(tmp_path / "raw"))
    args = parse_args(["--mode", "dua", "--output-root", str(tmp_path / "outputs")])

    with pytest.raises(SystemExit, match="valid Git checkout"):
        manuscript_runner.run_workflow(args)


def test_dua_mode_rejects_external_workflow_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_spec = tmp_path / "workflow.json"
    external_spec.write_text(SPEC.read_text())
    monkeypatch.setenv("CTB_RAW_DIR", str(tmp_path / "raw"))
    args = parse_args(
        [
            "--mode",
            "dua",
            "--spec",
            str(external_spec),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    with pytest.raises(SystemExit, match="inside the repository"):
        manuscript_runner.run_workflow(args)


def test_dua_mode_rejects_preexisting_declared_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_value(*args: str) -> str | None:
        if args and args[0] == "status":
            return ""
        if args == ("rev-parse", "HEAD"):
            return "fixed-commit"
        return "tracked"

    output_root = tmp_path / "outputs"
    stale = output_root / "data" / "processed" / "died_failure_model_matrices.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    monkeypatch.setattr(manuscript_runner, "_git_value", fake_git_value)
    monkeypatch.setenv("CTB_RAW_DIR", str(tmp_path / "raw"))
    args = parse_args(["--mode", "dua", "--output-root", str(output_root)])

    with pytest.raises(SystemExit, match="fresh empty output root"):
        manuscript_runner.run_workflow(args)


def test_clean_git_completion_check_rejects_head_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_value(*args: str) -> str | None:
        return "" if args and args[0] == "status" else "new-commit"

    monkeypatch.setattr(manuscript_runner, "_git_value", fake_git_value)

    with pytest.raises(RuntimeError, match="HEAD changed"):
        manuscript_runner._validate_clean_git_state(expected_commit="old-commit")


def test_completion_rehash_rejects_mutated_input(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("before")
    observed = {source: manuscript_runner.file_sha256(source)}
    source.write_text("after")

    with pytest.raises(RuntimeError, match="input changed"):
        manuscript_runner._verify_unchanged_files(observed, label="input")


def test_run_manifest_is_written_with_restricted_permissions(tmp_path: Path) -> None:
    path = tmp_path / "private" / "manifest.json"
    manuscript_runner._write_run_manifest(path, {"status": "test"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_endpoint_builder_direct_execution_is_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/05b_build_conversion_matrices.py"],
        cwd=ROOT,
        env=_current_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr


def test_workflow_requires_current_endpoint_builder(tmp_path: Path) -> None:
    spec = tmp_path / "workflow.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "cohort_report_schema_version": 2,
                "workflow_id": "missing_builder",
                "scope": "test",
                "submission_complete": False,
                "endpoint": "died_failure",
                "seed": 42,
                "known_release_gaps": ["test workflow intentionally omits the builder"],
                "panels": [],
                "stages": [
                    {
                        "id": "synthetic_data",
                        "modes": ["synthetic"],
                        "command": [sys.executable, "-c", "pass"],
                        "inputs": [],
                        "outputs": [],
                    }
                ],
            }
        )
    )
    args = parse_args(
        [
            "--mode",
            "synthetic",
            "--spec",
            str(spec),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    with pytest.raises(ValueError, match="exactly one endpoint-builder"):
        manuscript_runner.run_workflow(args)


def test_manifest_cannot_alias_declared_builder_output(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    manifest_template = "{output_root}/results/provenance/manuscript_run.json"
    spec = tmp_path / "alias-workflow.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "cohort_report_schema_version": 2,
                "workflow_id": "manifest_alias",
                "scope": "test",
                "submission_complete": False,
                "endpoint": "died_failure",
                "seed": 42,
                "known_release_gaps": ["test"],
                "panels": [],
                "stages": [
                    {
                        "id": "synthetic_data",
                        "modes": ["synthetic"],
                        "command": [sys.executable, "-c", "pass"],
                        "inputs": [],
                        "outputs": [],
                    },
                    {
                        "id": "build_endpoint_matrices",
                        "modes": ["synthetic"],
                        "command": [sys.executable, "-c", "pass"],
                        "inputs": [],
                        "outputs": [
                            "{output_root}/data/processed/died_failure_model_matrices.npz",
                            "{output_root}/data/processed/died_failure_cohort_report.json",
                            manifest_template,
                        ],
                    },
                ],
            }
        )
    )
    args = parse_args(
        [
            "--mode",
            "synthetic",
            "--spec",
            str(spec),
            "--output-root",
            str(output_root),
        ]
    )

    with pytest.raises(ValueError, match="must not alias"):
        manuscript_runner.run_workflow(args)


def test_bounded_runner_rejects_submission_complete_spec(tmp_path: Path) -> None:
    payload = json.loads(SPEC.read_text())
    payload["submission_complete"] = True
    spec = tmp_path / "unsafe-complete.json"
    spec.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="cannot certify submission_complete=true"):
        manuscript_runner.load_workflow_spec(spec)


def test_typed_output_validator_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.touch()

    with pytest.raises(ValueError, match="empty"):
        manuscript_runner._validate_typed_output(empty)


def test_metadata_validator_requires_endpoint_order_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "meta.parquet"
    frame = pd.DataFrame(
        {"died_failure": [1, 0]},
        index=pd.Index(["condition-a", "condition-b"], name="condition_id"),
    )
    frame.to_parquet(path)

    manuscript_runner._validate_metadata_alignment(
        path,
        expected_cond_ids=["condition-a", "condition-b"],
        required_columns={"died_failure"},
        expected_binary_columns={"died_failure": [1, 0]},
    )
    with pytest.raises(ValueError, match="condition order"):
        manuscript_runner._validate_metadata_alignment(
            path,
            expected_cond_ids=["condition-b", "condition-a"],
            required_columns={"died_failure"},
            expected_binary_columns={"died_failure": [0, 1]},
        )
    frame.loc["condition-a", "died_failure"] = 0
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="disagrees with endpoint artifact"):
        manuscript_runner._validate_metadata_alignment(
            path,
            expected_cond_ids=["condition-a", "condition-b"],
            required_columns={"died_failure"},
            expected_binary_columns={"died_failure": [1, 0]},
        )
    for unsafe_values in ([1.5, 0.5], ["1", "0"]):
        frame["died_failure"] = unsafe_values
        frame.to_parquet(path)
        with pytest.raises(ValueError, match="disagrees with endpoint artifact"):
            manuscript_runner._validate_metadata_alignment(
                path,
                expected_cond_ids=["condition-a", "condition-b"],
                required_columns={"died_failure"},
                expected_binary_columns={"died_failure": [1, 0]},
            )


def test_synthetic_dry_run_emits_machine_readable_plan(tmp_path: Path) -> None:
    output_root = tmp_path / "dry-run"
    output = output_root / "results" / "provenance" / "run_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_manuscript.py",
            "--mode",
            "synthetic",
            "--reduced",
            "--dry-run",
            "--output-root",
            str(output_root),
            "--output-manifest",
            str(output),
        ],
        cwd=ROOT,
        env=_current_source_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    run = json.loads(output.read_text())
    assert run["status"] == "planned"
    assert run["mode"] == "synthetic"
    assert run["reduced"]
    assert not run["submission_complete"]
    assert [stage["id"] for stage in run["stages"]] == [
        "synthetic_data",
        "build_endpoint_matrices",
        "figure_s1",
        "figure_s2",
    ]
    assert all(stage["status"] == "planned" for stage in run["stages"])


def test_minimum_synthetic_run_completes_with_guarded_provenance(tmp_path: Path) -> None:
    output_root = tmp_path / "minimum-smoke"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_manuscript.py",
            "--mode",
            "synthetic",
            "--reduced",
            "--synthetic-n",
            "50",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        env=_current_source_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    run = json.loads((output_root / "results" / "provenance" / "manuscript_run.json").read_text())
    endpoint = run["endpoint_artifact"]
    cohort_report_path = output_root / "data" / "processed" / "died_failure_cohort_report.json"
    cohort_report = json.loads(cohort_report_path.read_text())
    flow = cohort_report["flow_counts"]
    assert run["status"] == "completed"
    assert [stage["id"] for stage in run["stages"]] == [
        "synthetic_data",
        "build_endpoint_matrices",
        "figure_s1",
        "figure_s2",
    ]
    assert endpoint["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert endpoint["endpoint_definition_id"] == ENDPOINT_DEFINITION_IDS[DIED_FAILURE]
    assert endpoint["cohort_definition_id"] == COHORT_DEFINITION_IDS[DIED_FAILURE]
    assert endpoint["n_events"] >= 5
    assert endpoint["n_non_events"] >= 5
    assert len(endpoint["source_cohort_sha256"]) == 64
    assert len(endpoint["patient_sha256"]) == 64
    assert len(endpoint["provenance_sha256"]) == 64
    assert len(run["executed_source_tree"]["sha256"]) == 64
    assert run["executed_source_tree"]["n_files"] > 0
    assert len(run["executed_source_tree"]["files"]) == run["executed_source_tree"]["n_files"]
    assert len(run["runtime_environment"]["sha256"]) == 64
    assert run["runtime_environment"]["n_distributions"] > 0
    assert not run["runtime_environment"]["lock_match_verified"]
    assert cohort_report["report_schema_version"] == 2
    assert cohort_report["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert cohort_report["endpoint_contract"]["endpoint"] == DIED_FAILURE
    assert not cohort_report["contains_row_level_identifiers"]
    assert (
        flow["died_failure_assessed_conditions"] + flow["died_failure_unassessable_conditions"]
        == flow["primary_source_conditions"]
    )
    assert flow["primary_source_conditions"] == flow["consort_conditions"]
    assert flow["culture_conversion_sensitivity_conditions"] <= flow["consort_conditions"]
    assert (
        flow["died_failure_events"] + flow["died_failure_observed_non_events"]
        == flow["died_failure_assessed_conditions"]
    )
    assert all(len(value) == 64 for value in cohort_report["ordered_cohort_sha256"].values())
    assert run["cohort_report"]["validated_against_endpoint_artifact"]
    assert (
        run["cohort_report"]["died_failure_assessability"]["source_conditions"]
        == endpoint["source_n_rows"]
    )
    assert run["cohort_report"]["outcome_source_concordance"][
        "exact_assessability_and_label_concordance"
    ]
    assert run["cohort_report"]["patient_linkage"]["missing_links"] == 0
    assert run["cohort_report"]["patient_linkage"]["ambiguous_links"] == 0
    assert endpoint["cohort_report_sha256"] == run["cohort_report"]["report_file_sha256"]
    assert len(run["release_linkage"]["sha256"]) == 64
    assert run["release_linkage"]["endpoint_artifact_sha256"] == endpoint["artifact_file_sha256"]
    inventory = run["generated_file_inventory"]
    assert inventory["n_files"] >= sum(len(stage["outputs"]) for stage in run["stages"])
    assert len(inventory["sha256"]) == 64
    assert all(
        not item["declared_path"].endswith("manuscript_run.json")
        and not item["declared_path"].endswith(".manuscript_run.lock")
        for item in inventory["files"]
    )
    build_stage = next(stage for stage in run["stages"] if stage["id"] == "build_endpoint_matrices")
    assert any(
        output["declared_path"].endswith("died_failure_cohort_report.json")
        for output in build_stage["outputs"]
    )
    synthetic_command = run["stages"][0]["command"]
    assert synthetic_command[synthetic_command.index("--seed") + 1] == str(run["seed"])

    cohort_report["flow_counts"]["died_failure_events"] += 1
    cohort_report_path.write_text(json.dumps(cohort_report))
    with pytest.raises(ValueError, match="count mismatch"):
        manuscript_runner._cohort_report_record(
            cohort_report_path,
            endpoint_record=endpoint,
        )


def test_phase1_make_target_is_explicitly_blocked() -> None:
    result = subprocess.run(
        ["make", "phase1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "BLOCKED" in output
    assert "Do not fill" in output
    assert "scripts/01_build_dataset.py" not in output
    assert "scripts/02_train_baselines.py" not in output
    assert "scripts/03_generate_figures.py" not in output


@pytest.mark.parametrize("target", ["build", "train", "figures"])
def test_legacy_make_targets_are_explicitly_blocked(target: str) -> None:
    result = subprocess.run(
        ["make", target],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "BLOCKED" in result.stdout + result.stderr


def _main_function(path: Path) -> ast.FunctionDef | None:
    module = ast.parse(path.read_text(), filename=str(path))
    return next(
        (node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )


def _has_release_gate(main: ast.FunctionDef) -> bool:
    decorated = any(
        isinstance(decorator, ast.Name) and decorator.id == "unreleased_analysis_entrypoint"
        for decorator in main.decorator_list
    )
    executable = list(main.body)
    if (
        executable
        and isinstance(executable[0], ast.Expr)
        and isinstance(executable[0].value, ast.Constant)
        and isinstance(executable[0].value.value, str)
    ):
        executable = executable[1:]
    explicit = (
        len(executable) >= 2
        and isinstance(executable[0], ast.Expr)
        and isinstance(executable[0].value, ast.Call)
        and isinstance(executable[0].value.func, ast.Name)
        and executable[0].value.func.id == "print"
        and isinstance(executable[1], ast.Return)
        and isinstance(executable[1].value, ast.Constant)
        and executable[1].value.value == 2
    )
    return decorated or explicit


def test_all_python_analysis_entrypoints_are_allowlisted_or_blocked() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        main = _main_function(path)
        if main is None or path.name in ALLOWED_SCRIPT_ENTRYPOINTS:
            continue
        assert _has_release_gate(main), path.name

    figures_root = ROOT / "figures"
    for path in sorted(figures_root.rglob("*.py")):
        main = _main_function(path)
        relative = path.relative_to(figures_root).as_posix()
        if main is None or relative in ALLOWED_FIGURE_ENTRYPOINTS:
            continue
        assert _has_release_gate(main), relative


def test_representative_legacy_entrypoints_exit_blocked() -> None:
    candidates = [
        (ROOT / "scripts/01_build_dataset.py", [sys.executable, "scripts/01_build_dataset.py"]),
        (ROOT / "figures/fig01_schematic.py", [sys.executable, "-m", "figures.fig01_schematic"]),
        (ROOT / "figures/render_all.py", [sys.executable, "-m", "figures.render_all"]),
    ]
    for path, command in candidates:
        if not path.exists():
            continue
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_current_source_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, command
        assert "BLOCKED" in result.stdout + result.stderr, command


def test_legacy_slide_packaging_remains_hard_blocked() -> None:
    path = ROOT / "figures" / "build_pptx_deck.js"
    if not path.exists():
        return
    js = path.read_text()
    assert js.index("process.exit(2)") < js.index('require("pptxgenjs")')


def test_public_stager_is_fresh_target_only_and_scans_the_result(tmp_path: Path) -> None:
    destination = tmp_path / "public-release"
    staged = subprocess.run(
        ["bash", "scripts/sync_to_public.sh", str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert staged.returncode == 0, staged.stderr
    assert (destination / "PUBLIC_RELEASE_MANIFEST.json").is_file()
    assert (destination / "PUBLIC_FILE_MANIFEST.sha256").is_file()
    assert (destination / "manuscript/aggregate_results").is_dir()
    assert not (destination / "data").exists()
    assert not (destination / "scripts/run_all.sh").exists()
    assert not (destination / "figures/fig01_hero.py").exists()
    assert (destination / "scripts/run_episode_endpoint_audit.py").is_file()
    assert (destination / "figures/release/render_aggregate_figures.py").is_file()

    scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert scan.returncode == 0, scan.stderr
    assert "scan passed" in scan.stdout.lower()

    unlisted = destination / "manuscript/aggregate_results/UNLISTED.json"
    unlisted.write_text("{}\n")
    unlisted_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unlisted_scan.returncode != 0
    assert "file set" in unlisted_scan.stderr.lower()
    unlisted.unlink()

    nested_aggregate = destination / "manuscript/aggregate_results/nested"
    nested_aggregate.mkdir()
    (nested_aggregate / "rows.json").write_text("{}\n")
    nested_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert nested_scan.returncode != 0
    assert "nested aggregate" in nested_scan.stderr.lower()
    (nested_aggregate / "rows.json").unlink()
    nested_aggregate.rmdir()

    unlisted_document = destination / "docs/release_records.json"
    unlisted_document.write_text("{}\n")
    unlisted_document_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unlisted_document_scan.returncode != 0
    assert "unallowlisted json" in unlisted_document_scan.stderr.lower()
    unlisted_document.unlink()

    row_stream = destination / "docs/release_records.ndjson"
    row_stream.write_text("{}\n")
    row_stream_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert row_stream_scan.returncode != 0
    assert "patient-shaped" in row_stream_scan.stderr.lower()
    row_stream.unlink()

    arbitrary_source = destination / "docs/unreviewed_notes.md"
    arbitrary_source.write_text("not part of the reviewed release\n")
    arbitrary_source_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert arbitrary_source_scan.returncode != 0
    assert "exact public file manifest" in arbitrary_source_scan.stderr.lower()
    arbitrary_source.unlink()

    aggregate = destination / "manuscript/aggregate_results/episode_endpoint_audit.json"
    aggregate.write_text(aggregate.read_text() + "\n")
    tampered_scan = subprocess.run(
        ["bash", str(destination / "scripts/scan_public.sh"), str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_scan.returncode != 0
    assert "checksum" in tampered_scan.stderr.lower()

    repeated = subprocess.run(
        ["bash", "scripts/sync_to_public.sh", str(destination)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr

    inside_repository = subprocess.run(
        ["bash", "scripts/sync_to_public.sh", "public-release-inside-repository"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inside_repository.returncode == 2
    assert "outside" in inside_repository.stderr.lower()


def test_public_stager_refuses_an_explicit_allowlisted_symlink(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake-source"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/sync_to_public.sh", fake_scripts / "sync_to_public.sh")
    external = tmp_path / "external-private.txt"
    external.write_text("must not be dereferenced\n")
    (fake_root / "README.md").symlink_to(external)
    destination = tmp_path / "staged-release"
    result = subprocess.run(
        ["bash", str(fake_scripts / "sync_to_public.sh"), str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "allowlisted file is a symlink" in result.stderr
    assert not (destination / "README.md").exists()


def test_private_audit_output_guard_is_fresh_disjoint_and_mode_0700(tmp_path: Path) -> None:
    raw = tmp_path / "raw-input"
    raw.mkdir()
    destination = tmp_path / "private-audit"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_audit_output_root.py",
            "--output-root",
            str(destination),
            "--project-root",
            str(ROOT),
            "--input-root",
            str(raw),
            "--subdir",
            "aggregate",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "aggregate").stat().st_mode & 0o777 == 0o700

    repeated = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_audit_output_root.py",
            "--output-root",
            str(destination),
            "--project-root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "absent or empty" in repeated.stderr


def test_locked_residual_cli_rejects_repo_local_private_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    forbidden = ROOT / ".repo-local-private-output-must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_locked_residual_analysis.py",
            "--dua-acknowledge",
            "--raw-dir",
            str(raw),
            "--private-output-root",
            str(forbidden),
            "--aggregate-output",
            str(tmp_path / "analysis" / "aggregate" / "report.json"),
        ],
        cwd=ROOT,
        env=_current_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "outside the repository" in result.stderr
    assert not forbidden.exists()


def test_sinan_decompressor_refuses_overwrite_and_hardens_permissions() -> None:
    source = (ROOT / "scripts/decompress_sinan_dbc.py").read_text()
    assert "if output.exists()" in source
    assert "FileExistsError" in source
    assert "os.umask(0o077)" in source
    assert "os.chmod(output, 0o600)" in source


def test_internal_dca_cli_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    n_rows = 120
    groups = np.repeat(np.asarray([f"p{index}" for index in range(60)]), 2)
    outcome = np.tile(np.asarray([0, 1], dtype=np.int64), 60)
    additive = np.linspace(0.05, 0.95, n_rows)
    patch = np.clip(additive + 0.02 * (2 * outcome - 1), 0.001, 0.999)
    private = tmp_path / "private_predictions.npz"
    np.savez(
        private,
        outcome=outcome,
        patient_id=groups,
        additive_probability=additive,
        patch_probability=patch,
    )

    def run(output: Path, source: Path = private) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "scripts/run_internal_decision_curve.py",
                "--dua-acknowledge",
                "--private-predictions",
                str(source),
                "--output",
                str(output),
                "--bootstrap-resamples",
                "100",
            ],
            cwd=ROOT,
            env=_current_source_env(),
            check=False,
            capture_output=True,
            text=True,
        )

    first = tmp_path / "dca_first.json"
    second = tmp_path / "dca_second.json"
    first_run = run(first)
    second_run = run(second)
    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert payload["governance"]["contains_row_level_data"] is False
    assert payload["bootstrap"]["valid_resamples"] == 100
    assert "patient_id" not in first.read_text()
    assert "prediction" not in first.read_text()

    overwrite = run(first)
    assert overwrite.returncode != 0
    assert "already exists" in overwrite.stderr

    invalid_probability = patch.copy()
    invalid_probability[0] = 1.2
    invalid = tmp_path / "invalid_probability.npz"
    np.savez(
        invalid,
        outcome=outcome,
        patient_id=groups,
        additive_probability=additive,
        patch_probability=invalid_probability,
    )
    invalid_run = run(tmp_path / "invalid_probability.json", invalid)
    assert invalid_run.returncode != 0
    assert "closed interval" in invalid_run.stderr

    blank_groups = groups.copy()
    blank_groups[0] = ""
    missing = tmp_path / "missing_group.npz"
    np.savez(
        missing,
        outcome=outcome,
        patient_id=blank_groups,
        additive_probability=additive,
        patch_probability=patch,
    )
    missing_run = run(tmp_path / "missing_group.json", missing)
    assert missing_run.returncode != 0
    assert "missing or blank" in missing_run.stderr


def test_archived_shell_workflow_is_blocked() -> None:
    if not (ROOT / "scripts/run_all.sh").exists():
        return
    result = subprocess.run(
        ["bash", "scripts/run_all.sh"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr


def test_precommit_guard_blocks_row_level_matrix_and_table_formats() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text()

    for extension in (
        "csv",
        "tsv",
        "jsonl",
        "xlsx",
        "parquet",
        "arrow",
        "npz",
        "npy",
        "hdf5",
        "sqlite",
        "rds",
        "dta",
        "sav",
    ):
        assert extension in config
    assert "no-unreviewed-rendered-artifacts" in config
    assert "manuscript/figures/" in config
