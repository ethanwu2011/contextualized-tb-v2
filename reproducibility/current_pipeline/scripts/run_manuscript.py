"""Fail-fast, manifest-producing runner for the tracked PATCH remediation workflow.

The default public mode uses generated synthetic data and reduced model budgets. It exercises raw
table cleaning inside the endpoint builder plus the endpoint-artifact and S1/S2 rendering paths,
but its numerical output is fake and has no scientific meaning. DUA mode uses the same declared
stages without reduced budgets.

This is intentionally an honest bounded workflow: ``config/manuscript_workflow.json`` marks the
main/submission panels that are not yet covered instead of silently treating cached outputs as
reproduced.

Usage
-----
    uv run --frozen python scripts/run_manuscript.py --mode synthetic --reduced
    CTB_RAW_DIR=/secure/tb_portals uv run --frozen python scripts/run_manuscript.py --mode dua \
        --output-root /secure/new-run
    uv run --frozen python scripts/run_manuscript.py --mode synthetic --reduced --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
DEFAULT_SPEC = ROOT / "config" / "manuscript_workflow.json"
DEFAULT_ARTIFACT_ROOT = ROOT / ".artifacts" / "manuscript-smoke"
COHORT_REPORT_SCHEMA_VERSION = 2
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment_record() -> dict[str, Any]:
    """Record installed distribution names/versions without exposing installation paths."""
    packages = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).strip().lower(),
                str(distribution.version),
            )
            for distribution in distributions()
            if str(distribution.metadata.get("Name", "")).strip()
        }
    )
    records = [{"name": name, "version": version} for name, version in packages]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "n_distributions": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "distributions": records,
        "lock_match_verified": False,
    }


def source_tree_record(*, require_tracked: bool = False) -> dict[str, Any]:
    """Fingerprint and inventory every file that can affect the bounded workflow."""
    roots = ("src", "scripts", "figures", "config")
    paths: list[Path] = []
    for relative_root in roots:
        paths.extend(
            path
            for path in (ROOT / relative_root).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    paths.extend(ROOT / filename for filename in ("Makefile", "pyproject.toml", "uv.lock"))
    ordered = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())

    digest = hashlib.sha256()
    records: list[dict[str, str]] = []
    for path in ordered:
        relative_text = path.relative_to(ROOT).as_posix()
        if require_tracked:
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError(f"DUA source inventory may not traverse symlinks: {relative_text}")
            tracked = _git_value("ls-files", "--error-unmatch", relative_text)
            if not tracked:
                raise ValueError(
                    f"DUA source inventory includes an untracked file: {relative_text}"
                )
            working_blob = _git_value("hash-object", relative_text)
            head_blob = _git_value("rev-parse", f"HEAD:{relative_text}")
            if not working_blob or not head_blob or working_blob != head_blob:
                raise ValueError(
                    "DUA source inventory differs from HEAD, including a possible "
                    f"assume-unchanged/skip-worktree path: {relative_text}"
                )
        relative = relative_text.encode("utf-8")
        fingerprint = file_sha256(path)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(fingerprint))
        records.append({"declared_path": relative_text, "sha256": fingerprint})
    return {
        "declared_roots": [*roots, "Makefile", "pyproject.toml", "uv.lock"],
        "n_files": len(ordered),
        "sha256": digest.hexdigest(),
        "files": records,
    }


def load_workflow_spec(path: Path) -> dict[str, Any]:
    """Load and minimally validate the tracked workflow declaration."""
    payload: dict[str, Any] = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported workflow schema in {path}")
    from ctb.data.model_matrices import ARTIFACT_SCHEMA_VERSION

    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Workflow {path} does not declare the current artifact schema")
    if payload.get("cohort_report_schema_version") != COHORT_REPORT_SCHEMA_VERSION:
        raise ValueError(f"Workflow {path} does not declare the current cohort-report schema")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"Workflow {path} has no stages")
    ids = [stage.get("id") for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Workflow {path} contains duplicate stage ids")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Workflow {path} must declare an integer seed")
    if payload.get("submission_complete") is not False:
        raise ValueError("This bounded remediation runner cannot certify submission_complete=true")
    gaps = payload.get("known_release_gaps")
    if not isinstance(gaps, list) or not gaps:
        raise ValueError("Bounded workflow must declare at least one known release gap")
    return payload


def _render(value: str, variables: dict[str, str]) -> str:
    """Substitute the workflow's allowlisted placeholders."""
    try:
        return value.format_map(variables)
    except KeyError as exc:
        raise ValueError(f"Unknown workflow placeholder {exc.args[0]!r} in {value!r}") from exc


def _resolve_path(value: str, variables: dict[str, str]) -> Path:
    """Resolve a rendered workflow path relative to the repository."""
    rendered = Path(_render(value, variables))
    return rendered if rendered.is_absolute() else ROOT / rendered


def _git_value(*args: str) -> str | None:
    """Return a small Git query result, or ``None`` outside a usable checkout."""
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _is_repository_checkout() -> bool:
    """Return whether ROOT itself is the active Git worktree root."""
    git_root = _git_value("rev-parse", "--show-toplevel")
    return git_root is not None and Path(git_root).resolve() == ROOT.resolve()


def _public_release_hashes() -> dict[str, str]:
    """Read the exact public file manifest for a source-only archive."""
    manifest = ROOT / "PUBLIC_FILE_MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError("Source archive has no exact public file manifest")
    hashes: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative or relative in hashes:
            raise ValueError("Exact public file manifest contains an invalid entry")
        hashes[relative] = digest
    return hashes


def _validate_clean_git_state(*, expected_commit: str | None = None) -> tuple[str, str]:
    """Require a clean, resolvable Git state and optionally the same HEAD as an earlier snapshot."""
    status = _git_value("status", "--porcelain")
    commit = _git_value("rev-parse", "HEAD")
    if status is None or commit is None:
        message = (
            "--require-clean requires a valid Git checkout with a resolvable HEAD; "
            "Git provenance could not be established"
        )
        if expected_commit is not None:
            raise RuntimeError(message)
        raise SystemExit(message)
    if status:
        message = "--require-clean was set, but the checkout has uncommitted changes"
        if expected_commit is not None:
            raise RuntimeError(f"{message} during execution")
        raise SystemExit(message)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "Git HEAD changed during the governed run; refusing to certify mixed provenance"
        )
    return status, commit


def _require_tracked_repo_file(path: Path, *, label: str) -> str:
    """Return a repository-relative path only when it is inside ROOT and Git-tracked."""
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"DUA {label} must be inside the repository and Git-tracked") from exc
    tracked = _git_value("ls-files", "--error-unmatch", relative)
    if not tracked:
        raise SystemExit(f"DUA {label} is not Git-tracked: {relative}")
    return relative


def _verify_unchanged_files(observed: dict[Path, str], *, label: str) -> None:
    """Reject missing or mutated files after stage execution."""
    for path, expected_hash in observed.items():
        if not path.is_file():
            raise RuntimeError(f"Workflow {label} disappeared during execution: {path.name}")
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Workflow {label} changed during execution: {path.name}")


def validate_source_mapping(spec: dict[str, Any]) -> None:
    """Ensure every declared panel has exactly one governed producer stage."""
    stages = {stage["id"]: stage for stage in spec["stages"]}
    repository_checkout = _is_repository_checkout()
    release_hashes = {} if repository_checkout else _public_release_hashes()
    seen: set[str] = set()
    for panel in spec.get("panels", []):
        panel_id = str(panel["panel"])
        if panel_id in seen:
            raise ValueError(f"Panel {panel_id} is mapped more than once")
        seen.add(panel_id)
        stage_id = panel["stage"]
        if stage_id not in stages:
            raise ValueError(f"Panel {panel_id} references missing stage {stage_id!r}")
        producer = ROOT / panel["producer"]
        if not producer.is_file():
            raise FileNotFoundError(f"Panel {panel_id} producer is missing: {producer}")
        producer_relative = producer.relative_to(ROOT).as_posix()
        if repository_checkout:
            tracked = _git_value("ls-files", "--error-unmatch", producer_relative)
            if tracked is None:
                raise ValueError(f"Panel {panel_id} producer is not tracked: {producer}")
        elif release_hashes.get(producer_relative) != file_sha256(producer):
            raise ValueError(
                f"Panel {panel_id} producer does not match the exact public manifest: {producer}"
            )
        if panel["outputs"] != stages[stage_id]["outputs"]:
            raise ValueError(f"Panel {panel_id} outputs disagree with stage {stage_id!r}")


def _artifact_record(declared_path: str, path: Path) -> dict[str, Any]:
    """Build a path-safe checksum record without exposing external absolute directories."""
    return {
        "declared_path": declared_path,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _endpoint_record(path: Path, *, expected_endpoint: str) -> dict[str, Any]:
    """Validate an endpoint artifact, then expose aggregate provenance metadata only."""
    from ctb.data.model_matrices import ARTIFACT_SCHEMA_VERSION, load_model_matrices

    data = load_model_matrices(path, expected_endpoint=expected_endpoint)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "endpoint": data.endpoint,
        "endpoint_definition_id": data.endpoint_definition_id,
        "endpoint_definition": data.endpoint_definition,
        "cohort_definition_id": data.cohort_definition_id,
        "cohort_definition": data.cohort_definition,
        "source_n_rows": data.source_n_rows,
        "n_unassessable": data.n_unassessable,
        "n_rows": data.n_rows,
        "n_events": data.n_events,
        "n_non_events": data.n_non_events,
        "cohort_sha256": data.cohort_sha256,
        "patient_sha256": data.patient_sha256,
        "source_cohort_sha256": data.source_cohort_sha256,
        "outcome_sha256": data.outcome_sha256,
        "design_sha256": data.design_sha256,
        "provenance_sha256": data.provenance_sha256,
        "artifact_file_sha256": file_sha256(path),
    }


def _cohort_report_record(
    path: Path,
    *,
    endpoint_record: dict[str, Any],
) -> dict[str, Any]:
    """Validate the aggregate cohort report against the primary endpoint artifact."""
    payload: dict[str, Any] = json.loads(path.read_text())
    required = {
        "report_schema_version",
        "artifact_schema_version",
        "endpoint_contract",
        "flow_counts",
        "ordered_cohort_sha256",
        "cohort_steps",
        "culture_eligibility",
        "died_failure_assessability",
        "outcome_source_concordance",
        "patient_linkage",
        "initial_regimen_selection_report",
        "culture_date_report",
        "patient_cases_died_failure_report",
        "regimen_history_died_failure_report",
        "contains_row_level_identifiers",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Cohort report is incomplete; missing {missing}")
    extra = sorted(set(payload).difference(required))
    if extra:
        raise ValueError(f"Cohort report contains undeclared fields: {extra}")
    if payload["report_schema_version"] != COHORT_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported cohort-report schema")
    if payload["artifact_schema_version"] != endpoint_record["artifact_schema_version"]:
        raise ValueError("Cohort report artifact schema disagrees with endpoint artifact")
    if payload["contains_row_level_identifiers"] is not False:
        raise ValueError("Cohort report must not declare or contain row-level identifiers")

    contract = payload["endpoint_contract"]
    contract_keys = {
        "endpoint",
        "endpoint_definition_id",
        "endpoint_definition",
        "cohort_definition_id",
        "cohort_definition",
    }
    if set(contract) != contract_keys:
        raise ValueError("Cohort report endpoint contract fields are not canonical")
    for key in (
        "endpoint",
        "endpoint_definition_id",
        "endpoint_definition",
        "cohort_definition_id",
        "cohort_definition",
    ):
        if contract.get(key) != endpoint_record[key]:
            raise ValueError(f"Cohort report endpoint contract mismatch for {key}")

    flow = payload["flow_counts"]
    flow_keys = {
        "all_patient_conditions",
        "consort_conditions",
        "primary_source_conditions",
        "culture_conversion_sensitivity_conditions",
        "died_failure_assessed_conditions",
        "died_failure_unassessable_conditions",
        "died_failure_events",
        "died_failure_observed_non_events",
    }
    if set(flow) != flow_keys:
        raise ValueError("Cohort report flow-count fields are not canonical")
    if any(
        isinstance(flow.get(key), bool) or not isinstance(flow.get(key), int) or flow[key] < 0
        for key in flow_keys
    ):
        raise ValueError("Cohort report flow counts must be nonnegative integers")
    expected_counts = {
        "primary_source_conditions": endpoint_record["source_n_rows"],
        "died_failure_assessed_conditions": endpoint_record["n_rows"],
        "died_failure_unassessable_conditions": endpoint_record["n_unassessable"],
        "died_failure_events": endpoint_record["n_events"],
        "died_failure_observed_non_events": endpoint_record["n_non_events"],
    }
    for key, expected in expected_counts.items():
        value = flow.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"Cohort report count mismatch for {key}")
    if (
        flow["died_failure_assessed_conditions"] + flow["died_failure_unassessable_conditions"]
        != flow["primary_source_conditions"]
        or flow["died_failure_events"] + flow["died_failure_observed_non_events"]
        != flow["died_failure_assessed_conditions"]
        or flow["consort_conditions"] != flow["primary_source_conditions"]
        or flow["culture_conversion_sensitivity_conditions"] > flow["consort_conditions"]
        or flow["consort_conditions"] > flow["all_patient_conditions"]
    ):
        raise ValueError("Cohort report denominator arithmetic is inconsistent")

    hashes = payload["ordered_cohort_sha256"]
    hash_keys = {
        "all_patient_conditions",
        "consort_conditions",
        "primary_source_conditions",
        "died_failure_assessed_conditions",
    }
    if set(hashes) != hash_keys:
        raise ValueError("Cohort report ordered-cohort hash fields are not canonical")
    if (
        hashes.get("primary_source_conditions") != endpoint_record["source_cohort_sha256"]
        or hashes.get("consort_conditions") != endpoint_record["source_cohort_sha256"]
        or hashes.get("died_failure_assessed_conditions") != endpoint_record["cohort_sha256"]
    ):
        raise ValueError(
            "Cohort report ordered-cohort fingerprints disagree with endpoint artifact"
        )

    eligibility = payload["culture_eligibility"]
    reason_keys = {
        "no_culture_row",
        "no_interpretable_culture_result",
        "no_valid_interpretable_culture_timing",
        "quarantined_same_day_conflict",
        "never_positive",
        "first_positive_after_day_zero",
        "baseline_eligible_first_positive_at_or_before_day_zero",
    }
    eligibility_keys = {
        *reason_keys,
        "consort_conditions",
        "excluded_before_baseline_eligibility",
    }
    if set(eligibility) != eligibility_keys:
        raise ValueError("Cohort report culture-eligibility fields are not canonical")
    if any(
        isinstance(eligibility.get(key), bool)
        or not isinstance(eligibility.get(key), int)
        or eligibility[key] < 0
        for key in reason_keys
    ):
        raise ValueError("Cohort report culture-eligibility counts must be nonnegative integers")
    if eligibility.get("consort_conditions") != flow["consort_conditions"]:
        raise ValueError("Cohort report culture eligibility disagrees with CONSORT count")
    if sum(eligibility[key] for key in reason_keys) != flow["consort_conditions"]:
        raise ValueError("Cohort report culture exclusions do not reconstruct CONSORT")
    if (
        eligibility["baseline_eligible_first_positive_at_or_before_day_zero"]
        != flow["culture_conversion_sensitivity_conditions"]
    ):
        raise ValueError("Cohort report culture eligibility disagrees with sensitivity cohort")
    if (
        eligibility.get("excluded_before_baseline_eligibility")
        != flow["consort_conditions"] - flow["culture_conversion_sensitivity_conditions"]
    ):
        raise ValueError("Cohort report pre-baseline exclusion count is inconsistent")

    assessability = payload["died_failure_assessability"]
    assessability_reason_keys = {
        "event_recorded_died_failure",
        "observed_nonevent_recorded_cured_completed",
        "lost_to_follow_up",
        "still_on_treatment",
        "palliative_care",
        "unknown",
        "missing_or_blank_outcome",
        "other_unclassified_outcome",
    }
    assessability_keys = {
        *assessability_reason_keys,
        "source_conditions",
        "assessed_conditions",
        "unassessable_conditions",
    }
    if set(assessability) != assessability_keys or any(
        isinstance(assessability.get(key), bool)
        or not isinstance(assessability.get(key), int)
        or assessability[key] < 0
        for key in assessability_keys
    ):
        raise ValueError("Cohort report died/failure assessability fields are not canonical")
    if (
        sum(assessability[key] for key in assessability_reason_keys)
        != assessability["source_conditions"]
    ):
        raise ValueError("Died/failure assessability reasons do not reconstruct the source")
    if (
        assessability["source_conditions"] != endpoint_record["source_n_rows"]
        or assessability["assessed_conditions"] != endpoint_record["n_rows"]
        or assessability["unassessable_conditions"] != endpoint_record["n_unassessable"]
        or assessability["event_recorded_died_failure"] != endpoint_record["n_events"]
        or assessability["observed_nonevent_recorded_cured_completed"]
        != endpoint_record["n_non_events"]
    ):
        raise ValueError("Died/failure assessability reasons disagree with endpoint artifact")

    concordance = payload["outcome_source_concordance"]
    concordance_count_keys = {
        "source_conditions",
        "patient_cases_assessed",
        "regimen_history_assessed",
        "both_assessed",
        "patient_cases_only_assessed",
        "regimen_history_only_assessed",
        "label_disagreements_among_both_assessed",
    }
    concordance_keys = {*concordance_count_keys, "exact_assessability_and_label_concordance"}
    if set(concordance) != concordance_keys or any(
        isinstance(concordance.get(key), bool)
        or not isinstance(concordance.get(key), int)
        or concordance[key] < 0
        for key in concordance_count_keys
    ):
        raise ValueError("Cohort report outcome-concordance fields are not canonical")
    if concordance["exact_assessability_and_label_concordance"] is not True:
        raise ValueError("Cohort report requires exact outcome-source concordance")
    if (
        concordance["source_conditions"] != endpoint_record["source_n_rows"]
        or concordance["patient_cases_assessed"] != endpoint_record["n_rows"]
        or concordance["regimen_history_assessed"] != endpoint_record["n_rows"]
        or concordance["both_assessed"] != endpoint_record["n_rows"]
        or concordance["patient_cases_only_assessed"] != 0
        or concordance["regimen_history_only_assessed"] != 0
        or concordance["label_disagreements_among_both_assessed"] != 0
    ):
        raise ValueError("Cohort report outcome concordance disagrees with endpoint artifact")

    patient_linkage = payload["patient_linkage"]
    patient_linkage_keys = {
        "source_conditions",
        "source_unique_patients",
        "assessed_conditions",
        "assessed_unique_patients",
        "missing_links",
        "ambiguous_links",
    }
    if set(patient_linkage) != patient_linkage_keys or any(
        isinstance(patient_linkage.get(key), bool)
        or not isinstance(patient_linkage.get(key), int)
        or patient_linkage[key] < 0
        for key in patient_linkage_keys
    ):
        raise ValueError("Cohort report patient-linkage fields are not canonical")
    if (
        patient_linkage["source_conditions"] != endpoint_record["source_n_rows"]
        or patient_linkage["assessed_conditions"] != endpoint_record["n_rows"]
        or patient_linkage["missing_links"] != 0
        or patient_linkage["ambiguous_links"] != 0
        or patient_linkage["source_unique_patients"] > patient_linkage["source_conditions"]
        or patient_linkage["assessed_unique_patients"] > patient_linkage["assessed_conditions"]
    ):
        raise ValueError("Cohort report patient linkage disagrees with endpoint artifact")

    steps = payload["cohort_steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Cohort report must contain an ordered nonempty cohort flow")
    previous_count: int | None = None
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Cohort report steps must be objects")
        current = step.get("n_conditions")
        excluded = step.get("n_excluded")
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or current < 0
            or isinstance(excluded, bool)
            or not isinstance(excluded, int)
            or excluded < 0
        ):
            raise ValueError("Cohort report step counts must be nonnegative integers")
        if previous_count is not None and previous_count - excluded != current:
            raise ValueError("Cohort report steps do not reconcile sequentially")
        previous_count = current
    if previous_count != endpoint_record["n_rows"]:
        raise ValueError("Cohort report final step disagrees with assessed endpoint rows")

    expected_diagnostic_fields = {
        "initial_regimen_selection_report": {
            "total_rows",
            "zero_drug_rows",
            "undated_nonempty_rows",
            "invalid_start_rows",
            "conflicting_start_rows",
            "same_start_tie_conditions",
            "conflicting_tie_conditions",
            "conditions_skipping_leading_zero_drug_row",
            "conditions_without_initial_regimen",
            "temporal_coordinate",
        },
        "culture_date_report": {
            "total_rows",
            "invalid_day_rows",
            "missing_day_rows",
            "conflicting_same_day_groups",
            "conditions_quarantined_for_same_day_conflict",
        },
        "patient_cases_died_failure_report": {
            "source_table",
            "source_column",
            "n_conditions",
            "n_assessable",
            "n_events",
            "n_observed_nonevents",
            "n_unassessable",
            "event_values",
            "observed_nonevent_values",
        },
        "regimen_history_died_failure_report": {
            "n_conditions",
            "n_assessable",
            "n_events",
            "n_observed_nonevents",
            "n_unassessable",
            "event_codes",
            "observed_nonevent_codes",
        },
    }
    for field, expected_fields in expected_diagnostic_fields.items():
        value = payload[field]
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ValueError(f"Cohort report diagnostic fields are not canonical for {field}")
    for field in ("patient_cases_died_failure_report", "regimen_history_died_failure_report"):
        diagnostic = payload[field]
        count_fields = (
            "n_conditions",
            "n_assessable",
            "n_events",
            "n_observed_nonevents",
            "n_unassessable",
        )
        if any(
            isinstance(diagnostic.get(key), bool)
            or not isinstance(diagnostic.get(key), int)
            or diagnostic[key] < 0
            for key in count_fields
        ):
            raise ValueError(f"Cohort report diagnostic counts are invalid for {field}")
        if (
            diagnostic["n_assessable"] + diagnostic["n_unassessable"] != diagnostic["n_conditions"]
            or diagnostic["n_events"] + diagnostic["n_observed_nonevents"]
            != diagnostic["n_assessable"]
        ):
            raise ValueError(f"Cohort report diagnostic counts do not reconcile for {field}")

    return {
        "report_schema_version": COHORT_REPORT_SCHEMA_VERSION,
        "report_file_sha256": file_sha256(path),
        "validated_against_endpoint_artifact": True,
        "flow_counts": flow,
        "ordered_cohort_sha256": hashes,
        "culture_eligibility": eligibility,
        "died_failure_assessability": assessability,
        "outcome_source_concordance": concordance,
        "patient_linkage": patient_linkage,
    }


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write the run manifest after creating only its aggregate-output directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _stage_environment(
    *,
    output_root: Path,
    raw_dir: Path,
    mode: str,
    reduced: bool,
    synthetic_stage: bool,
    seed: int,
) -> dict[str, str]:
    """Return a sanitized child environment with all CTB controls set explicitly."""
    env = os.environ.copy()
    for key in (
        "CTB_OUTPUT_ROOT",
        "CTB_RAW_DIR",
        "CTB_WORKFLOW_MODE",
        "CTB_REDUCED_RUN",
        "CTB_GOVERNED_RUN",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONHASHSEED"] = str(seed)
    env["CTB_OUTPUT_ROOT"] = str(output_root)
    env["CTB_WORKFLOW_MODE"] = mode
    env["CTB_GOVERNED_RUN"] = "1"
    if not synthetic_stage:
        env["CTB_RAW_DIR"] = str(raw_dir)
    if reduced:
        env["CTB_REDUCED_RUN"] = "1"
    return env


def _validate_typed_output(path: Path) -> None:
    """Reject empty or structurally corrupt declared outputs."""
    if path.stat().st_size <= 0:
        raise ValueError(f"Declared output is empty: {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        from ctb.data.model_matrices import (
            CULTURE_CONVERSION,
            DIED_FAILURE,
            ENDPOINT_ARTIFACTS,
            load_model_matrices,
        )

        endpoint_by_filename = {
            ENDPOINT_ARTIFACTS[CULTURE_CONVERSION]: CULTURE_CONVERSION,
            ENDPOINT_ARTIFACTS[DIED_FAILURE]: DIED_FAILURE,
        }
        try:
            expected_endpoint = endpoint_by_filename[path.name]
        except KeyError as exc:
            raise ValueError(f"Undeclared NPZ output contract: {path.name}") from exc
        load_model_matrices(path, expected_endpoint=expected_endpoint)
    elif suffix == ".parquet":
        import pyarrow.parquet as parquet

        metadata = parquet.ParquetFile(path).metadata
        if metadata.num_rows <= 0 or metadata.num_columns <= 0:
            raise ValueError(f"Declared Parquet output has no rows or columns: {path.name}")
    elif suffix == ".pdf":
        with path.open("rb") as handle:
            header = handle.read(5)
            handle.seek(max(path.stat().st_size - 1024, 0))
            trailer = handle.read()
        if header != b"%PDF-" or b"%%EOF" not in trailer:
            raise ValueError(f"Declared PDF output is structurally invalid: {path.name}")
    elif suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Declared PNG output has invalid dimensions: {path.name}")
    elif suffix == ".json":
        json.loads(path.read_text())


def _validate_metadata_alignment(
    path: Path,
    *,
    expected_cond_ids: list[str],
    required_columns: set[str],
    expected_binary_columns: dict[str, list[int]],
) -> None:
    """Require a row-level metadata Parquet to align exactly with its endpoint artifact."""
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.index.name != "condition_id":
        raise ValueError(f"Metadata index must be condition_id: {path.name}")
    actual_ids = [str(value) for value in frame.index]
    if actual_ids != expected_cond_ids:
        raise ValueError(f"Metadata condition order disagrees with endpoint artifact: {path.name}")
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata output {path.name} is missing required columns {missing}")
    for column, expected in expected_binary_columns.items():
        actual = frame[column]
        supported_dtype = bool(
            pd.api.types.is_bool_dtype(actual.dtype) or pd.api.types.is_integer_dtype(actual.dtype)
        )
        if (
            not supported_dtype
            or actual.isna().any()
            or not actual.isin([0, 1]).all()
            or actual.astype(int).tolist() != expected
        ):
            raise ValueError(
                f"Metadata column {column!r} disagrees with endpoint artifact: {path.name}"
            )


def _same_file_or_target(first: Path, second: Path) -> bool:
    """Return whether two paths resolve to, or already alias, the same filesystem object."""
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def _generated_file_inventory(
    root: Path,
    *,
    excluded: set[Path],
) -> dict[str, Any]:
    """Hash every generated file except the self-referential manifest and transient run lock."""
    excluded_resolved = {path.resolve(strict=False) for path in excluded}
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve(strict=False) not in excluded_resolved
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        fingerprint = file_sha256(path)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(fingerprint))
        records.append(
            {
                "declared_path": f"{{output_root}}/{relative}",
                "size_bytes": path.stat().st_size,
                "sha256": fingerprint,
            }
        )
    return {"n_files": len(records), "sha256": digest.hexdigest(), "files": records}


def run_workflow(args: argparse.Namespace) -> int:
    """Execute selected stages and write a machine-readable provenance record."""
    spec_path = args.spec.resolve()
    spec = load_workflow_spec(spec_path)
    validate_source_mapping(spec)

    if args.mode == "dua" and args.output_root is None:
        raise SystemExit(
            "DUA mode requires an explicit fresh --output-root; repository/cached outputs "
            "cannot be certified"
        )
    output_root = DEFAULT_ARTIFACT_ROOT if args.output_root is None else args.output_root.resolve()
    synthetic_dir = (
        args.synthetic_dir.resolve()
        if args.synthetic_dir is not None
        else output_root / "data" / "raw" / "synthetic"
    )
    output_manifest = (
        args.output_manifest.resolve()
        if args.output_manifest is not None
        else output_root / "results" / "provenance" / "manuscript_run.json"
    )
    if args.mode == "synthetic":
        raw_dir = synthetic_dir
        protected_raw_value = os.environ.get("CTB_RAW_DIR")
        if protected_raw_value:
            protected_raw = Path(protected_raw_value).expanduser().resolve()
            synthetic_resolved = synthetic_dir.resolve(strict=False)
            if (
                synthetic_resolved == protected_raw
                or synthetic_resolved.is_relative_to(protected_raw)
                or protected_raw.is_relative_to(synthetic_resolved)
            ):
                raise SystemExit(
                    "Synthetic output overlaps the configured governed CTB_RAW_DIR; "
                    "choose a disjoint synthetic directory"
                )
    else:
        raw_value = os.environ.get("CTB_RAW_DIR")
        if not raw_value:
            raise SystemExit(
                "DUA mode requires CTB_RAW_DIR to name the authorized raw-data directory"
            )
        raw_dir = Path(raw_value).expanduser().resolve()

    variables = {
        "python": sys.executable,
        "output_root": str(output_root),
        "raw_dir": str(raw_dir),
        "synthetic_dir": str(synthetic_dir),
        "synthetic_n": str(args.synthetic_n),
        "seed": str(spec["seed"]),
    }
    selected = [stage for stage in spec["stages"] if args.mode in stage["modes"]]
    if not selected:
        raise ValueError(f"Workflow has no stages selected for mode {args.mode!r}")
    selected_ids = {str(stage["id"]) for stage in selected}
    required_panel_stages = {str(panel["stage"]) for panel in spec.get("panels", [])}
    if not required_panel_stages.issubset(selected_ids):
        raise ValueError("Selected workflow omits a declared panel producer stage")
    builders = [stage for stage in selected if stage["id"] == "build_endpoint_matrices"]
    if len(builders) != 1:
        raise ValueError("Selected workflow must contain exactly one endpoint-builder stage")
    required_mode_stage = "synthetic_data" if args.mode == "synthetic" else "verify_dua_inputs"
    if sum(stage["id"] == required_mode_stage for stage in selected) != 1:
        raise ValueError(
            f"Selected {args.mode!r} workflow must contain exactly one "
            f"{required_mode_stage!r} stage"
        )

    resolved_inputs: list[Path] = []
    resolved_outputs: list[Path] = []
    for stage in selected:
        resolved_inputs.extend(_resolve_path(str(value), variables) for value in stage["inputs"])
        resolved_outputs.extend(_resolve_path(str(value), variables) for value in stage["outputs"])
    normalized_outputs = [path.resolve(strict=False) for path in resolved_outputs]
    if len(normalized_outputs) != len(set(normalized_outputs)):
        raise ValueError("Workflow stages must not declare the same output path more than once")
    endpoint_path = output_root / "data" / "processed" / "died_failure_model_matrices.npz"
    cohort_report_path = output_root / "data" / "processed" / "died_failure_cohort_report.json"
    builder_outputs = {
        _resolve_path(str(value), variables).resolve(strict=False)
        for value in builders[0]["outputs"]
    }
    if endpoint_path.resolve(strict=False) not in builder_outputs:
        raise ValueError("Endpoint-builder stage does not declare the primary endpoint artifact")
    if cohort_report_path.resolve(strict=False) not in builder_outputs:
        raise ValueError("Endpoint-builder stage does not declare the cohort report")

    output_root_resolved = output_root.resolve(strict=False)
    raw_dir_resolved = raw_dir.resolve(strict=False)
    if args.mode == "dua":
        broad_roots = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
        if output_root_resolved in broad_roots or ROOT.resolve().is_relative_to(
            output_root_resolved
        ):
            raise SystemExit("DUA output root must be a dedicated non-broad directory")
        if (
            output_root_resolved == raw_dir_resolved
            or output_root_resolved.is_relative_to(raw_dir_resolved)
            or raw_dir_resolved.is_relative_to(output_root_resolved)
        ):
            raise SystemExit("DUA raw-data and output roots must be disjoint")
    for path in resolved_outputs:
        if not path.resolve(strict=False).is_relative_to(output_root_resolved):
            raise ValueError(f"Declared output escapes output root: {path}")
    for path in resolved_inputs:
        resolved = path.resolve(strict=False)
        if not (
            resolved.is_relative_to(output_root_resolved)
            or resolved.is_relative_to(raw_dir_resolved)
        ):
            raise ValueError(f"Declared input is outside raw/output roots: {path}")

    provenance_root = (output_root / "results" / "provenance").resolve(strict=False)
    if not provenance_root.is_relative_to(output_root_resolved):
        raise ValueError("Provenance directory escapes the resolved output root")
    if not output_manifest.resolve(strict=False).is_relative_to(provenance_root):
        raise ValueError("Run manifest must be inside the dedicated output provenance directory")
    for path in [*resolved_inputs, *resolved_outputs]:
        if _same_file_or_target(output_manifest, path):
            raise ValueError("Run manifest must not alias a declared workflow input or output")

    lock_path = ROOT / "uv.lock"
    if args.require_clean:
        spec_relative = _require_tracked_repo_file(spec_path, label="workflow specification")
        if not spec_relative.startswith("config/"):
            raise SystemExit(
                "DUA workflow specification must be tracked under the repository config directory"
            )
        initial_status, git_commit = _validate_clean_git_state()
        executed_source_tree = source_tree_record(require_tracked=True)
        if output_root.exists() and any(output_root.iterdir()):
            raise SystemExit(
                "DUA mode requires a fresh empty output root with no pre-existing files "
                "or directories. Choose a new --output-root."
            )
    else:
        initial_status = _git_value("status", "--porcelain")
        git_commit = _git_value("rev-parse", "HEAD")
        executed_source_tree = source_tree_record()

    previous_umask: int | None = None
    if args.mode == "dua" and not args.dry_run:
        previous_umask = os.umask(0o077)

    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "workflow_id": spec["workflow_id"],
        "workflow_spec_sha256": file_sha256(spec_path),
        "workflow_scope": spec["scope"],
        "submission_complete": spec["submission_complete"],
        "known_release_gaps": spec["known_release_gaps"],
        "mode": args.mode,
        "reduced": bool(args.reduced),
        "synthetic_results_have_scientific_meaning": False if args.mode == "synthetic" else None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.dry_run else "running",
        "git_commit": git_commit,
        "git_status_before": initial_status.splitlines() if initial_status else [],
        "require_clean": bool(args.require_clean),
        "executed_source_tree": executed_source_tree,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_environment": runtime_environment_record(),
        "environment_lock": {
            "declared_path": "uv.lock",
            "sha256": file_sha256(lock_path),
            "verification": "declared lock hash only; frozen environment sync is an external gate",
        },
        "endpoint": spec["endpoint"],
        "seed": spec["seed"],
        "stages": [],
    }
    observed_inputs: dict[Path, str] = {}
    observed_outputs: dict[Path, str] = {}
    run_lock: Path | None = None
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if args.mode == "dua":
            os.chmod(output_root, 0o700)
        run_lock = output_root / ".manuscript_run.lock"
        try:
            descriptor = os.open(run_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise SystemExit(f"Another run owns the output root lock: {run_lock}") from exc
        else:
            os.close(descriptor)

    try:
        for stage in selected:
            command = [_render(str(part), variables) for part in stage["command"]]
            recorded_command = ["{python}" if part == sys.executable else part for part in command]
            declared_inputs = [str(value) for value in stage["inputs"]]
            declared_outputs = [str(value) for value in stage["outputs"]]
            input_paths = [_resolve_path(value, variables) for value in declared_inputs]
            output_paths = [_resolve_path(value, variables) for value in declared_outputs]

            if stage["id"] != "synthetic_data" and not args.dry_run:
                missing_inputs = [
                    declared
                    for declared, path in zip(declared_inputs, input_paths, strict=True)
                    if not path.is_file()
                ]
                if missing_inputs:
                    raise FileNotFoundError(
                        f"Stage {stage['id']} has missing prerequisites: {missing_inputs}"
                    )

            input_records = (
                []
                if stage["id"] == "synthetic_data"
                else [{"declared_path": value} for value in declared_inputs]
                if args.dry_run
                else [
                    _artifact_record(declared, path)
                    for declared, path in zip(declared_inputs, input_paths, strict=True)
                ]
            )
            if not args.dry_run:
                for path, record in zip(input_paths, input_records, strict=True):
                    fingerprint = str(record["sha256"])
                    previous = observed_inputs.get(path)
                    if previous is not None and previous != fingerprint:
                        raise RuntimeError(f"Workflow input changed between stages: {path.name}")
                    observed_inputs[path] = fingerprint

            stage_record: dict[str, Any] = {
                "id": stage["id"],
                "command": recorded_command,
                "inputs": input_records,
                "outputs": [],
                "status": "planned" if args.dry_run else "running",
            }
            run_manifest["stages"].append(stage_record)
            print(f">>> [{stage['id']}] {' '.join(command)}", flush=True)
            if args.dry_run:
                continue

            previous_mtimes = {
                path: path.stat().st_mtime_ns for path in output_paths if path.is_file()
            }
            started = time.monotonic()
            env = _stage_environment(
                output_root=output_root,
                raw_dir=raw_dir,
                mode=args.mode,
                reduced=bool(args.reduced),
                synthetic_stage=stage["id"] == "synthetic_data",
                seed=int(spec["seed"]),
            )
            subprocess.run(command, cwd=ROOT, env=env, check=True)

            missing_outputs = [
                declared
                for declared, path in zip(declared_outputs, output_paths, strict=True)
                if not path.is_file()
            ]
            if missing_outputs:
                raise FileNotFoundError(
                    f"Stage {stage['id']} completed without required outputs: {missing_outputs}"
                )
            stale_outputs = [
                declared
                for declared, path in zip(declared_outputs, output_paths, strict=True)
                if path in previous_mtimes and path.stat().st_mtime_ns <= previous_mtimes[path]
            ]
            if stale_outputs:
                raise RuntimeError(
                    f"Stage {stage['id']} did not refresh declared outputs: {stale_outputs}"
                )
            for path in output_paths:
                _validate_typed_output(path)

            stage_record["duration_seconds"] = round(time.monotonic() - started, 3)
            output_records = [
                _artifact_record(declared, path)
                for declared, path in zip(declared_outputs, output_paths, strict=True)
            ]
            for path, record in zip(output_paths, output_records, strict=True):
                observed_outputs[path] = str(record["sha256"])
            stage_record["outputs"] = output_records
            stage_record["status"] = "completed"

        if args.dry_run:
            run_manifest["status"] = "planned"
        else:
            _verify_unchanged_files(observed_inputs, label="input")
            _verify_unchanged_files(observed_outputs, label="output")
            for path in observed_outputs:
                _validate_typed_output(path)

            if args.require_clean:
                final_status, final_commit = _validate_clean_git_state(expected_commit=git_commit)
                final_source_tree = source_tree_record(require_tracked=True)
                if final_source_tree != executed_source_tree:
                    raise RuntimeError(
                        "Executable source inventory changed during the governed run"
                    )
            else:
                final_status = _git_value("status", "--porcelain")
                final_commit = _git_value("rev-parse", "HEAD")
            run_manifest["git_commit_after"] = final_commit
            run_manifest["git_status_after"] = final_status.splitlines() if final_status else []
            run_manifest["source_tree_verified_unchanged_at_completion"] = (
                final_source_tree == executed_source_tree if args.require_clean else None
            )

            endpoint_record = _endpoint_record(
                endpoint_path,
                expected_endpoint=str(spec["endpoint"]),
            )
            if endpoint_record["endpoint"] != spec["endpoint"]:
                raise RuntimeError("Run endpoint disagrees with workflow declaration")
            from ctb.data.model_matrices import (
                CULTURE_CONVERSION,
                DIED_FAILURE,
                load_model_matrices,
            )

            primary_matrices = load_model_matrices(
                endpoint_path,
                expected_endpoint=DIED_FAILURE,
            )
            conversion_matrices = load_model_matrices(
                output_root / "data" / "processed" / "conversion_model_matrices.npz",
                expected_endpoint=CULTURE_CONVERSION,
            )
            _validate_metadata_alignment(
                output_root / "data" / "processed" / "died_failure_context_meta.parquet",
                expected_cond_ids=primary_matrices.cond_ids.tolist(),
                required_columns={"died_failure"},
                expected_binary_columns={
                    "died_failure": primary_matrices.y.tolist(),
                },
            )
            _validate_metadata_alignment(
                output_root / "data" / "processed" / "conversion_context_meta.parquet",
                expected_cond_ids=conversion_matrices.cond_ids.tolist(),
                required_columns={"died_failure", "event", "time_to_event"},
                expected_binary_columns={"event": conversion_matrices.y.tolist()},
            )
            cohort_record = _cohort_report_record(
                cohort_report_path,
                endpoint_record=endpoint_record,
            )
            endpoint_output_sha = observed_outputs.get(endpoint_path)
            report_output_sha = observed_outputs.get(cohort_report_path)
            if endpoint_output_sha != endpoint_record["artifact_file_sha256"]:
                raise RuntimeError(
                    "Final endpoint artifact does not match the current builder-stage output"
                )
            if report_output_sha != cohort_record["report_file_sha256"]:
                raise RuntimeError(
                    "Final cohort report does not match the current builder-stage output"
                )
            endpoint_record["cohort_report_sha256"] = cohort_record["report_file_sha256"]
            run_manifest["endpoint_artifact"] = endpoint_record
            run_manifest["cohort_report"] = cohort_record

            linked_inputs: dict[str, str] = {}
            for stage_record in run_manifest["stages"]:
                for input_record in stage_record["inputs"]:
                    declared = str(input_record["declared_path"])
                    fingerprint = str(input_record["sha256"])
                    previous = linked_inputs.get(declared)
                    if previous is not None and previous != fingerprint:
                        raise RuntimeError(f"Input linkage changed across stages: {declared}")
                    linked_inputs[declared] = fingerprint
            linkage = {
                "endpoint_artifact_sha256": endpoint_record["artifact_file_sha256"],
                "cohort_report_sha256": cohort_record["report_file_sha256"],
                "workflow_spec_sha256": run_manifest["workflow_spec_sha256"],
                "executed_source_tree_sha256": executed_source_tree["sha256"],
                "inputs": [
                    {"declared_path": declared, "sha256": fingerprint}
                    for declared, fingerprint in sorted(linked_inputs.items())
                ],
            }
            linkage["sha256"] = hashlib.sha256(
                json.dumps(linkage, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            run_manifest["release_linkage"] = linkage
            run_manifest["generated_file_inventory"] = _generated_file_inventory(
                output_root,
                excluded={output_manifest, run_lock} if run_lock is not None else {output_manifest},
            )
            run_manifest["status"] = "completed"
        run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output_manifest, run_manifest)
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(exc).__name__}: {exc}"
        run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output_manifest, run_manifest)
        raise
    finally:
        if run_lock is not None:
            run_lock.unlink(missing_ok=True)
        if previous_umask is not None:
            os.umask(previous_umask)

    print(f">>> provenance: {output_manifest}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "dua"), default="synthetic")
    parser.add_argument("--reduced", action="store_true", help="use reduced S1/S2 training budgets")
    parser.add_argument("--synthetic-n", type=int, default=300)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="generated-artifact root (synthetic default: .artifacts/manuscript-smoke)",
    )
    parser.add_argument("--synthetic-dir", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_n < 50:
        parser.error("--synthetic-n must be at least 50 so all smoke-test code paths are populated")
    if args.mode == "dua" and args.reduced:
        parser.error("--reduced is a synthetic smoke-test option, not a DUA analysis mode")
    if args.mode == "dua":
        args.require_clean = True
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return run_workflow(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
