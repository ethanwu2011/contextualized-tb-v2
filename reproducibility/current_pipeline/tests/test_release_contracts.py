"""Tests for the row-free release contract and temporal parsing guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from figures.palette import NARRATIVE_ORDER, subtype_name

from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes
from ctb.data.load import load_table
from ctb.data.temporal import extract_calendar_year


def test_registration_years_are_not_parsed_as_nanoseconds() -> None:
    """Integer calendar years remain the intended calendar years."""
    values = pd.Series([2020, 2021, 2022])
    parsed = extract_calendar_year(values)
    assert parsed.tolist() == [2020, 2021, 2022]


def test_registration_iso_dates_are_supported() -> None:
    """ISO registration dates retain their calendar year."""
    values = pd.Series(["2020-03-01", "2021-07-15"])
    parsed = extract_calendar_year(values)
    assert parsed.tolist() == [2020, 2021]


def test_dst_day_offsets_remain_numeric(tmp_path: Path) -> None:
    """Culture day offsets are not silently converted to Unix-epoch timestamps."""
    raw = tmp_path / "TB_Portals_DST_August_2023.csv"
    pd.DataFrame(
        {
            "condition_id": ["c1", "c1"],
            "specimen_collection_date": [0, 60],
            "test_date": [0, 60],
            "cultureresults": ["Positive", "Negative"],
        }
    ).to_csv(raw, index=False)
    loaded = load_table("dst", tmp_path)
    assert loaded["specimen_collection_date"].tolist() == [0, 60]
    assert loaded["test_date"].tolist() == [0, 60]
    assert pd.api.types.is_numeric_dtype(loaded["specimen_collection_date"])


def test_public_manifest_names_corrected_endpoint_and_release_paths() -> None:
    """The public manifest must describe the corrected cohort and aggregate release."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["primary_endpoint"]["name"] == "recorded_final_patient_cases_died_or_failure"
    assert manifest["primary_endpoint"]["loss_to_follow_up_included"] is False
    assert manifest["primary_endpoint"]["culture_positive_required"] is False
    assert manifest["analysis_universe"] == {
        "assessed_conditions": 9647,
        "patients": 9481,
        "events": 1770,
        "observed_non_events": 7877,
        "retained_context_features": 279,
        "retained_regimen_components": 26,
    }
    assert manifest["public_smoke_command"] == "make manuscript-smoke"
    assert manifest["aggregate_figure_command"] == "make aggregate-figures"
    assert (root / manifest["aggregate_checksums"]).is_file()
    assert manifest["exact_public_file_manifest"] == "PUBLIC_FILE_MANIFEST.sha256"
    assert (root / manifest["exact_public_file_manifest"]).is_file()
    assert manifest["internal_decision_curve_command"] == "make locked-validation-dca-dua"


def test_public_data_dictionary_suppresses_known_small_cells() -> None:
    """Known rare restricted-cohort categories cannot expose exact counts."""
    root = Path(__file__).resolve().parents[1]
    dictionary = (root / "docs/DATA_DICTIONARY.md").read_text(encoding="utf-8")
    # Rare categories are merged with a neighbour, so no count below 30 can be recovered by
    # subtracting the listed categories from the stated totals.
    for merged in ("Other or unknown 284", "Poly DR or negative 277", "ended or terminated 7666"):
        assert merged in dictionary
    assert "<30" not in dictionary


def test_public_manifest_claims_match_checksummed_aggregate_reports() -> None:
    """Duplicated public claims must equal their governed aggregate sources."""
    root = Path(__file__).resolve().parents[1]
    aggregate_dir = root / "manuscript/aggregate_results"
    checksum_path = aggregate_dir / "CHECKSUMS.sha256"
    checksum_rows = [line.split() for line in checksum_path.read_text().splitlines() if line]
    assert {row[1] for row in checksum_rows} == {path.name for path in aggregate_dir.glob("*.json")}
    for expected, filename in checksum_rows:
        observed = hashlib.sha256((aggregate_dir / filename).read_bytes()).hexdigest()
        assert observed == expected

    manifest = json.loads((root / "PUBLIC_RELEASE_MANIFEST.json").read_text())
    k_report = json.loads((aggregate_dir / "shared_k_sensitivity.json").read_text())
    composition = json.loads(
        (aggregate_dir / "shared_profile_reference_composition.json").read_text()
    )
    stability = json.loads(
        (aggregate_dir / "shared_profile_full_pipeline_stability.json").read_text()
    )
    brazil = json.loads((aggregate_dir / "brazil_source_rebuild_audit.json").read_text())

    universe = manifest["analysis_universe"]
    for key in ("assessed_conditions", "patients", "events", "observed_non_events"):
        assert universe[key] == k_report["cohort"][key]
    assert universe["assessed_conditions"] == composition["cohort"]["assessed_conditions"]
    assert universe["events"] == composition["cohort"]["events"]
    assert (
        universe["retained_context_features"] == composition["cohort"]["retained_context_features"]
    )
    assert (
        universe["retained_regimen_components"]
        == composition["cohort"]["retained_regimen_features"]
    )

    manifest_profiles = sorted(manifest["profiles"], key=lambda row: row["risk_rank"])
    report_profiles = sorted(composition["profiles"], key=lambda row: row["risk_rank"])
    assert len(manifest_profiles) == len(report_profiles) == 5
    assert len({row["label"] for row in manifest_profiles}) == 5
    assert [subtype_name(index) for index in NARRATIVE_ORDER] == [
        row["label"] for row in manifest_profiles
    ]
    for profile in report_profiles:
        for field in ("country_composition", "resistance_composition"):
            entries = profile[field]
            if not any(item["percent"] is None for item in entries.values()):
                continue
            visible_count = sum(
                int(item["count"]) for item in entries.values() if isinstance(item["count"], int)
            )
            assert int(profile["hard_count"]) - visible_count >= 30
    for declared, observed in zip(manifest_profiles, report_profiles, strict=True):
        assert declared["risk_rank"] == observed["risk_rank"]
        assert declared["hard_display_count"] == observed["hard_count"]
        assert declared["hard_event_rate"] == pytest.approx(observed["hard_event_rate"])
    assert (
        sum(row["hard_display_count"] for row in manifest_profiles)
        == universe["assessed_conditions"]
    )

    shared = k_report["oof_metrics"]["shared_k5"]
    additive = k_report["oof_metrics"]["additive"]
    declared_metrics = manifest["shared_patch_k5"]
    for key in ("auroc", "auprc", "log_loss", "brier"):
        assert declared_metrics[f"held_out_{key}"] == pytest.approx(shared[key])
    assert declared_metrics["additive_parent_auroc"] == pytest.approx(additive["auroc"])

    surface = stability["profile_by_regimen_uncertainty"]
    declared_surface = manifest["coefficient_surface"]
    assert declared_surface["cells_total"] == surface["cells_total"]
    assert (
        declared_surface["marginally_supported_cells"]
        == surface["cells_supported_by_soft_marginal_gate"]
    )
    assert declared_surface["suppressed_cells"] == surface["cells_suppressed_by_soft_marginal_gate"]
    assert (
        declared_surface["supported_cells_interval_excluding_zero"]
        == surface["supported_cells_with_interval_excluding_zero"]
    )
    assert (
        declared_surface["supported_cells_sign_stable_at_least_0_80"]
        == surface["supported_cells_with_sign_agreement_at_least_0.80"]
    )

    rebuilt = brazil["cohort_rebuild"]
    strict = brazil["external_discrimination"]["strict_primary_excluding_postmortem"]
    declared_brazil = manifest["brazil_transport"]
    assert (
        declared_brazil["legacy_cohort_conditions"]
        == rebuilt["legacy_manuscript_primary_conditions"]
    )
    assert (
        declared_brazil["strict_cohort_conditions"]
        == rebuilt["strict_primary_conditions_excluding_postmortem_entry"]
    )
    assert declared_brazil["strict_patch_profile_auroc"] == pytest.approx(
        strict["patch_profile_score"]["auroc"]
    )
    assert declared_brazil["strict_context_logistic_auroc"] == pytest.approx(
        strict["context_logistic"]["auroc"]
    )
    assert declared_brazil["strict_resistance_only_auroc"] == pytest.approx(
        strict["resistance_only"]["auroc"]
    )


def test_primary_matrix_loader_rejects_legacy_endpoint(tmp_path: Path) -> None:
    """The manuscript loader must reject a conversion matrix even if its shape matches."""
    path = tmp_path / "matrix.npz"
    np.savez(
        path,
        X=np.zeros((2, 1)),
        C=np.zeros((2, 1)),
        y=np.zeros(2),
        drugs=np.asarray(["drug_R"], dtype=object),
        cond_ids=np.asarray(["a", "b"], dtype=object),
        context_names=np.asarray(["age"], dtype=object),
        outcome_name=np.asarray("conversion"),
    )
    with pytest.raises(ValueError, match="Expected endpoint"):
        load_primary_matrix(path)


def test_subtype_loader_rejects_mismatched_primary_rows(tmp_path: Path) -> None:
    """Subtype labels cannot be silently paired with a different row order."""
    path = tmp_path / "subtypes.npz"
    np.savez(
        path,
        subtypes=np.asarray([0, 1]),
        loadings=np.zeros((2, 2)),
        cond_ids=np.asarray(["a", "b"], dtype=object),
        outcome_name=np.asarray("died_or_failure"),
    )
    with pytest.raises(ValueError, match="do not match"):
        load_primary_subtypes(path, expected_condition_ids=["b", "a"])
