"""Audit PATCH episode construction, time zero, and endpoint flow with aggregate output only.

All row-level TB Portals objects remain in memory. The serialized report suppresses counts below
the disclosure threshold and contains no identifiers, row-level labels, treatment vectors, or
predictions. This script does not edit manuscript, figure, submission, or public-repository files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ctb.config import RESISTANCE_EXPECTED_DAYS, TB_PORTALS_FILES
from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_dst, load_patient_cases, load_regimens
from ctb.data.outcomes import derive_died_failure
from ctb.data.treatments import select_initial_regimens
from ctb.models.fold_preprocessing import FoldPreprocessor
from ctb.models.locked_validation import LockedValidationConfig, RawValidationData
from run_aggregate_sensitivities import (
    SMALL_CELL_MIN,
    _build_raw_data,
    _cluster_bootstrap_metrics,
    _model_summary,
    _run_grouped_logistic,
    _safe_count,
    _strict_baseline_positive_ids,
    _write_report,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_outcome(values: pd.Series[Any]) -> pd.Series[str]:
    normalized = values.astype("string").str.strip().str.casefold().replace("", pd.NA)
    mapped = normalized.map(
        {
            "completed": "completed",
            "cured": "cured",
            "died": "died",
            "failure": "failure",
            "lost to follow up": "lost_to_follow_up",
            "ltfu": "lost_to_follow_up",
            "palliative care": "palliative_care",
            "palliativecare": "palliative_care",
            "still on treatment": "still_on_treatment",
            "stillontreatment": "still_on_treatment",
            "unknown": "unknown",
        }
    )
    return mapped.fillna("missing_or_unclassified").astype(str)


def _category_counts(values: pd.Series[str], mask: pd.Series[bool]) -> dict[str, Any]:
    levels = (
        "completed",
        "cured",
        "died",
        "failure",
        "lost_to_follow_up",
        "palliative_care",
        "still_on_treatment",
        "unknown",
        "missing_or_unclassified",
    )
    counts = values.loc[mask].value_counts()
    return {level: _safe_count(int(counts.get(level, 0))) for level in levels}


def _cohort_flow(
    patient_cases: pd.DataFrame,
    initial: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series[bool], pd.Series[bool], pd.Series[bool]]:
    condition_index = pd.Index(patient_cases["condition_id"], name="condition_id")
    if not condition_index.is_unique:
        raise ValueError("Patient Cases must contain one row per condition")
    has_regimen = initial["drug_set"].map(len).gt(0).reindex(condition_index, fill_value=False)
    resistance = patient_cases.set_index("condition_id")["type_of_resistance"]
    has_defined_resistance = resistance.isin(set(RESISTANCE_EXPECTED_DAYS))
    normalized = _normalize_outcome(patient_cases["outcome"])
    normalized.index = condition_index
    primary_assessable = normalized.isin({"died", "failure", "cured", "completed"})
    who_assessable = primary_assessable | normalized.eq("lost_to_follow_up")
    eligible_exposure = has_regimen & has_defined_resistance
    primary = eligible_exposure & primary_assessable
    who = eligible_exposure & who_assessable
    steps = [
        {
            "step": "all_patient_case_conditions",
            "remaining": int(len(condition_index)),
            "excluded_at_step": 0,
        },
        {
            "step": "valid_earliest_nonempty_regimen",
            "remaining": int(has_regimen.sum()),
            "excluded_at_step": _safe_count(int((~has_regimen).sum())),
        },
        {
            "step": "defined_resistance_class",
            "remaining": int(eligible_exposure.sum()),
            "excluded_at_step": _safe_count(int((has_regimen & ~has_defined_resistance).sum())),
        },
        {
            "step": "assessable_recorded_died_failure_endpoint",
            "remaining": int(primary.sum()),
            "excluded_at_step": _safe_count(int((eligible_exposure & ~primary_assessable).sum())),
        },
    ]
    return (
        {
            "steps": steps,
            "who_unfavorable_assessable_conditions": int(who.sum()),
            "who_unfavorable_added_by_ltfu": int((who & ~primary).sum()),
        },
        eligible_exposure,
        primary,
        who,
    )


def _record_order_treatments(
    regimens: pd.DataFrame,
    condition_index: pd.Index[Any],
    *,
    skip_empty: bool,
) -> pd.Series[Any]:
    frame = regimens[["condition_id", "drug_set"]].copy()
    if skip_empty:
        frame = frame.loc[frame["drug_set"].map(len).gt(0)]
    first = frame.groupby("condition_id", sort=False)["drug_set"].first()
    return pd.Series(
        first.reindex(condition_index).map(
            lambda value: value if isinstance(value, frozenset | set) else frozenset()
        ),
        index=condition_index,
        name="record_order_regimen",
    )


def _modified_treatments(data: RawValidationData, treatments: pd.Series[Any]) -> RawValidationData:
    aligned = treatments.reindex(data.context.index)
    if aligned.isna().any():
        raise ValueError("Modified treatment policy did not cover the locked cohort")
    return RawValidationData(
        context=data.context,
        treatments=aligned,
        outcome=data.outcome,
        patient_groups=data.patient_groups,
        numeric_columns=data.numeric_columns,
        categorical_columns=data.categorical_columns,
    )


def _treatment_policy_audit(
    regimens: pd.DataFrame,
    initial: pd.DataFrame,
    base: RawValidationData,
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    assessed = base.context.index
    first_row = _record_order_treatments(regimens, assessed, skip_empty=False)
    first_nonempty_record = _record_order_treatments(regimens, assessed, skip_empty=True)
    current = base.treatments.reindex(assessed)

    raw_empty_first = first_row.map(len).eq(0)
    record_nonempty_differs = pd.Series(
        [first_nonempty_record.loc[key] != current.loc[key] for key in assessed],
        index=assessed,
    )
    first_row_differs = pd.Series(
        [first_row.loc[key] != current.loc[key] for key in assessed],
        index=assessed,
    )

    preprocessor = FoldPreprocessor(
        base.numeric_columns,
        base.categorical_columns,
        min_treatment_prevalence=0.01,
    )
    full = preprocessor.fit_transform(base.context, base.treatments)
    encoded_all_zero = np.isclose(full.X.sum(axis=1), 0.0)

    seed = LockedValidationConfig().seed
    policy_data = {
        "chronology_first_nonempty": base,
        "record_order_first_nonempty": _modified_treatments(base, first_nonempty_record),
        "record_order_first_row_including_empty": _modified_treatments(base, first_row),
    }
    runs = {
        name: _run_grouped_logistic(data, "additive", seed=seed)
        for name, data in policy_data.items()
    }
    y = np.asarray(base.outcome, dtype=np.int64)
    probabilities = {
        "additive": runs["chronology_first_nonempty"].probability,
        "record_order_first_nonempty": runs["record_order_first_nonempty"].probability,
        "record_order_first_row_including_empty": runs[
            "record_order_first_row_including_empty"
        ].probability,
    }
    uncertainty = _cluster_bootstrap_metrics(
        y,
        probabilities,
        base.patient_groups.to_numpy(dtype=str),
        n_resamples=bootstrap_resamples,
        seed=seed + 9017,
    )
    return {
        "authoritative_policy": "earliest validated nonempty regimen by chronology",
        "authoritative_selector_report": {
            key: _safe_count(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else value
            for key, value in initial.attrs.get("selection_report", {}).items()
        },
        "locked_cohort_diagnostics": {
            "conditions": int(len(assessed)),
            "raw_selected_regimen_empty": _safe_count(int(current.map(len).eq(0).sum())),
            "encoded_vector_all_zero_after_one_percent_filter": _safe_count(
                int(encoded_all_zero.sum())
            ),
            "record_order_first_row_empty": _safe_count(int(raw_empty_first.sum())),
            "record_order_first_row_differs_from_authoritative": _safe_count(
                int(first_row_differs.sum())
            ),
            "record_order_first_nonempty_differs_from_authoritative": _safe_count(
                int(record_nonempty_differs.sum())
            ),
        },
        "prediction_sensitivity": {
            name: _model_summary(policy_data[name], run) for name, run in runs.items()
        },
        "patient_clustered_uncertainty": uncertainty,
        "interpretation": (
            "The primary policy already implements the requested first-nonempty-line repair. "
            "The record-order variants are diagnostics, not proposed clinical episode definitions."
        ),
    }


def _outcome_audit(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    eligible_exposure: pd.Series[bool],
    primary: pd.Series[bool],
    who: pd.Series[bool],
) -> dict[str, Any]:
    condition_index = pd.Index(patient_cases["condition_id"], name="condition_id")
    normalized = _normalize_outcome(patient_cases["outcome"])
    normalized.index = condition_index
    masks = {
        "all_patient_case_conditions": pd.Series(True, index=condition_index),
        "eligible_exposure_and_resistance": eligible_exposure,
        "primary_died_failure_cohort": primary,
        "who_unfavorable_assessable_cohort": who,
    }

    regimen_endpoint = derive_died_failure(regimens)["died_failure"].reindex(condition_index)
    patient_event = normalized.isin({"died", "failure"})
    both_assessable = primary & regimen_endpoint.notna()
    concordant = both_assessable & regimen_endpoint.astype("boolean").eq(
        patient_event.astype("boolean")
    )
    discordant = both_assessable & ~concordant

    regimen_codes = regimens.assign(_outcome=_normalize_outcome(regimens["outcome_cd"]))
    grouped_codes = regimen_codes.groupby("condition_id", sort=True)["_outcome"].agg(
        lambda values: frozenset(value for value in values if value != "missing_or_unclassified")
    )
    grouped_codes = grouped_codes.reindex(condition_index).map(
        lambda value: value if isinstance(value, frozenset) else frozenset()
    )
    terminal_event = {"died", "failure"}
    terminal_nonevent = {"cured", "completed"}
    other_state = {"lost_to_follow_up", "palliative_care", "still_on_treatment", "unknown"}
    mixed_event_nonevent = grouped_codes.map(
        lambda values: bool(values & terminal_event) and bool(values & terminal_nonevent)
    )
    event_plus_other = grouped_codes.map(
        lambda values: bool(values & terminal_event) and bool(values & other_state)
    )
    multiple_states = grouped_codes.map(len).gt(1)

    return {
        "category_counts": {
            name: _category_counts(normalized, mask) for name, mask in masks.items()
        },
        "primary_endpoint": {
            "definition": "recorded final Patient Cases outcome; Died or Failure versus Cured or Completed",
            "conditions": int(primary.sum()),
            "died": int((primary & normalized.eq("died")).sum()),
            "failure": int((primary & normalized.eq("failure")).sum()),
            "cured": int((primary & normalized.eq("cured")).sum()),
            "completed": int((primary & normalized.eq("completed")).sum()),
        },
        "who_sensitivity": {
            "definition": "primary recorded terminal states plus loss to follow-up as unfavorable",
            "conditions": int(who.sum()),
            "events": int((who & normalized.isin({"died", "failure", "lost_to_follow_up"})).sum()),
            "loss_to_follow_up": int((who & normalized.eq("lost_to_follow_up")).sum()),
        },
        "regimen_history_concordance": {
            "both_sources_assessable_in_primary": int(both_assessable.sum()),
            "concordant": int(concordant.sum()),
            "discordant": _safe_count(int(discordant.sum())),
            "primary_without_assessable_regimen_history_endpoint": _safe_count(
                int((primary & regimen_endpoint.isna()).sum())
            ),
        },
        "regimen_line_state_conflicts_in_exposure_eligible_conditions": {
            "multiple_distinct_recorded_states": _safe_count(
                int((eligible_exposure & multiple_states).sum())
            ),
            "terminal_event_and_terminal_nonevent": _safe_count(
                int((eligible_exposure & mixed_event_nonevent).sum())
            ),
            "terminal_event_and_other_state": _safe_count(
                int((eligible_exposure & event_plus_other).sum())
            ),
        },
        "endpoint_time_status": (
            "Patient Cases supplies a final outcome state but no outcome timestamp. Temporal "
            "precedence of the recorded final state cannot be verified from that table alone."
        ),
    }


def _relative_event_report(
    frame: pd.DataFrame,
    index_starts: pd.Series[Any],
    *,
    date_column: str,
    condition_mask: pd.Index[Any],
) -> dict[str, Any]:
    selected = frame.loc[
        frame["condition_id"].isin(condition_mask), ["condition_id", date_column]
    ].copy()
    selected["_event"] = pd.to_numeric(selected[date_column], errors="coerce")
    selected["_index"] = pd.to_numeric(selected["condition_id"].map(index_starts), errors="coerce")
    valid = selected["_event"].notna() & selected["_index"].notna()
    delta = selected.loc[valid, "_event"] - selected.loc[valid, "_index"]
    return {
        "rows": int(len(selected)),
        "rows_with_comparable_event_and_index_time": int(valid.sum()),
        "rows_on_or_before_index": int((delta <= 0).sum()),
        "rows_after_index": int((delta > 0).sum()),
        "conditions_with_at_least_one_on_or_before_index_row": int(
            selected.loc[
                valid & ((selected["_event"] - selected["_index"]) <= 0), "condition_id"
            ].nunique()
        ),
        "temporal_coordinate": "signed day offset",
    }


def _timing_audit(
    patient_cases: pd.DataFrame,
    biochemistry: pd.DataFrame,
    dst: pd.DataFrame,
    initial: pd.DataFrame,
    primary_ids: pd.Index[Any],
) -> dict[str, Any]:
    starts = initial["initial_start"].reindex(primary_ids)
    numeric_starts = pd.to_numeric(starts, errors="coerce")
    if numeric_starts.isna().any() or not np.isfinite(numeric_starts).all():
        raise ValueError("Primary cohort contains invalid initial-regimen time zero")

    context, spec = build_context_matrix(
        patient_cases,
        biochemistry,
        primary_ids.tolist(),
        baseline_index_dates=starts,
        preserve_degenerate_columns=True,
    )
    lab_features = [name for name in spec.numeric if name.startswith("lab_")]
    untimed_features = [name for name in spec.all_columns if name not in lab_features]
    baseline_positive = _strict_baseline_positive_ids(dst)
    primary_set = set(primary_ids.tolist())

    return {
        "index_regimen": {
            "conditions": int(len(primary_ids)),
            "time_zero_equal_to_zero": int(
                np.isclose(numeric_starts.to_numpy(dtype=float), 0.0).sum()
            ),
            "time_zero_before_zero": _safe_count(int((numeric_starts < 0).sum())),
            "time_zero_after_zero": _safe_count(int((numeric_starts > 0).sum())),
            "coordinate": initial.attrs.get("selection_report", {}).get(
                "temporal_coordinate", "unknown"
            ),
        },
        "biochemistry": _relative_event_report(
            biochemistry,
            initial["initial_start"],
            date_column="test_date",
            condition_mask=primary_ids,
        ),
        "dst_specimens": _relative_event_report(
            dst,
            initial["initial_start"],
            date_column="specimen_collection_date",
            condition_mask=primary_ids,
        ),
        "culture_positive_eligibility": {
            "primary_with_positive_culture_on_or_before_index": int(
                len(primary_set.intersection(baseline_positive))
            ),
            "primary_without_verified_positive_culture_on_or_before_index": int(
                len(primary_set.difference(baseline_positive))
            ),
        },
        "context_feature_provenance": {
            "raw_context_columns": int(context.shape[1]),
            "timestamp_constrained_laboratory_columns": len(lab_features),
            "condition_snapshot_columns_without_field_specific_timestamp": len(untimed_features),
            "groups": [
                {
                    "group": "demographic_socioeconomic_comorbidity",
                    "source": "Patient Cases",
                    "timing": "condition-level snapshot; field-specific time unavailable",
                },
                {
                    "group": "resistance_clinical_imaging_genomic",
                    "source": "Patient Cases",
                    "timing": "condition-level snapshot; field-specific time unavailable",
                },
                {
                    "group": "per_drug_bactec_dst",
                    "source": "Patient Cases",
                    "timing": "condition-level snapshot; test time unavailable in this table",
                },
                {
                    "group": "biochemistry_laboratory",
                    "source": "Biochemistry",
                    "timing": "test_date required on or before selected initial-regimen index",
                },
                {
                    "group": "recorded_final_outcome",
                    "source": "Patient Cases",
                    "timing": "final state supplied without outcome timestamp",
                },
            ],
        },
        "adjudication": (
            "The export can enforce regimen and laboratory chronology and a strict baseline-culture "
            "sensitivity. It cannot prove that untimed Patient Cases fields preceded treatment or "
            "assign an exact time to the recorded final outcome."
        ),
    }


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    dst = load_dst(raw_dir)
    initial = select_initial_regimens(regimens)

    cohort_flow, eligible_exposure, primary_mask, who_mask = _cohort_flow(patient_cases, initial)
    primary_ids = pd.Index(
        patient_cases.loc[primary_mask.to_numpy(), "condition_id"], name="condition_id"
    )
    base = _build_raw_data(
        patient_cases,
        biochemistry,
        initial,
        endpoint="died_failure",
        restrict_to=set(primary_ids.tolist()),
    )
    if not base.context.index.equals(primary_ids):
        raise ValueError("Episode audit cohort order differs from locked raw-data builder")

    source_hashes = {
        table: _sha256(raw_dir / filename) for table, filename in TB_PORTALS_FILES.items()
    }
    report = {
        "report_schema_version": 1,
        "analysis_id": "patch_episode_endpoint_audit",
        "analysis_status": "analysis-only; manuscript and public artifacts untouched",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "governance": {
            "dua_acknowledged": bool(args.dua_acknowledge),
            "contains_identifiers": False,
            "contains_row_level_data": False,
            "small_cell_minimum": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
        "source_sha256": source_hashes,
        "cohort_flow": cohort_flow,
        "treatment_episode": _treatment_policy_audit(
            regimens,
            initial,
            base,
            bootstrap_resamples=args.bootstrap_resamples,
        ),
        "endpoint": _outcome_audit(
            patient_cases,
            regimens,
            eligible_exposure,
            primary_mask,
            who_mask,
        ),
        "temporal_provenance": _timing_audit(
            patient_cases,
            biochemistry,
            dst,
            initial,
            primary_ids,
        ),
        "decision": {
            "time_zero_contract_supported": "partial",
            "supported": [
                "earliest validated nonempty recorded regimen can be selected reproducibly",
                "dated biochemistry can be restricted to on or before the selected regimen index",
                "culture-positive eligibility can be restricted to on or before the index in sensitivity analysis",
            ],
            "not_supported_by_export": [
                "field-specific timing for most Patient Cases context variables",
                "an exact timestamp for the recorded final Patient Cases outcome",
                "counterfactual or causal interpretation of drug coefficients",
            ],
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
