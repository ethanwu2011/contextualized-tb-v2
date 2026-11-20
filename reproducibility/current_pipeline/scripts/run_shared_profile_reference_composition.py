"""Describe the corrected endpoint-locked shared K=5 reference fit using safe aggregates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, train
from ctb.models.fold_preprocessing import FoldPreprocessor
from ctb.models.locked_validation import LockedValidationConfig, build_schema6_raw_validation_data
from run_aggregate_sensitivities import SMALL_CELL_MIN, _safe_count, _write_report
from run_shared_profile_full_pipeline_stability import PUBLISHED_CONFIG, REFERENCE_SEED, K


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe_binary(values: pd.Series[Any], selected: np.ndarray) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    known = selected & np.isfinite(numeric)
    positive = known & (numeric == 1)
    count = int(positive.sum())
    denominator = int(known.sum())
    both_supported = min(count, denominator - count) >= SMALL_CELL_MIN
    return {
        "known_count": _safe_count(denominator),
        "positive_count": (
            count if both_supported else {"suppressed": True, "threshold": SMALL_CELL_MIN}
        ),
        "percent_among_known": float(100 * count / denominator) if both_supported else None,
    }


def _safe_category(values: pd.Series[Any], selected: np.ndarray) -> dict[str, Any]:
    normalized = values.astype("string").fillna("Missing").to_numpy(dtype=str)
    denominator = int(selected.sum())
    counts = {
        str(level): int(np.sum(selected & (normalized == level)))
        for level in sorted(np.unique(normalized[selected]))
    }
    hidden = {level for level, count in counts.items() if count < SMALL_CELL_MIN}
    hidden_total = sum(counts[level] for level in hidden)
    # A visible total must not recover a single rare category by subtraction.
    if 0 < hidden_total < SMALL_CELL_MIN:
        visible = [level for level in counts if level not in hidden]
        if visible:
            hidden.add(min(visible, key=lambda level: (counts[level], level)))
    return {
        level: {
            "count": (
                {"suppressed": True, "threshold": SMALL_CELL_MIN} if level in hidden else count
            ),
            "percent": None if level in hidden else float(100 * count / denominator),
        }
        for level, count in counts.items()
    }


def _modality_of(name: str) -> str:
    normalized = name.casefold()
    if normalized.startswith(("type_of_resistance", "bactec_")):
        return "resistance_dst"
    if normalized.startswith("country"):
        return "region"
    if normalized.startswith(("main_lineage", "sub_lineage", "lineage")):
        return "genomics"
    if normalized.startswith("comorb_"):
        return "comorbidity"
    if normalized.startswith("lab_"):
        return "baseline_laboratory"
    if normalized.startswith(
        (
            "dissemination",
            "affect_pleura",
            "shadow_pattern",
            "timika",
            "imaging",
            "nodul",
        )
    ):
        return "radiology_imaging"
    if normalized.startswith(
        ("baseline_smear", "cavitation", "case_definition", "lung_localization")
    ):
        return "bacillary_burden_clinical"
    if normalized.startswith(("gender", "education", "employment")) or normalized in {
        "age_of_onset",
        "bmi",
        "number_of_children",
        "number_of_daily_contacts",
    }:
        return "sociodemographic"
    return "other_clinical"


def _modality_attribution(
    model: LinearArchetypeContextualized,
    context: np.ndarray,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    baseline = model.predict_loadings(context).argmax(axis=1)
    columns: dict[str, list[int]] = {}
    for position, name in enumerate(feature_names):
        columns.setdefault(_modality_of(name), []).append(position)
    rng = np.random.default_rng(REFERENCE_SEED)
    rows: list[dict[str, Any]] = []
    for modality, positions in columns.items():
        disruptions: list[float] = []
        for _ in range(5):
            permuted = context.copy()
            order = rng.permutation(len(context))
            permuted[:, positions] = context[order][:, positions]
            current = model.predict_loadings(permuted).argmax(axis=1)
            disruptions.append(1.0 - float(adjusted_rand_score(baseline, current)))
        rows.append(
            {
                "modality": modality,
                "encoded_features": len(positions),
                "mean_hard_assignment_disruption": float(np.mean(disruptions)),
            }
        )
    total = sum(float(row["mean_hard_assignment_disruption"]) for row in rows)
    per_feature_total = sum(
        float(row["mean_hard_assignment_disruption"]) / int(row["encoded_features"]) for row in rows
    )
    for row in rows:
        disruption = float(row["mean_hard_assignment_disruption"])
        feature_count = int(row["encoded_features"])
        row["total_share"] = disruption / total if total > 0 else 0.0
        row["per_feature_share"] = (
            (disruption / feature_count) / per_feature_total if per_feature_total > 0 else 0.0
        )
    return sorted(rows, key=lambda row: (-float(row["total_share"]), row["modality"]))


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    logging.getLogger("ctb").setLevel(logging.ERROR)
    config = LockedValidationConfig()
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    outcome = np.asarray(data.outcome, dtype=np.int64)
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=config.min_treatment_prevalence,
    )
    matrices = preprocessor.fit_transform(data.context, data.treatments)
    model = LinearArchetypeContextualized(
        matrices.C.shape[1],
        matrices.X.shape[1],
        n_archetypes=K,
        simplex=True,
        seed=REFERENCE_SEED,
    )
    model = train(
        model,
        matrices.C,
        matrices.X,
        outcome.astype(np.float64),
        PUBLISHED_CONFIG,
        seed=REFERENCE_SEED,
    )
    loadings = np.asarray(model.predict_loadings(matrices.C), dtype=np.float64)
    indexed = patient_cases.set_index("condition_id").reindex(data.context.index)
    hard = loadings.argmax(axis=1)
    soft_rates = np.sum(loadings * outcome[:, None], axis=0) / np.sum(loadings, axis=0)
    risk_order = np.argsort(soft_rates)
    rows: list[dict[str, Any]] = []
    for risk_rank, profile in enumerate(risk_order, start=1):
        selected = hard == profile
        hard_count = int(selected.sum())
        age = pd.to_numeric(data.context["age_of_onset"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        smear = data.context["baseline_smear"].astype("string")
        smear_positive = smear.str.casefold().eq("positive").astype(int)
        regimen_rows: list[dict[str, Any]] = []
        for component, feature in enumerate(matrices.treatment_feature_names):
            count = int(matrices.X[selected, component].sum())
            if count >= SMALL_CELL_MIN:
                regimen_rows.append(
                    {
                        "regimen_feature": feature,
                        "count": count,
                        "percent": float(100 * count / hard_count),
                    }
                )
        regimen_rows.sort(key=lambda row: (-row["percent"], row["regimen_feature"]))
        rows.append(
            {
                "risk_rank": risk_rank,
                "fitted_profile": int(profile + 1),
                "hard_count": _safe_count(hard_count),
                "hard_event_rate": float(outcome[selected].mean()),
                "soft_effective_count": float(loadings[:, profile].sum()),
                "soft_weighted_event_rate": float(soft_rates[profile]),
                "age_mean_among_known": float(np.nanmean(age[selected])),
                "hiv": _safe_binary(data.context["comorb_hiv"], selected),
                "smear_positive": _safe_binary(smear_positive, selected),
                "resistance_composition": _safe_category(indexed["type_of_resistance"], selected),
                "country_composition": _safe_category(data.context["country"], selected),
                "top_recorded_regimen_components": regimen_rows[:5],
            }
        )
    report = {
        "report_schema_version": 1,
        "analysis_id": "shared_patch_k5_corrected_reference_composition",
        "analysis_role": "analysis-only remediation; no submission artifact was modified",
        "cohort": {
            "assessed_conditions": len(outcome),
            "events": int(outcome.sum()),
            "retained_context_features": matrices.C.shape[1],
            "retained_regimen_features": matrices.X.shape[1],
        },
        "fit": {
            "k": K,
            "reference_seed": REFERENCE_SEED,
            "profile_order": "ascending soft-weighted recorded died-or-failure rate",
        },
        "profiles": rows,
        "modality_attribution": _modality_attribution(
            model, matrices.C, matrices.context_feature_names
        ),
        "interpretation": (
            "Descriptive composition of an outcome-supervised shared representation. Hard bins "
            "are display summaries, not biological classes or causal treatment-response groups."
        ),
        "governance": {
            "row_level_objects_serialized": False,
            "identifiers_serialized": False,
            "small_cell_threshold": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
    }
    _write_report(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
