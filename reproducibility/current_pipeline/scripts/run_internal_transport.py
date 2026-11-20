"""Run patient-disjoint temporal and held-out-country transport diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.data.temporal import extract_calendar_year, patient_disjoint_temporal_masks
from ctb.models.fold_preprocessing import FoldMatrices, FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    PatchCandidate,
    _additive_probability,
    _assert_release_safe_report,
    _fit_additive,
    _fit_residual,
    _metric_vector,
    _seed,
    _validate_probability,
    build_schema6_raw_validation_data,
    patient_clustered_paired_metrics,
)
from ctb.utils.reproducibility import set_global_seed

MIN_COUNTRY_TEST_N = 200
MIN_COUNTRY_CLASS_N = 30


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


def _safe_count(value: int) -> int | dict[str, Any]:
    count = int(value)
    return count if count >= 30 else {"suppressed": True, "threshold": 30}


def _fit_gbdt(
    matrices: FoldMatrices, y: np.ndarray, *, seed: int
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(np.hstack([matrices.C, matrices.X]), y)
    return model


def _gbdt_probability(model: HistGradientBoostingClassifier, matrices: FoldMatrices) -> np.ndarray:
    probability = np.asarray(
        model.predict_proba(np.hstack([matrices.C, matrices.X]))[:, 1],
        dtype=np.float64,
    )
    if probability.shape != (matrices.n_rows,) or not np.isfinite(probability).all():
        raise ValueError("GBDT returned invalid probabilities")
    return probability


def _fit_models(
    train: FoldMatrices,
    test: FoldMatrices,
    y_train: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    baseline = _fit_additive(train, y_train, c_value=0.1, max_iter=2000, seed=seed)
    additive = _additive_probability(baseline, test)
    candidate = PatchCandidate(
        name="residual_k5_strong_shrinkage",
        n_archetypes=5,
        loading_l2=1e-3,
        deviation_l2=1e-2,
        residual_scale=0.5,
        epochs=300,
        learning_rate=0.01,
    )
    patch_model = _fit_residual(baseline, train, y_train, candidate, seed=_seed(seed, 31))
    patch = _validate_probability(
        patch_model.predict_proba(test.C, test.X),
        n_rows=test.n_rows,
        model_name=candidate.name,
    )
    gbdt_model = _fit_gbdt(train, y_train, seed=_seed(seed, 9001))
    gbdt = _gbdt_probability(gbdt_model, test)
    return {"additive": additive, "patch_k5": patch, "nonlinear_gbdt": gbdt}


def _prepare(
    data: Any,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> tuple[FoldMatrices, FoldMatrices]:
    preprocessor = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=0.01,
    )
    train = preprocessor.fit_transform(
        data.context.iloc[train_index],
        data.treatments.iloc[train_index],
    )
    test = preprocessor.transform(
        data.context.iloc[test_index],
        data.treatments.iloc[test_index],
    )
    return train, test


def _rename(result: dict[str, Any], *, baseline: str, extension: str) -> dict[str, Any]:
    return {
        baseline: result["additive_lr"],
        extension: result["residual_patch"],
        f"paired_difference_{extension}_minus_{baseline}": result[
            "paired_difference_patch_minus_additive"
        ],
        "bootstrap": result["bootstrap"],
    }


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    _assert_release_safe_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    config = LockedValidationConfig()
    set_global_seed(config.seed)
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    y = np.asarray(data.outcome, dtype=np.int64)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    condition_index = data.context.index
    patient_case_index = patient_cases.drop_duplicates("condition_id").set_index("condition_id")

    years = extract_calendar_year(patient_case_index["registration_date"]).reindex(condition_index)
    patient_series = pd.Series(groups, index=condition_index, dtype="string")
    temporal_train, temporal_test, temporal_flow = patient_disjoint_temporal_masks(
        years,
        patient_series,
        train_through=2020,
        test_from=2021,
    )
    train_index = np.flatnonzero(temporal_train.to_numpy(dtype=bool))
    test_index = np.flatnonzero(temporal_test.to_numpy(dtype=bool))
    if len(train_index) < 100 or len(test_index) < 100:
        raise ValueError("Temporal transport split is too small")
    if len(np.unique(y[train_index])) != 2 or len(np.unique(y[test_index])) != 2:
        raise ValueError("Temporal transport split lacks both outcome classes")
    temporal_train_matrices, temporal_test_matrices = _prepare(data, train_index, test_index)
    temporal_predictions = _fit_models(
        temporal_train_matrices,
        temporal_test_matrices,
        y[train_index],
        seed=config.seed,
    )
    temporal_metrics = {
        name: _metric_vector(y[test_index], probability)
        for name, probability in temporal_predictions.items()
    }
    temporal_patch_vs_additive = patient_clustered_paired_metrics(
        y[test_index],
        temporal_predictions["additive"],
        temporal_predictions["patch_k5"],
        groups[test_index],
        n_resamples=args.bootstrap_resamples,
        seed=_seed(config.seed, 7001),
    )
    temporal_patch_vs_gbdt = patient_clustered_paired_metrics(
        y[test_index],
        temporal_predictions["nonlinear_gbdt"],
        temporal_predictions["patch_k5"],
        groups[test_index],
        n_resamples=args.bootstrap_resamples,
        seed=_seed(config.seed, 7002),
    )

    country = patient_case_index["country"].astype("string").reindex(condition_index)
    country_counts = country.value_counts(dropna=True)
    country_rows: list[dict[str, Any]] = []
    for country_name in sorted(
        str(value)
        for value, count in country_counts.items()
        if int(count) >= MIN_COUNTRY_TEST_N and str(value).strip() not in {"", "<NA>"}
    ):
        test_mask = country.eq(country_name).fillna(False).to_numpy(dtype=bool)
        test_country_index = np.flatnonzero(test_mask)
        if len(test_country_index) < MIN_COUNTRY_TEST_N:
            continue
        test_y = y[test_country_index]
        class_counts = np.bincount(test_y, minlength=2)
        if int(class_counts.min()) < MIN_COUNTRY_CLASS_N:
            continue
        held_out_patients = set(groups[test_country_index])
        train_mask = ~test_mask & ~np.isin(groups, list(held_out_patients))
        train_country_index = np.flatnonzero(train_mask)
        if len(np.unique(y[train_country_index])) != 2:
            continue
        train_matrices, test_matrices = _prepare(
            data,
            train_country_index,
            test_country_index,
        )
        predictions = _fit_models(
            train_matrices,
            test_matrices,
            y[train_country_index],
            seed=_seed(config.seed, len(country_rows), 8100),
        )
        country_rows.append(
            {
                "held_out_country": country_name,
                "test_conditions": int(len(test_country_index)),
                "test_patients": int(len(np.unique(groups[test_country_index]))),
                "test_events": int(test_y.sum()),
                "train_conditions": int(len(train_country_index)),
                "cross_country_patient_records_excluded_from_training": _safe_count(
                    int(np.sum(~test_mask & np.isin(groups, list(held_out_patients))))
                ),
                "metrics": {
                    name: _metric_vector(test_y, probability)
                    for name, probability in predictions.items()
                },
            }
        )
    if not country_rows:
        raise ValueError("No held-out country met the disclosure and class-count criteria")

    def country_metric_summary(model: str, metric: str) -> dict[str, float]:
        values = np.asarray(
            [row["metrics"][model][metric] for row in country_rows],
            dtype=np.float64,
        )
        return {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }

    report = {
        "report_schema_version": 1,
        "analysis_id": "patient_disjoint_internal_transport",
        "analysis_status": "analysis-only; manuscript and submission artifacts untouched",
        "endpoint": "recorded_final_patient_cases_died_or_failure",
        "temporal": {
            "definition": "train registration year through 2020; test registration year 2021 or later",
            "train_conditions": int(len(train_index)),
            "test_conditions": int(len(test_index)),
            "train_events": int(y[train_index].sum()),
            "test_events": int(y[test_index].sum()),
            "cross_period_patients_excluded": _safe_count(temporal_flow.cross_period_patients),
            "cross_period_records_excluded": _safe_count(
                temporal_flow.cross_period_records_excluded
            ),
            "missing_year_records_excluded": _safe_count(temporal_flow.missing_year_records),
            "metrics": temporal_metrics,
            "paired_metrics": {
                "patch_vs_additive": _rename(
                    temporal_patch_vs_additive,
                    baseline="additive",
                    extension="patch_k5",
                ),
                "patch_vs_nonlinear_gbdt": _rename(
                    temporal_patch_vs_gbdt,
                    baseline="nonlinear_gbdt",
                    extension="patch_k5",
                ),
            },
        },
        "held_out_country": {
            "minimum_test_conditions": MIN_COUNTRY_TEST_N,
            "minimum_events_and_non_events": MIN_COUNTRY_CLASS_N,
            "countries_assessed": int(len(country_rows)),
            "rows": country_rows,
            "summary": {
                model: {
                    metric: country_metric_summary(model, metric)
                    for metric in ("auroc", "auprc", "log_loss", "brier")
                }
                for model in ("additive", "patch_k5", "nonlinear_gbdt")
            },
        },
        "model_contract": {
            "additive_c": 0.1,
            "patch_k": 5,
            "patch_loading_l2": 1e-3,
            "patch_deviation_l2": 1e-2,
            "patch_residual_scale": 0.5,
            "gbdt_candidate": "shallow_15 selected most often in nested internal validation",
            "preprocessing": "fit on the transport training partition only",
            "patient_overlap": "excluded",
        },
        "governance": {
            "dua_acknowledged": bool(args.dua_acknowledge),
            "contains_identifiers": False,
            "contains_row_level_data": False,
            "minimum_reported_country_cell": MIN_COUNTRY_CLASS_N,
            "submission_artifacts_modified": False,
        },
        "interpretation": "internal prognostic transport; not external treatment-response validation",
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "status": "complete",
                "aggregate_output": str(args.output.expanduser().resolve()),
                "submission_artifacts_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
