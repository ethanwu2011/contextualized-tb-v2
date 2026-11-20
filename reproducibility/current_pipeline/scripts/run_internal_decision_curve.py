"""Compute an aggregate internal decision-curve sensitivity from private locked OOF predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SMALL_CELL_MIN = 30


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--private-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    return args


def _net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    positive = p >= threshold
    true_positive = int(np.sum(positive & (y == 1)))
    false_positive = int(np.sum(positive & (y == 0)))
    return float(true_positive / len(y) - false_positive / len(y) * threshold / (1.0 - threshold))


def _safe_count(value: int) -> int | dict[str, Any]:
    return int(value) if value >= SMALL_CELL_MIN else {"suppressed": True, "threshold": 30}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    private_path = args.private_predictions.expanduser().resolve()
    if not private_path.is_file():
        raise FileNotFoundError("Private OOF prediction artifact does not exist")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("Aggregate decision-curve output already exists")
    with np.load(private_path, allow_pickle=False) as artifact:
        y = np.asarray(artifact["outcome"], dtype=np.int64)
        raw_groups = np.asarray(artifact["patient_id"])
        additive = np.asarray(artifact["additive_probability"], dtype=np.float64)
        patch = np.asarray(artifact["patch_probability"], dtype=np.float64)
    if any(array.shape != (len(y),) for array in (raw_groups, additive, patch)):
        raise ValueError("Private prediction arrays are not aligned")
    groups = raw_groups.astype(str)
    normalized_groups = np.char.strip(groups)
    missing_group_tokens = {"", "nan", "none", "<na>", "nat"}
    if any(value.casefold() in missing_group_tokens for value in normalized_groups.tolist()):
        raise ValueError("Patient-group identifiers contain missing or blank values")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("Outcome is not binary")
    if not np.isfinite(additive).all() or not np.isfinite(patch).all():
        raise ValueError("Predictions contain non-finite values")
    if ((additive < 0) | (additive > 1)).any() or ((patch < 0) | (patch > 1)).any():
        raise ValueError("Predictions must lie in the closed interval [0, 1]")

    thresholds = np.arange(0.05, 0.401, 0.025)
    unique_groups = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(20260827)
    bootstrap_differences: dict[float, list[float]] = {
        float(threshold): [] for threshold in thresholds
    }
    valid = 0
    for _ in range(args.bootstrap_resamples):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        rows = np.concatenate([rows_by_group[index] for index in sampled])
        if len(np.unique(y[rows])) != 2:
            continue
        valid += 1
        for threshold in thresholds:
            difference = _net_benefit(y[rows], patch[rows], float(threshold)) - _net_benefit(
                y[rows], additive[rows], float(threshold)
            )
            bootstrap_differences[float(threshold)].append(difference)
    if valid < max(100, args.bootstrap_resamples // 2):
        raise RuntimeError("Too few valid patient-cluster bootstrap samples")

    prevalence = float(np.mean(y))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        additive_benefit = _net_benefit(y, additive, threshold_value)
        patch_benefit = _net_benefit(y, patch, threshold_value)
        differences = np.asarray(bootstrap_differences[threshold_value], dtype=float)
        additive_positive = int(np.sum(additive >= threshold_value))
        patch_positive = int(np.sum(patch >= threshold_value))
        rows.append(
            {
                "threshold": threshold_value,
                "treat_none_net_benefit": 0.0,
                "treat_all_net_benefit": float(
                    prevalence - (1.0 - prevalence) * threshold_value / (1.0 - threshold_value)
                ),
                "additive_net_benefit": additive_benefit,
                "patch_net_benefit": patch_benefit,
                "patch_minus_additive_net_benefit": patch_benefit - additive_benefit,
                "patch_minus_additive_ci95": [
                    float(np.percentile(differences, 2.5)),
                    float(np.percentile(differences, 97.5)),
                ],
                "additive_flagged_n": _safe_count(additive_positive),
                "patch_flagged_n": _safe_count(patch_positive),
            }
        )
    report = {
        "report_schema_version": 1,
        "analysis_id": "internal_locked_oof_decision_curve_sensitivity",
        "analysis_status": "analysis-only; manuscript and submission artifacts untouched",
        "interpretation": "internal prognostic decision-curve sensitivity, not external utility validation",
        "cohort": {
            "conditions": int(len(y)),
            "patients": int(len(unique_groups)),
            "events": int(y.sum()),
            "observed_non_events": int(len(y) - y.sum()),
        },
        "bootstrap": {
            "unit": "patient",
            "requested_resamples": args.bootstrap_resamples,
            "valid_resamples": valid,
            "seed": 20260827,
        },
        "threshold_rows": rows,
        "governance": {
            "dua_acknowledged": bool(args.dua_acknowledge),
            "contains_identifiers": False,
            "contains_row_level_data": False,
            "small_cell_minimum": SMALL_CELL_MIN,
            "submission_artifacts_modified": False,
        },
        "limitations": [
            "Thresholds are exploratory and were not clinically prespecified.",
            "This is internal out-of-fold evaluation and does not validate utility in Brazil or another external cohort.",
            "Net benefit depends on calibration and the chosen outcome definition.",
        ],
    }
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "aggregate_output": str(output_path),
                "submission_artifacts_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
