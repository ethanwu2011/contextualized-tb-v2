"""Rerun the locked residual PATCH analysis without touching manuscript or release artifacts.

Row-level predictions are written only below an explicit private output root outside the workspace.
The requested aggregate JSON contains no identifiers or row-level fields and may be stored in the
analysis-only directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

# Reject an explicitly repository-local private-output request before importing scientific
# runtimes. This safety preflight also works where OpenMP cannot initialize (for example,
# a restricted smoke-test sandbox). Valid scientific execution and its later guards are unchanged.
if __name__ == "__main__" and "--private-output-root" in sys.argv:
    _position = sys.argv.index("--private-output-root") + 1
    if _position < len(sys.argv):
        _requested = Path(sys.argv[_position]).expanduser().resolve()
        _repository = Path(__file__).resolve().parents[1]
        if _requested == _repository or _requested.is_relative_to(_repository):
            print("Private row-level output must be outside the repository", file=sys.stderr)
            raise SystemExit(2)

import numpy as np
import pandas as pd
import sklearn
import torch

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.locked_validation import (
    LockedValidationConfig,
    _assert_release_safe_report,
    build_schema6_raw_validation_data,
    prepare_dua_output_root,
    run_locked_validation,
    write_validation_outputs,
)
from ctb.utils.reproducibility import set_global_seed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--private-output-root", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(p, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for bin_number, indices in enumerate(np.array_split(order, n_bins), start=1):
        if len(indices) < 30:
            raise ValueError("Calibration bin violates small-cell policy")
        rows.append(
            {
                "bin": bin_number,
                "n": int(len(indices)),
                "mean_predicted": float(np.mean(p[indices])),
                "observed_rate": float(np.mean(y[indices])),
            }
        )
    return rows


def _metric_estimates(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics = report["metrics"]
    return {
        model: {
            metric: float(values["estimate"])
            for metric, values in metrics[model].items()
            if isinstance(values, dict) and "estimate" in values
        }
        for model in ("additive_lr", "residual_patch", "paired_difference_patch_minus_additive")
    }


def _reference_comparison(
    current: dict[str, Any],
    reference_path: Path | None,
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    current_values = _metric_estimates(current)
    reference_values = _metric_estimates(reference)
    differences = {
        model: {
            metric: current_values[model][metric] - reference_values[model][metric]
            for metric in current_values[model]
        }
        for model in current_values
    }
    maximum_absolute = max(abs(value) for model in differences.values() for value in model.values())
    return {
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _hash_file(reference_path),
        "metric_estimate_differences_current_minus_reference": differences,
        "maximum_absolute_metric_estimate_difference": float(maximum_absolute),
        "matches_to_1e_12": bool(maximum_absolute <= 1e-12),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _assert_release_safe_report(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw TB Portals directory does not exist")
    aggregate_output = args.aggregate_output.expanduser().resolve()
    private_requested = args.private_output_root.expanduser().resolve()
    source_repository = Path(__file__).resolve().parents[1]
    analysis_workspace = aggregate_output.parents[1]
    protected_roots = (source_repository, analysis_workspace, raw_dir)
    if any(
        private_requested == protected
        or private_requested.is_relative_to(protected)
        or protected.is_relative_to(private_requested)
        for protected in protected_roots
    ):
        raise ValueError(
            "Private row-level output must be outside the repository, analysis workspace, and raw data"
        )
    if aggregate_output.is_relative_to(source_repository):
        raise ValueError("Aggregate rerun output must remain outside the source repository")
    if aggregate_output.exists():
        raise FileExistsError("Aggregate rerun output already exists; refusing to overwrite it")
    private_root = prepare_dua_output_root(private_requested, source_repository)

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    config = LockedValidationConfig()
    set_global_seed(config.seed)
    patient_cases = clean_patient_cases(load_patient_cases(raw_dir))
    regimens = clean_regimens(load_regimens(raw_dir))
    biochemistry = load_biochemistry(raw_dir)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    result = run_locked_validation(data, config)
    write_validation_outputs(result, private_root)

    source_root = Path(__file__).resolve().parents[1] / "src"
    source_files = [
        source_root / "ctb" / "models" / "locked_validation.py",
        source_root / "ctb" / "models" / "residual_archetype.py",
        source_root / "ctb" / "models" / "fold_preprocessing.py",
        source_root / "ctb" / "data" / "treatments.py",
        source_root / "ctb" / "data" / "context.py",
    ]
    report = dict(result.aggregate_report)
    report["analysis_status"] = "analysis-only rerun; manuscript and submission artifacts untouched"
    report["runtime"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    report["source_sha256"] = {
        str(path.relative_to(source_root)): _hash_file(path) for path in source_files
    }
    report["calibration_bins"] = {
        "additive_lr": _calibration_bins(result.y_true, result.additive_probability),
        "residual_patch": _calibration_bins(result.y_true, result.patch_probability),
    }
    comparison = _reference_comparison(report, args.reference_json)
    if comparison is not None:
        report["reference_reproduction"] = comparison
    report["governance"] = {
        **report["governance"],
        "dua_acknowledged": bool(args.dua_acknowledge),
        "submission_artifacts_modified": False,
        "row_level_output_root": "private external path retained by local operator",
    }
    _atomic_json(aggregate_output, report)

    private_aggregate = private_root / "aggregate_only" / "locked_nested_validation.json"
    if not private_aggregate.exists():
        raise RuntimeError("Governed writer did not create its aggregate report")
    shutil.copyfile(private_aggregate, private_root / "dua_private" / "aggregate_report_copy.json")
    os.chmod(private_root / "dua_private" / "aggregate_report_copy.json", 0o600)
    print(
        json.dumps(
            {
                "status": "complete",
                "aggregate_output": str(aggregate_output),
                "private_output_retained": True,
                "submission_artifacts_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
