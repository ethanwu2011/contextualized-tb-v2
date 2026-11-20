"""Run an expanded synthetic residual-PATCH benchmark in the isolated analysis workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from ctb.models.residual_simulation import (
    SimulationBenchmarkConfig,
    run_simulation_benchmark,
    write_benchmark_artifacts,
)

SEEDS = (
    101,
    211,
    307,
    401,
    503,
    607,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output_root = args.output_root.expanduser().resolve()
    config = replace(SimulationBenchmarkConfig(), seeds=SEEDS)
    results, diagnostics = run_simulation_benchmark(config)
    paths = write_benchmark_artifacts(results, diagnostics, config, output_root)
    aggregate_output = None
    if args.aggregate_output is not None:
        aggregate_output = args.aggregate_output.expanduser().resolve()
        if aggregate_output.exists():
            raise FileExistsError("Aggregate benchmark output already exists")
        aggregate_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = aggregate_output.with_suffix(aggregate_output.suffix + ".tmp")
        shutil.copyfile(paths.summary_json, temporary)
        os.chmod(temporary, 0o600)
        temporary.replace(aggregate_output)
    report = {
        "status": "complete",
        "analysis_status": "analysis-only; manuscript and submission artifacts untouched",
        "interpretation": "synthetic predictive coefficient-surface recovery, not causal treatment-effect validation",
        "seeds": len(SEEDS),
        "replicate_rows": int(len(results)),
        "synthetic_data_only": True,
        "artifacts": {
            "replicate_csv": str(paths.replicate_csv),
            "summary_json": str(paths.summary_json),
            "figure_png": str(paths.figure_png),
            "figure_pdf": str(paths.figure_pdf),
        },
        "submission_artifacts_modified": False,
    }
    if aggregate_output is not None:
        report["aggregate_output"] = str(aggregate_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
