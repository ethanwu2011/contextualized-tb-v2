"""Verify a PATCH reviewer snapshot without reading any clinical input file."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_CELL = 30
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".feather",
    ".arrow",
    ".orc",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".h5",
    ".hdf5",
    ".db",
    ".dbf",
    ".dbc",
    ".sqlite",
    ".sqlite3",
    ".xlsx",
    ".xls",
    ".rds",
    ".dta",
    ".sav",
    ".ipynb",
}
COUNT_FIELDS = {
    "count",
    "hard_count",
    "events",
    "non_events",
    "patients",
    "conditions",
    "n_patients",
    "n_conditions",
    "n_events",
    "observed_non_events",
    "assessed_conditions",
    "exposed_count",
    "unexposed_count",
    "n",
}
ROW_VECTOR_FIELDS = {
    "patient_ids",
    "condition_ids",
    "cond_ids",
    "labels",
    "predictions",
    "patient_predictions",
    "hidden_states",
    "embeddings",
}


def aggregate_issues(obj: object) -> set[str]:
    """Return schema-level issue classes; never return offending values."""
    issues: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).lower()
            if (
                normalized in ROW_VECTOR_FIELDS
                and isinstance(value, list)
                and any(not isinstance(entry, dict) for entry in value)
            ):
                issues.add("potential_row_vector")
            if (
                normalized in COUNT_FIELDS
                and isinstance(value, int | float)
                and not isinstance(value, bool)
                and 0 <= value < MIN_CELL
            ):
                issues.add("potential_unsuppressed_small_count")
            issues.update(aggregate_issues(value))
    elif isinstance(obj, list):
        if len(obj) > 1000:
            issues.add("long_array_requires_review")
        for entry in obj:
            issues.update(aggregate_issues(entry))
    return issues


def verify(root: Path) -> dict[str, object]:
    """Check the exact file inventory and aggregate schema gate."""
    failures: list[dict[str, object]] = []
    manifest = root / "PUBLIC_FILE_MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if re.fullmatch(r"[0-9a-f]{64}  [A-Za-z0-9._/-]+", line) is None:
            failures.append({"file": manifest.name, "issue_classes": ["invalid_manifest"]})
            continue
        digest, relative = line.split("  ", 1)
        if relative.startswith("/") or ".." in Path(relative).parts or relative in expected:
            failures.append({"file": manifest.name, "issue_classes": ["unsafe_manifest_path"]})
            continue
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(expected):
        failures.append({"file": manifest.name, "issue_classes": ["file_inventory_mismatch"]})
    for relative in sorted(actual):
        path = root / relative
        issues: set[str] = set()
        if path.is_symlink():
            issues.add("symlink")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.add("excluded_data_or_model_file")
        if expected.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
            issues.add("checksum_mismatch")
        if issues:
            failures.append({"file": relative, "issue_classes": sorted(issues)})
    aggregate = root / "manuscript/aggregate_results"
    for path in sorted(aggregate.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        issues = aggregate_issues(payload)
        if path.name == "shared_geometry_aggregate.json":
            required_tile_fields = {
                "xbin",
                "ybin",
                "conditions",
                "profile",
                "resistance",
                "region",
                "hiv_prevalence",
                "smear_prevalence",
            }
            tiles = payload.get("tiles")
            if not isinstance(tiles, list) or not tiles:
                issues.add("missing_geometry_tiles")
            elif any(not required_tile_fields.issubset(tile) for tile in tiles):
                issues.add("missing_required_geometry_fields")
            if payload.get("embedding", {}).get("coordinates_serialized") is not False:
                issues.add("geometry_coordinate_boundary_unverified")
        if issues:
            failures.append({"file": path.name, "issue_classes": sorted(issues)})
    return {
        "status": "failed" if failures else "passed",
        "scope": "exact source files and frozen aggregate JSON; no clinical inputs opened",
        "files_checked": len(actual) + 1,
        "small_cell_threshold": MIN_CELL,
        "private_values_printed": False,
        "limitations": "Schema checks supplement the reviewed allowlist; they do not prove anonymization.",
        "failures": failures,
    }


def main() -> int:
    """Run the local verification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    report = verify(args.snapshot.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
