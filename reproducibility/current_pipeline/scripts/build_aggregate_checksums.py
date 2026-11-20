"""Rebuild SHA-256 checksums for the frozen disclosure-safe aggregate JSON files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_DIR = ROOT / "manuscript" / "aggregate_results"
EXPECTED_JSON = {
    "architecture_composition_ablations.json",
    "brazil_source_rebuild_audit.json",
    "episode_endpoint_audit.json",
    "expanded_simulation_benchmark.json",
    "internal_decision_curve.json",
    "internal_transport.json",
    "sensitivity_and_support.json",
    "shared_k_sensitivity.json",
    "shared_profile_full_pipeline_stability.json",
    "shared_profile_reference_composition.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    json_files = sorted(AGGREGATE_DIR.glob("*.json"))
    observed = {path.name for path in json_files}
    if observed != EXPECTED_JSON:
        raise ValueError("Aggregate JSON file set differs from the reviewed release contract")
    rows = [f"{_sha256(path)}  {path.name}" for path in json_files]
    destination = AGGREGATE_DIR / "CHECKSUMS.sha256"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o644)
    temporary.replace(destination)
    print(f"Wrote {len(rows)} aggregate checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
