#!/usr/bin/env bash
# Build the manuscript endpoint and the core source artifacts from approved raw data.
#
# This entry point deliberately stops before the optional, compute-heavy figure and bootstrap
# scripts. Run those numbered scripts from the order documented in docs/SUPPLEMENTARY_FIGURES.md.
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">>> [0/2] Verifying raw data is reachable"
uv run --locked python scripts/00_download_data.py

echo ">>> [1/2] Building analysis dataset (raw -> processed)"
uv run --locked python scripts/01_build_dataset.py

echo ">>> [2/2] Building the endpoint-explicit PATCH matrix"
uv run --locked python scripts/05c_build_unfavorable_matrices.py

echo ">>> Done. Run the numbered manuscript analyses after reviewing the endpoint manifest."
