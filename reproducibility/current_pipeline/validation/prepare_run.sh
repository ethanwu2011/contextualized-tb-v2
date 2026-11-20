#!/usr/bin/env bash
# Stage the locked validation driver in a fresh directory. The driver expects its source snapshot
# under ./source/src and writes aggregate JSON next to itself, so it never runs inside the repository.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -P)"
DEST="${1:?usage: prepare_run.sh <new-run-directory>}"
[ ! -e "$DEST" ] || { echo "REFUSED: $DEST already exists." >&2; exit 2; }
mkdir -p "$DEST/source"
cp "$HERE/run_validation.py" "$HERE/protocol_before_run.json" "$DEST/"
cp -R "$HERE/../src" "$DEST/source/src"
find "$DEST/source" -name '__pycache__' -type d -prune -exec rm -rf {} +
echo "Staged locked validation run at $DEST"
