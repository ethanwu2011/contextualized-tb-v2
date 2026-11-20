#!/usr/bin/env bash
# Materialise a source-only PUBLIC PATCH tree from an explicit allowlist.
#
# The staged tree contains code, tests, CI, and documentation. It deliberately omits the entire
# data/ and results/ trees. Static figures may be added only after a separate disclosure review and
# an explicit release marker is created. Run scan_public.sh on the staged tree before publishing.
#
# Usage
#   bash scripts/sync_to_public.sh [TARGET_DIR]
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST_INPUT="${1:-$SRC/../patch-public}"
DEST_PARENT="$(cd "$(dirname "$DEST_INPUT")" && pwd)"
DEST="$DEST_PARENT/$(basename "$DEST_INPUT")"

if [ "$DEST" = "/" ] || [ "$DEST" = "/tmp" ] || [ "$DEST" = "/private/tmp" ] || \
   [ "$DEST" = "$SRC" ] || [[ "$DEST" == "$SRC"/* ]]; then
  echo "Refusing to stage into the source tree or a broad root: $DEST" >&2
  exit 2
fi

if [ -L "$DEST" ]; then
  echo "Refusing to stage through a symlink: $DEST" >&2
  exit 2
fi
if [ -e "$DEST" ]; then
  if [ ! -d "$DEST" ] || [ -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Refusing to clear a non-empty existing target. Use a new dedicated directory: $DEST" >&2
    exit 2
  fi
else
  mkdir -p "$DEST"
fi

echo ">>> Source: $SRC"
echo ">>> Target: $DEST"

# Directories published in full. These paths contain source code only. The allowlist intentionally
# excludes data/, results/, notebooks/, logs/, models/, and local review material.
ALLOW_DIRS=(
  "src"
  "scripts"
  "figures"
  "tests"
  ".github"
  "reproducibility"
)

ALLOW_FILES=(
  "README.md"
  "LICENSE"
  "CITATION.cff"
  ".gitignore"
  ".pre-commit-config.yaml"
  "pyproject.toml"
  "uv.lock"
  "Makefile"
  "PUBLIC_RELEASE_MANIFEST.json"
  "docs/PUBLIC_REPRODUCIBILITY.md"
  "docs/PATCH_OUTCOME_DEFINITION.md"
  "docs/DATA_DICTIONARY.md"
  "docs/OUTCOME_DEFINITION.md"
  "docs/REPRODUCIBILITY.md"
  "docs/SUPPLEMENTARY_FIGURES.md"
  "docs/EVIDENCE_LEDGER.md"
)

for directory in "${ALLOW_DIRS[@]}"; do
  if [ -d "$SRC/$directory" ]; then
    mkdir -p "$DEST/$directory"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='.venv' \
      --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' --exclude='.coverage' \
      "$SRC/$directory/" "$DEST/$directory/"
  fi
done

for file in "${ALLOW_FILES[@]}"; do
  if [ -f "$SRC/$file" ]; then
    mkdir -p "$DEST/$(dirname "$file")"
    cp "$SRC/$file" "$DEST/$file"
  fi
done

# Fail closed on data-shaped files even if an allowlisted source directory changes later.
find "$DEST" -type f \( \
  -name '*.parquet' -o -name '*.npz' -o -name '*.feather' -o -name '*.pkl' -o -name '*.pickle' \
\) -delete
find "$DEST" -name '.DS_Store' -delete

# Fail closed on rendered/static artifacts. They can contain derived values or embedded metadata and
# therefore require a separate disclosure review. Keep the public tree source-only by construction.
STATIC_HITS="$(find "$DEST/figures" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.pdf' -o -name '*.html' -o -name '*.svg' \) -print 2>/dev/null)"
if [ -n "$STATIC_HITS" ]; then
  echo "Refusing to stage rendered/static figure artifacts without a separate disclosure review:" >&2
  echo "$STATIC_HITS" >&2
  exit 3
fi

echo ">>> Source-only tree staged at $DEST"
echo ">>> Next: bash scripts/scan_public.sh $DEST"
