#!/usr/bin/env bash
# Pre-publish hygiene scan for the staged PUBLIC PATCH repo.
# Fails (exit 1) if it finds anything that must never be published: real-data files, the real-data
# path, secret-shaped strings, absolute home paths, or notebook outputs. Run on the TARGET dir.
#
# Usage: bash scripts/scan_public.sh <PUBLIC_DIR>
set -euo pipefail
DIR_INPUT="${1:?usage: scan_public.sh <PUBLIC_DIR>}"
if [ ! -d "$DIR_INPUT" ]; then
  echo "Directory does not exist: $DIR_INPUT" >&2
  exit 2
fi
DIR="$(cd "$DIR_INPUT" && pwd)"
FAIL=0
note() { printf '  [%s] %s\n' "$1" "$2"; }

echo ">>> Scanning $DIR"

echo "== 1. data-shaped files =="
HITS="$(find "$DIR" -type f \( \
  -name '*.parquet' -o -name '*.npz' -o -name '*.feather' -o -name '*.pkl' -o -name '*.pickle' \
  -o -name '*.joblib' -o -name '*.npy' -o -name '*.h5' -o -name '*.hdf5' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.rds' -o -name '*.arrow' -o -name '*.sqlite' -o -name '*.sqlite3' \
  -o -name '*.db' -o -name '*.xls' -o -name '*.xlsx' -o -name '*.csv' -o -name '*.tsv' \
  -o -name '*.jsonl' -o -name '*.ndjson' -o -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' \
  -o -name '*.tgz' \
\) -not -path "$DIR/.git/*" -print 2>/dev/null)"
# Reviewed source data: a table may pass only if figures/source_data/SHA256SUMS lists its exact path
# and its SHA-256 still matches. Anything unlisted or changed still fails.
ALLOWLIST="$DIR/figures/source_data/SHA256SUMS"
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
if [ -n "$HITS" ] && [ -f "$ALLOWLIST" ]; then
  REMAIN=""
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    rel="${hit#"$DIR"/}"
    expected="$(awk -v p="$rel" '$2 == p {print $1}' "$ALLOWLIST")"
    if [ -n "$expected" ] && [ ! -L "$hit" ] && [ "$(sha256_of "$hit")" = "$expected" ]; then
      note OK "reviewed source data, hash matches: $rel"
    else
      REMAIN="${REMAIN}${hit}"$'\n'
    fi
  done <<< "$HITS"
  HITS="$(printf '%s' "$REMAIN" | sed '/^$/d')"
fi
if [ -n "$HITS" ]; then
  note FAIL "data-shaped files are not publishable"
  echo "$HITS"
  FAIL=1
else
  note OK "no unreviewed data-shaped files"
fi

echo "== 2. symlinks =="
HITS="$(find "$DIR" -type l -print 2>/dev/null)"
if [ -n "$HITS" ]; then
  note FAIL "symlinks are not allowed in a release tree"
  echo "$HITS"
  FAIL=1
else
  note OK "none"
fi

echo "== 3. local paths =="
PATHHITS="$(grep -rIlE --exclude='scan_public.sh' --exclude-dir=.git \
  'Documents/Tuberculosis|/Users/[A-Za-z]' "$DIR" 2>/dev/null || true)"
if [ -n "$PATHHITS" ]; then
  note FAIL "local path references found"
  echo "$PATHHITS"
  FAIL=1
else
  note OK "none"
fi

echo "== 4. secret-shaped strings =="
SECRET_RE="(AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{20,}|sk-[0-9A-Za-z]{20,}|-----BEGIN (RSA|OPENSSH|PRIVATE) KEY-----|password[[:space:]]*=[[:space:]]*['\"])"
SECRET_HITS="$(grep -rInE --exclude='scan_public.sh' --exclude-dir=.git "$SECRET_RE" "$DIR" 2>/dev/null || true)"
if [ -n "$SECRET_HITS" ]; then
  note FAIL "possible secret(s)"
  echo "$SECRET_HITS" | head -20
  FAIL=1
else
  note OK "none"
fi

echo "== 5. notebooks =="
NOTEBOOKS="$(find "$DIR" -type f -name '*.ipynb' -print 2>/dev/null)"
if [ -n "$NOTEBOOKS" ]; then
    OUTPUT_HITS="$(grep -Il --exclude-dir=.git '"output_type"' $NOTEBOOKS 2>/dev/null || true)"
  if [ -n "$OUTPUT_HITS" ]; then
    note FAIL "notebooks with output cells found"
    echo "$OUTPUT_HITS"
    FAIL=1
  else
    note OK "notebooks are output-free"
  fi
else
  note OK "none"
fi

echo "== 6. excluded directories =="
EXCLUDED="$(find "$DIR" -type d \( \
  -name logs -o -name lightning_logs -o -name .venv \
  -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \
  -o -name data -o -name results -o -name notebooks -o -name models \
	\) -print 2>/dev/null | while IFS= read -r path; do
	  # A local Git checkout necessarily contains .git metadata, which is not part of the
	  # published working tree. Ignore that metadata while scanning the checked-out files.
	  if [[ "$path" == "$DIR/.git" || "$path" == "$DIR/.git/"* ]]; then
	    continue
	  fi
	  # Installable source packages are intentionally named ctb/data and ctb/models.
	  if [ "$path" = "$DIR/src/ctb/data" ] || [ "$path" = "$DIR/src/ctb/models" ] || \
	     [ "$path" = "$DIR/reproducibility/current_pipeline/src/ctb/data" ] || \
	     [ "$path" = "$DIR/reproducibility/current_pipeline/src/ctb/models" ]; then
	    continue
	  fi
	  printf '%s\n' "$path"
done)"
if [ -n "$EXCLUDED" ]; then
  note FAIL "excluded directories found"
  echo "$EXCLUDED"
  FAIL=1
else
  note OK "none"
fi

echo
if [ "$FAIL" = 0 ]; then
  echo ">>> SCAN PASSED. Source-only tree is safe to publish."
else
  echo ">>> SCAN FAILED. Do not publish this tree."
fi
exit "$FAIL"
