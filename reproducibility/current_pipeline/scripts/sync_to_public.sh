#!/usr/bin/env bash
# Copy only the reviewed exact current release. Never copy repository history or runtime outputs.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd -P)"
DEST_INPUT="${1:?usage: sync_to_public.sh <new-target-directory>}"
PARENT="$(dirname "$DEST_INPUT")"
[ -d "$PARENT" ] || { echo "REFUSED: destination parent must exist." >&2; exit 2; }
DEST="$(cd "$PARENT" && pwd -P)/$(basename "$DEST_INPUT")"
case "$DEST" in
  "/"|"$HOME"|"$HOME/"*|"$SRC"|"$SRC/"*)
    echo "REFUSED: target must resolve outside the source repository and home directory." >&2; exit 2;;
esac
[ ! -e "$DEST" ] || { echo "REFUSED: target already exists." >&2; exit 2; }
if [ -L "$SRC/README.md" ]; then
  echo "REFUSED: allowlisted file is a symlink." >&2; exit 2
fi
python3 - "$SRC" "$DEST" <<'PYSOURCE'
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("release_verify", source / "verify_package.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_members()
manifest = json.loads((source / "PACKAGE_CONTENTS.json").read_text())
files = [*manifest["files"], "PACKAGE_CONTENTS.json", "PUBLIC_FILE_MANIFEST.sha256"]
for name in files:
    path = source / name
    if path.is_symlink():
        raise SystemExit("REFUSED: allowlisted file is a symlink")
destination.mkdir(mode=0o700)
for name in files:
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, target)
print("Public source release staged successfully.")
PYSOURCE
