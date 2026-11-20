#!/usr/bin/env bash
# Hash-bound hygiene scan; never print matching values or line contents.
set -euo pipefail
DIR="${1:?usage: scan_public.sh <staged-public-directory>}"
[ -d "$DIR" ] && [ ! -L "$DIR" ] || { echo "SCAN FAILED: real directory required." >&2; exit 2; }
python3 - "$DIR" <<'PYSCAN'
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
def fail(label):
    print("SCAN FAILED: " + label, file=sys.stderr)
    raise SystemExit(1)
manifest = root / "PUBLIC_FILE_MANIFEST.sha256"
if not manifest.is_file():
    fail("exact public file manifest missing")
rows = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
if any(len(row) != 2 or not re.fullmatch("[0-9a-f]{64}", row[0]) or Path(row[1]).is_absolute() or ".." in Path(row[1]).parts for row in rows):
    fail("unsafe exact public file manifest")
listed = {name: digest for digest, name in rows}
for excluded in ("data", "logs", "models", "lightning_logs", ".venv"):
    if (root / excluded).exists():
        fail("excluded directory")
aggregate = root / "manuscript/aggregate_results"
if any(path.is_dir() for path in aggregate.rglob("*")):
    fail("nested aggregate directory")
secrets = re.compile(rb"AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{20,}|sk-[0-9A-Za-z]{20,}|-----BEGIN (?:RSA|OPENSSH|PRIVATE) KEY-----")
private = {".tsv", ".jsonl", ".ndjson", ".parquet", ".feather", ".arrow", ".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth", ".dbf", ".dbc", ".joblib", ".h5", ".hdf5", ".sqlite", ".safetensors", ".xlsx", ".xls"}
for path in root.rglob("*"):
    relative = path.relative_to(root)
    if any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in relative.parts):
        continue
    if path.is_symlink():
        fail("symlink")
    if not path.is_file():
        continue
    name = relative.as_posix()
    if path.suffix in private or (path.suffix in {".csv", ".zip"} and name not in listed):
        fail("patient-shaped or unreviewed data artifact")
    if path.suffix == ".json" and name not in listed:
        fail("unallowlisted JSON; file set mismatch")
    if path.suffix in {".py", ".md", ".txt", ".json", ".log", ".sh", ".toml", ".yml", ".yaml"}:
        data = path.read_bytes()
        if secrets.search(data):
            fail("secret-shaped literal")
        # Immutable reviewed historical paths are provenance, not execution dependencies.
        if re.search(rb"/(?:Users|home)/[A-Za-z0-9._-]+", data) and name not in listed:
            fail("local absolute path in unreviewed source")
try:
    spec = importlib.util.spec_from_file_location("release_verify", root / "verify_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_members()
except Exception:
    fail("exact public file manifest file set or checksum verification failed")
print("Public release scan passed.")
PYSCAN
