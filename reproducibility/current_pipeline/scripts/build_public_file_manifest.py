"""Rebuild the exact file and SHA-256 manifest for the staged public release."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "PUBLIC_FILE_MANIFEST.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="patch-public-manifest.", dir="/tmp") as temporary:
        staged = Path(temporary) / "release"
        subprocess.run(
            ["bash", str(ROOT / "scripts" / "sync_to_public.sh"), str(staged)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        files = sorted(
            path for path in staged.rglob("*") if path.is_file() and path != staged / MANIFEST_NAME
        )
        rows = [f"{_sha256(path)}  {path.relative_to(staged).as_posix()}" for path in files]

    destination = ROOT / MANIFEST_NAME
    temporary_manifest = destination.with_suffix(destination.suffix + ".tmp")
    temporary_manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.chmod(temporary_manifest, 0o644)
    temporary_manifest.replace(destination)
    print(f"Wrote {len(rows)} public release file hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
