"""Create a fresh private output root for access-gated claim audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--subdir", action="append", default=[])
    return parser.parse_args()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def prepare_output_root(
    output_root: Path,
    project_root: Path,
    input_roots: list[Path],
    subdirectories: list[str],
) -> Path:
    """Validate and create a private, empty output root with mode 0700."""
    root = output_root.expanduser().resolve(strict=False)
    project = project_root.expanduser().resolve()
    home = Path.home().resolve()
    broad_roots = {Path("/").resolve(), home, Path("/tmp").resolve()}
    if root in broad_roots or _overlaps(root, project):
        raise ValueError("Output root must be a dedicated directory outside the source repository")
    for raw in input_roots:
        resolved = raw.expanduser().resolve(strict=False)
        if _overlaps(root, resolved):
            raise ValueError("Output and input roots must be disjoint")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Output root must be absent or empty")
    if any(not name or name in {".", ".."} or "/" in name for name in subdirectories):
        raise ValueError("Subdirectories must be simple names")

    previous_umask = os.umask(0o077)
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        for name in subdirectories:
            child = root / name
            child.mkdir(mode=0o700)
            os.chmod(child, 0o700)
    finally:
        os.umask(previous_umask)
    return root


def main() -> int:
    args = _arguments()
    root = prepare_output_root(
        args.output_root,
        args.project_root,
        args.input_root,
        args.subdir,
    )
    print(
        json.dumps(
            {
                "status": "private output root prepared",
                "created_subdirectories": sorted(args.subdir),
                "permissions": oct(root.stat().st_mode & 0o777),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
