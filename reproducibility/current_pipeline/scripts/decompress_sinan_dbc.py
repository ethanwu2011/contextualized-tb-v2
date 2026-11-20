"""Decompress one DATASUS DBC file without reading or printing its records."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasus_dbc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != ".dbc":
        raise FileNotFoundError("Source must be an existing DBC file")
    if output.suffix.casefold() != ".dbf":
        raise ValueError("Output must use the DBF suffix")
    if output.exists():
        raise FileExistsError("Output DBF already exists; refusing to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    previous_umask = os.umask(0o077)
    try:
        datasus_dbc.decompress(str(source), str(output))
    finally:
        os.umask(previous_umask)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("DBC decompression did not produce a nonempty DBF")
    os.chmod(output, 0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
