"""Endpoint choice for the seed-reproducibility figures (Supplementary Figs. S1 and S2).

The published Supplementary Figs. S1 and S2 were fitted on observed culture conversion, the matrix
written by ``scripts/05b_build_conversion_matrices.py``. That is the default here, so a plain run
redraws the published figures. ``--endpoint died_or_failure`` refits them on the primary endpoint
matrix (``scripts/05c_build_unfavorable_matrices.py``) and writes files with a ``_died_or_failure``
suffix, so the published version is never overwritten by the other endpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np

from ctb.config import PROCESSED_DIR
from ctb.data.artifacts import load_primary_matrix

ENDPOINTS = ("conversion", "died_or_failure")
CONVERSION_MATRIX = PROCESSED_DIR / "conversion_model_matrices.npz"
_REQUIRED = frozenset({"X", "C", "y", "drugs", "cond_ids", "context_names"})


def parse_endpoint(argv: Sequence[str] | None = None, *, description: str = "") -> str:
    """Read ``--endpoint`` from the command line (default: conversion, as published)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--endpoint", choices=ENDPOINTS, default="conversion",
                        help="outcome the model is fitted on (default: conversion, as published)")
    return str(parser.parse_args(argv).endpoint)


def figure_stem(base: str, endpoint: str) -> str:
    """Output name: the published name for conversion, a suffixed name otherwise."""
    return base if endpoint == "conversion" else f"{base}_{endpoint}"


def load_matrix(endpoint: str) -> Any:
    """Open the model matrix for ``endpoint`` and check its fields (fails closed)."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}; choose one of {ENDPOINTS}")
    if endpoint == "died_or_failure":
        return load_primary_matrix()
    if not CONVERSION_MATRIX.exists():
        raise FileNotFoundError(
            f"Missing culture-conversion matrix at {CONVERSION_MATRIX}. "
            "Run scripts/05b_build_conversion_matrices.py first."
        )
    data = np.load(CONVERSION_MATRIX, allow_pickle=True)
    missing = _REQUIRED.difference(data.files)
    if missing:
        data.close()
        raise ValueError(f"Conversion matrix is missing required fields: {sorted(missing)}")
    if not np.isin(np.asarray(data["y"]).reshape(-1).astype(int), [0, 1]).all():
        data.close()
        raise ValueError("Conversion matrix outcome must be binary")
    return data
