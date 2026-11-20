"""Fail-closed decorators for analysis entry points outside the tracked release manifest."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec

P = ParamSpec("P")


def unreleased_analysis_entrypoint(function: Callable[P, int]) -> Callable[P, int]:
    """Replace a legacy analysis CLI with an explicit non-submission release gate."""

    @wraps(function)
    def blocked(*_args: P.args, **_kwargs: P.kwargs) -> int:
        entrypoint = Path(function.__code__.co_filename).name
        print(
            f"BLOCKED: {entrypoint} is outside the endpoint-guarded manuscript manifest. "
            "Use scripts/run_manuscript.py for the bounded S1/S2 diagnostic workflow; "
            "do not treat cached or legacy outputs as evidence.",
            file=sys.stderr,
        )
        return 2

    return blocked
