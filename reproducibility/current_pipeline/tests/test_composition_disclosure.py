"""Synthetic disclosure checks for fresh shared-profile composition reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "composition_disclosure_fixture",
        root / "scripts/run_shared_profile_reference_composition.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rare_category_cannot_be_recovered_from_visible_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _driver(monkeypatch)
    values = pd.Series(["common"] * 100 + ["secondary"] * 50 + ["rare"] * 10)
    rows = module._safe_category(values, np.ones(len(values), dtype=bool))
    visible = sum(item["count"] for item in rows.values() if isinstance(item["count"], int))
    assert len(values) - visible >= 30
    assert rows["rare"]["percent"] is None
    assert rows["secondary"]["percent"] is None
    assert rows["common"]["count"] == 100


def test_binary_complement_and_rate_are_hidden_when_complement_is_rare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _driver(monkeypatch)
    values = pd.Series([1] * 100 + [0] * 10)
    row = module._safe_binary(values, np.ones(len(values), dtype=bool))
    assert row["known_count"] == 110
    assert row["positive_count"]["suppressed"] is True
    assert row["percent_among_known"] is None


def test_supported_binary_summary_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _driver(monkeypatch)
    values = pd.Series([1] * 100 + [0] * 50)
    row = module._safe_binary(values, np.ones(len(values), dtype=bool))
    assert row["positive_count"] == 100
    assert row["percent_among_known"] == pytest.approx(100 * 100 / 150)
