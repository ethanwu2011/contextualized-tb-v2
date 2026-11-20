"""Synthetic alignment and leakage contract for the shared transport figure builder."""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ctb.data.temporal import extract_calendar_year, patient_disjoint_temporal_masks
from ctb.models.fold_preprocessing import FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    RawValidationData,
    _additive_probability,
    _fit_additive,
    _seed,
)


def test_reordered_case_rows_preserve_outcome_direction_and_train_only_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_aggregate_sensitivities import _point_metrics
    from run_shared_k_sensitivity import OOF_CONFIG

    # Every value is generated here; no clinical file or stored patient artifact is opened.
    index = pd.Index([f"synthetic-condition-{i}" for i in range(900)], name="condition_id")
    outcome = pd.Series(np.arange(900) % 2, index=index)
    future = np.arange(900) >= 600
    country = np.where(future, "Synthetic-future", "Synthetic-history")
    context = pd.DataFrame({"signal": outcome, "country": country}, index=index)
    treatment = pd.Series(
        [
            frozenset({"common", "future_only"}) if flag else frozenset({"common"})
            for flag in future
        ],
        index=index,
    )
    groups = pd.Series([f"synthetic-patient-{i}" for i in range(900)], index=index)
    groups.iloc[650] = groups.iloc[0]  # A crossing patient must be excluded from both periods.
    data = RawValidationData(
        context=context,
        treatments=treatment,
        outcome=outcome,
        patient_groups=groups,
        numeric_columns=("signal",),
        categorical_columns=("country",),
    )
    cases = pd.DataFrame(
        {
            "condition_id": index,
            "registration_date": np.where(future, 2022, 2019),
            "country": country,
        }
    ).sample(frac=1, random_state=17)
    calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    class KnownPositiveClassModel:
        def predict_proba(self, c: np.ndarray, x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-c[:, 0]))

    def fit(matrices: Any, y_train: np.ndarray, seed: int, config: Any) -> Any:
        assert config is OOF_CONFIG
        assert np.corrcoef(matrices.C[:, 0], y_train)[0, 1] > 0.99
        calls.append(
            {
                "index": matrices.row_index,
                "features": matrices.context_feature_names,
                "regimens": matrices.treatment_feature_names,
            }
        )
        return KnownPositiveClassModel()

    def write(path: Path, report: dict[str, Any]) -> None:
        captured.update(report)

    namespace = {
        "np": np,
        "pd": pd,
        "FoldPreprocessor": FoldPreprocessor,
        "extract_calendar_year": extract_calendar_year,
        "patient_disjoint_temporal_masks": patient_disjoint_temporal_masks,
        "_fit_additive": _fit_additive,
        "_additive_probability": _additive_probability,
        "_seed": _seed,
        "_point_metrics": _point_metrics,
        "fit": fit,
        "_cluster_bootstrap_metrics": lambda *args, **kwargs: {"synthetic_contract": True},
        "OOF_CONFIG": OOF_CONFIG,
        "MIN_CELL": 30,
        "write": write,
    }
    source = root / "scripts/build_shared_primary_aggregates.py"
    tree = ast.parse(source.read_text())
    contract = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"prepare", "transport"}
        ],
        type_ignores=[],
    )
    exec(
        compile(contract, str(source), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    namespace["transport"](cases, data, LockedValidationConfig(), Path("unused-synthetic-output"))

    temporal = calls[0]
    assert len(temporal["index"]) == 599
    assert index[0] not in temporal["index"] and index[650] not in temporal["index"]
    assert not any("Synthetic-future" in name for name in temporal["features"])
    assert not any("future_only" in name for name in temporal["regimens"])
    report = captured["temporal"]
    assert report["train_conditions"] == 599 and report["test_conditions"] == 299
    assert report["metrics"]["shared_k5"]["auroc"] == 1.0
    assert report["metrics"]["additive"]["auroc"] == 1.0
    assert captured["configuration"]["epochs"] == OOF_CONFIG.epochs == 500
