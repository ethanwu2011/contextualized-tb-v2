"""Unit tests for :mod:`ctb.models.evaluation` (deterministic, synthetic scores)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ctb.models.evaluation import (
    bootstrap_ci,
    compute_metrics,
    youden_threshold,
)


def _separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100, dtype=np.float64)
    score = np.concatenate([rng.normal(0.3, 0.1, 100), rng.normal(0.7, 0.1, 100)]).astype(
        np.float64
    )
    return y, np.clip(score, 0, 1)


def test_bootstrap_ci_brackets_point() -> None:
    y, s = _separable()
    point, lo, hi = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=42)
    assert lo <= point <= hi
    assert 0.5 < point <= 1.0
    assert hi - lo < 0.3  # reasonably tight for clean signal


def test_bootstrap_ci_is_deterministic() -> None:
    y, s = _separable()
    a = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=42)
    b = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=42)
    assert a == b


def test_youden_threshold_range() -> None:
    y, s = _separable()
    thr = youden_threshold(y, s)
    assert 0.0 <= thr <= 1.0


def test_compute_metrics_keys_and_ranges() -> None:
    y, s = _separable()
    m = compute_metrics(y, s, n_boot=100, seed=42)
    for key in ["auroc", "auprc", "brier", "sensitivity", "specificity", "prevalence"]:
        assert key in m
    assert 0.5 < m["auroc"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["prevalence"] == 0.5
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)


def test_compute_metrics_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    m = compute_metrics(y, s, n_boot=50, seed=42)
    assert m["auroc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
