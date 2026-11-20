"""Evaluation metrics: AUROC, AUPRC, Brier, calibration, bootstrap CIs, and thresholded summaries.

All randomness (bootstrap resampling) is seeded from :data:`ctb.config.SEED` for reproducibility.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ctb.config import N_BOOTSTRAP, SEED

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class PredictionResult(Protocol):
    """Structural type for an object carrying out-of-fold predictions.

    Attributes
    ----------
    name : str
        Model identifier.
    y_true : numpy.ndarray
        Ground-truth labels.
    y_score : numpy.ndarray
        Predicted positive-class probabilities.
    """

    name: str
    y_true: FloatArray
    y_score: FloatArray


def bootstrap_ci(
    y_true: FloatArray,
    y_score: FloatArray,
    metric_fn: Callable[[FloatArray, FloatArray], float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute a metric and its percentile bootstrap confidence interval.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    y_score : numpy.ndarray
        Predicted positive-class probabilities.
    metric_fn : callable
        Function ``(y_true, y_score) -> float``.
    n_boot : int, default :data:`ctb.config.N_BOOTSTRAP`
        Number of bootstrap resamples.
    seed : int, default :data:`ctb.config.SEED`
        Seed for the resampling generator.
    alpha : float, default 0.05
        Two-sided significance level (0.05 → 95% CI).

    Returns
    -------
    tuple of float
        ``(point_estimate, ci_low, ci_high)``.
    """
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))
    n = len(y_true)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:  # undefined for single-class resample
            continue
        stats.append(float(metric_fn(yt, ys)))
    lo = float(np.percentile(stats, 100 * alpha / 2)) if stats else float("nan")
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2))) if stats else float("nan")
    return point, lo, hi


def youden_threshold(y_true: FloatArray, y_score: FloatArray) -> float:
    """Return the probability threshold maximising Youden's J (sensitivity + specificity − 1).

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    y_score : numpy.ndarray
        Predicted positive-class probabilities.

    Returns
    -------
    float
        Optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_metrics(
    y_true: FloatArray, y_score: FloatArray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, object]:
    """Compute the full metric suite with bootstrap CIs and a thresholded confusion matrix.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    y_score : numpy.ndarray
        Predicted positive-class probabilities.
    n_boot : int, default :data:`ctb.config.N_BOOTSTRAP`
        Bootstrap resamples for CIs.
    seed : int, default :data:`ctb.config.SEED`
        Bootstrap seed.

    Returns
    -------
    dict
        Metrics including ``auroc``/``auprc`` (with ``*_ci``), ``brier``, ``prevalence``,
        ``threshold_youden``, ``confusion_matrix`` (tn, fp, fn, tp), ``sensitivity``,
        ``specificity``, and per-class accuracy.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    auroc, auroc_lo, auroc_hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, seed)
    auprc, auprc_lo, auprc_hi = bootstrap_ci(y_true, y_score, average_precision_score, n_boot, seed)
    brier = float(brier_score_loss(y_true, y_score))

    thr = youden_threshold(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": auroc,
        "auroc_ci": [auroc_lo, auroc_hi],
        "auprc": auprc,
        "auprc_ci": [auprc_lo, auprc_hi],
        "brier": brier,
        "threshold_youden": thr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity": sens,
        "specificity": spec,
        "accuracy_pos": sens,
        "accuracy_neg": spec,
    }


def roc_points(y_true: FloatArray, y_score: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ROC curve points.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    y_score : numpy.ndarray
        Predicted probabilities.

    Returns
    -------
    tuple of numpy.ndarray
        ``(fpr, tpr)``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def calibration_points(
    y_true: FloatArray, y_score: FloatArray, n_bins: int = 10
) -> tuple[FloatArray, FloatArray]:
    """Return calibration-curve points (fraction of positives vs mean predicted probability).

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    y_score : numpy.ndarray
        Predicted probabilities.
    n_bins : int, default 10
        Number of calibration bins.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean_predicted, fraction_positive)``.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos
