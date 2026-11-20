"""Figure-generating functions for the Phase 1 report.

Each ``figure_*`` function takes already-computed data (never raw patient rows) and writes a
300-dpi PNG+PDF to ``results/figures/``. Functions return the list of written paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from ctb.config import FIGURES_DIR, N_BOOTSTRAP, SEED
from ctb.data.clean import CohortStep
from ctb.models.evaluation import PredictionResult, calibration_points, compute_metrics
from ctb.viz.style import MODEL_COLORS, PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)


def figure_cohort_flow(steps: list[CohortStep], out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Figure 1 — CONSORT-style cohort flow diagram.

    Parameters
    ----------
    steps : list of CohortStep
        Ordered inclusion steps from :func:`ctb.data.clean.build_cohort`.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(4.5, 0.9 * len(steps) + 1))
    ax.axis("off")
    n = len(steps)
    y_positions = np.linspace(1, 0, n)
    for i, (step, y) in enumerate(zip(steps, y_positions, strict=True)):
        ax.annotate(
            f"{step.label}\nn = {step.n_conditions:,}",
            xy=(0.5, y),
            xycoords="axes fraction",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": PALETTE[0],
                "alpha": 0.15,
                "edgecolor": PALETTE[0],
            },
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.06),
                xytext=(0.5, y - 0.06),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#444444"},
            )
            if step.n_excluded:
                mid = (y + y_positions[i + 1]) / 2
                ax.annotate(
                    f"excluded n = {step.n_excluded:,}",
                    xy=(0.78, mid),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color="#777777",
                )
    ax.set_title("Cohort selection")
    return save_figure(fig, out_dir / "figure1_cohort_flow")


def figure_outcome_prevalence(
    prevalence: pd.DataFrame, outcomes: pd.DataFrame, out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Figure 2 — component prevalence bars + UpSet plot of intensification subcomponents.

    Parameters
    ----------
    prevalence : pandas.DataFrame
        Output of :func:`ctb.data.outcomes.outcome_prevalence_table`.
    outcomes : pandas.DataFrame
        Per-condition outcome frame (boolean component columns).
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    components = [
        "escalation",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
        "treatment_intensification",
    ]
    sub = prevalence[prevalence["component"].isin(components)].copy()
    order = {c: i for i, c in enumerate(components)}
    sub = sub.sort_values("component", key=lambda s: s.map(order))
    cohort_n = len(outcomes)
    for _, row in sub.iterrows():
        component = str(row["component"])
        if component not in outcomes:
            raise ValueError(f"Prevalence component {component!r} is absent from outcomes")
        observed = outcomes[component]
        supported_dtype = bool(
            pd.api.types.is_bool_dtype(observed.dtype)
            or pd.api.types.is_integer_dtype(observed.dtype)
            or pd.api.types.is_float_dtype(observed.dtype)
        )
        invalid = (
            int((observed.notna() & ~observed.isin([0, 1, False, True])).sum())
            if supported_dtype
            else int(observed.notna().sum())
        )
        if invalid:
            raise ValueError(
                f"Outcome component {component!r} contains {invalid} non-binary values; "
                "values are not coerced for plotting"
            )
        actual_total = int(observed.notna().sum())
        actual_positive = int(observed.sum(skipna=True))
        stated_total = int(row["n_total"])
        stated_positive = int(row["n_positive"])
        if stated_total != actual_total or stated_positive != actual_positive:
            raise ValueError(
                f"Prevalence counts for {component!r} disagree with outcomes: "
                f"stated positive/assessable={stated_positive}/{stated_total}, "
                f"actual={actual_positive}/{actual_total}"
            )
        stated_prevalence = row["prevalence"]
        if actual_total:
            actual_prevalence = actual_positive / actual_total
            if pd.isna(stated_prevalence) or not np.isclose(stated_prevalence, actual_prevalence):
                raise ValueError(f"Prevalence value for {component!r} disagrees with its counts")
        elif pd.notna(stated_prevalence):
            raise ValueError(f"Prevalence for unassessable component {component!r} must be missing")

    with mpl.rc_context({"figure.constrained_layout.use": False}):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.16, top=0.86)
    colors = [
        PALETTE[2] if c != "treatment_intensification" else PALETTE[3] for c in sub["component"]
    ]
    bar_values = sub["prevalence"].fillna(0) * 100
    y_positions = np.arange(len(sub))
    ax.barh(y_positions, bar_values, color=colors)
    ax.set_yticks(y_positions, sub["component"])
    for y_position, (_, row) in zip(y_positions, sub.iterrows(), strict=True):
        n_positive = int(row["n_positive"])
        n_assessable = int(row["n_total"])
        if n_assessable and pd.notna(row["prevalence"]):
            x_position = float(row["prevalence"]) * 100 + 0.5
            label = (
                f"{float(row['prevalence']) * 100:.1f}% ({n_positive:,}/{n_assessable:,}); "
                f"assessable {n_assessable:,}/{cohort_n:,}"
            )
        else:
            x_position = 0.5
            label = f"not assessable (0/{cohort_n:,})"
        ax.text(
            x_position,
            y_position,
            label,
            va="center",
            fontsize=7,
        )
    ax.set_xlabel("Prevalence among assessable conditions (%)")
    ax.set_title("Archived treatment-intensification components")
    ax.invert_yaxis()
    paths = save_figure(fig, out_dir / "figure2a_outcome_prevalence")

    # UpSet of the three primary intensification components.
    try:
        from upsetplot import UpSet, from_indicators

        prim = ["escalation", "extension", "re_initiation"]
        validated_indicators = pd.DataFrame(index=outcomes.index)
        for component in prim:
            values = outcomes[component]
            supported_dtype = bool(
                pd.api.types.is_bool_dtype(values.dtype)
                or pd.api.types.is_integer_dtype(values.dtype)
                or pd.api.types.is_float_dtype(values.dtype)
            )
            invalid = (
                int((values.notna() & ~values.isin([0, 1, False, True])).sum())
                if supported_dtype
                else int(values.notna().sum())
            )
            if invalid:
                raise ValueError(
                    f"Outcome component {component!r} contains {invalid} non-binary values; "
                    "values are not coerced for plotting"
                )
            validated_indicators[component] = values.astype("boolean")
        complete = validated_indicators.notna().all(axis=1)
        n_total = len(outcomes)
        n_complete = int(complete.sum())
        n_excluded = int((~complete).sum())
        if not n_complete:
            logger.warning(
                "UpSet plot skipped: no complete cases across %s (excluded n=%d)",
                prim,
                n_excluded,
            )
            with mpl.rc_context({"figure.constrained_layout.use": False}):
                upset_fig, upset_ax = plt.subplots(figsize=(5, 3))
            upset_ax.axis("off")
            upset_ax.text(
                0.5,
                0.5,
                f"Component intersections not estimable\ncomplete cases: 0/{n_total:,}",
                ha="center",
                va="center",
            )
            paths += save_figure(upset_fig, out_dir / "figure2b_outcome_upset")
        else:
            if n_excluded:
                logger.warning(
                    "UpSet plot uses complete cases only: n=%d included, n=%d excluded",
                    n_complete,
                    n_excluded,
                )
            ind = validated_indicators.loc[complete, prim].astype(bool)
            data = from_indicators(prim, ind)
            upset_fig = plt.figure(figsize=(5, 3))
            UpSet(data, show_counts=True, sort_by="cardinality").plot(fig=upset_fig)
            upset_fig.suptitle(
                f"Complete-case component intersections "
                f"(n={n_complete:,}/{n_total:,}; excluded {n_excluded:,})"
            )
            paths += save_figure(upset_fig, out_dir / "figure2b_outcome_upset")
    except Exception as exc:  # pragma: no cover - dependency/backend-specific plotting fallback
        logger.warning(
            "UpSet plot failed safely (%s); replacing any stale output",
            type(exc).__name__,
        )
        with mpl.rc_context({"figure.constrained_layout.use": False}):
            upset_fig, upset_ax = plt.subplots(figsize=(5, 3))
        upset_ax.axis("off")
        upset_ax.text(
            0.5,
            0.5,
            "Component intersections not generated\n"
            f"plotting failed safely ({type(exc).__name__})\n"
            "stale output replaced",
            ha="center",
            va="center",
        )
        paths += save_figure(upset_fig, out_dir / "figure2b_outcome_upset")
    return paths


def _bootstrap_curve_band(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    kind: str,
    grid: NDArray[np.float64],
    n_boot: int = 200,
    seed: int = SEED,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bootstrap a confidence band for an ROC or PR curve on a fixed x-grid.

    Parameters
    ----------
    y_true, y_score : numpy.ndarray
        Labels and predicted probabilities.
    kind : str
        ``"roc"`` or ``"pr"``.
    grid : numpy.ndarray
        x-axis grid (FPR for ROC, recall for PR).
    n_boot : int, default 200
        Bootstrap resamples (kept modest for runtime).
    seed : int, default :data:`ctb.config.SEED`
        Seed.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lo, hi)`` 95% band values on the grid.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    curves = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            curves[b] = np.nan
            continue
        if kind == "roc":
            x, y, _ = roc_curve(yt, ys)
        else:
            prec, rec, _ = precision_recall_curve(yt, ys)
            x, y = rec[::-1], prec[::-1]
        curves[b] = np.interp(grid, x, y)
    lo = np.nanpercentile(curves, 2.5, axis=0)
    hi = np.nanpercentile(curves, 97.5, axis=0)
    return lo, hi


def figure_model_performance(
    results: dict[str, PredictionResult], out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Figure 4 — ROC and PR curves with bootstrap CI bands for all baselines.

    Parameters
    ----------
    results : dict
        Mapping model name -> ``BaselineResult`` (with ``y_true``/``y_score``).
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.0, 3.2))  # type: ignore[misc]
    grid = np.linspace(0, 1, 100)
    for name, res in results.items():
        color = MODEL_COLORS.get(name, PALETTE[0])
        yt = np.asarray(res.y_true, dtype=np.float64)
        ys = np.asarray(res.y_score, dtype=np.float64)
        m = compute_metrics(yt, ys, n_boot=N_BOOTSTRAP, seed=SEED)

        fpr, tpr, _ = roc_curve(yt, ys)
        ax_roc.plot(fpr, tpr, color=color, label=f"{name} (AUROC {m['auroc']:.2f})")
        lo, hi = _bootstrap_curve_band(yt, ys, "roc", grid)
        ax_roc.fill_between(grid, lo, hi, color=color, alpha=0.12)

        prec, rec, _ = precision_recall_curve(yt, ys)
        ax_pr.plot(rec, prec, color=color, label=f"{name} (AUPRC {m['auprc']:.2f})")
        lo_p, hi_p = _bootstrap_curve_band(yt, ys, "pr", grid)
        ax_pr.fill_between(grid, lo_p, hi_p, color=color, alpha=0.12)

    ax_roc.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax_roc.legend(loc="lower right")
    prevalence = float(np.mean(next(iter(results.values())).y_true))
    ax_pr.axhline(prevalence, ls="--", color="#999999", linewidth=0.8)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    ax_pr.legend(loc="upper right")
    return save_figure(fig, out_dir / "figure4_model_performance")


def figure_calibration(
    results: dict[str, PredictionResult], out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Calibration plot for all baselines (saved alongside Figure 4).

    Parameters
    ----------
    results : dict
        Mapping model name -> ``BaselineResult``.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8, label="Perfect")
    for name, res in results.items():
        color = MODEL_COLORS.get(name, PALETTE[0])
        mp, fp = calibration_points(
            np.asarray(res.y_true, dtype=np.float64),
            np.asarray(res.y_score, dtype=np.float64),
            n_bins=10,
        )
        ax.plot(mp, fp, marker="o", markersize=3, color=color, label=name)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction positive",
        title="Calibration",
    )
    ax.legend(loc="upper left")
    return save_figure(fig, out_dir / "figure4b_calibration")


def figure_shap_summary(
    pipe: Pipeline, transformed: pd.DataFrame, out_dir: Path = FIGURES_DIR, max_display: int = 20
) -> list[Path]:
    """Figure 5 — SHAP summary for the GBDT-with-context model.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Fitted pipeline whose final step is the LightGBM classifier.
    transformed : pandas.DataFrame
        Model-ready feature frame (post-preprocessing) with named columns.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.
    max_display : int, default 20
        Number of top features to display.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    import shap

    model = pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):  # binary classifier returns per-class list
        shap_values = shap_values[1]
    # shap.summary_plot builds its own colorbar, which conflicts with constrained_layout.
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(5.5, 5))
        shap.summary_plot(
            shap_values,
            transformed,
            max_display=max_display,
            show=False,
            color_bar=True,
            plot_size=None,
        )
        fig.tight_layout()
        paths = save_figure(fig, out_dir / "figure5_shap_summary")
    return paths
