"""Render the corrected PATCH figures from disclosure-safe aggregate JSON files.

This module never reads a patient-level artifact. It is the public rendering path for the frozen
aggregate results under ``manuscript/aggregate_results``. The figures describe prediction,
continuous response profiles, and regularized regimen-outcome associations. They do not report
causal treatment effects or prescribing rules.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "manuscript" / "aggregate_results"
DEFAULT_OUTPUT = ROOT / "manuscript" / "figures"

INK = "#202124"
BLUE = "#2864A9"
ORANGE = "#D97706"
GREEN = "#2A7F62"
PURPLE = "#7851A9"
GRAY = "#8A8F98"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GRAY]
PROFILE_COLORS = ["#009E73", "#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
RELEASE_TIMESTAMP = datetime(2026, 8, 27, tzinfo=timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    """Load one aggregate JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _profile_labels() -> list[str]:
    """Read canonical profile labels from the public release manifest."""
    manifest = _read_json(ROOT / "PUBLIC_RELEASE_MANIFEST.json")
    profiles = sorted(manifest["profiles"], key=lambda row: int(row["risk_rank"]))
    return [textwrap.fill(str(row["label"]), width=30) for row in profiles]


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one figure as reviewer-friendly PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "PATCH aggregate release renderer",
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _style() -> None:
    """Apply a compact, print-safe matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _schematic_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    color: str,
) -> None:
    """Draw one compact schematic stage."""
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
    )
    axis.add_patch(
        Rectangle(
            (x, y + height - 0.12),
            width,
            0.12,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
    )
    axis.text(
        x + width / 2,
        y + height - 0.06,
        title,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )
    axis.text(
        x + 0.018,
        y + height - 0.17,
        "\n".join(lines),
        ha="left",
        va="top",
        color=INK,
        fontsize=8.4,
        linespacing=1.55,
    )


def render_conceptual_schematic(output_dir: Path) -> None:
    """Render a data-free overview of the governed PATCH analysis."""
    figure, axis = plt.subplots(figsize=(11.4, 4.0), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = [
        (
            0.02,
            "Recorded inputs",
            [
                "Context C",
                "279 retained features",
                "Initial regimen X",
                "26 recorded components",
            ],
            BLUE,
        ),
        (
            0.275,
            "Shared representation",
            [
                "Context encoder Φ(C)",
                "K = 5 coefficient archetypes",
                "Regularized joint fit",
                "Patient-grouped validation",
            ],
            PURPLE,
        ),
        (
            0.53,
            "Patient-specific map",
            [
                "Soft mixture weights",
                "β(C) regimen coefficients",
                "μ(C) context intercept",
                "Hard profiles are summaries",
            ],
            GREEN,
        ),
        (
            0.785,
            "Audited outputs",
            [
                "Prognostic risk",
                "Recurring response profiles",
                "Support-gated association map",
                "Transport and stability checks",
            ],
            ORANGE,
        ),
    ]
    width = 0.195
    height = 0.54
    y = 0.27
    for x, title, lines, color in stages:
        _schematic_box(axis, x, y, width, height, title, lines, color)
    for left in (0.02, 0.275, 0.53):
        axis.add_patch(
            FancyArrowPatch(
                (left + width + 0.008, y + height / 2),
                (left + 0.247, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.3,
                color=GRAY,
            )
        )
    axis.text(
        0.5,
        0.92,
        "PATCH contextualizes recorded regimen-response associations through shared archetypes",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.5,
        0.12,
        r"logit P(Y = 1 | X, C) = μ(C) + Xβ(C)",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    axis.text(
        0.5,
        0.045,
        (
            "β(C) is a regularized observational coefficient. It is not an identified causal "
            "effect or a prescribing rule."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        color=GRAY,
    )
    _save(figure, output_dir, "fig01_patch_schematic")


def render_performance(input_dir: Path, output_dir: Path) -> None:
    """Render architecture-relevant held-out performance and calibration."""
    sensitivity = _read_json(input_dir / "sensitivity_and_support.json")
    k_report = _read_json(input_dir / "shared_k_sensitivity.json")
    primary = sensitivity["primary_ablation"]
    shared = k_report["oof_metrics"]["shared_k5"]
    primary_uncertainty = sensitivity["primary_patient_clustered_uncertainty"]["models"]
    shared_uncertainty = k_report["patient_clustered_uncertainty"]["models"]["shared_k5"]
    models = [
        ("Regimen only", primary["regimen_only"]["metrics"], primary_uncertainty["regimen_only"]),
        ("Context only", primary["context_only"]["metrics"], primary_uncertainty["context_only"]),
        ("Additive", primary["additive"]["metrics"], primary_uncertainty["additive"]),
        ("Shared PATCH K=5", shared, shared_uncertainty),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), constrained_layout=True)
    x = np.arange(len(models))
    labels = [item[0] for item in models]
    for axis, metric, title in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "auprc", "AUPRC"),
        (axes[2], "brier", "Brier score"),
    ]:
        values = [float(item[1][metric]) for item in models]
        intervals = [item[2][metric]["ci95"] for item in models]
        lower = np.asarray([float(value[0]) for value in intervals])
        upper = np.asarray([float(value[1]) for value in intervals])
        values_array = np.asarray(values)
        for position, value, low, high, color in zip(
            x, values_array, lower, upper, PALETTE[: len(models)], strict=True
        ):
            axis.errorbar(
                position,
                value,
                yerr=np.asarray([[value - low], [high - value]]),
                fmt="o",
                markersize=6,
                color=color,
                capsize=3,
                linewidth=1.2,
            )
            axis.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_xticks(x, labels, rotation=28, ha="right")
        span = float(upper.max() - lower.min())
        padding = max(span * 0.12, 0.002)
        axis.set_ylim(float(lower.min() - padding), float(upper.max() + 2 * padding))
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    axes[2].text(
        0.02,
        0.98,
        "Lower is better",
        transform=axes[2].transAxes,
        va="top",
        color=GRAY,
        fontsize=8,
    )
    figure.suptitle(
        "Held-out performance with patient-clustered 95% intervals",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, output_dir, "fig02_locked_performance")


def render_profiles(input_dir: Path, output_dir: Path) -> None:
    """Render corrected hard-bin profile summaries in soft-risk order."""
    report = _read_json(input_dir / "shared_profile_reference_composition.json")
    profiles = sorted(report["profiles"], key=lambda row: int(row["risk_rank"]))
    profile_labels = _profile_labels()
    counts = np.asarray([row["hard_count"] for row in profiles], dtype=float)
    events = 100 * np.asarray([row["hard_event_rate"] for row in profiles], dtype=float)
    hiv = np.asarray([row["hiv"]["percent_among_known"] for row in profiles], dtype=float)

    resistance_groups = ["Sensitive", "Mono DR", "Poly DR", "MDR non XDR", "Pre-XDR", "XDR"]
    resistance_colors = ["#80B1D3", "#B3DE69", "#FDB462", "#FB8072", "#BC80BD", "#8C6D31"]
    resistance_rows: list[list[float]] = []
    for row in profiles:
        entries = row["resistance_composition"]
        known = [
            0.0 if entries[name]["percent"] is None else float(entries[name]["percent"])
            for name in resistance_groups
        ]
        has_suppressed = any(entries[name]["percent"] is None for name in resistance_groups)
        suppressed_remainder = max(0.0, 100.0 - sum(known)) if has_suppressed else 0.0
        resistance_rows.append([*known, suppressed_remainder])
    resistance_groups.append("Suppressed categories")
    resistance_colors.append("#D9D9D9")
    resistance = np.asarray(resistance_rows)

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    y = np.arange(len(profiles))
    axes[0].barh(y, counts, color=PROFILE_COLORS)
    axes[0].set_yticks(y, profile_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Conditions in hard display bin")
    axes[0].set_title("a  Profile size", loc="left", fontweight="bold")
    for position, value in zip(y, counts, strict=True):
        axes[0].text(value + 25, position, f"{int(value):,}", va="center", fontsize=8)

    width = 0.36
    axes[1].barh(y - width / 2, events, height=width, color=ORANGE, label="Death or failure")
    axes[1].barh(y + width / 2, hiv, height=width, color=BLUE, label="HIV")
    axes[1].invert_yaxis()
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Percent")
    axes[1].set_title("b  Outcome and HIV", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")

    left = np.zeros(len(profiles))
    for index, name in enumerate(resistance_groups):
        axes[2].barh(
            y,
            resistance[:, index],
            left=left,
            color=resistance_colors[index],
            label=name,
            hatch="////" if name == "Suppressed categories" else None,
            edgecolor="#9A9A9A" if name == "Suppressed categories" else "none",
            linewidth=0.5,
        )
        left += resistance[:, index]
    axes[2].invert_yaxis()
    axes[2].set_yticks(y, [])
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Resistance composition, percent")
    axes[2].set_title("c  Recorded resistance", loc="left", fontweight="bold")
    axes[2].legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    figure.suptitle(
        "Five recurring continuous profiles with hard assignments shown only as summaries",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, output_dir, "fig03_profiles")


def render_association_heatmap(input_dir: Path, output_dir: Path) -> None:
    """Render the profile-by-regimen coefficient surface with uncertainty marks."""
    stability = _read_json(input_dir / "shared_profile_full_pipeline_stability.json")
    composition = _read_json(input_dir / "shared_profile_reference_composition.json")
    profile_labels = _profile_labels()
    ordered_profiles = [
        int(row["fitted_profile"])
        for row in sorted(composition["profiles"], key=lambda row: int(row["risk_rank"]))
    ]
    cells = stability["profile_by_regimen_uncertainty"]["cells"]
    regimen_labels = list(dict.fromkeys(str(cell["regimen_label"]) for cell in cells))
    regimen_index = {name: index for index, name in enumerate(regimen_labels)}
    profile_index = {profile: index for index, profile in enumerate(ordered_profiles)}
    values = np.full((len(ordered_profiles), len(regimen_labels)), np.nan)
    interval = np.zeros_like(values, dtype=bool)
    stable_sign = np.zeros_like(values, dtype=bool)
    suppressed = np.ones_like(values, dtype=bool)
    for cell in cells:
        row = profile_index[int(cell["profile"])]
        column = regimen_index[str(cell["regimen_label"])]
        if cell["support_status"] != "supported":
            continue
        values[row, column] = float(cell["reference_shared_logit_coefficient"])
        interval[row, column] = bool(cell["interval_excludes_zero"])
        stable_sign[row, column] = bool(cell["sign_agreement_at_least_0.80"])
        suppressed[row, column] = False

    finite = np.abs(values[np.isfinite(values)])
    scale = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
    figure, axis = plt.subplots(figsize=(12.4, 4.5), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axis.set_xticks(np.arange(len(regimen_labels)), regimen_labels, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(profile_labels)), profile_labels)
    axis.set_title(
        "Regularized profile-by-regimen logit coefficients",
        loc="left",
        fontweight="bold",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if suppressed[row, column]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="#E6E6E6",
                        edgecolor="#B8B8B8",
                        hatch="////",
                        linewidth=0.4,
                    )
                )
            elif interval[row, column]:
                axis.text(column, row, "*", ha="center", va="center", color=INK, fontsize=12)
            elif stable_sign[row, column]:
                axis.text(column, row, "•", ha="center", va="center", color=INK, fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Logit coefficient")
    axis.text(
        0,
        -0.34,
        "* full-pipeline 95% interval excludes zero    • sign retained in at least 80% of "
        "resamples    hatched insufficient marginal support",
        transform=axis.transAxes,
        fontsize=8,
        color=INK,
    )
    axis.text(
        0,
        -0.42,
        "Associations conditional on recorded context. Not causal drug effects or prescribing "
        "recommendations.",
        transform=axis.transAxes,
        fontsize=8,
        color=GRAY,
    )
    _save(figure, output_dir, "fig04_association_heatmap")


def render_brazil(input_dir: Path, output_dir: Path) -> None:
    """Render prognostic transport to the strict SINAN cohort."""
    report = _read_json(input_dir / "brazil_source_rebuild_audit.json")
    strict = report["external_discrimination"]["strict_primary_excluding_postmortem"]
    models = [
        ("Resistance only", strict["resistance_only"]),
        ("PATCH profile score", strict["patch_profile_score"]),
        ("Context logistic", strict["context_logistic"]),
    ]
    seed_rows = report["profile_transport"]["seeds"]
    patch_seed_auroc = np.asarray(
        [float(row["external_metrics"]["auroc"]) for row in seed_rows], dtype=float
    )
    years = sorted({int(row["year"]) for seed in seed_rows for row in seed["yearly_transport"]})
    yearly_auroc_matrix = np.asarray(
        [
            [
                float(
                    next(row for row in seed["yearly_transport"] if int(row["year"]) == year)[
                        "patch_profile_score_auroc"
                    ]
                )
                for year in years
            ]
            for seed in seed_rows
        ]
    )
    yearly_order_matrix = np.asarray(
        [
            [
                float(
                    next(row for row in seed["yearly_transport"] if int(row["year"]) == year)[
                        "profile_risk_spearman_vs_source"
                    ]
                )
                for year in years
            ]
            for seed in seed_rows
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    x = np.arange(len(models))
    auroc = [float(item[1]["auroc"]) for item in models]
    axes[0].scatter(x, auroc, color=[GRAY, BLUE, ORANGE], s=46, zorder=3)
    jitter = np.linspace(-0.12, 0.12, len(patch_seed_auroc))
    axes[0].scatter(
        np.full_like(patch_seed_auroc, x[1]) + jitter,
        patch_seed_auroc,
        facecolors="none",
        edgecolors=BLUE,
        s=28,
        linewidths=0.9,
        zorder=2,
    )
    axes[0].set_xticks(x, [item[0] for item in models], rotation=22, ha="right")
    axes[0].set_ylim(0.5, 0.76)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("a  Strict SINAN cohort", loc="left", fontweight="bold")
    for position, value in zip(x, auroc, strict=True):
        axes[0].annotate(
            f"{value:.3f}",
            (position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].text(
        0.02,
        0.03,
        "Open circles show seven optimization-sensitivity seeds",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color=GRAY,
    )

    yearly_auroc = np.median(yearly_auroc_matrix, axis=0)
    yearly_auroc_min = np.min(yearly_auroc_matrix, axis=0)
    yearly_auroc_max = np.max(yearly_auroc_matrix, axis=0)
    yearly_order = np.median(yearly_order_matrix, axis=0)
    yearly_order_min = np.min(yearly_order_matrix, axis=0)
    yearly_order_max = np.max(yearly_order_matrix, axis=0)
    axes[1].fill_between(years, yearly_auroc_min, yearly_auroc_max, color=BLUE, alpha=0.16)
    axes[1].plot(years, yearly_auroc, marker="o", color=BLUE, label="Profile-score AUROC")
    axes[1].set_ylim(0.70, 0.75)
    axes[1].set_xticks(years)
    axes[1].set_ylabel("AUROC", color=BLUE)
    order_axis = axes[1].twinx()
    order_axis.fill_between(years, yearly_order_min, yearly_order_max, color=ORANGE, alpha=0.14)
    order_axis.plot(years, yearly_order, marker="s", color=ORANGE, label="Risk-order Spearman")
    order_axis.set_ylim(0.4, 1.05)
    order_axis.set_ylabel("Risk-order Spearman", color=ORANGE)
    axes[1].set_title("b  Year-specific median and seed range", loc="left", fontweight="bold")
    figure.suptitle(
        "Brazil supports prognostic transport, not transport of regimen coefficients",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, output_dir, "fig05_brazil_transport")


def render_k_sensitivity(input_dir: Path, output_dir: Path) -> None:
    """Render held-out discrimination across the prespecified K range."""
    report = _read_json(input_dir / "shared_k_sensitivity.json")
    candidates = [int(value) for value in report["candidate_k"]]
    metrics = report["oof_metrics"]
    values = [float(metrics[f"shared_k{k}"]["auroc"]) for k in candidates]
    figure, axis = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    axis.plot(candidates, values, marker="o", color=BLUE, linewidth=1.8)
    axis.scatter([5], [values[candidates.index(5)]], color=ORANGE, zorder=3, label="K=5 display")
    axis.axhline(float(metrics["additive"]["auroc"]), color=GRAY, linestyle="--", label="Additive")
    axis.set_xticks(candidates)
    axis.set_xlabel("Number of shared profiles")
    axis.set_ylabel("Held-out AUROC")
    upper = max(float(metrics["additive"]["auroc"]), max(values)) + 0.002
    axis.set_ylim(min(values) - 0.003, upper)
    axis.set_title("K sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.text(
        0.01,
        -0.26,
        "K=5 is a clinically useful descriptive resolution, not a uniquely identified class count.",
        transform=axis.transAxes,
        fontsize=8,
        color=GRAY,
    )
    _save(figure, output_dir, "figS_k_sensitivity")


def render_simulation(input_dir: Path, output_dir: Path) -> None:
    """Render coefficient recovery for the expanded twenty-seed simulation."""
    report = _read_json(input_dir / "expanded_simulation_benchmark.json")
    rows = [
        row
        for row in report["group_summary"]
        if int(row["n_train"]) == 1000 and row["scenario"] in {"additive_null", "low_rank"}
    ]
    model_order = [
        "additive_logistic_null",
        "residual_patch",
        "unrestricted_interaction_logistic",
    ]
    labels = ["Additive", "Residual PATCH", "Unrestricted interaction"]
    scenarios = ["additive_null", "low_rank"]
    scenario_labels = ["Additive null", "Low-rank heterogeneity"]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    for axis, scenario, scenario_label in zip(axes, scenarios, scenario_labels, strict=True):
        selected = {row["model"]: row for row in rows if row["scenario"] == scenario}
        means = [float(selected[model]["coefficient_rmse_mean"]) for model in model_order]
        errors = [float(selected[model]["coefficient_rmse_se"]) for model in model_order]
        axis.bar(np.arange(3), means, yerr=errors, color=PALETTE[:3], capsize=3)
        axis.set_xticks(np.arange(3), labels, rotation=20, ha="right")
        axis.set_ylabel("Coefficient-surface RMSE")
        axis.set_title(scenario_label, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    figure.suptitle(
        "Twenty-seed synthetic recovery benchmark at n=1,000",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, output_dir, "figS_simulation_benchmark")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render corrected PATCH figures from public aggregate JSON files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Render every disclosure-safe aggregate figure."""
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    _style()
    render_conceptual_schematic(output_dir)
    render_performance(input_dir, output_dir)
    render_profiles(input_dir, output_dir)
    render_association_heatmap(input_dir, output_dir)
    render_brazil(input_dir, output_dir)
    render_k_sensitivity(input_dir, output_dir)
    render_simulation(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
