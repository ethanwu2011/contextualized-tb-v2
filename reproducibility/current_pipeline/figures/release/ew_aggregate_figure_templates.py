"""Render Figures 2 and 5 from frozen safe aggregates.

This script does not read patient records or fitted per-patient artifacts. It
keeps the original panel arrangement for Figures 2 and 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "manuscript" / "aggregate_results"
OUT = ROOT / "manuscript" / "submission_figures" / "main"

INK = "#202124"
MUTED = "#535963"
GRID = "#E2E5E9"
PROFILE_COLORS = ["#009E73", "#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
PROFILE_SHORT = [
    "Drug-sensitive\nresponders",
    "Mixed-spectrum\nresistance responders",
    "MDR-predominant\nresponders",
    "High-burden\nregional failure",
    "HIV-enriched\nsevere-resistance failure",
]
TABLE_RANKS = [1, 2, 3, 5, 4]


def read_report(name: str) -> dict:
    """Read an aggregate report and check its declared disclosure boundary."""
    report = json.loads((REPORTS / name).read_text(encoding="utf-8"))
    governance = report.get("governance", {})
    assert governance.get("row_level_objects_serialized", False) is False
    assert governance.get("contains_row_level_data", False) is False
    assert governance.get("contains_identifiers", False) is False
    return report


def set_style() -> None:
    """Apply the shared print style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure and register its source lineage."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{name}.pdf",
        metadata={
            "Creator": "PATCH shared-primary aggregate renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def percent(row: dict, key: str, category: str) -> float:
    """Return an aggregate percentage for a named category."""
    value = row[key][category]["percent"]
    return float(value) if value is not None else float("nan")


def figure2() -> None:
    """Original EW organization, top outcome and composition, bottom modalities."""
    report = read_report("shared_profile_reference_composition.json")
    assert report["cohort"]["assessed_conditions"] == 9647
    by_rank = {int(row["risk_rank"]): row for row in report["profiles"]}
    rows = [by_rank[rank] for rank in TABLE_RANKS]
    counts = np.asarray([int(row["hard_count"]) for row in rows])
    assert int(counts.sum()) == 9647
    rates = 100 * np.asarray([float(row["hard_event_rate"]) for row in rows])

    composition_labels = [
        "Sensitive",
        "MDR",
        "XDR",
        "HIV+",
        "Smear+",
        "Ukraine",
        "Georgia",
        "Age",
    ]
    values = np.asarray(
        [
            [
                percent(row, "resistance_composition", "Sensitive"),
                percent(row, "resistance_composition", "MDR non XDR"),
                percent(row, "resistance_composition", "XDR"),
                float(row["hiv"]["percent_among_known"]),
                float(row["smear_positive"]["percent_among_known"]),
                percent(row, "country_composition", "Ukraine"),
                percent(row, "country_composition", "Georgia"),
                float(row["age_mean_among_known"]),
            ]
            for row in rows
        ]
    )
    z = np.zeros_like(values)
    for j in range(values.shape[1]):
        column = values[:, j]
        mean = np.nanmean(column)
        sd = np.nanstd(column)
        z[:, j] = (np.where(np.isnan(column), mean, column) - mean) / max(sd, 1e-9)

    fig = plt.figure(figsize=(7.0, 5.0))
    ax_a = fig.add_axes([0.245, 0.555, 0.245, 0.360])
    # Leave a narrow label gutter so each heatmap row names its profile directly.
    ax_b = fig.add_axes([0.635, 0.555, 0.295, 0.360])
    ax_c = fig.add_axes([0.09, 0.140, 0.84, 0.250])

    y = np.arange(5)
    ax_a.barh(y, rates, color=PROFILE_COLORS, height=0.67)
    ax_a.set_yticks(
        y,
        [f"{label}\n(n={count:,})" for label, count in zip(PROFILE_SHORT, counts, strict=True)],
        fontsize=6.8,
    )
    ax_a.tick_params(axis="y", length=0, pad=4)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 48)
    ax_a.set_xlabel("Recorded death or failure (%)", fontsize=8)
    ax_a.set_title("a  Outcome by profile", loc="left", fontsize=9.1, weight="bold", pad=9)
    ax_a.grid(axis="x", color=GRID, linewidth=0.6)
    ax_a.set_axisbelow(True)
    for yy, rate in zip(y, rates, strict=True):
        ax_a.text(rate + 0.8, yy, f"{rate:.1f}%", va="center", fontsize=7.2)

    image = ax_b.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax_b.set_yticks(
        y,
        [
            "Drug-sensitive",
            "Mixed resistance",
            "MDR-predominant",
            "High-burden",
            "HIV-enriched",
        ],
        fontsize=5.9,
    )
    ax_b.tick_params(axis="y", length=0, pad=4)
    ax_b.set_xticks(np.arange(8), composition_labels, rotation=40, ha="right", fontsize=7.1)
    ax_b.tick_params(axis="x", length=0, pad=4)
    ax_b.set_title(
        "b  Defining clinical composition",
        loc="left",
        fontsize=9.1,
        weight="bold",
        pad=9,
    )
    for i in range(5):
        for j in range(8):
            value = values[i, j]
            if np.isnan(value):
                ax_b.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="////",
                        edgecolor=MUTED,
                        linewidth=0,
                    )
                )
                label = "—"
            elif j == 7:
                label = f"{value:.0f}y"
            else:
                label = f"{value:.0f}"
            color = "white" if abs(z[i, j]) > 1.15 and not np.isnan(value) else INK
            ax_b.text(j, i, label, ha="center", va="center", fontsize=6.7, color=color)
    cax = fig.add_axes([0.947, 0.605, 0.011, 0.260])
    cb = fig.colorbar(image, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.ax.set_title("z", fontsize=6.5)

    modality_names = {
        "resistance_dst": "Resistance / DST",
        "sociodemographic": "Sociodemographic",
        "region": "Region",
        "bacillary_burden_clinical": "Burden / clinical",
        "radiology_imaging": "Imaging",
        "comorbidity": "Comorbidity",
        "genomics": "Genomics",
        "baseline_laboratory": "Laboratory",
        "other_clinical": "Other clinical",
    }
    modalities = sorted(
        report["modality_attribution"], key=lambda item: -float(item["total_share"])
    )
    total = 100 * np.asarray([float(item["total_share"]) for item in modalities])
    per_feature = 100 * np.asarray([float(item["per_feature_share"]) for item in modalities])
    x = np.arange(len(modalities))
    original_hues = [
        "#8B1E52",
        "#352B83",
        "#A49E25",
        "#A04490",
        "#D7BE65",
        "#C45C69",
        "#168342",
        "#777777",
        "#777777",
    ]
    ax_c.bar(x - 0.18, total, width=0.36, color=original_hues, label="Total share")
    ax_c.bar(
        x + 0.18,
        per_feature,
        width=0.36,
        color="#E7E8EB",
        edgecolor="#A2A5AA",
        hatch="///",
        linewidth=0.5,
        label="Per-feature share",
    )
    ax_c.set_ylim(0, max(40, float(max(total.max(), per_feature.max())) + 3))
    ax_c.set_ylabel("Contribution (%)", fontsize=8)
    ax_c.set_xticks(
        x,
        [modality_names[item["modality"]] for item in modalities],
        rotation=30,
        ha="right",
        fontsize=6.8,
    )
    ax_c.set_title(
        "c  Context modality contribution",
        loc="left",
        fontsize=9.1,
        weight="bold",
        pad=8,
    )
    ax_c.grid(axis="y", color=GRID, linewidth=0.6)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
    save(fig, "fig02_EW_labeled")


def figure5() -> None:
    """Original two-panel EW layout, corrected strict SINAN analysis."""
    report = read_report("brazil_source_rebuild_audit.json")
    assert (
        report["cohort_rebuild"]["strict_primary_conditions_excluding_postmortem_entry"] == 296079
    )
    metrics = report["external_discrimination"]["strict_primary_excluding_postmortem"]
    auc_resistance = float(metrics["resistance_only"]["auroc"])
    auc_patch = float(metrics["patch_profile_score"]["auroc"])
    seed = next(row for row in report["profile_transport"]["seeds"] if row["seed"] == 0)
    source = sorted(seed["source_profile_summary"], key=lambda row: int(row["profile_risk_rank"]))
    brazil = sorted(
        seed["brazil_strict_profile_summary"],
        key=lambda row: int(row["profile_risk_rank"]),
    )
    assert len(source) == len(brazil) == 5
    assert all(float(row["soft_effective_count"]) >= 30 for row in [*source, *brazil])
    source_rates = 100 * np.asarray([float(row["soft_event_rate"]) for row in source])
    brazil_rates = 100 * np.asarray([float(row["soft_event_rate"]) for row in brazil])

    fig = plt.figure(figsize=(7.0, 2.70))
    ax_a = fig.add_axes([0.155, 0.25, 0.305, 0.58])
    ax_b = fig.add_axes([0.57, 0.25, 0.39, 0.58])
    ax_a.barh(
        [1, 0],
        [auc_resistance - 0.5, auc_patch - 0.5],
        left=0.5,
        height=0.50,
        color=["#A8A9AC", "#4D5C72"],
    )
    ax_a.set_yticks([1, 0], ["Resistance class", "PATCH profile score"], fontsize=7.5)
    ax_a.set_xlim(0.50, 0.79)
    ax_a.set_xticks([0.50, 0.60, 0.70, 0.80])
    ax_a.set_xlabel("AUROC", fontsize=8)
    ax_a.set_title("a  External discrimination", loc="left", fontsize=9.1, weight="bold", pad=9)
    ax_a.tick_params(axis="y", length=0, pad=4)
    for y, value in [(1, auc_resistance), (0, auc_patch)]:
        ax_a.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=7.6)
    ax_a.grid(axis="x", color=GRID, linewidth=0.6)
    ax_a.set_axisbelow(True)

    x = np.arange(5)
    width = 0.35
    for i in range(5):
        color = PROFILE_COLORS[i]
        ax_b.bar(x[i] - width / 2, source_rates[i], width, color=color)
        ax_b.bar(
            x[i] + width / 2,
            brazil_rates[i],
            width,
            facecolor="white",
            edgecolor=color,
            hatch="///",
            linewidth=0.9,
        )
    ax_b.set_xticks(x, ["Low", "2", "3", "4", "High"], fontsize=7.2)
    ax_b.set_ylim(0, max(34, float(max(source_rates.max(), brazil_rates.max())) + 3))
    ax_b.set_ylabel("Recorded death or failure (%)", fontsize=7.5)
    ax_b.set_title(
        "b  Profile risk in both registries",
        loc="left",
        fontsize=9.1,
        weight="bold",
        pad=9,
    )
    ax_b.legend(
        handles=[
            Patch(facecolor="#4D5C72", label="TB Portals"),
            Patch(facecolor="white", edgecolor="#4D5C72", hatch="///", label="Brazil"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    ax_b.grid(axis="y", color=GRID, linewidth=0.6)
    ax_b.set_axisbelow(True)
    fig.text(
        0.01,
        0.965,
        "Strict SINAN cohort, n = 296,079",
        fontsize=8,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.995,
        0.035,
        "Profiles ordered by source risk using seven harmonized context fields",
        fontsize=6.8,
        color=MUTED,
        ha="right",
    )
    save(fig, "fig05_EW_corrected")


def main() -> None:
    """Parse paths and render the aggregate figures."""
    set_style()
    figure2()
    figure5()
    print("Built corrected EW-style Figures 2 and 5 from frozen aggregate reports.")


if __name__ == "__main__":
    main()
