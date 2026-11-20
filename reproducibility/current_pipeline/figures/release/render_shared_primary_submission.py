"""Render EW-layout PATCH figures from disclosure-safe aggregate source data only.

No patient row, prediction, latent artifact, embedding or private fitted model is
read here. Main figures preserve EW's panel sequence and canonical clinical names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    Patch,
    Polygon,
    Rectangle,
    Wedge,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "manuscript" / "submission_figures"
OUTPUT_ROOT = HERE
DEFAULT_FROZEN = ROOT / "manuscript" / "aggregate_results"

COLORS = ["#009E73", "#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
NAMES = [
    "Drug-sensitive\nresponders",
    "Mixed-spectrum resistance\nresponders",
    "MDR-predominant\nresponders",
    "High-burden\nregional failure",
    "HIV-enriched severe-\nresistance failure",
]
SHORT = [
    "Drug-sensitive",
    "Mixed resistance",
    "MDR-predominant",
    "High-burden",
    "HIV-enriched",
]
RANKS = [1, 2, 3, 5, 4]
INK = "#202124"
MUTED = "#62676D"
GRID = "#E4E7EA"
TEAL = "#217C7E"
BLUE = "#0072B2"
PURPLE = "#882255"
MODALITY = {
    "resistance_dst": "#8B1E52",
    "sociodemographic": "#352B83",
    "region": "#A49E25",
    "bacillary_burden_clinical": "#A04490",
    "radiology_imaging": "#D7BE65",
    "comorbidity": "#C45C69",
    "genomics": "#168342",
    "baseline_laboratory": "#777777",
    "other_clinical": "#777777",
}
MANIFEST = []


def read(root, name):
    """Read one disclosure-gated aggregate report."""
    value = json.loads((root / name).read_text())
    gov = value.get("governance", {})
    assert not gov.get("row_level_objects_serialized", False)
    assert not gov.get("identifiers_serialized", False)
    assert not gov.get("contains_row_level_data", False)
    assert not gov.get("contains_identifiers", False)
    return value


def save(fig, out, stem, sources, panels):
    """Save the figure and register its source lineage."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out / f"{stem}.pdf",
        metadata={
            "Creator": "PATCH shared-primary aggregate renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(out / f"{stem}.png", dpi=600)
    MANIFEST.append(
        {
            "figure": stem,
            "files": [str((out / f"{stem}.{s}").relative_to(OUTPUT_ROOT)) for s in ["pdf", "png"]],
            "sources": [str(s) for s in sources],
            "panels": panels,
            "vector_pdf": True,
            "png_dpi": 600,
        }
    )
    plt.close(fig)


def title(ax, letter, text):
    """Add a panel letter and title."""
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=9.1, weight="bold", pad=10)


def simple_axes(ax):
    """Apply the common print axis style."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def schematic(frozen, out):
    """Square EW four-panel reading path; all dots and profile bars illustrative."""
    fig = plt.figure(figsize=(7.4, 7.6))
    axa = fig.add_axes([0.055, 0.64, 0.42, 0.29])
    axb = fig.add_axes([0.555, 0.64, 0.42, 0.29])
    rng = np.random.default_rng(4)
    anchors = np.array([[0.12, 0.75], [0.85, 0.80], [0.75, 0.14], [0.18, 0.15], [0.5, 0.45]])
    pts = np.clip(np.vstack([a + rng.normal(0, 0.11, (7, 2)) for a in anchors]), 0.05, 0.95)
    for ax, letter, heading in [
        (axa, "a", "Conventional: group summaries"),
        (axb, "b", "PATCH: continuous response profiles"),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 1.05)
        simple_axes(ax)
        title(ax, letter, heading)
    for k in range(5):
        if k < 3:
            x0, y0, width = k / 3, 0.55, 1 / 3
        else:
            x0, y0, width = (k - 3) / 2, 0.15, 0.5
        axa.add_patch(
            Rectangle(
                (x0, y0),
                width,
                0.4,
                facecolor=COLORS[k],
                alpha=0.15,
                edgecolor=INK,
                lw=0.7,
            )
        )
    axa.scatter(
        pts[:, 0],
        0.2 + 0.7 * pts[:, 1],
        s=16,
        marker="s",
        c=[
            COLORS[min(2, int(x * 3)) if 0.2 + 0.7 * y > 0.55 else 3 + min(1, int(x * 2))]
            for x, y in pts
        ],
        alpha=0.5,
        edgecolors="white",
        lw=0.4,
    )
    axa.text(
        0.5,
        -0.06,
        "One coefficient profile per group",
        ha="center",
        fontsize=8,
        weight="bold",
    )
    axb.add_patch(
        Polygon(
            [[0.02, 0.88], [0.97, 0.90], [0.78, 0.18], [0.16, 0.18]],
            facecolor="#DFE8EB",
            edgecolor="#ADB7BD",
            lw=1,
        )
    )
    for x, y in pts:
        w = np.exp(-((anchors - [x, y]) ** 2).sum(axis=1) / 0.12)
        w /= w.sum()
        col = w @ np.array([mcolors.to_rgb(c) for c in COLORS])
        axb.scatter([x], [0.20 + 0.66 * y], s=22, color=col, edgecolor="white", lw=0.5)
    for k, (x, y) in enumerate(anchors):
        axb.add_patch(
            Circle(
                (x, 0.2 + 0.66 * y),
                0.055,
                facecolor="white",
                edgecolor=COLORS[k],
                lw=1.4,
            )
        )
        axb.text(
            x,
            0.2 + 0.66 * y,
            f"A{k+1}",
            ha="center",
            va="center",
            fontsize=6.8,
            weight="bold",
        )
    axb.text(
        0.5,
        -0.06,
        "Each episode is a mixture of five profiles",
        ha="center",
        fontsize=8,
        weight="bold",
    )
    ax = fig.add_axes([0.055, 0.315, 0.92, 0.25])
    simple_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title(ax, "c", "From recorded context and regimen to modeled risk")
    centers = [0.105, 0.36, 0.625, 0.875]
    headings = [
        "1  Recorded context",
        "2  Shared profile mixture",
        "3  Regimen associations",
        "4  Predicted risk",
    ]
    for x, label in zip(centers, headings, strict=False):
        ax.text(x, 0.88, label, ha="center", fontsize=7.5, weight="bold")
    ax.text(
        centers[0],
        0.48,
        "Imaging · laboratory\nDrug susceptibility\nDemographics\nComorbidities · genomics",
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.6,
    )
    mixture = [0.30, 0.20, 0.10, 0.25, 0.15]
    left = 0.252
    for k, v in enumerate(mixture):
        ax.add_patch(
            Rectangle(
                (left, 0.41),
                v * 0.215,
                0.16,
                facecolor=COLORS[k],
                edgecolor="white",
                lw=0.6,
            )
        )
        left += v * 0.215
    ax.text(centers[1], 0.28, "Z(C) sums to 1", ha="center", fontsize=7.5, color=MUTED)
    bars = [0.10, -0.08, 0.06, -0.045, -0.09]
    for k, v in enumerate(bars):
        ax.barh(
            0.65 - k * 0.075,
            v,
            left=0.625,
            color=BLUE if v < 0 else "#C45C69",
            height=0.047,
        )
    ax.plot([0.625, 0.625], [0.28, 0.70], color=MUTED, lw=0.6)
    ax.text(
        centers[2],
        0.15,
        "Lower / higher modeled risk",
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )
    ax.add_patch(Wedge((0.875, 0.40), 0.09, 0, 180, width=0.027, facecolor="#DFE2E5"))
    ax.add_patch(Wedge((0.875, 0.40), 0.09, 0, 115, width=0.027, facecolor=TEAL))
    ax.plot([0.875, 0.85], [0.40, 0.46], color=INK, lw=1.2)
    ax.text(0.875, 0.27, "Illustrative episode", ha="center", fontsize=7.2, color=MUTED)
    for x in [0.22, 0.49, 0.75]:
        ax.add_patch(
            FancyArrowPatch(
                (x, 0.5),
                (x + 0.025, 0.5),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.2,
                color=INK,
            )
        )
    fig.text(
        0.515,
        0.303,
        r"logit P(Y = 1 | X, C) = Xβ(C) + μ(C);   β(C) = Z(C)A",
        ha="center",
        fontsize=8,
    )
    report = read(frozen, "shared_profile_full_pipeline_stability.json")
    assert report["profile_by_regimen_uncertainty"]["cells_supported_by_soft_marginal_gate"] == 104
    ax = fig.add_axes([0.055, 0.045, 0.92, 0.205])
    simple_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title(ax, "d", "Audited evidence")
    items = [
        (104, 130, "Marginally supported\nprofile–regimen cells"),
        (69, 104, "Supported cells with\n≥80% sign agreement"),
        (37, 104, "Supported cells with\n95% interval excluding zero"),
    ]
    for x, (n, d, label) in zip([0.115, 0.36, 0.60], items, strict=False):
        size = 0.11
        donut = fig.add_axes(
            [
                0.055 + 0.92 * x - size / 2,
                0.045 + 0.205 * 0.58 - size * 7.4 / 7.6 / 2,
                size,
                size * 7.4 / 7.6,
            ]
        )
        donut.pie(
            [n, d - n],
            colors=[TEAL, "#E5E7EB"],
            radius=1,
            wedgeprops={"width": 0.28, "linewidth": 0},
            startangle=90,
        )
        donut.text(0, 0, f"{n}/{d}", ha="center", va="center", fontsize=8.4, weight="bold")
        ax.text(x, 0.28, label, ha="center", va="top", fontsize=7.1, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.text(0.85, 0.64, "AUROC 0.724", ha="center", fontsize=11, weight="bold", color=TEAL)
    ax.text(
        0.85,
        0.42,
        "Brazil strict cohort\nn = 296,079\nPrognostic transport",
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.5,
    )
    fig.text(
        0.52,
        0.012,
        (
            "Regularized observational associations; no regimen component "
            "passed the conditional support gate."
        ),
        ha="center",
        fontsize=7,
        color=MUTED,
    )
    save(
        fig,
        out,
        "fig01_shared_schematic",
        [
            frozen / "shared_profile_full_pipeline_stability.json",
            frozen / "brazil_source_rebuild_audit.json",
        ],
        {
            "a-c": "data-free illustrative schematic; unchanged model factorization",
            "d": "20 full-pipeline bootstrap summaries and strict Brazil seed0 AUROC",
        },
    )


def geometry(data, out):
    """Draw supported tiles from the aggregate shared-profile embedding."""
    report = read(data, "shared_geometry_aggregate.json")
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(1, 5, left=0.055, right=0.985, bottom=0.34, top=0.85, wspace=0.22)
    specs = [
        ("profile", "Response\nprofile"),
        ("resistance", "Resistance\nclass"),
        ("hiv_prevalence", "HIV\nprevalence"),
        ("smear_prevalence", "Smear\npositivity"),
        ("region", "Region\n "),
    ]
    resistance = ["Sensitive", "Mono DR", "Poly DR", "MDR non XDR", "Pre-XDR", "XDR"]
    regions = ["Other", "Ukraine", "Georgia", "Nigeria", "South Africa"]
    rcolors = ["#BDBDBD", "#3F348B", "#D9C66F", "#009E73", "#56B4E9"]
    for idx, (key, label) in enumerate(specs):
        ax = fig.add_subplot(gs[idx])
        grid = np.full((12, 12), np.nan)
        for row in report["tiles"]:
            value = row.get(key)
            if value is None:
                continue
            if key == "resistance":
                value = resistance.index(value) if value in resistance else np.nan
            elif key == "region":
                value = regions.index(value)
            grid[row["ybin"], row["xbin"]] = value
        if key == "profile":
            cmap = mcolors.ListedColormap(COLORS)
            vmin = -0.5
            vmax = 4.5
        elif key == "region":
            cmap = mcolors.ListedColormap(rcolors)
            vmin = -0.5
            vmax = 4.5
        elif key == "resistance":
            cmap = mcolors.ListedColormap(plt.get_cmap("Purples")(np.linspace(0.18, 0.92, 6)))
            vmin = -0.5
            vmax = 5.5
        else:
            cmap = plt.get_cmap("Reds" if idx == 2 else "Purples").copy()
            vmin = 0
            vmax = 1
        cmap.set_bad("#F4F5F6")
        im = ax.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        title(ax, "abcde"[idx], label)
        ax.title.set_fontsize(7.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=7)
        if idx == 0:
            ax.set_ylabel("UMAP 2", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        if idx in [2, 3]:
            pos = ax.get_position()
            cb = fig.colorbar(
                im,
                cax=fig.add_axes([pos.x0, 0.29, pos.width, 0.014]),
                orientation="horizontal",
                ticks=[0, 0.5, 1],
            )
            cb.ax.set_xticklabels(["0%", "50%", "100%"], fontsize=6)
    fig.legend(
        handles=[Patch(color=c, label=n) for c, n in zip(COLORS, SHORT, strict=False)],
        loc="lower left",
        bbox_to_anchor=(0.055, 0.15),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        columnspacing=1.1,
    )
    fig.legend(
        handles=[Patch(color=c, label=n) for c, n in zip(rcolors, regions, strict=False)],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.115),
        ncol=2,
        frameon=False,
        fontsize=6.3,
        columnspacing=0.8,
    )
    fig.legend(
        handles=[
            Patch(color=c, label=n)
            for c, n in zip(
                plt.get_cmap("Purples")(np.linspace(0.18, 0.92, 6)),
                ["Sensitive", "Mono-DR", "Poly-DR", "MDR", "Pre-XDR", "XDR"],
                strict=False,
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.055, 0.095),
        ncol=6,
        frameon=False,
        fontsize=6.1,
        columnspacing=0.9,
    )
    fig.text(
        0.055,
        0.075,
        "Common embedding across panels. Each visible category summarizes ≥30 conditions.",
        fontsize=7,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.03,
        (
            "HIV/smear fractions require ≥30 positive and ≥30 negative "
            "observations; grey tiles are suppressed."
        ),
        fontsize=6.7,
        color=MUTED,
    )
    save(
        fig,
        out,
        "fig03_shared_geometry",
        [data / "shared_geometry_aggregate.json"],
        {
            "a-e": "same private shared-reference UMAP, aggregate tiles only",
            "b": "dominant resistance class; ordered light sensitive to dark XDR",
        },
    )


def feature_label(name):
    """Format an encoded context feature while retaining its category."""
    text = unquote(name).replace("__", ": ").replace("_", " ")
    for old, new in [
        ("type of resistance: ", "Resistance: "),
        ("baseline smear: ", "Smear: "),
        ("comorb ", "Comorbidity: "),
        ("country: ", "Region: "),
        ("main lineage: ", "Lineage: "),
        ("sub lineage: ", "Sublineage: "),
    ]:
        if text.lower().startswith(old):
            text = new + text[len(old) :]
            break
    text = text.replace("bactec ", "DST ").replace(
        "aminoglycosides injectable agents", "injectable aminoglycosides"
    )
    if len(text) > 41:
        if ": " in text:
            head, tail = text.rsplit(": ", 1)
            text = head[: max(1, 38 - len(tail))] + "…: " + tail
        else:
            text = text[:40] + "…"
    return text


def profile_matrix(frozen):
    """Arrange supported coefficients in canonical display order."""
    report = read(frozen, "shared_profile_full_pipeline_stability.json")
    cells = report["profile_by_regimen_uncertainty"]["cells"]
    composition = read(frozen, "shared_profile_reference_composition.json")
    profiles = [
        int(next(p["fitted_profile"] for p in composition["profiles"] if p["risk_rank"] == r))
        for r in RANKS
    ]
    drugs = list(dict.fromkeys(r["regimen_feature"] for r in cells))
    labels = [next(r["regimen_label"] for r in cells if r["regimen_feature"] == d) for d in drugs]
    matrix = np.full((5, len(drugs)), np.nan)
    rows = {}
    for i, p in enumerate(profiles):
        for j, d in enumerate(drugs):
            r = next(r for r in cells if r["profile"] == p and r["regimen_feature"] == d)
            rows[(i, j)] = r
            if r["support_status"] == "supported":
                matrix[i, j] = r["reference_shared_logit_coefficient"]
    return report, drugs, labels, matrix, rows


def atlas(data, frozen, out, supp):
    """Draw the supported context and regimen-association atlas panels."""
    r = read(data, "shared_context_atlas.json")
    report, drugs, labels, matrix, cells = profile_matrix(frozen)
    indexes = [r["regimen_features"].index(d) for d in drugs]
    allrows = r["context_features"]
    selected = [allrows[i] for i in r["main_figure_feature_indices"]]

    def render_context(ax, rows, fontsize=5.8):
        values = np.array(
            [[np.nan if v is None else v for v in row["covariation"]] for row in rows]
        )[:, indexes]
        bound = max(0.01, float(np.nanpercentile(abs(values), 98)))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#E4E7EB")
        im = ax.imshow(
            values,
            cmap=cmap,
            vmin=-bound,
            vmax=bound,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(
            np.arange(len(rows)),
            [feature_label(row["feature"]) for row in rows],
            fontsize=fontsize,
        )
        for tick, row in zip(ax.get_yticklabels(), rows, strict=False):
            tick.set_color(MODALITY[row["modality"]])
        ax.set_xticks(np.arange(26), labels, rotation=60, ha="right", fontsize=5.7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        return im

    fig = plt.figure(figsize=(7.6, 8.6))
    axa = fig.add_axes([0.225, 0.205, 0.36, 0.725])
    axb = fig.add_axes([0.70, 0.78, 0.28, 0.145])
    axc = fig.add_axes([0.755, 0.465, 0.215, 0.105])
    axd = fig.add_axes([0.745, 0.17, 0.225, 0.18])
    im = render_context(axa, selected)
    title(axa, "a", f"Context–regimen atlas ({len(selected)} features)")
    cb = fig.colorbar(im, cax=fig.add_axes([0.23, 0.075, 0.345, 0.012]), orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Context–coefficient covariation per SD", fontsize=6.5)
    bound = max(0.1, float(np.nanpercentile(abs(matrix), 98)))
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#E4E7EB")
    im2 = axb.imshow(matrix, aspect="auto", cmap=cm, vmin=-bound, vmax=bound)
    axb.set_yticks(np.arange(5), SHORT, fontsize=5.8)
    axb.set_xticks(np.arange(26), labels, rotation=65, ha="right", fontsize=5.3)
    axb.tick_params(length=0)
    for (i, j), cell in cells.items():
        if cell["support_status"] != "supported":
            continue
        if cell["interval_excludes_zero"]:
            axb.text(j, i, "*", ha="center", va="center", fontsize=7)
        elif cell["sign_agreement_at_least_0.80"]:
            axb.text(j, i, "·", ha="center", va="center", fontsize=9)
    title(axb, "b", "Shared profile–regimen\ncoefficients")
    axb.title.set_fontsize(8.3)
    cb2 = fig.colorbar(im2, cax=fig.add_axes([0.71, 0.665, 0.26, 0.009]), orientation="horizontal")
    cb2.ax.tick_params(labelsize=5.7)
    cb2.set_label("Regularized logit coefficient", fontsize=6.2)
    axc.barh([2, 1, 0], [104, 69, 37], height=0.6, color=[TEAL, BLUE, PURPLE])
    axc.set_yticks(
        [2, 1, 0],
        ["Marginal support", "Sign agreement ≥80%", "95% interval excludes 0"],
        fontsize=5.8,
    )
    axc.set_xlim(0, 140)
    axc.set_xticks([0, 65, 130], ["0", "65", "130"], fontsize=6)
    axc.set_xlabel("Cells", fontsize=6.5)
    for y, v in zip([2, 1, 0], [104, 69, 37], strict=False):
        axc.text(v + 2, y, str(v), va="center", fontsize=6.2)
    title(axc, "c", "Support and\nuncertainty")
    axc.title.set_fontsize(8)
    fig.text(
        0.70,
        0.405,
        "Conditional support gate: 0/26 components",
        fontsize=5.9,
        color=PURPLE,
    )
    # Clinically familiar components selected before inspecting intervals;
    # show all supported profiles for each pair.
    examples = []
    for name in ["R", "Lfx", "Bdq", "Lzd"]:
        j = labels.index(name)
        candidates = [
            (i, cells[(i, j)]) for i in range(5) if cells[(i, j)]["support_status"] == "supported"
        ]
        if candidates:
            examples.append(
                (
                    name,
                    *max(
                        candidates,
                        key=lambda pair: abs(pair[1]["reference_shared_logit_coefficient"]),
                    ),
                )
            )
    for y, (_drug, i, cell) in zip(np.arange(len(examples))[::-1], examples, strict=False):
        value = cell["reference_shared_logit_coefficient"]
        lo, hi = cell["bootstrap_ci95"]
        # The reference can lie outside a percentile bootstrap interval; draw endpoints directly.
        axd.plot([lo, hi], [y, y], color=COLORS[i], lw=1.1)
        axd.plot(value, y, "o", color=COLORS[i], ms=3)
    axd.set_yticks(
        np.arange(len(examples))[::-1],
        [f"{drug} · {SHORT[i]}" for drug, i, _ in examples],
        fontsize=5.8,
    )
    axd.axvline(0, color=MUTED, lw=0.6)
    axd.set_xlabel("Logit coefficient (95% interval)", fontsize=6.2)
    axd.tick_params(axis="x", labelsize=5.8)
    title(axd, "d", "Illustrative supported\nassociations")
    axd.title.set_fontsize(8)
    fig.text(
        0.225,
        0.025,
        (
            "Grey cells: suppressed support. * 95% bootstrap interval "
            "excludes zero; · sign agreement ≥80%."
        ),
        fontsize=6.1,
        color=MUTED,
    )
    fig.text(
        0.225,
        0.009,
        (
            "Twenty full-pipeline bootstraps; exploratory intervals, no "
            "multiplicity-adjusted significance claims."
        ),
        fontsize=6.1,
        color=MUTED,
    )
    save(
        fig,
        out,
        "fig04_shared_atlas",
        [
            data / "shared_context_atlas.json",
            frozen / "shared_profile_full_pipeline_stability.json",
        ],
        {
            "a": r["definition"],
            "b": (
                "104 marginally supported profile–regimen coefficients; 26 "
                "suppressed; canonical profile order"
            ),
            "c": "support and 20 full-pipeline-bootstrap uncertainty summaries",
            "d": (
                "R, Lfx, Bdq, Lzd examples: largest absolute supported reference "
                "coefficient within each preselected component; selection "
                "descriptive"
            ),
        },
    )
    # Full row atlas uses multiple pages/assets to retain label readability.
    for page, start in enumerate(range(0, len(allrows), 65), 1):
        rows = allrows[start : start + 65]
        fig = plt.figure(figsize=(7.2, max(5.2, len(rows) * 0.112 + 1.3)))
        ax = fig.add_axes([0.31, 0.21, 0.65, 0.71])
        im = render_context(ax, rows, 6.4)
        title(ax, "a", f"Full supported context atlas · part {page}")
        cb = fig.colorbar(im, cax=fig.add_axes([0.32, 0.07, 0.63, 0.012]), orientation="horizontal")
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(
            "Context–coefficient covariation per SD",
            fontsize=6.4,
        )
        save(
            fig,
            supp,
            f"figS05_full_context_atlas_part{page}",
            [data / "shared_context_atlas.json"],
            {
                "all": (
                    "all eligible encoded features, same support gate and row-free "
                    "aggregation as main Figure4a"
                )
            },
        )


def dca(data, out, stem="fig06_shared_monitoring"):
    """Draw shared-primary internal risk-monitoring diagnostics."""
    r = read(data, "shared_oof_clinical_validation.json")
    rows = r["threshold_rows"]
    t = 100 * np.array([x["threshold"] for x in rows])
    chosen = min(rows, key=lambda r: abs(r["threshold"] - 0.2))
    fig = plt.figure(figsize=(7.1, 2.95))
    a = fig.add_axes([0.085, 0.25, 0.465, 0.59])
    b = fig.add_axes([0.735, 0.25, 0.245, 0.59])
    for key, col, label in [
        ("shared_k5", BLUE, "Shared PATCH"),
        ("additive", PURPLE, "Additive context + regimen"),
    ]:
        a.plot(t, [x[key + "_net_benefit"] for x in rows], color=col, lw=1.7, label=label)
    a.plot(
        t,
        [x["monitor_all_net_benefit"] for x in rows],
        color="#A5A5A5",
        ls="--",
        lw=1,
        label="Monitor all",
    )
    a.axhline(0, color=MUTED, ls=":", lw=0.8, label="Monitor none")
    a.set_xlim(5, 40)
    a.set_ylim(-0.025, 0.16)
    a.set_xlabel("Threshold for intensified monitoring (%)", fontsize=7.4)
    a.set_ylabel("Net benefit", fontsize=7.5)
    a.legend(frameon=False, fontsize=6.5, loc="upper right")
    a.grid(axis="y", color=GRID, lw=0.6)
    title(a, "a", "Internal out-of-fold decision curves")
    alltp = 1000 * r["cohort"]["events"] / r["cohort"]["conditions"]
    for y, key, col in [
        (2, "all", MUTED),
        (1, "additive", PURPLE),
        (0, "shared_k5", BLUE),
    ]:
        tp = alltp if key == "all" else chosen[key + "_event_per_1000"]
        fp = 1000 - alltp if key == "all" else chosen[key + "_nonevent_per_1000"]
        b.barh(y, tp, color=col, height=0.56)
        b.barh(y, fp, left=tp, color=col, alpha=0.22, height=0.56)
        b.text(tp + fp + 12, y, f"{tp+fp:.0f}", va="center", fontsize=7)
    b.set_yticks([2, 1, 0], ["Monitor all", "Additive", "Shared PATCH"], fontsize=7)
    b.set_xlim(0, 1100)
    b.set_xticks([0, 500, 1000], ["0", "500", "1000"], fontsize=7)
    b.set_xlabel("Flagged per 1,000 at 20%", fontsize=7.2)
    title(b, "b", "Monitoring yield")
    b.legend(
        handles=[
            Patch(color=MUTED, label="Event"),
            Patch(color=MUTED, alpha=0.22, label="No event"),
        ],
        frameon=False,
        fontsize=6.2,
        loc="lower right",
    )
    lo, hi = chosen["shared_minus_additive_ci95"]
    delta = chosen["shared_minus_additive_net_benefit"]
    fig.text(
        0.085,
        0.06,
        (
            f"At 20%: shared minus additive net benefit {delta:.5f} "
            f"(patient-bootstrap 95% interval {lo:.5f} to {hi:.5f})."
        ),
        fontsize=6.4,
        color=MUTED,
    )
    fig.text(
        0.085,
        0.018,
        (
            "Exploratory prognostic risk targeting; no intervention benefit "
            "or treatment-selection utility was tested."
        ),
        fontsize=6.4,
        color=MUTED,
    )
    save(
        fig,
        out,
        stem,
        [data / "shared_oof_clinical_validation.json"],
        {
            "a": (
                "same shared K=5 OOF probabilities as frozen shared K "
                "sensitivity; five patient-grouped folds"
            ),
            "b": "direct aggregate event/non-event flagged counts, 20% threshold",
        },
    )


def summary_metrics(ax, source, keys, labels, heading):
    """Plot aggregate model metrics on a common axis."""
    yy = np.arange(len(keys))[::-1]
    for y, key, col in zip(yy, keys, [TEAL, BLUE, PURPLE, "#E69F00", MUTED], strict=False):
        item = source[key]
        m = item["median"]
        lo, hi = item["ci95"]
        ax.plot([lo, hi], [y, y], color=col, lw=1.2)
        ax.plot(m, y, "o", color=col, ms=4)
    ax.set_yticks(yy, labels, fontsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("Median and percentile interval", fontsize=7)
    title(ax, "a", heading)


def supplements(frozen, data, out):
    """Render supplementary evidence from aggregate reports."""
    s = read(frozen, "shared_profile_full_pipeline_stability.json")
    source = frozen / "shared_profile_full_pipeline_stability.json"
    keys = [
        "canonical_beta_surface_cosine",
        "mean_matched_soft_loading_pearson",
        "fitted_probability_pearson",
        "soft_event_rate_rank_spearman",
        "hard_profile_ari_secondary",
    ]
    labels = [
        "Coefficient-surface cosine",
        "Matched loading correlation",
        "Fitted-risk correlation",
        "Soft risk-order Spearman",
        "Hard-profile ARI",
    ]
    for stem, field, heading in [
        (
            "figS01_seed_stability",
            "optimization_seed_stability",
            "Six optimization seeds; 15 matched pairs",
        ),
        (
            "figS03_full_pipeline_stability",
            "full_pipeline_patient_cluster_bootstrap",
            "Twenty patient-cluster full-pipeline bootstraps",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        fig.subplots_adjust(left=0.35, right=0.97, bottom=0.19, top=0.83)
        summary_metrics(ax, s[field]["global"], keys, labels, heading)
        save(fig, out, stem, [source], {"a": field + ".global"})
    k = read(frozen, "shared_k_sensitivity.json")
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.22, top=0.83)
    kk = k["candidate_k"]
    v = [k["oof_metrics"][f"shared_k{i}"]["auroc"] for i in kk]
    ax.plot(kk, v, "o-", color=BLUE)
    ax.scatter([5], [v[kk.index(5)]], color="#E69F00", zorder=4, label="K=5 display")
    ax.axhline(k["oof_metrics"]["additive"]["auroc"], color=MUTED, ls="--", label="Additive")
    ax.set_xticks(kk)
    ax.set_xlabel("Number of shared profiles")
    ax.set_ylabel("Patient-grouped out-of-fold AUROC")
    ax.legend(frameon=False, fontsize=7)
    title(ax, "a", "Nearby K values give similar predictive performance")
    save(
        fig,
        out,
        "figS02_k_sensitivity",
        [frozen / "shared_k_sensitivity.json"],
        {"a": "all forced K=3,...8; K5 descriptive granularity"},
    )
    support = read(frozen, "sensitivity_and_support.json")
    components = support["treatment_design"]["components"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.38, top=0.83)
    _, drugfeatures, druglabels, _, _ = profile_matrix(frozen)
    component_labels = dict(zip(drugfeatures, druglabels, strict=False))
    labels2 = [
        component_labels.get(r["component"], unquote(r["component"].removeprefix("drug_")))
        for r in components
    ]
    ax.bar(
        range(len(components)),
        [100 * r["marginal_prevalence"] for r in components],
        color=TEAL,
    )
    ax.set_xticks(range(len(components)), labels2, rotation=65, ha="right", fontsize=6)
    ax.set_ylabel("Recorded receipt (%)")
    title(ax, "a", "Marginal regimen-component support")
    save(
        fig,
        out,
        "figS04_receipt_support",
        [frozen / "sensitivity_and_support.json"],
        {"a": "treatment_design.components marginal_prevalence"},
    )
    # Context dependence of profile coefficients is an aggregate model summary.
    _, _, druglabels, matrix, _ = profile_matrix(frozen)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.35, top=0.83)
    for i in range(5):
        ax.plot(range(26), matrix[i], ".-", color=COLORS[i], label=SHORT[i], lw=1, ms=3)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(26), druglabels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Regularized logit coefficient")
    ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper right")
    title(ax, "a", "Recorded regimen associations vary across shared profiles")
    save(
        fig,
        out,
        "figS06_component_context_dependence",
        [source],
        {"a": "same supported coefficient matrix as Fig4b; suppressed cells omitted"},
    )
    val = read(data, "shared_oof_clinical_validation.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.81, wspace=0.39)
    for key, col, label in [
        ("shared_k5", BLUE, "Shared PATCH"),
        ("additive", PURPLE, "Additive"),
    ]:
        bins = val["calibration_bins"][key]
        axes[0].plot(
            [b["mean_predicted"] for b in bins],
            [b["observed_rate"] for b in bins],
            "o-",
            color=col,
            label=label,
            ms=3,
            lw=1,
        )
        curve = val["precision_recall_curves"][key]
        axes[1].plot(
            [b["recall"] for b in curve],
            [b["precision"] for b in curve],
            color=col,
            label=f"{label} ({val['metrics'][key]['auprc']:.3f})",
            lw=1.3,
        )
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, lw=0.7)
    axes[0].set(
        xlim=(0, 0.8),
        ylim=(0, 0.8),
        xlabel="Mean predicted risk",
        ylabel="Observed event fraction",
    )
    axes[0].legend(frameon=False, fontsize=6.8)
    title(axes[0], "a", "Out-of-fold calibration")
    axes[1].axhline(
        val["cohort"]["events"] / val["cohort"]["conditions"],
        color=MUTED,
        ls="--",
        lw=0.7,
    )
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    axes[1].legend(frameon=False, fontsize=6.5)
    title(axes[1], "b", "Precision–recall (AUPRC)")
    save(
        fig,
        out,
        "figS07_shared_calibration_pr",
        [data / "shared_oof_clinical_validation.json"],
        {
            "a": "suppression-gated risk deciles",
            "b": (
                "fixed risk-threshold aggregate precision-recall curve; points "
                "with any TP/FP/FN<30 omitted"
            ),
        },
    )
    arch = read(frozen, "architecture_composition_ablations.json")
    metrics = arch["oof_validation"]["metrics"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.81)
    primary = support["primary_ablation"]
    baseline_unc = support["primary_patient_clustered_uncertainty"]["models"]
    arch_unc = arch["oof_validation"]["patient_clustered_uncertainty"]["models"]
    k_unc = k["patient_clustered_uncertainty"]["models"]
    models = [
        (
            "Regimen\nonly",
            primary["regimen_only"]["metrics"],
            baseline_unc["regimen_only"],
            MUTED,
        ),
        (
            "Context\nonly",
            primary["context_only"]["metrics"],
            baseline_unc["context_only"],
            "#979BA2",
        ),
        (
            "Additive",
            primary["additive"]["metrics"],
            baseline_unc["additive"],
            "#535963",
        ),
        (
            "Shared PATCH\nprimary",
            k["oof_metrics"]["shared_k5"],
            k_unc["shared_k5"],
            BLUE,
        ),
        ("Risk-only\nK=5", metrics["risk_only_k5"], arch_unc["risk_only_k5"], TEAL),
        (
            "Residual PATCH\nsecondary",
            metrics["residual_k5"],
            arch_unc["residual_k5"],
            PURPLE,
        ),
    ]
    for i, (_label, point, unc, color) in enumerate(models):
        value = point["auroc"]
        lo, hi = unc["auroc"]["ci95"]
        ax.bar(i, value, color=color, width=0.68)
        ax.errorbar(
            i,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt="none",
            ecolor=INK,
            capsize=3,
            lw=0.8,
        )
        ax.text(i, hi + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=6.7)
    ax.set_ylim(0.60, 0.855)
    ax.set_xticks(range(6), [m[0] for m in models], fontsize=7)
    ax.set_ylabel("Out-of-fold AUROC")
    title(ax, "a", "Architecture comparison under common grouped folds")
    fig.text(
        0.12,
        0.12,
        (
            "Whiskers: 500 patient-cluster bootstrap 95% intervals. Shared "
            "primary uses the frozen K-sensitivity run."
        ),
        fontsize=6.2,
        color=MUTED,
    )
    fig.text(
        0.12,
        0.07,
        (
            "Risk-only and residual bars are separately seeded architecture "
            "sensitivities; they do not define the primary clinical profiles."
        ),
        fontsize=6.2,
        color=MUTED,
    )
    save(
        fig,
        out,
        "figS08_architecture_ablations",
        [
            frozen / "architecture_composition_ablations.json",
            frozen / "sensitivity_and_support.json",
            frozen / "shared_k_sensitivity.json",
        ],
        {
            "a": (
                "Regimen/context/additive: primary_ablation in "
                "sensitivity_and_support; shared-primary: shared_k5 in "
                "shared_k_sensitivity; risk-only K5 and residual K5: separately "
                "seeded architecture_composition_ablations. All use common "
                "patient-grouped five folds, training-only preprocessing and "
                "fixed additive C=0.1. Shared/risk-only use500epochs; residual "
                "uses300epochs. Error bars use each frozen "
                "run's500-patient-cluster-bootstrap interval."
            )
        },
    )
    trans = read(data, "shared_internal_transport.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.82, wspace=0.55)
    for i, (key, col, _label) in enumerate(
        [("additive", PURPLE, "Additive"), ("shared_k5", BLUE, "Shared PATCH")]
    ):
        axes[0].bar(i, trans["temporal"]["metrics"][key]["auroc"], color=col)
    axes[0].set_ylim(0.5, 0.9)
    axes[0].set_xticks([0, 1], ["Additive", "Shared PATCH"], fontsize=7)
    axes[0].set_ylabel("AUROC")
    title(axes[0], "a", "Temporal holdout: 2021 onward")
    rows = trans["held_out_country"]["rows"]
    yy = np.arange(len(rows))
    for key, col in [("additive", PURPLE), ("shared_k5", BLUE)]:
        axes[1].plot(
            [r["metrics"][key]["auroc"] for r in rows],
            yy,
            "o",
            color=col,
            label="Shared PATCH" if key == "shared_k5" else "Additive",
            ms=4,
        )
    axes[1].set_yticks(yy, [r["held_out_country"] for r in rows], fontsize=6.8)
    axes[1].set_xlabel("AUROC")
    axes[1].legend(frameon=False, fontsize=6.4)
    title(axes[1], "b", "Held-out-country transport")
    save(
        fig,
        out,
        "figS09_shared_internal_transport",
        [data / "shared_internal_transport.json"],
        {
            "a-b": (
                "new shared-primary 500epoch models; patient-disjoint and "
                "training-only preprocessing"
            )
        },
    )
    dca(data, out, "figS10_shared_internal_dca")
    cond = support["conditional_component_support"]
    rows = cond["components"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.34, top=0.81, wspace=0.38)
    labs = [
        component_labels.get(r["component"], unquote(r["component"].removeprefix("drug_")))
        for r in rows
    ]
    axes[0].bar(range(len(rows)), [r["selection_auroc"] for r in rows], color=TEAL)
    axes[1].bar(
        range(len(rows)),
        [r["fraction_propensity_0.10_to_0.90"] for r in rows],
        color=BLUE,
    )
    for ax in axes:
        ax.set_xticks(range(len(rows)), labs, rotation=65, ha="right", fontsize=4.9)
    axes[0].set_ylabel("Receipt-model AUROC", fontsize=7)
    axes[1].set_ylabel("Fraction with propensity 0.10–0.90", fontsize=7)
    title(axes[0], "a", "Component selection is predictable")
    title(axes[1], "b", "Conditional overlap is limited")
    save(
        fig,
        out,
        "figS11_conditional_support_audit",
        [frozen / "sensitivity_and_support.json"],
        {"a-b": "conditional_component_support; 0/26 passed joint descriptive gate"},
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.23, top=0.81, wspace=0.45)
    for ax, field, heading, letter in [
        (axes[0], "optimization_seed_stability", "Six optimization seeds", "a"),
        (
            axes[1],
            "full_pipeline_patient_cluster_bootstrap",
            "Twenty patient-cluster\nbootstraps",
            "b",
        ),
    ]:
        item = s[field]["global"]["soft_event_rate_rank_spearman"]
        ax.plot(item["ci95"], [0, 0], color=TEAL, lw=2)
        ax.plot(item["median"], 0, "o", color=TEAL)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Soft profile risk-order Spearman")
        title(ax, letter, heading)
    save(
        fig,
        out,
        "figS12_risk_order_stability",
        [source],
        {"a-b": "soft_event_rate_rank_spearman; outcomes supervise profile geometry"},
    )
    simpath = frozen / "expanded_simulation_benchmark.json"
    if not simpath.exists():
        simpath = frozen / "expanded_simulation_benchmark.json"
    sim = json.loads(simpath.read_text())
    rows = [
        r
        for r in sim["group_summary"]
        if r["n_train"] == 1000 and r["scenario"] in ["additive_null", "low_rank"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.30, top=0.8, wspace=0.4)
    models = [
        "additive_logistic_null",
        "residual_patch",
        "unrestricted_interaction_logistic",
    ]
    for ax, scenario, letter in zip(axes, ["additive_null", "low_rank"], "ab", strict=False):
        selected = {r["model"]: r for r in rows if r["scenario"] == scenario}
        ax.bar(
            range(3),
            [selected[m]["coefficient_rmse_mean"] for m in models],
            yerr=[selected[m]["coefficient_rmse_se"] for m in models],
            color=[MUTED, PURPLE, TEAL],
            capsize=3,
        )
        ax.set_xticks(
            range(3),
            ["Additive", "Residual PATCH", "Unrestricted\ninteraction"],
            rotation=20,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Coefficient-surface RMSE", fontsize=7)
        title(ax, letter, scenario.replace("_", " ").title())
    fig.text(
        0.5,
        0.02,
        (
            "Twenty-seed synthetic benchmark; residual architecture is a "
            "secondary methodological sensitivity."
        ),
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    save(
        fig,
        out,
        "figS13_residual_synthetic_sensitivity",
        [simpath],
        {
            "a-b": (
                "20seed synthetic ntrain1000 coefficient recovery; does not "
                "validate shared-primary estimator"
            )
        },
    )
    brazil = read(frozen, "brazil_source_rebuild_audit.json")
    models = brazil["temporal_recalibration"]["models"]
    names = ["resistance_only", "context_logistic", "patch_profile_score"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.80, wspace=0.4)
    for ax, metric, letter in zip(axes, ["brier", "calibration_slope"], "ab", strict=False):
        ax.bar(range(3), [models[m][metric] for m in names], color=[MUTED, PURPLE, TEAL])
        ax.set_xticks(
            range(3),
            ["Resistance", "Context logistic", "PATCH score"],
            rotation=20,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Brier score" if metric == "brier" else "Calibration slope")
        title(ax, letter, "Brazil temporal recalibration")
    fig.text(
        0.5,
        0.02,
        (
            "Calibrated using 2015–2017 Brazil labels; evaluated in "
            "2018–2019. Prognostic validation only."
        ),
        ha="center",
        fontsize=6.7,
        color=MUTED,
    )
    save(
        fig,
        out,
        "figS14_brazil_temporal_calibration",
        [frozen / "brazil_source_rebuild_audit.json"],
        {
            "a-b": (
                "temporal_recalibration.models; source-only zero-shot transport "
                "and target-label recalibration distinguished"
            )
        },
    )


def main():
    """Parse paths and render the aggregate figures."""
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--main-dir", type=Path, default=HERE / "main")
    parser.add_argument("--supp-dir", type=Path, default=HERE / "supplementary")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Portable directory containing all frozen and newly built safe aggregate JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Portable figure root; contains main/, supplementary/ and manifest",
    )
    args = parser.parse_args()
    if args.input_dir:
        args.frozen_dir = args.input_dir
        args.data_dir = args.input_dir
    if args.output_dir:
        OUTPUT_ROOT = args.output_dir.resolve()
        args.main_dir = OUTPUT_ROOT / "main"
        args.supp_dir = OUTPUT_ROOT / "supplementary"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    schematic(args.frozen_dir, args.main_dir)
    from figures.release import ew_aggregate_figure_templates as ew

    ew.REPORTS = args.frozen_dir
    ew.OUT = args.main_dir
    ew.figure2()
    ew.figure5()
    for number, old, new in [
        (2, "fig02_EW_labeled", "fig02_shared_profiles"),
        (5, "fig05_EW_corrected", "fig05_shared_brazil"),
    ]:
        for suffix in ["pdf", "png"]:
            (args.main_dir / f"{old}.{suffix}").rename(args.main_dir / f"{new}.{suffix}")
        source = (
            "shared_profile_reference_composition.json"
            if number == 2
            else "brazil_source_rebuild_audit.json"
        )
        MANIFEST.append(
            {
                "figure": new,
                "files": [
                    str((args.main_dir / f"{new}.{s}").relative_to(OUTPUT_ROOT))
                    for s in ["pdf", "png"]
                ],
                "sources": [str(args.frozen_dir / source)],
                "panels": {
                    "all": (
                        "EW aggregate template; corrected cohort; named primary profiles "
                        "only in Figure2, anonymous source-risk ranks in harmonized "
                        "Brazil Figure5"
                    )
                },
                "vector_pdf": True,
                "png_dpi": 600,
            }
        )
    geometry(args.data_dir, args.main_dir)
    atlas(args.data_dir, args.frozen_dir, args.main_dir, args.supp_dir)
    dca(args.data_dir, args.main_dir)
    supplements(args.frozen_dir, args.data_dir, args.supp_dir)
    # Restore Figure 4 only after every other drawing is complete. The isolated
    # presentation process cannot alter the other panels' matplotlib settings.
    restoration_scripts = ROOT / "PATCH_EW_Figures_Restored_2026-10-26" / "figures" / "scripts"
    subprocess.run(
        [
            sys.executable,
            str(restoration_scripts / "render_restored_fig4.py"),
            "--data-dir",
            str(args.data_dir),
            "--frozen-dir",
            str(args.frozen_dir),
            "--output-dir",
            str(args.main_dir),
            "--qa-dir",
            str(OUTPUT_ROOT / "restoration_layout_qa"),
        ],
        check=True,
        capture_output=True,
    )
    restoration = json.loads((args.main_dir / "fig04_restored_source_manifest.json").read_text())
    assert restoration["presentation_only"] and not restoration["scientific_selection_changes"]
    for suffix in ("pdf", "png"):
        shutil.copyfile(
            args.main_dir / f"fig04_EW_restored_shared_atlas.{suffix}",
            args.main_dir / f"fig04_shared_atlas.{suffix}",
        )
    for item in MANIFEST:
        if item["figure"] == "fig04_shared_atlas":
            item["panels"] = {letter: restoration[f"{letter}_definition"] for letter in "abcd"}
            item["presentation_restoration"] = {
                "scope": "Landscape layout and modality-group axis cues only",
                "scientific_selection_changes": False,
                "renderer_sha256": restoration["renderer_sha256"],
                "layout_module_sha256": restoration["layout_module_sha256"],
                "panel_data_identity_sha256": restoration["panel_data_identity_sha256"],
            }
    for item in MANIFEST:
        actual_sources = item["sources"]

        def logical_source(source):
            filename = Path(source).name
            if filename == "residual_simulation_benchmark.json":
                filename = "expanded_simulation_benchmark.json"
            return "manuscript/aggregate_results/" + filename

        item["source_sha256"] = {
            logical_source(s): hashlib.sha256(Path(s).read_bytes()).hexdigest()
            for s in actual_sources
        }
        item["sources"] = [logical_source(s) for s in actual_sources]
        item["output_sha256"] = {
            s: hashlib.sha256((OUTPUT_ROOT / s).read_bytes()).hexdigest() for s in item["files"]
        }
    (OUTPUT_ROOT / "source_to_panel_manifest.json").write_text(
        json.dumps(
            {
                "primary_estimator": "shared K=5 simplex PATCH",
                "canonical_profile_order": NAMES,
                "figures": MANIFEST,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "main_figures": 6,
                "source_to_panel_manifest": True,
                "private_data_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
