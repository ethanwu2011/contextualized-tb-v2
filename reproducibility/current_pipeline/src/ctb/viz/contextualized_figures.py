"""Figures for the contextualized regimen-coefficient decomposition.

All functions consume already-computed arrays/tables (never raw patient rows) and write 300-dpi
PNG+PDF. They visualize the output of
:class:`ctb.models.archetype_contextualized.LinearArchetypeContextualized`:

* :func:`figure_homogeneous_effects` — population-average regimen coefficient per drug.
* :func:`figure_heterogeneity_by_drug` — coefficient context dependence measured by SD of β(C).
* :func:`figure_archetype_atlas` — the K archetype drug-response profiles.
* :func:`figure_subtype_umap` — UMAP of patient loadings ``Z`` coloured by covariates.
* :func:`figure_subtype_characterization` — per-subtype conversion rate + clinical makeup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from ctb.config import FIGURES_DIR
from ctb.viz.style import PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)


def figure_homogeneous_effects(
    effects: pd.Series[float], out_dir: Path = FIGURES_DIR, name: str = "homogeneous_effects"
) -> list[Path]:
    """Forest-style bar of population-average regimen coefficients.

    Parameters
    ----------
    effects : pandas.Series
        Index = drug names and values = mean ``β`` on the model's logit scale.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    e = effects.sort_values()
    fig, ax = plt.subplots(figsize=(4.0, 0.32 * len(e) + 1))
    colors = ["#0173B2" if v >= 0 else "#D55E00" for v in e]
    ax.barh(list(e.index), np.asarray(e.values, dtype=float), color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set(
        xlabel="Population regimen coefficient (log-odds)",
        title="Population-average regimen coefficients",
    )
    return save_figure(fig, out_dir / name)


def figure_heterogeneity_by_drug(
    het: pd.Series[float], out_dir: Path = FIGURES_DIR, name: str = "heterogeneity_by_drug"
) -> list[Path]:
    """Bar of per-drug coefficient heterogeneity (SD of β(C) across episodes).

    Parameters
    ----------
    het : pandas.Series
        Index = drug names, values = SD of β(C) (higher => more context-dependent).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    h = het.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(4.0, 0.32 * len(h) + 1))
    ax.barh(list(h.index), np.asarray(h.values, dtype=float), color=PALETTE[2])
    ax.set(
        xlabel="Coefficient heterogeneity  (SD of β(C))",
        title="Which regimen coefficients vary most with context",
    )
    return save_figure(fig, out_dir / name)


def figure_archetype_atlas(
    profiles: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "archetype_atlas"
) -> list[Path]:
    """Heatmap of the K archetype regimen-coefficient profiles.

    Parameters
    ----------
    profiles : pandas.DataFrame
        ``(K, n_drugs)`` archetype drug-response profiles (normalized).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    vmax = float(np.nanmax(np.abs(profiles.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.45 * profiles.shape[1] + 2, 0.5 * profiles.shape[0] + 1.5))
    sns.heatmap(
        profiles,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "archetype loading"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set(title="Treatment-response archetypes", xlabel="Drug", ylabel="Archetype")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / name)


def figure_model_ladder(
    table: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "model_ladder"
) -> list[Path]:
    """Plot the methods-headline ladder: AUC from population -> +context -> +archetypes.

    Parameters
    ----------
    table : pandas.DataFrame
        Columns ``model`` (ordered), ``auc``, ``ci_low``, ``ci_high``, and optional ``p_vs_prev``
        (bootstrap p-value vs the previous rung; annotated as significance stars).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.0, 0.5 * len(t) + 1.5))
    ypos = np.arange(len(t))
    err = np.vstack([t["auc"] - t["ci_low"], t["ci_high"] - t["auc"]])
    ax.errorbar(t["auc"], ypos, xerr=err, fmt="o", color=PALETTE[0], capsize=3, markersize=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(t["model"])
    ax.invert_yaxis()
    ax.set(
        xlabel="Cross-validated AUROC (95% CI)", title="Contextualization improves discrimination"
    )
    if "p_vs_prev" in t.columns:
        for i in range(len(t)):
            p = t["p_vs_prev"].iloc[i]
            if pd.notna(p) and i > 0:
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(
                    float(t["ci_high"].iloc[i]) + 0.002,
                    i,
                    f"{star}",
                    va="center",
                    fontsize=8,
                    color="#333",
                )
    return save_figure(fig, out_dir / name)


def figure_archetype_k_sweep(
    table: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "archetype_k_sweep"
) -> list[Path]:
    """Plot AUROC vs number of archetypes K with 95% CIs (the value-of-archetypes curve).

    Parameters
    ----------
    table : pandas.DataFrame
        Columns ``k``, ``auc``, ``ci_low``, ``ci_high``.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    t = table.sort_values("k")
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    err = np.vstack([t["auc"] - t["ci_low"], t["ci_high"] - t["auc"]])
    ax.errorbar(t["k"], t["auc"], yerr=err, fmt="o-", color=PALETTE[2], capsize=3, markersize=5)
    best = t.loc[t["auc"].idxmax()]
    ax.annotate(
        f"peak K={int(best['k'])}",
        xy=(float(best["k"]), float(best["auc"])),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=7,
    )
    ax.set(
        xlabel="Number of archetypes (K)",
        ylabel="Cross-validated AUROC",
        title="Archetype count vs performance",
    )
    return save_figure(fig, out_dir / name)


def figure_archetype_distribution_sweep(
    dist: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "archetype_distribution_sweep"
) -> list[Path]:
    """Stacked-bar of how patients distribute across archetypes as K grows.

    For each K, patients are hard-assigned to their top-loading archetype; the bar is split into the
    sorted (largest-first) subtype fractions. Shows whether extra archetypes carve out *balanced*
    new subtypes (good, interpretable) or just shave off tiny slivers (over-fractionation).

    Parameters
    ----------
    dist : pandas.DataFrame
        Long-form with columns ``k``, ``slot`` (0-indexed sorted subtype), ``frac``.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    ks = sorted(dist["k"].unique())
    wide = dist.pivot(index="k", columns="slot", values="frac").reindex(ks).fillna(0.0)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(ks), 4.2))
    palette = sns.color_palette("crest", n_colors=wide.shape[1])
    bottom = np.zeros(len(ks))
    for slot in wide.columns:
        vals = wide[slot].to_numpy()
        ax.bar(range(len(ks)), vals, bottom=bottom, color=palette[slot], width=0.82, linewidth=0)
        bottom += vals
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set(
        xlabel="Number of archetypes (K)",
        ylabel="Fraction of patients (sorted subtypes)",
        title="Patient distribution across archetypes",
        ylim=(0, 1),
    )
    ax.margins(x=0.01)
    return save_figure(fig, out_dir / name)


def figure_heterogeneous_effects_with_ci(
    sig_table: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "heterogeneous_effects_ci"
) -> list[Path]:
    """Coefficient heatmap with bootstrap uncertainty overlaid.

    Parameters
    ----------
    sig_table : pandas.DataFrame
        Long format from :func:`ctb.models.archetype_contextualized.bootstrap_context_effects`
        (columns ``stratum``, ``drug``, ``mean``, ``significant``).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    pivot = sig_table.pivot(index="stratum", columns="drug", values="mean")
    sig = sig_table.pivot(index="stratum", columns="drug", values="significant")
    annot = sig.replace({True: "*", False: ""})
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.5 * pivot.shape[1] + 2.5, 0.5 * pivot.shape[0] + 1.8))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"size": 10, "color": "black"},
        cbar_kws={"label": "regimen coefficient β (log-odds)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set(
        title="Context-conditioned regimen coefficients β(C)  (* = interval excludes 0)",
        xlabel="Drug",
        ylabel="Context stratum",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / name)


def figure_heterogeneity_pvals(
    scan: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "heterogeneity_pvals"
) -> list[Path]:
    """Heatmap of heterogeneity significance (-log10 p) per drug x context, with direction.

    Legacy descriptive figure: cell color = signed -log10(p), and ``*`` marks p < 0.05.
    Direction indicates the fitted regimen-coefficient shift, not a causal drug effect.

    Parameters
    ----------
    scan : pandas.DataFrame
        Output of :func:`ctb.models.archetype_contextualized.heterogeneous_effect_pvals` (columns
        ``context``, ``drug``, ``delta_beta``, ``pval``, ``significant``).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    s = scan.copy()
    s["signed_nlp"] = -np.log10(s["pval"].clip(lower=1e-6)) * np.sign(s["delta_beta"])
    pivot = s.pivot(index="context", columns="drug", values="signed_nlp")
    sig = s.pivot(index="context", columns="drug", values="significant").fillna(False)
    annot = sig.replace({True: "*", False: ""})
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(0.62 * pivot.shape[1] + 2.6, 0.62 * pivot.shape[0] + 2.0))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        square=True,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"size": 15, "color": "#111111", "weight": "bold"},
        cbar_kws={"label": "signed  -log10(p)", "shrink": 0.55, "aspect": 14, "pad": 0.02},
        linewidths=1.4,
        linecolor="white",
    )
    ax.set_title("Context-dependent regimen-coefficient heterogeneity", pad=10, weight="semibold")
    ax.set(xlabel="Anti-tuberculosis drug", ylabel="Patient context")
    ax.text(
        0.0,
        -0.16,
        "* p < 0.05    red = more positive coefficient    blue = more negative coefficient",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#777777",
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(left=False, bottom=False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=8, length=0)
        cbar.outline.set_visible(False)  # type: ignore[operator]
    return save_figure(fig, out_dir / name)


def figure_heterogeneity_clustermap(
    scan: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
    name: str = "heterogeneity_clustermap",
    top_n: int | None = None,
    title: str = "Clustered regimen-coefficient heterogeneity across patient contexts",
) -> list[Path]:
    """Bi-clustered heatmap of regimen-coefficient heterogeneity, drugs by contexts.

    Same signed-significance signal as :func:`figure_heterogeneity_pvals`, but rows (contexts) and
    columns (drugs) are hierarchically clustered (Ward linkage) so that drugs sharing a
    heterogeneity pattern and contexts that modulate the same drugs form visible blocks. ``*`` marks
    p < 0.05. The fitted coefficients are observational associations.

    Parameters
    ----------
    scan : pandas.DataFrame
        Output of :func:`ctb.models.archetype_contextualized.heterogeneous_effect_pvals` or
        :func:`ctb.models.archetype_contextualized.heterogeneity_atlas` (columns ``context``,
        ``drug``, ``delta_beta``, ``pval``, ``significant``).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.
    top_n : int or None
        If set, keep only the ``top_n`` most heterogeneous contexts (ranked by total ``-log10 p``
        summed across drugs). Keeps a large all-context atlas readable; ``None`` plots all contexts.
    title : str
        Figure title.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    # clustermap manages its own layout (tight_layout); constrained_layout conflicts with its cbar.
    plt.rcParams["figure.constrained_layout.use"] = False
    s = scan.copy()
    s["signed_nlp"] = -np.log10(s["pval"].clip(lower=1e-6)) * np.sign(s["delta_beta"])
    if top_n is not None:
        rank = (
            s.assign(nlp=s["signed_nlp"].abs())
            .groupby("context")["nlp"]
            .sum()
            .sort_values(ascending=False)
        )
        keep = set(rank.head(top_n).index)
        s = s[s["context"].isin(keep)]
    pivot = s.pivot(index="context", columns="drug", values="signed_nlp").fillna(0.0)
    sig = (
        s.pivot(index="context", columns="drug", values="significant")
        .reindex(index=pivot.index, columns=pivot.columns)
        .fillna(False)
    )
    annot = sig.replace({True: "*", False: ""})
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0

    grid = sns.clustermap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        method="ward",
        metric="euclidean",
        annot=annot,
        fmt="",
        annot_kws={"size": 13, "color": "#111111", "weight": "bold"},
        linewidths=1.2,
        linecolor="white",
        figsize=(0.66 * pivot.shape[1] + 3.2, 0.62 * pivot.shape[0] + 2.6),
        dendrogram_ratio=(0.12, 0.16),
        cbar_pos=(0.02, 0.83, 0.03, 0.14),
        cbar_kws={"label": "signed  -log10(p)"},
    )
    grid.ax_heatmap.set_xlabel("Anti-tuberculosis drug", labelpad=8)
    grid.ax_heatmap.set_ylabel("Patient context", labelpad=8)
    label_fs = 11 if pivot.shape[0] <= 35 else 9
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=label_fs)
    plt.setp(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fs)
    grid.ax_heatmap.tick_params(left=False, bottom=False)
    grid.ax_cbar.tick_params(labelsize=8, length=0)
    grid.ax_cbar.set_title("signed\n-log10(p)", fontsize=8, pad=6)
    grid.ax_cbar.yaxis.set_label_text("")
    grid.figure.suptitle(title, x=0.55, y=1.01, weight="semibold")
    written = save_figure(grid.figure, out_dir / name)
    plt.rcParams["figure.constrained_layout.use"] = True  # restore for subsequent figures
    return written


def figure_interaction_landscape(
    scan: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "interaction_landscape"
) -> list[Path]:
    """Full drug x context interaction heatmap with significance markers.

    Legacy descriptive figure of fitted regimen-coefficient shifts across context features, with
    ``*`` marking bootstrap-significant interactions. Positive values indicate a more positive
    fitted coefficient in the context-present group, not greater causal efficacy.

    Parameters
    ----------
    scan : pandas.DataFrame
        Output of :func:`ctb.models.archetype_contextualized.interaction_scan` (columns ``drug``,
        ``context``, ``interaction``, ``significant``).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    pivot = scan.pivot(index="context", columns="drug", values="interaction")
    sig = scan.pivot(index="context", columns="drug", values="significant").fillna(False)
    annot = sig.replace({True: "*", False: ""})
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.5 * pivot.shape[1] + 2.5, 0.5 * pivot.shape[0] + 1.8))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"size": 11, "color": "black"},
        cbar_kws={"label": "interaction: shift in regimen coefficient (log-odds)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set(
        title="Drug \u00d7 context interaction landscape  (* = bootstrap CI excludes 0)",
        xlabel="Drug",
        ylabel="Patient context",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / name)


def figure_subtype_drug_response(
    profiles: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "subtype_drug_response"
) -> list[Path]:
    """Heatmap of mean regimen coefficient β per drug and profile.

    Describes which recorded-regimen coefficients characterize each response profile.

    Parameters
    ----------
    profiles : pandas.DataFrame
        ``(n_subtypes, n_drugs)`` mean β(drug) within each subtype (positive => faster conversion).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    vmax = float(np.nanmax(np.abs(profiles.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.62 * profiles.shape[1] + 2.6, 0.55 * profiles.shape[0] + 2.0))
    sns.heatmap(
        profiles,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7, "color": "#222222"},
        cbar_kws={
            "label": "regimen coefficient β (log-odds)",
            "shrink": 0.55,
            "aspect": 14,
            "pad": 0.02,
        },
        linewidths=1.4,
        linecolor="white",
    )
    ax.set_title("Drug response by patient subtype", pad=10, weight="semibold")
    ax.set(xlabel="Anti-tuberculosis drug", ylabel="Treatment-response subtype")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(left=False, bottom=False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=8, length=0)
        cbar.outline.set_visible(False)  # type: ignore[operator]
    return save_figure(fig, out_dir / name)


def figure_drug_heterogeneity_ranking(
    summary: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "drug_heterogeneity_ranking"
) -> list[Path]:
    """Rank drugs by how context-dependent their fitted coefficient is.

    Parameters
    ----------
    summary : pandas.DataFrame
        Columns ``drug``, ``magnitude`` (total significant drug-specific |Δβ|), ``n_sig`` (number of
        significant context modifiers).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    t = summary.sort_values("magnitude", ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(t) + 1.2))
    ax.barh(t["drug"], t["magnitude"], color=PALETTE[2])
    for i, (_, r) in enumerate(t.iterrows()):
        if r["n_sig"] > 0:
            ax.text(
                r["magnitude"], i, f"  {int(r['n_sig'])}", va="center", fontsize=7, color="#555555"
            )
    ax.set(
        xlabel="Total drug-specific heterogeneity (Σ|Δβ| over significant contexts)",
        ylabel="Anti-tuberculosis drug",
        title="Which drugs have context-dependent coefficients",
    )
    ax.margins(y=0.01)
    return save_figure(fig, out_dir / name)


def figure_effect_modifier_forests(
    forest: pd.DataFrame,
    drug_order: list[str],
    out_dir: Path = FIGURES_DIR,
    name: str = "effect_modifier_forests",
) -> list[Path]:
    """Small-multiple forest plots of the strongest context modifiers per key drug.

    Parameters
    ----------
    forest : pandas.DataFrame
        Columns ``drug``, ``context``, ``delta_beta``, ``ci_low``, ``ci_high``.
    drug_order : list of str
        Drugs to panel (in order).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    ndrug = len(drug_order)
    ncol = min(3, ndrug)
    nrow = int(np.ceil(ndrug / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, drug in zip(axes.flat, drug_order, strict=False):
        ax.set_visible(True)
        sub = forest[forest["drug"] == drug].copy()
        sub = sub.reindex(sub["delta_beta"].abs().sort_values().index).tail(8)
        ypos = np.arange(len(sub))
        colors = [PALETTE[3] if v < 0 else PALETTE[1] for v in sub["delta_beta"]]
        ax.errorbar(
            sub["delta_beta"],
            ypos,
            xerr=[sub["delta_beta"] - sub["ci_low"], sub["ci_high"] - sub["delta_beta"]],
            fmt="none",
            ecolor="#999999",
            elinewidth=1.2,
            capsize=2,
            zorder=1,
        )
        ax.scatter(sub["delta_beta"], ypos, c=colors, s=28, zorder=2)
        ax.axvline(0, color="#cccccc", lw=1, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["context"], fontsize=7)
        ax.set_title(drug, fontsize=10, weight="semibold")
        ax.set_xlabel("Δβ (log-odds)", fontsize=8)
    fig.suptitle("Top context effect-modifiers per drug", weight="semibold")
    return save_figure(fig, out_dir / name)


def figure_subtype_resistance_validation(
    table: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "subtype_resistance_validation"
) -> list[Path]:
    """Render a descriptive resistance/co-treatment heatmap for context-derived hard bins.

    Resistance is itself included in the model context and is correlated with recorded treatment.
    Accordingly, this panel is not independent validation of a treatment-response phenotype.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by subtype; percentage columns (e.g. ``got Rifampicin``, ``Drug-sensitive``,
        ``MDR``, ``XDR``) plus an ``unfavorable %`` column used only for row ordering.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    t = table.sort_values("unfavorable %")
    cols = [c for c in t.columns if c != "unfavorable %"]
    mat = t[cols]
    fig, ax = plt.subplots(figsize=(0.95 * len(cols) + 2.4, 0.62 * len(t) + 1.8))
    sns.heatmap(
        mat,
        cmap="rocket_r",
        vmin=0,
        vmax=100,
        ax=ax,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 9},
        cbar_kws={"label": "% of subtype", "shrink": 0.6, "pad": 0.02},
        linewidths=1.4,
        linecolor="white",
    )
    ax.set_title("Unsupervised subtypes recover the resistance axis", pad=10, weight="semibold")
    ax.set(xlabel="", ylabel="Treatment-response subtype")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(left=False, bottom=False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=8, length=0)
        cbar.outline.set_visible(False)  # type: ignore[operator]
    return save_figure(fig, out_dir / name)


def figure_subtype_interaction(
    effect: pd.DataFrame,
    sig: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
    name: str = "subtype_interaction",
    title: str = "Profile by recorded-drug coefficient interaction",
    cbar_label: str = "β subtype − β population (log-odds)",
) -> list[Path]:
    """Heatmap of a profile-by-drug coefficient matrix with bootstrap markers.

    Used for both the per-profile coefficient ``β`` and the **interaction** (each profile's β
    minus the population-average β). ``*`` marks cells whose bootstrap 95% CI excludes zero.

    Parameters
    ----------
    effect : pandas.DataFrame
        ``(n_subtypes, n_drugs)`` effect (or interaction) estimates.
    sig : pandas.DataFrame
        Boolean ``(n_subtypes, n_drugs)`` significance mask aligned to ``effect``.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.
    title : str
        Figure title.
    cbar_label : str
        Colorbar label.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    annot = (
        sig.reindex(index=effect.index, columns=effect.columns)
        .fillna(False)
        .replace({True: "*", False: ""})
    )
    vmax = float(np.nanmax(np.abs(effect.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(0.62 * effect.shape[1] + 2.6, 0.6 * effect.shape[0] + 2.2))
    sns.heatmap(
        effect,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        square=True,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"size": 15, "color": "#111111", "weight": "bold"},
        cbar_kws={"label": cbar_label, "shrink": 0.55, "aspect": 14, "pad": 0.02},
        linewidths=1.4,
        linecolor="white",
    )
    ax.set_title(title, pad=10, weight="semibold")
    ax.set(xlabel="Anti-tuberculosis drug", ylabel="Treatment-response subtype")
    ax.text(
        0.0,
        -0.18,
        "* bootstrap 95% CI excludes 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#777777",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(left=False, bottom=False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=8, length=0)
        cbar.outline.set_visible(False)  # type: ignore[operator]
    return save_figure(fig, out_dir / name)


def figure_heterogeneous_effects_by_context(
    het: pd.DataFrame, out_dir: Path = FIGURES_DIR, name: str = "heterogeneous_effects_context"
) -> list[Path]:
    """Heatmap of mean regimen coefficient β per drug and clinical context stratum.

    The display shows how each recorded drug's association with culture conversion shifts across
    patient context such as HIV, smear, and resistance class.

    Parameters
    ----------
    het : pandas.DataFrame
        ``(n_strata, n_drugs)`` mean β(drug) within each context stratum.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    vmax = float(np.nanmax(np.abs(het.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.5 * het.shape[1] + 2.5, 0.45 * het.shape[0] + 1.8))
    sns.heatmap(
        het,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "regimen coefficient β (log-odds)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set(
        title="Context-conditioned regimen coefficients β(C)",
        xlabel="Drug",
        ylabel="Context stratum",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / name)


def figure_subtype_umap(
    embedding: NDArray[np.float64],
    color_by: dict[str, pd.Series[Any]],
    out_dir: Path = FIGURES_DIR,
    name: str = "subtype_umap",
) -> list[Path]:
    """Small-multiple UMAP of patient loadings ``Z`` coloured by several covariates.

    Parameters
    ----------
    embedding : numpy.ndarray
        ``(n, 2)`` 2-D embedding of the loadings.
    color_by : dict of str to pandas.Series
        Panel title -> per-patient colouring variable.
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    npan = len(color_by)
    fig, axes = plt.subplots(1, npan, figsize=(3.3 * npan, 3.1), squeeze=False)
    for ax, (title, series) in zip(axes[0], color_by.items(), strict=True):
        vals = pd.Series(np.asarray(series)).reset_index(drop=True)
        if vals.dtype == object or vals.dtype.name == "string" or vals.nunique() <= 12:
            cats = vals.astype("category")
            pal = sns.color_palette("colorblind", len(cats.cat.categories))
            for i, cat in enumerate(cats.cat.categories):
                msk = (cats == cat).to_numpy()
                ax.scatter(
                    embedding[msk, 0],
                    embedding[msk, 1],
                    s=2,
                    color=pal[i],
                    label=str(cat),
                    alpha=0.5,
                    linewidths=0,
                )
            if len(cats.cat.categories) <= 8:
                ax.legend(markerscale=3, fontsize=5, loc="best", frameon=False)
        else:
            sc = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=2,
                c=vals.to_numpy(),
                cmap="viridis",
                alpha=0.5,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set(title=title, xticks=[], yticks=[], xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle("Patient treatment-response subtypes (loadings Z)")
    return save_figure(fig, out_dir / name)


def figure_subtype_characterization(
    table: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
    name: str = "subtype_characterization",
    outcome_label: str = "Conversion",
) -> list[Path]:
    """Conversion rate per subtype (bars) + clinical-feature heatmap.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by subtype; must contain ``conversion_pct``, ``n``, and standardized clinical
        feature columns (prefixed ``feat_``).
    out_dir : pathlib.Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list of pathlib.Path
        Written paths.
    """
    set_style()
    feat_cols = [c for c in table.columns if c.startswith("feat_")]
    fig, (ax1, ax2) = plt.subplots(  # type: ignore[misc]
        1,
        2,
        figsize=(4 + 0.5 * len(feat_cols), 0.5 * len(table) + 2),
        gridspec_kw={"width_ratios": [1, max(1, len(feat_cols) * 0.4)]},
    )
    order = table.sort_values("conversion_pct").index
    t = table.loc[order]
    ax1.barh([str(i) for i in t.index], t["conversion_pct"], color=PALETTE[0])
    for i, (_, r) in enumerate(t.iterrows()):
        ax1.text(r["conversion_pct"], i, f" n={int(r['n'])}", va="center", fontsize=6)
    ax1.set(xlabel=f"{outcome_label} rate (%)", ylabel="Subtype", title="Outcome by subtype")

    mat = t[feat_cols].copy()
    mat.columns = pd.Index([c.replace("feat_", "") for c in feat_cols])
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 1.0
    sns.heatmap(
        mat,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax2,
        cbar_kws={"label": "enrichment (SD)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax2.set(title="Subtype clinical profile", ylabel="")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / name)
