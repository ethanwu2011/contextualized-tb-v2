"""Publication figures for the archetype and response-profile analysis.

Each function consumes already-computed tables (never raw patient rows) and writes a 300-dpi
PNG+PDF. Together they form the contextualized-model figure suite:

* :func:`figure_archetype_atlas` — the K regimen-coefficient archetypes.
* :func:`figure_subtype_prevalence_by_country` — regional archetype/subtype map.
* :func:`figure_context_archetype_loadings` — which contexts characterise each subtype.
* :func:`figure_heterogeneous_effects` / :func:`figure_homogeneous_effects` — beta(C)
  heterogeneity and mu(C)/beta significance heatmaps.
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
from ctb.viz.style import save_figure, set_style

logger = logging.getLogger(__name__)


def figure_archetype_atlas(
    profiles: pd.DataFrame, outcome: str, out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Heatmap of the K archetype regimen-coefficient profiles.

    Parameters
    ----------
    profiles : pandas.DataFrame
        ``(K, n_treatments)`` log-odds coefficients from
        :func:`ctb.models.archetypes.archetype_treatment_profiles`.
    outcome : str
        Outcome name (used in title and filename).
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    vmax = float(np.nanmax(np.abs(profiles.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.5 * profiles.shape[1] + 2.5, 0.45 * profiles.shape[0] + 1.8))
    sns.heatmap(
        profiles,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "regimen coefficient (log-odds)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(f"Regimen-coefficient archetypes — {outcome}")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Archetype")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / f"figure_archetype_atlas_{outcome}")


def figure_subtype_prevalence_by_country(
    prevalence: pd.DataFrame, outcome: str, out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Heatmap of subtype prevalence across countries (regional archetype map).

    Parameters
    ----------
    prevalence : pandas.DataFrame
        Output of :func:`ctb.models.archetypes.subtype_prevalence_by_group` (rows countries,
        subtype columns plus an ``n`` column).
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    mat = prevalence.drop(columns=["n"]) * 100
    labels = [f"{idx} (n={int(prevalence.loc[idx, 'n'])})" for idx in mat.index]
    fig, ax = plt.subplots(figsize=(0.5 * mat.shape[1] + 3, 0.4 * mat.shape[0] + 2))
    sns.heatmap(
        mat,
        cmap="rocket_r",
        ax=ax,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 6},
        cbar_kws={"label": "% of country's patients"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title(f"Subtype prevalence by country — {outcome}")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Country")
    return save_figure(fig, out_dir / f"figure_subtype_by_country_{outcome}")


def figure_context_archetype_loadings(
    context_profile: pd.DataFrame, outcome: str, out_dir: Path = FIGURES_DIR, top_n: int = 25
) -> list[Path]:
    """Heatmap of standardised context enrichment per subtype (clinical interpretation).

    Parameters
    ----------
    context_profile : pandas.DataFrame
        Output of :func:`ctb.models.archetypes.subtype_context_profile` (rows features, columns
        subtypes); values are standardised enrichment scores.
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.
    top_n : int, default 25
        Show the ``top_n`` features with the largest cross-subtype spread.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    spread = context_profile.max(axis=1) - context_profile.min(axis=1)
    sel = spread.sort_values(ascending=False).head(top_n).index
    mat = context_profile.loc[sel]
    vmax = float(np.nanmax(np.abs(mat.to_numpy())))
    fig, ax = plt.subplots(figsize=(0.5 * mat.shape[1] + 3, 0.32 * mat.shape[0] + 1.8))
    sns.heatmap(
        mat,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "enrichment (SD units)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(f"Context enrichment by subtype — {outcome}")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Context feature")
    return save_figure(fig, out_dir / f"figure_context_loadings_{outcome}")


def figure_heterogeneous_effects(
    heterogeneous: pd.DataFrame, outcome: str, out_dir: Path = FIGURES_DIR, top_n: int = 25
) -> list[Path]:
    """Heatmap of context-conditioned coefficient significance beta(C).

    Parameters
    ----------
    heterogeneous : pandas.DataFrame
        ``-log10(p)`` matrix (rows context, columns treatment) from
        :func:`ctb.models.archetypes.compute_effect_tables`.
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.
    top_n : int, default 25
        Show the ``top_n`` context rows with the strongest maximum signal.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    order = heterogeneous.max(axis=1).sort_values(ascending=False).head(top_n).index
    mat = heterogeneous.loc[order]
    fig, ax = plt.subplots(figsize=(0.5 * mat.shape[1] + 3, 0.32 * mat.shape[0] + 1.8))
    sns.heatmap(
        mat,
        cmap="rocket_r",
        ax=ax,
        cbar_kws={"label": "-log10(p)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(f"Context-conditioned regimen coefficients \u03b2(C) — {outcome}")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Context feature")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return save_figure(fig, out_dir / f"figure_heterogeneous_effects_{outcome}")


def figure_subtype_umap(
    embedding: NDArray[np.float64],
    color_by: dict[str, pd.Series[Any]],
    outcome: str,
    out_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Small-multiple 2-D embedding of the loadings ``Z``, coloured by several covariates.

    Parameters
    ----------
    embedding : numpy.ndarray
        2-D embedding ``(n, 2)`` from :func:`ctb.models.archetypes.embed_2d`.
    color_by : dict of str to pandas.Series
        Mapping of panel title -> per-patient colouring variable (categorical or numeric).
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    n_panels = len(color_by)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.2), squeeze=False)
    for ax, (title, series) in zip(axes[0], color_by.items(), strict=True):
        values = pd.Series(np.asarray(series)).reset_index(drop=True)
        if values.dtype == object or values.dtype.name == "string" or values.nunique() <= 12:
            cats = values.astype("category")
            palette = sns.color_palette("colorblind", len(cats.cat.categories))
            for i, cat in enumerate(cats.cat.categories):
                m = (cats == cat).to_numpy()
                ax.scatter(
                    embedding[m, 0],
                    embedding[m, 1],
                    s=3,
                    color=palette[i],
                    label=str(cat),
                    alpha=0.6,
                    linewidths=0,
                )
            if len(cats.cat.categories) <= 8:
                ax.legend(markerscale=2, fontsize=5, loc="best", frameon=False)
        else:
            sc = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=3,
                c=values.to_numpy(),
                cmap="viridis",
                alpha=0.6,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set(title=title, xticks=[], yticks=[], xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle(f"Patient-subtype embedding (Z) — {outcome}")
    return save_figure(fig, out_dir / f"figure_subtype_umap_{outcome}")


def figure_subtype_prognostic(
    prevalence: pd.DataFrame,
    aurocs: dict[str, float],
    outcome: str,
    out_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Subtype outcome-rate bars and a subtype-vs-baseline AUROC comparison.

    Parameters
    ----------
    prevalence : pandas.DataFrame
        Columns ``subtype``, ``outcome_rate``, ``n`` from
        :func:`ctb.models.archetypes.subtype_outcome_separation`.
    aurocs : dict of str to float
        AUROCs keyed ``subtype``, ``resistance``, ``combined``.
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))  # type: ignore[misc]
    prev = prevalence.sort_values("outcome_rate")
    ax1.bar(prev["subtype"].astype(str), prev["outcome_rate"] * 100, color="#0173B2")
    for _, r in prev.iterrows():
        ax1.text(
            str(int(r["subtype"])),
            r["outcome_rate"] * 100,
            f"n={int(r['n'])}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax1.set(xlabel="Subtype", ylabel=f"{outcome} rate (%)", title="Outcome rate by subtype")

    keys = ["resistance", "subtype", "combined"]
    ax2.bar(keys, [aurocs[k] for k in keys], color=["#949494", "#029E73", "#D55E00"])
    for k in keys:
        ax2.text(k, aurocs[k], f"{aurocs[k]:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set(ylim=(0.5, 1.0), ylabel="AUROC (5-fold)", title="Outcome prediction")
    fig.suptitle(f"Subtype prognostic validation — {outcome}")
    return save_figure(fig, out_dir / f"figure_subtype_prognostic_{outcome}")


def figure_homogeneous_effects(
    homo_predictor: pd.DataFrame,
    homo_context: pd.DataFrame,
    outcome: str,
    out_dir: Path = FIGURES_DIR,
    top_n: int = 20,
) -> list[Path]:
    """Bar charts of population coefficients beta and context intercepts mu(C).

    Parameters
    ----------
    homo_predictor : pandas.DataFrame
        Columns ``treatment``, ``effect``, ``neg_log10_p``.
    homo_context : pandas.DataFrame
        Columns ``context``, ``neg_log10_p``.
    outcome : str
        Outcome name.
    out_dir : pathlib.Path, default :data:`ctb.config.FIGURES_DIR`
        Output directory.
    top_n : int, default 20
        Number of context features to show.

    Returns
    -------
    list of pathlib.Path
        Written file paths.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))  # type: ignore[misc]

    hp = homo_predictor.sort_values("effect")
    colors = ["#0173B2" if v < 0 else "#D55E00" for v in hp["effect"]]
    ax1.barh(hp["treatment"], hp["effect"], color=colors)
    ax1.axvline(0, color="#444444", linewidth=0.8)
    ax1.set(
        xlabel="Population regimen coefficient (log-odds)",
        title="Population-average predictor coefficients",
    )

    hc = homo_context.sort_values("neg_log10_p", ascending=False).head(top_n).iloc[::-1]
    ax2.barh(hc["context"], hc["neg_log10_p"], color="#029E73")
    ax2.set(xlabel="-log10(p)", title=f"Context intercepts \u03bc(C) (top {top_n})")

    fig.suptitle(f"Population-average coefficients — {outcome}")
    return save_figure(fig, out_dir / f"figure_homogeneous_effects_{outcome}")
