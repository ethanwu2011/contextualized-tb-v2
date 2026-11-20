"""Archetype extraction, patient-profile assignment, and coefficient stability analysis.

The contextualized classifier with ``num_archetypes=K`` represents every patient's
context-conditioned regimen coefficients as a mix of ``K`` shared archetypes:

``theta_i = [beta(C_i) | mu(C_i)] = sum_k Z_ik * A_k``,  with  ``Z_i = encoder(C_i)``,

where each archetype ``A_k`` is a full coefficient vector (one weight per treatment plus an
intercept). The per-patient loadings ``Z_i`` are a low-dimensional latent representation of the
patient's regimen-coefficient profile and are used here to assign display profiles.

Implementation notes (verified against the ``contextualized`` package internals):

* A fitted bootstrap model is ``clf.models[b]``; its metamodel is a ``SubtypeMetamodel``.
* ``Z = clf.models[b].metamodel.context_encoder(C_dense)`` has shape ``(n, K)``. Under the default
  identity link it is **not** softmax-normalised, so it is treated as a real-valued loading vector,
  not a probability simplex. Subtypes are assigned by the dominant (arg-max) archetype loading.
* Archetypes ``A = clf.models[b].metamodel.explainer.archetypes`` have shape
  ``(y_dim, x_dim+1, K)``;
  archetype ``k``'s regimen-coefficient profile is ``A[0, :x_dim, k]`` and its intercept is
  ``A[0, x_dim, k]``.
* Archetypes are **not aligned across bootstraps** (label switching); :func:`archetype_stability`
  aligns them with the Hungarian algorithm before quantifying reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from contextualized.easy import ContextualizedClassifier

FloatArray = NDArray[np.float64]


def extract_archetype_weights(
    clf: ContextualizedClassifier, c_dense: FloatArray, bootstrap: int = 0
) -> FloatArray:
    """Return per-patient archetype loadings ``Z`` from one bootstrap's context encoder.

    Parameters
    ----------
    clf : ContextualizedClassifier
        Fitted classifier.
    c_dense : numpy.ndarray
        Dense, NaN-free context matrix ``(n, context_dim)`` (post-preprocessing).
    bootstrap : int, default 0
        Index of the bootstrap model to read.

    Returns
    -------
    numpy.ndarray
        Loadings ``Z`` of shape ``(n, K)``.
    """
    encoder = clf.models[bootstrap].metamodel.context_encoder
    with torch.no_grad():
        z = encoder(torch.tensor(c_dense, dtype=torch.float32)).cpu().numpy()
    return np.asarray(z, dtype=np.float64)


def extract_archetypes(clf: ContextualizedClassifier, bootstrap: int = 0) -> FloatArray:
    """Return the ``K`` archetype coefficient vectors from one bootstrap.

    Parameters
    ----------
    clf : ContextualizedClassifier
        Fitted classifier.
    bootstrap : int, default 0
        Index of the bootstrap model to read.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(K, x_dim + 1)``; columns ``0:x_dim`` are regimen coefficients, the last
        column is the intercept.
    """
    arche = clf.models[bootstrap].metamodel.explainer.archetypes.detach().cpu().numpy()
    # shape (y_dim, x_dim+1, K) -> (K, x_dim+1)
    return np.asarray(arche[0].T, dtype=np.float64)


def assign_subtypes(z: FloatArray) -> NDArray[np.int_]:
    """Assign each patient to its dominant archetype (arg-max loading).

    Parameters
    ----------
    z : numpy.ndarray
        Archetype loadings ``(n, K)``.

    Returns
    -------
    numpy.ndarray
        Integer subtype label per patient ``(n,)``.
    """
    return np.asarray(z.argmax(axis=1), dtype=np.int_)


def assign_subtypes_kmeans(z: FloatArray, n_subtypes: int, seed: int = 42) -> NDArray[np.int_]:
    """Assign subtypes by clustering the archetype loadings ``Z`` (collapse-robust alternative).

    When arg-max assignment collapses most patients onto a single archetype (which happens when the
    loadings vary smoothly rather than peaking), K-means on the full loading vectors recovers a
    balanced display summary that still respects the learned coefficient geometry.

    Parameters
    ----------
    z : numpy.ndarray
        Archetype loadings ``(n, K)``.
    n_subtypes : int
        Number of clusters.
    seed : int, default 42
        Random seed for K-means.

    Returns
    -------
    numpy.ndarray
        Integer subtype label per patient ``(n,)``.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_subtypes, random_state=seed, n_init=10)
    return np.asarray(km.fit_predict(z), dtype=np.int_)


def choose_kmeans_subtypes(
    z: FloatArray, k_min: int = 3, k_max: int = 8, seed: int = 42
) -> tuple[NDArray[np.int_], int, float]:
    """Cluster the loadings ``Z`` into balanced subtypes, choosing K by silhouette score.

    This realises the *medical subtypes used in analysis* (in the terminology of Ellington et al.),
    as distinct from the raw per-patient archetype loadings (the *model subtype*). K-means on ``Z``
    yields comparably sized, interpretable groups even when arg-max assignment collapses onto a
    dominant archetype.

    Parameters
    ----------
    z : numpy.ndarray
        Archetype loadings ``(n, K)``.
    k_min, k_max : int
        Inclusive range of cluster counts to search.
    seed : int, default 42
        K-means seed.

    Returns
    -------
    tuple of (numpy.ndarray, int, float)
        Subtype labels ``(n,)``, the selected number of subtypes, and its silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best: tuple[NDArray[np.int_], int, float] | None = None
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(z)
        score = float(silhouette_score(z, labels))
        if best is None or score > best[2]:
            best = (np.asarray(labels, dtype=np.int_), k, score)
    assert best is not None
    logger.info("Selected %d subtypes by silhouette (score=%.3f)", best[1], best[2])
    return best


def subtype_outcome_separation(
    subtypes: NDArray[np.int_],
    y: NDArray[np.int_],
    resistance: pd.Series[Any],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assess whether subtypes stratify the outcome, vs a resistance-class baseline.

    Mirrors the Ellington et al. prognostic-validation step (model-derived subtypes vs reference
    subtypes). Reports per-subtype outcome prevalence and the cross-validated AUROC of predicting
    the outcome from (a) subtype membership, (b) resistance class, and (c) both — quantifying the
    added value of the learned subtypes.

    Parameters
    ----------
    subtypes : numpy.ndarray
        Subtype labels ``(n,)``.
    y : numpy.ndarray
        Binary outcome ``(n,)``.
    resistance : pandas.Series
        Resistance-class baseline aligned to ``subtypes``.
    seed : int, default 42
        CV seed.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Per-subtype prevalence table, and a dict of AUROCs
        (``subtype``, ``resistance``, ``combined``).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = np.asarray(y, dtype=np.int_)
    sub_dummies = pd.get_dummies(pd.Series(subtypes, name="subtype"), prefix="s").to_numpy(float)
    res_dummies = pd.get_dummies(resistance.astype("string"), prefix="r").to_numpy(float)
    combined = np.hstack([sub_dummies, res_dummies])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def _auroc(features: FloatArray) -> float:
        proba = cross_val_predict(
            LogisticRegression(max_iter=1000, random_state=seed),
            features,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        return float(roc_auc_score(y, proba))

    aurocs = {
        "subtype": _auroc(sub_dummies),
        "resistance": _auroc(res_dummies),
        "combined": _auroc(combined),
    }
    prevalence = (
        pd.DataFrame({"subtype": subtypes, "y": y})
        .groupby("subtype")["y"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "outcome_rate", "size": "n"})
        .reset_index()
    )
    logger.info(
        "Subtype prognostic AUROC: subtype=%.3f resistance=%.3f combined=%.3f",
        aurocs["subtype"],
        aurocs["resistance"],
        aurocs["combined"],
    )
    return prevalence, aurocs


def embed_2d(z: FloatArray, seed: int = 42) -> FloatArray:
    """Embed the loadings ``Z`` into 2-D for visualisation (UMAP, PCA fallback).

    Parameters
    ----------
    z : numpy.ndarray
        Archetype loadings ``(n, K)``.
    seed : int, default 42
        Random seed.

    Returns
    -------
    numpy.ndarray
        2-D embedding ``(n, 2)``.
    """
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3)
        return np.asarray(reducer.fit_transform(z), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - environment fallback
        logger.warning("UMAP unavailable (%s); falling back to PCA.", exc)
        from sklearn.decomposition import PCA

        return np.asarray(PCA(n_components=2, random_state=seed).fit_transform(z), dtype=np.float64)


def archetype_treatment_profiles(
    archetypes: FloatArray, treatment_names: list[str]
) -> pd.DataFrame:
    """Tabulate archetype regimen-coefficient profiles for the atlas figure.

    Parameters
    ----------
    archetypes : numpy.ndarray
        Array ``(K, x_dim + 1)`` from :func:`extract_archetypes`.
    treatment_names : list of str
        Treatment column names (length ``x_dim``).

    Returns
    -------
    pandas.DataFrame
        Rows ``archetype_0..K-1``, columns the treatments (intercept dropped). Values are
        context-conditioned log-odds regimen coefficients.
    """
    x_dim = len(treatment_names)
    betas = archetypes[:, :x_dim]
    idx = [f"archetype_{k}" for k in range(archetypes.shape[0])]
    return pd.DataFrame(betas, index=idx, columns=treatment_names)


def subtype_context_profile(
    subtypes: NDArray[np.int_], context_raw: pd.DataFrame, numeric_cols: list[str]
) -> pd.DataFrame:
    """Characterise each subtype by its mean context features (clinical interpretation).

    Numeric features are reported as the subtype mean minus the cohort mean, in cohort standard-
    deviation units (so positive = enriched). Binary/indicator numeric features (e.g. comorbidity
    flags coded 0/1) are included as numeric and read as prevalence differences.

    Parameters
    ----------
    subtypes : numpy.ndarray
        Subtype labels ``(n,)``.
    context_raw : pandas.DataFrame
        Raw (pre-one-hot) context matrix aligned to ``subtypes``.
    numeric_cols : list of str
        Numeric context columns to profile.

    Returns
    -------
    pandas.DataFrame
        Rows are features, columns are subtypes; values are standardised enrichment scores.
    """
    df = context_raw.copy()
    df["_subtype"] = subtypes
    mu = df[numeric_cols].mean()
    sd = df[numeric_cols].std(ddof=0).replace(0, np.nan)
    rows = {}
    for label in sorted(np.unique(subtypes).tolist()):
        grp = df[df["_subtype"] == label]
        rows[f"subtype_{int(label)}"] = (grp[numeric_cols].mean() - mu) / sd
    return pd.DataFrame(rows)


def subtype_prevalence_by_group(
    subtypes: NDArray[np.int_], group: pd.Series[Any], min_group_n: int = 30
) -> pd.DataFrame:
    """Compute subtype prevalence within each group level (e.g. country).

    Parameters
    ----------
    subtypes : numpy.ndarray
        Subtype labels ``(n,)``.
    group : pandas.Series
        Grouping variable aligned to ``subtypes`` (e.g. ``country``).
    min_group_n : int, default 30
        Groups with fewer than this many patients are dropped (unstable proportions).

    Returns
    -------
    pandas.DataFrame
        Rows are group levels, columns are subtypes; values are within-group proportions
        (each row sums to 1). A ``n`` column records the group size.
    """
    df = pd.DataFrame({"group": np.asarray(group), "subtype": subtypes})
    counts = df.groupby("group")["subtype"].size()
    keep = counts[counts >= min_group_n].index
    df = df[df["group"].isin(keep)]
    tab = pd.crosstab(df["group"], df["subtype"], normalize="index")
    tab.columns = pd.Index([f"subtype_{int(c)}" for c in tab.columns])
    tab["n"] = counts.reindex(tab.index).astype(int)
    return tab.sort_values("n", ascending=False)


def _match_archetypes(reference: FloatArray, other: FloatArray) -> tuple[NDArray[np.int_], float]:
    """Align two archetype sets by maximising total absolute correlation (Hungarian).

    Parameters
    ----------
    reference, other : numpy.ndarray
        Archetype matrices ``(K, d)``.

    Returns
    -------
    tuple of (numpy.ndarray, float)
        The column permutation mapping ``other`` rows onto ``reference`` rows, and the mean matched
        absolute Pearson correlation.
    """
    k = reference.shape[0]
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            corr[i, j] = abs(np.corrcoef(reference[i], other[j])[0, 1])
    corr = np.nan_to_num(corr)
    row, col = linear_sum_assignment(-corr)
    return col, float(corr[row, col].mean())


def archetype_stability(clf: ContextualizedClassifier) -> tuple[float, FloatArray]:
    """Quantify archetype reproducibility across the model's bootstraps.

    Each bootstrap learns its own archetypes (subject to label switching). Bootstrap 0 is taken as
    the reference; every other bootstrap is aligned to it with the Hungarian algorithm, and the mean
    matched absolute correlation is reported. High values indicate the archetypes are a stable,
    reproducible decomposition rather than a fitting artifact.

    Parameters
    ----------
    clf : ContextualizedClassifier
        Fitted classifier with ``n_bootstraps > 1``.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        Mean matched correlation across bootstraps, and the per-bootstrap matched-correlation array.
    """
    n_boot = len(clf.models)
    ref = extract_archetypes(clf, bootstrap=0)
    scores = []
    for b in range(1, n_boot):
        other = extract_archetypes(clf, bootstrap=b)
        _, score = _match_archetypes(ref, other)
        scores.append(score)
    arr = np.asarray(scores, dtype=np.float64)
    mean_score = float(arr.mean()) if len(arr) else float("nan")
    logger.info(
        "Archetype stability across %d bootstraps: mean matched |r| = %.3f", n_boot, mean_score
    )
    return mean_score, arr


@dataclass
class EffectTables:
    """Tidy coefficient tables for the homogeneous and heterogeneous heatmaps.

    Attributes
    ----------
    homogeneous_predictor : pandas.DataFrame
        Population-average regimen coefficients ``beta`` with significance (columns ``treatment``,
        ``effect``, ``pval``, ``neg_log10_p``).
    homogeneous_context : pandas.DataFrame
        Direct context-intercept ``mu(C)`` significance (columns ``context``, ``pval``,
        ``neg_log10_p``).
    heterogeneous : pandas.DataFrame
        Context-by-regimen coefficient heterogeneity ``beta(C)`` as ``neg_log10_p``
        (rows context, columns treatment).
    """

    homogeneous_predictor: pd.DataFrame
    homogeneous_context: pd.DataFrame
    heterogeneous: pd.DataFrame


def compute_effect_tables(
    clf: ContextualizedClassifier,
    c_dense: FloatArray,
    context_names: list[str],
    treatment_names: list[str],
) -> EffectTables:
    """Compute homogeneous and heterogeneous coefficient-significance tables.

    Wraps the ``contextualized.analysis`` p-value functions (which derive significance from
    bootstrap variation; the minimum attainable p-value is ``1 / n_bootstraps``) and the
    per-episode coefficient magnitudes from ``predict_params``.

    Parameters
    ----------
    clf : ContextualizedClassifier
        Fitted classifier.
    c_dense : numpy.ndarray
        Dense context matrix used for fitting.
    context_names : list of str
        Names of the (expanded) context dimensions.
    treatment_names : list of str
        Treatment names.

    Returns
    -------
    EffectTables
        The three tidy effect tables.
    """
    from contextualized.analysis import (
        calc_heterogeneous_predictor_effects_pvals,
        calc_homogeneous_context_effects_pvals,
        calc_homogeneous_predictor_effects_pvals,
    )

    eps = 1e-12
    homo_pred_p = np.asarray(calc_homogeneous_predictor_effects_pvals(clf, c_dense)).reshape(-1)
    homo_ctx_p = np.asarray(calc_homogeneous_context_effects_pvals(clf, c_dense)).reshape(-1)
    het_p = np.asarray(calc_heterogeneous_predictor_effects_pvals(clf, c_dense))
    if het_p.ndim == 3:
        het_p = het_p[:, :, 0]  # (context, treatment)

    betas, _ = clf.predict_params(c_dense, model_includes_mus=True)
    mean_beta = np.asarray(betas)[:, 0, :].mean(axis=0)  # population-average regimen coefficient

    homo_pred = pd.DataFrame(
        {
            "treatment": treatment_names,
            "effect": mean_beta,
            "pval": homo_pred_p,
            "neg_log10_p": -np.log10(homo_pred_p + eps),
        }
    ).sort_values("neg_log10_p", ascending=False)

    homo_ctx = pd.DataFrame(
        {
            "context": context_names,
            "pval": homo_ctx_p,
            "neg_log10_p": -np.log10(homo_ctx_p + eps),
        }
    ).sort_values("neg_log10_p", ascending=False)

    het = pd.DataFrame(-np.log10(het_p + eps), index=context_names, columns=treatment_names)

    return EffectTables(
        homogeneous_predictor=homo_pred,
        homogeneous_context=homo_ctx,
        heterogeneous=het,
    )
