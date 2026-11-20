"""Phase 2 archetype / patient-subtype analysis and effect heatmaps.

For each outcome, fits a high-bootstrap contextualized classifier (so effect p-values have
resolution and archetype stability can be assessed), then derives and saves:

* the K=8 treatment-effect **archetype atlas**,
* per-patient **subtype** assignments (arg-max archetype, with a K-means fallback if arg-max
  collapses) and their distribution,
* **archetype stability** across bootstraps (Hungarian-matched correlation),
* **subtype context enrichment** (clinical interpretation),
* **subtype prevalence by country** (regional map),
* **homogeneous** (beta, mu(C)) and **heterogeneous** (beta(C)) effect-significance tables/heatmaps.

A dedicated effects configuration (more bootstraps, fewer epochs) is used here; the headline AUROC
configuration lives in ``scripts/04_train_contextualized.py``.

Usage
-----
    caffeinate -i uv run python scripts/05_archetype_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.context import FeatureSpec
from ctb.models.archetypes import (
    archetype_stability,
    archetype_treatment_profiles,
    assign_subtypes,
    choose_kmeans_subtypes,
    compute_effect_tables,
    embed_2d,
    extract_archetype_weights,
    extract_archetypes,
    subtype_context_profile,
    subtype_outcome_separation,
    subtype_prevalence_by_group,
)
from ctb.models.contextualized import (
    ContextualizedConfig,
    _fit_one,
    make_context_preprocessor,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.archetype_figures import (
    figure_archetype_atlas,
    figure_context_archetype_loadings,
    figure_heterogeneous_effects,
    figure_homogeneous_effects,
    figure_subtype_prevalence_by_country,
    figure_subtype_prognostic,
    figure_subtype_umap,
)

logger = logging.getLogger(__name__)

OUTCOMES = ["mortality", "treatment_intensification"]
N_ARCHETYPES = 8
# Effects/stability config: many bootstraps for p-value resolution; fewer epochs for tractability.
EFFECTS_CONFIG = ContextualizedConfig(
    encoder_type="mlp", num_archetypes=N_ARCHETYPES, n_bootstraps=20, max_epochs=60, es_patience=6
)
# Key categorical context columns to expand for the subtype interpretation heatmap.
INTERP_CATEGORICALS = ["type_of_resistance", "case_definition", "cavitation", "baseline_culture"]


def _interpretation_frame(
    context_raw: pd.DataFrame, spec: FeatureSpec
) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric frame for subtype context enrichment (numeric + one-hot key categoricals).

    Parameters
    ----------
    context_raw : pandas.DataFrame
        Raw context matrix (NaNs allowed).
    spec : FeatureSpec
        Feature specification.

    Returns
    -------
    tuple of (pandas.DataFrame, list of str)
        The interpretation frame and its numeric column names.
    """
    numeric = [c for c in spec.numeric if c in context_raw.columns]
    # comorbidity / indicator flags are stored numeric (0/1) but live in spec.categorical.
    flag_cols = [
        c
        for c in spec.categorical
        if c in context_raw.columns
        and context_raw[c].dropna().isin([0, 1]).all()
        and context_raw[c].dtype != object
    ]
    frame = context_raw[numeric + flag_cols].apply(pd.to_numeric, errors="coerce")
    for col in INTERP_CATEGORICALS:
        if col in context_raw.columns:
            dummies = pd.get_dummies(context_raw[col].astype("string"), prefix=col)
            frame = pd.concat([frame, dummies.astype(float)], axis=1)
    return frame, list(frame.columns)


def _analyse_outcome(
    outcome: str, analysis: pd.DataFrame, spec: FeatureSpec, treatment_cols: list[str]
) -> dict[str, object]:
    """Fit and analyse one outcome; write all tables and figures.

    Parameters
    ----------
    outcome : str
        Outcome column name.
    analysis : pandas.DataFrame
        Analysis frame (treatments + context + outcomes).
    spec : FeatureSpec
        Context feature specification.
    treatment_cols : list of str
        Treatment column names.

    Returns
    -------
    dict
        Summary metrics for this outcome (subtype method, distribution, stability).
    """
    logger.info("=== Archetype analysis: %s ===", outcome)
    X = analysis[treatment_cols]
    context_raw = analysis[spec.all_columns]
    y = analysis[outcome].astype(int)

    pre = make_context_preprocessor(spec)
    c_dense = pre.fit_transform(context_raw).astype(np.float64)
    context_names = list(pre.get_feature_names_out())

    clf = _fit_one(c_dense, X.to_numpy(np.float64), y.to_numpy(np.float64), EFFECTS_CONFIG)

    # --- archetype loadings (persisted so downstream analysis never re-fits the slow model) ---
    z = extract_archetype_weights(clf, c_dense, bootstrap=0)
    archetypes = extract_archetypes(clf, bootstrap=0)
    z_df = pd.DataFrame(z, index=analysis.index, columns=[f"z{k}" for k in range(z.shape[1])])
    z_df.to_parquet(PROCESSED_DIR / f"z_loadings_{outcome}.parquet")

    # --- medical subtypes: balanced K-means on the loadings (silhouette-selected K) ---
    subtypes, n_sub, silhouette = choose_kmeans_subtypes(z)
    argmax_dist = np.bincount(assign_subtypes(z), minlength=N_ARCHETYPES).tolist()
    dist = np.bincount(subtypes, minlength=n_sub)
    logger.info(
        "%s subtypes: K=%d (silhouette=%.3f), sizes=%s; argmax archetype use=%s",
        outcome,
        n_sub,
        silhouette,
        dist.tolist(),
        argmax_dist,
    )

    atlas = archetype_treatment_profiles(archetypes, treatment_cols)
    atlas.to_csv(TABLES_DIR / f"archetype_atlas_{outcome}.csv")

    stability_mean, stability_arr = archetype_stability(clf)

    interp, _ = _interpretation_frame(context_raw, spec)
    ctx_profile = subtype_context_profile(subtypes, interp, list(interp.columns))
    ctx_profile.to_csv(TABLES_DIR / f"subtype_context_profile_{outcome}.csv")

    prevalence = subtype_prevalence_by_group(subtypes, analysis["country"], min_group_n=30)
    prevalence.to_csv(TABLES_DIR / f"subtype_prevalence_by_country_{outcome}.csv")

    # --- prognostic validation: subtypes vs resistance-class baseline (Ellington-style) ---
    prog_prev, aurocs = subtype_outcome_separation(
        subtypes, y.to_numpy(np.int_), analysis["type_of_resistance"]
    )
    prog_prev.to_csv(TABLES_DIR / f"subtype_outcome_rate_{outcome}.csv", index=False)

    effects = compute_effect_tables(clf, c_dense, context_names, treatment_cols)
    effects.homogeneous_predictor.to_csv(
        TABLES_DIR / f"homogeneous_predictor_effects_{outcome}.csv", index=False
    )
    effects.homogeneous_context.to_csv(
        TABLES_DIR / f"homogeneous_context_effects_{outcome}.csv", index=False
    )
    effects.heterogeneous.to_csv(TABLES_DIR / f"heterogeneous_effects_{outcome}.csv")

    # --- figures ---
    figure_archetype_atlas(atlas, outcome)
    figure_subtype_prevalence_by_country(prevalence, outcome)
    figure_context_archetype_loadings(ctx_profile, outcome)
    figure_heterogeneous_effects(effects.heterogeneous, outcome)
    figure_homogeneous_effects(effects.homogeneous_predictor, effects.homogeneous_context, outcome)
    figure_subtype_prognostic(prog_prev, aurocs, outcome)
    embedding = embed_2d(z)
    figure_subtype_umap(
        embedding,
        {
            "Subtype": pd.Series(subtypes),
            "Country": analysis["country"].reset_index(drop=True),
            "Resistance": analysis["type_of_resistance"].reset_index(drop=True),
            outcome: y.reset_index(drop=True),
        },
        outcome,
    )
    logger.info("%s figures written to %s", outcome, FIGURES_DIR)

    return {
        "outcome": outcome,
        "subtype_method": "kmeans",
        "n_subtypes": int(n_sub),
        "silhouette": float(silhouette),
        "subtype_sizes": dist.tolist(),
        "argmax_archetype_use": argmax_dist,
        "archetype_stability_mean_abs_corr": stability_mean,
        "archetype_stability_per_bootstrap": stability_arr.tolist(),
        "prognostic_auroc": aurocs,
        "n_countries_mapped": int(prevalence.shape[0]),
    }


def main() -> int:
    """Run the archetype/subtype analysis for all outcomes.

    Returns
    -------
    int
        Process exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    analysis = pd.read_parquet(PROCESSED_DIR / "analysis_frame.parquet")
    payload = json.loads((PROCESSED_DIR / "feature_spec.json").read_text())
    spec = FeatureSpec(numeric=payload["numeric"], categorical=payload["categorical"])
    treatment_cols = payload["treatment_cols"]

    summary = [_analyse_outcome(o, analysis, spec, treatment_cols) for o in OUTCOMES]
    (METRICS_DIR / "archetype_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved archetype analysis summary for %d outcomes.", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
