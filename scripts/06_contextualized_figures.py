"""Generate a legacy contextualized decomposition on the PATCH primary endpoint.

Fits the linear-archetype contextualized model (K=5, simplex loadings) on the multi-drug
baseline culture-positive data and writes:

* homogeneous treatment effects (population β per drug),
* treatment-effect heterogeneity by drug (SD of β(C)),
* the K archetype drug-response profiles (atlas),
* a UMAP of patient loadings Z coloured by subtype, resistance, and outcome,
* per-subtype died-or-failure rate and clinical profile.

Usage
-----
    uv run python scripts/06_contextualized_figures.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import METRICS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_archetype_atlas,
    figure_heterogeneity_by_drug,
    figure_heterogeneous_effects_by_context,
    figure_homogeneous_effects,
    figure_subtype_characterization,
    figure_subtype_drug_response,
    figure_subtype_umap,
)

logger = logging.getLogger(__name__)

N_ARCHETYPES = 5
STABLE_SEED = 9  # from select_stable_seed: balanced, distinct, populated subtypes
USE_SIMPLEX = True


def main() -> int:
    """Fit the model and generate all contextualized-decomposition figures.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x, c, y = data["X"], data["C"], data["y"]
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = data["cond_ids"]
    meta = pd.read_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet").loc[cond_ids]

    model = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=N_ARCHETYPES, simplex=USE_SIMPLEX, seed=STABLE_SEED
    )
    # Stable seed (selected by select_stable_seed) gives balanced, distinct, populated subtypes;
    # most seeds collapse one or more subtypes to ~0.
    train(
        model, c, x, y,
        TrainConfig(epochs=2500, lr=0.02, weight_decay=1e-3, archetype_l1=2e-3,
                    entropy_weight=1.0),
        seed=STABLE_SEED,
    )

    beta, _ = model.predict_beta(c)
    loadings = model.predict_loadings(c)
    archetypes = model.normalized_archetypes()
    subtypes = loadings.argmax(axis=1)
    beta_df = pd.DataFrame(beta, columns=drugs)

    # --- tables ---
    homo = pd.Series(beta.mean(axis=0), index=drugs, name="homogeneous_effect")
    het = pd.Series(beta.std(axis=0), index=drugs, name="heterogeneity_sd")
    atlas = pd.DataFrame(archetypes, columns=drugs, index=[f"arch{k}" for k in range(N_ARCHETYPES)])
    homo.to_csv(TABLES_DIR / "ctx_homogeneous_effects.csv")
    het.to_csv(TABLES_DIR / "ctx_heterogeneity_by_drug.csv")
    atlas.to_csv(TABLES_DIR / "ctx_archetype_atlas.csv")

    # --- subtype characterization ---
    meta = meta.copy()
    meta["subtype"] = subtypes
    rows = []
    for s in range(N_ARCHETYPES):
        g = meta[meta["subtype"] == s]
        if len(g) == 0:
            continue
        age = pd.to_numeric(g["age_of_onset"], errors="coerce")
        rows.append(
            {
                "subtype": s,
                "n": len(g),
                "conversion_pct": float(g["died_or_failure"].mean() * 100),
                "feat_age": float(age.mean()),
                "feat_pct_MDR": float((g["type_of_resistance"] == "MDR non XDR").mean() * 100),
                "feat_pct_XDR": float((g["type_of_resistance"] == "XDR").mean() * 100),
                "feat_pct_HIV": float(g["comorb_hiv"].mean() * 100),
                "feat_pct_smearpos": float((g["baseline_smear"] == "positive").mean() * 100),
                "feat_pct_anemia": float(g["comorb_anemia"].mean() * 100),
            }
        )
    char = pd.DataFrame(rows).set_index("subtype")
    # standardize feature columns to enrichment (z) for the heatmap
    feat_cols = [col for col in char.columns if col.startswith("feat_")]
    char_z = char.copy()
    for col in feat_cols:
        mu, sd = char[col].mean(), char[col].std(ddof=0)
        char_z[col] = (char[col] - mu) / sd if sd > 0 else 0.0
    char.to_csv(TABLES_DIR / "ctx_subtype_characterization.csv")

    # --- subtype x drug-response profile (the money figure) ---
    beta_df = beta_df.copy()
    beta_df["subtype"] = subtypes
    subtype_drug = beta_df.groupby("subtype").mean()
    subtype_drug.index = pd.Index([f"S{i}" for i in subtype_drug.index])
    subtype_drug.to_csv(TABLES_DIR / "ctx_subtype_drug_response.csv")

    # --- heterogeneous effects by context stratum ---
    strata = {
        "HIV+": meta["comorb_hiv"] == 1,
        "HIV-": meta["comorb_hiv"] == 0,
        "smear+": meta["baseline_smear"] == "positive",
        "smear-": meta["baseline_smear"] == "negative",
        "MDR": meta["type_of_resistance"] == "MDR non XDR",
        "XDR": meta["type_of_resistance"] == "XDR",
        "Sensitive": meta["type_of_resistance"] == "Sensitive",
    }
    beta_only = beta_df.drop(columns="subtype")
    het_rows = {
        nm: beta_only[mask.to_numpy()].mean()
        for nm, mask in strata.items()
        if mask.to_numpy().sum() > 30
    }
    het_context = pd.DataFrame(het_rows).T
    het_context.to_csv(TABLES_DIR / "ctx_heterogeneous_effects_by_context.csv")

    # --- figures ---
    figure_homogeneous_effects(homo)
    figure_heterogeneity_by_drug(het)
    figure_subtype_drug_response(subtype_drug)
    figure_heterogeneous_effects_by_context(het_context)
    figure_archetype_atlas(atlas)
    figure_subtype_characterization(
        char_z.assign(conversion_pct=char["conversion_pct"], n=char["n"]),
        outcome_label="Died/failure",
    )

    # UMAP of loadings
    try:
        import umap

        emb = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=30, min_dist=0.3
        ).fit_transform(loadings)
    except Exception as exc:  # pragma: no cover
        logger.warning("UMAP failed (%s); using PCA.", exc)
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=42).fit_transform(loadings)
    figure_subtype_umap(
        np.asarray(emb, dtype=np.float64),
        {
            "Subtype": pd.Series(subtypes),
            "Resistance": meta["type_of_resistance"].reset_index(drop=True),
            "Converted": pd.Series(y),
        },
    )

    summary = {
        "n_archetypes": N_ARCHETYPES,
        "simplex": USE_SIMPLEX,
        "subtype_sizes": np.bincount(subtypes, minlength=N_ARCHETYPES).tolist(),
            "subtype_unfavorable_pct": char["conversion_pct"].round(1).tolist(),
        "most_heterogeneous_drugs": het.sort_values(ascending=False).head(4).round(3).to_dict(),
    }
    (METRICS_DIR / "ctx_decomposition_summary.json").write_text(
        pd.Series(summary).to_json(indent=2)
    )
    logger.info("Contextualized decomposition figures + tables written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
