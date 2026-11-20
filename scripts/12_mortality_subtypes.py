"""Characterize the contextualized subtypes for the unfavorable-treatment-response outcome.

The archetype sweep (script 11) shows the cohort supports ~4 effective subtypes regardless of K, so
this fits the headline ``K = 5`` contextualized model on the **unfavorable treatment response**
(died OR failure) outcome, hard-assigns each patient to their top-loading subtype, and describes:

* who each subtype is (clinical / comorbidity / resistance / regional enrichment),
* their outcome rate (unfavorable %),
* their per-drug context-conditioned coefficient profile ``beta``,
* a 2-D UMAP of the loadings coloured by subtype, resistance, and outcome.

Usage
-----
    uv run python scripts/12_mortality_subtypes.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, SEED, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    select_n_archetypes,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_subtype_characterization,
    figure_subtype_drug_response,
    figure_subtype_umap,
)

logger = logging.getLogger(__name__)

MIN_SUBTYPE_FRAC = 0.03  # every retained subtype must hold >=3% of patients (interpretability rule)
EPOCHS = 2000  # selection and final fit MUST share epochs (the 5-way split is epoch-sensitive)
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)


def main() -> int:
    """Fit and characterize the K=5 subtypes for the unfavorable-outcome model.

    Returns
    -------
    int
        Process exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = list(data["cond_ids"])
    y = data["y"].astype(int)
    meta = pd.read_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet").loc[cond_ids]
    meta = meta.reset_index(drop=True)
    meta["unfavorable"] = y

    # Interpretability-first: largest K whose every subtype clears the 3% floor, with the
    # cross-seed stability assessment in Supplementary Fig. S2 used to support the K=5 choice.
    K, seed = select_n_archetypes(
        c, x, y, drug_names=drugs, k_range=range(8, 3, -1),
        min_subtype_frac=MIN_SUBTYPE_FRAC, epochs=EPOCHS,
    )
    model = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=seed
    )
    train(model, c, x, y, CFG, seed=seed)
    loadings = model.predict_loadings(c)
    subtypes = loadings.argmax(axis=1)
    beta, _ = model.predict_beta(c)
    beta_df = pd.DataFrame(beta, columns=drugs)
    sizes = np.bincount(subtypes, minlength=K).tolist()
    logger.info("K=%d, seed=%d; subtype sizes=%s", K, seed, sizes)

    age = pd.to_numeric(meta["age_of_onset"], errors="coerce")
    ssa = meta["country"].isin(["Nigeria", "South Africa"])
    rows = []
    for s in range(K):
        m = subtypes == s
        if m.sum() == 0:
            continue
        g = meta[m]
        rows.append(
            {
                "subtype": f"S{s}",
                "n": int(m.sum()),
                "conversion_pct": float(g["unfavorable"].mean() * 100),  # outcome rate (reused col)
                "feat_age": float(age[m].mean()),
                "feat_pct_MDR": float((g["type_of_resistance"] == "MDR non XDR").mean() * 100),
                "feat_pct_XDR": float((g["type_of_resistance"] == "XDR").mean() * 100),
                "feat_pct_HIV": float((g["comorb_hiv"] == 1).mean() * 100),
                "feat_pct_smearpos": float((g["baseline_smear"] == "positive").mean() * 100),
                "feat_pct_anemia": float((g["comorb_anemia"] == 1).mean() * 100),
                "feat_pct_diabetes": float((g["comorb_diabetes"] == 1).mean() * 100),
                "feat_pct_psychiatric": float((g["comorb_psychiatric"] == 1).mean() * 100),
                "feat_pct_SSAfrica": float(ssa[m].mean() * 100),
            }
        )
    char = pd.DataFrame(rows).set_index("subtype")
    char.to_csv(TABLES_DIR / "mortality_subtype_characterization.csv")

    feat_cols = [col for col in char.columns if col.startswith("feat_")]
    char_z = char.copy()
    for col in feat_cols:
        mu, sd = char[col].mean(), char[col].std(ddof=0)
        char_z[col] = (char[col] - mu) / sd if sd > 0 else 0.0

    subtype_drug = beta_df.assign(subtype=subtypes).groupby("subtype").mean()
    subtype_drug.index = pd.Index([f"S{i}" for i in subtype_drug.index])
    subtype_drug.to_csv(TABLES_DIR / "mortality_subtype_drug_response.csv")

    figure_subtype_characterization(
        char_z.assign(conversion_pct=char["conversion_pct"], n=char["n"]),
        name="mortality_subtype_characterization",
        outcome_label="Unfavorable",
    )
    figure_subtype_drug_response(subtype_drug, name="mortality_subtype_drug_response")

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.3)
        emb = reducer.fit_transform(loadings)
    except Exception as exc:  # pragma: no cover
        logger.warning("UMAP failed (%s); using PCA.", exc)
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=SEED).fit_transform(loadings)
    figure_subtype_umap(
        np.asarray(emb, dtype=np.float64),
        {
            "Subtype": pd.Series([f"S{s}" for s in subtypes]),
            "Resistance": meta["type_of_resistance"].reset_index(drop=True),
            "Unfavorable": pd.Series(y),
        },
        name="mortality_subtype_umap",
    )
    logger.info("Mortality subtype characterization complete (K=%d).", K)
    return 0


if __name__ == "__main__":
    sys.exit(main())
