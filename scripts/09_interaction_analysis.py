"""Heterogeneity analysis (faithful to OG ``test_each_context``) on the linear-archetype model.

For each interpretable context feature, isolates it, and tests whether each drug's effect varies
with it (sign-consistency bootstrap p-value). This replaces the earlier, statistically-flawed
``interaction_scan`` (which conflated context main effects with interactions and produced
sign-flipped results). Writes a signed -log10(p) heatmap + a ranked significance table.

Usage
-----
    uv run python scripts/09_interaction_analysis.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import METRICS_DIR, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import heterogeneous_effect_pvals, select_stable_seed
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import (
    figure_heterogeneity_clustermap,
    figure_heterogeneity_pvals,
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the faithful heterogeneity analysis and write figure + table.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x, c, y = (
        data["X"].astype(np.float64),
        data["C"].astype(np.float64),
        data["y"].astype(np.float64),
    )
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = data["cond_ids"]
    meta = pd.read_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet").loc[cond_ids]
    meta = meta.reset_index(drop=True)

    age = pd.to_numeric(meta["age_of_onset"], errors="coerce").fillna(40)
    country = meta["country"].astype("string")
    # Sub-Saharan Africa cohort (HIV-TB co-epidemic) vs former-Soviet cohort (MDR-driven).
    ssa = country.isin(["Nigeria", "South Africa"])
    context_features = {
        "age>50": (age > 50).astype(float).to_numpy(),
        "HIV": (meta["comorb_hiv"] == 1).astype(float).to_numpy(),
        "anemia": (meta["comorb_anemia"] == 1).astype(float).to_numpy(),
        "diabetes": (meta["comorb_diabetes"] == 1).astype(float).to_numpy(),
        "psychiatric": (meta["comorb_psychiatric"] == 1).astype(float).to_numpy(),
        "renal": (meta["comorb_renal"] == 1).astype(float).to_numpy(),
        "smear+": (meta["baseline_smear"] == "positive").astype(float).to_numpy(),
        "MDR": (meta["type_of_resistance"] == "MDR non XDR").astype(float).to_numpy(),
        "XDR": (meta["type_of_resistance"] == "XDR").astype(float).to_numpy(),
        "drug-sensitive": (meta["type_of_resistance"] == "Sensitive").astype(float).to_numpy(),
        "Ukraine": (country == "Ukraine").astype(float).to_numpy(),
        "Georgia": (country == "Georgia").astype(float).to_numpy(),
        "Moldova": (country == "Moldova").astype(float).to_numpy(),
        "Sub-Saharan Africa": ssa.astype(float).to_numpy(),
    }

    logger.info(
        "Heterogeneity test: %d contexts x %d drugs (faithful method)...",
        len(context_features),
        len(drugs),
    )
    scan = heterogeneous_effect_pvals(
        c,
        x,
        y,
        context_features,
        drugs,
        n_archetypes=5,
        n_boot=100,
        epochs=250,
    )
    scan.to_csv(TABLES_DIR / "heterogeneity_pvals.csv", index=False)
    figure_heterogeneity_pvals(scan)
    figure_heterogeneity_clustermap(scan)

    n_sig = int(scan["significant"].sum())
    sig = scan[scan["significant"]].copy()
    sig = sig.reindex(sig["pval"].sort_values().index)
    logger.info("Significant heterogeneous effects: %d of %d", n_sig, len(scan))
    logger.info(
        "Significant (p<0.05):\n%s",
        sig[["context", "drug", "delta_beta", "pval"]].round(3).to_string(index=False),
    )

    stable_seed = select_stable_seed(c, x, y, n_archetypes=5, drug_names=drugs, seeds=range(10))
    (METRICS_DIR / "heterogeneity_summary.json").write_text(
        pd.Series(
            {
                "n_tested": int(len(scan)),
                "n_significant": n_sig,
                "stable_subtype_seed": stable_seed,
                "significant": sig[["context", "drug", "delta_beta", "pval"]]
                .round(3)
                .to_dict(orient="records"),
            }
        ).to_json(indent=2)
    )
    logger.info("Heterogeneity analysis written (stable seed=%d).", stable_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
