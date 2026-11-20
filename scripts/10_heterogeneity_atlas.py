"""All-context heterogeneity atlas for the PATCH primary endpoint.

Trains the single joint 335-feature contextualized model (the headline model that gives every
episode a context-conditioned coefficient vector ``beta(C_i)``) and decomposes that episode-level
variation across **every interpretable context** such as demographics, comorbidities, resistance,
per-drug DST, imaging, lineage, and region. It writes a full
``heterogeneity_atlas_unfavorable_{view}.csv`` table and a clustered heatmap. See
:func:`ctb.models.archetype_contextualized.heterogeneity_atlas` for the method and its caveats.

Usage
-----
    uv run python scripts/10_heterogeneity_atlas.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_masks, build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import atlas_bootstrap, scan_from_atlas_boot
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_heterogeneity_clustermap

logger = logging.getLogger(__name__)

# Level values that denote missingness rather than a real patient context (excluded from the atlas).
FIG_TOP_N = 45  # contexts shown in each clustermap (full set is always written to CSV)

_OUTCOME_TITLE = "Unfavorable-outcome (died/failure) coefficient heterogeneity"




def main() -> int:
    """Run the Method-B heterogeneity atlas for all three outcomes.

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

    # Raw context (for interpretable masks/labels) and the three outcomes on the same cohort.
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    c_raw, spec = build_context_matrix(pc, bx, cond_ids)
    c_raw = c_raw.reindex(pd.Index(cond_ids, name="condition_id"))
    masks = build_context_masks(c_raw, spec)

    y = data["y"].astype(np.float64)
    logger.info("=== Heterogeneity atlas: died_or_failure (prevalence %.1f%%) ===", 100 * y.mean())
    # One bootstrap of the joint model; derive both the raw and interaction-isolated views.
    boot, names = atlas_bootstrap(c, x, y, masks, n_archetypes=5, n_boot=40, epochs=2000)
    for view, center in (("raw", False), ("interaction", True)):
        scan = scan_from_atlas_boot(boot, names, drugs, center_drugs=center)
        scan = scan.sort_values("pval").reset_index(drop=True)
        stem = f"heterogeneity_atlas_unfavorable_{view}"
        scan.to_csv(TABLES_DIR / f"{stem}.csv", index=False)
        n_sig = int(scan["significant"].sum())
        logger.info("died_or_failure [%s]: %d of %d pairs significant", view, n_sig, len(scan))
        suffix = "" if view == "raw" else " (drug-specific)"
        figure_heterogeneity_clustermap(
            scan,
            name=stem,
            top_n=FIG_TOP_N,
            title=_OUTCOME_TITLE + suffix,
        )

    logger.info("Heterogeneity atlas complete for the died_or_failure endpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
