"""Modality-coloured UMAP of the multimodal treatment-response subtypes.

Embeds the per-patient archetype loadings in 2-D and colours the same embedding by subtype and by
each dominant modality (resistance class, HIV, bacillary burden, region) — showing the subtypes
physically separate along *different* modality axes.

Usage
-----
    uv run python scripts/23_multimodal_umap.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_subtypes
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_subtype_umap

logger = logging.getLogger(__name__)


def main() -> int:
    """Compute the loadings UMAP and render modality colourings."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    mm = load_primary_subtypes(PROCESSED_DIR / "multimodal_subtypes.npz")
    loadings = mm["loadings"].astype(np.float64)
    sub = mm["subtypes"]
    cond_ids = list(mm["cond_ids"])

    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, _ = build_context_matrix(pc, bx, cond_ids)
    craw = craw.reindex(pd.Index(cond_ids, name="condition_id")).reset_index(drop=True)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        emb = reducer.fit_transform(loadings)
    except Exception as exc:  # pragma: no cover
        logger.warning("UMAP failed (%s); using PCA.", exc)
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=42).fit_transform(loadings)

    res = craw["type_of_resistance"].astype("string")
    res = res.where(res.isin(["Sensitive", "MDR non XDR", "XDR"]), other="Other")
    res = res.reset_index(drop=True)
    color_by = {
        "Subtype": pd.Series([f"S{s}" for s in sub]),
        "Resistance": res,
        "HIV": pd.Series(np.where(craw["comorb_hiv"] == 1, "HIV+", "HIV-")),
        "Bacillary burden (smear)": pd.Series(
            np.where(craw["baseline_smear"] == "positive", "smear+", "smear-")
        ),
        "Region": pd.Series(
            np.where(craw["country"].isin(["Nigeria", "South Africa"]), "Sub-Saharan Africa",
                     np.where(craw["country"] == "Ukraine", "Ukraine", "Other"))
        ),
    }
    figure_subtype_umap(np.asarray(emb, dtype=np.float64), color_by, name="multimodal_umap")
    logger.info("Multimodal UMAP written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
