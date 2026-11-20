"""Validation: treatment-response subtypes align with the recorded resistance context.

The contextualized model is trained on treatments (X), context (C), and the died/failure outcome
(Y), with recorded resistance and susceptibility variables included in the context. This script
refits the reproducible K=5 model, hard-assigns subtypes, and describes how the two good-prognosis
the subtypes split on rifampicin receipt and recorded drug-susceptibility labels.

Usage
-----
    uv run python scripts/16_subtype_validation.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    select_n_archetypes,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_subtype_resistance_validation

logger = logging.getLogger(__name__)

EPOCHS = 2000
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)


def main() -> int:
    """Build the subtype and resistance-context validation table and figure.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = list(data["cond_ids"])
    meta = pd.read_parquet(PROCESSED_DIR / "unfavorable_context_meta.parquet").loc[cond_ids]
    meta = meta.reset_index(drop=True)

    y = data["y"].astype(np.float64)

    k, seed = select_n_archetypes(
        c, x, y, drug_names=drugs, k_range=range(8, 3, -1), min_subtype_frac=0.03, epochs=EPOCHS
    )
    model = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=seed
    )
    train(model, c, x, y, CFG, seed=seed)
    subtypes = model.predict_loadings(c).argmax(axis=1)

    got_r = x[:, drugs.index("R")]
    res = meta["type_of_resistance"].astype("string")
    rows = []
    for s in range(k):
        m = subtypes == s
        if m.sum() == 0:
            continue
        rows.append(
            {
                "subtype": f"S{s}",
                "unfavorable %": float(y[m].mean() * 100),
                "got Rifampicin": float(got_r[m].mean() * 100),
                "Drug-sensitive": float((res[m] == "Sensitive").mean() * 100),
                "MDR": float((res[m] == "MDR non XDR").mean() * 100),
                "XDR": float((res[m] == "XDR").mean() * 100),
            }
        )
    table = pd.DataFrame(rows).set_index("subtype")
    table.to_csv(TABLES_DIR / "subtype_resistance_validation.csv")
    figure_subtype_resistance_validation(table)
    logger.info("Subtype-resistance validation:\n%s", table.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
