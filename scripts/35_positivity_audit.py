"""Persist the positivity (treatment-overlap) audit behind the §2.4 firewall.

For each of the five treatment-response subtypes and each of the 26 initial-regimen drugs, compute the
RECEIPT proportion (mean of the binary drug indicator) and classify the subtype x drug cell as
"estimable" (10-90% receipt -> within-subtype overlap exists) or "near-deterministic" (<10% or >90% ->
no counterfactual, effect non-identified). Underpins the manuscript's 61/130-estimable, 69/130
near-deterministic, rifampicin-88%-in-drug-sensitive, ART-<10%-everywhere claims.

DUA-safe: reads only the de-identified processed matrices, emits only group-level aggregates
(proportions and counts) to results/tables/. No row-level record is printed or written.

Run: ``uv run python scripts/35_positivity_audit.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctb.config import PROCESSED_DIR as PROC, TABLES_DIR as TABLES  # noqa: E402
from figures.palette import NARRATIVE_ORDER, subtype_name  # noqa: E402
from ctb.data.artifacts import load_primary_matrix, load_primary_subtypes  # noqa: E402

LOW, HIGH = 10.0, 90.0  # estimable receipt band (%)


def main() -> int:
    cm = load_primary_matrix(PROC / "unfavorable_model_matrices.npz")
    X = cm["X"].astype(float)
    drugs = [str(d) for d in cm["drugs"]]
    cid_x = [str(c) for c in cm["cond_ids"]]

    ms = load_primary_subtypes(PROC / "multimodal_subtypes.npz")
    sub_by_cid = dict(zip([str(c) for c in ms["cond_ids"]], ms["subtypes"].astype(int)))
    subt = np.array([sub_by_cid[c] for c in cid_x])  # subtype per row of X (aligned by cond_id)

    rows = []
    for k in NARRATIVE_ORDER:
        mask = subt == k
        n = int(mask.sum())
        for j, d in enumerate(drugs):
            pct = round(100.0 * float(X[mask, j].mean()), 1)
            band = "estimable" if LOW <= pct <= HIGH else "near-deterministic"
            rows.append({"subtype": subtype_name(k, full=True), "n_subtype": n, "drug": d,
                         "receipt_pct": pct, "band": band})
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "positivity_audit.csv", index=False)

    est = int((df.band == "estimable").sum())
    det = int((df.band == "near-deterministic").sum())
    tot = len(df)
    summary = pd.DataFrame([{
        "total_cells": tot, "estimable_cells": est, "near_deterministic_cells": det,
        "estimable_frac": round(est / tot, 3), "deterministic_frac": round(det / tot, 3),
    }])
    summary.to_csv(TABLES / "positivity_audit_summary.csv", index=False)

    print(f"drugs ({len(drugs)}): {drugs}")
    print(f"\nestimable {est} / near-deterministic {det} / total {tot} "
          f"({100*est/tot:.0f}% estimable)")
    # canonical verification cells (aggregate only)
    rif = [d for d in drugs if d in ("R", "drug_R", "Rifampicin", "rifampicin")]
    art = [d for d in drugs if "ART" in d or d in ("drug_ART",)]
    for label, ds in [("rifampicin", rif), ("antiretroviral", art)]:
        for d in ds:
            sub_df = df[df.drug == d][["subtype", "receipt_pct"]]
            print(f"\n{label} ({d}) receipt by subtype:")
            print(sub_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
