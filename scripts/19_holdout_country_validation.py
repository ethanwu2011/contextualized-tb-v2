"""Held-out-country (internal external) validation.

A stopgap for true external validation: treat each major country as an unseen "site." (A) Leave one
country out, train the contextualized model on the rest, and measure held-out AUROC — does the model
generalize across sites with different case mixes? (B) For the largest held-out country, refit the
subtype model on the remaining cohort and apply it to the held-out patients, checking that the
rifampicin / drug-sensitivity split re-emerges out-of-distribution.

Usage
-----
    uv run python scripts/19_holdout_country_validation.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctb.config import FIGURES_DIR, SEED, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import apply_context_encoder, build_context_matrix, fit_context_encoder
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    select_n_archetypes,
    train,
)
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.style import PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt  # noqa: E402

EPOCHS = 2000
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
MIN_TEST_N = 200  # countries with at least this many patients are held out


def main() -> int:
    """Run leave-one-country-out AUROC and held-out subtype-structure checks."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    x = data["X"].astype(np.float64)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    cond_ids = list(data["cond_ids"])
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, spec = build_context_matrix(pc, bx, cond_ids)
    craw = craw.reindex(pd.Index(cond_ids, name="condition_id")).reset_index(drop=True)
    country = craw["country"].astype("string").to_numpy()
    y = data["y"].astype(np.float64)
    got_r = x[:, drugs.index("R")]

    # --- (A) leave-one-country-out AUROC ---
    counts = pd.Series(country).value_counts()
    test_countries = [ct for ct, n in counts.items() if n >= MIN_TEST_N and str(ct) != "<NA>"]
    rows = []
    for ct in test_countries:
        te = country == ct
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue
        encoder = fit_context_encoder(craw[tr], spec)
        c_tr = apply_context_encoder(craw[tr], spec, encoder).to_numpy(dtype=np.float64)
        c_te = apply_context_encoder(craw[te], spec, encoder).to_numpy(dtype=np.float64)
        m = LinearArchetypeContextualized(
            c_tr.shape[1], x.shape[1], n_archetypes=5, simplex=True, seed=SEED
        )
        train(m, c_tr, x[tr], y[tr], CFG, seed=SEED)
        auc = roc_auc_score(y[te], m.predict_proba(c_te, x[te]))
        rows.append({"held_out_country": str(ct), "n": int(te.sum()),
                     "unfavorable_pct": float(y[te].mean() * 100), "heldout_auroc": float(auc)})
        logger.info("Held out %-14s n=%4d AUROC=%.3f", ct, int(te.sum()), auc)
    tab = pd.DataFrame(rows).sort_values("heldout_auroc")
    tab.to_csv(TABLES_DIR / "holdout_country_auroc.csv", index=False)

    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 0.35 * len(tab) + 1.4))
    ax.barh(tab["held_out_country"], tab["heldout_auroc"], color=PALETTE[0])
    ax.axvline(0.79, color="#999999", ls="--", lw=1, label="full-cohort CV AUROC")
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.text(r["heldout_auroc"], i, f"  {r['heldout_auroc']:.2f}", va="center", fontsize=7)
    ax.set(xlim=(0.5, 0.9), xlabel="Held-out AUROC (trained on other countries)",
           ylabel="Held-out country", title="Cross-site generalization")
    ax.legend(fontsize=7, frameon=False)
    save_figure(fig, FIGURES_DIR / "holdout_country_auroc")

    # --- (B) held-out subtype recovery for the largest country ---
    big = str(counts.index[0])
    te = country == big
    tr = ~te
    encoder = fit_context_encoder(craw[tr], spec)
    c_tr = apply_context_encoder(craw[tr], spec, encoder).to_numpy(dtype=np.float64)
    c_te = apply_context_encoder(craw[te], spec, encoder).to_numpy(dtype=np.float64)
    k, seed = select_n_archetypes(
        c_tr, x[tr], y[tr], drug_names=drugs, k_range=range(8, 3, -1),
        min_subtype_frac=0.03, epochs=EPOCHS,
    )
    m = LinearArchetypeContextualized(
        c_tr.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=seed
    )
    train(m, c_tr, x[tr], y[tr], CFG, seed=seed)
    sub_te = m.predict_loadings(c_te).argmax(axis=1)
    sizes = np.bincount(sub_te, minlength=k)
    bigs = np.argsort(sizes)[::-1][:2]
    recov = [{"held_out": big, "subtype": f"S{int(s)}", "n": int((sub_te == s).sum()),
              "got_rifampicin_pct": float(got_r[te][sub_te == s].mean() * 100)} for s in bigs]
    pd.DataFrame(recov).to_csv(TABLES_DIR / "holdout_subtype_recovery.csv", index=False)
    logger.info("Held-out (%s) two largest subtypes rifampicin%%: %s",
                big, [round(r["got_rifampicin_pct"]) for r in recov])
    return 0


if __name__ == "__main__":
    sys.exit(main())
