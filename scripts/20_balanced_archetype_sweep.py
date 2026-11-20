"""Balanced archetype sweep (K=5..9) on non-resistance context.

Tests how many reproducible, *distinct* treatment-response subtypes the cohort supports once (a)
resistance/DST features are removed from the context for a resistance-excluded sensitivity analysis and (b) a
load-balancing penalty prevents the degenerate one-dominant-blob optimum. For each K the penalty cap
scales as 2/K so higher K is genuinely forced to spread out. We report, per K: how many subtypes
clear the 3% floor, cross-seed reproducibility (mean adjusted Rand index), and subtype distinctness
(mean / min pairwise distance of standardized clinical profiles). The right number of subtypes is
the largest K that stays populated, reproducible, and non-redundant.

Usage
-----
    uv run python scripts/20_balanced_archetype_sweep.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ctb.config import TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K_RANGE = range(5, 10)
N_SEEDS = 3
BALANCE_WEIGHT = 3.0
EPOCHS = 2000


def _build_no_resistance_context() -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Return (encoded C without resistance, X, drug names, eval-feature frame)."""
    data = load_primary_matrix()
    ids = list(data["cond_ids"])
    x = data["X"].astype(float)
    drugs = [s.replace("drug_", "") for s in data["drugs"]]
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, spec = build_context_matrix(pc, bx, ids)
    craw = craw.reindex(pd.Index(ids, name="condition_id"))
    drop = {c for c in spec.categorical if c == "type_of_resistance" or c.startswith("bactec_")}
    num = list(spec.numeric)
    cat = [c for c in spec.categorical if c not in drop]
    nn = craw[num].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num if nn[c].notna().any()]
    ns = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(nn[keep]))
    oh = pd.get_dummies(craw[cat].astype("string"), dummy_na=True)
    c = np.hstack([ns, oh.to_numpy(float)])
    evalf = pd.DataFrame(
        {
            "age": pd.to_numeric(craw["age_of_onset"], errors="coerce").to_numpy(),
            "hiv": (craw["comorb_hiv"] == 1).to_numpy().astype(float),
            "smear": (craw["baseline_smear"] == "positive").to_numpy().astype(float),
            "ssa": craw["country"].isin(["Nigeria", "South Africa"]).to_numpy().astype(float),
            "sens": (craw["type_of_resistance"].astype("string") == "Sensitive")
            .to_numpy()
            .astype(float),
            "gotR": x[:, drugs.index("R")],
        }
    )
    return c, x, drugs, evalf


def main() -> int:
    """Run the balanced archetype sweep and report populated/reproducible/distinct subtypes."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    c, x, drugs, evalf = _build_no_resistance_context()
    npz = load_primary_matrix()
    y = npz["y"].astype(float)
    evalf["unfav"] = y
    logger.info("C (no resistance) dim = %d", c.shape[1])

    rows = []
    for k in K_RANGE:
        cap = 2.0 / k
        cfg = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                          balance_weight=BALANCE_WEIGHT, balance_cap=cap)
        subs = []
        for seed in range(N_SEEDS):
            m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=k,
                                              simplex=True, seed=seed)
            train(m, c, x, y, cfg, seed=seed)
            subs.append(m.predict_loadings(c).argmax(axis=1))
        # reproducibility: mean pairwise ARI across seeds
        aris = [adjusted_rand_score(subs[i], subs[j])
                for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
        # representative = most balanced seed (max min subtype fraction)
        best = max(subs, key=lambda s: (np.bincount(s, minlength=k) / len(y)).min())
        fr = np.bincount(best, minlength=k) / len(y)
        n_pop = int((fr >= 0.03).sum())
        # distinctness: standardized subtype profiles on eval features (populated subtypes only)
        prof = evalf.assign(_s=best).groupby("_s").mean()
        profz = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
        dmat = pdist(profz.to_numpy())
        rows.append({
            "K": k, "n_populated_>=3%": n_pop, "min_frac": float(fr.min()),
            "mean_ARI": float(np.mean(aris)), "mean_profile_dist": float(dmat.mean()),
            "min_profile_dist": float(dmat.min()),
        })
        logger.info(
            "K=%d cap=%.2f | populated=%d/%d min_frac=%.0f%% | ARI=%.2f | dist mean=%.2f min=%.2f",
            k, cap, n_pop, k, 100 * fr.min(), np.mean(aris), dmat.mean(), dmat.min(),
        )
        # print representative profiles
        rep = prof.copy()
        rep["n"] = np.bincount(best, minlength=k)[prof.index.to_numpy()]
        rep["unfav%"] = (rep["unfav"] * 100).round(0)
        for col in ["hiv", "smear", "ssa", "sens", "gotR"]:
            rep[col] = (rep[col] * 100).round(0)
        rep["age"] = rep["age"].round(0)
        logger.info("K=%d representative subtypes:\n%s", k,
                    rep[["n", "unfav%", "age", "hiv", "smear", "ssa", "sens", "gotR"]].to_string())

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "balanced_archetype_sweep.csv", index=False)
    logger.info("Balanced sweep summary:\n%s", summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
