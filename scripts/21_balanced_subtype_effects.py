"""Bootstrap subtype × drug treatment effects for the balanced, non-resistance K=5 subtypes.

Fits the reproducible K=5 contextualized model on non-resistance context with the load-balancing
penalty (no dominant-blob), then estimates per-subtype drug effects with label-matched bootstrap
CIs in three views:

* **raw** ``beta`` — per-subtype mean treatment effect (negative = protective vs died/failure),
* **interaction** ``beta_subtype - beta_population`` — how a subtype differs from cohort average,
* **drug-specific** — the interaction with each subtype's across-drug mean removed, isolating
  "this subtype favors *this drug*" from the global responsiveness gradient.

The drug-specific view with CIs is a check of whether the subtypes carry drug-specific
signal beyond overall responsiveness.

Usage
-----
    uv run python scripts/21_balanced_subtype_effects.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.contextualized_figures import figure_subtype_interaction

logger = logging.getLogger(__name__)

K = 5
N_BOOT = 100
EPOCHS = 2000
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  balance_weight=3.0, balance_cap=0.40)
BASE_SEED = 0


def _no_resistance_c() -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], pd.DataFrame]:
    """Build the encoded context matrix with resistance/DST features removed."""
    data = load_primary_matrix()
    ids = list(data["cond_ids"])
    x = data["X"].astype(np.float64)
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
    meta = pd.DataFrame({
        "hiv": (craw["comorb_hiv"] == 1).to_numpy().astype(int),
        "smear_pos": (craw["baseline_smear"] == "positive").to_numpy().astype(int),
        "age": pd.to_numeric(craw["age_of_onset"], errors="coerce").to_numpy(),
    })
    return c, x, drugs, meta


def _match(ref: NDArray[np.int_], other: NDArray[np.int_], k: int) -> dict[int, int]:
    """Map ``other`` labels to ``ref`` by maximum patient overlap (Hungarian)."""
    overlap = np.zeros((k, k))
    for i in range(k):
        ri = ref == i
        for j in range(k):
            overlap[i, j] = int(np.logical_and(ri, other == j).sum())
    r_idx, o_idx = linear_sum_assignment(-overlap)
    return {int(o): int(r) for r, o in zip(r_idx, o_idx, strict=True)}


def _summary(arr: NDArray[np.float64], drugs: list[str], k: int, label: str) -> pd.DataFrame:
    """Mean, 95% CI, significance per (subtype, drug) from a (boot, k, drug) array."""
    est = np.nanmean(arr, axis=0)
    lo = np.nanpercentile(arr, 2.5, axis=0)
    hi = np.nanpercentile(arr, 97.5, axis=0)
    rows = []
    for s in range(k):
        for j, dn in enumerate(drugs):
            rows.append({"subtype": f"S{s}", "drug": dn, label: float(est[s, j]),
                         "ci_low": float(lo[s, j]), "ci_high": float(hi[s, j]),
                         "significant": bool(lo[s, j] > 0 or hi[s, j] < 0)})
    return pd.DataFrame(rows)


def main() -> int:
    """Run the balanced-subtype bootstrap treatment-effect analysis."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    c, x, drugs, meta = _no_resistance_c()
    npz = load_primary_matrix()
    y = npz["y"].astype(np.float64)

    ref = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=BASE_SEED
    )
    train(ref, c, x, y, CFG, seed=BASE_SEED)
    ref_sub = ref.predict_loadings(c).argmax(axis=1)
    for s in range(K):
        m = ref_sub == s
        logger.info("S%d n=%d unfav=%.0f%% HIV=%.0f%% smear+=%.0f%%", s, int(m.sum()),
                    100 * y[m].mean(), 100 * meta["hiv"][m].mean(),
                    100 * meta["smear_pos"][m].mean())

    nd = x.shape[1]
    eff = np.full((N_BOOT, K, nd), np.nan)
    inter = np.full((N_BOOT, K, nd), np.nan)
    spec_ = np.full((N_BOOT, K, nd), np.nan)
    rng = np.random.default_rng(BASE_SEED)
    n = c.shape[0]
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=BASE_SEED + b
        )
        train(m, c[idx], x[idx], y[idx], CFG, seed=BASE_SEED + b)
        beta, _ = m.predict_beta(c)
        beta_pop = beta.mean(axis=0)
        sub = m.predict_loadings(c).argmax(axis=1)
        mapping = _match(ref_sub, sub, K)
        for bl, rl in mapping.items():
            sel = sub == bl
            if sel.sum() == 0:
                continue
            mb = beta[sel].mean(axis=0)
            eff[b, rl, :] = mb
            inter[b, rl, :] = mb - beta_pop
            spec_[b, rl, :] = (mb - beta_pop) - (mb - beta_pop).mean()
        logger.info("bootstrap %d/%d", b + 1, N_BOOT)

    raw = _summary(eff, drugs, K, "beta")
    it = _summary(inter, drugs, K, "interaction")
    ds = _summary(spec_, drugs, K, "drug_specific")
    raw.to_csv(TABLES_DIR / "balanced_subtype_effects.csv", index=False)
    it.to_csv(TABLES_DIR / "balanced_subtype_interactions.csv", index=False)
    ds.to_csv(TABLES_DIR / "balanced_subtype_drugspecific.csv", index=False)

    common = [dn for dn in drugs if x[:, drugs.index(dn)].mean() >= 0.05]
    it_p = it[it["drug"].isin(common)].pivot(index="subtype", columns="drug", values="interaction")
    it_s = it[it["drug"].isin(common)].pivot(index="subtype", columns="drug", values="significant")
    figure_subtype_interaction(it_p, it_s, name="balanced_subtype_interactions",
                               title="Subtype × drug interaction vs cohort (balanced, no-DST)",
                               cbar_label="β subtype − β population")
    ds_p = ds[ds["drug"].isin(common)].pivot(
        index="subtype", columns="drug", values="drug_specific")
    ds_s = ds[ds["drug"].isin(common)].pivot(index="subtype", columns="drug", values="significant")
    figure_subtype_interaction(ds_p, ds_s, name="balanced_subtype_drugspecific",
                               title="Drug-specific subtype effects (gradient removed)",
                               cbar_label="drug-specific Δβ")

    n_ds = int(ds["significant"].sum())
    logger.info("Significant DRUG-SPECIFIC subtype effects: %d", n_ds)
    logger.info("Top drug-specific:\n%s",
                ds[ds.significant].reindex(ds[ds.significant]["drug_specific"].abs().sort_values()
                .index).tail(12)[["subtype", "drug", "drug_specific", "ci_low", "ci_high"]]
                .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
