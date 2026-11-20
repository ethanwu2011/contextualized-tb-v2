"""External validation of PATCH on Brazil (SINAN) via a harmonized shared-feature model.

TB Portals and SINAN share only a subset of context (resistance status, HIV, diabetes, smear, age,
sex, case type) and a died/failure outcome. We therefore fit a *harmonized* PATCH on the shared
feature space in TB Portals with a harmonized first-/second-line regimen X, then transfer it to the
298k-patient Brazilian cohort to test three things:

1. **Risk transfer** — held-out AUROC for died/failure in Brazil (PATCH and a context-LR baseline),
   vs the within-Brazil cross-validated ceiling.
2. **Prognostic-structure replication** — assign every Brazilian patient to a subtype via the
   TB-Portals-fit context->archetype map Z(C); do the subtypes preserve their risk ordering?
3. **Subtype recurrence** — fit the harmonized PATCH natively on Brazil; do the same response
   profiles (and the responder/failure asymmetry) re-emerge?

Brazil is ~99.5% RIF-sensitive, so this validates the drug-sensitive + multimodal-failure regime of
PATCH (HIV/burden/comorbidity driving failure), not the resistant-responder subtypes. Aggregate
results only.

Usage
-----
    uv run python scripts/32_external_validation.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
SEED = 0
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
# Shared context features in both registries and the harmonized six-drug regimen.
CTX = ["rif_resistant", "hiv", "diabetes", "smear_pos", "age_z", "male", "new_case"]
XCOLS = ["R", "H", "Z", "E", "S", "second_line"]


def _standardize(age: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return (np.nan_to_num(age, nan=mean) - mean) / (sd if sd else 1.0)


def _tbportals() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Harmonized TB Portals: shared context C, regimen X, died/failure y."""
    data = load_primary_matrix()
    drugs = list(data["drugs"])
    xfull = data["X"].astype(np.float64)
    cond = list(data["cond_ids"])
    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, _ = build_context_matrix(pc, bx, cond)
    craw = craw.reindex(pd.Index(cond, name="condition_id")).reset_index(drop=True)
    y = data["y"].astype(np.float64).reshape(-1)

    res = craw["type_of_resistance"].astype("string")
    age = pd.to_numeric(craw["age_of_onset"], errors="coerce").to_numpy()
    amean, asd = float(np.nanmean(age)), float(np.nanstd(age))
    c = pd.DataFrame(
        {
            "rif_resistant": res.isin(["MDR non XDR", "Pre-XDR", "XDR"]).astype(float).to_numpy(),
            "hiv": (craw["comorb_hiv"] == 1).astype(float).to_numpy(),
            "diabetes": (craw["comorb_diabetes"] == 1).astype(float).to_numpy(),
            "smear_pos": (craw["baseline_smear"] == "positive").astype(float).to_numpy(),
            "age_z": _standardize(age, amean, asd),
            "male": (craw["gender"] == "Male").astype(float).to_numpy(),
            "new_case": craw.get("case_definition", pd.Series(index=craw.index, dtype="string"))
            .astype("string")
            .str.contains("New", case=False, na=False)
            .astype(float)
            .to_numpy(),
        }
    )[CTX].to_numpy()

    def col(d: str) -> np.ndarray:
        return xfull[:, drugs.index(f"drug_{d}")] if f"drug_{d}" in drugs else np.zeros(len(cond))

    second = ((xfull.sum(1) - sum(col(d) for d in ["R", "H", "Z", "E", "S"])) > 0).astype(float)
    x = np.column_stack([col("R"), col("H"), col("Z"), col("E"), col("S"), second])
    return c, x, y, amean, asd


def _brazil(amean: float, asd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonized Brazil: shared context C (same cols), standard regimen X, died/failure y."""
    path = PROCESSED_DIR / "brazil_harmonized.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/31_build_brazil_cohort.py with the public "
            "SINAN files first."
        )
    bz = pd.read_parquet(path)
    age = bz["age"].to_numpy()
    c = pd.DataFrame(
        {
            "rif_resistant": bz["rif_resistant"].fillna(0).to_numpy(),
            "hiv": bz["hiv"].fillna(bz["hiv"].mean()).to_numpy(),
            "diabetes": bz["diabetes"].fillna(0).to_numpy(),
            "smear_pos": bz["smear_pos"].fillna(bz["smear_pos"].mean()).to_numpy(),
            "age_z": _standardize(age, amean, asd),
            "male": bz["male"].fillna(bz["male"].mean()).to_numpy(),
            "new_case": bz["new_case"].fillna(1).to_numpy(),
        }
    )[CTX].to_numpy()
    y = bz["y"].to_numpy()
    # Standard first-line regimen (RHZE) for all — Brazil is ~99.5% drug-sensitive.
    x = np.tile(np.array([1, 1, 1, 1, 0, 0], float), (len(bz), 1))
    return c, x, y


def _profile(c: np.ndarray, y: np.ndarray, sub: np.ndarray) -> pd.DataFrame:
    """Per-subtype size, died/failure rate, and mean of each shared feature."""
    rows = []
    for k in range(K):
        m = sub == k
        if m.sum() == 0:
            continue
        row = {"subtype": k, "n": int(m.sum()), "unfav%": round(100 * y[m].mean(), 1)}
        for j, name in enumerate(CTX):
            row[name] = round(float(c[m, j].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    """Run the harmonized cross-registry transfer validation."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    c_tb, x_tb, y_tb, amean, asd = _tbportals()
    c_bz, x_bz, y_bz = _brazil(amean, asd)
    logger.info(
        "TB Portals harmonized: n=%d, died/fail %.1f%% | Brazil: n=%d, died/fail %.1f%%",
        len(y_tb),
        100 * y_tb.mean(),
        len(y_bz),
        100 * y_bz.mean(),
    )

    # --- fit harmonized PATCH on TB Portals ---
    m = LinearArchetypeContextualized(
        c_tb.shape[1], x_tb.shape[1], n_archetypes=K, simplex=True, seed=SEED
    )
    train(m, c_tb, x_tb, y_tb, CFG, seed=SEED)

    # (1) risk transfer: TB Portals model -> Brazil
    auc_tb_in = roc_auc_score(y_tb, m.predict_proba(c_tb, x_tb))
    auc_transfer = roc_auc_score(y_bz, m.predict_proba(c_bz, x_bz))
    lr = LogisticRegression(max_iter=2000).fit(np.hstack([x_tb, c_tb]), y_tb)
    auc_lr_transfer = roc_auc_score(y_bz, lr.predict_proba(np.hstack([x_bz, c_bz]))[:, 1])
    # within-Brazil ceiling (5-fold CV, context-LR)
    oof = np.zeros(len(y_bz))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(c_bz, y_bz):
        lrb = LogisticRegression(max_iter=2000).fit(np.hstack([x_bz[tr], c_bz[tr]]), y_bz[tr])
        oof[te] = lrb.predict_proba(np.hstack([x_bz[te], c_bz[te]]))[:, 1]
    auc_bz_ceiling = roc_auc_score(y_bz, oof)
    logger.info(
        "RISK TRANSFER | TBP in-sample %.3f | -> Brazil PATCH %.3f | -> Brazil LR %.3f | "
        "Brazil CV ceiling %.3f",
        auc_tb_in,
        auc_transfer,
        auc_lr_transfer,
        auc_bz_ceiling,
    )

    # (2) prognostic-structure replication: apply TBP Z(C) to Brazil
    sub_tb = m.predict_loadings(c_tb).argmax(1)
    sub_bz = m.predict_loadings(c_bz).argmax(1)
    p_tb = _profile(c_tb, y_tb, sub_tb).assign(cohort="TB_Portals")
    p_bz = _profile(c_bz, y_bz, sub_bz).assign(cohort="Brazil_applied")

    # (3) native recurrence: fit harmonized PATCH on Brazil itself
    mb = LinearArchetypeContextualized(
        c_bz.shape[1], x_bz.shape[1], n_archetypes=K, simplex=True, seed=SEED
    )
    train(mb, c_bz, x_bz, y_bz, CFG, seed=SEED)
    sub_bz_native = mb.predict_loadings(c_bz).argmax(1)
    p_bz_native = _profile(c_bz, y_bz, sub_bz_native).assign(cohort="Brazil_native")

    prof = pd.concat([p_tb, p_bz, p_bz_native], ignore_index=True)
    prof.to_csv(TABLES_DIR / "brazil_subtype_profiles.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_tbportals": len(y_tb),
                "n_brazil": len(y_bz),
                "unfav_tbportals": round(100 * y_tb.mean(), 1),
                "unfav_brazil": round(100 * y_bz.mean(), 1),
                "auc_tbp_insample": round(auc_tb_in, 3),
                "auc_transfer_patch": round(auc_transfer, 3),
                "auc_transfer_lr": round(auc_lr_transfer, 3),
                "auc_brazil_ceiling": round(auc_bz_ceiling, 3),
                "subtype_risk_spread_tbp": round(p_tb["unfav%"].max() - p_tb["unfav%"].min(), 1),
                "subtype_risk_spread_brazil": round(p_bz["unfav%"].max() - p_bz["unfav%"].min(), 1),
            }
        ]
    ).to_csv(TABLES_DIR / "brazil_external_validation.csv", index=False)
    logger.info("TBP subtype risk:\n%s", p_tb[["subtype", "n", "unfav%"]].to_string(index=False))
    logger.info(
        "Brazil (applied) subtype risk:\n%s",
        p_bz[["subtype", "n", "unfav%"]].to_string(index=False),
    )
    logger.info(
        "Brazil (native) subtype risk:\n%s",
        p_bz_native[["subtype", "n", "unfav%"]].to_string(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
