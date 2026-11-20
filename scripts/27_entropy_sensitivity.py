"""Entropy-penalty (λ) sensitivity: is the 5th subtype data-driven or manufactured by λ?

Motivation: the max-entropy load-balancing penalty (λ=1.0) is what keeps all five
archetypes populated ("collapses to four without it"), and held-out AUROC is flat in K, so AUROC
cannot adjudicate whether archetype 5 is real. This script trains the canonical K=5 multimodal model
across λ ∈ {0, 0.3, 1.0, 3.0} and reports, per λ: how many subtypes stay populated (≥3%), the
minimum subtype fraction, and each subtype's defining profile (unfavorable rate + resistance/HIV/
smear composition). If the SAME five clinical profiles recur across λ (only their balance changes),
the five-subtype structure is data-driven and λ controls balance, not existence. If profiles appear/
disappear with λ, the fifth subtype is a penalty artifact. Writes an aggregate table only.

Usage
-----
    uv run python scripts/27_entropy_sensitivity.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

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
LAMBDAS = [0.0, 0.3, 1.0, 3.0]
PROFILE_COLS = ["unfav%", "sens%", "MDR%", "XDR%", "HIV%", "smear+%"]


def _eqfrac(series: pd.Series, value: object) -> float:
    """NA-safe percentage of ``series`` equal to ``value`` (nullable dtypes -> False)."""
    return 100.0 * (series == value).fillna(False).to_numpy(dtype=float).mean()


def _profile(craw: pd.DataFrame, y: np.ndarray, sub: np.ndarray, s: int) -> dict[str, float]:
    """Defining profile of subtype ``s`` (percentages)."""
    m = sub == s
    return {
        "unfav%": 100.0 * float(y[m].mean()),
        "sens%": _eqfrac(craw["type_of_resistance"][m], "Sensitive"),
        "MDR%": _eqfrac(craw["type_of_resistance"][m], "MDR non XDR"),
        "XDR%": _eqfrac(craw["type_of_resistance"][m], "XDR"),
        "HIV%": _eqfrac(craw["comorb_hiv"][m], 1),
        "smear+%": _eqfrac(craw["baseline_smear"][m], "positive"),
    }


def main() -> int:
    """Train K=5 across a grid of entropy weights and tabulate subtype profiles."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    cond_ids = list(data["cond_ids"])

    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, _ = build_context_matrix(pc, bx, cond_ids)
    craw = craw.reindex(pd.Index(cond_ids, name="condition_id")).reset_index(drop=True)
    y = data["y"].astype(np.float64)

    # Reference profiles at the canonical λ=1.0, used to match subtypes across λ by clinical profile.
    ref_profiles: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for lam in LAMBDAS:
        cfg = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                          entropy_weight=lam)
        m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True,
                                          seed=SEED)
        train(m, c, x, y, cfg, seed=SEED)
        sub = m.predict_loadings(c).argmax(axis=1)
        auc = roc_auc_score(y, m.predict_proba(c, x))
        fracs = np.bincount(sub, minlength=K) / len(sub)
        n_pop = int((fracs >= 0.03).sum())

        prof = np.array([[_profile(craw, y, sub, s)[col] for col in PROFILE_COLS]
                         for s in range(K)], dtype=float)
        if lam == 1.0:
            ref_profiles = prof
        # Match this λ's subtypes to the reference (λ=1) by nearest standardized profile (Hungarian).
        if ref_profiles is not None:
            denom = ref_profiles.std(0)
            denom[denom == 0] = 1.0
            cost = np.linalg.norm(
                (prof[:, None, :] - ref_profiles[None, :, :]) / denom, axis=2)
            r, col = linear_sum_assignment(cost)
            match = {int(rr): int(cc) for rr, cc in zip(r, col, strict=True)}
        else:
            match = {s: s for s in range(K)}

        for s in range(K):
            p = _profile(craw, y, sub, s)
            rows.append({"lambda": lam, "subtype": f"S{s}",
                         "matches_canonical": f"S{match.get(s, s)}",
                         "n": int((sub == s).sum()), "frac%": round(100 * fracs[s], 1),
                         **{k2: round(v2, 1) for k2, v2 in p.items()}})
        summary.append({"lambda": lam, "n_populated_>=3%": n_pop,
                        "min_frac%": round(100 * fracs.min(), 1),
                        "in_sample_auroc": round(float(auc), 3)})
        logger.info("λ=%.1f | populated>=3%%: %d/5 | min frac %.1f%% | AUROC %.3f",
                    lam, n_pop, 100 * fracs.min(), auc)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "entropy_sensitivity_profiles.csv", index=False)
    pd.DataFrame(summary).to_csv(TABLES_DIR / "entropy_sensitivity_summary.csv", index=False)
    logger.info("Summary:\n%s", pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
