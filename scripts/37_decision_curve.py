"""Decision-curve (net-benefit) analysis: prognostic risk-targeting, PATCH vs resistance class.

Operationalizes PATCH's prognostic use by evaluating intensified monitoring or adjunctive
support to high-risk patients — as a decision-analytic deliverable. Writes two aggregate tables that
``figures.fig06_clinical_utility`` renders into main Figure 6:

* ``decision_curve.csv`` — net benefit (Vickers & Elkin) vs decision threshold for intervene-on-all,
  intervene-on-none, resistance-class-only, and the full multimodal PATCH risk.
* ``decision_curve_yield.csv`` — at a representative monitoring threshold, the per-1,000-patient
  yield (true high-risk flagged vs unnecessary intensification) for each strategy.

DUA-safe: trains locally; writes only aggregate values.

Run: ``uv run python scripts/37_decision_curve.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ctb.config import TABLES_DIR
from ctb.data.artifacts import load_primary_matrix, load_primary_raw_context
from ctb.data.context import apply_context_encoder, fit_context_encoder
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train

SEED = 0
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
T_OPERATING = 0.20  # representative monitoring threshold for the per-1,000 yield panel


def main() -> int:
    data = load_primary_matrix()
    X = data["X"].astype(float)
    cond = list(data["cond_ids"])
    raw_context, spec = load_primary_raw_context(cond)
    y = data["y"].astype(int)

    res = raw_context["type_of_resistance"].astype("string").fillna("Unknown")
    res_oh = pd.get_dummies(res).to_numpy(float)  # resistance-class one-hot = current standard

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def oof_lr(F: np.ndarray) -> np.ndarray:
        p = np.zeros(len(y))
        for tr, te in cv.split(F, y):
            sc = StandardScaler().fit(F[tr])
            m = LogisticRegression(max_iter=3000).fit(sc.transform(F[tr]), y[tr])
            p[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
        return p

    def oof_patch() -> np.ndarray:
        p = np.zeros(len(y))
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            encoder = fit_context_encoder(raw_context.iloc[tr], spec)
            c_tr = apply_context_encoder(raw_context.iloc[tr], spec, encoder).to_numpy(float)
            c_te = apply_context_encoder(raw_context.iloc[te], spec, encoder).to_numpy(float)
            fold_seed = SEED + fold
            m = LinearArchetypeContextualized(
                c_tr.shape[1], X.shape[1], n_archetypes=5, simplex=True, seed=fold_seed
            )
            train(m, c_tr, X[tr], y[tr], CFG, seed=fold_seed)
            p[te] = m.predict_proba(c_te, X[te])
        return p

    p_res, p_patch = oof_lr(res_oh), oof_patch()
    auc_res, auc_patch = roc_auc_score(y, p_res), roc_auc_score(y, p_patch)
    print(f"AUROC | resistance-class only {auc_res:.3f} | PATCH {auc_patch:.3f}")

    n, prev = len(y), float(y.mean())
    ths = np.arange(0.02, 0.51, 0.01)

    def nb(p: np.ndarray, t: float) -> float:
        pos = p >= t
        tp, fp = int(np.sum(pos & (y == 1))), int(np.sum(pos & (y == 0)))
        return tp / n - fp / n * (t / (1 - t))

    nb_all = np.array([prev - (1 - prev) * (t / (1 - t)) for t in ths])
    pd.DataFrame({"threshold": ths, "treat_all": nb_all, "treat_none": np.zeros_like(ths),
                  "resistance_class_only": [nb(p_res, t) for t in ths],
                  "PATCH_multimodal": [nb(p_patch, t) for t in ths],
                  "auroc_resistance": auc_res, "auroc_patch": auc_patch}).to_csv(
        TABLES_DIR / "decision_curve.csv", index=False)

    # per-1,000 yield at the operating threshold
    def yield_row(name: str, p: np.ndarray | None) -> dict:
        flagged = np.ones(n, bool) if p is None else p >= T_OPERATING
        tp, fp = int(np.sum(flagged & (y == 1))), int(np.sum(flagged & (y == 0)))
        return {"strategy": name, "threshold": T_OPERATING,
                "flagged_per_1000": round(1000 * flagged.mean(), 0),
                "true_highrisk_per_1000": round(1000 * tp / n, 0),
                "overmonitored_per_1000": round(1000 * fp / n, 0)}

    yld = pd.DataFrame([yield_row("Monitor all", None), yield_row("Resistance class", p_res),
                        yield_row("PATCH (multimodal)", p_patch)])
    yld.to_csv(TABLES_DIR / "decision_curve_yield.csv", index=False)
    print(yld.to_string(index=False))

    band = (ths >= 0.10) & (ths <= 0.30)
    gain = float(np.mean(np.array([nb(p_patch, t) for t in ths])[band]
                         - np.array([nb(p_res, t) for t in ths])[band]))
    print(f"mean net-benefit gain PATCH vs resistance-class over 10-30%: {gain:+.4f}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
