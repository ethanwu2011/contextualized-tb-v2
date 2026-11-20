"""Head-to-head: do PATCH and the black-box baseline transfer differently to Brazil?

On the harmonized shared-feature space, train four models on TB Portals and evaluate (a) in-sample
out-of-fold on TB Portals and (b) transferred to the 298k-patient Brazilian registry. Reports AUROC,
AUPRC, and Brier for each, plus the in-sample->Brazil generalization gap. The question: does the
black box's tiny in-sample edge survive the distribution shift, and is PATCH more robust?

Brazil is public (SINAN); TB Portals features are read locally; only aggregate metrics are written.

Run: ``uv run python scripts/36_brazil_model_comparison.py``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# reuse the exact harmonized loaders from the external-validation script
sys.path.insert(0, "scripts")
ext = __import__("32_external_validation")
from ctb.config import TABLES_DIR  # noqa: E402
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, train  # noqa: E402

SEED, CFG = ext.SEED, ext.CFG


def _oof_lr(F, y):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(F, y):
        m = LogisticRegression(max_iter=2000).fit(F[tr], y[tr]); p[te] = m.predict_proba(F[te])[:, 1]
    return p


def main() -> int:
    c_tb, x_tb, y_tb, amean, asd = ext._tbportals()
    c_bz, x_bz, y_bz = ext._brazil(amean, asd)
    xc_tb, xc_bz = np.hstack([x_tb, c_tb]), np.hstack([x_bz, c_bz])
    print(f"TB Portals harmonized n={len(y_tb)} ({100*y_tb.mean():.1f}% died/fail) | "
          f"Brazil n={len(y_bz)} ({100*y_bz.mean():.1f}%)\n")

    def metrics(y, p):
        return roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)

    rows = []
    # --- three logistic baselines (regimen / context / concatenated) ---
    for name, Ftb, Fbz in [("Regimen only", x_tb, x_bz), ("Context only", c_tb, c_bz),
                           ("Concatenated (black box)", xc_tb, xc_bz)]:
        ins = roc_auc_score(y_tb, _oof_lr(Ftb, y_tb))                 # in-sample OOF AUROC
        lr = LogisticRegression(max_iter=2000).fit(Ftb, y_tb)        # full fit -> transfer
        a, ap, br = metrics(y_bz, lr.predict_proba(Fbz)[:, 1])
        rows.append({"model": name, "insample_auroc": round(ins, 3), "brazil_auroc": round(a, 3),
                     "brazil_auprc": round(ap, 3), "brazil_brier": round(br, 3),
                     "gap": round(ins - a, 3)})

    # --- PATCH (archetype contextualized) ---
    def fit_patch(c, x, y):
        m = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=ext.K, simplex=True,
                                          seed=SEED)
        train(m, c, x, y, CFG, seed=SEED); return m
    # in-sample OOF for PATCH (5 retrains)
    poof = np.zeros(len(y_tb))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(c_tb, y_tb):
        mp = fit_patch(c_tb[tr], x_tb[tr], y_tb[tr]); poof[te] = mp.predict_proba(c_tb[te], x_tb[te])
    ins = roc_auc_score(y_tb, poof)
    mfull = fit_patch(c_tb, x_tb, y_tb)
    a, ap, br = metrics(y_bz, mfull.predict_proba(c_bz, x_bz))
    rows.append({"model": "PATCH", "insample_auroc": round(ins, 3), "brazil_auroc": round(a, 3),
                 "brazil_auprc": round(ap, 3), "brazil_brier": round(br, 3), "gap": round(ins - a, 3)})

    # within-Brazil ceiling (5-fold context-LR)
    ceil = roc_auc_score(y_bz, _oof_lr(c_bz, y_bz))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "brazil_model_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwithin-Brazil context-LR ceiling (5-fold) = {ceil:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
