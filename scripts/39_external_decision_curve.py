"""Decision-curve analysis against a realistic comparator, internally AND in Brazil.

The main-text decision curve (script 37, Fig 6) compares PATCH's multimodal risk against
resistance-class-only stratification. It has two limits: (a) resistance-class-only is a
straw-man comparator no clinician uses alone, and (b) the analysis is internal-only and rests
on internal calibration. This script addresses both:

* Realistic comparator: a plain MULTIVARIABLE clinical logistic model on the seven harmonized
  features shared by both registries (rif-resistance, HIV, diabetes, smear, age, sex, case type) ---
  the kind of simple risk score a programme would actually use --- alongside resistance-class-only
  and PATCH.
* External decision curve: every model is trained on TB Portals and the decision curve is recomputed
  in the 298,900-patient Brazilian (SINAN) cohort, with calibration slope and intercept reported in
  both cohorts.

Internally we use the full 335-feature PATCH out of fold. Externally we use the harmonized seven-
feature PATCH that transfers. Aggregate outputs only are written.

Usage
-----
    .venv/bin/python scripts/39_external_decision_curve.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
ext = __import__("32_external_validation")
from ctb.config import TABLES_DIR, ensure_output_dirs  # noqa: E402
from ctb.data.artifacts import load_primary_matrix  # noqa: E402
from ctb.data.clean import clean_patient_cases  # noqa: E402
from ctb.data.context import (  # noqa: E402
    apply_context_encoder,
    build_context_matrix,
    fit_context_encoder,
)
from ctb.data.load import load_biochemistry, load_patient_cases  # noqa: E402
from ctb.models.archetype_contextualized import (  # noqa: E402
    LinearArchetypeContextualized,
    TrainConfig,
    train,
)
from ctb.utils.logging import configure_logging  # noqa: E402
from ctb.utils.reproducibility import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)
SEED = 0
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
THS = np.arange(0.02, 0.51, 0.01)
BAND = (THS >= 0.10) & (THS <= 0.30)
T_OP = 0.20


def _nb(y, p, t):
    n = len(y)
    pos = p >= t
    tp, fp = int(np.sum(pos & (y == 1))), int(np.sum(pos & (y == 0)))
    return tp / n - fp / n * (t / (1 - t))


def _curve(y, scores: dict) -> pd.DataFrame:
    prev = float(y.mean())
    out = {
        "threshold": THS,
        "treat_all": [prev - (1 - prev) * (t / (1 - t)) for t in THS],
        "treat_none": np.zeros_like(THS),
    }
    for name, p in scores.items():
        out[name] = [_nb(y, p, t) for t in THS]
    return pd.DataFrame(out)


def _cal_slope(y, p):
    """Calibration slope (coef of logit(p)) and intercept; perfect = slope 1, intercept 0."""
    lo = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(lo / (1 - lo)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def _yield(y, p, t=T_OP):
    n = len(y)
    pos = p >= t
    return {
        "flagged_per_1000": round(1000 * pos.mean()),
        "true_highrisk_per_1000": round(1000 * int(np.sum(pos & (y == 1))) / n),
        "overmonitored_per_1000": round(1000 * int(np.sum(pos & (y == 0))) / n),
    }


def main() -> int:
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    # ---------- full 335-feature internal cohort (for the internal curve) ----------
    data = load_primary_matrix()
    X = data["X"].astype(float)
    cond = list(data["cond_ids"])
    y = data["y"].astype(int).reshape(-1)
    craw, spec = build_context_matrix(
        clean_patient_cases(load_patient_cases()), load_biochemistry(), cond
    )
    craw = craw.reindex(pd.Index(cond, name="condition_id")).reset_index(drop=True)
    res_oh = pd.get_dummies(craw["type_of_resistance"].astype("string").fillna("Unknown")).to_numpy(
        float
    )

    # harmonized matrices (shared 7 features) for the realistic comparator + external transfer
    c_tb, x_tb, y_tb, amean, asd = ext._tbportals()
    c_bz, x_bz, y_bz = ext._brazil(amean, asd)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def oof_lr(F, yy):
        p = np.zeros(len(yy))
        for tr, te in cv.split(F, yy):
            sc = StandardScaler().fit(F[tr])
            p[te] = (
                LogisticRegression(max_iter=3000)
                .fit(sc.transform(F[tr]), yy[tr])
                .predict_proba(sc.transform(F[te]))[:, 1]
            )
        return p

    def oof_patch_full():
        p = np.zeros(len(y))
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            encoder = fit_context_encoder(craw.iloc[tr], spec)
            c_tr = apply_context_encoder(craw.iloc[tr], spec, encoder).to_numpy(float)
            c_te = apply_context_encoder(craw.iloc[te], spec, encoder).to_numpy(float)
            m = LinearArchetypeContextualized(
                c_tr.shape[1], X.shape[1], n_archetypes=5, simplex=True, seed=SEED + fold
            )
            train(m, c_tr, X[tr], y[tr], CFG, seed=SEED + fold)
            p[te] = m.predict_proba(c_te, X[te])
        return p

    # ---------- INTERNAL curves (TB Portals, out-of-fold) ----------
    p_res_i = oof_lr(res_oh, y)  # resistance-class only (straw man)
    p_mv_i = oof_lr(c_tb, y_tb)  # realistic multivariable clinical (7 feat)
    p_patch_i = oof_patch_full()  # full 335-feature PATCH
    internal = {
        "resistance_class_only": p_res_i,
        "multivariable_clinical": p_mv_i,
        "PATCH_full": p_patch_i,
    }
    _curve(y, internal).to_csv(TABLES_DIR / "decision_curve_realistic.csv", index=False)

    # ---------- EXTERNAL curves (train on TB Portals harmonized -> Brazil) ----------
    res_col = ext.CTX.index("rif_resistant")
    lr_res = LogisticRegression(max_iter=3000).fit(c_tb[:, [res_col]], y_tb)
    lr_mv = LogisticRegression(max_iter=3000).fit(StandardScaler().fit_transform(c_tb), y_tb)
    sc_mv = StandardScaler().fit(c_tb)
    mp = LinearArchetypeContextualized(
        c_tb.shape[1], x_tb.shape[1], n_archetypes=5, simplex=True, seed=SEED
    )
    train(mp, c_tb, x_tb, y_tb, CFG, seed=SEED)

    p_res_e = lr_res.predict_proba(c_bz[:, [res_col]])[:, 1]
    p_mv_e = lr_mv.predict_proba(sc_mv.transform(c_bz))[:, 1]
    p_patch_e = mp.predict_proba(c_bz, x_bz)
    external = {
        "resistance_class_only": p_res_e,
        "multivariable_clinical": p_mv_e,
        "PATCH_harmonized": p_patch_e,
    }
    _curve(y_bz, external).to_csv(TABLES_DIR / "decision_curve_brazil.csv", index=False)

    # ---------- summary: net-benefit gains, AUROC, calibration, yield ----------
    rows = []
    for cohort, yy, scores, ref in [
        ("internal", y, internal, "resistance_class_only"),
        ("brazil", y_bz, external, "resistance_class_only"),
    ]:
        patch_key = "PATCH_full" if cohort == "internal" else "PATCH_harmonized"
        nb_patch = np.array([_nb(yy, scores[patch_key], t) for t in THS])
        nb_mv = np.array([_nb(yy, scores["multivariable_clinical"], t) for t in THS])
        nb_res = np.array([_nb(yy, scores[ref], t) for t in THS])
        for name, p in scores.items():
            sl, ic = _cal_slope(yy, p)
            rows.append(
                {
                    "cohort": cohort,
                    "model": name,
                    "auroc": round(roc_auc_score(yy, p), 3),
                    "cal_slope": round(sl, 3),
                    "cal_intercept": round(ic, 3),
                    **{f"yield_{k}": v for k, v in _yield(yy, p).items()},
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "model": "__gains__",
                "nb_gain_patch_vs_resistance": round(float(np.mean((nb_patch - nb_res)[BAND])), 4),
                "nb_gain_patch_vs_multivariable": round(
                    float(np.mean((nb_patch - nb_mv)[BAND])), 4
                ),
                "nb_gain_multivariable_vs_resistance": round(
                    float(np.mean((nb_mv - nb_res)[BAND])), 4
                ),
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(TABLES_DIR / "decision_curve_realistic_summary.csv", index=False)
    logger.info("SUMMARY:\n%s", summ.to_string(index=False))

    # ---------- figure ----------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dc_i = pd.read_csv(TABLES_DIR / "decision_curve_realistic.csv")
        dc_e = pd.read_csv(TABLES_DIR / "decision_curve_brazil.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
        for a, dc, title, pk in [
            (ax[0], dc_i, "Internal (TB Portals)", "PATCH_full"),
            (ax[1], dc_e, "External (Brazil / SINAN)", "PATCH_harmonized"),
        ]:
            # Unified palette (figures/palette.py): PATCH ink slate, resistance class wine,
            # other comparator grey, reference strategies dark grey dotted/dashed.
            a.plot(dc.threshold, dc.treat_all, color="#666666", ls=":", label="treat all")
            a.plot(dc.threshold, dc.treat_none, color="#666666", ls="--", label="treat none")
            a.plot(
                dc.threshold,
                dc.resistance_class_only,
                color="#882255",
                label="resistance class only",
            )
            a.plot(
                dc.threshold,
                dc.multivariable_clinical,
                color="#8C8C8C",
                label="multivariable clinical",
            )
            a.plot(dc.threshold, dc[pk], color="#0072B2", lw=2, label="PATCH")
            a.axvspan(0.10, 0.30, color="#000", alpha=0.04)
            a.set_xlabel("risk threshold")
            a.set_ylabel("net benefit")
            a.set_title(title)
            a.set_xlim(0.02, 0.5)
            a.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        from ctb.config import FIGURES_DIR

        for e in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"figS14_external_decision_curve.{e}", dpi=200)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure skipped: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
