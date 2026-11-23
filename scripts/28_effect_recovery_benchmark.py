"""Semi-synthetic effect-recovery benchmark.

PATCH's thesis is per-patient treatment-EFFECT estimation, yet every quantitative
comparison in the paper is an outcome-AUROC comparison. This script benchmarks effect recovery
directly, on semi-synthetic data built from the real context matrix C (so the covariate geometry is
realistic) with KNOWN ground-truth heterogeneous effects and no unmeasured confounding.

Part 1 --- archetype-factorization ablation (multi-drug).
    Ground-truth per-patient effect vector beta*(C) in R^26 has low-rank (archetype) structure.
    Outcomes are simulated from the real regimens X. We fit PATCH (K=5, simplex archetypes) and an
    UNRESTRICTED contextualized regression (dense linear varying coefficients: simplex=False, K=26)
    and measure beta-recovery RMSE on held-out patients across training-set sizes. This isolates
    whether the archetype factorization improves EFFECT recovery (not just discrimination), and
    whether the advantage grows as data get sparse.

Part 2 --- HTE baselines (single treatment).
    One binary treatment with a confounded propensity e(C) and a heterogeneous effect tau*(C).
    We compare PATCH against S-, T-, and X-learners and an RF "causal-forest proxy" (econml/grf are
    not installed) on CATE recovery (PEHE), under a low-rank and a nonlinear ground truth.

DUA-safe: reads the fitted context/regimen matrices; simulates its own outcomes; writes only
aggregate error metrics + a figure.

Usage
-----
    .venv/bin/python scripts/28_effect_recovery_benchmark.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ctb.config import FIGURES_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:  # noqa: BLE001
    HAVE_LGBM = False

CFG = TrainConfig(epochs=1500, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)
N_REPS = 3
SIG = lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))  # noqa: E731


def _dense_contextualized_beta(c_tr, x_tr, y_tr, c_te, seed, epochs=1500, lr=0.01, wd=1e-3):
    """Unrestricted contextualized logistic (no archetype factorization): beta(c)=Wc+b, mu(c)=v.c+a.

    This is the ablation baseline -- dense linear varying coefficients, the natural 'contextualized
    regression without the low-rank archetype bottleneck.' Returns beta_hat on the held-out patients.
    """
    import torch
    torch.manual_seed(seed)
    ct = torch.tensor(c_tr, dtype=torch.float32)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    p, d = c_tr.shape[1], x_tr.shape[1]
    W = torch.zeros(d, p, requires_grad=True)
    b0 = torch.zeros(d, requires_grad=True)
    v = torch.zeros(p, requires_grad=True)
    a = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b0, v, a], lr=lr, weight_decay=wd)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        opt.zero_grad()
        beta = ct @ W.t() + b0
        mu = ct @ v + a
        loss = bce((xt * beta).sum(1) + mu, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (torch.tensor(c_te, dtype=torch.float32) @ W.t() + b0).numpy()


def _base_clf():
    if HAVE_LGBM:
        return LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.8, min_child_samples=40,
                              verbosity=-1, n_jobs=1)
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=20, n_jobs=1)


def _proba1(clf, X):
    p = clf.predict_proba(X)
    # column for class "1"
    j = list(clf.classes_).index(1) if 1 in getattr(clf, "classes_", [1]) else -1
    return p[:, j]


# --------------------------------------------------------------------------- Part 1
def part1(c: NDArray[np.float64], x: NDArray[np.float64]) -> pd.DataFrame:
    n, p = c.shape
    d = x.shape[1]
    K_STAR = 5
    rows = []
    for rep in range(N_REPS):
        rng = np.random.default_rng(1000 + rep)
        # ground-truth low-rank effect surface
        Mstar = rng.normal(0, 1.0, size=(K_STAR, p)) / np.sqrt(p)
        Astar = rng.normal(0, 0.7, size=(K_STAR, d))
        vstar = rng.normal(0, 1.0, size=p) / np.sqrt(p)
        z = c @ Mstar.T
        z = np.exp(z - z.max(1, keepdims=True))
        z = z / z.sum(1, keepdims=True)                      # softmax loadings
        beta_true = z @ Astar                                # (n, d) ground-truth effects
        mu_true = c @ vstar
        logit = (x * beta_true).sum(1) + mu_true
        y = (rng.uniform(size=n) < SIG(logit)).astype(np.float64)

        # fixed 70/30 split
        idx = rng.permutation(n)
        n_test = int(0.3 * n)
        te, tr_all = idx[:n_test], idx[n_test:]
        for n_train in (1500, 5000, len(tr_all)):
            tr = tr_all[:n_train]
            mp = LinearArchetypeContextualized(p, d, n_archetypes=5, simplex=True, seed=rep)
            train(mp, c[tr], x[tr], y[tr], CFG, seed=rep)
            bhat_patch, _ = mp.predict_beta(c[te])
            bhat_dense = _dense_contextualized_beta(c[tr], x[tr], y[tr], c[te], rep)
            for name, bhat in (("PATCH (K=5 archetype)", bhat_patch),
                               ("Unrestricted contextualized", bhat_dense)):
                rmse = float(np.sqrt(np.mean((bhat - beta_true[te]) ** 2)))
                rows.append({"part": 1, "regime": "low-rank", "n_train": int(n_train),
                             "method": name, "rep": rep, "metric": "beta_RMSE", "value": rmse})
        logger.info("part1 rep %d done", rep)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Part 2
def _patch_cate(c_tr, t_tr, y_tr, c_te, seed):
    m = LinearArchetypeContextualized(c_tr.shape[1], 1, n_archetypes=5, simplex=True, seed=seed)
    train(m, c_tr, t_tr.reshape(-1, 1).astype(np.float64), y_tr, CFG, seed=seed)
    beta, mu = m.predict_beta(c_te)
    return SIG(mu + beta[:, 0]) - SIG(mu)


def _slearner(c_tr, t_tr, y_tr, c_te):
    clf = _base_clf().fit(np.c_[c_tr, t_tr], y_tr)
    return _proba1(clf, np.c_[c_te, np.ones(len(c_te))]) - _proba1(clf, np.c_[c_te, np.zeros(len(c_te))])


def _tlearner(c_tr, t_tr, y_tr, c_te):
    m1 = _base_clf().fit(c_tr[t_tr == 1], y_tr[t_tr == 1])
    m0 = _base_clf().fit(c_tr[t_tr == 0], y_tr[t_tr == 0])
    return _proba1(m1, c_te) - _proba1(m0, c_te)


def _rf_tlearner(c_tr, t_tr, y_tr, c_te):
    m1 = RandomForestClassifier(n_estimators=300, min_samples_leaf=15, n_jobs=1).fit(c_tr[t_tr == 1], y_tr[t_tr == 1])
    m0 = RandomForestClassifier(n_estimators=300, min_samples_leaf=15, n_jobs=1).fit(c_tr[t_tr == 0], y_tr[t_tr == 0])
    return _proba1(m1, c_te) - _proba1(m0, c_te)


def _xlearner(c_tr, t_tr, y_tr, c_te):
    from sklearn.ensemble import GradientBoostingRegressor
    m1 = _base_clf().fit(c_tr[t_tr == 1], y_tr[t_tr == 1])
    m0 = _base_clf().fit(c_tr[t_tr == 0], y_tr[t_tr == 0])
    d1 = y_tr[t_tr == 1] - _proba1(m0, c_tr[t_tr == 1])
    d0 = _proba1(m1, c_tr[t_tr == 0]) - y_tr[t_tr == 0]
    tau1 = GradientBoostingRegressor(n_estimators=200, max_depth=3).fit(c_tr[t_tr == 1], d1)
    tau0 = GradientBoostingRegressor(n_estimators=200, max_depth=3).fit(c_tr[t_tr == 0], d0)
    e = LogisticRegression(max_iter=500, C=1.0).fit(c_tr, t_tr)
    ec = e.predict_proba(c_te)[:, list(e.classes_).index(1)]
    return ec * tau0.predict(c_te) + (1 - ec) * tau1.predict(c_te)


def part2(c: NDArray[np.float64]) -> pd.DataFrame:
    n, p = c.shape
    rows = []
    methods = {"PATCH (K=5)": None, "S-learner": _slearner, "T-learner": _tlearner,
               "X-learner": _xlearner, "RF T-learner (causal-forest proxy)": _rf_tlearner}
    for regime in ("low-rank", "nonlinear"):
        for rep in range(N_REPS):
            rng = np.random.default_rng(2000 + rep)
            we = rng.normal(0, 1, p) / np.sqrt(p)
            e = SIG(c @ we)
            t = (rng.uniform(size=n) < e).astype(np.float64)          # confounded treatment
            b = c @ (rng.normal(0, 1, p) / np.sqrt(p))                 # prognostic baseline logit
            if regime == "low-rank":
                Ms = rng.normal(0, 1, (5, p)) / np.sqrt(p)
                zz = c @ Ms.T
                zz = np.exp(zz - zz.max(1, keepdims=True)); zz /= zz.sum(1, keepdims=True)
                tau_lo = zz @ rng.normal(0, 1.2, 5)
            else:
                f = c @ (rng.normal(0, 1, (p, 3)) / np.sqrt(p))       # 3 latent factors
                tau_lo = 1.3 * np.tanh(f[:, 0]) * (f[:, 1] > 0) + 0.8 * np.sin(f[:, 2])
            y = (rng.uniform(size=n) < SIG(b + t * tau_lo)).astype(np.float64)
            tau_star = SIG(b + tau_lo) - SIG(b)                        # ground-truth prob-scale CATE

            idx = rng.permutation(n)
            te, tr = idx[: int(0.3 * n)], idx[int(0.3 * n):][:5000]
            for name, fn in methods.items():
                tau_hat = _patch_cate(c[tr], t[tr], y[tr], c[te], rep) if fn is None else fn(c[tr], t[tr], y[tr], c[te])
                pehe = float(np.sqrt(np.mean((tau_hat - tau_star[te]) ** 2)))
                rows.append({"part": 2, "regime": regime, "n_train": 5000, "method": name,
                             "rep": rep, "metric": "PEHE", "value": pehe})
            logger.info("part2 regime=%s rep=%d done", regime, rep)
    return pd.DataFrame(rows)


def main() -> int:
    configure_logging()
    set_global_seed()
    ensure_output_dirs()
    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    logger.info("LightGBM available: %s", HAVE_LGBM)

    r1 = part1(c, x)
    r2 = part2(c)
    res = pd.concat([r1, r2], ignore_index=True)
    res.to_csv(TABLES_DIR / "effect_recovery_benchmark_raw.csv", index=False)

    summ = (res.groupby(["part", "regime", "n_train", "method", "metric"])["value"]
            .agg(["mean", "std"]).reset_index())
    summ.to_csv(TABLES_DIR / "effect_recovery_benchmark.csv", index=False)

    logger.info("PART 1 (beta-recovery RMSE, lower is better):\n%s",
                summ[summ.part == 1].to_string(index=False))
    logger.info("PART 2 (CATE PEHE, lower is better):\n%s",
                summ[summ.part == 2].to_string(index=False))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))
        p1 = summ[summ.part == 1]
        for name in p1.method.unique():
            s = p1[p1.method == name].sort_values("n_train")
            # PATCH in PATCH_COLOR (figures/palette.py); the comparator in grey.
            col = "#0072B2" if str(name).startswith("PATCH") else "#8C8C8C"
            ax[0].plot(s.n_train, s["mean"], marker="o", color=col, label=name)
            ax[0].fill_between(s.n_train, s["mean"] - s["std"], s["mean"] + s["std"], color=col,
                               alpha=0.15)
        ax[0].set_xscale("log")
        ax[0].set_xticks(sorted(p1.n_train.unique()))
        ax[0].xaxis.set_major_formatter(ScalarFormatter())
        ax[0].xaxis.set_minor_formatter(NullFormatter())
        ax[0].set_xlabel("training patients"); ax[0].set_ylabel("$\\beta(C)$ recovery RMSE")
        ax[0].set_title("Part 1: factorization ablation (low-rank)")
        ax[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                     ncol=1)
        for k, regime in enumerate(("low-rank", "nonlinear")):
            s = summ[(summ.part == 2) & (summ.regime == regime)].sort_values("mean")
            ax[k + 1].barh(s.method, s["mean"], xerr=s["std"],
                           color=["#0072B2" if str(m).startswith("PATCH") else "#8C8C8C"
                                  for m in s.method])
            ax[k + 1].set_xlabel("PEHE"); ax[k + 1].set_title(f"Part 2: CATE recovery\n({regime})")
            ax[k + 1].invert_yaxis()
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"figS13_effect_recovery.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure skipped: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
