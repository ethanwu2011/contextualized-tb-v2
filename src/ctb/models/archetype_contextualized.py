"""Linear-archetype contextualized regression, with archetypes inside one model.

This is the paper's primary model. It is a **linear contextualized regression** whose
context-conditioned regimen-association coefficient vector is factored through ``K`` shared
**archetypes**:

``beta(C) = (Z(C)) @ A``, with ``Z(C) = softmax(M . C + q)`` (linear loadings) and ``A`` the
archetypes.

Because ``Z`` is linear in ``C``, ``beta(C)`` is linear in ``C`` — so the model is an *interaction
model* (the interpretable, exact-inference instance of the contextualized framework), but with the
context-by-treatment interaction matrix ``W = M^T A`` constrained to **rank <= K**. The archetypes
and the per-patient subtype loadings therefore fall out of the *single* model, rather than from a
separate clustering step.

Interpretability is first-class — every quantity a contextualized analysis reports is a direct
read-off:

* **Population-average coefficients** ``beta`` — the mean recorded regimen association.
* **Context-conditioned coefficients** ``beta(C)`` — episode-level coefficients (the
  ``predict_beta`` map).
* **Archetypes** ``A`` — ``K`` interpretable coefficient profiles that summarize response
  phenotypes.
* **Subtype loadings** ``Z(C)`` — each patient's mixture over the archetypes (patient subtypes).
* **Context effects** ``mu(C)`` — direct effect of context on the outcome.

Special cases:

* ``K = 1`` collapses to a plain (full-rank-1) interaction model; for the confirmatory single-
  contrast analysis prefer the closed-form GLM/Cox (exact p-values).
* ``K > 1`` on a multi-treatment matrix yields the archetype subtypes (the Ellington-style figure).

Identifiability: the factorization ``Z @ A`` has a sign/scale/rotation ambiguity (``Z W^{-1}`` and
``W A`` give the same ``beta`` for invertible ``W``). :meth:`normalized_archetypes` returns a
canonical form (unit-norm archetype rows, sign fixed by the largest-magnitude entry) so figures are
reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from ctb.config import SEED

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class LinearArchetypeContextualized(nn.Module):
    """Linear contextualized logistic regression with an internal archetype factorization.

    Parameters
    ----------
    context_dim : int
        Number of context features (columns of ``C``).
    x_dim : int
        Number of treatment predictors (columns of ``X``).
    n_archetypes : int
        Number of archetypes ``K`` (``K = 1`` => plain rank-1 interaction model).
    simplex : bool, default False
        Use softmax (convex) loadings.
    seed : int or None, default None
        If given, seed the RNG **before** weight initialization so the starting point is
        deterministic. The archetype factorization has multiple local optima; an unseeded
        ``__init__`` draws from global RNG state, making which optimum is reached
        irreproducible. Pass a seed (matching the one used for :func:`train`) for reproducible fits.
    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        n_archetypes: int,
        simplex: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.n_archetypes = n_archetypes
        self.simplex = simplex
        # Linear context -> archetype loadings Z. The bias q is part of the declared
        # parameterization.
        self.loading_map = nn.Linear(context_dim, n_archetypes)
        # Archetype profiles: each row is [beta over x_dim treatments | intercept mu component].
        self.archetypes = nn.Parameter(torch.randn(n_archetypes, x_dim + 1) * 0.1)

    def _loadings(self, c: torch.Tensor) -> torch.Tensor:
        """Return archetype loadings ``Z`` of shape ``(n, K)``.

        With ``simplex=True`` the loadings are softmax-normalized so each patient is a **convex
        combination** of archetypes (loadings sum to 1, non-negative) — the Ellington-style
        parameterization in which archetypes are interpretable "pure subtype" extremes. With
        ``simplex=False`` the loadings are an unconstrained linear map (more flexible, less
        directly interpretable as subtype membership).
        """
        z: torch.Tensor = self.loading_map(c)
        if self.simplex:
            z = torch.softmax(z, dim=1)
        return z

    def _params(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(beta(C), mu(C))`` via the archetype factorization."""
        z = self._loadings(c)  # (n, K)
        w = z @ self.archetypes  # (n, x_dim + 1)
        return w[:, : self.x_dim], w[:, self.x_dim]

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the logit ``X . beta(C) + mu(C)``."""
        beta, mu = self._params(c)
        return (x * beta).sum(dim=1) + mu

    # ---- interpretability read-offs (all from this one model) ----

    @torch.no_grad()
    def predict_beta(self, c: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return episode-level context-conditioned coefficients ``beta(C)`` and ``mu(C)``.

        Parameters
        ----------
        c : numpy.ndarray
            Dense context matrix ``(n, context_dim)``.

        Returns
        -------
        tuple of numpy.ndarray
            ``beta`` ``(n, x_dim)`` and ``mu`` ``(n,)``.
        """
        beta, mu = self._params(torch.tensor(c, dtype=torch.float32))
        return beta.cpu().numpy().astype(np.float64), mu.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_loadings(self, c: FloatArray) -> FloatArray:
        """Per-patient archetype loadings ``Z(C)`` (the patient subtypes).

        Parameters
        ----------
        c : numpy.ndarray
            Dense context matrix.

        Returns
        -------
        numpy.ndarray
            Loadings ``(n, K)``.
        """
        z = self._loadings(torch.tensor(c, dtype=torch.float32))
        return np.asarray(z.cpu().numpy(), dtype=np.float64)

    @torch.no_grad()
    def homogeneous_effects(self, c: FloatArray) -> FloatArray:
        """Return the population-average coefficient vector ``beta`` = mean of ``beta(C)``.

        Parameters
        ----------
        c : numpy.ndarray
            Dense context matrix over which to average.

        Returns
        -------
        numpy.ndarray
            Mean per-treatment effect ``(x_dim,)``.
        """
        beta, _ = self.predict_beta(c)
        return np.asarray(beta.mean(axis=0), dtype=np.float64)

    @torch.no_grad()
    def normalized_archetypes(self) -> FloatArray:
        """Return canonicalized archetype coefficient profiles ``(K, x_dim)``.

        Each archetype row (treatment part only) is scaled to unit L2 norm and sign-fixed so that
        its largest-magnitude entry is positive, resolving the factorization's sign/scale ambiguity
        for reproducible figures.

        Returns
        -------
        numpy.ndarray
            Canonical archetype profiles ``(K, x_dim)``.
        """
        a = self.archetypes.detach().cpu().numpy()[:, : self.x_dim].astype(np.float64)
        out = np.zeros_like(a)
        for k in range(a.shape[0]):
            row = a[k]
            norm = np.linalg.norm(row)
            if norm > 0:
                row = row / norm
                if row[np.argmax(np.abs(row))] < 0:
                    row = -row
            out[k] = row
        return out

    @torch.no_grad()
    def predict_proba(self, c: FloatArray, x: FloatArray) -> FloatArray:
        """Positive-class probabilities for ``(C, X)``.

        Parameters
        ----------
        c, x : numpy.ndarray
            Dense context and treatment matrices.

        Returns
        -------
        numpy.ndarray
            Probabilities ``(n,)``.
        """
        logit = self.forward(
            torch.tensor(c, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
        )
        return torch.sigmoid(logit).cpu().numpy().astype(np.float64)


def select_stable_seed(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    n_archetypes: int,
    drug_names: list[str],
    seeds: range = range(10),
    epochs: int = 1500,
    min_subtype_frac: float = 0.03,
) -> int:
    """Select the training seed giving the most balanced, distinct subtypes.

    The archetype factorization is sensitive to initialization; many seeds collapse one or more
    subtypes. This trains across ``seeds`` and scores each by
    ``min_subtype_fraction * n_effective_subtypes * mean_pairwise_profile_distance`` (favoring
    balanced, well-populated, distinct subtypes), returning the best seed. Reproducibility is
    improved by always using the returned seed downstream.

    Parameters
    ----------
    c, x, y : numpy.ndarray
        Dense context, treatments, binary outcome.
    n_archetypes : int
        Number of archetypes.
    drug_names : list of str
        Treatment names (for profile-distance computation).
    seeds : range, default ``range(10)``
        Seeds to evaluate.
    epochs : int, default 1500
        Training epochs per seed.
    min_subtype_frac : float, default 0.02
        Minimum fraction for a subtype to count as "effective".

    Returns
    -------
    int
        The seed maximizing the balance/distinctness score.
    """
    from scipy.spatial.distance import pdist

    best_seed, best_score = int(seeds[0]), -np.inf
    for seed in seeds:
        model = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=n_archetypes, simplex=True, seed=seed
        )
        train(
            model,
            c,
            x,
            y,
            TrainConfig(
                epochs=epochs, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0
            ),
            seed=seed,
        )
        z = model.predict_loadings(c)
        sub = z.argmax(axis=1)
        counts = np.bincount(sub, minlength=n_archetypes)
        frac = counts / counts.sum()
        n_eff = int((counts >= len(y) * min_subtype_frac).sum())
        beta, _ = model.predict_beta(c)
        prof = pd.DataFrame(beta, columns=drug_names)
        prof["_s"] = sub
        means = prof.groupby("_s").mean()
        distinct = float(pdist(means.to_numpy()).mean()) if len(means) >= 2 else 0.0
        score = float(frac.min()) * n_eff * distinct
        if score > best_score:
            best_score, best_seed = score, int(seed)
    logger.info("Selected stable seed=%d (balance/distinctness score=%.3f)", best_seed, best_score)
    return best_seed


def select_n_archetypes(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    drug_names: list[str],
    k_range: range = range(8, 3, -1),
    min_subtype_frac: float = 0.03,
    seeds: range = range(40),
    epochs: int = 1500,
) -> tuple[int, int]:
    """Pick the largest archetype count ``K`` admitting a non-degenerate subtype solution.

    Interpretability-first model selection: rather than maximizing AUROC (flat in ``K`` here), pick
    the **largest** ``K`` for which some seed yields *every* archetype populated by at least
    ``min_subtype_frac`` of patients (no collapsed/empty subtypes). Among qualifying seeds the most
    distinct (largest mean pairwise profile distance) is returned. Falls back to the smallest ``K``
    in ``k_range`` if none qualifies.

    Parameters
    ----------
    c, x, y : numpy.ndarray
        Dense context, treatments, binary outcome.
    drug_names : list of str
        Treatment names (for profile-distance computation).
    k_range : range, default ``range(8, 3, -1)``
        Candidate ``K`` values, evaluated in the given (descending) order.
    min_subtype_frac : float, default 0.03
        Minimum fraction of patients each archetype must hold to count as populated.
    seeds : range, default ``range(40)``
        Seeds to evaluate per ``K``.
    epochs : int, default 1500
        Training epochs per fit.

    Returns
    -------
    tuple of (int, int)
        ``(k, seed)`` — the chosen archetype count and the qualifying seed.
    """
    from scipy.spatial.distance import pdist

    for k in k_range:
        best_seed, best_score = -1, -np.inf
        for seed in seeds:
            model = LinearArchetypeContextualized(
                c.shape[1], x.shape[1], n_archetypes=k, simplex=True, seed=seed
            )
            train(
                model,
                c,
                x,
                y,
                TrainConfig(
                    epochs=epochs,
                    lr=0.01,
                    weight_decay=1e-3,
                    archetype_l1=5e-4,
                    entropy_weight=1.0,
                ),
                seed=seed,
            )
            sub = model.predict_loadings(c).argmax(axis=1)
            frac = np.bincount(sub, minlength=k) / len(y)
            if (frac >= min_subtype_frac).sum() < k:
                continue  # some archetype collapsed below the floor -> reject
            beta, _ = model.predict_beta(c)
            prof = pd.DataFrame(beta, columns=drug_names)
            prof["_s"] = sub
            means = prof.groupby("_s").mean()
            distinct = float(pdist(means.to_numpy()).mean()) if len(means) >= 2 else 0.0
            # Prefer a *balanced* solution (large second subtype) over a dominant-blob carve-out:
            # the second-largest subtype fraction rewards a genuine split of the majority.
            second = float(np.sort(frac)[::-1][1])
            score = second * distinct
            if score > best_score:
                best_score, best_seed = score, int(seed)
        if best_seed >= 0:
            logger.info(
                "Selected K=%d (largest with all archetypes >=%.0f%%), seed=%d",
                k,
                100 * min_subtype_frac,
                best_seed,
            )
            return k, best_seed
    fallback_k = k_range[-1]
    logger.warning(
        "No K satisfied the >=%.0f%% floor; falling back to K=%d",
        100 * min_subtype_frac,
        fallback_k,
    )
    return fallback_k, int(seeds[0])


def heterogeneous_effect_pvals(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    context_features: dict[str, FloatArray],
    drug_names: list[str],
    n_archetypes: int = 5,
    n_boot: int = 20,
    epochs: int = 300,
    seed: int = SEED,
) -> pd.DataFrame:
    """Heterogeneity p-values per (context feature, drug), faithful to OG ``test_each_context``.

    This is the correct way to test whether a drug's effect varies with a context feature, and it
    replaces the flawed :func:`interaction_scan` (which used a group-mean-of-beta metric that
    conflates the context *main* effect with the interaction). The method, following Lengerich /
    Ellington / Xing:

    1. **Isolate** each context feature: fit a contextualized model on that *single* context column
       (linear encoder, no softmax — the package default), so the context main effect is absorbed
       by ``mu(C)`` and ``beta(C)``'s variation is the clean interaction.
    2. **Vary** the context from its minimum to its maximum and read each drug's effect ``beta`` at
       each end; the test statistic is ``delta_beta = beta(max) - beta(min)``.
    3. **Bootstrap** and compute a sign-consistency p-value: ``p = (1 + #{bootstraps whose
       delta_beta disagrees with the majority sign}) / (n_boot + 1)`` (Laplace-smoothed; minimum
       attainable p is ``1 / (n_boot + 1)``).

    Parameters
    ----------
    c : numpy.ndarray
        Dense context matrix (unused except for n; the per-context columns are passed explicitly).
    x, y : numpy.ndarray
        Treatments and binary outcome.
    context_features : dict of str to numpy.ndarray
        Named context columns to test (each shape ``(n,)``; binary or continuous).
    drug_names : list of str
        Treatment names.
    n_archetypes : int, default 5
        Archetypes in the (single-context) model.
    n_boot : int, default 20
        Bootstrap resamples. Determines p-value resolution (min p = 1/(n_boot+1)).
    epochs : int, default 300
        Training epochs per bootstrap.
    seed : int, default :data:`ctb.config.SEED`
        Base seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``context``, ``drug``, ``delta_beta`` (mean change in effect, min->max context),
        ``pval``, ``significant`` (p < 0.05). ``delta_beta`` sign indicates direction but is
        secondary to the p-value (the test is for *existence* of heterogeneity).
    """
    rng = np.random.default_rng(seed)
    n = c.shape[0]
    rows: list[dict[str, object]] = []
    for cname, cvals in context_features.items():
        col = np.asarray(cvals, dtype=np.float64).reshape(-1, 1)
        grid = np.array([[float(col.min())], [float(col.max())]])
        diffs = np.zeros((n_boot, x.shape[1]))
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            model = LinearArchetypeContextualized(
                1, x.shape[1], n_archetypes=n_archetypes, simplex=False, seed=seed + b
            )
            train(
                model,
                col[idx],
                x[idx],
                y[idx],
                TrainConfig(epochs=epochs, lr=0.02, weight_decay=1e-4, archetype_l1=0.0),
                seed=seed + b,
            )
            beta_grid, _ = model.predict_beta(grid)  # (2, n_drugs)
            diffs[b] = beta_grid[-1] - beta_grid[0]
        for j, dn in enumerate(drug_names):
            est = diffs[:, j]
            mean = float(est.mean())
            sign = np.sign(mean) if mean != 0 else 1.0
            pval = float((1 + np.sum(est * sign < 0)) / (n_boot + 1))
            rows.append(
                {
                    "context": cname,
                    "drug": dn,
                    "delta_beta": mean,
                    "pval": pval,
                    "significant": bool(pval < 0.05),
                }
            )
    return pd.DataFrame(rows)


def heterogeneity_atlas(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    context_masks: dict[str, NDArray[np.bool_]],
    drug_names: list[str],
    n_archetypes: int = 5,
    n_boot: int = 40,
    epochs: int = 2000,
    lr: float = 0.02,
    weight_decay: float = 1e-3,
    archetype_l1: float = 2e-3,
    simplex: bool = True,
    center_drugs: bool = False,
    seed: int = SEED,
) -> pd.DataFrame:
    """Decompose a *single* joint contextualized model's per-patient ``beta(C)`` across contexts.

    This is the "Method B" heterogeneity atlas: rather than isolating each context in its own model
    (:func:`heterogeneous_effect_pvals`), it trains **one** model on the full context matrix — the
    same headline model that produces a personalized effect vector ``beta(C_i)`` for every patient —
    and then asks, for each interpretable context indicator and each drug, how much that
    personalized effect shifts between patients who have the feature and those who do not:

    ``delta_beta = mean_i[beta_drug(C_i) | feature] - mean_i[beta_drug(C_i) | not feature]``.

    Significance is a sign-consistency bootstrap: the joint model is refit on ``n_boot`` patient
    resamples and ``delta_beta`` recomputed on the fixed population each time;
    ``p = (1 + #{resamples disagreeing with the majority sign}) / (n_boot + 1)``.

    **Interpretation / caveat.** Because the effects come from the *joint* model, ``delta_beta``
    reflects a feature's association with the personalized effect *including* signal carried by
    correlated context (e.g. region ↔ resistance ↔ DST). It is an exploratory atlas of the
    personalized model, not the isolated causal effect-modification of a single feature.

    Parameters
    ----------
    c : numpy.ndarray
        Dense (encoded) context matrix ``(n, context_dim)`` used to fit the joint model.
    x, y : numpy.ndarray
        Treatments ``(n, x_dim)`` and binary outcome ``(n,)``.
    context_masks : dict of str to numpy.ndarray
        Named boolean indicators ``(n,)`` over which to decompose the effects (one per interpretable
        context level / binarized numeric).
    drug_names : list of str
        Treatment names.
    n_archetypes : int, default 5
        Archetypes in the joint model.
    n_boot : int, default 40
        Bootstrap refits of the joint model (determines p-value resolution, min ``1/(n_boot+1)``).
    epochs, lr, weight_decay, archetype_l1 : training hyperparameters
        Passed to :class:`TrainConfig` (defaults match the headline model in script 06).
    simplex : bool, default True
        Use simplex (softmax) loadings, matching the headline subtype model.
    center_drugs : bool, default False
        If True, subtract each context's across-drug mean ``delta_beta`` before the test, isolating
        **drug-specific** effect modification (interactions) from the global responsiveness gradient
        (a context that shifts *all* drugs together — closer to a prognostic / context main effect).
    seed : int, default :data:`ctb.config.SEED`
        Base seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``context``, ``drug``, ``delta_beta``, ``pval``, ``significant`` — the same schema
        as :func:`heterogeneous_effect_pvals`, so it feeds the heterogeneity clustermap directly.
    """
    boot, names = atlas_bootstrap(
        c,
        x,
        y,
        context_masks,
        n_archetypes=n_archetypes,
        n_boot=n_boot,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        archetype_l1=archetype_l1,
        simplex=simplex,
        seed=seed,
    )
    return scan_from_atlas_boot(boot, names, drug_names, center_drugs=center_drugs)


def atlas_bootstrap(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    context_masks: dict[str, NDArray[np.bool_]],
    n_archetypes: int = 5,
    n_boot: int = 40,
    epochs: int = 2000,
    lr: float = 0.02,
    weight_decay: float = 1e-3,
    archetype_l1: float = 2e-3,
    simplex: bool = True,
    seed: int = SEED,
) -> tuple[FloatArray, list[str]]:
    """Bootstrap the joint model and return *raw* per-context, per-drug ``delta_beta`` resamples.

    Separated from :func:`heterogeneity_atlas` so that both the raw and drug-centered (interaction)
    scans can be derived from a *single* bootstrap run (see :func:`scan_from_atlas_boot`).

    Parameters
    ----------
    c, x, y : numpy.ndarray
        Context, treatments, outcome.
    context_masks : dict of str to numpy.ndarray
        Named boolean indicators over patients.
    n_archetypes, n_boot, epochs, lr, weight_decay, archetype_l1, simplex, seed
        See :func:`heterogeneity_atlas`.

    Returns
    -------
    tuple of (numpy.ndarray, list of str)
        ``boot`` array ``(n_boot, n_context, n_drug)`` of raw ``delta_beta`` (NaN where a mask is
        empty in a resample) and the ordered context names.
    """
    rng = np.random.default_rng(seed)
    n, context_dim = c.shape
    names = list(context_masks)
    masks = {k: np.asarray(v, dtype=bool) for k, v in context_masks.items()}
    cfg = TrainConfig(epochs=epochs, lr=lr, weight_decay=weight_decay, archetype_l1=archetype_l1)

    boot = np.full((n_boot, len(names), x.shape[1]), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearArchetypeContextualized(
            context_dim, x.shape[1], n_archetypes=n_archetypes, simplex=simplex, seed=seed + b
        )
        train(model, c[idx], x[idx], y[idx], cfg, seed=seed + b)
        beta, _ = model.predict_beta(c)  # personalized effects on the fixed population (n, n_drugs)
        for fi, nm in enumerate(names):
            m = masks[nm]
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            boot[b, fi, :] = beta[m].mean(axis=0) - beta[~m].mean(axis=0)
        logger.info("Heterogeneity-atlas bootstrap %d/%d complete", b + 1, n_boot)
    return boot, names


def scan_from_atlas_boot(
    boot: FloatArray, names: list[str], drug_names: list[str], center_drugs: bool = False
) -> pd.DataFrame:
    """Build a heterogeneity scan (sign-consistency p-values) from bootstrap ``delta_beta``.

    Parameters
    ----------
    boot : numpy.ndarray
        ``(n_boot, n_context, n_drug)`` raw ``delta_beta`` from :func:`atlas_bootstrap`.
    names : list of str
        Context names (row order of ``boot``).
    drug_names : list of str
        Drug names (last axis of ``boot``).
    center_drugs : bool, default False
        If True, subtract each (resample, context) across-drug mean before testing, isolating
        drug-specific effect modification from the global responsiveness gradient.

    Returns
    -------
    pandas.DataFrame
        Columns ``context``, ``drug``, ``delta_beta``, ``ci_low``, ``ci_high``, ``pval``,
        ``significant``.
    """
    arr = boot - np.nanmean(boot, axis=2, keepdims=True) if center_drugs else boot
    rows: list[dict[str, object]] = []
    for fi, nm in enumerate(names):
        for j, dn in enumerate(drug_names):
            est = arr[:, fi, j]
            est = est[~np.isnan(est)]
            if len(est) == 0:
                continue
            mean = float(est.mean())
            sign = np.sign(mean) if mean != 0 else 1.0
            pval = float((1 + np.sum(est * sign < 0)) / (len(est) + 1))
            rows.append(
                {
                    "context": nm,
                    "drug": dn,
                    "delta_beta": mean,
                    "ci_low": float(np.percentile(est, 2.5)),
                    "ci_high": float(np.percentile(est, 97.5)),
                    "pval": pval,
                    "significant": bool(pval < 0.05),
                }
            )
    return pd.DataFrame(rows)


def interaction_scan(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    context_masks: dict[str, NDArray[np.bool_]],
    drug_names: list[str],
    n_archetypes: int = 5,
    n_boot: int = 20,
    epochs: int = 700,
    seed: int = SEED,
) -> pd.DataFrame:
    """Scan interactions via group-mean-of-beta (DEPRECATED — statistically flawed).

    .. warning::
        This metric (mean ``beta`` among context-present minus context-absent) **conflates the
        context main effect with the interaction** and was empirically shown to produce
        sign-flipped, unreliable results (it disagreed with both adjusted logistic interaction
        models and the faithful framework method). Use :func:`heterogeneous_effect_pvals` instead.
        Retained only for provenance/reproducibility of the superseded analysis.

    Parameters
    ----------
    c, x, y : numpy.ndarray
        Dense context, treatments, binary outcome.
    context_masks : dict of str to numpy.ndarray
        Named boolean patient masks (e.g. ``{"anemia": ..., "XDR": ...}``).
    drug_names : list of str
        Treatment names.
    n_archetypes : int, default 5
        Number of archetypes.
    n_boot : int, default 20
        Bootstrap resamples.
    epochs : int, default 700
        Training epochs per bootstrap.
    seed : int, default :data:`ctb.config.SEED`
        Base seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``drug``, ``context``, ``interaction`` (mean), ``ci_low``, ``ci_high``,
        ``significant``, sorted by absolute interaction magnitude.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    acc: dict[tuple[str, str], list[float]] = {}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=n_archetypes, simplex=True, seed=seed + b
        )
        train(
            model,
            c[idx],
            x[idx],
            y[idx],
            TrainConfig(epochs=epochs, lr=0.015, weight_decay=1e-3, archetype_l1=5e-4),
            seed=seed + b,
        )
        beta, _ = model.predict_beta(c)
        bdf = pd.DataFrame(beta, columns=drug_names)
        for fn, mask in context_masks.items():
            if mask.sum() < 40 or (~mask).sum() < 40:
                continue
            diff = bdf[mask].mean() - bdf[~mask].mean()
            for dn in drug_names:
                acc.setdefault((dn, fn), []).append(float(diff[dn]))
    rows: list[dict[str, object]] = []
    for (dn, fn), vals in acc.items():
        arr = np.asarray(vals)
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        rows.append(
            {
                "drug": dn,
                "context": fn,
                "interaction": float(arr.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    out = pd.DataFrame(rows)
    return out.reindex(out["interaction"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def bootstrap_context_effects(
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    stratum_masks: dict[str, NDArray[np.bool_]],
    drug_names: list[str],
    n_archetypes: int = 5,
    simplex: bool = True,
    n_boot: int = 25,
    epochs: int = 600,
    seed: int = SEED,
) -> pd.DataFrame:
    """Bootstrap CIs for the mean treatment effect of each drug within each context stratum.

    Refits the model on ``n_boot`` bootstrap resamples; for each, evaluates per-patient ``beta(C)``
    on the full data and averages within each stratum. Returns the mean and 95% percentile CI per
    (stratum, drug), plus a significance flag (CI excludes 0). This converts the heterogeneous-
    effect heatmaps from point estimates into statistically-supported claims.

    Parameters
    ----------
    c, x, y : numpy.ndarray
        Dense context, treatment matrix, binary outcome.
    stratum_masks : dict of str to numpy.ndarray
        Named boolean masks over patients (e.g. ``{"XDR": ..., "HIV+": ...}``).
    drug_names : list of str
        Treatment column names.
    n_archetypes : int, default 5
        Number of archetypes.
    simplex : bool, default True
        Use softmax (convex) loadings.
    n_boot : int, default 25
        Number of bootstrap resamples.
    epochs : int, default 600
        Training epochs per bootstrap (kept modest for runtime).
    seed : int, default :data:`ctb.config.SEED`
        Base seed.

    Returns
    -------
    pandas.DataFrame
        Long-format columns: ``stratum``, ``drug``, ``mean``, ``ci_low``, ``ci_high``,
        ``significant`` (bool).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    acc: dict[str, list[FloatArray]] = {k: [] for k in stratum_masks}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=n_archetypes, simplex=simplex, seed=seed + b
        )
        train(
            model,
            c[idx],
            x[idx],
            y[idx],
            TrainConfig(epochs=epochs, lr=0.02, weight_decay=1e-3, archetype_l1=1e-3),
            seed=seed + b,
        )
        beta, _ = model.predict_beta(c)
        bdf = pd.DataFrame(beta, columns=drug_names)
        for k, mask in stratum_masks.items():
            acc[k].append(bdf[mask].mean().to_numpy())
    rows: list[dict[str, object]] = []
    for k, mats in acc.items():
        arr = np.asarray(mats)  # (n_boot, n_drugs)
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        for j, dn in enumerate(drug_names):
            rows.append(
                {
                    "stratum": k,
                    "drug": dn,
                    "mean": float(mean[j]),
                    "ci_low": float(lo[j]),
                    "ci_high": float(hi[j]),
                    "significant": bool(lo[j] > 0 or hi[j] < 0),
                }
            )
    return pd.DataFrame(rows)


def label_subtypes_by_profile(
    subtype_drug_effects: FloatArray, drug_names: list[str]
) -> tuple[NDArray[np.int_], list[str]]:
    """Assign stable, profile-based labels to subtypes so figures are reproducible across runs.

    Because the archetype factorization is initialized randomly, raw subtype indices permute
    between runs. This relabels subtypes by their dominant drug-response signature (the drug each
    subtype most strongly responds to) and returns a canonical ordering (by that drug's column
    index), so a subtype is identified by *what it responds to*, not by an arbitrary integer.

    Parameters
    ----------
    subtype_drug_effects : numpy.ndarray
        ``(n_subtypes, n_drugs)`` mean treatment effect per drug per subtype.
    drug_names : list of str
        Drug column names (length ``n_drugs``).

    Returns
    -------
    tuple of (numpy.ndarray, list of str)
        A permutation mapping old subtype index -> new ordered index, and the descriptive labels
        (e.g. ``"resp:Lzd"``) in the new order.
    """
    dominant = subtype_drug_effects.argmax(axis=1)
    order = np.argsort(dominant)  # order subtypes by their dominant-drug column
    labels = [f"resp:{drug_names[dominant[i]]}" for i in order]
    remap = np.empty(len(order), dtype=np.int_)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    return remap, labels


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes
    ----------
    epochs : int
        Full-batch gradient steps.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 penalty (regularizes loadings + archetypes; important for the flexible factorization).
    archetype_l1 : float
        Optional L1 penalty on archetype profiles to encourage sparse, interpretable archetypes
        (each archetype responds to a few drugs). 0 disables it.
    balance_weight : float
        Optional load-balancing penalty weight. Penalizes any archetype whose mean soft loading
        exceeds ``balance_cap``, preventing the degenerate "one dominant blob" optimum and forcing
        the model to use multiple balanced subtypes. 0 disables it.
    balance_cap : float
        Per-archetype soft-usage cap above which the balance penalty activates (e.g. 0.40 = no
        archetype may absorb more than ~40% of the population's loading mass).
    entropy_weight : float
        Optional maximum-entropy load-balancing penalty. Subtracts ``entropy_weight * H(mean_load)``
        from the loss, rewarding a *flat* archetype-usage distribution and preventing archetype
        collapse (the standard cluster-collapse regularizer used in balanced deep clustering, e.g.
        DeepCluster/SwAV). More stable than the one-sided ``balance_cap`` hinge: it penalizes both
        over- and under-used archetypes. 0 disables it.
    """

    epochs: int = 2000
    lr: float = 0.02
    weight_decay: float = 1e-4
    archetype_l1: float = 0.0
    balance_weight: float = 0.0
    balance_cap: float = 0.40
    entropy_weight: float = 0.0


def train(
    model: LinearArchetypeContextualized,
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    config: TrainConfig | None = None,
    seed: int = SEED,
) -> LinearArchetypeContextualized:
    """Train the linear-archetype contextualized model (full-batch Adam, BCE loss).

    Parameters
    ----------
    model : LinearArchetypeContextualized
        Model to train (modified in place and returned).
    c, x, y : numpy.ndarray
        Dense context ``(n, context_dim)``, treatments ``(n, x_dim)``, binary outcome ``(n,)``.
    config : TrainConfig or None
        Hyperparameters (defaults used if ``None``).
    seed : int, default :data:`ctb.config.SEED`
        Torch manual seed.

    Returns
    -------
    LinearArchetypeContextualized
        Trained model.
    """
    config = config or TrainConfig()
    torch.manual_seed(seed)
    ct = torch.tensor(c, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(config.epochs):
        opt.zero_grad()
        loss = loss_fn(model(ct, xt), yt)
        if config.archetype_l1 > 0:
            loss = loss + config.archetype_l1 * model.archetypes.abs().mean()
        if config.balance_weight > 0 or config.entropy_weight > 0:
            mean_load = model._loadings(ct).mean(dim=0)
            if config.balance_weight > 0:
                # Squared-hinge cap: penalize any archetype above the usage cap (one-sided).
                overflow = torch.relu(mean_load - config.balance_cap)
                loss = loss + config.balance_weight * overflow.pow(2).sum()
            if config.entropy_weight > 0:
                # Maximum-entropy balance: reward a flat usage distribution (prevents collapse,
                # both over- and under-use). Standard cluster-collapse regularizer.
                entropy = -(mean_load * torch.log(mean_load + 1e-9)).sum()
                loss = loss - config.entropy_weight * entropy
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    model.eval()
    logger.info(
        "Trained linear-archetype contextualized (K=%d, final loss=%.4f)",
        model.n_archetypes,
        float(loss.detach()),
    )
    return model
