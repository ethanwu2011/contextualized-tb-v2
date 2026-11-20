"""Unit tests for the linear-archetype contextualized model."""

from __future__ import annotations

import numpy as np

from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    bootstrap_context_effects,
    label_subtypes_by_profile,
    train,
)


def _interaction_data(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One treatment, one binary modifier: beta_moxi = -1.5 when modifier=1, ~0 otherwise."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 3)).astype(np.float64)
    c[:, 0] = (rng.uniform(size=n) < 0.3).astype(np.float64)  # modifier
    x = (rng.uniform(size=n) < 0.4).reshape(-1, 1).astype(np.float64)
    logit = 0.3 - 1.5 * x[:, 0] * c[:, 0]
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=n)).astype(np.float64)
    return c, x, y


def test_shapes_and_readoffs() -> None:
    c, x, y = _interaction_data()
    m = LinearArchetypeContextualized(c.shape[1], 1, n_archetypes=3)
    train(m, c, x, y, TrainConfig(epochs=300, lr=0.03))
    beta, mu = m.predict_beta(c)
    assert beta.shape == (len(c), 1)
    assert mu.shape == (len(c),)
    assert m.predict_loadings(c).shape == (len(c), 3)
    assert m.homogeneous_effects(c).shape == (1,)
    assert m.normalized_archetypes().shape == (3, 1)
    assert m.predict_proba(c, x).shape == (len(c),)


def test_k1_recovers_interaction_sign() -> None:
    # K=1 is a rank-1 interaction model; should recover beta_moxi more negative when modifier=1.
    c, x, y = _interaction_data()
    m = LinearArchetypeContextualized(c.shape[1], 1, n_archetypes=1)
    train(m, c, x, y, TrainConfig(epochs=1500, lr=0.02))
    beta, _ = m.predict_beta(c)
    b_mod = beta[c[:, 0] == 1, 0].mean()
    b_nomod = beta[c[:, 0] == 0, 0].mean()
    assert b_mod < b_nomod - 0.3  # effect more negative in the modifier=1 group


def test_archetype_recovery_multidrug() -> None:
    # 5 drugs, effects driven by 2 latent archetypes -> model should recover per-patient beta well.
    rng = np.random.default_rng(0)
    n, m_ctx, d = 4000, 6, 5
    c = rng.normal(size=(n, m_ctx)).astype(np.float64)
    s1 = 1 / (1 + np.exp(-(c[:, 0] + 0.5 * c[:, 1])))
    s2 = 1 / (1 + np.exp(-(c[:, 2] - 0.5 * c[:, 3])))
    a1 = np.array([-1.5, -1.2, 0, 0, 0])
    a2 = np.array([0, 0, -1.4, -1.1, 0])
    beta_true = s1[:, None] * a1 + s2[:, None] * a2
    x = (rng.uniform(size=(n, d)) < 0.5).astype(np.float64)
    y = (1 / (1 + np.exp(-((x * beta_true).sum(1) + 0.2))) > rng.uniform(size=n)).astype(np.float64)
    model = LinearArchetypeContextualized(m_ctx, d, n_archetypes=2)
    train(model, c, x, y, TrainConfig(epochs=2500, lr=0.02, weight_decay=1e-4))
    beta_hat, _ = model.predict_beta(c)
    corrs = [np.corrcoef(beta_hat[:, k], beta_true[:, k])[0, 1] for k in range(d)]
    assert np.nanmean(corrs) > 0.8  # recovers the low-rank treatment-effect structure


def test_normalized_archetypes_canonical() -> None:
    c, x, y = _interaction_data()
    m = LinearArchetypeContextualized(c.shape[1], 4, n_archetypes=3)
    # inject known archetype values
    with __import__("torch").no_grad():
        m.archetypes.copy_(
            __import__("torch").tensor(
                [[2.0, -1.0, 0.0, 0.0, 9.0], [-3.0, 0.5, 0.0, 0.0, 9.0], [0.0, 0.0, 0.0, 0.0, 9.0]]
            )
        )
    norm = m.normalized_archetypes()
    assert norm.shape == (3, 4)
    # rows with signal are unit-norm; largest-magnitude entry positive
    for k in range(2):
        assert abs(np.linalg.norm(norm[k]) - 1.0) < 1e-6
        assert norm[k][np.argmax(np.abs(norm[k]))] > 0


def test_label_subtypes_by_profile() -> None:
    # subtype 0 responds most to drug index 2, subtype 1 to index 0, subtype 2 to index 1
    eff = np.array([[0.1, 0.2, 0.9], [0.8, 0.1, 0.0], [0.2, 0.7, 0.1]])
    remap, labels = label_subtypes_by_profile(eff, ["A", "B", "C"])
    # ordering by dominant-drug column: subtype1(dom A=0), subtype2(dom B=1), subtype0(dom C=2)
    assert labels == ["resp:A", "resp:B", "resp:C"]
    assert remap[1] == 0 and remap[2] == 1 and remap[0] == 2


def test_bootstrap_context_effects_recovers_known_modifier() -> None:
    # one treatment helps only when modifier=1; bootstrap CI should be sig-positive in that stratum
    rng = np.random.default_rng(0)
    n = 1500
    c = rng.normal(size=(n, 3)).astype(np.float64)
    c[:, 0] = (rng.uniform(size=n) < 0.5).astype(np.float64)
    x = (rng.uniform(size=n) < 0.5).reshape(-1, 1).astype(np.float64)
    # drug helps (raises conversion) only when modifier=1
    logit = -0.2 + 1.5 * x[:, 0] * c[:, 0]
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=n)).astype(np.float64)
    masks = {"mod+": c[:, 0] == 1, "mod-": c[:, 0] == 0}
    res = bootstrap_context_effects(c, x, y, masks, ["drug"], n_archetypes=2, n_boot=8, epochs=300)
    pos = res[(res["stratum"] == "mod+") & (res["drug"] == "drug")].iloc[0]
    assert pos["mean"] > 0  # positive effect in the modifier-present stratum
    assert {"stratum", "drug", "mean", "ci_low", "ci_high", "significant"} <= set(res.columns)


def test_k1_predicts_reasonably() -> None:
    c, x, y = _interaction_data()
    m = LinearArchetypeContextualized(c.shape[1], 1, n_archetypes=1)
    train(m, c, x, y, TrainConfig(epochs=800, lr=0.03))
    p = m.predict_proba(c, x)
    assert ((p >= 0) & (p <= 1)).all()
