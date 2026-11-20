"""Tests for the identified additive residual PATCH model."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ctb.models.residual_archetype import (
    ResidualArchetypeLogistic,
    ResidualTrainConfig,
    compute_residual_loss_components,
    train_residual,
)


def test_identified_coordinates_and_manuscript_parameter_count() -> None:
    model = ResidualArchetypeLogistic(279, 26, n_archetypes=5, seed=0)
    c = torch.randn(13, 279)

    torch.testing.assert_close(
        model._loading_logits(c).sum(dim=1),
        torch.zeros(13),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        model._deviations().sum(dim=0),
        torch.zeros(26),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        model.contrast_basis.T @ model.contrast_basis,
        torch.eye(4),
        atol=1e-6,
        rtol=0,
    )
    assert model.parameter_count == 1530


def test_exactly_nests_supplied_logistic_baseline() -> None:
    rng = np.random.default_rng(4)
    c = rng.normal(size=(17, 3)).astype(np.float64)
    x = rng.normal(size=(17, 2)).astype(np.float64)
    context_coef = np.array([0.4, -0.2, 0.7])
    treatment_coef = np.array([-0.8, 0.3])
    intercept = -0.15
    model = ResidualArchetypeLogistic(3, 2, n_archetypes=4, seed=9)
    model.initialize_from_logistic(context_coef, treatment_coef, intercept)

    expected = intercept + c @ context_coef + x @ treatment_coef
    actual = (
        model(torch.tensor(c, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
        .detach()
        .numpy()
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)


def test_k1_is_exactly_baseline_logistic() -> None:
    rng = np.random.default_rng(12)
    c = rng.normal(size=(9, 2)).astype(np.float64)
    x = rng.normal(size=(9, 3)).astype(np.float64)
    model = ResidualArchetypeLogistic(2, 3, n_archetypes=1, seed=0)
    model.initialize_from_logistic(
        np.array([0.2, -0.1]),
        np.array([0.3, -0.4, 0.5]),
        0.6,
    )

    expected = 0.6 + c @ np.array([0.2, -0.1]) + x @ np.array([0.3, -0.4, 0.5])
    actual = (
        model(torch.tensor(c, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
        .detach()
        .numpy()
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
    np.testing.assert_array_equal(model.predict_loadings(c), np.ones((len(c), 1)))


def test_zero_residual_scale_returns_exact_baseline_after_interaction_fit() -> None:
    rng = np.random.default_rng(13)
    c = rng.normal(size=(11, 3)).astype(np.float64)
    x = rng.normal(size=(11, 2)).astype(np.float64)
    model = ResidualArchetypeLogistic(3, 2, n_archetypes=4, seed=5)
    model.initialize_from_logistic(
        np.array([0.2, -0.4, 0.1]),
        np.array([0.8, -0.3]),
        -0.2,
    )
    with torch.no_grad():
        model.deviation_coordinates.normal_(mean=0.0, std=2.0)
    model.set_residual_scale(0.0)

    expected = -0.2 + c @ np.array([0.2, -0.4, 0.1]) + x @ np.array([0.8, -0.3])
    actual = (
        model(torch.tensor(c, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
        .detach()
        .numpy()
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)


def test_prediction_coefficients_are_mixtures_of_total_prototypes() -> None:
    rng = np.random.default_rng(3)
    c = rng.normal(size=(21, 4)).astype(np.float64)
    model = ResidualArchetypeLogistic(4, 3, n_archetypes=4, seed=1)
    with torch.no_grad():
        model.deviation_coordinates.normal_(mean=0.0, std=0.4)
        model.treatment_main.copy_(torch.tensor([0.2, -0.3, 0.7]))

    beta, _ = model.predict_beta(c)
    expected = model.predict_loadings(c) @ model.prototype_coefficients()

    np.testing.assert_allclose(beta, expected, rtol=1e-6, atol=2e-7)


def test_changing_loadings_cannot_change_context_only_prognosis() -> None:
    rng = np.random.default_rng(8)
    c = rng.normal(size=(15, 5)).astype(np.float64)
    x = np.zeros((15, 2), dtype=np.float64)
    model = ResidualArchetypeLogistic(5, 2, n_archetypes=3, seed=2)
    before = model.predict_proba(c, x)
    assert model.loading_map is not None

    with torch.no_grad():
        model.loading_map.weight.normal_(mean=0.0, std=20.0)
        model.loading_map.bias.normal_(mean=0.0, std=20.0)
    after = model.predict_proba(c, x)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)


def test_baseline_checkpoint_has_nonzero_deviation_gradient() -> None:
    rng = np.random.default_rng(14)
    c = rng.normal(size=(300, 3)).astype(np.float32)
    x = (rng.uniform(size=(300, 1)) < 0.5).astype(np.float32)
    logit = 2.0 * x[:, 0] * np.tanh(2.0 * c[:, 0])
    y = (rng.uniform(size=300) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    model = ResidualArchetypeLogistic(3, 1, n_archetypes=2, seed=4)
    model.initialize_from_logistic(np.zeros(3), np.zeros(1), 0.0, freeze=True)
    config = ResidualTrainConfig(
        epochs=1,
        loading_l2=0.0,
        deviation_l2=0.0,
    )

    loss = compute_residual_loss_components(
        model,
        torch.tensor(c),
        torch.tensor(x),
        torch.tensor(y),
        config,
    ).total
    loss.backward()

    assert model.deviation_coordinates.grad is not None
    assert float(model.deviation_coordinates.grad.norm()) > 1e-5


def test_archetype_label_permutation_preserves_predictions() -> None:
    rng = np.random.default_rng(11)
    c = rng.normal(size=(19, 5)).astype(np.float64)
    x = rng.normal(size=(19, 2)).astype(np.float64)
    model = ResidualArchetypeLogistic(5, 2, n_archetypes=4, seed=3)
    assert model.loading_map is not None
    with torch.no_grad():
        model.deviation_coordinates.normal_(mean=0.0, std=0.5)
    before = model.predict_proba(c, x)

    permutation = torch.tensor([2, 0, 3, 1])
    permutation_matrix = torch.eye(4)[permutation]
    coordinate_transform = model.contrast_basis.T @ permutation_matrix @ model.contrast_basis
    with torch.no_grad():
        model.loading_map.weight.copy_(coordinate_transform @ model.loading_map.weight.clone())
        model.loading_map.bias.copy_(coordinate_transform @ model.loading_map.bias.clone())
        model.deviation_coordinates.copy_(
            coordinate_transform @ model.deviation_coordinates.clone()
        )
    after = model.predict_proba(c, x)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)


def test_baseline_ridge_term_excludes_intercept() -> None:
    model = ResidualArchetypeLogistic(2, 1, n_archetypes=2, seed=0)
    with torch.no_grad():
        model.context_main.weight.copy_(torch.tensor([[1.0, -2.0]]))
        model.treatment_main.copy_(torch.tensor([3.0]))
    c = torch.zeros((4, 2))
    x = torch.zeros((4, 1))
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    config = ResidualTrainConfig(
        epochs=1,
        baseline_l2=2.0,
        loading_l2=0.0,
        deviation_l2=0.0,
    )
    before = compute_residual_loss_components(model, c, x, y, config)
    with torch.no_grad():
        model.context_main.bias.fill_(50.0)
    after = compute_residual_loss_components(model, c, x, y, config)

    torch.testing.assert_close(before.baseline_l2, torch.tensor(14.0))
    torch.testing.assert_close(after.baseline_l2, before.baseline_l2)


def test_low_rank_interaction_improves_heldout_log_loss() -> None:
    rng = np.random.default_rng(21)
    n_train, n_test = 1800, 800
    c = rng.normal(size=(n_train + n_test, 3)).astype(np.float64)
    x = (rng.uniform(size=(n_train + n_test, 1)) < 0.5).astype(np.float64)
    logit = 0.3 * c[:, 1] + 3.0 * x[:, 0] * np.tanh(2.0 * c[:, 0])
    y = (rng.uniform(size=len(c)) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
    train_index = np.arange(n_train)
    test_index = np.arange(n_train, len(c))
    baseline = LogisticRegression(C=1.0, max_iter=2000).fit(
        np.hstack([c[train_index], x[train_index]]),
        y[train_index],
    )
    baseline_probability = baseline.predict_proba(np.hstack([c[test_index], x[test_index]]))[:, 1]

    model = ResidualArchetypeLogistic(3, 1, n_archetypes=2, seed=6)
    model.initialize_from_logistic(
        baseline.coef_[0, :3],
        baseline.coef_[0, 3:],
        float(baseline.intercept_[0]),
        freeze=True,
    )
    train_residual(
        model,
        c[train_index],
        x[train_index],
        y[train_index],
        ResidualTrainConfig(
            epochs=500,
            lr=0.03,
            loading_l2=1e-4,
            deviation_l2=1e-4,
        ),
        seed=6,
    )
    residual_probability = model.predict_proba(c[test_index], x[test_index])

    assert log_loss(y[test_index], residual_probability) < (
        log_loss(y[test_index], baseline_probability) - 0.03
    )


def test_checkpoint_roundtrip_preserves_predictions() -> None:
    rng = np.random.default_rng(5)
    c = rng.normal(size=(12, 3)).astype(np.float64)
    x = rng.normal(size=(12, 2)).astype(np.float64)
    model = ResidualArchetypeLogistic(3, 2, n_archetypes=3, seed=7)
    with torch.no_grad():
        model.deviation_coordinates.normal_()
    expected = model.predict_proba(c, x)
    restored = ResidualArchetypeLogistic(3, 2, n_archetypes=3, seed=99)

    restored.load_state_dict(model.state_dict())

    np.testing.assert_allclose(restored.predict_proba(c, x), expected, rtol=0, atol=0)
