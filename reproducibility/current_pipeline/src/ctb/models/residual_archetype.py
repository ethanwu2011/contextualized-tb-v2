"""Identified residual PATCH model for associative treatment heterogeneity.

The legacy PATCH estimator routes both baseline prognosis and regimen coefficients through one
simplex-archetype bottleneck.  That is useful as a compact representation but it does not contain
ordinary logistic regression on context and regimen as a special case.

This module defines an additive extension:

``logit = alpha + C gamma + X [delta + s D.T z(C)]``, with post-fit shrinkage ``s >= 0``.

The context/regimen logistic component is exact when ``D = 0`` or ``s = 0``.  A fixed orthonormal
contrast basis constrains both archetype deviations and loading logits to sum to zero across
archetypes, removing their common-shift redundancies.  The total prototype profiles are
``delta + s D_k``.

The model is descriptive and predictive.  Without a treatment-assignment identification strategy,
its context-varying regimen coefficients describe treatment-associated heterogeneity and are not
causal effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim.adam import Adam

from ctb.config import SEED

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


def _contrast_basis(n_archetypes: int) -> torch.Tensor:
    """Return a deterministic Helmert basis orthogonal to the all-ones vector."""
    if n_archetypes < 1:
        raise ValueError("n_archetypes must be at least 1")
    basis = np.zeros((n_archetypes, n_archetypes - 1), dtype=np.float32)
    for column in range(n_archetypes - 1):
        denominator = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return torch.from_numpy(basis)


class ResidualArchetypeLogistic(nn.Module):
    """Additive logistic baseline plus identified simplex-archetype deviations.

    Parameters
    ----------
    context_dim : int
        Number of context features.
    x_dim : int
        Number of recorded regimen indicators.
    n_archetypes : int
        Number of total regimen prototype profiles. ``K=1`` is exactly the additive baseline.
    seed : int or None, default None
        Seed applied before parameter initialization.
    """

    contrast_basis: torch.Tensor
    loading_map: nn.Linear | None
    residual_scale: torch.Tensor

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        n_archetypes: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if context_dim < 1 or x_dim < 1:
            raise ValueError("context_dim and x_dim must both be at least 1")
        if n_archetypes < 1:
            raise ValueError("n_archetypes must be at least 1")
        if seed is not None:
            torch.manual_seed(seed)

        self.context_dim = context_dim
        self.x_dim = x_dim
        self.n_archetypes = n_archetypes
        self.register_buffer("contrast_basis", _contrast_basis(n_archetypes))
        self.register_buffer("residual_scale", torch.tensor(1.0))

        contrast_dim = n_archetypes - 1
        self.loading_map = nn.Linear(context_dim, contrast_dim) if contrast_dim else None
        # Zero deviations make initialization exactly equal to the additive baseline.  The random
        # loading map still gives the deviation coordinates a nonzero gradient when interactions
        # are present; initializing both blocks at zero would create a stationary symmetry.
        self.deviation_coordinates = nn.Parameter(torch.zeros(contrast_dim, x_dim))
        self.context_main = nn.Linear(context_dim, 1)
        self.treatment_main = nn.Parameter(torch.zeros(x_dim))

    @property
    def parameter_count(self) -> int:
        """Return the number of learnable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    def _loading_logits(self, c: torch.Tensor) -> torch.Tensor:
        """Return identified sum-zero loading logits of shape ``(n, K)``."""
        if self.loading_map is None:
            return c.new_zeros((c.shape[0], 1))
        coordinates = self.loading_map(c)
        logits: torch.Tensor = coordinates @ self.contrast_basis.T
        return logits

    def _loadings(self, c: torch.Tensor) -> torch.Tensor:
        """Return simplex loadings ``z(C)`` of shape ``(n, K)``."""
        return torch.softmax(self._loading_logits(c), dim=1)

    def _deviations(self) -> torch.Tensor:
        """Return sum-zero archetype deviations ``D`` of shape ``(K, d)``."""
        deviations: torch.Tensor = self.contrast_basis @ self.deviation_coordinates
        return deviations

    def _prototype_tensor(self) -> torch.Tensor:
        """Return total regimen profiles ``delta + D_k`` of shape ``(K, d)``."""
        return self.treatment_main.unsqueeze(0) + self.residual_scale * self._deviations()

    def _params(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context-varying regimen coefficients and independent prognosis."""
        beta = self.treatment_main.unsqueeze(0) + self.residual_scale * (
            self._loadings(c) @ self._deviations()
        )
        mu = self.context_main(c).squeeze(dim=1)
        return beta, mu

    def set_residual_scale(self, scale: float) -> None:
        """Set a nonnegative post-fit shrinkage multiplier for the residual interaction."""
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("residual scale must be finite and nonnegative")
        with torch.no_grad():
            self.residual_scale.fill_(scale)

    def prognostic_logit(self, c: torch.Tensor) -> torch.Tensor:
        """Return the direct context-only logit ``alpha + C gamma``."""
        logit: torch.Tensor = self.context_main(c).squeeze(dim=1)
        return logit

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``alpha + C gamma + X [delta + D.T z(C)]``."""
        beta, mu = self._params(c)
        return mu + (x * beta).sum(dim=1)

    def initialize_from_logistic(
        self,
        context_coef: FloatArray,
        treatment_coef: FloatArray,
        intercept: float,
        *,
        freeze: bool = False,
    ) -> None:
        """Initialize the exact additive component from same-scale logistic coefficients."""
        context_array = np.asarray(context_coef, dtype=np.float64)
        treatment_array = np.asarray(treatment_coef, dtype=np.float64)
        if context_array.shape != (self.context_dim,):
            raise ValueError(
                f"context_coef must have shape ({self.context_dim},), got {context_array.shape}"
            )
        if treatment_array.shape != (self.x_dim,):
            raise ValueError(
                f"treatment_coef must have shape ({self.x_dim},), got {treatment_array.shape}"
            )
        if not (
            np.isfinite(context_array).all()
            and np.isfinite(treatment_array).all()
            and np.isfinite(intercept)
        ):
            raise ValueError("Baseline coefficients and intercept must all be finite")

        with torch.no_grad():
            self.context_main.weight.copy_(
                self.context_main.weight.new_tensor(context_array).unsqueeze(0)
            )
            self.context_main.bias.copy_(self.context_main.bias.new_tensor([intercept]))
            self.treatment_main.copy_(self.treatment_main.new_tensor(treatment_array))
            self.deviation_coordinates.zero_()
            self.residual_scale.fill_(1.0)

        requires_grad = not freeze
        self.context_main.requires_grad_(requires_grad)
        self.treatment_main.requires_grad_(requires_grad)

    @torch.no_grad()
    def predict_beta(self, c: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return fitted regimen coefficients ``beta(C)`` and direct prognosis ``mu(C)``."""
        beta, mu = self._params(torch.tensor(c, dtype=torch.float32))
        return (
            beta.cpu().numpy().astype(np.float64),
            mu.cpu().numpy().astype(np.float64),
        )

    @torch.no_grad()
    def predict_loadings(self, c: FloatArray) -> FloatArray:
        """Return simplex archetype loadings ``z(C)``."""
        loadings = self._loadings(torch.tensor(c, dtype=torch.float32))
        return loadings.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def prototype_coefficients(self) -> FloatArray:
        """Return the prediction-scale total regimen profiles ``delta + D_k``."""
        return self._prototype_tensor().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def homogeneous_effects(self, c: FloatArray) -> FloatArray:
        """Return the cohort mean of the context-varying regimen coefficients."""
        beta, _ = self.predict_beta(c)
        return np.asarray(beta.mean(axis=0), dtype=np.float64)

    @torch.no_grad()
    def normalized_prototypes(self) -> FloatArray:
        """Return display-only, row-normalized total prototype profiles."""
        profiles = self.prototype_coefficients()
        output = np.zeros_like(profiles)
        for archetype in range(profiles.shape[0]):
            row = profiles[archetype]
            norm = np.linalg.norm(row)
            if norm > 0:
                row = row / norm
                if row[np.argmax(np.abs(row))] < 0:
                    row = -row
            output[archetype] = row
        return output

    @torch.no_grad()
    def predict_proba(self, c: FloatArray, x: FloatArray) -> FloatArray:
        """Return positive-class probabilities."""
        logits = self.forward(
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
        )
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class ResidualTrainConfig:
    """Explicit optimization and regularization settings for residual PATCH."""

    epochs: int = 500
    lr: float = 0.01
    baseline_l2: float = 0.0
    loading_l2: float = 1e-4
    deviation_l2: float = 1e-3
    deviation_l1: float = 0.0
    entropy_weight: float = 0.0


@dataclass(frozen=True)
class ResidualLossComponents:
    """Scalar components of the residual PATCH objective."""

    bce: torch.Tensor
    baseline_l2: torch.Tensor
    loading_l2: torch.Tensor
    deviation_l2: torch.Tensor
    deviation_l1: torch.Tensor
    entropy_reward: torch.Tensor
    total: torch.Tensor


def compute_residual_loss_components(
    model: ResidualArchetypeLogistic,
    c: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    config: ResidualTrainConfig,
) -> ResidualLossComponents:
    """Evaluate BCE plus explicit baseline, loading, deviation, and entropy terms."""
    bce = nn.functional.binary_cross_entropy_with_logits(model(c, x), y)
    zero = bce.new_zeros(())
    baseline_l2 = zero
    loading_l2 = zero
    deviation_l2 = zero
    deviation_l1 = zero
    entropy_reward = zero

    if config.baseline_l2 > 0:
        # The intercept is intentionally excluded from ridge regularization.
        baseline_norm = model.context_main.weight.pow(2).sum() + model.treatment_main.pow(2).sum()
        baseline_l2 = 0.5 * config.baseline_l2 * baseline_norm
    if config.loading_l2 > 0 and model.loading_map is not None:
        loading_norm = model.loading_map.weight.pow(2).sum() + model.loading_map.bias.pow(2).sum()
        loading_l2 = 0.5 * config.loading_l2 * loading_norm
    deviations = model._deviations()
    if config.deviation_l2 > 0:
        deviation_l2 = 0.5 * config.deviation_l2 * deviations.pow(2).sum()
    if config.deviation_l1 > 0:
        deviation_l1 = config.deviation_l1 * deviations.abs().mean()
    if config.entropy_weight > 0:
        mean_loading = model._loadings(c).mean(dim=0)
        entropy = -(mean_loading * torch.log(mean_loading + 1e-9)).sum()
        entropy_reward = -config.entropy_weight * entropy

    total = bce + baseline_l2 + loading_l2 + deviation_l2 + deviation_l1 + entropy_reward
    return ResidualLossComponents(
        bce=bce,
        baseline_l2=baseline_l2,
        loading_l2=loading_l2,
        deviation_l2=deviation_l2,
        deviation_l1=deviation_l1,
        entropy_reward=entropy_reward,
        total=total,
    )


def train_residual(
    model: ResidualArchetypeLogistic,
    c: FloatArray,
    x: FloatArray,
    y: NDArray[np.floating[Any]],
    config: ResidualTrainConfig | None = None,
    seed: int = SEED,
) -> ResidualArchetypeLogistic:
    """Fit residual PATCH by full-batch Adam using only explicit objective penalties."""
    config = config or ResidualTrainConfig()
    torch.manual_seed(seed)
    ct = torch.tensor(c, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Residual PATCH has no trainable parameters")
    optimizer = Adam(trainable, lr=config.lr, weight_decay=0.0)
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = compute_residual_loss_components(model, ct, xt, yt, config).total
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    logger.info(
        "Trained residual PATCH (K=%d, final loss=%.4f)",
        model.n_archetypes,
        float(loss.detach()),
    )
    return model
