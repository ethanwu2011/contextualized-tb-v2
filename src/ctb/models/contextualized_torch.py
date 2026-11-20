"""From-scratch PyTorch contextualized regression — full control, no package dependency.

Implements the Lengerich-Ellington-Xing decomposition directly:

``logodds(Y | X, C) = X . beta(C) + mu(C)``

where a context encoder maps context ``C`` to the per-patient parameters ``(beta(C), mu(C))`` of a
logistic model operating on predictors ``X``. Three encoder classes are provided, spanning the
interpretability/flexibility axis:

* ``"linear"`` — ``beta(C) = W.C + b``. This instance is **algebraically identical to an interaction
  GLM**: the term ``X . (W.C)`` expands to ``sum_jk W_jk (X_j C_k)``, i.e. pairwise X-by-C
  interaction terms. Use it when the confirmatory question is a small set of pre-specified linear
  interactions (then the closed-form GLM in :mod:`statsmodels` gives exact inference and should be
  preferred for p-values).
* ``"mlp"`` — ``beta(C) = MLP(C)``. Captures **nonlinear** and **higher-order** context interactions
  that the linear/GLM form cannot represent. This is where the framework does work a GLM cannot.
* ``"archetype"`` — ``beta(C) = sum_k Z_k(C) A_k`` with ``Z(C)`` an encoder output and ``A_k``
  learned shared archetypes (the subtype bottleneck). Yields patient subtypes via loadings ``Z``.

The forward pass, training loop, per-patient parameter extraction, and (for the archetype encoder)
archetype/loading extraction are all explicit and inspectable. Determinism is controlled by
:data:`ctb.config.SEED`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ctb.config import SEED

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class _Encoder(nn.Module):
    """Context encoder mapping ``C`` to a flat parameter vector of length ``output_dim``."""

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        kind: str = "linear",
        width: int = 64,
        layers: int = 2,
    ) -> None:
        """Build the encoder.

        Parameters
        ----------
        context_dim : int
            Number of context features.
        output_dim : int
            Size of the parameter vector to emit (``x_dim + 1`` for beta + intercept, or
            ``num_archetypes`` for the archetype encoder).
        kind : str
            ``"linear"`` or ``"mlp"``.
        width, layers : int
            Hidden width and hidden-layer count for the MLP.
        """
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(context_dim, output_dim)
        elif kind == "mlp":
            mods: list[nn.Module] = [nn.Linear(context_dim, width), nn.ReLU()]
            for _ in range(layers - 1):
                mods += [nn.Linear(width, width), nn.ReLU()]
            mods.append(nn.Linear(width, output_dim))
            self.net = nn.Sequential(*mods)
        else:
            raise ValueError(f"Unknown encoder kind {kind!r}")

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Map context to the flat parameter vector."""
        out: torch.Tensor = self.net(c)
        return out


class ContextualizedLogisticRegression(nn.Module):
    """Contextualized logistic regression with a selectable context encoder.

    Parameters
    ----------
    context_dim : int
        Number of context features.
    x_dim : int
        Number of treatment predictors.
    encoder : str
        ``"linear"``, ``"mlp"``, or ``"archetype"``.
    width, layers : int
        MLP hidden width / depth (used by ``"mlp"`` and the archetype encoder).
    num_archetypes : int
        Number of archetypes (used only by ``encoder="archetype"``).
    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        encoder: str = "linear",
        width: int = 64,
        layers: int = 2,
        num_archetypes: int = 8,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder_kind = encoder
        out = x_dim + 1  # beta (x_dim) + intercept mu (1)
        if encoder == "archetype":
            self.subtype_encoder = _Encoder(context_dim, num_archetypes, "mlp", width, layers)
            # archetypes: (num_archetypes, x_dim + 1)
            self.archetypes = nn.Parameter(torch.randn(num_archetypes, out) * 0.01)
        else:
            self.encoder = _Encoder(context_dim, out, encoder, width, layers)

    def _params(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(beta(C), mu(C))`` for a context batch."""
        if self.encoder_kind == "archetype":
            z = self.subtype_encoder(c)  # (n, K)
            p = z @ self.archetypes  # (n, x_dim + 1)
        else:
            p = self.encoder(c)
        return p[:, : self.x_dim], p[:, self.x_dim]

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the logit ``X . beta(C) + mu(C)``."""
        beta, mu = self._params(c)
        return (x * beta).sum(dim=1) + mu

    @torch.no_grad()
    def predict_beta(self, c: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return per-patient ``(beta(C), mu(C))`` as numpy arrays.

        Parameters
        ----------
        c : numpy.ndarray
            Dense context matrix ``(n, context_dim)``.

        Returns
        -------
        tuple of numpy.ndarray
            ``beta`` of shape ``(n, x_dim)`` and ``mu`` of shape ``(n,)``.
        """
        ct = torch.tensor(c, dtype=torch.float32)
        beta, mu = self._params(ct)
        return beta.cpu().numpy().astype(np.float64), mu.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_proba(self, c: FloatArray, x: FloatArray) -> FloatArray:
        """Return positive-class probabilities for ``(C, X)``."""
        logit = self.forward(
            torch.tensor(c, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
        )
        return torch.sigmoid(logit).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def get_archetypes(self) -> FloatArray:
        """Return the archetype coefficient matrix ``(K, x_dim + 1)`` (archetype encoder only).

        Returns
        -------
        numpy.ndarray
            Archetype coefficients; last column is the intercept component.

        Raises
        ------
        AttributeError
            If the encoder is not ``"archetype"``.
        """
        if self.encoder_kind != "archetype":
            raise AttributeError("Archetypes only exist for encoder='archetype'.")
        return self.archetypes.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_loadings(self, c: FloatArray) -> FloatArray:
        """Return per-patient archetype loadings ``Z(C)`` ``(n, K)`` (archetype encoder only).

        Parameters
        ----------
        c : numpy.ndarray
            Dense context matrix.

        Returns
        -------
        numpy.ndarray
            Loadings ``(n, num_archetypes)``.

        Raises
        ------
        AttributeError
            If the encoder is not ``"archetype"``.
        """
        if self.encoder_kind != "archetype":
            raise AttributeError("Loadings only exist for encoder='archetype'.")
        z = self.subtype_encoder(torch.tensor(c, dtype=torch.float32))
        return np.asarray(z.cpu().numpy(), dtype=np.float64)


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
        L2 regularization on encoder weights.
    """

    epochs: int = 2000
    lr: float = 0.05
    weight_decay: float = 0.0


def train(
    model: ContextualizedLogisticRegression,
    c: FloatArray,
    x: FloatArray,
    y: FloatArray,
    config: TrainConfig | None = None,
    seed: int = SEED,
) -> ContextualizedLogisticRegression:
    """Train a contextualized model by full-batch Adam on binary cross-entropy.

    Parameters
    ----------
    model : ContextualizedLogisticRegression
        The model to train (modified in place and returned).
    c, x, y : numpy.ndarray
        Dense context ``(n, context_dim)``, treatments ``(n, x_dim)``, binary outcome ``(n,)``.
    config : TrainConfig or None
        Training hyperparameters (defaults used if ``None``).
    seed : int, default :data:`ctb.config.SEED`
        Torch manual seed for reproducibility.

    Returns
    -------
    ContextualizedLogisticRegression
        The trained model.
    """
    config = config or TrainConfig()
    torch.manual_seed(seed)
    ct = torch.tensor(c, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(config.epochs):
        opt.zero_grad()
        loss = loss_fn(model(ct, xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    logger.info(
        "Trained contextualized model (encoder=%s, final BCE=%.4f)",
        model.encoder_kind,
        float(loss.detach()),
    )
    return model
