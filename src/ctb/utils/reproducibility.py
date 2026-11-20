"""Deterministic seeding utilities.

Sets seeds for the Python ``random`` module, NumPy, and the ``PYTHONHASHSEED`` environment
variable so that every stage of the pipeline is reproducible. The active seed is logged.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

from ctb.config import SEED

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = SEED) -> int:
    """Seed all relevant random number generators for reproducibility.

    Parameters
    ----------
    seed : int, default :data:`ctb.config.SEED`
        The seed applied to :mod:`random`, :mod:`numpy`, and ``PYTHONHASHSEED``.

    Returns
    -------
    int
        The seed that was set (echoed for logging/record-keeping).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global seed set to %d", seed)
    return seed


def new_rng(seed: int = SEED) -> np.random.Generator:
    """Return a fresh NumPy random generator with a fixed seed.

    Prefer this over the legacy global NumPy state when a local, isolated generator is needed
    (e.g. bootstrap resampling), so that call order elsewhere cannot perturb results.

    Parameters
    ----------
    seed : int, default :data:`ctb.config.SEED`
        Seed for the generator.

    Returns
    -------
    numpy.random.Generator
        A seeded PCG64 generator.
    """
    return np.random.default_rng(seed)
