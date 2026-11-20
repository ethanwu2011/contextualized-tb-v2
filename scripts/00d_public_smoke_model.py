"""Run a small synthetic-only PATCH model smoke test.

This script consumes only ``data/processed/unfavorable_model_matrices.npz``. The companion
``reproduce_synthetic.py`` script creates that matrix from the guarded synthetic generator, so this
check never needs the access-gated TB Portals export. It writes aggregate metadata only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from ctb.config import METRICS_DIR, PROCESSED_DIR, SEED, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging

logger = logging.getLogger(__name__)

MATRIX_PATH = PROCESSED_DIR / "unfavorable_model_matrices.npz"
METRIC_PATH = METRICS_DIR / "public_synthetic_smoke.json"
ENDPOINT_NAME = "died_or_failure"


def run_smoke(epochs: int = 50, seed: int = SEED) -> dict[str, object]:
    """Fit PATCH on the synthetic endpoint and return disclosure-safe aggregates."""
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"{MATRIX_PATH} is missing. Run scripts/reproduce_synthetic.py first."
        )
    with load_primary_matrix(MATRIX_PATH) as data:
        x = np.asarray(data["X"], dtype=np.float64)
        context = np.asarray(data["C"], dtype=np.float64)
        outcome = np.asarray(data["y"], dtype=np.int64).reshape(-1)
        drugs = [str(value) for value in data["drugs"].tolist()]
        context_names = [str(value) for value in data["context_names"].tolist()]
        endpoint = str(np.asarray(data["outcome_name"]).item())
    if x.ndim != 2 or context.ndim != 2 or len(outcome) != len(x) or len(x) != len(context):
        raise ValueError("Synthetic matrix dimensions are inconsistent.")
    if len(outcome) < 20 or outcome.min() == outcome.max():
        raise ValueError("Synthetic smoke cohort must contain at least two outcome classes.")
    if not np.isfinite(x).all() or not np.isfinite(context).all():
        raise ValueError("Synthetic matrix contains non-finite values after preprocessing.")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    model = LinearArchetypeContextualized(
        context_dim=context.shape[1],
        x_dim=x.shape[1],
        n_archetypes=min(5, max(2, x.shape[1])),
        simplex=True,
        seed=seed,
    )
    train(
        model,
        context,
        x,
        outcome.astype(np.float64),
        TrainConfig(epochs=epochs, lr=0.02, weight_decay=1e-3, archetype_l1=5e-4,
                    entropy_weight=0.5),
        seed=seed,
    )
    probability = model.predict_proba(context, x)
    loadings = model.predict_loadings(context)
    aggregate = {
        "synthetic_only": True,
        "endpoint": endpoint,
        "n_conditions": int(len(outcome)),
        "n_positive": int(outcome.sum()),
        "n_drugs": int(len(drugs)),
        "n_context_features": int(len(context_names)),
        "n_archetypes": int(model.n_archetypes),
        "epochs": int(epochs),
        "seed": int(seed),
        "auroc": float(roc_auc_score(outcome, probability)),
        "min_archetype_usage": float(loadings.mean(axis=0).min()),
        "max_archetype_usage": float(loadings.mean(axis=0).max()),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> int:
    """Run the public synthetic smoke test."""
    parser = argparse.ArgumentParser(description="Fit a small PATCH model on synthetic data only.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    configure_logging()
    ensure_output_dirs()
    aggregate = run_smoke(epochs=args.epochs, seed=args.seed)
    logger.info(
        "Synthetic smoke passed: n=%d, positive=%d, endpoint=%s, AUROC=%.3f",
        aggregate["n_conditions"],
        aggregate["n_positive"],
        aggregate["endpoint"],
        aggregate["auroc"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
