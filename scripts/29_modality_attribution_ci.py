"""Modality attribution with confidence intervals.

The headline modality-attribution figure (Fig 2c) previously reported point estimates from 5
permutations. This recomputes the permutation importance with B=50 permutation rounds and reports the
mean and 95% percentile interval of each modality's total share and per-feature share, so the figure
can show uncertainty and the per-feature ordering can be read with its instability visible. Writes an
aggregate table only.

Usage
-----
    uv run python scripts/29_modality_attribution_ci.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import load_primary_matrix
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
SEED = 0
B = 50  # permutation rounds
CFG = TrainConfig(epochs=2000, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4, entropy_weight=1.0)


def _modality_of(name: str) -> str:
    """Map an encoded context column name to its modality group (identical to script 22)."""
    if name.startswith("type_of_resistance") or name.startswith("bactec_"):
        return "resistance"
    if name.startswith("comorb_"):
        return "comorbidity"
    if name.startswith(("baseline_smear", "cavitation", "case_definition", "lung_localization")):
        return "burden_clinical"
    if name.startswith(("dissemination", "affect_pleura", "shadow_pattern")) or name == "timika_score":
        return "imaging"
    if name.startswith(("gender", "education", "employment")) or name in {"age_of_onset", "bmi"}:
        return "sociodemographic"
    if name.startswith("country"):
        return "region"
    if name.startswith(("main_lineage", "sub_lineage")):
        return "genomics"
    if name.startswith("lab_"):
        return "labs"
    return "other"


def main() -> int:
    """Recompute modality attribution with bootstrap-style permutation CIs."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    names = list(data["context_names"])
    y = data["y"].astype(np.float64)

    model = LinearArchetypeContextualized(c.shape[1], x.shape[1], n_archetypes=K, simplex=True,
                                          seed=SEED)
    train(model, c, x, y, CFG, seed=SEED)
    sub = model.predict_loadings(c).argmax(axis=1)

    mod_cols: dict[str, list[int]] = {}
    for j, nm in enumerate(names):
        mod_cols.setdefault(_modality_of(nm), []).append(j)
    mods = list(mod_cols)
    rng = np.random.default_rng(SEED)

    # B rounds: each round permutes every modality once and records its disruption (1-ARI); from the
    # round-wise disruption vectors we form the share series, so CIs reflect both permutation noise and
    # the coupling between modalities' shares.
    disrupt = np.zeros((B, len(mods)))
    for b in range(B):
        for mi, mod in enumerate(mods):
            cperm = c.copy()
            perm = rng.permutation(c.shape[0])
            cperm[:, mod_cols[mod]] = c[perm][:, mod_cols[mod]]
            sub_p = model.predict_loadings(cperm).argmax(axis=1)
            disrupt[b, mi] = 1 - adjusted_rand_score(sub, sub_p)

    nfeat = np.array([len(mod_cols[m]) for m in mods], dtype=float)
    total_share = disrupt / disrupt.sum(axis=1, keepdims=True)
    perfeat = disrupt / nfeat
    perfeat_share = perfeat / perfeat.sum(axis=1, keepdims=True)

    rows = []
    for mi, mod in enumerate(mods):
        rows.append({
            "modality": mod, "n_features": int(nfeat[mi]),
            "total_share": round(float(total_share[:, mi].mean()), 4),
            "total_lo": round(float(np.percentile(total_share[:, mi], 2.5)), 4),
            "total_hi": round(float(np.percentile(total_share[:, mi], 97.5)), 4),
            "per_feature_share": round(float(perfeat_share[:, mi].mean()), 4),
            "perfeat_lo": round(float(np.percentile(perfeat_share[:, mi], 2.5)), 4),
            "perfeat_hi": round(float(np.percentile(perfeat_share[:, mi], 97.5)), 4),
        })
    out = pd.DataFrame(rows).sort_values("total_share", ascending=False)
    out.to_csv(TABLES_DIR / "multimodal_modality_attribution_ci.csv", index=False)
    logger.info("Modality attribution with 95%% CIs (B=%d permutations):\n%s", B,
                out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
