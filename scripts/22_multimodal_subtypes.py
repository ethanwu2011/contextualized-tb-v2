"""Multimodal treatment-response subtypes + modality-attribution (the paper's headline analysis).

Fits the balanced K=5 contextualized model on the FULL multimodal context (resistance, comorbidity,
bacillary burden, imaging, socioeconomic, region, genomics) and shows that the treatment-response
subtypes are *dominated by different modalities* — resistance is one axis among several. Quantifies
each modality's contribution to the subtype structure by permutation importance (drop in subtype
agreement when a modality's features are shuffled). Saves subtype assignments + loadings for the
modality-coloured UMAP.

Unlike outcome-prediction multimodal work (e.g. Sambarey 2024), the subtypes here are defined by how
patients *respond* to treatment (per-patient drug effects beta(C)), not by risk-factor profile.

Usage
-----
    uv run python scripts/22_multimodal_subtypes.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ctb.config import PROCESSED_DIR, TABLES_DIR, ensure_output_dirs
from ctb.data.artifacts import PRIMARY_ENDPOINT, load_primary_matrix
from ctb.data.clean import clean_patient_cases
from ctb.data.context import build_context_matrix
from ctb.data.load import load_biochemistry, load_patient_cases
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

K = 5
EPOCHS = 2000
CFG = TrainConfig(epochs=EPOCHS, lr=0.01, weight_decay=1e-3, archetype_l1=5e-4,
                  entropy_weight=1.0)
SEED = 0


def _modality_of(name: str) -> str:
    """Map an encoded context column name to its modality group."""
    if name.startswith("type_of_resistance") or name.startswith("bactec_"):
        return "resistance"
    if name.startswith("comorb_"):
        return "comorbidity"
    if name.startswith(("baseline_smear", "cavitation", "case_definition", "lung_localization")):
        return "burden_clinical"
    if name.startswith(("dissemination", "affect_pleura", "shadow_pattern")):
        return "imaging"
    if name == "timika_score":
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
    """Fit multimodal balanced subtypes, characterize, and attribute modalities."""
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    data = load_primary_matrix()
    c = data["C"].astype(np.float64)
    x = data["X"].astype(np.float64)
    names = list(data["context_names"])
    cond_ids = list(data["cond_ids"])

    pc = clean_patient_cases(load_patient_cases())
    bx = load_biochemistry()
    craw, _ = build_context_matrix(pc, bx, cond_ids)
    craw = craw.reindex(pd.Index(cond_ids, name="condition_id")).reset_index(drop=True)
    y = data["y"].astype(np.float64)

    model = LinearArchetypeContextualized(
        c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=SEED
    )
    train(model, c, x, y, CFG, seed=SEED)
    sub = model.predict_loadings(c).argmax(axis=1)
    loadings = model.predict_loadings(c)
    np.savez(PROCESSED_DIR / "multimodal_subtypes.npz", subtypes=sub, loadings=loadings,
             cond_ids=np.asarray(cond_ids, dtype=object),
             outcome_name=np.asarray(PRIMARY_ENDPOINT))
    logger.info("Subtype sizes: %s", np.bincount(sub, minlength=K).tolist())

    # --- characterize each subtype across ALL modalities ---
    age = pd.to_numeric(craw["age_of_onset"], errors="coerce")
    feats = {
        "unfavorable%": y * 100,
        "age": age.to_numpy(),
        "MDR%": (craw["type_of_resistance"] == "MDR non XDR").to_numpy() * 100,
        "XDR%": (craw["type_of_resistance"] == "XDR").to_numpy() * 100,
        "sensitive%": (craw["type_of_resistance"] == "Sensitive").to_numpy() * 100,
        "HIV%": (craw["comorb_hiv"] == 1).to_numpy() * 100,
        "diabetes%": (craw["comorb_diabetes"] == 1).to_numpy() * 100,
        "smear+%": (craw["baseline_smear"] == "positive").to_numpy() * 100,
        "SSAfrica%": craw["country"].isin(["Nigeria", "South Africa"]).to_numpy() * 100,
        "Ukraine%": (craw["country"] == "Ukraine").to_numpy() * 100,
    }
    rows = []
    for s in range(K):
        m = sub == s
        row = {"subtype": f"S{s}", "n": int(m.sum())}
        for fn, fv in feats.items():
            row[fn] = float(np.nanmean(np.asarray(fv, float)[m]))
        rows.append(row)
    char = pd.DataFrame(rows).set_index("subtype")
    char.to_csv(TABLES_DIR / "multimodal_subtype_characterization.csv")
    logger.info("Multimodal subtype characterization:\n%s", char.round(1).to_string())

    # --- modality attribution: permutation importance on subtype assignment ---
    mod_cols: dict[str, list[int]] = {}
    for j, nm in enumerate(names):
        mod_cols.setdefault(_modality_of(nm), []).append(j)
    rng = np.random.default_rng(SEED)
    attr = []
    for mod, cols in mod_cols.items():
        drops = []
        for _ in range(5):
            cperm = c.copy()
            perm = rng.permutation(c.shape[0])
            cperm[:, cols] = c[perm][:, cols]
            sub_p = model.predict_loadings(cperm).argmax(axis=1)
            drops.append(1 - adjusted_rand_score(sub, sub_p))  # 1-ARI = disruption
        attr.append({"modality": mod, "n_features": len(cols),
                     "subtype_disruption": float(np.mean(drops))})
    attr_df = pd.DataFrame(attr)
    # Total share (aggregate importance) AND per-feature share (importance / n_features), so the
    # feature-count confound is transparent: resistance has many sparse columns, so its total share
    # overstates per-feature contribution. Both are reported.
    attr_df["total_share"] = attr_df["subtype_disruption"] / attr_df["subtype_disruption"].sum()
    per_feat = attr_df["subtype_disruption"] / attr_df["n_features"]
    attr_df["per_feature_share"] = per_feat / per_feat.sum()
    attr_df = attr_df.sort_values("total_share", ascending=False)
    attr_df.to_csv(TABLES_DIR / "multimodal_modality_attribution.csv", index=False)
    logger.info("Modality attribution (total vs per-feature):\n%s",
                attr_df.round(3).to_string(index=False))

    # --- canonical model reproducibility: cross-seed ARI on THIS configuration ---
    seed_subs = []
    for s2 in range(6):
        m2 = LinearArchetypeContextualized(
            c.shape[1], x.shape[1], n_archetypes=K, simplex=True, seed=s2
        )
        train(m2, c, x, y, CFG, seed=s2)
        seed_subs.append(m2.predict_loadings(c).argmax(axis=1))
    aris = [adjusted_rand_score(seed_subs[i], seed_subs[j])
            for i in range(6) for j in range(i + 1, 6)]
    n_pop = [int((np.bincount(s, minlength=K) / len(s) >= 0.03).sum()) for s in seed_subs]
    logger.info("Canonical reproducibility: mean cross-seed ARI=%.3f (min %.3f); subtypes>=3%%: %s",
                float(np.mean(aris)), float(np.min(aris)), n_pop)
    pd.DataFrame([{"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris)),
                   "min_subtypes_populated": int(min(n_pop))}]).to_csv(
        TABLES_DIR / "multimodal_subtype_stability.csv", index=False)

    # --- assign each subtype its dominant modality (max standardized enrichment) ---
    z = char.drop(columns="n")
    zz = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    mod_map = {"MDR%": "resistance", "XDR%": "resistance", "sensitive%": "resistance",
               "HIV%": "comorbidity", "diabetes%": "comorbidity", "smear+%": "burden",
               "SSAfrica%": "region", "Ukraine%": "region", "age": "sociodemographic"}
    dom = {}
    for s in zz.index:
        contrib: dict[str, float] = {}
        for col, mod in mod_map.items():
            contrib[mod] = contrib.get(mod, 0) + abs(zz.loc[s, col])
        dom[s] = max(contrib, key=lambda k: contrib[k])
    logger.info("Dominant modality per subtype: %s", dom)
    pd.Series(dom, name="dominant_modality").to_csv(TABLES_DIR / "multimodal_subtype_dominant.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
