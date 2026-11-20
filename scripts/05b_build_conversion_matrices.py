"""Build the culture-conversion modelling matrices (X, C, y) for the contextualized analysis.

Assembles the design used by the contextualized treatment-effect figures (scripts 06/07/09):

* **X** — binary initial-regimen drug matrix (one column per drug at >=1% prevalence).
* **C** — encoded context matrix: numeric features median-imputed + standardized; categorical
  features one-hot encoded with an explicit ``<NA>`` category. Source features (demographics,
  comorbidities, imaging, per-drug DST, lineage, socioeconomic/region, baseline labs) come from
  :func:`ctb.data.context.build_context_matrix`.
* **y** — binary culture-conversion event (1 = sustained conversion ever observed, 0 = censored).

Cohort = the conversion-estimable population (>=1 baseline-positive culture) intersected with the
CONSORT analysis cohort (:func:`ctb.data.clean.build_cohort`). Writes:

* ``data/processed/conversion_model_matrices.npz`` — ``X``, ``C``, ``y``, ``drugs``,
  ``cond_ids``, ``context_names``.
* ``data/processed/conversion_context_meta.parquet`` — per-condition metadata (outcome + raw
  interpretable context columns) used for subtype characterization and heterogeneity strata.

Only aggregate shapes are logged; no row-level data is emitted.

Usage
-----
    uv run python scripts/05b_build_conversion_matrices.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ctb.config import PROCESSED_DIR, ensure_output_dirs
from ctb.data.clean import build_cohort, clean_patient_cases, clean_regimens
from ctb.data.context import FeatureSpec, build_context_matrix
from ctb.data.conversion import derive_conversion
from ctb.data.load import load_biochemistry, load_dst, load_patient_cases, load_regimens
from ctb.data.treatments import build_treatment_matrix
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

# Raw interpretable context columns carried into the metadata frame (subtype / strata analysis).
_META_CONTEXT_COLS = [
    "type_of_resistance",
    "baseline_smear",
    "age_of_onset",
    "country",
    "comorb_hiv",
    "comorb_anemia",
    "comorb_diabetes",
    "comorb_covid",
    "comorb_psychiatric",
    "comorb_renal",
    "comorb_hepatic",
]


def encode_context(context: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Encode the raw context matrix for modelling.

    Numeric features are median-imputed (all-missing columns dropped) and standardized; categorical
    features are one-hot encoded with an explicit ``<NA>`` category (``dummy_na=True``).

    Parameters
    ----------
    context : pandas.DataFrame
        Raw (un-imputed) context matrix from :func:`build_context_matrix`.
    spec : FeatureSpec
        Numeric/categorical column metadata.

    Returns
    -------
    pandas.DataFrame
        Dense float design matrix indexed like ``context``.
    """
    num = context[spec.numeric].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in spec.numeric if num[c].notna().any()]
    dropped = [c for c in spec.numeric if c not in keep]
    if dropped:
        logger.info("Dropping all-missing numeric features: %s", dropped)
    imputed = SimpleImputer(strategy="median").fit_transform(num[keep])
    num_std = pd.DataFrame(
        StandardScaler().fit_transform(imputed), columns=keep, index=context.index
    )
    cat = pd.get_dummies(context[spec.categorical].astype("string"), dummy_na=True)
    return pd.concat([num_std, cat], axis=1)


def main() -> int:
    """Assemble and persist the conversion modelling matrices and metadata.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    pc = clean_patient_cases(load_patient_cases())
    reg = clean_regimens(load_regimens())
    bx = load_biochemistry()
    dst = load_dst()

    # Conversion-estimable population (>=1 baseline-positive culture) within the CONSORT cohort.
    conv = derive_conversion(dst)
    baseline_pos = set(conv[conv["first_pos_day"].notna()].index)
    cohort = build_cohort(pc, reg)
    ids = [cid for cid in cohort.condition_ids if cid in baseline_pos]
    logger.info("Conversion cohort: %d conditions (baseline-positive ∩ CONSORT cohort)", len(ids))

    idx: pd.Index[Any] = pd.Index(ids, name="condition_id")
    x = build_treatment_matrix(reg, ids).reindex(idx)
    c_raw, spec = build_context_matrix(pc, bx, ids)
    c_raw = c_raw.reindex(idx)
    c_enc = encode_context(c_raw, spec)
    y = conv["event"].reindex(idx).astype(int)

    meta = pd.DataFrame(index=idx)
    meta["event"] = y
    meta["time_to_event"] = conv["time_to_event"].reindex(idx)
    for col in _META_CONTEXT_COLS:
        if col in c_raw.columns:
            meta[col] = c_raw[col].to_numpy()
        else:
            logger.warning("Meta context column %r absent from context matrix; skipping", col)

    np.savez(
        PROCESSED_DIR / "conversion_model_matrices.npz",
        X=x.to_numpy(dtype=np.float64),
        C=c_enc.to_numpy(dtype=np.float64),
        y=y.to_numpy(dtype=np.int64),
        drugs=np.asarray(list(x.columns), dtype=object),
        cond_ids=np.asarray(ids, dtype=object),
        context_names=np.asarray(list(c_enc.columns), dtype=object),
    )
    meta.to_parquet(PROCESSED_DIR / "conversion_context_meta.parquet")

    logger.info(
        "Wrote conversion matrices: X=%s, C=%s (%d context features), y mean=%.3f",
        x.shape,
        c_enc.shape,
        c_enc.shape[1],
        float(y.mean()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
