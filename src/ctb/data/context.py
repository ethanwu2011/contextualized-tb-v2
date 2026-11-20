"""Construction of the context matrix ``C`` (patient covariates).

Builds demographic, comorbidity, clinical-baseline, imaging, socioeconomic, and baseline-laboratory
features from the Patient Cases and Biochemistry tables. Features are returned *un-imputed* (NaNs
preserved); imputation is performed inside the modelling cross-validation pipeline so that it is fit
on training folds only and cannot leak. A :class:`FeatureSpec` records which columns are numeric vs
categorical for the downstream preprocessor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Interpretable-context-mask construction (shared by the heterogeneity analyses).
MASK_MISSING_LEVELS = {"<NA>", "nan", "None", "", "Not Reported", "Unknown", "Unknown result"}
MASK_MIN_LEVEL_N = 30  # minimum patients for a context level / numeric split to be tested

# Comorbidity tokens parsed from the free-text ``comorbidity`` column -> binary flag column names.
_COMORBIDITY_TOKENS: dict[str, str] = {
    "hiv": "comorb_hiv",
    "diabetes": "comorb_diabetes",
    "hepatitis c": "comorb_hepatitis_c",
    "hepatitis b": "comorb_hepatitis_b",
    "anemia": "comorb_anemia",
    "covid": "comorb_covid",  # matches both "covid-19" and "post-covid-19"
    "psychiatric": "comorb_psychiatric",
    "renal disease": "comorb_renal",
    "hepatic disease": "comorb_hepatic",  # "hepatic" is not a substring of "hepatitis"
}

# Baseline laboratory columns to extract (earliest non-null value per condition).
_LAB_COLS: dict[str, str] = {
    "Alanine Aminotransferase (u/l)": "lab_alt",
    "Aspartate Aminotransferase (u/l)": "lab_ast",
    "Creatinine (umol/l)": "lab_creatinine",
    "Glucose (mmol/l)": "lab_glucose",
    "Albumin (g/l)": "lab_albumin",
    "Hemoglobin (g/l)": "lab_hemoglobin",
}

# Context features dropped after construction. These fields are zero-information in the locked
# August 2023 analysis cohort. ``lab_hemoglobin`` (92% missing) is also removed because an ablation
# confirmed that the five treatment-response subtypes survive its removal. Keep this list explicit
# so small synthetic fixtures can still exercise otherwise constant laboratory fields.
_EXCLUDE_FEATURES: frozenset[str] = frozenset(
    {
        "lab_hemoglobin",
        "totalcavernum",
        "overall_percent_of_abnormal_volume",
        "pleural_effusion_percent_of_hemithorax_involved",
        "baseline_culture",
        "bactec_mycobutin",
        "bactec_imipenem_cilastatin",
    }
)


@dataclass
class FeatureSpec:
    """Column-type metadata for the context matrix.

    Attributes
    ----------
    numeric : list of str
        Numeric feature column names (imputed by IterativeImputer downstream).
    categorical : list of str
        Categorical feature column names (one-hot encoded with an explicit missing category).
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def all_columns(self) -> list[str]:
        """Return all feature columns (numeric followed by categorical)."""
        return [*self.numeric, *self.categorical]


def fit_context_encoder(context: pd.DataFrame, spec: FeatureSpec) -> dict[str, Any]:
    """Fit numeric and categorical transforms on one declared training split.

    The returned object contains only preprocessing state and column names. Keeping this operation
    separate from :func:`build_context_matrix` lets temporal and country holdouts fit every
    imputer, scaler, and category vocabulary on training rows only.

    Parameters
    ----------
    context : pandas.DataFrame
        Un-imputed context rows used to fit the transform.
    spec : FeatureSpec
        Numeric and categorical columns to encode.

    Returns
    -------
    dict
        Serializable-by-convention preprocessing state used by
        :func:`apply_context_encoder`.
    """
    numeric = context[spec.numeric].apply(pd.to_numeric, errors="coerce")
    numeric_columns = [col for col in spec.numeric if numeric[col].notna().any()]
    imputer = SimpleImputer(strategy="median") if numeric_columns else None
    scaler = StandardScaler() if numeric_columns else None
    if imputer is not None and scaler is not None:
        scaler.fit(imputer.fit_transform(numeric[numeric_columns]))
    categorical = pd.get_dummies(context[spec.categorical].astype("string"), dummy_na=True)
    return {
        "numeric_columns": numeric_columns,
        "imputer": imputer,
        "scaler": scaler,
        "categorical_columns": list(categorical.columns),
    }


def apply_context_encoder(
    context: pd.DataFrame, spec: FeatureSpec, encoder: dict[str, Any]
) -> pd.DataFrame:
    """Apply a training-fitted context encoder and return a stable numeric DataFrame.

    Unseen categorical levels in held-out rows are dropped. Missing levels are represented by the
    training vocabulary and therefore cannot add columns after the split.

    Parameters
    ----------
    context : pandas.DataFrame
        Raw context rows to transform.
    spec : FeatureSpec
        Numeric and categorical columns used when fitting the encoder.
    encoder : dict
        Object returned by :func:`fit_context_encoder`.

    Returns
    -------
    pandas.DataFrame
        Numeric encoded matrix with training-derived column order.
    """
    numeric_columns = list(encoder["numeric_columns"])
    if numeric_columns:
        numeric = context[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_array = encoder["scaler"].transform(encoder["imputer"].transform(numeric))
    else:
        numeric_array = np.empty((len(context), 0), dtype=float)
    categorical_columns = list(encoder["categorical_columns"])
    categorical = pd.get_dummies(context[spec.categorical].astype("string"), dummy_na=True)
    categorical = categorical.reindex(columns=categorical_columns, fill_value=0)
    encoded = np.hstack([numeric_array, categorical.to_numpy(dtype=float)])
    return pd.DataFrame(
        encoded,
        index=context.index,
        columns=[*numeric_columns, *categorical_columns],
    )


def parse_comorbidity_flags(comorbidity: pd.Series[Any]) -> pd.DataFrame:
    """Parse the free-text comorbidity column into binary flags.

    A flag is 1 if the token appears, 0 if the field is present but the token is absent, and the
    ``comorb_missing`` indicator marks rows where the field itself is missing.

    Parameters
    ----------
    comorbidity : pandas.Series
        The raw ``comorbidity`` column (string / NaN).

    Returns
    -------
    pandas.DataFrame
        Binary flag columns (see :data:`_COMORBIDITY_TOKENS`) plus ``comorb_missing``.
    """
    lowered = comorbidity.astype("string").str.lower()
    out = pd.DataFrame(index=comorbidity.index)
    for token, col in _COMORBIDITY_TOKENS.items():
        out[col] = lowered.str.contains(token, na=False).astype(int)
    out["comorb_missing"] = comorbidity.isna().astype(int)
    return out


def _micro_status(value: object) -> str:
    """Classify a culture/microscopy free-text value as positive/negative/unknown.

    Parameters
    ----------
    value : object
        Raw cell (e.g. ``"Negative, Positive"``, ``"20 to 100"``, ``"Not Reported"``).

    Returns
    -------
    str
        One of ``"positive"``, ``"negative"``, ``"unknown"``.
    """
    if not isinstance(value, str):
        return "unknown"
    s = value.lower()
    if "positive" in s or re.search(r"\d", s):
        return "positive"
    if "negative" in s:
        return "negative"
    return "unknown"


def _cavitation(value: object) -> str:
    """Map ``lungcavity_size`` to a cavitation category.

    Parameters
    ----------
    value : object
        Raw ``lungcavity_size`` value.

    Returns
    -------
    str
        ``"yes"``, ``"no"``, or ``"unknown"``.
    """
    if not isinstance(value, str):
        return "unknown"
    s = value.lower()
    if "no cavities" in s:
        return "no"
    if "not reported" in s or s.strip() == "":
        return "unknown"
    return "yes"


def _baseline_labs(biochemistry: pd.DataFrame, condition_ids: pd.Index[Any]) -> pd.DataFrame:
    """Extract earliest-available baseline labs per condition.

    Parameters
    ----------
    biochemistry : pandas.DataFrame
        Biochemistry table with ``condition_id``, ``test_date``, and lab columns.
    condition_ids : pandas.Index
        Conditions to index the result by.

    Returns
    -------
    pandas.DataFrame
        One column per lab in :data:`_LAB_COLS`, indexed by ``condition_id`` (NaN if untested).
    """
    bx = biochemistry.copy()
    bx["test_date"] = pd.to_datetime(bx["test_date"], errors="coerce")
    bx = bx.sort_values("test_date", kind="mergesort")
    out = pd.DataFrame(index=condition_ids)
    for src, dst in _LAB_COLS.items():
        if src not in bx.columns:
            out[dst] = np.nan
            continue
        vals = pd.to_numeric(bx[src], errors="coerce")
        first_valid = bx.assign(_v=vals).dropna(subset=["_v"]).groupby("condition_id")["_v"].first()
        out[dst] = first_valid.reindex(condition_ids)
    return out


def build_context_matrix(
    patient_cases: pd.DataFrame,
    biochemistry: pd.DataFrame,
    condition_ids: list[object],
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build the un-imputed context matrix ``C`` and its feature specification.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Cleaned Patient Cases table.
    biochemistry : pandas.DataFrame
        Biochemistry table (longitudinal labs).
    condition_ids : list
        Conditions to include (defines the matrix index).

    Returns
    -------
    tuple of (pandas.DataFrame, FeatureSpec)
        The context matrix (NaNs preserved) and column-type metadata.
    """
    idx: pd.Index[Any] = pd.Index(condition_ids, name="condition_id")
    pc = patient_cases.drop_duplicates("condition_id").set_index("condition_id").reindex(idx)

    numeric: list[str] = []
    categorical: list[str] = []
    C = pd.DataFrame(index=idx)

    # --- Demographics ---
    demo_numeric = ["age_of_onset", "bmi", "number_of_children", "number_of_daily_contacts"]
    for col in demo_numeric:
        if col in pc.columns:
            C[col] = pd.to_numeric(pc[col], errors="coerce")
            numeric.append(col)
    for col in ["gender", "education", "employment"]:
        if col in pc.columns:
            C[col] = pc[col].astype("string")
            categorical.append(col)

    # --- Socioeconomic ---
    if "country" in pc.columns:
        C["country"] = pc["country"].astype("string")
        categorical.append("country")

    # --- Comorbidities (parsed from free text) ---
    if "comorbidity" in pc.columns:
        flags = parse_comorbidity_flags(pc["comorbidity"])
        for col in flags.columns:
            C[col] = flags[col].astype(int)
            categorical.append(col)

    # --- Clinical baseline ---
    for col in ["type_of_resistance", "case_definition", "lung_localization"]:
        if col in pc.columns:
            C[col] = pc[col].astype("string")
            categorical.append(col)
    if "lungcavity_size" in pc.columns:
        C["cavitation"] = pc["lungcavity_size"].map(_cavitation).astype("string")
        categorical.append("cavitation")
    if "culture" in pc.columns:
        C["baseline_culture"] = pc["culture"].map(_micro_status).astype("string")
        categorical.append("baseline_culture")
    if "microscopy" in pc.columns:
        C["baseline_smear"] = pc["microscopy"].map(_micro_status).astype("string")
        categorical.append("baseline_smear")
    if "timika_score" in pc.columns:
        C["timika_score"] = pd.to_numeric(pc["timika_score"], errors="coerce")
        numeric.append("timika_score")

    # --- Radiology / imaging severity (mostly 100% complete in TB Portals) ---
    img_numeric = [
        "totalcavernum",
        "overall_percent_of_abnormal_volume",
        "pleural_effusion_percent_of_hemithorax_involved",
    ]
    for col in img_numeric:
        if col in pc.columns:
            C[col] = pd.to_numeric(pc[col], errors="coerce")
            numeric.append(col)
    img_categorical = ["dissemination", "affect_pleura", "shadow_pattern"]
    for col in img_categorical:
        if col in pc.columns:
            C[col] = pc[col].astype("string")
            categorical.append(col)

    # --- Genomics (M. tuberculosis lineage) ---
    for col in ["main_lineage", "sub_lineage"]:
        if col in pc.columns:
            C[col] = pc[col].astype("string")
            categorical.append(col)

    # --- Baseline per-drug drug-susceptibility (BACTEC; the key prescribing confounder) ---
    bactec_cols = [c for c in pc.columns if c.startswith("bactec_") and c != "bactec_test"]
    for col in bactec_cols:
        C[col] = pc[col].astype("string")
        categorical.append(col)

    # --- Baseline labs ---
    labs = _baseline_labs(biochemistry, idx)
    for col in labs.columns:
        C[col] = labs[col]
        numeric.append(col)

    # Drop explicitly-excluded and all-missing columns. The resulting 335-column encoded schema is
    # the manuscript contract for the locked August 2023 cohort.
    canonical_zero_variance = len(C) >= 1000
    dead = [
        col
        for col in (*numeric, *categorical)
        if (canonical_zero_variance and col in _EXCLUDE_FEATURES) or C[col].dropna().empty
    ]
    if dead:
        logger.info("Dropping %d excluded / zero-variance features: %s", len(dead), dead)
        C = C.drop(columns=dead)
        numeric = [c for c in numeric if c not in dead]
        categorical = [c for c in categorical if c not in dead]

    spec = FeatureSpec(numeric=numeric, categorical=categorical)
    miss = C.isna().mean().sort_values(ascending=False)
    logger.info(
        "Context matrix: %d conditions x %d features (%d numeric, %d categorical). "
        "Max feature missingness %.1f%%",
        C.shape[0],
        C.shape[1],
        len(numeric),
        len(categorical),
        100 * miss.iloc[0] if len(miss) else 0,
    )
    return C[spec.all_columns], spec


def context_missingness_table(context: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Summarise per-feature missingness and type.

    Parameters
    ----------
    context : pandas.DataFrame
        Context matrix from :func:`build_context_matrix`.
    spec : FeatureSpec
        Feature specification.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``kind``, ``missing_rate`` sorted by missingness.
    """
    kinds = {c: "numeric" for c in spec.numeric}
    kinds.update({c: "categorical" for c in spec.categorical})
    rows = [
        {"feature": c, "kind": kinds.get(c, "?"), "missing_rate": float(context[c].isna().mean())}
        for c in context.columns
    ]
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def build_context_masks(c_raw: pd.DataFrame, spec: FeatureSpec) -> dict[str, NDArray[np.bool_]]:
    """Construct interpretable binary context indicators from the raw context matrix.

    Numeric features are split at their median (``feature>med``); each categorical level with at
    least :data:`MASK_MIN_LEVEL_N` patients (excluding missingness levels) becomes an indicator.
    For a binary categorical the two levels are exact complements (mirror effects), so only the
    rarer "present" level is kept.

    Parameters
    ----------
    c_raw : pandas.DataFrame
        Raw (un-encoded) context matrix from :func:`build_context_matrix`.
    spec : FeatureSpec
        Numeric/categorical column metadata.

    Returns
    -------
    dict of str to numpy.ndarray
        Named boolean masks over patients.
    """
    masks: dict[str, NDArray[np.bool_]] = {}
    for col in spec.numeric:
        vals = pd.to_numeric(c_raw[col], errors="coerce")
        if vals.notna().sum() < 2 * MASK_MIN_LEVEL_N:
            continue
        med = float(vals.median())
        mask = (vals > med).fillna(False).to_numpy()
        if MASK_MIN_LEVEL_N <= int(mask.sum()) <= len(mask) - MASK_MIN_LEVEL_N:
            masks[f"{col}>med"] = mask
    for col in spec.categorical:
        s = c_raw[col].astype("string")
        levels = [
            (level, int(n))
            for level, n in s.value_counts().items()
            if n >= MASK_MIN_LEVEL_N and str(level) not in MASK_MISSING_LEVELS
        ]
        if len(levels) == 2 and s.isna().sum() == 0:
            level, _ = min(levels, key=lambda kv: kv[1])
            masks[f"{col}={level}"] = (s == level).fillna(False).to_numpy()
            continue
        for level, _ in levels:
            masks[f"{col}={level}"] = (s == level).fillna(False).to_numpy()
    logger.info(
        "Built %d interpretable context indicators (min n=%d)", len(masks), MASK_MIN_LEVEL_N
    )  # noqa: E501
    return masks
