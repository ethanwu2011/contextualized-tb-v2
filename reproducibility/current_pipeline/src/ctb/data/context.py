"""Construction of the context matrix ``C`` (patient covariates).

Builds demographic, comorbidity, clinical-baseline, imaging, socioeconomic, and baseline-laboratory
features from the Patient Cases and Biochemistry tables. Features are returned *un-imputed* (NaNs
preserved). A :class:`FeatureSpec` records which columns are numeric versus categorical for a
downstream preprocessor. The current endpoint-artifact builder fits an exploratory full-cohort
encoder for the non-claim-bearing S1/S2 scaffolds; every held-out or claim-bearing analysis must
instead fit imputation, scaling, and categorical vocabulary inside each training fold.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ctb.data.temporal import baseline_event_mask, parse_temporal_coordinate

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

# Context features dropped after construction. ``lab_hemoglobin`` (92% missing) is removed —
# ablation confirmed the 5 treatment-response subtypes survive its removal. Zero-variance /
# all-missing columns are pruned automatically alongside it (totalcavernum,
# overall_percent_of_abnormal_volume, pleural_effusion_*, baseline_culture, bactec_mycobutin,
# bactec_imipenem_cilastatin). Verify the subtypes survive before adding more here.
_EXCLUDE_FEATURES: frozenset[str] = frozenset({"lab_hemoglobin"})


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


def _baseline_labs(
    biochemistry: pd.DataFrame,
    condition_ids: pd.Index[Any],
    *,
    index_dates: pd.Series[Any] | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Extract earliest-available labs, optionally constrained to a baseline window.

    Parameters
    ----------
    biochemistry : pandas.DataFrame
        Biochemistry table with ``condition_id``, ``test_date``, and lab columns.
    condition_ids : pandas.Index
        Conditions to index the result by.
    index_dates : pandas.Series or None, default None
        Optional time-zero value for each condition, indexed by ``condition_id``. Values may be
        integral day offsets or calendar dates but must match ``test_date``. When supplied,
        post-index and unalignable measurements are excluded.
    lookback_days : int or None, default None
        Optional maximum number of days before time zero.  Requires ``index_dates``.

    Returns
    -------
    pandas.DataFrame
        One column per lab in :data:`_LAB_COLS`, indexed by ``condition_id`` (NaN if untested).
    """
    bx = biochemistry.copy()
    test_order, _ = parse_temporal_coordinate(bx["test_date"], field="test_date")
    bx["_test_order"] = test_order
    if index_dates is None:
        if lookback_days is not None:
            raise ValueError("lookback_days requires index_dates")
        logger.warning(
            "No index dates supplied for laboratory extraction; using earliest available values "
            "without claiming that their timing is pre-index"
        )
    else:
        eligible, report = baseline_event_mask(
            bx,
            index_dates,
            event_date_col="test_date",
            lookback_days=lookback_days,
        )
        bx = bx.loc[eligible].copy()
        logger.info(
            "Baseline lab window: %d/%d eligible; excluded %d post-index, %d before "
            "lookback, %d missing event date, %d missing index date",
            report.eligible_events,
            report.total_events,
            report.post_index_events,
            report.before_lookback_events,
            report.missing_event_date,
            report.missing_index_date,
        )
    bx = bx.sort_values("_test_order", kind="mergesort")
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
    *,
    baseline_index_dates: pd.Series[Any] | None = None,
    baseline_lookback_days: int | None = None,
    preserve_degenerate_columns: bool = False,
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
    baseline_index_dates : pandas.Series or None, default None
        Optional time-zero value per condition, using the same integral day-offset or calendar-date
        coordinate as laboratory ``test_date``. Supplying this activates temporal validation and
        excludes post-index measurements. ``None`` preserves the historical earliest-available
        behavior for backward compatibility.
    baseline_lookback_days : int or None, default None
        Optional pre-index laboratory window length.  Requires ``baseline_index_dates``.
    preserve_degenerate_columns : bool, default False
        Retain raw all-missing and constant columns so a held-out analysis can decide which
        columns are usable from each training fold only. Explicitly excluded features remain
        excluded. The default preserves the historical full-cohort pruning behavior.

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
    labs = _baseline_labs(
        biochemistry,
        idx,
        index_dates=baseline_index_dates,
        lookback_days=baseline_lookback_days,
    )
    for col in labs.columns:
        C[col] = labs[col]
        numeric.append(col)

    # Explicit exclusions are prespecified and do not depend on the observed cohort. Degenerate
    # columns are historically pruned here, but a claim-bearing held-out analysis preserves them
    # so its FoldPreprocessor can make that decision using training rows only.
    dead = [
        col
        for col in (*numeric, *categorical)
        if col in _EXCLUDE_FEATURES
        or (not preserve_degenerate_columns and C[col].dropna().nunique() <= 1)
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
