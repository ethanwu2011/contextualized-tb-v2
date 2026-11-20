"""Build endpoint-specific modelling matrices for the contextualized analysis.

Assembles the design used by the archived contextualized coefficient figures (scripts 06/07/09):

* **X** — binary initial-regimen drug matrix (one column per drug at >=1% prevalence).
* **C** — encoded context matrix: numeric features median-imputed + standardized; categorical
  features one-hot encoded with an explicit ``<NA>`` category. Source features (demographics,
  comorbidities, imaging, per-drug DST, lineage, socioeconomic/region, baseline labs) come from
  :func:`ctb.data.context.build_context_matrix`.
Two endpoint-specific cohorts are written rather than swapping outcomes on one convenient matrix:

* **culture conversion** — the baseline-eligible population (first observed positive culture at or
  before day 0) within the CONSORT analysis cohort;
* **died/failure** — all CONSORT conditions with an assessable recorded final Patient Cases
  outcome. Culture measurements are not an eligibility requirement.

Each endpoint gets context preprocessing fitted on its own declared cohort. Writes:

* ``data/processed/conversion_model_matrices.npz`` — culture-conversion ``y`` plus provenance.
* ``data/processed/died_failure_model_matrices.npz`` — died/failure ``y`` plus provenance.
* ``data/processed/conversion_context_meta.parquet`` — per-condition metadata (outcome + raw
  interpretable context columns) for the conversion cohort.
* ``data/processed/died_failure_context_meta.parquet`` — corresponding metadata for the primary
  died/failure cohort.
* ``data/processed/died_failure_cohort_report.json`` — aggregate-only denominator flow,
  exclusion diagnostics, canonical endpoint contract, and ordered cohort fingerprints.

Only aggregate shapes are logged; no row-level data is emitted.

Usage
-----
    uv run --frozen python scripts/05b_build_conversion_matrices.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ctb.config import PROCESSED_DIR, ensure_output_dirs
from ctb.data.clean import CohortStep, build_cohort, clean_patient_cases, clean_regimens
from ctb.data.context import FeatureSpec, build_context_matrix
from ctb.data.conversion import derive_conversion
from ctb.data.load import load_biochemistry, load_dst, load_patient_cases, load_regimens
from ctb.data.model_matrices import (
    ARTIFACT_SCHEMA_VERSION,
    COHORT_DEFINITION_IDS,
    COHORT_DEFINITIONS,
    CULTURE_CONVERSION,
    DIED_FAILURE,
    ENDPOINT_ARTIFACTS,
    ENDPOINT_DEFINITION_IDS,
    ENDPOINT_DEFINITIONS,
    cohort_fingerprint,
    write_culture_conversion_matrices,
    write_died_failure_matrices,
)
from ctb.data.outcomes import derive_died_failure, derive_patient_case_died_failure
from ctb.data.treatments import build_treatment_matrix, select_initial_regimens
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
    if keep:
        imputed = SimpleImputer(strategy="median").fit_transform(num[keep])
        num_std = pd.DataFrame(
            StandardScaler().fit_transform(imputed), columns=keep, index=context.index
        )
    else:
        num_std = pd.DataFrame(index=context.index)
    cat = (
        pd.get_dummies(context[spec.categorical].astype("string"), dummy_na=True)
        if spec.categorical
        else pd.DataFrame(index=context.index)
    )
    return pd.concat([num_std, cat], axis=1)


def build_design(
    patient_cases: pd.DataFrame,
    biochemistry: pd.DataFrame,
    regimens: pd.DataFrame,
    condition_ids: list[Any],
    *,
    lab_index_dates: pd.Series[Any],
) -> tuple[pd.Index[Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one endpoint cohort's ordered treatment and time-restricted context matrices."""
    idx: pd.Index[Any] = pd.Index(condition_ids, name="condition_id")
    if not lab_index_dates.index.is_unique:
        raise ValueError("Laboratory index dates must contain at most one row per condition")
    ordered_lab_index = lab_index_dates.reindex(idx)
    missing_index_dates = int(ordered_lab_index.isna().sum())
    if missing_index_dates:
        raise ValueError(
            "Laboratory time-zero linkage is incomplete: "
            f"missing_index_dates={missing_index_dates}, requested_conditions={len(idx)}"
        )
    treatment = build_treatment_matrix(regimens, condition_ids).reindex(idx)
    context_raw, spec = build_context_matrix(
        patient_cases,
        biochemistry,
        condition_ids,
        baseline_index_dates=ordered_lab_index,
    )
    context_raw = context_raw.reindex(idx)
    context_encoded = encode_context(context_raw, spec)
    return idx, treatment, context_raw, context_encoded


def resolve_patient_ids(
    patient_cases: pd.DataFrame,
    condition_ids: list[Any],
) -> pd.Series[str]:
    """Resolve one nonempty patient identifier per requested condition or fail closed."""
    required = {"condition_id", "patient_id"}
    missing_columns = sorted(required.difference(patient_cases.columns))
    if missing_columns:
        raise KeyError(f"Patient linkage requires columns {missing_columns}")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Requested condition identifiers must be unique")

    requested = set(condition_ids)
    links = patient_cases.loc[
        patient_cases["condition_id"].isin(requested),
        ["condition_id", "patient_id"],
    ].copy()
    links["_patient_id"] = links["patient_id"].astype("string").str.strip().replace("", pd.NA)
    grouped = links.groupby("condition_id", sort=False)["_patient_id"]
    distinct = grouped.nunique(dropna=True)
    ambiguous_conditions = int(distinct.gt(1).sum())
    resolved = grouped.first().reindex(condition_ids)
    missing_conditions = int(resolved.isna().sum())
    if missing_conditions or ambiguous_conditions:
        raise ValueError(
            "Condition-to-patient linkage is not modelable: "
            f"requested_conditions={len(condition_ids)}, missing_links={missing_conditions}, "
            f"ambiguous_links={ambiguous_conditions}"
        )
    return pd.Series(
        resolved.astype(str).to_numpy(),
        index=pd.Index(condition_ids, name="condition_id"),
        dtype="string",
        name="patient_id",
    )


def build_metadata(
    index: pd.Index[Any],
    context_raw: pd.DataFrame,
    *,
    died_failure: pd.Series[Any],
    conversion: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct endpoint-aligned, interpretable aggregate-analysis metadata."""
    meta = pd.DataFrame(index=index)
    meta["died_failure"] = died_failure.reindex(index).astype("Int64")
    if conversion is not None:
        meta["event"] = conversion["event"].reindex(index).astype("Int64")
        meta["time_to_event"] = conversion["time_to_event"].reindex(index)
    for column in _META_CONTEXT_COLS:
        if column in context_raw.columns:
            meta[column] = context_raw[column].to_numpy()
        else:
            logger.warning("Meta context column %r absent from context matrix; skipping", column)
    # The endpoint artifact canonicalizes integer/text identifiers to strings. Keep the companion
    # parquet on the same representation so a future guarded consumer cannot silently misalign it.
    meta.index = pd.Index([str(value) for value in meta.index], name="condition_id")
    return meta


def build_culture_eligibility_report(
    *,
    consort_ids: list[Any],
    conversion: pd.DataFrame,
    private_condition_sets: dict[str, frozenset[Any]],
) -> dict[str, int]:
    """Return mutually exclusive, CONSORT-intersected culture-eligibility counts."""
    consort = set(consort_ids)
    has_row = set(private_condition_sets.get("has_any_culture_row", frozenset()))
    interpretable = set(private_condition_sets.get("has_interpretable_culture_result", frozenset()))
    valid_dated = set(
        private_condition_sets.get("has_valid_interpretable_dated_culture", frozenset())
    )
    conflict = set(private_condition_sets.get("quarantined_same_day_conflict", frozenset()))

    no_row = consort - has_row
    no_interpretable = (consort & has_row) - interpretable
    no_valid_timing = (consort & interpretable) - valid_dated
    quarantined = consort & conflict
    never_positive = consort & set(conversion[conversion["first_pos_day"].isna()].index)
    first_positive_after_day_zero = consort & set(
        conversion[conversion["first_pos_day"].gt(0)].index
    )
    baseline_eligible = consort & set(conversion[conversion["first_pos_day"].le(0)].index)
    categories = {
        "no_culture_row": no_row,
        "no_interpretable_culture_result": no_interpretable,
        "no_valid_interpretable_culture_timing": no_valid_timing,
        "quarantined_same_day_conflict": quarantined,
        "never_positive": never_positive,
        "first_positive_after_day_zero": first_positive_after_day_zero,
        "baseline_eligible_first_positive_at_or_before_day_zero": baseline_eligible,
    }
    category_union: set[Any] = set()
    for label, identifiers in categories.items():
        overlap = category_union.intersection(identifiers)
        if overlap:
            raise ValueError(f"Culture-eligibility category {label!r} overlaps an earlier category")
        category_union.update(identifiers)
    if category_union != consort:
        raise ValueError("Culture-eligibility categories do not reconstruct the CONSORT cohort")

    return {
        "consort_conditions": len(consort),
        **{label: len(identifiers) for label, identifiers in categories.items()},
        "excluded_before_baseline_eligibility": len(consort) - len(baseline_eligible),
    }


def build_died_failure_assessability_report(
    *,
    patient_cases: pd.DataFrame,
    source_ids: list[Any],
) -> dict[str, int]:
    """Return mutually exclusive final Patient Cases outcome counts for the source cohort."""
    derived = derive_patient_case_died_failure(patient_cases)["died_failure"].reindex(source_ids)
    raw = (
        patient_cases.set_index("condition_id")["outcome"]
        .reindex(source_ids)
        .astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )
    counts = {
        "event_recorded_died_failure": int(derived.eq(True).sum()),  # noqa: E712
        "observed_nonevent_recorded_cured_completed": int(derived.eq(False).sum()),  # noqa: E712
        "lost_to_follow_up": int(raw.eq("lost to follow up").sum()),
        "still_on_treatment": int(raw.eq("still on treatment").sum()),
        "palliative_care": int(raw.eq("palliative care").sum()),
        "unknown": int(raw.eq("unknown").sum()),
        "missing_or_blank_outcome": int(raw.isna().sum()),
        "other_unclassified_outcome": 0,
    }
    recognized = {
        "died",
        "failure",
        "cured",
        "completed",
        "lost to follow up",
        "still on treatment",
        "palliative care",
        "unknown",
    }
    counts["other_unclassified_outcome"] = int((raw.notna() & ~raw.isin(recognized)).sum())
    assessed = (
        counts["event_recorded_died_failure"] + counts["observed_nonevent_recorded_cured_completed"]
    )
    if sum(counts.values()) != len(source_ids):
        raise ValueError(
            "Patient Cases outcome categories do not reconstruct the primary source cohort"
        )
    if assessed != int(derived.notna().sum()):
        raise ValueError(
            "Patient Cases outcome categories disagree with the derived died/failure endpoint"
        )
    return {
        "source_conditions": len(source_ids),
        **counts,
        "assessed_conditions": assessed,
        "unassessable_conditions": len(source_ids) - assessed,
    }


def build_outcome_concordance_report(
    *,
    patient_case_labels: pd.Series[Any],
    regimen_history_labels: pd.Series[Any],
    source_ids: list[Any],
) -> dict[str, int | bool]:
    """Verify aggregate and row-aligned concordance of two recorded outcome sources."""
    direct = patient_case_labels.reindex(source_ids).astype("boolean")
    regimen = regimen_history_labels.reindex(source_ids).astype("boolean")
    both_assessed = direct.notna() & regimen.notna()
    direct_only = direct.notna() & regimen.isna()
    regimen_only = direct.isna() & regimen.notna()
    disagreements = both_assessed & direct.ne(regimen).fillna(False)
    report: dict[str, int | bool] = {
        "source_conditions": len(source_ids),
        "patient_cases_assessed": int(direct.notna().sum()),
        "regimen_history_assessed": int(regimen.notna().sum()),
        "both_assessed": int(both_assessed.sum()),
        "patient_cases_only_assessed": int(direct_only.sum()),
        "regimen_history_only_assessed": int(regimen_only.sum()),
        "label_disagreements_among_both_assessed": int(disagreements.sum()),
        "exact_assessability_and_label_concordance": bool(
            not direct_only.any() and not regimen_only.any() and not disagreements.any()
        ),
    }
    if not report["exact_assessability_and_label_concordance"]:
        raise ValueError(
            "Patient Cases and regimen-history died/failure sources disagree: "
            f"patient_cases_only={report['patient_cases_only_assessed']}, "
            f"regimen_history_only={report['regimen_history_only_assessed']}, "
            f"label_disagreements={report['label_disagreements_among_both_assessed']}"
        )
    return report


def build_cohort_report(
    *,
    patient_cases: pd.DataFrame,
    consort_ids: list[Any],
    endpoint_candidate_ids: list[Any],
    primary_ids: list[Any],
    primary_y: pd.Series[Any],
    cohort_steps: list[CohortStep],
    culture_eligibility_report: dict[str, int],
    died_failure_assessability_report: dict[str, int],
    initial_regimen_report: dict[str, Any],
    culture_date_report: dict[str, Any],
    patient_case_died_failure_report: dict[str, Any],
    regimen_history_died_failure_report: dict[str, Any],
    outcome_concordance_report: dict[str, int | bool],
    patient_linkage_report: dict[str, int],
) -> dict[str, Any]:
    """Build an aggregate-only, machine-readable primary-endpoint denominator report."""
    all_ids = patient_cases["condition_id"].drop_duplicates().tolist()
    source_n_rows = len(endpoint_candidate_ids)
    assessed_n_rows = len(primary_ids)
    n_events = int(primary_y.sum())
    n_non_events = assessed_n_rows - n_events
    n_unassessable = source_n_rows - assessed_n_rows

    if not cohort_steps or cohort_steps[-1].n_conditions != len(consort_ids):
        raise ValueError("CONSORT step counts disagree with the final CONSORT identifier set")
    if n_unassessable < 0 or n_events < 0 or n_non_events < 0:
        raise ValueError("Primary endpoint denominator counts are internally inconsistent")
    if assessed_n_rows + n_unassessable != source_n_rows:
        raise ValueError("Assessed and unassessable counts do not reconstruct the source cohort")
    if endpoint_candidate_ids != consort_ids:
        raise ValueError("Primary endpoint source must be the complete ordered CONSORT cohort")
    if culture_eligibility_report["consort_conditions"] != len(consort_ids):
        raise ValueError("Culture-eligibility reason counts disagree with the CONSORT cohort")
    if (
        died_failure_assessability_report["source_conditions"] != source_n_rows
        or died_failure_assessability_report["assessed_conditions"] != assessed_n_rows
        or died_failure_assessability_report["unassessable_conditions"] != n_unassessable
        or died_failure_assessability_report["event_recorded_died_failure"] != n_events
        or died_failure_assessability_report["observed_nonevent_recorded_cured_completed"]
        != n_non_events
    ):
        raise ValueError("Died/failure assessability reasons disagree with endpoint denominators")
    if (
        outcome_concordance_report["source_conditions"] != source_n_rows
        or outcome_concordance_report["patient_cases_assessed"] != assessed_n_rows
        or not outcome_concordance_report["exact_assessability_and_label_concordance"]
    ):
        raise ValueError("Outcome-source concordance report disagrees with endpoint denominators")
    if (
        patient_linkage_report["source_conditions"] != source_n_rows
        or patient_linkage_report["assessed_conditions"] != assessed_n_rows
    ):
        raise ValueError("Patient-linkage report disagrees with endpoint denominators")

    full_flow = [asdict(step) for step in cohort_steps]
    full_flow.append(
        {
            "label": "With an assessable recorded final Patient Cases died/failure outcome",
            "n_conditions": assessed_n_rows,
            "n_excluded": n_unassessable,
        }
    )

    def ordered_hash(values: list[Any]) -> str:
        return cohort_fingerprint(np.asarray(values, dtype=str))

    return {
        "report_schema_version": 2,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "endpoint_contract": {
            "endpoint": DIED_FAILURE,
            "endpoint_definition_id": ENDPOINT_DEFINITION_IDS[DIED_FAILURE],
            "endpoint_definition": ENDPOINT_DEFINITIONS[DIED_FAILURE],
            "cohort_definition_id": COHORT_DEFINITION_IDS[DIED_FAILURE],
            "cohort_definition": COHORT_DEFINITIONS[DIED_FAILURE],
        },
        "flow_counts": {
            "all_patient_conditions": len(all_ids),
            "consort_conditions": len(consort_ids),
            "primary_source_conditions": source_n_rows,
            "culture_conversion_sensitivity_conditions": culture_eligibility_report[
                "baseline_eligible_first_positive_at_or_before_day_zero"
            ],
            "died_failure_assessed_conditions": assessed_n_rows,
            "died_failure_unassessable_conditions": n_unassessable,
            "died_failure_events": n_events,
            "died_failure_observed_non_events": n_non_events,
        },
        "ordered_cohort_sha256": {
            "all_patient_conditions": ordered_hash(all_ids),
            "consort_conditions": ordered_hash(consort_ids),
            "primary_source_conditions": ordered_hash(endpoint_candidate_ids),
            "died_failure_assessed_conditions": ordered_hash(primary_ids),
        },
        "cohort_steps": full_flow,
        "culture_eligibility": culture_eligibility_report,
        "died_failure_assessability": died_failure_assessability_report,
        "outcome_source_concordance": outcome_concordance_report,
        "patient_linkage": patient_linkage_report,
        "initial_regimen_selection_report": initial_regimen_report,
        "culture_date_report": culture_date_report,
        "patient_cases_died_failure_report": patient_case_died_failure_report,
        "regimen_history_died_failure_report": regimen_history_died_failure_report,
        "contains_row_level_identifiers": False,
    }


def main() -> int:
    """Assemble and persist endpoint-specific modelling matrices and metadata.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    workflow_mode = os.environ.get("CTB_WORKFLOW_MODE")
    if os.environ.get("CTB_GOVERNED_RUN") != "1" or workflow_mode not in {"synthetic", "dua"}:
        print(
            "BLOCKED: endpoint matrices may be written only by scripts/run_manuscript.py.",
            file=sys.stderr,
        )
        return 2
    if not os.environ.get("CTB_OUTPUT_ROOT"):
        print("BLOCKED: governed output root is not set.", file=sys.stderr)
        return 2

    configure_logging()
    set_global_seed()
    ensure_output_dirs()

    pc = clean_patient_cases(load_patient_cases())
    reg = clean_regimens(load_regimens())
    bx = load_biochemistry()
    dst = load_dst()

    # Culture conversion remains a separate sensitivity cohort. It does not restrict the primary
    # recorded-final-outcome cohort.
    conv = derive_conversion(dst)
    private_culture_condition_sets = dict(conv.attrs.pop("_private_eligibility_condition_sets", {}))
    baseline_pos = set(conv[conv["first_pos_day"].le(0)].index)
    cohort = build_cohort(pc, reg)
    consort_ids = list(cohort.condition_ids)
    endpoint_candidate_ids = list(consort_ids)
    conversion_ids = [condition_id for condition_id in consort_ids if condition_id in baseline_pos]

    patient_case_died_failure_frame = derive_patient_case_died_failure(pc)
    died_failure = patient_case_died_failure_frame["died_failure"]
    regimen_history_died_failure_frame = derive_died_failure(reg)
    outcome_concordance_report = build_outcome_concordance_report(
        patient_case_labels=died_failure,
        regimen_history_labels=regimen_history_died_failure_frame["died_failure"],
        source_ids=endpoint_candidate_ids,
    )
    assessable_died_failure = set(died_failure.index[died_failure.notna()])
    primary_ids = [
        condition_id
        for condition_id in endpoint_candidate_ids
        if condition_id in assessable_died_failure
    ]
    n_primary_unassessable = len(endpoint_candidate_ids) - len(primary_ids)
    culture_eligibility_report = build_culture_eligibility_report(
        consort_ids=consort_ids,
        conversion=conv,
        private_condition_sets=private_culture_condition_sets,
    )
    died_failure_assessability_report = build_died_failure_assessability_report(
        patient_cases=pc,
        source_ids=endpoint_candidate_ids,
    )

    source_patient_ids = resolve_patient_ids(pc, endpoint_candidate_ids)
    primary_patient_ids = source_patient_ids.reindex(primary_ids)
    conversion_patient_ids = source_patient_ids.reindex(conversion_ids)
    patient_linkage_report = {
        "source_conditions": len(source_patient_ids),
        "source_unique_patients": int(source_patient_ids.nunique()),
        "assessed_conditions": len(primary_patient_ids),
        "assessed_unique_patients": int(primary_patient_ids.nunique()),
        "missing_links": 0,
        "ambiguous_links": 0,
    }
    logger.info(
        "Endpoint cohorts: primary source=%d; died/failure assessed=%d; "
        "died/failure unassessable=%d; culture-conversion sensitivity=%d",
        len(consort_ids),
        len(primary_ids),
        n_primary_unassessable,
        len(conversion_ids),
    )

    initial_regimens = select_initial_regimens(reg)
    primary_lab_index = initial_regimens["initial_start"].reindex(primary_ids)
    conversion_lab_index = pd.Series(
        0,
        index=pd.Index(conversion_ids, name="condition_id"),
        dtype="Int64",
        name="culture_conversion_day_zero",
    )
    conversion_idx, conversion_x, conversion_c_raw, conversion_c_enc = build_design(
        pc,
        bx,
        reg,
        conversion_ids,
        lab_index_dates=conversion_lab_index,
    )
    primary_idx, primary_x, primary_c_raw, primary_c_enc = build_design(
        pc,
        bx,
        reg,
        primary_ids,
        lab_index_dates=primary_lab_index,
    )
    conversion_y = conv["event"].reindex(conversion_idx).astype(int)
    died_failure_y = died_failure.reindex(primary_idx).astype(int)

    conversion_meta = build_metadata(
        conversion_idx,
        conversion_c_raw,
        died_failure=died_failure,
        conversion=conv,
    )
    primary_meta = build_metadata(
        primary_idx,
        primary_c_raw,
        died_failure=died_failure,
    )

    write_culture_conversion_matrices(
        PROCESSED_DIR / ENDPOINT_ARTIFACTS[CULTURE_CONVERSION],
        X=conversion_x.to_numpy(dtype=np.float64),
        C=conversion_c_enc.to_numpy(dtype=np.float64),
        dst=dst,
        drugs=np.asarray(list(conversion_x.columns), dtype=object),
        cond_ids=np.asarray(conversion_ids, dtype=object),
        patient_ids=conversion_patient_ids.to_numpy(dtype=object),
        source_cond_ids=np.asarray(conversion_ids, dtype=object),
        context_names=np.asarray(list(conversion_c_enc.columns), dtype=object),
    )
    write_died_failure_matrices(
        PROCESSED_DIR / ENDPOINT_ARTIFACTS[DIED_FAILURE],
        X=primary_x.to_numpy(dtype=np.float64),
        C=primary_c_enc.to_numpy(dtype=np.float64),
        patient_cases=pc,
        drugs=np.asarray(list(primary_x.columns), dtype=object),
        cond_ids=np.asarray(primary_ids, dtype=object),
        patient_ids=primary_patient_ids.to_numpy(dtype=object),
        source_cond_ids=np.asarray(endpoint_candidate_ids, dtype=object),
        context_names=np.asarray(list(primary_c_enc.columns), dtype=object),
    )
    conversion_meta.to_parquet(PROCESSED_DIR / "conversion_context_meta.parquet")
    primary_meta.to_parquet(PROCESSED_DIR / "died_failure_context_meta.parquet")
    cohort_report = build_cohort_report(
        patient_cases=pc,
        consort_ids=consort_ids,
        endpoint_candidate_ids=endpoint_candidate_ids,
        primary_ids=primary_ids,
        primary_y=died_failure_y,
        cohort_steps=cohort.steps,
        culture_eligibility_report=culture_eligibility_report,
        died_failure_assessability_report=died_failure_assessability_report,
        initial_regimen_report=initial_regimens.attrs["selection_report"],
        culture_date_report=conv.attrs["date_report"],
        patient_case_died_failure_report=patient_case_died_failure_frame.attrs[
            "died_failure_report"
        ],
        regimen_history_died_failure_report=regimen_history_died_failure_frame.attrs[
            "died_failure_report"
        ],
        outcome_concordance_report=outcome_concordance_report,
        patient_linkage_report=patient_linkage_report,
    )
    (PROCESSED_DIR / "died_failure_cohort_report.json").write_text(
        json.dumps(cohort_report, indent=2, sort_keys=True) + "\n"
    )

    logger.info(
        "Wrote conversion matrices X=%s C=%s, events=%d/%d; "
        "died/failure matrices X=%s C=%s, events=%d/%d",
        conversion_x.shape,
        conversion_c_enc.shape,
        int(conversion_y.sum()),
        len(conversion_y),
        primary_x.shape,
        primary_c_enc.shape,
        int(died_failure_y.sum()),
        len(died_failure_y),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
