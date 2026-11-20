"""Rebuild SINAN 2015-2019 and audit PATCH's prognostic external validation.

This analysis starts from the five original DATASUS DBC files, applies the official SINAN coding
dictionary, rebuilds the manuscript cohort, and uses the current locked TB Portals cohort as the
source. Brazil is used only to evaluate prognostic profile structure. No regimen is assigned to a
Brazil record and no claim about transported treatment heterogeneity is made. Row-level objects
remain in memory and only disclosure-safe aggregates are written.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dbfread import DBF
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.models.archetype_contextualized import (
    LinearArchetypeContextualized,
    TrainConfig,
    train,
)
from ctb.models.locked_validation import LockedValidationConfig, build_schema6_raw_validation_data
from ctb.utils.reproducibility import set_global_seed
from run_aggregate_sensitivities import SMALL_CELL_MIN, _safe_count, _write_report

YEARS = (2015, 2016, 2017, 2018, 2019)
PROFILE_SEEDS = (0, 9, 42, 137, 314, 2718, 8675)
LOCKED_EXTERNAL_SEED = 0
FEATURES = (
    "rif_resistant",
    "hiv",
    "diabetes",
    "smear_pos",
    "age",
    "male",
    "new_case",
)
FIRST_LINE = ("R", "H", "Z", "E", "S")
KEEP = (
    "SITUA_ENCE",
    "TEST_MOLEC",
    "HIV",
    "AGRAVDIABE",
    "BACILOSC_E",
    "FORMA",
    "TRATAMENTO",
    "NU_IDADE_N",
    "CS_SEXO",
    "SG_UF",
)
PROFILE_CONFIG = TrainConfig(
    epochs=2000,
    lr=0.01,
    weight_decay=1e-3,
    archetype_l1=5e-4,
    entropy_weight=1.0,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument("--tb-raw-dir", type=Path, required=True)
    parser.add_argument("--sinan-src", type=Path, required=True)
    parser.add_argument(
        "--decompress-helper",
        type=Path,
        default=Path(__file__).with_name("decompress_sinan_dbc.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap_resamples < 200:
        parser.error("--bootstrap-resamples must be at least 200")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _code(value: object) -> str:
    """Normalize numeric category codes while preserving nonnumeric text."""
    text = _text(value)
    if text.isdigit():
        return str(int(text))
    return text


def _read_year(dbf_path: Path, year: int) -> pd.DataFrame:
    table = DBF(
        str(dbf_path),
        encoding="latin-1",
        load=False,
        ignore_missing_memofile=True,
        char_decode_errors="ignore",
    )
    missing = sorted(set(KEEP).difference(table.field_names))
    if missing:
        raise KeyError(f"SINAN DBF is missing required columns: {missing}")
    rows = ({name: record.get(name) for name in KEEP} for record in table)
    frame = pd.DataFrame.from_records(rows, columns=KEEP)
    frame["year"] = year
    return frame


def _load_sinan(
    src: Path,
    *,
    helper: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="patch_sinan_") as temporary:
        temp = Path(temporary)
        for year in YEARS:
            source = src / f"TUBEBR{str(year)[-2:]}.dbc"
            if not source.is_file():
                raise FileNotFoundError(f"Missing required SINAN source file for {year}")
            output = temp / f"TUBEBR{str(year)[-2:]}.dbf"
            subprocess.run(
                [
                    sys.executable,
                    str(helper),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            frames.append(_read_year(output, year))
            files.append(
                {
                    "year": year,
                    "filename": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                    "notifications": len(frames[-1]),
                }
            )
    return pd.concat(frames, ignore_index=True), files


def _binary_codes(series: pd.Series[Any], *, yes: set[str], no: set[str]) -> pd.Series:
    text = series.map(_code)
    return pd.Series(
        np.where(text.isin(yes), 1.0, np.where(text.isin(no), 0.0, np.nan)),
        index=series.index,
        dtype=float,
    )


def _decode_age(series: pd.Series[Any]) -> pd.Series:
    text = series.map(_text)
    unit = text.str[:1]
    value = pd.to_numeric(text.str[1:], errors="coerce")
    age = pd.Series(np.nan, index=series.index, dtype=float)
    age.loc[unit.eq("4")] = value.loc[unit.eq("4")]
    age.loc[unit.isin(["1", "2", "3"])] = 0.0
    age.loc[~age.between(0, 110, inclusive="both")] = np.nan
    return age


def _decode_sinan(raw: pd.DataFrame) -> pd.DataFrame:
    raw_code = raw["SITUA_ENCE"].map(_text)
    code = raw["SITUA_ENCE"].map(_code)
    entry = raw["TRATAMENTO"].map(_code)
    form = raw["FORMA"].map(_code)
    sex = raw["CS_SEXO"].map(_text).str.upper()
    out = pd.DataFrame(index=raw.index)
    out["closure_code_raw"] = raw_code
    out["closure_code"] = code
    out["year"] = pd.to_numeric(raw["year"], errors="raise").astype(int)
    out["uf"] = raw["SG_UF"].map(_text).replace("", pd.NA)
    out["pulmonary"] = form.isin({"1", "3"})
    out["postmortem_entry"] = entry.eq("6")
    out["legacy_primary_event"] = np.where(
        raw_code.isin({"3", "4", "9"}),
        1.0,
        np.where(raw_code.eq("1"), 0.0, np.nan),
    )
    out["primary_event"] = np.where(
        code.isin({"3", "4", "9"}),
        1.0,
        np.where(code.eq("1"), 0.0, np.nan),
    )
    out["who_event"] = np.where(
        code.isin({"2", "3", "4", "9", "10"}),
        1.0,
        np.where(code.eq("1"), 0.0, np.nan),
    )
    out["death_event"] = np.where(
        code.isin({"3", "4"}),
        1.0,
        np.where(code.eq("1"), 0.0, np.nan),
    )
    out["rif_resistant"] = _binary_codes(raw["TEST_MOLEC"], yes={"2"}, no={"1"})
    out["hiv"] = _binary_codes(raw["HIV"], yes={"1"}, no={"2"})
    out["diabetes"] = _binary_codes(raw["AGRAVDIABE"], yes={"1"}, no={"2"})
    out["smear_pos"] = _binary_codes(raw["BACILOSC_E"], yes={"1"}, no={"2"})
    out["age"] = _decode_age(raw["NU_IDADE_N"])
    out["male"] = pd.Series(
        np.where(sex.eq("M"), 1.0, np.where(sex.eq("F"), 0.0, np.nan)),
        index=raw.index,
        dtype=float,
    )
    out["new_case"] = _binary_codes(raw["TRATAMENTO"], yes={"1"}, no={"2", "3"})
    return out


def _tb_source(data: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    context = data.context.copy()
    resistance = context["type_of_resistance"].astype("string").str.strip()
    gender = context["gender"].astype("string").str.strip().str.casefold()
    smear = context["baseline_smear"].astype("string").str.strip().str.casefold()
    case = context["case_definition"].astype("string").str.strip()
    comorb_missing = pd.to_numeric(context["comorb_missing"], errors="coerce").eq(1)
    harmonized = pd.DataFrame(index=context.index)
    harmonized["rif_resistant"] = resistance.isin({"MDR non XDR", "Pre-XDR", "XDR"}).astype(float)
    for raw_name, shared_name in (("comorb_hiv", "hiv"), ("comorb_diabetes", "diabetes")):
        values = pd.to_numeric(context[raw_name], errors="coerce").astype(float)
        values.loc[comorb_missing] = np.nan
        harmonized[shared_name] = values
    harmonized["smear_pos"] = np.where(
        smear.eq("positive"), 1.0, np.where(smear.eq("negative"), 0.0, np.nan)
    )
    harmonized["age"] = pd.to_numeric(context["age_of_onset"], errors="coerce")
    harmonized["male"] = np.where(
        gender.eq("male"), 1.0, np.where(gender.eq("female"), 0.0, np.nan)
    )
    unknown_case = case.isna() | case.str.casefold().isin({"", "unknown", "not reported", "<na>"})
    harmonized["new_case"] = np.where(
        unknown_case,
        np.nan,
        case.str.contains("new", case=False, na=False).astype(float),
    )
    treatment_rows: list[list[float]] = []
    for regimen in data.treatments.tolist():
        drugs = set(regimen)
        treatment_rows.append(
            [
                *[float(drug in drugs) for drug in FIRST_LINE],
                float(bool(drugs.difference(FIRST_LINE))),
            ]
        )
    treatment = np.asarray(treatment_rows, dtype=np.float64)
    outcome = np.asarray(data.outcome, dtype=np.int64)
    return harmonized.loc[:, FEATURES], treatment, outcome, context


def _preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def _summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Summary values must be a finite nonempty vector")
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _soft_rate(weights: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & np.isfinite(values)
    denominator = float(weights[usable].sum())
    if denominator < SMALL_CELL_MIN:
        raise ValueError("Soft profile summary violates the small-cell threshold")
    return float(np.dot(weights[usable], values[usable]) / denominator)


def _profile_rows(
    loadings: np.ndarray,
    event: np.ndarray,
    hiv: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    hard = loadings.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for profile in range(loadings.shape[1]):
        weights = loadings[:, profile]
        hard_mask = mask & (hard == profile)
        hard_count = int(hard_mask.sum())
        hard_events = int(np.nansum(event[hard_mask]))
        hard_nonevents = hard_count - hard_events
        hard_safe = min(hard_count, hard_events, hard_nonevents) >= SMALL_CELL_MIN
        rows.append(
            {
                "profile_risk_rank": profile + 1,
                "soft_effective_count": float(weights[mask].sum()),
                "soft_event_rate": _soft_rate(weights, event, mask),
                "soft_hiv_prevalence_among_known": _soft_rate(
                    weights,
                    hiv,
                    mask & np.isfinite(hiv),
                ),
                "hard_count_secondary": _safe_count(hard_count),
                "hard_event_rate_secondary": (
                    float(hard_events / hard_count) if hard_safe else None
                ),
                "hard_rate_suppressed": not hard_safe,
            }
        )
    return rows


def _cluster_bootstrap_pair(
    event: np.ndarray,
    reference_score: np.ndarray,
    patch_score: np.ndarray,
    clusters: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.asarray(clusters).astype(str)
    unique = np.unique(groups)
    grouped_rows = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    ref_values: list[float] = []
    patch_values: list[float] = []
    differences: list[float] = []
    for _ in range(resamples):
        sampled = np.concatenate(
            [grouped_rows[position] for position in rng.integers(0, len(unique), len(unique))]
        )
        if len(np.unique(event[sampled])) != 2:
            continue
        ref = float(roc_auc_score(event[sampled], reference_score[sampled]))
        patch = float(roc_auc_score(event[sampled], patch_score[sampled]))
        ref_values.append(ref)
        patch_values.append(patch)
        differences.append(patch - ref)
    if len(differences) < int(0.9 * resamples):
        raise RuntimeError("Too many geographic-year bootstrap replicates lost one event class")
    return {
        "resampling_unit": "Brazil state by notification year",
        "requested_resamples": resamples,
        "completed_resamples": len(differences),
        "context_logistic_auroc": _summary(ref_values),
        "patch_profile_score_auroc": _summary(patch_values),
        "paired_difference_patch_minus_context_logistic": _summary(differences),
    }


def _calibrator(score: np.ndarray, event: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000).fit(score.reshape(-1, 1), event)


def _calibration_summary(event: np.ndarray, fitted_risk: np.ndarray) -> dict[str, float]:
    clipped = np.clip(fitted_risk, 1e-6, 1 - 1e-6)
    logit_score = np.log(clipped / (1 - clipped))
    calibration = _calibrator(logit_score, event)
    return {
        "brier": float(brier_score_loss(event, fitted_risk)),
        "log_loss": float(log_loss(event, fitted_risk)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def _net_benefit(event: np.ndarray, fitted_risk: np.ndarray, threshold: float) -> float:
    positive = fitted_risk >= threshold
    true_positive = int(np.sum(positive & (event == 1)))
    false_positive = int(np.sum(positive & (event == 0)))
    return float(
        true_positive / len(event) - false_positive / len(event) * threshold / (1.0 - threshold)
    )


def _temporal_recalibration(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    year = frame["year"].to_numpy(dtype=int)
    event = frame["primary_event"].to_numpy(dtype=float)
    calibration_mask = mask & (year <= 2017)
    evaluation_mask = mask & (year >= 2018)
    y_cal = event[calibration_mask].astype(int)
    y_eval = event[evaluation_mask].astype(int)
    fitted: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, score in scores.items():
        model = _calibrator(score[calibration_mask], y_cal)
        risk = model.predict_proba(score[evaluation_mask].reshape(-1, 1))[:, 1]
        fitted[name] = risk
        metrics[name] = {
            **_metric(y_eval, risk),
            **_calibration_summary(y_eval, risk),
        }
    thresholds = np.arange(0.02, 0.301, 0.01)
    patch = np.asarray(
        [_net_benefit(y_eval, fitted["patch_profile_score"], value) for value in thresholds]
    )
    context = np.asarray(
        [_net_benefit(y_eval, fitted["context_logistic"], value) for value in thresholds]
    )
    band = (thresholds >= 0.05) & (thresholds <= 0.15)
    return {
        "calibration_period": "2015-2017",
        "evaluation_period": "2018-2019",
        "calibration_conditions": int(calibration_mask.sum()),
        "evaluation_conditions": int(evaluation_mask.sum()),
        "models": metrics,
        "exploratory_decision_curve": {
            "threshold_range": [0.02, 0.30],
            "prespecified_summary_band": [0.05, 0.15],
            "mean_net_benefit_difference_patch_minus_context_logistic": float(
                np.mean((patch - context)[band])
            ),
            "differences_at_selected_thresholds": {
                format(value, ".2f"): float(
                    patch[np.argmin(np.abs(thresholds - value))]
                    - context[np.argmin(np.abs(thresholds - value))]
                )
                for value in (0.05, 0.10, 0.15)
            },
        },
    }


def _feature_summary(source: pd.DataFrame, brazil: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        source_values = source[feature].to_numpy(dtype=float)
        brazil_values = brazil[feature].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "feature": feature,
            "source_missing_fraction": float(np.mean(~np.isfinite(source_values))),
            "brazil_missing_fraction": float(np.mean(~np.isfinite(brazil_values))),
        }
        if feature == "age":
            row.update(
                {
                    "source_median_among_known": float(np.nanmedian(source_values)),
                    "brazil_median_among_known": float(np.nanmedian(brazil_values)),
                    "brazil_fraction_outside_source_observed_range": float(
                        np.mean(
                            (brazil_values < np.nanmin(source_values))
                            | (brazil_values > np.nanmax(source_values))
                        )
                    ),
                }
            )
        else:
            row.update(
                {
                    "source_prevalence_among_known": float(np.nanmean(source_values)),
                    "brazil_prevalence_among_known": float(np.nanmean(brazil_values)),
                }
            )
        rows.append(row)
    return rows


def main() -> int:
    args = _arguments()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
    logging.getLogger("ctb").setLevel(logging.ERROR)
    tb_raw = args.tb_raw_dir.expanduser().resolve()
    sinan_src = args.sinan_src.expanduser().resolve()
    helper = args.decompress_helper.expanduser().resolve()
    for path, label in (
        (tb_raw, "TB Portals directory"),
        (sinan_src, "SINAN directory"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist")
    if not helper.is_file():
        raise FileNotFoundError("SINAN decompression helper does not exist")

    config = LockedValidationConfig()
    set_global_seed(config.seed)
    patient_cases = clean_patient_cases(load_patient_cases(tb_raw))
    regimens = clean_regimens(load_regimens(tb_raw))
    biochemistry = load_biochemistry(tb_raw)
    data = build_schema6_raw_validation_data(patient_cases, regimens, biochemistry, config)
    source_raw, source_treatment, source_event, _ = _tb_source(data)

    sinan_raw, source_files = _load_sinan(
        sinan_src,
        helper=helper,
    )
    brazil_all = _decode_sinan(sinan_raw)
    brazil = brazil_all.loc[brazil_all["pulmonary"]].reset_index(drop=True)
    legacy_primary_mask = np.isfinite(brazil["legacy_primary_event"].to_numpy(dtype=float))
    primary_mask = np.isfinite(brazil["primary_event"].to_numpy(dtype=float))
    strict_primary_mask = primary_mask & ~brazil["postmortem_entry"].to_numpy(dtype=bool)
    who_mask = np.isfinite(brazil["who_event"].to_numpy(dtype=float)) & ~brazil[
        "postmortem_entry"
    ].to_numpy(dtype=bool)
    death_mask = np.isfinite(brazil["death_event"].to_numpy(dtype=float)) & ~brazil[
        "postmortem_entry"
    ].to_numpy(dtype=bool)
    if int(legacy_primary_mask.sum()) != 298900:
        raise ValueError("From-source rebuild does not reproduce the manuscript Brazil cohort size")

    encoder = _preprocessor()
    source_c = encoder.fit_transform(source_raw)
    brazil_c = encoder.transform(brazil.loc[:, FEATURES])
    context_model = LogisticRegression(C=0.1, max_iter=2000, random_state=config.seed).fit(
        source_c, source_event
    )
    context_score = context_model.predict_proba(brazil_c)[:, 1]
    resistance_model = LogisticRegression(
        C=0.1,
        max_iter=2000,
        random_state=config.seed,
    ).fit(source_c[:, [0]], source_event)
    resistance_score = resistance_model.predict_proba(brazil_c[:, [0]])[:, 1]

    hiv_raw = brazil["hiv"].to_numpy(dtype=float)
    primary_event = brazil["primary_event"].to_numpy(dtype=float)
    source_hiv = source_raw["hiv"].to_numpy(dtype=float)
    seed_profile_scores: dict[int, np.ndarray] = {}
    seed_reports: list[dict[str, Any]] = []
    for seed in PROFILE_SEEDS:
        model = LinearArchetypeContextualized(
            source_c.shape[1],
            source_treatment.shape[1],
            n_archetypes=5,
            simplex=True,
            seed=seed,
        )
        train(
            model,
            source_c,
            source_treatment,
            source_event.astype(np.float64),
            PROFILE_CONFIG,
            seed=seed,
        )
        source_loadings = model.predict_loadings(source_c)
        brazil_loadings = model.predict_loadings(brazil_c)
        source_rates = np.asarray(
            [
                np.dot(source_loadings[:, profile], source_event)
                / source_loadings[:, profile].sum()
                for profile in range(5)
            ],
            dtype=float,
        )
        order = np.argsort(source_rates, kind="mergesort")
        ordered_source = source_loadings[:, order]
        ordered_brazil = brazil_loadings[:, order]
        ordered_rates = source_rates[order]
        profile_score = ordered_brazil @ ordered_rates
        seed_profile_scores[seed] = profile_score
        source_profile = _profile_rows(
            ordered_source,
            source_event.astype(float),
            source_hiv,
            np.ones(len(source_event), dtype=bool),
        )
        brazil_profile = _profile_rows(
            ordered_brazil,
            primary_event,
            hiv_raw,
            strict_primary_mask,
        )
        brazil_rates = np.asarray([row["soft_event_rate"] for row in brazil_profile])
        source_hard_rates = np.asarray(
            [row["hard_event_rate_secondary"] for row in source_profile], dtype=float
        )
        brazil_hard_rates = np.asarray(
            [row["hard_event_rate_secondary"] for row in brazil_profile], dtype=float
        )
        source_hiv_rates = np.asarray(
            [row["soft_hiv_prevalence_among_known"] for row in source_profile], dtype=float
        )
        hiv_profile = int(np.argmax(source_hiv_rates))
        yearly: list[dict[str, Any]] = []
        for year in YEARS:
            year_mask = strict_primary_mask & brazil["year"].eq(year).to_numpy(dtype=bool)
            year_rates = np.asarray(
                [
                    _soft_rate(ordered_brazil[:, profile], primary_event, year_mask)
                    for profile in range(5)
                ]
            )
            yearly.append(
                {
                    "year": year,
                    "conditions": int(year_mask.sum()),
                    "profile_risk_spearman_vs_source": float(
                        spearmanr(ordered_rates, year_rates).statistic
                    ),
                    "source_highest_risk_profile_remains_highest": bool(
                        int(np.argmax(year_rates)) == 4
                    ),
                    "patch_profile_score_auroc": float(
                        roc_auc_score(
                            primary_event[year_mask].astype(int),
                            profile_score[year_mask],
                        )
                    ),
                }
            )
        seed_reports.append(
            {
                "seed": seed,
                "source_profile_summary": source_profile,
                "brazil_strict_profile_summary": brazil_profile,
                "external_metrics": _metric(
                    primary_event[strict_primary_mask].astype(int),
                    profile_score[strict_primary_mask],
                ),
                "profile_risk_spearman_source_vs_brazil": float(
                    spearmanr(ordered_rates, brazil_rates).statistic
                ),
                "source_highest_risk_profile_remains_highest_in_brazil": bool(
                    int(np.argmax(brazil_rates)) == 4
                ),
                "hard_profile_risk_spearman_source_vs_brazil": float(
                    spearmanr(source_hard_rates, brazil_hard_rates).statistic
                ),
                "source_hard_highest_risk_profile_remains_highest_in_brazil": bool(
                    int(np.argmax(source_hard_rates)) == int(np.argmax(brazil_hard_rates))
                ),
                "source_hiv_enriched_profile": {
                    "profile_risk_rank": hiv_profile + 1,
                    "highest_soft_risk_in_source": bool(
                        hiv_profile == int(np.argmax(ordered_rates))
                    ),
                    "highest_hard_risk_in_source": bool(
                        hiv_profile == int(np.argmax(source_hard_rates))
                    ),
                    "highest_soft_risk_in_brazil": bool(
                        hiv_profile == int(np.argmax(brazil_rates))
                    ),
                    "highest_hard_risk_in_brazil": bool(
                        hiv_profile == int(np.argmax(brazil_hard_rates))
                    ),
                },
                "yearly_transport": yearly,
            }
        )

    patch_score = seed_profile_scores[LOCKED_EXTERNAL_SEED]
    ensemble_score = np.mean(np.stack(list(seed_profile_scores.values())), axis=0)
    models = {
        "resistance_only": resistance_score,
        "context_logistic": context_score,
        "patch_profile_score": patch_score,
        "patch_profile_seed9_sensitivity": seed_profile_scores[9],
    }

    def endpoint_metrics(event_column: str, mask: np.ndarray) -> dict[str, Any]:
        event = brazil[event_column].to_numpy(dtype=float)[mask].astype(int)
        return {name: _metric(event, score[mask]) for name, score in models.items()}

    strict_event = primary_event[strict_primary_mask].astype(int)
    clusters = (
        brazil.loc[strict_primary_mask, "uf"].astype("string").fillna("unknown")
        + "_"
        + brazil.loc[strict_primary_mask, "year"].astype(str)
    ).to_numpy(dtype=str)
    bootstrap = _cluster_bootstrap_pair(
        strict_event,
        context_score[strict_primary_mask],
        patch_score[strict_primary_mask],
        clusters,
        resamples=args.bootstrap_resamples,
        seed=20260827,
    )
    seed9_bootstrap = _cluster_bootstrap_pair(
        strict_event,
        context_score[strict_primary_mask],
        seed_profile_scores[9][strict_primary_mask],
        clusters,
        resamples=args.bootstrap_resamples,
        seed=20260828,
    )

    closure_counts = (
        brazil["closure_code"].replace("", "blank").value_counts(dropna=False).sort_index()
    )
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "analysis_id": "sinan_from_source_prognostic_transport_audit",
        "analysis_role": "analysis-only remediation; no submission artifact was modified",
        "source_files": source_files,
        "official_dictionary": {
            "title": "SINAN Tuberculosis Data Dictionary",
            "url": "https://portalsinan.saude.gov.br/images/documentos/Agravos/Tuberculose/DICI_DADOS_Tuberculose.pdf",
            "coding_corrections_to_legacy_builder": [
                "Numeric category strings are normalized, so 03 and 04 map to 3 and 4.",
                "SITUA_ENCE 8 is change of regimen, not loss to follow-up.",
                "SITUA_ENCE 9 is treatment failure.",
                "SITUA_ENCE 10 is primary abandonment.",
                "Unknown or not-performed binary fields are missing, not negative.",
                "Unknown age units are missing, not age zero.",
            ],
        },
        "cohort_rebuild": {
            "years": list(YEARS),
            "all_notifications": len(brazil_all),
            "pulmonary_notifications": len(brazil),
            "legacy_manuscript_primary_conditions": int(legacy_primary_mask.sum()),
            "legacy_manuscript_primary_cohort_reproduced_exactly": bool(
                legacy_primary_mask.sum() == 298900
            ),
            "corrected_primary_conditions": int(primary_mask.sum()),
            "corrected_records_added_by_numeric_code_normalization": int(
                primary_mask.sum() - legacy_primary_mask.sum()
            ),
            "strict_primary_conditions_excluding_postmortem_entry": int(strict_primary_mask.sum()),
            "strict_who_composite_conditions": int(who_mask.sum()),
            "strict_death_only_conditions": int(death_mask.sum()),
            "closure_code_counts_in_pulmonary_notifications": {
                str(code): _safe_count(int(count)) for code, count in closure_counts.items()
            },
        },
        "endpoint_definitions": {
            "primary": "SINAN codes 3, 4, or 9 versus code 1",
            "primary_labels": "death due to TB, death due to other causes, or failure versus cure",
            "strict_primary_sensitivity": "primary with postmortem entries excluded",
            "who_composite_sensitivity": (
                "codes 2, 3, 4, 9, or 10 versus code 1, with postmortem entries excluded"
            ),
            "death_only_sensitivity": (
                "codes 3 or 4 versus code 1, with postmortem entries excluded"
            ),
            "cross_registry_mismatch": (
                "SINAN has cure but no separate completed category; TB Portals non-events include "
                "recorded cured and completed"
            ),
        },
        "harmonization": {
            "features": list(FEATURES),
            "source_only_imputation_and_scaling": True,
            "feature_shift": _feature_summary(source_raw, brazil.loc[:, FEATURES]),
            "brazil_regimen_assigned": False,
            "treatment_layer_externally_validated": False,
            "claim_scope": "prognostic profile transport only",
        },
        "tb_portals_source": {
            "locked_conditions": len(source_event),
            "events": int(source_event.sum()),
            "harmonized_context_features": len(FEATURES),
            "collapsed_observed_regimen_features": [*FIRST_LINE, "other_or_second_line"],
        },
        "external_discrimination": {
            "primary_patch_estimator_seed": LOCKED_EXTERNAL_SEED,
            "manuscript_primary_all_entries": endpoint_metrics("primary_event", primary_mask),
            "legacy_manuscript_primary_literal_code_reproduction": endpoint_metrics(
                "legacy_primary_event", legacy_primary_mask
            ),
            "strict_primary_excluding_postmortem": endpoint_metrics(
                "primary_event", strict_primary_mask
            ),
            "strict_who_composite": endpoint_metrics("who_event", who_mask),
            "strict_death_only": endpoint_metrics("death_event", death_mask),
            "seedwise_patch_profile_auroc": _summary(
                [row["external_metrics"]["auroc"] for row in seed_reports]
            ),
            "posthoc_seed_ensemble_diagnostic": _metric(
                primary_event[strict_primary_mask].astype(int),
                ensemble_score[strict_primary_mask],
            ),
            "paired_geographic_year_bootstrap": bootstrap,
            "paired_geographic_year_bootstrap_seed9_sensitivity": seed9_bootstrap,
        },
        "profile_transport": {
            "profile_ordering": "ordered by source soft event rate independently within seed",
            "seed_roles": {
                "0": "locked legacy external-validation estimator",
                "9": "legacy full-profile figure seed selected for balance",
                "others": "optimization sensitivity only",
            },
            "seeds": seed_reports,
            "seeds_preserving_source_highest_risk_profile_as_highest_in_brazil": int(
                sum(
                    row["source_highest_risk_profile_remains_highest_in_brazil"]
                    for row in seed_reports
                )
            ),
            "source_to_brazil_profile_risk_spearman": _summary(
                [row["profile_risk_spearman_source_vs_brazil"] for row in seed_reports]
            ),
            "hard_profile_risk_spearman_source_vs_brazil": _summary(
                [row["hard_profile_risk_spearman_source_vs_brazil"] for row in seed_reports]
            ),
            "seeds_preserving_source_hard_highest_profile_in_brazil": int(
                sum(
                    row["source_hard_highest_risk_profile_remains_highest_in_brazil"]
                    for row in seed_reports
                )
            ),
            "seeds_where_source_hiv_enriched_profile_is_highest_soft_risk_in_both": int(
                sum(
                    row["source_hiv_enriched_profile"]["highest_soft_risk_in_source"]
                    and row["source_hiv_enriched_profile"]["highest_soft_risk_in_brazil"]
                    for row in seed_reports
                )
            ),
            "seeds_where_source_hiv_enriched_profile_is_highest_hard_risk_in_both": int(
                sum(
                    row["source_hiv_enriched_profile"]["highest_hard_risk_in_source"]
                    and row["source_hiv_enriched_profile"]["highest_hard_risk_in_brazil"]
                    for row in seed_reports
                )
            ),
        },
        "temporal_recalibration": _temporal_recalibration(
            brazil,
            strict_primary_mask,
            models,
        ),
        "interpretive_limits": [
            "Brazil validates only prognostic transport on seven harmonized fields.",
            "SINAN does not provide an observed regimen vector suitable for this PATCH treatment layer.",
            "The assigned-RHZE analysis in the legacy script was not rerun and must not support treatment-response claims.",
            "HIV may be recorded after notification and has no field-specific date in the downloaded extract.",
            "SINAN notifications do not provide a patient identifier suitable for patient-cluster resampling.",
            "Target-year recalibration uses 2015-2017 Brazil labels and is evaluated only in 2018-2019.",
            "The seed ensemble is a post hoc stability diagnostic and is not a claim-bearing estimator.",
        ],
        "governance": {
            "row_level_objects_serialized": False,
            "identifiers_serialized": False,
            "temporary_decompressed_files_deleted": True,
            "small_cell_threshold": SMALL_CELL_MIN,
        },
    }
    _write_report(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
