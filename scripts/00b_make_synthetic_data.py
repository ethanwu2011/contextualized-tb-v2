"""Generate SYNTHETIC, schema-faithful TB Portals CSVs so the pipeline runs without real data.

THE OUTPUT OF THIS SCRIPT IS ENTIRELY FAKE. It exists only so that this repository is runnable
end-to-end by anyone (e.g. reviewers) who has NOT executed the NIH TB Portals Data Use Agreement.
The real, access-gated row-level data is governed by a DUA and is never committed to this repo
(see ``scripts/00_download_data.py`` and the README). Every value below is drawn from a random
number generator; column *names* and *value vocabularies* mirror the August-2023 export so that
``ctb.data.load`` / ``clean`` / ``context`` / ``conversion`` parse it, but **no statistic, subtype,
drug effect, or figure computed from this synthetic data carries any biological meaning**.

The five tables written match :data:`ctb.config.TB_PORTALS_FILES`:
``patient_cases``, ``regimens``, ``specimen``, ``biochemistry``, ``dst``.

By default this writes to ``data/raw/synthetic/`` — NOT the real-data location ``RAW_DIR``. To run
the pipeline on the synthetic data, point the loaders at it via the environment variable:

    CTB_RAW_DIR=data/raw/synthetic uv run python scripts/01_build_dataset.py

The generator REFUSES to write into ``RAW_DIR`` (or any directory that already holds a real TB
Portals export) unless given ``--force``, so it can never clobber the access-gated source data.

Usage
-----
    uv run python scripts/00b_make_synthetic_data.py            # -> data/raw/synthetic/
    uv run python scripts/00b_make_synthetic_data.py --n 3000   # custom cohort size
    uv run python scripts/00b_make_synthetic_data.py --out /tmp/syn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ctb.config import PROJECT_ROOT, RAW_DIR, TB_PORTALS_FILES
from ctb.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# Default synthetic output: a dedicated directory that is NEVER the real-data RAW_DIR.
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"


def _assert_safe_output(out: Path, force: bool) -> None:
    """Abort if ``out`` is (or holds) the real, access-gated TB Portals export.

    Guards against the failure mode where ``out`` resolves — possibly through a symlink — to the
    real-data ``RAW_DIR``, or already contains a full-size real Patient Cases file. Synthetic data
    must never overwrite DUA-restricted source files.

    Parameters
    ----------
    out : pathlib.Path
        Requested output directory.
    force : bool
        If True, bypass the guard (explicit opt-in only).

    Raises
    ------
    SystemExit
        If the target looks like the real-data location and ``force`` is not set.
    """
    if force:
        logger.warning("--force set: safety guard bypassed; writing to %s", out)
        return
    try:
        out_real = out.resolve()
        raw_real = RAW_DIR.resolve()
    except OSError:
        out_real, raw_real = out, RAW_DIR
    if out_real == raw_real or raw_real in out_real.parents or out_real in raw_real.parents:
        raise SystemExit(
            f"REFUSING to write synthetic data to {out} — it resolves to the real-data RAW_DIR "
            f"({raw_real}). Use the default (data/raw/synthetic) or pass --force only if you are "
            f"certain no real data lives there."
        )
    pc = out / TB_PORTALS_FILES["patient_cases"]
    if pc.exists():
        with pc.open() as fh:
            n_lines = sum(1 for _ in fh)
        if n_lines > 5000:
            raise SystemExit(
                f"REFUSING to overwrite {pc}: it has {n_lines} rows, which looks like the REAL "
                f"TB Portals export (synthetic cohorts are a few thousand). Pass --force to override."
            )

# --- Synthetic value vocabularies (mirror the real export's categories) -------------------------
COUNTRIES = ["Belarus", "Ukraine", "Georgia", "Moldova", "Romania", "Azerbaijan",
             "Kazakhstan", "India", "Nigeria", "South Africa"]
RESISTANCE = ["Sensitive", "Mono DR", "Poly DR", "MDR non XDR", "Pre-XDR", "XDR"]
RESISTANCE_P = [0.34, 0.07, 0.06, 0.34, 0.10, 0.09]
GENDERS = ["Male", "Female"]
EDUCATION = ["Secondary school", "Higher education", "Primary education",
             "Vocational education", "Not Reported"]
EMPLOYMENT = ["Unemployed", "Employed", "Disabled", "Retired", "Student", "Not Reported"]
CASE_DEF = ["New", "Relapse", "Treatment after failure",
            "Treatment after loss to follow up", "Other"]
LUNG_LOC = ["Bilateral", "Left", "Right", "Not Reported"]
DISSEMINATION = ["Limited", "Widespread", "Not Reported"]
AFFECT_PLEURA = ["Yes", "No", "Not Reported"]
SHADOW = ["Focus", "Infiltrate", "Cavity", "Mixed", "Not Reported"]
MAIN_LINEAGE = ["Lineage 2", "Lineage 4", "Lineage 3", "Lineage 1", "Not Reported"]
SUB_LINEAGE = ["Beijing", "Euro-American", "Delhi-CAS", "Ural", "Haarlem", "Not Reported"]
CAVITY = ["Multiple cavities", "Single cavity", "No cavities", "Not Reported"]
MICRO_POS = ["1 to 9 per 100", "10 to 99 per 100", "Positive", "1+", "2+", "3+"]
CULTURE_POS = ["Positive", "1 to 19", "20 to 100", "100 to 200", "more than 200"]
COMORB_TOKENS = ["HIV", "Diabetes", "Hepatitis C", "Hepatitis B", "Anemia",
                 "Psychiatric disorder", "Renal disease", "Hepatic disease"]
BACTEC_COLS = ["bactec_isoniazid", "bactec_rifampicin", "bactec_ethambutol",
               "bactec_streptomycin", "bactec_ofloxacin", "bactec_amikacin",
               "bactec_capreomycin", "bactec_kanamycin", "bactec_moxifloxacin",
               "bactec_pyrazinamide", "bactec_mycobutin", "bactec_imipenem_cilastatin"]
FIRST_LINE = ["H", "R", "Z", "E"]
SECOND_LINE = ["Bdq", "Lzd", "Lfx", "Mfx", "Cfz", "Cs", "Cm", "Am", "Km",
               "Pto", "Eto", "Pas", "Dld", "Pa", "Trd", "S"]
RARE_DRUGS = ["Ipm/Cln", "Amx/Clv", "Ofx", "Cotrimoxazol", "ART"]
OUTCOMES = ["cured", "completed", "died", "failure", "ltfu", "still on treatment"]
STATUSREASONS = ["", "resistance", "adverse", "newDrugAvailable", "completed", "other"]
LAB_COLS = ["Alanine Aminotransferase (u/l)", "Aspartate Aminotransferase (u/l)",
            "Creatinine (umol/l)", "Glucose (mmol/l)", "Albumin (g/l)", "Hemoglobin (g/l)"]


def _regimen_for(res: str, rng: np.random.Generator) -> str:
    """Draw a plausible brace-wrapped initial regimen string for a resistance class."""
    if res in {"Sensitive", "Mono DR"}:
        drugs = list(FIRST_LINE)
        if rng.random() < 0.15:
            drugs.append(rng.choice(SECOND_LINE))
    else:
        k = int(rng.integers(4, 6))
        drugs = list(rng.choice(SECOND_LINE, size=k, replace=False))
        if rng.random() < 0.4:
            drugs.append(rng.choice(["H", "Z", "E"]))
    if rng.random() < 0.05:  # occasional rare drug -> exercises the "other_second_line" bucket
        drugs.append(rng.choice(RARE_DRUGS))
    rng.shuffle(drugs)
    return "{" + ",".join(dict.fromkeys(drugs)) + "}"


def _comorbidity(rng: np.random.Generator) -> object:
    """Draw a free-text comorbidity string (or NaN for a missing field)."""
    if rng.random() < 0.2:
        return np.nan
    present = [t for t in COMORB_TOKENS if rng.random() < 0.12]
    return ", ".join(present) if present else "None"


def make_patient_cases(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build the synthetic Patient Cases table (one row per condition/episode)."""
    # A minority of patients contribute two episodes (exercises the re-initiation logic).
    n_patients = int(n * 0.92)
    patient_ids = [f"SYN-P{idx:06d}" for idx in range(n_patients)]
    pid = list(rng.choice(patient_ids, size=n, replace=True))
    res = rng.choice(RESISTANCE, size=n, p=RESISTANCE_P)
    years = rng.choice([2018, 2019, 2020, 2021, 2022], size=n, p=[0.15, 0.2, 0.25, 0.22, 0.18])
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 28, size=n)
    reg_dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days, strict=True)]

    df = pd.DataFrame({
        "patient_id": pid,
        "condition_id": [f"SYN-C{idx:06d}" for idx in range(n)],
        "registration_date": reg_dates,
        "country": rng.choice(COUNTRIES, size=n),
        "age_of_onset": np.clip(rng.normal(42, 15, n).round(0), 16, 90),
        "bmi": np.clip(rng.normal(22, 4, n).round(1), 12, 45),
        "number_of_children": rng.integers(0, 5, n),
        "number_of_daily_contacts": rng.integers(0, 12, n),
        "gender": rng.choice(GENDERS, size=n, p=[0.65, 0.35]),
        "education": rng.choice(EDUCATION, size=n),
        "employment": rng.choice(EMPLOYMENT, size=n),
        "comorbidity": [_comorbidity(rng) for _ in range(n)],
        "type_of_resistance": res,
        "case_definition": rng.choice(CASE_DEF, size=n),
        "lung_localization": rng.choice(LUNG_LOC, size=n),
        "lungcavity_size": rng.choice(CAVITY, size=n),
        "culture": rng.choice(CULTURE_POS + ["Negative", "Not Reported"], size=n),
        "microscopy": rng.choice(MICRO_POS + ["Negative", "Not Reported"], size=n),
        "timika_score": np.clip(rng.normal(40, 25, n).round(0), 0, 140),
        "totalcavernum": rng.integers(0, 4, n),
        "overall_percent_of_abnormal_volume": np.clip(rng.normal(25, 18, n).round(0), 0, 100),
        "pleural_effusion_percent_of_hemithorax_involved": rng.integers(0, 3, n),
        "dissemination": rng.choice(DISSEMINATION, size=n),
        "affect_pleura": rng.choice(AFFECT_PLEURA, size=n),
        "shadow_pattern": rng.choice(SHADOW, size=n),
        "main_lineage": rng.choice(MAIN_LINEAGE, size=n),
        "sub_lineage": rng.choice(SUB_LINEAGE, size=n),
    })
    for col in BACTEC_COLS:
        # Susceptible-leaning for first-line drugs; mostly "S"/"R" with some missing.
        df[col] = rng.choice(["S", "R", "Not Reported"], size=n, p=[0.55, 0.3, 0.15])
    return df


def make_regimens(pc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build the synthetic Regimens table (1-3 lines per condition)."""
    rows: list[dict[str, object]] = []
    for cid, res, regdate in zip(pc["condition_id"], pc["type_of_resistance"],
                                 pc["registration_date"], strict=True):
        n_lines = 1 + int(rng.random() < 0.25) + int(rng.random() < 0.1)
        base_days = int(rng.normal(190 if res in {"Sensitive", "Mono DR"} else 520, 90))
        start = pd.Timestamp(regdate)
        outcome = rng.choice(OUTCOMES, p=[0.40, 0.22, 0.10, 0.06, 0.12, 0.10])
        for line in range(n_lines):
            span = max(30, int(base_days / n_lines + rng.normal(0, 30)))
            end = start + pd.Timedelta(days=span)
            rows.append({
                "condition_id": cid,
                "registration_date": regdate,
                "regimen_drug": _regimen_for(res, rng),
                "period_span": span,
                "period_start": start.date().isoformat(),
                "period_end": end.date().isoformat(),
                "activities_period_start": start.date().isoformat(),
                "activities_period_end": end.date().isoformat(),
                "activities_statusreason_cd": rng.choice(STATUSREASONS),
                # Outcome recorded on the last line only; earlier lines blank.
                "outcome_cd": outcome if line == n_lines - 1 else "",
            })
            start = end
    return pd.DataFrame(rows)


def make_dst(pc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build the synthetic DST table with serial cultures (integer day-offsets)."""
    rows: list[dict[str, object]] = []
    for cid in pc["condition_id"]:
        # Most episodes start culture-positive (the conversion-estimable population), then
        # serially convert to negative at a random month.
        n_cult = int(rng.integers(2, 7))
        offsets = sorted(int(d) for d in rng.choice(range(0, 400), size=n_cult, replace=False))
        convert_at = rng.integers(30, 200)
        for day in offsets:
            if day == 0 or rng.random() < 0.85:
                status = rng.choice(CULTURE_POS) if day < convert_at else "Negative"
            else:
                status = rng.choice(["MOTT", "Nonspecific microflora", "Unknown result"])
            rows.append({
                "condition_id": cid,
                "specimen_collection_date": day,   # integer day-offset, per real schema
                "test_date": day,
                "cultureresults": status,
            })
    return pd.DataFrame(rows)


def make_biochemistry(pc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build the synthetic Biochemistry (longitudinal labs) table."""
    rows: list[dict[str, object]] = []
    means = {"Alanine Aminotransferase (u/l)": 30, "Aspartate Aminotransferase (u/l)": 32,
             "Creatinine (umol/l)": 80, "Glucose (mmol/l)": 5.5, "Albumin (g/l)": 38,
             "Hemoglobin (g/l)": 130}
    for cid, regdate in zip(pc["condition_id"], pc["registration_date"], strict=True):
        n_tests = int(rng.integers(1, 4))
        start = pd.Timestamp(regdate)
        for t in range(n_tests):
            date = (start + pd.Timedelta(days=int(t * 30))).date().isoformat()
            row: dict[str, object] = {"condition_id": cid, "test_date": date,
                                      "specimen_collection_date": date}
            for lab, mu in means.items():
                # Hemoglobin is ~92% missing in the real export; mirror that sparsity.
                if lab == "Hemoglobin (g/l)" and rng.random() < 0.92:
                    row[lab] = np.nan
                else:
                    row[lab] = round(float(rng.normal(mu, mu * 0.25)), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def make_specimen(pc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build a minimal synthetic Specimen table."""
    rows: list[dict[str, object]] = []
    for cid in pc["condition_id"]:
        for _ in range(int(rng.integers(1, 4))):
            day = int(rng.integers(0, 400))
            rows.append({"condition_id": cid, "specimen_collection_date": day, "test_date": day,
                         "specimen_type": rng.choice(["Sputum", "Other"])})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Generate and write the five synthetic TB Portals CSVs."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC TB Portals CSVs.")
    parser.add_argument("--n", type=int, default=2500, help="number of synthetic conditions")
    parser.add_argument("--out", type=Path, default=SYNTHETIC_DIR,
                        help="output directory (default: data/raw/synthetic; never RAW_DIR)")
    parser.add_argument("--seed", type=int, default=20230815)
    parser.add_argument("--force", action="store_true",
                        help="bypass the real-data safety guard (dangerous; off by default)")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out: Path = args.out
    _assert_safe_output(out, args.force)
    out.mkdir(parents=True, exist_ok=True)

    pc = make_patient_cases(args.n, rng)
    tables = {
        "patient_cases": pc,
        "regimens": make_regimens(pc, rng),
        "dst": make_dst(pc, rng),
        "biochemistry": make_biochemistry(pc, rng),
        "specimen": make_specimen(pc, rng),
    }
    for key, df in tables.items():
        path = out / TB_PORTALS_FILES[key]
        df.to_csv(path, index=False)
        logger.info("Wrote SYNTHETIC %-13s %6d rows x %2d cols -> %s",
                    key, df.shape[0], df.shape[1], path.name)

    (out / "SYNTHETIC_DATA_README.txt").write_text(
        "These CSV files are 100% SYNTHETIC, generated by scripts/00b_make_synthetic_data.py.\n"
        "They contain NO real patient data. All values are random. Any results, subtypes, drug\n"
        "effects, or figures computed from them are MEANINGLESS and exist only to demonstrate that\n"
        "the PATCH pipeline runs end-to-end. The real NIH TB Portals data is access-gated under a\n"
        "Data Use Agreement (https://tbportals.niaid.nih.gov/) and is never included in this repo.\n"
    )
    logger.info("Synthetic dataset ready in %s (FAKE DATA — results are meaningless).", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
