"""Build a harmonized Brazil (SINAN-TB) cohort for external validation of PATCH.

Reads the public DATASUS/SINAN tuberculosis notification files (TUBEBR{YY}.dbc, open data, no DUA),
pools the requested final years, and derives the died/failure outcome and cross-registry
*harmonized* context features that also exist in TB Portals. Only aggregate summaries are logged.
The per-patient harmonized frame is written to data/processed/ (git-ignored).

SINAN is public, de-identified surveillance data; this script computes derived/aggregate features
only and never prints row-level records.

Usage
-----
    uv run --with datasus-dbc --with dbfread python scripts/31_build_brazil_cohort.py \
        --years 15 16 17 18 19 --src ~/Downloads
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ctb.config import PROCESSED_DIR, ensure_output_dirs
from ctb.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# SINAN field to harmonized meaning. Codes follow the SINAN-TB dictionary.
KEEP = [
    "SITUA_ENCE",
    "TEST_MOLEC",
    "TEST_SENSI",
    "HIV",
    "AGRAVAIDS",
    "AGRAVDIABE",
    "AGRAVALCOO",
    "AGRAVDROGA",
    "BACILOSC_E",
    "FORMA",
    "TRATAMENTO",
    "NU_IDADE_N",
    "CS_SEXO",
    "SG_UF",
    "NU_ANO",
]


def _years_age(idade: pd.Series) -> pd.Series:
    """Decode SINAN NU_IDADE_N (unit+value; 4### = years) to age in years."""
    s = idade.astype("string").str.strip()
    yrs = pd.to_numeric(s.where(s.str.startswith("4")).str[1:], errors="coerce")
    # rows coded in days/months (units 1-3) are infants -> 0 years
    return yrs.where(s.str.startswith("4"), 0.0)


def _decode(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the harmonized outcome + context features from raw SINAN columns."""
    s = {c: df[c].astype("string").str.strip() for c in KEEP if c in df.columns}
    out = pd.DataFrame(index=df.index)
    # --- outcome: died/failure (3,4,9) = 1 ; cure (1) = 0 ; LTFU/transfer/ongoing -> drop ---
    se = s["SITUA_ENCE"]
    out["y"] = np.where(se.isin(["3", "4", "9"]), 1.0, np.where(se.eq("1"), 0.0, np.nan))
    out["ltfu"] = se.isin(["2", "8"]).astype(int)  # for descriptive WHO-composite reporting
    # --- harmonized context (each also present in TB Portals) ---
    out["rif_resistant"] = (s["TEST_MOLEC"].eq("2")).astype(float)  # Xpert RIF-resistant
    out["hiv"] = np.where(s["HIV"].eq("1"), 1.0, np.where(s["HIV"].eq("2"), 0.0, np.nan))
    out["diabetes"] = (s["AGRAVDIABE"].eq("1")).astype(float)
    out["smear_pos"] = np.where(
        s["BACILOSC_E"].eq("1"), 1.0, np.where(s["BACILOSC_E"].eq("2"), 0.0, np.nan)
    )
    out["age"] = _years_age(df["NU_IDADE_N"])
    out["male"] = (s["CS_SEXO"].eq("M")).astype(float)
    out["pulmonary"] = (s["FORMA"].isin(["1", "3"])).astype(float)
    out["new_case"] = (s["TRATAMENTO"].eq("1")).astype(float)
    # Brazil-only context (descriptive; not in the shared transfer set)
    out["alcohol"] = (s["AGRAVALCOO"].eq("1")).astype(float)
    out["drugs"] = (s["AGRAVDROGA"].eq("1")).astype(float)
    out["uf"] = s["SG_UF"]
    out["year"] = s.get("NU_ANO", pd.Series(index=df.index, dtype="string"))
    return out


def main(argv: list[str] | None = None) -> int:
    """Decompress, pool, decode, and persist the harmonized Brazil cohort."""
    configure_logging()
    ensure_output_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["15", "16", "17", "18", "19"])
    ap.add_argument("--src", type=Path, default=Path("~/Downloads").expanduser())
    args = ap.parse_args(argv)

    import datasus_dbc
    import dbfread

    frames = []
    with TemporaryDirectory(prefix="patch_sinan_") as temp_dir:
        temp_root = Path(temp_dir)
        for yy in args.years:
            src = args.src.expanduser() / f"TUBEBR{yy}.dbc"
            if not src.exists():
                logger.warning("missing %s; skipping", src.name)
                continue
            tmp = temp_root / f"TUBEBR{yy}.dbf"
            datasus_dbc.decompress(str(src), str(tmp))
            df = pd.DataFrame(iter(dbfread.DBF(str(tmp), encoding="latin-1")))
            missing = sorted(set(KEEP) - set(df.columns))
            if missing:
                raise ValueError(f"{src.name} is missing required SINAN fields: {missing}")
            dec = _decode(df)
            frames.append(dec)
            usable = dec["y"].notna()
            logger.info(
                "20%s: %d notifications -> %d with usable outcome (died/fail %.1f%%)",
                yy,
                len(dec),
                int(usable.sum()),
                100 * dec.loc[usable, "y"].mean(),
            )

    if not frames:
        raise FileNotFoundError(
            f"No SINAN files found in {args.src.expanduser()}. Expected TUBEBR15.dbc through "
            "TUBEBR19.dbc or pass --years and --src."
        )

    pooled = pd.concat(frames, ignore_index=True)
    # Analysis cohort: pulmonary, known outcome (cure vs died/failure), like the PATCH cohort.
    cohort = pooled[(pooled["pulmonary"] == 1) & pooled["y"].notna()].reset_index(drop=True)
    cohort.to_parquet(PROCESSED_DIR / "brazil_harmonized.parquet")
    logger.info("=== BRAZIL COHORT ===")
    logger.info("pooled notifications: %d | pulmonary w/ outcome: %d", len(pooled), len(cohort))
    logger.info(
        "died/failure prevalence: %.1f%% | HIV+ %.1f%% (of tested) | RIF-resistant %.2f%% | "
        "diabetes %.1f%% | smear+ %.1f%% (of tested) | age median %.0f",
        100 * cohort["y"].mean(),
        100 * cohort["hiv"].mean(),
        100 * cohort["rif_resistant"].mean(),
        100 * cohort["diabetes"].mean(),
        100 * cohort.loc[cohort["smear_pos"].notna(), "smear_pos"].mean(),
        cohort["age"].median(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
