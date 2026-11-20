"""Raw data loaders for the TB Portals (August 2023) clinical tables.

Each loader reads exactly one source CSV from :data:`ctb.config.RAW_DIR` and returns a typed
:class:`pandas.DataFrame`. Loaders never mutate the raw files. Date columns are parsed where the
schema is well understood; all other cleaning is deferred to :mod:`ctb.data.clean`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ctb.config import RAW_DIR, TB_PORTALS_FILES

logger = logging.getLogger(__name__)

# Date columns known to exist in each table (parsed on load where unambiguous).
_DATE_COLS: dict[str, list[str]] = {
    "regimens": ["period_start", "period_end", "activities_period_start", "activities_period_end"],
    "patient_cases": ["registration_date"],
    "specimen": ["specimen_collection_date"],
    "biochemistry": ["test_date", "specimen_collection_date"],
    "dst": ["test_date", "specimen_collection_date"],
}

# Culture and specimen dates in the August 2023 DST/specimen exports are integer day offsets from
# treatment start. They must not be passed through ``to_datetime`` because pandas would interpret
# small integers as nanoseconds after the Unix epoch.
_DAY_OFFSET_COLS: dict[str, list[str]] = {
    "dst": ["test_date", "specimen_collection_date"],
    "specimen": ["test_date", "specimen_collection_date"],
}


def _parse_registration_date(values: pd.Series[Any]) -> pd.Series[Any]:
    """Parse the TB Portals registration field without losing year-only exports.

    Some TB Portals releases encode ``registration_date`` as an integer calendar year. Passing
    those integers directly to :func:`pandas.to_datetime` interprets them as nanoseconds after
    the Unix epoch. That silently turns 2021 into a timestamp in 1970 and makes downstream time
    splits depend on ``Timestamp.nanosecond``. Preserve integer years explicitly and use normal
    date parsing for ISO dates.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_nonnull = numeric.dropna()
    is_year_only = (
        len(numeric_nonnull) > 0
        and numeric_nonnull.between(1900, 2100).all()
        and (numeric_nonnull % 1 == 0).all()
    )
    if is_year_only:
        years = numeric.round().astype("Int64").astype("string")
        return pd.to_datetime(years, format="%Y", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def _resolve(table: str, raw_dir: Path | None) -> Path:
    """Resolve the on-disk path for a named TB Portals table.

    Parameters
    ----------
    table : str
        Logical table name; a key of :data:`ctb.config.TB_PORTALS_FILES`.
    raw_dir : pathlib.Path or None
        Directory containing the CSVs. Defaults to :data:`ctb.config.RAW_DIR`.

    Returns
    -------
    pathlib.Path
        Full path to the CSV file.

    Raises
    ------
    KeyError
        If ``table`` is not a recognised TB Portals table.
    FileNotFoundError
        If the resolved file does not exist.
    """
    if table not in TB_PORTALS_FILES:
        raise KeyError(
            f"Unknown TB Portals table {table!r}; expected one of {list(TB_PORTALS_FILES)}"
        )
    base = RAW_DIR if raw_dir is None else raw_dir
    path = base / TB_PORTALS_FILES[table]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file for {table!r} not found at {path}. Set CTB_RAW_DIR or place the TB Portals "
            f"export under {base}."
        )
    return path


def load_table(table: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load a single TB Portals table as a DataFrame.

    Parameters
    ----------
    table : str
        Logical table name; a key of :data:`ctb.config.TB_PORTALS_FILES`.
    raw_dir : pathlib.Path or None, optional
        Directory containing the CSVs. Defaults to :data:`ctb.config.RAW_DIR`.

    Returns
    -------
    pandas.DataFrame
        The loaded table, with calendar-date columns parsed to ``datetime64`` and culture-day
        offsets retained as numeric values.
    """
    path = _resolve(table, raw_dir)
    date_cols = _DATE_COLS.get(table, [])
    df = pd.read_csv(path, low_memory=False)
    for col in _DAY_OFFSET_COLS.get(table, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in date_cols:
        if col in df.columns:
            if col in _DAY_OFFSET_COLS.get(table, []):
                continue
            if table == "patient_cases" and col == "registration_date":
                df[col] = _parse_registration_date(df[col])
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info("Loaded %s: %d rows x %d cols from %s", table, df.shape[0], df.shape[1], path.name)
    return df


def load_patient_cases(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the Patient Cases table (one row per condition/episode).

    Parameters
    ----------
    raw_dir : pathlib.Path or None, optional
        Directory containing the CSVs.

    Returns
    -------
    pandas.DataFrame
        Patient Cases table.
    """
    return load_table("patient_cases", raw_dir)


def load_regimens(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the Regimens table (one row per regimen line within an episode).

    Parameters
    ----------
    raw_dir : pathlib.Path or None, optional
        Directory containing the CSVs.

    Returns
    -------
    pandas.DataFrame
        Regimens table.
    """
    return load_table("regimens", raw_dir)


def load_biochemistry(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the Biochemistry (longitudinal labs) table.

    Parameters
    ----------
    raw_dir : pathlib.Path or None, optional
        Directory containing the CSVs.

    Returns
    -------
    pandas.DataFrame
        Biochemistry table.
    """
    return load_table("biochemistry", raw_dir)


def load_dst(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the Drug-Susceptibility-Testing (DST) table.

    Parameters
    ----------
    raw_dir : pathlib.Path or None, optional
        Directory containing the CSVs.

    Returns
    -------
    pandas.DataFrame
        DST table.
    """
    return load_table("dst", raw_dir)
