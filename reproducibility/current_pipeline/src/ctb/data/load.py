"""Raw data loaders for the TB Portals (August 2023) clinical tables.

Each loader reads exactly one source CSV from :data:`ctb.config.RAW_DIR` and returns a typed
:class:`pandas.DataFrame`. Loaders never mutate the raw files. Date columns are parsed where the
schema is well understood; all other cleaning is deferred to :mod:`ctb.data.clean`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ctb.config import RAW_DIR, TB_PORTALS_FILES

logger = logging.getLogger(__name__)

# Date columns known to be calendar dates in each table. ``registration_date`` is intentionally
# excluded: the audited Patient Cases release stores four-digit calendar years, while synthetic and
# other releases may store ISO dates. Coercing integer years with ``pd.to_datetime`` turns them into
# nanoseconds after 1970 and destroys temporal-split semantics. Consumers must use
# ``ctb.data.temporal.extract_calendar_year`` or another field-specific parser.
_DATE_COLS: dict[str, list[str]] = {
    # Regimen starts/ends are signed integer day offsets in the audited release, although
    # synthetic/other releases may use ISO dates. Keep them raw for field-specific parsing.
    "regimens": [],
    "patient_cases": [],
    # Specimen, DST, and biochemistry releases may use integer day offsets rather than calendar
    # dates. Keep them raw so conversion and baseline-window code can apply declared units.
    "specimen": [],
    "biochemistry": [],
    "dst": [],
}


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
        The loaded table, with unambiguous calendar-date columns parsed to ``datetime64``.
        Schema-dependent year/offset fields are retained in their source representation.
    """
    path = _resolve(table, raw_dir)
    date_cols = _DATE_COLS.get(table, [])
    df = pd.read_csv(path, low_memory=False)
    for col in date_cols:
        if col in df.columns:
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
