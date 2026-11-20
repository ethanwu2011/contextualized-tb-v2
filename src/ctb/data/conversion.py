"""Derivation of the **culture-conversion** treatment-response endpoint from TB Portals DST data.

Culture conversion (a sustained change from culture-positive to culture-negative) is the canonical
microbiological efficacy endpoint in tuberculosis: 2-month conversion is prognostic for relapse and
is a primary endpoint in TB treatment trials. This module derives, per treatment episode, two
prespecified endpoints:

* **Day-60 conversion (binary):** culture status at the result nearest day 60 within the
  ``[45, 75]`` window. Episodes with no culture in the window are flagged (excluded from the
  day-60 analysis as an explicit selection layer).
* **Time-to-conversion (survival):** days from treatment start (the first positive culture) to the
  first negative culture, with right-censoring at the last observed culture for episodes that never
  convert. Uses *all* serial cultures, addressing the post-120-day follow-up thinning.

**Date semantics (critical):** in this release the culture date columns are **integer day-offsets
from treatment start** (0 = baseline; range ~0–5000), *not* calendar dates. They must be read as
numeric. (An earlier ``to_datetime`` parse silently collapsed them.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Culture-result vocabulary classification.
_NEGATIVE = {"negative"}
_POSITIVE = {"positive", "1 to 19", "20 to 100", "100 to 200", "more than 200"}
# Everything else (nonspecific microflora, MOTT, unknown result, study in progress, not done, NaN)
# is uninterpretable for conversion and dropped.

# Day-60 window (prespecified).
DAY60_TARGET = 60
DAY60_WINDOW = (45, 75)


def classify_culture(value: object) -> str:
    """Classify a raw culture result as ``"positive"``, ``"negative"``, or ``"unknown"``.

    Parameters
    ----------
    value : object
        Raw ``cultureresults`` cell. Colony-count bins (e.g. ``"20 to 100"``) are positive growth;
        ``"Negative"`` is no growth; ``"MOTT"``/``"Nonspecific microflora"``/``"Unknown result"``/
        ``"Study in progress"``/``"Not done"``/missing are uninterpretable.

    Returns
    -------
    str
        ``"positive"``, ``"negative"``, or ``"unknown"``.
    """
    if not isinstance(value, str):
        return "unknown"
    v = value.strip().lower()
    if v in _NEGATIVE:
        return "negative"
    if v in _POSITIVE:
        return "positive"
    return "unknown"


@dataclass
class ConversionConfig:
    """Configuration for conversion derivation.

    Attributes
    ----------
    day60_target : int
        Target day for the binary endpoint.
    day60_window : tuple of int
        Inclusive ``(low, high)`` day window for selecting the day-60 culture.
    require_sustained : bool
        If True, time-to-conversion requires the first negative to be followed by no later positive
        (sustained conversion); if False, first-negative-after-first-positive is used.
    """

    day60_target: int = DAY60_TARGET
    day60_window: tuple[int, int] = DAY60_WINDOW
    require_sustained: bool = True


def _prepare(dst: pd.DataFrame) -> pd.DataFrame:
    """Filter DST rows to interpretable, dated cultures with a parsed day-offset and status.

    Parameters
    ----------
    dst : pandas.DataFrame
        Raw DST table with ``condition_id``, ``specimen_collection_date``, ``cultureresults``.

    Returns
    -------
    pandas.DataFrame
        Columns ``condition_id``, ``day`` (int), ``status`` (positive/negative); unknowns dropped.
    """
    df = dst[["condition_id", "specimen_collection_date", "cultureresults"]].copy()
    df["day"] = pd.to_numeric(df["specimen_collection_date"], errors="coerce")
    df["status"] = df["cultureresults"].map(classify_culture)
    df = df[(df["status"] != "unknown") & df["day"].notna()]
    return df.sort_values(["condition_id", "day"], kind="mergesort")


def _episode_conversion(
    days: NDArray[np.float64], status: NDArray[np.object_], cfg: ConversionConfig
) -> dict[str, object]:
    """Compute conversion endpoints for one episode's ordered cultures.

    Parameters
    ----------
    days : numpy.ndarray
        Sorted day-offsets.
    status : numpy.ndarray
        Culture status (``"positive"``/``"negative"``) aligned to ``days``.
    cfg : ConversionConfig
        Configuration.

    Returns
    -------
    dict
        Keys: ``n_cultures``, ``first_pos_day``, ``conversion_day`` (NaN if none), ``event``
        (1 converted / 0 censored / NaN if not assessable), ``time_to_event`` (days; NaN if n/a),
        ``day60_status`` (1 negative / 0 positive / NaN if no window culture), ``last_day``.
    """
    out: dict[str, object] = {
        "n_cultures": int(len(days)),
        "first_pos_day": np.nan,
        "conversion_day": np.nan,
        "event": np.nan,
        "time_to_event": np.nan,
        "day60_status": np.nan,
        "last_day": float(days.max()) if len(days) else np.nan,
    }
    pos_idx = np.flatnonzero(status == "positive")
    if len(pos_idx) == 0:
        return out  # never culture-positive: not in the conversion-estimable population
    fp = int(days[pos_idx[0]])
    out["first_pos_day"] = float(fp)

    # --- time-to-conversion (survival) ---
    neg_after = np.flatnonzero((status == "negative") & (days > fp))
    if len(neg_after) > 0:
        conv_day = int(days[neg_after[0]])
        sustained = True
        if cfg.require_sustained:
            later_pos = np.flatnonzero((status == "positive") & (days > conv_day))
            sustained = len(later_pos) == 0
        if sustained:
            out["conversion_day"] = float(conv_day)
            out["event"] = 1
            out["time_to_event"] = float(conv_day - fp)
        else:
            out["event"] = 0
            out["time_to_event"] = float(days.max() - fp)
    else:
        out["event"] = 0
        out["time_to_event"] = float(days.max() - fp)

    # --- day-60 binary (culture nearest target within window) ---
    lo, hi = cfg.day60_window
    win = np.flatnonzero((days >= lo) & (days <= hi))
    if len(win) > 0:
        nearest = win[np.argmin(np.abs(days[win] - cfg.day60_target))]
        out["day60_status"] = 1 if status[nearest] == "negative" else 0
    return out


def derive_conversion(dst: pd.DataFrame, cfg: ConversionConfig | None = None) -> pd.DataFrame:
    """Derive culture-conversion endpoints per episode from the DST table.

    Parameters
    ----------
    dst : pandas.DataFrame
        Raw DST table (``condition_id``, ``specimen_collection_date``, ``cultureresults``).
    cfg : ConversionConfig or None
        Configuration (defaults used if ``None``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id``, one row per episode that has >=1 interpretable dated culture,
        with columns ``n_cultures``, ``first_pos_day``, ``conversion_day``, ``event``,
        ``time_to_event``, ``day60_status``, ``last_day``. Only episodes with >=1 positive culture
        (the conversion-estimable population) have non-NaN ``event``/``time_to_event``.
    """
    cfg = cfg or ConversionConfig()
    prep = _prepare(dst)
    records: list[dict[str, object]] = []
    for cid, grp in prep.groupby("condition_id", sort=True):
        rec = _episode_conversion(grp["day"].to_numpy(), grp["status"].to_numpy(), cfg)
        rec["condition_id"] = cid
        records.append(rec)
    out = pd.DataFrame.from_records(records).set_index("condition_id")
    n_pos = int(out["first_pos_day"].notna().sum())
    n_conv = int((out["event"] == 1).sum())
    n_d60 = int(out["day60_status"].notna().sum())
    logger.info(
        "Conversion derived: %d episodes w/ cultures, %d baseline-positive, %d converted "
        "(event=1), %d with a day-60 window culture",
        len(out),
        n_pos,
        n_conv,
        n_d60,
    )
    return out
