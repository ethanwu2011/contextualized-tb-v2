"""Derivation of the *treatment-intensification* outcome from the TB Portals regimen timeline.

The composite outcome flags episodes whose initial regimen proved insufficient. See
``docs/OUTCOME_DEFINITION.md`` for the full specification and the rationale behind each design
decision (notably the surrogate escalation rule, which is required because the regimen table does
not provide sequentially dated phases).

Primary composite (locked decisions):

``treatment_intensification = escalation OR extension OR re_initiation``

where

* **escalation** — a later regimen line adds ≥1 drug absent from the first (baseline) line
  (drug-set growth), OR any line carries ``activities_statusreason_cd`` in
  :data:`ctb.config.ESCALATION_STATUSREASONS`.
* **extension** — total treatment duration (``period_span``) exceeds
  :data:`ctb.config.EXTENSION_THRESHOLD` × the class-expected duration
  (:data:`ctb.config.RESISTANCE_EXPECTED_DAYS`).
* **re_initiation** — a subsequent treatment episode for the same patient begins within
  :data:`ctb.config.REINITIATION_WINDOW_DAYS` days of a prior episode's end.

``unfavorable_outcome`` (WHO died/failure/LTFU) is computed and returned for validation but is
**excluded** from the primary composite, to keep the outcome distinct from mortality.

All functions are pure with respect to their inputs (no raw files are read or written here) so that
they can be unit-tested against small synthetic frames.
"""

from __future__ import annotations

import logging

import pandas as pd

from ctb.config import (
    ESCALATION_STATUSREASONS,
    EXTENSION_THRESHOLD,
    KEY_INTENSIFICATION_DRUGS,
    REINITIATION_WINDOW_DAYS,
    RESISTANCE_EXPECTED_DAYS,
    UNFAVORABLE_OUTCOMES,
)

logger = logging.getLogger(__name__)


def parse_drug_set(value: object) -> frozenset[str]:
    """Parse a TB Portals ``regimen_drug`` set-string into a set of drug codes.

    The source encodes a regimen line as a brace-wrapped, comma-separated set, e.g.
    ``"{Bdq,Cfz,Cs,Lfx,Lzd}"``. Whitespace is stripped; an empty or missing value yields an empty
    set.

    Parameters
    ----------
    value : object
        The raw cell value (typically ``str`` or ``float`` NaN).

    Returns
    -------
    frozenset of str
        The set of drug codes; empty if the value is missing or ``"{}"``.
    """
    if not isinstance(value, str):
        return frozenset()
    inner = value.strip().strip("{}").strip()
    if not inner:
        return frozenset()
    return frozenset(tok.strip() for tok in inner.split(",") if tok.strip())


def derive_escalation(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive the escalation component per condition (surrogate rule A).

    A condition is flagged when either (a) a regimen line beyond the first adds a drug not present
    in the baseline (first) line, or (b) any line's ``activities_statusreason_cd`` indicates a
    change consistent with escalation. Because regimen lines lack distinct timestamps, the first
    line in record order is treated as the baseline.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with at least ``condition_id``, ``regimen_drug``, and
        ``activities_statusreason_cd`` columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with boolean columns ``escalation``,
        ``escalation_drugset``, ``escalation_key_drug``, and ``escalation_statusreason``.
    """
    df = regimens.copy()
    df["_drugset"] = df["regimen_drug"].map(parse_drug_set)
    sr = df.get("activities_statusreason_cd", pd.Series(index=df.index, dtype=object))

    records: list[dict[str, object]] = []
    for cid, grp in df.groupby("condition_id", sort=True):
        sets = list(grp["_drugset"])
        baseline = sets[0] if sets else frozenset()
        later_union: frozenset[str] = frozenset().union(*sets[1:]) if len(sets) > 1 else frozenset()
        added = later_union - baseline
        drugset_grow = len(added) > 0
        key_grow = len(added & KEY_INTENSIFICATION_DRUGS) > 0
        reasons = sr.loc[grp.index].dropna().astype(str)
        sr_flag = bool(reasons.isin(ESCALATION_STATUSREASONS).any())
        records.append(
            {
                "condition_id": cid,
                "escalation_drugset": drugset_grow,
                "escalation_key_drug": key_grow,
                "escalation_statusreason": sr_flag,
                "escalation": drugset_grow or sr_flag,
            }
        )
    out = pd.DataFrame.from_records(records).set_index("condition_id")
    logger.info(
        "Escalation derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["escalation"].mean() if len(out) else 0.0,
    )
    return out


def _episode_duration_days(regimens: pd.DataFrame) -> pd.Series[float]:
    """Compute treatment duration (days) per condition from ``period_span``.

    Uses the maximum ``period_span`` across an episode's regimen lines as the episode duration,
    since lines share the episode start date and ``period_span`` carries per-line length.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table with ``condition_id`` and ``period_span``.

    Returns
    -------
    pandas.Series
        Duration in days indexed by ``condition_id`` (NaN where ``period_span`` is missing).
    """
    return regimens.groupby("condition_id", sort=True)["period_span"].max()


def derive_extension(regimens: pd.DataFrame, patient_cases: pd.DataFrame) -> pd.DataFrame:
    """Derive the treatment-extension component per condition.

    A condition is flagged when its duration exceeds ``EXTENSION_THRESHOLD`` times the expected
    duration for its resistance class. Conditions whose resistance class has no defined expected
    duration are not flagged (``extension = False``, ``expected_days = NaN``).

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``period_span``).
    patient_cases : pandas.DataFrame
        Patient Cases table (``condition_id``, ``type_of_resistance``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with columns ``duration_days`` (float),
        ``expected_days`` (float), and ``extension`` (bool).
    """
    duration = _episode_duration_days(regimens).rename("duration_days")
    res = (
        patient_cases[["condition_id", "type_of_resistance"]]
        .drop_duplicates("condition_id")
        .set_index("condition_id")["type_of_resistance"]
    )
    expected = res.map(RESISTANCE_EXPECTED_DAYS).rename("expected_days")

    out = pd.concat([duration, expected], axis=1)
    out["expected_days"] = out["expected_days"].astype("float64")
    out["duration_days"] = out["duration_days"].astype("float64")
    threshold = EXTENSION_THRESHOLD * out["expected_days"]
    out["extension"] = (out["duration_days"] > threshold).fillna(False)
    logger.info(
        "Extension derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["extension"].mean() if len(out) else 0.0,
    )
    return out


def derive_reinitiation(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    window_days: int = REINITIATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Derive the treatment re-initiation component per condition.

    For patients with multiple episodes, episodes are ordered by ``registration_date``. A
    (non-first) episode is flagged when it begins within ``window_days`` days *after* the prior
    episode's end (the prior episode's latest ``period_end``). Concurrent/overlapping episodes
    (negative gap) and gaps beyond the window are not flagged.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Patient Cases table (``patient_id``, ``condition_id``, ``registration_date``).
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``period_end``).
    window_days : int, default :data:`ctb.config.REINITIATION_WINDOW_DAYS`
        Maximum gap (days) between prior end and new start to count as re-initiation.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a single boolean column ``re_initiation``.
    """
    end_by_cond = regimens.groupby("condition_id", sort=True)["period_end"].max()
    pc = (
        patient_cases[["patient_id", "condition_id", "registration_date"]]
        .drop_duplicates("condition_id")
        .copy()
    )
    pc["registration_date"] = pd.to_datetime(pc["registration_date"], errors="coerce")
    pc["prior_end"] = pc["condition_id"].map(end_by_cond)

    flagged: set[object] = set()
    for _, grp in pc.groupby("patient_id", sort=True):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("registration_date", kind="mergesort")
        prior_end = grp["prior_end"].to_numpy()
        starts = grp["registration_date"].to_numpy()
        cids = grp["condition_id"].to_numpy()
        for i in range(1, len(grp)):
            if pd.isna(prior_end[i - 1]) or pd.isna(starts[i]):
                continue
            gap = (pd.Timestamp(starts[i]) - pd.Timestamp(prior_end[i - 1])).days
            if 0 <= gap <= window_days:
                flagged.add(cids[i])

    idx = pc.set_index("condition_id").index
    out = pd.DataFrame({"re_initiation": idx.isin(flagged)}, index=idx)
    logger.info(
        "Re-initiation derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["re_initiation"].mean() if len(out) else 0.0,
    )
    return out


def derive_unfavorable(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive the WHO unfavorable-outcome flag per condition (validation column).

    A condition is flagged if any regimen line records an ``outcome_cd`` in
    :data:`ctb.config.UNFAVORABLE_OUTCOMES` (died/failure/LTFU).

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``outcome_cd``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a single boolean column ``unfavorable_outcome``.
    """
    grp = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    flag = grp.agg(lambda s: bool(s.dropna().astype(str).isin(UNFAVORABLE_OUTCOMES).any()))
    out = flag.rename("unfavorable_outcome").to_frame()
    logger.info(
        "Unfavorable outcome derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["unfavorable_outcome"].mean() if len(out) else 0.0,
    )
    return out


def derive_mortality(regimens: pd.DataFrame) -> pd.DataFrame:
    """Derive an all-cause mortality flag per condition.

    A condition is flagged if any regimen line records ``outcome_cd == "died"``. This reproduces the
    outcome used in the ML4H 2024 contextualized-TB analysis (X = treatments, Y = mortality), giving
    a clean treatment-effect target that is decoupled from the regimen-derived intensification
    outcome.

    Parameters
    ----------
    regimens : pandas.DataFrame
        Regimens table (``condition_id``, ``outcome_cd``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with a single boolean column ``mortality``.
    """
    grp = regimens.groupby("condition_id", sort=True)["outcome_cd"]
    flag = grp.agg(lambda s: bool(s.dropna().astype(str).eq("died").any()))
    out = flag.rename("mortality").to_frame()
    logger.info(
        "Mortality derived for %d conditions (%.1f%% positive)",
        len(out),
        100 * out["mortality"].mean() if len(out) else 0.0,
    )
    return out


def derive_treatment_intensification(
    patient_cases: pd.DataFrame,
    regimens: pd.DataFrame,
    window_days: int = REINITIATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Assemble the full per-condition treatment-intensification outcome frame.

    Combines the escalation, extension, re-initiation, and unfavorable components. The primary
    composite ``treatment_intensification`` is the logical OR of escalation, extension, and
    re-initiation (unfavorable is excluded). A secondary
    ``treatment_intensification_with_unfavorable`` column additionally ORs in the unfavorable flag.

    Parameters
    ----------
    patient_cases : pandas.DataFrame
        Patient Cases table.
    regimens : pandas.DataFrame
        Regimens table.
    window_days : int, default :data:`ctb.config.REINITIATION_WINDOW_DAYS`
        Re-initiation window passed to :func:`derive_reinitiation`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``condition_id`` with all component and composite columns. Every condition in
        ``patient_cases`` is represented (components default to ``False`` where not derivable).
    """
    base_idx = pd.Index(patient_cases["condition_id"].drop_duplicates(), name="condition_id")

    esc = derive_escalation(regimens)
    ext = derive_extension(regimens, patient_cases)
    rei = derive_reinitiation(patient_cases, regimens, window_days=window_days)
    unf = derive_unfavorable(regimens)

    out = pd.DataFrame(index=base_idx)
    out = out.join(esc).join(ext).join(rei).join(unf)

    bool_cols = [
        "escalation_drugset",
        "escalation_key_drug",
        "escalation_statusreason",
        "escalation",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
    ]
    for col in bool_cols:
        out[col] = out[col].where(out[col].notna(), other=False).astype(bool)

    out["treatment_intensification"] = out["escalation"] | out["extension"] | out["re_initiation"]
    out["treatment_intensification_with_unfavorable"] = (
        out["treatment_intensification"] | out["unfavorable_outcome"]
    )
    logger.info(
        "Composite treatment_intensification: %.1f%% positive (primary), %.1f%% (with unfavorable)",
        100 * out["treatment_intensification"].mean() if len(out) else 0.0,
        100 * out["treatment_intensification_with_unfavorable"].mean() if len(out) else 0.0,
    )
    return out


def outcome_prevalence_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarise prevalence of each component and composite.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        Output of :func:`derive_treatment_intensification`.

    Returns
    -------
    pandas.DataFrame
        Columns ``component``, ``n_positive``, ``n_total``, ``prevalence`` (one row per flag).
    """
    cols = [
        "escalation",
        "escalation_drugset",
        "escalation_key_drug",
        "escalation_statusreason",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
        "treatment_intensification",
        "treatment_intensification_with_unfavorable",
    ]
    n = len(outcomes)
    rows = [
        {
            "component": c,
            "n_positive": int(outcomes[c].sum()),
            "n_total": n,
            "prevalence": float(outcomes[c].mean()) if n else float("nan"),
        }
        for c in cols
        if c in outcomes.columns
    ]
    return pd.DataFrame(rows)
