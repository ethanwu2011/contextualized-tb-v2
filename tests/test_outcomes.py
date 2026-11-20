"""Unit tests for :mod:`ctb.data.outcomes`.

All fixtures are synthetic and schema-matching; no real TB Portals data is used. These tests are
the most safety-critical in the project because outcome mis-derivation would silently corrupt every
downstream result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctb.data import outcomes as oc


# --------------------------------------------------------------------------------------------- #
# parse_drug_set
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("{E,H,R,Z}", frozenset({"E", "H", "R", "Z"})),
        ("{ Bdq , Lzd }", frozenset({"Bdq", "Lzd"})),
        ("{R}", frozenset({"R"})),
        ("{}", frozenset()),
        ("", frozenset()),
        ("   ", frozenset()),
        (np.nan, frozenset()),
        (None, frozenset()),
        (42, frozenset()),
    ],
)
def test_parse_drug_set(value: object, expected: frozenset[str]) -> None:
    assert oc.parse_drug_set(value) == expected


# --------------------------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------------------------- #
def _regimens(rows: list[dict[str, object]]) -> pd.DataFrame:
    cols = [
        "patient_id",
        "condition_id",
        "period_start",
        "period_end",
        "period_span",
        "outcome_cd",
        "activities_statusreason_cd",
        "regimen_drug",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in ("period_start", "period_end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df[cols]


def _cases(rows: list[dict[str, object]]) -> pd.DataFrame:
    cols = ["patient_id", "condition_id", "registration_date", "type_of_resistance"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    return df[cols]


# --------------------------------------------------------------------------------------------- #
# derive_escalation
# --------------------------------------------------------------------------------------------- #
def test_escalation_drugset_growth() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{H,R}"},
            {"condition_id": 1, "regimen_drug": "{H,R,Bdq}"},  # adds Bdq (key)
        ]
    )
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is True
    assert bool(out.loc[1, "escalation_drugset"]) is True
    assert bool(out.loc[1, "escalation_key_drug"]) is True


def test_escalation_deescalation_not_flagged() -> None:
    # Intensive -> continuation phase (drugs DROPPED) is NOT escalation.
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{E,H,R,Z}"},
            {"condition_id": 1, "regimen_drug": "{H,R}"},
        ]
    )
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is False


def test_escalation_statusreason_flag() -> None:
    reg = _regimens(
        [
            {
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "activities_statusreason_cd": "resistance",
            },
        ]
    )
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is True
    assert bool(out.loc[1, "escalation_statusreason"]) is True
    assert bool(out.loc[1, "escalation_drugset"]) is False


def test_escalation_nonkey_drug_growth() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{H,R}"},
            {"condition_id": 1, "regimen_drug": "{H,R,E}"},  # adds E (not a key drug)
        ]
    )
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation_drugset"]) is True
    assert bool(out.loc[1, "escalation_key_drug"]) is False
    assert bool(out.loc[1, "escalation"]) is True


def test_escalation_single_line_not_flagged() -> None:
    reg = _regimens([{"condition_id": 1, "regimen_drug": "{E,H,R,Z}"}])
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is False


# --------------------------------------------------------------------------------------------- #
# derive_extension
# --------------------------------------------------------------------------------------------- #
def test_extension_over_threshold() -> None:
    # Sensitive expected 180d; 1.2x = 216d. 250 > 216 => extension.
    reg = _regimens([{"condition_id": 1, "period_span": 250}])
    pc = _cases([{"patient_id": "p", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_extension(reg, pc)
    assert bool(out.loc[1, "extension"]) is True
    assert out.loc[1, "expected_days"] == 180.0


def test_extension_under_threshold() -> None:
    reg = _regimens([{"condition_id": 1, "period_span": 200}])  # < 216
    pc = _cases([{"patient_id": "p", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_extension(reg, pc)
    assert bool(out.loc[1, "extension"]) is False


def test_extension_class_specific_mapping() -> None:
    # Poly DR -> 540 (locked clinical-standard mapping), 1.2x = 648.
    reg = _regimens(
        [
            {"condition_id": 1, "period_span": 700},  # > 648 => True
            {"condition_id": 2, "period_span": 600},  # < 648 => False
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p1", "condition_id": 1, "type_of_resistance": "Poly DR"},
            {"patient_id": "p2", "condition_id": 2, "type_of_resistance": "Poly DR"},
        ]
    )
    out = oc.derive_extension(reg, pc)
    assert bool(out.loc[1, "extension"]) is True
    assert bool(out.loc[2, "extension"]) is False
    assert out.loc[1, "expected_days"] == 540.0


def test_extension_unknown_class_not_flagged() -> None:
    reg = _regimens([{"condition_id": 1, "period_span": 9999}])
    pc = _cases([{"patient_id": "p", "condition_id": 1, "type_of_resistance": "Negative"}])
    out = oc.derive_extension(reg, pc)
    assert bool(out.loc[1, "extension"]) is False
    assert pd.isna(out.loc[1, "expected_days"])


def test_extension_missing_duration_not_flagged() -> None:
    reg = _regimens([{"condition_id": 1, "period_span": np.nan}])
    pc = _cases([{"patient_id": "p", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_extension(reg, pc)
    assert bool(out.loc[1, "extension"]) is False


def test_extension_uses_max_span_across_lines() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "period_span": 60},
            {"condition_id": 1, "period_span": 250},  # max -> over threshold
        ]
    )
    pc = _cases([{"patient_id": "p", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_extension(reg, pc)
    assert out.loc[1, "duration_days"] == 250.0
    assert bool(out.loc[1, "extension"]) is True


# --------------------------------------------------------------------------------------------- #
# derive_reinitiation
# --------------------------------------------------------------------------------------------- #
def test_reinitiation_within_window() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "period_end": "2020-06-01"},
            {"condition_id": 2, "period_end": "2021-06-01"},
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p", "condition_id": 1, "registration_date": "2020-01-01"},
            {"patient_id": "p", "condition_id": 2, "registration_date": "2020-09-01"},  # 92d gap
        ]
    )
    out = oc.derive_reinitiation(pc, reg)
    assert bool(out.loc[2, "re_initiation"]) is True
    assert bool(out.loc[1, "re_initiation"]) is False  # first episode never flagged


def test_reinitiation_outside_window() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "period_end": "2018-06-01"},
            {"condition_id": 2, "period_end": "2021-06-01"},
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p", "condition_id": 1, "registration_date": "2018-01-01"},
            {"patient_id": "p", "condition_id": 2, "registration_date": "2020-06-01"},  # >365d
        ]
    )
    out = oc.derive_reinitiation(pc, reg)
    assert bool(out.loc[2, "re_initiation"]) is False


def test_reinitiation_single_episode() -> None:
    reg = _regimens([{"condition_id": 1, "period_end": "2020-06-01"}])
    pc = _cases([{"patient_id": "p", "condition_id": 1, "registration_date": "2020-01-01"}])
    out = oc.derive_reinitiation(pc, reg)
    assert bool(out.loc[1, "re_initiation"]) is False


def test_reinitiation_overlap_not_flagged() -> None:
    # New episode starts BEFORE prior end (negative gap) => concurrent, not re-initiation.
    reg = _regimens(
        [
            {"condition_id": 1, "period_end": "2021-01-01"},
            {"condition_id": 2, "period_end": "2021-12-01"},
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p", "condition_id": 1, "registration_date": "2020-01-01"},
            {"patient_id": "p", "condition_id": 2, "registration_date": "2020-06-01"},
        ]
    )
    out = oc.derive_reinitiation(pc, reg)
    assert bool(out.loc[2, "re_initiation"]) is False


def test_reinitiation_missing_dates_skipped() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "period_end": np.nan},
            {"condition_id": 2, "period_end": "2021-06-01"},
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p", "condition_id": 1, "registration_date": "2020-01-01"},
            {"patient_id": "p", "condition_id": 2, "registration_date": "2020-09-01"},
        ]
    )
    out = oc.derive_reinitiation(pc, reg)
    assert bool(out.loc[2, "re_initiation"]) is False  # prior end missing -> cannot assess


# --------------------------------------------------------------------------------------------- #
# derive_unfavorable
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("code", "expected"),
    [("died", True), ("failure", True), ("ltfu", True), ("cured", False), ("completed", False)],
)
def test_unfavorable(code: str, expected: bool) -> None:
    reg = _regimens([{"condition_id": 1, "outcome_cd": code}])
    out = oc.derive_unfavorable(reg)
    assert bool(out.loc[1, "unfavorable_outcome"]) is expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [("died", True), ("failure", False), ("ltfu", False), ("cured", False)],
)
def test_mortality(code: str, expected: bool) -> None:
    reg = _regimens([{"condition_id": 1, "outcome_cd": code}])
    out = oc.derive_mortality(reg)
    assert bool(out.loc[1, "mortality"]) is expected


def test_mortality_any_line_triggers() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "outcome_cd": "cured"},
            {"condition_id": 1, "outcome_cd": "died"},
        ]
    )
    out = oc.derive_mortality(reg)
    assert bool(out.loc[1, "mortality"]) is True


def test_unfavorable_any_line_triggers() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "outcome_cd": "cured"},
            {"condition_id": 1, "outcome_cd": "died"},
        ]
    )
    out = oc.derive_unfavorable(reg)
    assert bool(out.loc[1, "unfavorable_outcome"]) is True


# --------------------------------------------------------------------------------------------- #
# derive_treatment_intensification (composite)
# --------------------------------------------------------------------------------------------- #
def test_composite_or_logic_and_unfavorable_excluded() -> None:
    reg = _regimens(
        [
            # cond 1: pure escalation
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "period_span": 100,
                "outcome_cd": "cured",
            },
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R,Bdq}",
                "period_span": 100,
                "outcome_cd": "cured",
            },
            # cond 2: pure extension (Sensitive, 250 > 216)
            {
                "patient_id": "b",
                "condition_id": 2,
                "regimen_drug": "{E,H,R,Z}",
                "period_span": 250,
                "outcome_cd": "completed",
            },
            # cond 3: ONLY unfavorable (short, no escalation) -> primary False, secondary True
            {
                "patient_id": "c",
                "condition_id": 3,
                "regimen_drug": "{E,H,R,Z}",
                "period_span": 100,
                "outcome_cd": "died",
            },
        ]
    )
    pc = _cases(
        [
            {"patient_id": "a", "condition_id": 1, "type_of_resistance": "Sensitive"},
            {"patient_id": "b", "condition_id": 2, "type_of_resistance": "Sensitive"},
            {"patient_id": "c", "condition_id": 3, "type_of_resistance": "Sensitive"},
        ]
    )
    out = oc.derive_treatment_intensification(pc, reg)
    assert bool(out.loc[1, "treatment_intensification"]) is True
    assert bool(out.loc[2, "treatment_intensification"]) is True
    # cond 3 is unfavorable-only: excluded from primary, included in secondary
    assert bool(out.loc[3, "treatment_intensification"]) is False
    assert bool(out.loc[3, "unfavorable_outcome"]) is True
    assert bool(out.loc[3, "treatment_intensification_with_unfavorable"]) is True


def test_composite_covers_all_conditions_even_without_regimen() -> None:
    # condition 99 exists in patient_cases but has NO regimen row -> all flags default False.
    reg = _regimens(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "period_span": 100,
                "outcome_cd": "cured",
            }
        ]
    )
    pc = _cases(
        [
            {"patient_id": "a", "condition_id": 1, "type_of_resistance": "Sensitive"},
            {"patient_id": "z", "condition_id": 99, "type_of_resistance": "Sensitive"},
        ]
    )
    out = oc.derive_treatment_intensification(pc, reg)
    assert 99 in out.index
    assert bool(out.loc[99, "treatment_intensification"]) is False
    assert bool(out.loc[99, "unfavorable_outcome"]) is False


def test_composite_output_dtypes_are_bool() -> None:
    reg = _regimens(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "period_span": 100,
                "outcome_cd": "cured",
            }
        ]
    )
    pc = _cases([{"patient_id": "a", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_treatment_intensification(pc, reg)
    for col in [
        "escalation",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
        "treatment_intensification",
        "treatment_intensification_with_unfavorable",
    ]:
        assert out[col].dtype == bool


# --------------------------------------------------------------------------------------------- #
# outcome_prevalence_table
# --------------------------------------------------------------------------------------------- #
def test_prevalence_table() -> None:
    reg = _regimens(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R,Bdq}",
                "period_span": 100,
                "outcome_cd": "cured",
            },
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "period_span": 100,
                "outcome_cd": "cured",
            },
        ]
    )
    pc = _cases([{"patient_id": "a", "condition_id": 1, "type_of_resistance": "Sensitive"}])
    out = oc.derive_treatment_intensification(pc, reg)
    tab = oc.outcome_prevalence_table(out)
    assert {"component", "n_positive", "n_total", "prevalence"} <= set(tab.columns)
    assert (tab["n_total"] == 1).all()
    assert tab["prevalence"].between(0, 1).all()
