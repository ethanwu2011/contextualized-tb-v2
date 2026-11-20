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
# require_modelable_binary_outcome
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["bool", "boolean", "int64", "Int64", "float64"])
def test_modelable_binary_outcome_accepts_complete_native_labels(dtype: str) -> None:
    values = pd.Series([0, 1, 0, 1], dtype=dtype, index=[4, 3, 2, 1], name="endpoint")
    original = values.copy()

    result = oc.require_modelable_binary_outcome(values, outcome_name="endpoint")

    assert result.tolist() == [0, 1, 0, 1]
    assert result.index.tolist() == [4, 3, 2, 1]
    assert result.name == "endpoint"
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (pd.Series([0, 1, pd.NA], dtype="Int64"), "missing=1"),
        (pd.Series([0, 1, None], dtype="Int64"), "missing=1"),
        (pd.Series([0.0, 1.0, np.nan]), "missing=1"),
        (pd.Series([0, 2]), "invalid=1"),
        (pd.Series([0.0, 0.5, 1.0]), "invalid=1"),
        (pd.Series([0.0, np.inf, 1.0]), "invalid=1"),
        (pd.Series([0.0, -np.inf, 1.0]), "invalid=1"),
        (pd.Series([-1, 0, 1]), "invalid=1"),
        (pd.Series([1, 1]), r"classes=\[1\]"),
        (pd.Series([0, 0]), r"classes=\[0\]"),
        (pd.Series([], dtype="Int64"), r"classes=\[\]"),
        (pd.Series([pd.NA, pd.NA], dtype="Int64"), "missing=2"),
    ],
)
def test_modelable_binary_outcome_rejects_unsafe_labels(
    values: pd.Series[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        oc.require_modelable_binary_outcome(values, outcome_name="archived_endpoint")


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["0", "1"]),
        pd.Series(["0.0", "1.0"]),
        pd.Series([b"0", b"1"]),
        pd.Series(np.asarray([0, 1], dtype="timedelta64[ns]")),
        pd.Series(np.asarray([0 + 0j, 1 + 0j], dtype=np.complex128)),
        pd.Series([False, True, 0, 1]),
    ],
)
def test_modelable_binary_outcome_never_coerces_unsupported_dtypes(
    values: pd.Series[object],
) -> None:
    with pytest.raises(ValueError, match="not coerced"):
        oc.require_modelable_binary_outcome(values, outcome_name="archived_endpoint")


def test_modelable_binary_outcome_reports_only_aggregate_counts() -> None:
    values = pd.Series(
        [0, 1, pd.NA, 2],
        dtype="Int64",
        index=["public-0", "SENSITIVE-PATIENT-ID", "public-2", "public-3"],
    )

    with pytest.raises(ValueError) as error:
        oc.require_modelable_binary_outcome(values, outcome_name="endpoint")

    message = str(error.value)
    assert "total=4" in message
    assert "assessable=2" in message
    assert "missing=1" in message
    assert "invalid=1" in message
    assert "SENSITIVE-PATIENT-ID" not in message


def test_modelable_binary_outcome_enforces_per_class_fold_count() -> None:
    values = pd.Series([0] * 5 + [1] * 4)

    with pytest.raises(ValueError, match="required_per_class=5"):
        oc.require_modelable_binary_outcome(
            values,
            outcome_name="endpoint",
            min_class_count=5,
        )


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_modelable_binary_outcome_rejects_invalid_minimum(minimum: object) -> None:
    with pytest.raises(ValueError, match="min_class_count"):
        oc.require_modelable_binary_outcome(
            pd.Series([0, 1]),
            outcome_name="endpoint",
            min_class_count=minimum,  # type: ignore[arg-type]
        )


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
def test_escalation_drugset_growth_is_unassessable_without_chronology() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{H,R}"},
            {"condition_id": 1, "regimen_drug": "{H,R,Bdq}"},  # adds Bdq (key)
        ]
    )
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is False
    assert pd.isna(out.loc[1, "escalation_drugset"])
    assert pd.isna(out.loc[1, "escalation_key_drug"])
    assert out.attrs["escalation_report"]["drugset_growth_assessable"] is False


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
    assert pd.isna(out.loc[1, "escalation_drugset"])


def test_escalation_nonkey_drug_growth_is_not_inferred_from_row_order() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{H,R}"},
            {"condition_id": 1, "regimen_drug": "{H,R,E}"},  # adds E (not a key drug)
        ]
    )
    out = oc.derive_escalation(reg)
    assert pd.isna(out.loc[1, "escalation_drugset"])
    assert pd.isna(out.loc[1, "escalation_key_drug"])
    assert bool(out.loc[1, "escalation"]) is False


def test_escalation_single_line_not_flagged() -> None:
    reg = _regimens([{"condition_id": 1, "regimen_drug": "{E,H,R,Z}"}])
    out = oc.derive_escalation(reg)
    assert bool(out.loc[1, "escalation"]) is False


def test_escalation_is_invariant_to_source_row_order() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "regimen_drug": "{H,R}"},
            {
                "condition_id": 1,
                "regimen_drug": "{H,R,Bdq}",
                "activities_statusreason_cd": "resistance",
            },
            {"condition_id": 2, "regimen_drug": "{E,H,R,Z}"},
            {"condition_id": 2, "regimen_drug": "{H,R}"},
        ]
    )

    expected = oc.derive_escalation(reg)
    reversed_result = oc.derive_escalation(reg.iloc[::-1].reset_index(drop=True))

    pd.testing.assert_frame_equal(reversed_result, expected)
    assert reversed_result.attrs == expected.attrs


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


def test_reinitiation_missing_prior_end_is_unassessable() -> None:
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
    assert pd.isna(out.loc[2, "re_initiation"])
    assert out.attrs["reinitiation_report"]["unassessable_conditions"] == 1


def test_reinitiation_ambiguous_episode_order_is_unassessable() -> None:
    reg = _regimens(
        [
            {"condition_id": 1, "period_end": "2020-06-01"},
            {"condition_id": 2, "period_end": "2021-06-01"},
        ]
    )
    pc = _cases(
        [
            {"patient_id": "p", "condition_id": 1, "registration_date": "2020-01-01"},
            {"patient_id": "p", "condition_id": 2, "registration_date": "2020-01-01"},
        ]
    )

    out = oc.derive_reinitiation(pc, reg)

    assert out["re_initiation"].isna().all()
    assert out.attrs["reinitiation_report"]["unassessable_conditions"] == 2


def test_reinitiation_disables_incompatible_calendar_year_and_day_offset_coordinates() -> None:
    pc = pd.DataFrame(
        {
            "patient_id": ["p", "p"],
            "condition_id": [1, 2],
            "registration_date": [2020, 2021],
        }
    )
    reg = pd.DataFrame(
        {
            "condition_id": [1, 2],
            "period_end": [180, 200],
        }
    )

    out = oc.derive_reinitiation(pc, reg)

    assert out["re_initiation"].isna().all()
    assert out.attrs["reinitiation_report"]["coordinate_compatible"] is False
    assert out.attrs["reinitiation_report"]["registration_coordinate"] == "calendar_year"
    assert out.attrs["reinitiation_report"]["period_end_coordinate"] == "day_offset"


@pytest.mark.parametrize("window_days", [1.5, float("inf"), float("nan"), True, "365"])
def test_reinitiation_rejects_noninteger_window(window_days: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        oc.derive_reinitiation(
            pd.DataFrame(
                {
                    "patient_id": ["p"],
                    "condition_id": [1],
                    "registration_date": ["2020-01-01"],
                }
            ),
            pd.DataFrame({"condition_id": [1], "period_end": ["2020-06-01"]}),
            window_days=window_days,  # type: ignore[arg-type]
        )


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


def test_died_failure_preserves_unassessable_and_mixed_states() -> None:
    reg = pd.DataFrame(
        [
            {"condition_id": 1, "outcome_cd": "ltfu"},
            {"condition_id": 2, "outcome_cd": "completed"},
            {"condition_id": 2, "outcome_cd": "failure"},
            {"condition_id": 3, "outcome_cd": "died"},
            {"condition_id": 4, "outcome_cd": "cured"},
            {"condition_id": 5, "outcome_cd": "unknown"},
            {"condition_id": 6, "outcome_cd": "stillOnTreatment"},
            {"condition_id": 7, "outcome_cd": "palliativeCare"},
            {"condition_id": 8, "outcome_cd": None},
            {"condition_id": 9, "outcome_cd": ""},
            {"condition_id": 10, "outcome_cd": "cured"},
            {"condition_id": 10, "outcome_cd": "unknown"},
            {"condition_id": 11, "outcome_cd": "unknown"},
            {"condition_id": 11, "outcome_cd": "failure"},
        ]
    )
    out = oc.derive_died_failure(reg)
    assert pd.isna(out.loc[1, "died_failure"])
    assert bool(out.loc[3, "died_failure"])
    assert not bool(out.loc[4, "died_failure"])
    for condition_id in (2, 5, 6, 7, 8, 9, 10, 11):
        assert pd.isna(out.loc[condition_id, "died_failure"])
    assert out["died_failure"].dtype == "boolean"
    assert out.attrs["died_failure_report"] == {
        "n_conditions": 11,
        "n_assessable": 2,
        "n_events": 1,
        "n_observed_nonevents": 1,
        "n_unassessable": 9,
        "event_codes": ["died", "failure"],
        "observed_nonevent_codes": ["completed", "cured"],
    }


def test_patient_case_died_failure_uses_recorded_final_outcome() -> None:
    patient_cases = pd.DataFrame(
        {
            "condition_id": list(range(1, 9)),
            "outcome": [
                "Died",
                "Failure",
                "Cured",
                "Completed",
                "Lost to follow up",
                "Still on treatment",
                "Palliative Care",
                "Unknown",
            ],
        }
    )

    out = oc.derive_patient_case_died_failure(patient_cases)

    assert out["died_failure"].dtype == "boolean"
    assert out.loc[[1, 2], "died_failure"].astype(bool).all()
    assert not out.loc[[3, 4], "died_failure"].astype(bool).any()
    assert out.loc[[5, 6, 7, 8], "died_failure"].isna().all()
    assert out.attrs["died_failure_report"] == {
        "source_table": "patient_cases",
        "source_column": "outcome",
        "n_conditions": 8,
        "n_assessable": 4,
        "n_events": 2,
        "n_observed_nonevents": 2,
        "n_unassessable": 4,
        "event_values": ["died", "failure"],
        "observed_nonevent_values": ["completed", "cured"],
    }


def test_patient_case_died_failure_rejects_duplicate_or_missing_condition_ids() -> None:
    duplicate = pd.DataFrame(
        {
            "condition_id": [1, 1],
            "outcome": ["Cured", "Died"],
        }
    )
    with pytest.raises(ValueError, match="one row per condition"):
        oc.derive_patient_case_died_failure(duplicate)

    missing = pd.DataFrame(
        {
            "condition_id": [1, None],
            "outcome": ["Cured", "Died"],
        }
    )
    with pytest.raises(ValueError, match="missing_condition_ids=1"):
        oc.derive_patient_case_died_failure(missing)


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
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
                "activities_statusreason_cd": "resistance",
            },
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R,Bdq}",
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
            },
            # cond 2: pure extension (Sensitive, 250 > 216)
            {
                "patient_id": "b",
                "condition_id": 2,
                "regimen_drug": "{E,H,R,Z}",
                "period_end": "2020-06-01",
                "period_span": 250,
                "outcome_cd": "completed",
            },
            # cond 3: ONLY unfavorable (short, no escalation) -> primary False, secondary True
            {
                "patient_id": "c",
                "condition_id": 3,
                "regimen_drug": "{E,H,R,Z}",
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "died",
            },
        ]
    )
    pc = _cases(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            },
            {
                "patient_id": "b",
                "condition_id": 2,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            },
            {
                "patient_id": "c",
                "condition_id": 3,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            },
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
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
            }
        ]
    )
    pc = _cases(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            },
            {
                "patient_id": "z",
                "condition_id": 99,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            },
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
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
            }
        ]
    )
    pc = _cases(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            }
        ]
    )
    out = oc.derive_treatment_intensification(pc, reg)
    for col in [
        "escalation",
        "extension",
        "re_initiation",
        "unfavorable_outcome",
        "treatment_intensification",
        "treatment_intensification_with_unfavorable",
    ]:
        expected_dtype = (
            "boolean"
            if col
            in {
                "re_initiation",
                "treatment_intensification",
                "treatment_intensification_with_unfavorable",
            }
            else bool
        )
        assert out[col].dtype == expected_dtype


def test_composite_preserves_unassessable_reinitiation_and_report() -> None:
    reg = pd.DataFrame(
        {
            "condition_id": [1],
            "regimen_drug": ["{H,R}"],
            "activities_statusreason_cd": [None],
            "period_span": [100],
            "period_end": [180],
            "outcome_cd": ["cured"],
        }
    )
    pc = pd.DataFrame(
        {
            "patient_id": ["p"],
            "condition_id": [1],
            "registration_date": [2020],
            "type_of_resistance": ["Sensitive"],
        }
    )

    out = oc.derive_treatment_intensification(pc, reg)

    assert pd.isna(out.loc[1, "re_initiation"])
    assert pd.isna(out.loc[1, "treatment_intensification"])
    assert pd.isna(out.loc[1, "treatment_intensification_with_unfavorable"])
    assert out.attrs["reinitiation_report"]["coordinate_compatible"] is False
    with pytest.raises(ValueError, match=r"assessable=0, missing=1.*classes=\[\]"):
        oc.require_modelable_binary_outcome(
            out["treatment_intensification"],
            outcome_name="treatment_intensification",
        )


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
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
            },
            {
                "patient_id": "a",
                "condition_id": 1,
                "regimen_drug": "{H,R}",
                "period_end": "2020-06-01",
                "period_span": 100,
                "outcome_cd": "cured",
            },
        ]
    )
    pc = _cases(
        [
            {
                "patient_id": "a",
                "condition_id": 1,
                "registration_date": "2020-01-01",
                "type_of_resistance": "Sensitive",
            }
        ]
    )
    out = oc.derive_treatment_intensification(pc, reg)
    tab = oc.outcome_prevalence_table(out)
    assert {"component", "n_positive", "n_total", "prevalence"} <= set(tab.columns)
    unassessable = tab["component"].isin({"escalation_drugset", "escalation_key_drug"})
    assert (tab.loc[unassessable, "n_total"] == 0).all()
    assert tab.loc[unassessable, "prevalence"].isna().all()
    assert (tab.loc[~unassessable, "n_total"] == 1).all()
    assert tab.loc[~unassessable, "prevalence"].between(0, 1).all()
