"""Safety tests for denominator-aware legacy figure generation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from matplotlib.figure import Figure

from ctb.data.outcomes import outcome_prevalence_table
from ctb.viz import figures as vf


def _outcomes() -> pd.DataFrame:
    outcomes = pd.DataFrame(
        {
            "escalation": pd.Series([True, False, True, False], dtype="boolean"),
            "extension": pd.Series([False, True, pd.NA, False], dtype="boolean"),
            "re_initiation": pd.Series([pd.NA, True, pd.NA, False], dtype="boolean"),
            "unfavorable_outcome": pd.Series([False, False, True, False], dtype="boolean"),
            "treatment_intensification": pd.Series(
                [True, True, pd.NA, False],
                dtype="boolean",
            ),
        }
    )
    outcomes.index = ["a", "b", "c", "d"]
    return outcomes


def _fake_upset_module(seen: dict[str, Any]) -> ModuleType:
    module = ModuleType("upsetplot")

    def from_indicators(components: list[str], indicators: pd.DataFrame) -> pd.DataFrame:
        seen["components"] = components
        seen["indicators"] = indicators.copy()
        return indicators

    class UpSet:
        def __init__(self, data: pd.DataFrame, **_: object) -> None:
            seen["data"] = data

        def plot(self, *, fig: Figure) -> None:
            fig.add_subplot(111)

    module.from_indicators = from_indicators  # type: ignore[attr-defined]
    module.UpSet = UpSet  # type: ignore[attr-defined]
    return module


def test_prevalence_annotations_disclose_positive_and_assessable_denominators(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outcomes = _outcomes()
    prevalence = outcome_prevalence_table(outcomes)
    seen: dict[str, Any] = {}
    captured: dict[str, Figure] = {}

    def capture(fig: Figure, path: Path) -> list[Path]:
        captured[path.name] = fig
        return [path.with_suffix(".png"), path.with_suffix(".pdf")]

    monkeypatch.setitem(sys.modules, "upsetplot", _fake_upset_module(seen))
    monkeypatch.setattr(vf, "save_figure", capture)

    vf.figure_outcome_prevalence(prevalence, outcomes, out_dir=tmp_path)

    prevalence_text = {
        text.get_text() for text in captured["figure2a_outcome_prevalence"].axes[0].texts
    }
    assert any("50.0% (1/2); assessable 2/4" in text for text in prevalence_text)
    assert any("assessable 3/4" in text for text in prevalence_text)
    assert list(seen["indicators"].index) == ["b", "d"]
    assert not seen["indicators"].isna().any().any()
    upset_title = captured["figure2b_outcome_upset"]._suptitle
    assert upset_title is not None
    assert "n=2/4; excluded 2" in upset_title.get_text()


def test_zero_assessable_components_remain_visible_and_replace_stale_upset(
    tmp_path: Path,
) -> None:
    outcomes = pd.DataFrame(
        {
            component: pd.Series([pd.NA] * 4, dtype="boolean")
            for component in [
                "escalation",
                "extension",
                "re_initiation",
                "unfavorable_outcome",
                "treatment_intensification",
            ]
        }
    )
    prevalence = outcome_prevalence_table(outcomes)
    stale_png = tmp_path / "figure2b_outcome_upset.png"
    stale_pdf = tmp_path / "figure2b_outcome_upset.pdf"
    stale_png.write_bytes(b"stale")
    stale_pdf.write_bytes(b"stale")

    paths = vf.figure_outcome_prevalence(prevalence, outcomes, out_dir=tmp_path)

    assert stale_png in paths
    assert stale_pdf in paths
    assert stale_png.stat().st_size > len(b"stale")
    assert stale_pdf.stat().st_size > len(b"stale")
    assert b"stale" not in stale_png.read_bytes()
    assert b"stale" not in stale_pdf.read_bytes()


def test_upset_failure_replaces_stale_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outcomes = _outcomes()
    prevalence = outcome_prevalence_table(outcomes)
    stale_png = tmp_path / "figure2b_outcome_upset.png"
    stale_pdf = tmp_path / "figure2b_outcome_upset.pdf"
    stale_png.write_bytes(b"STALE")
    stale_pdf.write_bytes(b"STALE")
    module = _fake_upset_module({})

    def fail_from_indicators(
        _components: list[str],
        _indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        raise RuntimeError("synthetic plotting failure")

    module.from_indicators = fail_from_indicators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "upsetplot", module)

    paths = vf.figure_outcome_prevalence(prevalence, outcomes, out_dir=tmp_path)

    assert stale_png in paths
    assert stale_pdf in paths
    assert b"STALE" not in stale_png.read_bytes()
    assert b"STALE" not in stale_pdf.read_bytes()


def test_plotting_rejects_string_booleans_without_coercion(tmp_path: Path) -> None:
    outcomes = _outcomes()
    outcomes["escalation"] = pd.Series(["False", "True", "False", "True"], index=outcomes.index)
    prevalence = pd.DataFrame(
        [
            {
                "component": "escalation",
                "n_positive": 2,
                "n_total": 4,
                "prevalence": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="not coerced"):
        vf.figure_outcome_prevalence(prevalence, outcomes, out_dir=tmp_path)


def test_plotting_rejects_prevalence_counts_that_disagree_with_cohort(tmp_path: Path) -> None:
    outcomes = _outcomes()
    prevalence = outcome_prevalence_table(outcomes)
    prevalence.loc[prevalence["component"].eq("extension"), "n_total"] = len(outcomes) + 1

    with pytest.raises(ValueError, match="disagree with outcomes"):
        vf.figure_outcome_prevalence(prevalence, outcomes, out_dir=tmp_path)
