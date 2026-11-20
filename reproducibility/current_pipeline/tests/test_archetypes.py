"""Unit tests for the pure (non-neural) archetype analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctb.models import archetypes as arch


def test_assign_subtypes_argmax() -> None:
    z = np.array([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8], [0.3, 0.5, 0.2]])
    assert arch.assign_subtypes(z).tolist() == [0, 2, 1]


def test_assign_subtypes_kmeans_balanced() -> None:
    rng = np.random.default_rng(0)
    z = np.vstack([rng.normal(loc, 0.05, size=(20, 3)) for loc in ([3, 0, 0], [0, 3, 0])])
    labels = arch.assign_subtypes_kmeans(z, n_subtypes=2, seed=0)
    assert len(np.unique(labels)) == 2
    assert np.bincount(labels).min() == 20  # two clean clusters of 20


def test_archetype_treatment_profiles_drops_intercept() -> None:
    archetypes = np.array([[1.0, -2.0, 0.5, 9.0], [0.0, 1.0, -1.0, 9.0]])  # last col = intercept
    prof = arch.archetype_treatment_profiles(archetypes, ["drug_a", "drug_b", "drug_c"])
    assert prof.shape == (2, 3)
    assert list(prof.columns) == ["drug_a", "drug_b", "drug_c"]
    assert prof.iloc[0].tolist() == [1.0, -2.0, 0.5]


def test_subtype_context_profile_enrichment_sign() -> None:
    # subtype 1 has higher age -> positive enrichment for age in subtype_1.
    ctx = pd.DataFrame({"age": [30.0, 32.0, 60.0, 62.0], "hiv": [0, 0, 1, 1]})
    subtypes = np.array([0, 0, 1, 1])
    prof = arch.subtype_context_profile(subtypes, ctx, ["age", "hiv"])
    assert prof.loc["age", "subtype_1"] > 0 > prof.loc["age", "subtype_0"]
    assert prof.loc["hiv", "subtype_1"] > prof.loc["hiv", "subtype_0"]


def test_subtype_prevalence_by_group_rows_sum_to_one() -> None:
    subtypes = np.array([0, 1, 0, 1, 0, 1] * 10)
    group = pd.Series((["A"] * 30) + (["B"] * 30))
    prev = arch.subtype_prevalence_by_group(subtypes, group, min_group_n=10)
    props = prev.drop(columns="n")
    assert np.allclose(props.sum(axis=1), 1.0)
    assert set(prev.index) == {"A", "B"}


def test_subtype_prevalence_drops_small_groups() -> None:
    subtypes = np.array([0, 1] * 25 + [0] * 5)
    group = pd.Series((["big"] * 50) + (["tiny"] * 5))
    prev = arch.subtype_prevalence_by_group(subtypes, group, min_group_n=30)
    assert "tiny" not in prev.index
    assert "big" in prev.index


def test_match_archetypes_identity() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 6))
    perm = [2, 0, 3, 1]
    permuted = a[perm]
    col, score = arch._match_archetypes(a, permuted)
    # matching should recover near-perfect correlation
    assert score > 0.99
    # col[i] is the 'other' row matched to reference row i; permuted[col[i]] == a[i]
    assert np.allclose(permuted[col[0]], a[0])
