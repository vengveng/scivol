"""Tests for bounded automatic conditional-mean selection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from scivol import ARMA, AutoDensity, AutoMean, AutoVol, GARCH, Normal


def test_automean_candidate_grid_is_complexity_ordered() -> None:
    auto = AutoMean(max_p=1, max_q=1)

    assert auto.signature == "AutoMean"
    assert auto.get_candidates() == [
        ("Zero", ()),
        ("ARMA", (0, 0)),
        ("ARMA", (0, 1)),
        ("ARMA", (1, 0)),
        ("ARMA", (1, 1)),
    ]


def test_automean_preview_composes_all_three_axes() -> None:
    spec = (
        AutoMean(max_p=1, max_q=0)
        + AutoVol(candidates=["GARCH"], max_p=1, max_q=1)
        + AutoDensity(candidates=["Normal", "StudentT"])
    )

    assert [str(candidate) for candidate in spec.preview_candidates()] == [
        "GARCH(1,1)+Normal",
        "GARCH(1,1)+StudentT",
        "ARMA(0,0)+GARCH(1,1)+Normal",
        "ARMA(0,0)+GARCH(1,1)+StudentT",
        "ARMA(1,0)+GARCH(1,1)+Normal",
        "ARMA(1,0)+GARCH(1,1)+StudentT",
    ]


def test_fixed_mean_is_preserved_during_volatility_selection() -> None:
    data = np.random.default_rng(20260820).normal(0.0, 0.01, size=360)
    spec = ARMA(1, 0) + AutoVol(
        candidates=["GARCH"], max_p=2, max_q=1
    ) + Normal()

    result = spec.fit(data, n_jobs=1, select_by="bic", show_progress=False)

    assert all(str(candidate.spec).startswith("ARMA(1,0)+") for candidate in result.selection.candidates)
    assert result.fit_info.selection_search_space["mean_candidates"] == ["ARMA(1,0)"]


def test_staged_automean_search_respects_budget_and_expands_finalists() -> None:
    data = np.random.default_rng(20260821).normal(0.0, 0.01, size=420)
    spec = (
        AutoMean(max_p=1, max_q=1, candidate_budget=12, finalists=2)
        + AutoVol(candidates=["GARCH"], max_p=2, max_q=1)
        + AutoDensity(candidates=["Normal", "StudentT"])
    )

    result = spec.fit(data, n_jobs=1, select_by="bic", show_progress=False)
    search = result.fit_info.selection_search_space
    counts = Counter(
        candidate.mean_signature for candidate in result.selection.candidates
    )

    assert search["mean_search_staged"] is True
    assert search["full_grid_candidates"] == 20
    assert search["n_candidates"] == 11
    assert search["n_candidates"] <= 12
    assert len(search["mean_finalists"]) == 2
    assert sorted(counts.values()) == [1, 1, 1, 4, 4]

