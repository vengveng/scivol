"""Tests for the public auto-selection result and OOS selection features."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scivol import AutoDensity, AutoVol, GARCH, Normal
from scivol._autoselect import _compute_oos_losses


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(20260705)
    return rng.standard_normal(420) * 0.01


def test_preview_candidates_matches_search_space() -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=2, max_q=1) + AutoDensity(candidates=["Normal", "StudentT"])
    candidates = spec.preview_candidates()

    assert len(candidates) == 4
    assert [str(candidate) for candidate in candidates] == [
        "GARCH(1,1)+Normal",
        "GARCH(1,1)+StudentT",
        "GARCH(2,1)+Normal",
        "GARCH(2,1)+StudentT",
    ]


def test_selection_object_wraps_existing_candidate_list(sample_data: np.ndarray) -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=2, max_q=1) + Normal()
    result = spec.fit(sample_data, n_jobs=1, select_by="bic")

    assert result.selection.candidates is result._selection_candidates
    assert result.selection.best is result._selection_candidates[0]
    assert result.fit_info.selected_by == "bic"
    assert result.fit_info.selection_search_space["n_candidates"] == 2
    for candidate in result.selection.candidates:
        assert candidate.score == pytest.approx(candidate.bic)


def test_selection_to_dataframe(sample_data: np.ndarray) -> None:
    pd = pytest.importorskip("pandas")

    spec = AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()
    result = spec.fit(sample_data, n_jobs=1)
    frame = result.selection.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert {"model", "p", "q", "density", "aic", "bic", "score", "converged"}.issubset(frame.columns)
    assert frame.loc[0, "model"] == str(result.selection.best.spec)


def test_custom_criterion_cannot_be_combined_with_nondefault_select_by(sample_data: np.ndarray) -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()

    with pytest.raises(ValueError, match="custom criterion"):
        spec.fit(sample_data, n_jobs=1, select_by="bic", criterion=lambda result, diagnostics: result.aic)


def test_oos_selection_records_losses_and_refits_full_sample(sample_data: np.ndarray) -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=2, max_q=1) + Normal()
    result = spec.fit(sample_data, n_jobs=1, select_by="oos", oos_test_size=80, min_train_size=250)

    assert result.fit_info.selected_by == "oos:logscore"
    assert result.fit_info.selection_refit_on_full is True
    assert result.fit_info.oos_test_size == 80
    assert result.fit_info.original_n_obs == sample_data.size

    for candidate in result.selection.candidates:
        assert candidate.oos_losses is not None
        assert candidate.oos_losses.shape == (80,)
        assert candidate.oos_loss == pytest.approx(float(np.mean(candidate.oos_losses)))
        assert candidate.score == pytest.approx(candidate.oos_loss)
        assert np.all(np.isfinite(candidate.oos_losses))


def test_oos_density_blind_loss_rejects_autodensity(sample_data: np.ndarray) -> None:
    spec = GARCH(1, 1) + AutoDensity(candidates=["Normal", "StudentT"])

    with pytest.raises(ValueError, match="density-blind"):
        spec.fit(sample_data, n_jobs=1, select_by="oos", oos_loss="qlike")


def test_explicit_small_oos_holdout_warns(sample_data: np.ndarray) -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.fit(sample_data, n_jobs=1, select_by="oos", oos_test_size=30, min_train_size=250)

    assert any("fewer than 50" in str(item.message) for item in caught)


def test_oos_training_fit_has_no_holdout_lookahead(sample_data: np.ndarray) -> None:
    train = sample_data[:340].copy()
    holdout_a = sample_data[340:].copy()
    holdout_b = holdout_a.copy()
    holdout_b *= -3.0

    spec = AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()
    result_a = spec.fit(np.concatenate([train, holdout_a]), n_jobs=1, select_by="oos", oos_test_size=80)
    result_b = spec.fit(np.concatenate([train, holdout_b]), n_jobs=1, select_by="oos", oos_test_size=80)

    params_a = result_a.selection.candidates[0].result.params
    params_b = result_b.selection.candidates[0].result.params
    np.testing.assert_array_equal(params_a, params_b)


def test_oos_logscore_matches_manual_normal_density(sample_data: np.ndarray) -> None:
    spec = GARCH(1, 1) + Normal()
    train = sample_data[:-60]
    result = spec.fit(train, n_jobs=1)
    losses = _compute_oos_losses(
        result,
        sample_data,
        full_x=None,
        start=sample_data.size - 60,
        stop=sample_data.size,
        loss="logscore",
    )

    state = result.spec.filter(sample_data, result.params)
    hold_back = int(result.fit_info.effective_hold_back or 0)
    resid = state.residuals[sample_data.size - 60 - hold_back:]
    sigma2 = np.maximum(state.sigma2[sample_data.size - 60 - hold_back:], 1e-12)
    manual = 0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + resid * resid / sigma2)

    np.testing.assert_allclose(losses, manual, rtol=1e-10, atol=1e-12)
