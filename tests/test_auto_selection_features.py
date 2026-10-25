"""Tests for the public auto-selection result and OOS selection features."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from scivol import AutoDensity, AutoVol, GARCH, Normal
from scivol._autoselect import (
    ModelCandidate,
    SelectionResult,
    _assess_candidate_quality,
    _compute_oos_losses,
    _diebold_mariano,
)


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
    assert {
        "model", "p", "q", "density", "aic", "bic", "score", "converged",
        "warm_started_from", "boundary_hugging", "near_integrated",
        "dm_pvalue_vs_best",
    }.issubset(frame.columns)
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

    assert np.isfinite(result.selection.candidates[1].dm_pvalue_vs_best)
    assert isinstance(result.selection.decisive, bool)


def test_auto_grid_records_lattice_and_density_warm_starts(sample_data: np.ndarray) -> None:
    spec = AutoVol(candidates=["GARCH"], max_p=2, max_q=1) + AutoDensity(
        candidates=["Normal", "StudentT"]
    )
    result = spec.fit(sample_data, n_jobs=1, select_by="bic")
    by_key = {
        (candidate.p, candidate.q, candidate.density): candidate
        for candidate in result.selection.candidates
    }

    assert by_key[(1, 1, "Normal")].warm_started_from is None
    assert by_key[(1, 1, "StudentT")].warm_started_from == "GARCH(1,1)+Normal"
    assert by_key[(2, 1, "Normal")].warm_started_from == "GARCH(1,1)+Normal"
    assert by_key[(2, 1, "StudentT")].warm_started_from == "GARCH(2,1)+Normal"
    assert by_key[(2, 1, "StudentT")].result.fit_info.warm_started is True


def test_candidate_quality_flags_the_domain_boundary() -> None:
    """White noise: alpha ends at its bound 0, so the fit is at the boundary."""
    spec = GARCH(1, 1) + Normal()
    result = spec.fit(np.random.default_rng(3).standard_normal(2000))
    assert result.fit_info.inference_status == "at_boundary"
    candidate = ModelCandidate(spec=spec, result=result)

    _assess_candidate_quality(candidate, degeneracy_weight=40.0, persistence_threshold=0.99)

    assert candidate.boundary_hugging is True
    assert candidate.boundary_parameters == ("alpha[1]",)
    assert candidate.near_integrated is True  # persistence 0.994
    assert candidate.degeneracy_penalty == 80.0


def test_diebold_mariano_detects_persistent_loss_advantage() -> None:
    winner = np.zeros(120, dtype=np.float64)
    runner = 0.2 + 0.01 * np.sin(np.arange(120, dtype=np.float64))

    statistic, pvalue = _diebold_mariano(runner, winner)

    assert statistic > 0.0
    assert pvalue < 0.01


def test_rolling_oos_refits_preserve_loss_alignment_and_no_lookahead(
    sample_data: np.ndarray,
) -> None:
    changed = sample_data.copy()
    changed[-40:] *= -2.0
    spec = AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()

    result_a = spec.fit(
        sample_data,
        n_jobs=1,
        select_by="oos",
        oos_test_size=60,
        oos_refit=20,
    )
    result_b = spec.fit(
        changed,
        n_jobs=1,
        select_by="oos",
        oos_test_size=60,
        oos_refit=20,
    )

    losses_a = result_a.selection.best.oos_losses
    losses_b = result_b.selection.best.oos_losses
    assert losses_a.shape == (60,)
    np.testing.assert_array_equal(losses_a[:20], losses_b[:20])
    assert result_a.fit_info.oos_refit == 20
    assert result_a.fit_info.selection_search_space["oos_refit"] == 20


def test_model_confidence_set_eliminates_clearly_inferior_model() -> None:
    rng = np.random.default_rng(20260820)
    common = rng.normal(1.0, 0.1, size=160)
    specs = [GARCH(order, 1) + Normal() for order in (1, 2, 3)]
    losses = [
        common,
        common + rng.normal(0.005, 0.03, size=160),
        common + 0.4 + rng.normal(0.0, 0.02, size=160),
    ]
    candidates = []
    for spec, candidate_losses in zip(specs, losses):
        candidates.append(ModelCandidate(
            spec=spec,
            oos_losses=candidate_losses,
            oos_loss=float(np.mean(candidate_losses)),
            score=float(np.mean(candidate_losses)),
        ))
    candidates.sort(key=lambda candidate: candidate.score)
    selection = SelectionResult(
        candidates,
        select_by="oos",
        criterion_name="oos:logscore",
        search_space={"n_candidates": 3},
    )

    confidence_set = selection.confidence_set(
        alpha=0.10,
        n_bootstrap=400,
        random_state=7,
    )

    assert candidates[2] in confidence_set.eliminated
    assert candidates[0] in confidence_set.candidates
    assert len(confidence_set) < 3


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
