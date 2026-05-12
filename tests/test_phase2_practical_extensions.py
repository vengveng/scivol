from __future__ import annotations

import numpy as np
import pytest

from scivol import ARMA, ARX, EGARCH, HARX, AutoDensity, GARCH, GED, GJRGARCH, Normal, SkewT, StudentT


def _standard_normal_x(n_obs: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n_obs, 1)), dtype=np.float64)


def test_ged_filter_fix_and_forecast_paths_are_finite() -> None:
    spec = GARCH(1, 1) + GED()
    params = np.array([1e-6, 0.07, 0.90, 1.6], dtype=np.float64)

    sim = spec.simulate(300, params, burn=50, seed=123)
    state = spec.filter(sim.data, params)
    fixed = spec.fix(sim.data, params)
    forecast = fixed.forecast(3)

    assert state.distribution == "GED"
    assert np.isfinite(state.log_likelihood)
    assert np.all(np.isfinite(state.sigma2))
    assert np.all(np.isfinite(forecast.variance))
    assert np.all(np.isfinite(forecast.quantile(0.05)))


@pytest.mark.parametrize(
    ("spec", "params"),
    [
        (
            ARMA(1, 1) + EGARCH(1, 1) + SkewT(),
            np.array([0.001, 0.20, -0.12, -0.18, 0.10, -0.05, 0.92, 8.0, -0.2], dtype=np.float64),
        ),
        (
            ARMA(1, 1) + EGARCH(1, 1) + GED(),
            np.array([0.001, 0.20, -0.12, -0.18, 0.10, -0.05, 0.92, 1.5], dtype=np.float64),
        ),
    ],
)
def test_arma_egarch_extended_density_public_and_fixed_paths_are_finite(spec, params) -> None:
    sim = spec.simulate(320, params, burn=120, seed=20260512)
    result = spec.fit(sim.data, solver="slsqp", log_mode=True, verbose=False)

    assert np.all(np.isfinite(result.params))
    assert np.isfinite(spec.loglikelihood(sim.data, params))
    assert np.all(np.isfinite(spec.score(sim.data, params)))
    assert np.all(np.isfinite(spec.hessian(sim.data, params)))

    state = spec.filter(sim.data, params)
    fixed = spec.fix(sim.data, params)
    forecast = fixed.forecast(3)

    assert np.all(np.isfinite(state.sigma2))
    assert np.all(np.isfinite(forecast.variance))


def test_arx_fixed_forecast_uses_future_regressors() -> None:
    spec = ARX(1) + GARCH(1, 1) + Normal()
    data = np.linspace(0.01, 0.04, 40, dtype=np.float64)
    x_train = np.zeros((40, 1), dtype=np.float64)
    params = np.array([0.10, 0.40, 1.20, 1e-6, 0.05, 0.90], dtype=np.float64)

    fixed = spec.fix(data, params, x=x_train)
    future_x = np.array([[2.0], [3.0]], dtype=np.float64)
    forecast = fixed.forecast(2, x=future_x)

    expected_mean_1 = 0.10 + 0.40 * data[-1] + 1.20 * future_x[0, 0]
    assert forecast.mean[0] == pytest.approx(expected_mean_1)
    assert np.all(np.isfinite(forecast.variance))


def test_harx_fixed_forecast_uses_horizon_averages_and_future_regressors() -> None:
    spec = HARX((1, 5)) + GARCH(1, 1) + Normal()
    data = np.linspace(0.0, 0.09, 10, dtype=np.float64)
    x_train = np.zeros((10, 1), dtype=np.float64)
    params = np.array([0.05, 0.60, 0.20, 1.50, 1e-6, 0.05, 0.90], dtype=np.float64)

    fixed = spec.fix(data, params, x=x_train)
    future_x = np.array([[2.0]], dtype=np.float64)
    forecast = fixed.forecast(1, x=future_x)

    expected_mean_1 = 0.05 + 0.60 * data[-1] + 0.20 * np.mean(data[-5:]) + 1.50 * future_x[0, 0]
    assert forecast.mean[0] == pytest.approx(expected_mean_1)


def test_harx_validation_and_simulation_require_full_regressor_paths() -> None:
    spec = HARX((1, 5)) + GARCH(1, 1) + Normal()
    params = np.array([0.03, 0.45, 0.10, 0.80, 1e-6, 0.06, 0.90], dtype=np.float64)
    data = np.linspace(0.0, 0.02, 25, dtype=np.float64)

    with pytest.raises(ValueError, match="requires exogenous regressors"):
        spec.fit(data, solver="slsqp", verbose=False)

    fixed = spec.fix(data, params, x=np.zeros((25, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="HARX forecast requires x with shape"):
        fixed.forecast(2, x=np.zeros((1, 1), dtype=np.float64))

    sim_x = _standard_normal_x(30 + 20, seed=17)
    sim = spec.simulate(30, params, burn=20, seed=17, x=sim_x)
    assert sim.data.shape == (30,)


def test_hold_back_and_scale_are_recorded_in_fit_info() -> None:
    rng = np.random.default_rng(2026)
    data = rng.standard_normal(260).astype(np.float64) * 0.01

    result = (GARCH(1, 1) + Normal()).fit(
        data,
        hold_back=7,
        scale=100.0,
        solver="slsqp",
        verbose=False,
    )

    assert result.fit_info.requested_hold_back == 7
    assert result.fit_info.effective_hold_back == 7
    assert result.fit_info.original_n_obs == data.size
    assert result.fit_info.effective_n_obs == data.size - 7
    assert result.fit_info.scale == pytest.approx(100.0)
    assert result.data.shape == (data.size - 7,)


def test_auto_selection_common_sample_uses_max_candidate_hold_back() -> None:
    rng = np.random.default_rng(91)
    data = rng.standard_normal(220).astype(np.float64) * 0.01

    spec = GARCH(auto={"max_p": 2, "max_q": 1}) + AutoDensity(candidates=["Normal", "GED"])
    result = spec.fit(
        data,
        n_jobs=1,
        common_sample=True,
        solver="slsqp",
        verbose=False,
    )

    successful = [candidate.result for candidate in result._selection_candidates if candidate.result is not None]
    ged_candidates = [candidate for candidate in result._selection_candidates if "GED" in str(candidate.spec)]

    assert result.fit_info.common_sample is True
    assert result.fit_info.effective_hold_back == 2
    assert successful
    assert ged_candidates
    assert any(candidate.result is not None for candidate in ged_candidates)
    for candidate_result in successful:
        assert candidate_result.fit_info.common_sample is True
        assert candidate_result.fit_info.effective_hold_back == 2
        assert candidate_result.fit_info.effective_n_obs == data.size - 2


def test_default_auto_density_candidates_exclude_unsupported_ged_fit() -> None:
    assert AutoDensity().get_candidates() == ["Normal", "StudentT", "SkewT"]


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (ARX(1) + GARCH(1, 1) + StudentT(), np.zeros((40, 1), dtype=np.float64)),
        (HARX((1, 5)) + GARCH(1, 1) + SkewT(), np.zeros((40, 1), dtype=np.float64)),
        (ARX(1) + GARCH(1, 1) + GED(), np.zeros((40, 1), dtype=np.float64)),
        (ARX(1) + GJRGARCH(1, 1) + StudentT(), np.zeros((40, 1), dtype=np.float64)),
        (HARX((1, 5)) + GJRGARCH(1, 1) + SkewT(), np.zeros((40, 1), dtype=np.float64)),
        (ARX(1) + GJRGARCH(1, 1) + GED(), np.zeros((40, 1), dtype=np.float64)),
    ],
)
def test_phase2_linked_meanx_vol_fit_surfaces_are_finite(spec, x) -> None:
    data = np.linspace(0.01, 0.04, 40, dtype=np.float64)
    result_theta = spec.fit(data, x=x, hold_back=1, solver="slsqp", log_mode=False, verbose=False)
    result_log = spec.fit(data, x=x, hold_back=1, solver="slsqp", log_mode=True, verbose=False)

    assert np.all(np.isfinite(result_theta.params))
    assert np.all(np.isfinite(result_log.params))
    assert result_theta.fit_info.optimization_space == "theta-space"
    assert result_log.fit_info.optimization_space == "z-space"


def test_linked_meanx_gjr_surfaces_route_to_analytical_derivatives() -> None:
    spec = ARX(1) + GJRGARCH(1, 1) + StudentT()
    data = np.linspace(0.01, 0.04, 40, dtype=np.float64)
    x = np.zeros((40, 1), dtype=np.float64)
    params = np.array([0.10, 0.40, 1.20, 1e-6, 0.05, 0.03, 0.90, 8.0], dtype=np.float64)

    score = spec.score(data, params, x=x)
    hessian = spec.hessian(data, params, x=x)

    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))
