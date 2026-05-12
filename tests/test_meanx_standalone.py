from __future__ import annotations

import numpy as np
import pytest

from scivol import ARX, GED, HARX, Normal, SkewT, StudentT


def _x_path(n_obs: int, *, seed: int, scale: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n_obs, 1)) * scale, dtype=np.float64)


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (ARX(1) + Normal(), _x_path(220, seed=1)),
        (HARX((1, 5)) + Normal(), _x_path(220, seed=2)),
    ],
)
def test_meanx_normal_public_fit_paths_are_finite(spec, x: np.ndarray) -> None:
    data = np.linspace(-0.02, 0.03, x.shape[0], dtype=np.float64)

    result_theta = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)
    result_log = spec.fit(data, x=x, solver="slsqp", log_mode=True, verbose=False)

    assert np.all(np.isfinite(result_theta.params))
    assert np.all(np.isfinite(result_log.params))
    assert result_theta.fit_info.optimization_space == "theta-space"
    assert result_log.fit_info.optimization_space == "z-space"


@pytest.mark.parametrize(
    ("spec", "x"),
    [
        (ARX(1) + StudentT(), _x_path(220, seed=3)),
        (HARX((1, 5)) + SkewT(), _x_path(220, seed=4)),
        (ARX(1) + GED(), _x_path(220, seed=5)),
    ],
)
def test_meanx_extended_public_fit_paths_are_finite(spec, x: np.ndarray) -> None:
    data = np.linspace(-0.015, 0.025, x.shape[0], dtype=np.float64)

    result_theta = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)
    result_log = spec.fit(data, x=x, solver="slsqp", log_mode=True, verbose=False)

    assert np.all(np.isfinite(result_theta.params))
    assert np.all(np.isfinite(result_log.params))
    assert result_theta.fit_info.optimization_space == "theta-space"
    assert result_log.fit_info.optimization_space == "z-space"


@pytest.mark.parametrize(
    ("spec", "params", "x", "future_x", "expected"),
    [
        (
            ARX(1) + Normal(),
            np.array([0.10, 0.40, 1.20], dtype=np.float64),
            np.zeros((40, 1), dtype=np.float64),
            np.array([[2.0], [3.0]], dtype=np.float64),
            lambda data, fx: 0.10 + 0.40 * data[-1] + 1.20 * fx[0, 0],
        ),
        (
            HARX((1, 5)) + Normal(),
            np.array([0.05, 0.60, 0.20, 1.50], dtype=np.float64),
            np.zeros((10, 1), dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            lambda data, fx: 0.05 + 0.60 * data[-1] + 0.20 * np.mean(data[-5:]) + 1.50 * fx[0, 0],
        ),
    ],
)
def test_meanx_normal_fixed_workflows_are_analytical_and_forecast_correct(
    spec,
    params: np.ndarray,
    x: np.ndarray,
    future_x: np.ndarray,
    expected,
) -> None:
    data = np.linspace(0.01, 0.04, x.shape[0], dtype=np.float64) if x.shape[0] == 40 else np.linspace(0.0, 0.09, x.shape[0], dtype=np.float64)

    assert np.isfinite(spec.loglikelihood(data, params, x=x))
    assert np.all(np.isfinite(spec.score(data, params, x=x)))
    assert np.all(np.isfinite(spec.hessian(data, params, x=x)))

    fixed = spec.fix(data, params, x=x)
    forecast = fixed.forecast(future_x.shape[0], x=future_x)

    assert forecast.mean[0] == pytest.approx(expected(data, future_x))
    assert np.all(np.isfinite(forecast.variance))


@pytest.mark.parametrize(
    ("spec", "params", "x", "future_x", "expected"),
    [
        (
            ARX(1) + StudentT(),
            np.array([0.10, 0.40, 1.20, 0.80, 8.0], dtype=np.float64),
            np.zeros((40, 1), dtype=np.float64),
            np.array([[2.0], [3.0]], dtype=np.float64),
            lambda data, fx: 0.10 + 0.40 * data[-1] + 1.20 * fx[0, 0],
        ),
        (
            HARX((1, 5)) + SkewT(),
            np.array([0.05, 0.60, 0.20, 1.50, 0.75, 8.0, -0.10], dtype=np.float64),
            np.zeros((10, 1), dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            lambda data, fx: 0.05 + 0.60 * data[-1] + 0.20 * np.mean(data[-5:]) + 1.50 * fx[0, 0],
        ),
        (
            ARX(1) + GED(),
            np.array([0.08, 0.35, 0.90, 0.65, 1.6], dtype=np.float64),
            np.zeros((40, 1), dtype=np.float64),
            np.array([[1.5]], dtype=np.float64),
            lambda data, fx: 0.08 + 0.35 * data[-1] + 0.90 * fx[0, 0],
        ),
    ],
)
def test_meanx_extended_fixed_workflows_are_analytical_and_forecast_correct(
    spec,
    params: np.ndarray,
    x: np.ndarray,
    future_x: np.ndarray,
    expected,
) -> None:
    data = (
        np.linspace(0.01, 0.04, x.shape[0], dtype=np.float64)
        if x.shape[0] == 40
        else np.linspace(0.0, 0.09, x.shape[0], dtype=np.float64)
    )

    assert np.isfinite(spec.loglikelihood(data, params, x=x))
    assert np.all(np.isfinite(spec.score(data, params, x=x)))
    assert np.all(np.isfinite(spec.hessian(data, params, x=x)))

    fixed = spec.fix(data, params, x=x)
    forecast = fixed.forecast(future_x.shape[0], x=future_x)

    assert forecast.mean[0] == pytest.approx(expected(data, future_x))
    assert np.all(np.isfinite(forecast.variance))


@pytest.mark.parametrize(
    ("spec", "params", "seed", "atol"),
    [
        (ARX(1) + Normal(), np.array([0.25, 0.40, 0.90], dtype=np.float64), 11, 0.18),
        (HARX((1, 5)) + Normal(), np.array([0.10, 0.55, 0.20, 0.80], dtype=np.float64), 17, 0.20),
    ],
)
def test_meanx_normal_parameter_recovery_on_simulated_data(
    spec,
    params: np.ndarray,
    seed: int,
    atol: float,
) -> None:
    burn = 200
    n_obs = 1800
    x = _x_path(n_obs + burn, seed=seed + 1000)
    sim = spec.simulate(n_obs, params, burn=burn, seed=seed, x=x)
    fit_x = x[burn:]

    result = spec.fit(sim.data, x=fit_x, solver="slsqp", log_mode=False, verbose=False)

    assert result.converged
    np.testing.assert_allclose(result.params, params, atol=atol, rtol=0.0)


def test_meanx_studentt_parameter_recovery_on_simulated_data() -> None:
    spec = ARX(1) + StudentT()
    params = np.array([0.15, 0.35, 0.70, 0.60, 8.0], dtype=np.float64)
    burn = 200
    n_obs = 1800
    x = _x_path(n_obs + burn, seed=2111)
    sim = spec.simulate(n_obs, params, burn=burn, seed=111, x=x)
    fit_x = x[burn:]

    result = spec.fit(sim.data, x=fit_x, solver="slsqp", log_mode=False, verbose=False)

    assert result.converged
    np.testing.assert_allclose(result.params[:4], params[:4], atol=0.22, rtol=0.0)
    assert result.params[4] == pytest.approx(params[4], abs=3.5)
