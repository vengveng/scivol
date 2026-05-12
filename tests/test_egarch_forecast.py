"""Regression coverage for shipped EGARCH forecast and simulation paths."""

from __future__ import annotations

import numpy as np
import pytest

from scivol import GED, Normal, SkewT, StudentT
from scivol._evaluation import filter_spec, forecast_from_state, simulate_spec
from scivol.components.vol import EGARCH


def test_egarch_normal_one_step_forecast_is_analytic() -> None:
    spec = EGARCH(1, 1) + Normal()
    params = {"omega": -0.15, "alpha": [0.12], "gamma": [-0.05], "beta": [0.92]}
    sim = simulate_spec(spec, 600, params, burn=200, seed=123)
    state = filter_spec(spec, sim.data, params)
    forecast = forecast_from_state(spec, state, 1)

    sigma_last = float(state.sigma2[-1])
    z_last = float(state.residuals[-1] / np.sqrt(sigma_last))
    expected = np.exp(
        params["omega"]
        + params["alpha"][0] * (abs(z_last) - np.sqrt(2.0 / np.pi))
        + params["gamma"][0] * z_last
        + params["beta"][0] * np.log(sigma_last)
    )

    np.testing.assert_allclose(forecast.residual_variance[0], expected, rtol=1e-12, atol=1e-12)


def test_egarch_studentt_multi_step_forecast_is_deterministic_and_positive() -> None:
    spec = EGARCH(1, 1) + StudentT()
    params = {"omega": -0.15, "alpha": [0.10], "gamma": [-0.04], "beta": [0.90], "nu": 8.0}
    sim = simulate_spec(spec, 500, params, burn=200, seed=321)
    state = filter_spec(spec, sim.data, params)

    forecast_a = forecast_from_state(spec, state, 5)
    forecast_b = forecast_from_state(spec, state, 5)

    np.testing.assert_allclose(forecast_a.residual_variance, forecast_b.residual_variance, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(forecast_a.residual_variance))
    assert np.all(forecast_a.residual_variance > 0.0)


def test_egarch_normal_pq_forecast_is_deterministic_and_positive() -> None:
    spec = EGARCH(2, 1) + Normal()
    params = {"omega": -0.12, "alpha": [0.08, 0.04], "gamma": [-0.03, 0.01], "beta": [0.90]}
    sim = simulate_spec(spec, 500, params, burn=200, seed=456)
    state = filter_spec(spec, sim.data, params)

    forecast_a = forecast_from_state(spec, state, 5)
    forecast_b = forecast_from_state(spec, state, 5)

    np.testing.assert_allclose(forecast_a.residual_variance, forecast_b.residual_variance, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(forecast_a.residual_variance))
    assert np.all(forecast_a.residual_variance > 0.0)


def test_egarch_skewt_forecast_is_deterministic_and_positive() -> None:
    spec = EGARCH(2, 1) + SkewT()
    params = {"omega": -0.15, "alpha": [0.10, 0.03], "gamma": [-0.04, 0.01], "beta": [0.90], "nu": 8.0, "lam": -0.2}
    sim = simulate_spec(spec, 500, params, burn=200, seed=456)
    state = filter_spec(spec, sim.data, params)

    forecast_a = forecast_from_state(spec, state, 5)
    forecast_b = forecast_from_state(spec, state, 5)

    np.testing.assert_allclose(forecast_a.residual_variance, forecast_b.residual_variance, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(forecast_a.residual_variance))
    assert np.all(forecast_a.residual_variance > 0.0)


def test_egarch_ged_fixed_forecast_and_quantiles_are_finite() -> None:
    spec = EGARCH(2, 1) + GED()
    params = {"omega": -0.14, "alpha": [0.09, 0.02], "gamma": [-0.03, 0.01], "beta": [0.91], "nu": 1.5}
    sim = simulate_spec(spec, 500, params, burn=200, seed=654)
    state = filter_spec(spec, sim.data, params)

    forecast_a = forecast_from_state(spec, state, 4)
    fixed = spec.fix(sim.data, params)
    forecast_b = fixed.forecast(4)

    np.testing.assert_allclose(forecast_a.residual_variance, forecast_b.residual_variance, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(forecast_a.residual_variance))
    assert np.all(forecast_a.residual_variance > 0.0)
    assert np.all(np.isfinite(forecast_b.quantile(np.array([0.01, 0.05]))))
