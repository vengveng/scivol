from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scivol import ARMA, GARCH, GJRGARCH, Normal, StudentT


HAS_ARCH = importlib.util.find_spec("arch") is not None


def _legacy_simulate_garch_normal(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return y


def _legacy_simulate_arma_garch_normal(
    n: int,
    c: float,
    phi: float,
    theta: float,
    omega: float,
    alpha: float,
    beta: float,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    y[0] = c / (1.0 - phi) + eps[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]
    return y


def _legacy_simulate_gjr_garch_normal(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - 0.5 * gamma - beta)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        e_prev = y[t - 1]
        sigma2[t] = omega + alpha * e_prev**2 + gamma * (e_prev < 0.0) * e_prev**2 + beta * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return y


def test_public_evaluation_api_returns_consistent_fixed_workflow() -> None:
    spec = GARCH(1, 1) + Normal()
    params = np.array([1e-6, 0.08, 0.90], dtype=np.float64)
    data = _legacy_simulate_garch_normal(300, *params, seed=123)

    ll = spec.loglikelihood(data, params)
    score = spec.score(data, params)
    hess = spec.hessian(data, params)
    state = spec.filter(data, params)
    fixed = spec.fix(data, params)

    assert ll == pytest.approx(state.log_likelihood)
    assert ll == pytest.approx(fixed.log_likelihood)
    assert score.shape == (3,)
    assert hess.shape == (3, 3)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hess))
    assert fixed.method == "FIXED"
    assert fixed.parameter_source == "supplied"
    assert fixed.to_dict()["parameter_source"] == "supplied"


def test_public_garch_simulation_matches_legacy_helper() -> None:
    spec = GARCH(1, 1) + Normal()
    params = np.array([1e-6, 0.08, 0.90], dtype=np.float64)

    sim = spec.simulate(250, params, burn=0, seed=77)
    legacy = _legacy_simulate_garch_normal(250, *params, seed=77)

    np.testing.assert_allclose(sim.data, legacy)


def test_public_arma_garch_simulation_matches_legacy_helper() -> None:
    spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
    params = np.array([0.0002, 0.1, -0.05, 1e-6, 0.08, 0.90], dtype=np.float64)

    sim = spec.simulate(250, params, burn=0, seed=91)
    legacy = _legacy_simulate_arma_garch_normal(250, *params, seed=91)

    np.testing.assert_allclose(sim.data, legacy)


def test_public_gjr_simulation_matches_legacy_helper() -> None:
    spec = GJRGARCH(1, 1) + Normal()
    params = np.array([1e-6, 0.05, 0.06, 0.90], dtype=np.float64)

    sim = spec.simulate(250, params, burn=0, seed=15)
    legacy = _legacy_simulate_gjr_garch_normal(250, *params, seed=15)

    np.testing.assert_allclose(sim.data, legacy)


def test_fixed_forecast_matches_fitted_forecast_for_garch() -> None:
    spec = GARCH(1, 1) + Normal()
    data = _legacy_simulate_garch_normal(400, 1e-6, 0.08, 0.90, seed=44)

    fitted = spec.fit(data, solver="slsqp", verbose=False)
    fixed = spec.fix(data, fitted.params)

    np.testing.assert_allclose(fitted.forecast(8).variance, fixed.forecast(8).variance)
    np.testing.assert_allclose(fitted.forecast(8).residual_variance, fixed.forecast(8).residual_variance)


def test_fixed_forecast_matches_fitted_forecast_for_arma_garch() -> None:
    spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
    data = _legacy_simulate_arma_garch_normal(450, 0.0002, 0.1, -0.05, 1e-6, 0.08, 0.90, seed=55)

    fitted = spec.fit(data, solver="slsqp", verbose=False)
    fixed = spec.fix(data, fitted.params)

    fc_fit = fitted.forecast(6)
    fc_fixed = fixed.forecast(6)
    np.testing.assert_allclose(fc_fit.mean, fc_fixed.mean)
    np.testing.assert_allclose(fc_fit.variance, fc_fixed.variance)
    np.testing.assert_allclose(fc_fit.residual_variance, fc_fixed.residual_variance)


def test_gjr_forecast_uses_negative_shock_leverage() -> None:
    spec = GJRGARCH(1, 1) + Normal()
    params = np.array([1e-6, 0.04, 0.12, 0.88], dtype=np.float64)

    base = np.zeros(120, dtype=np.float64)
    positive = base.copy()
    negative = base.copy()
    positive[-1] = 0.01
    negative[-1] = -0.01

    pos_fc = spec.fix(positive, params).forecast(1)
    neg_fc = spec.fix(negative, params).forecast(1)

    assert neg_fc.residual_variance[0] > pos_fc.residual_variance[0]


def test_std_resid_uses_filtered_residuals_for_arma_garch() -> None:
    spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
    params = np.array([0.0002, 0.15, -0.08, 1e-6, 0.08, 0.90], dtype=np.float64)
    data = _legacy_simulate_arma_garch_normal(220, *params, seed=18)

    fixed = spec.fix(data, params)
    state = fixed.filter()

    np.testing.assert_allclose(np.asarray(fixed.std_resid), state.residuals / np.sqrt(state.sigma2))
    assert not np.allclose(np.asarray(fixed.std_resid), np.asarray(fixed.data) / np.sqrt(state.sigma2))


@pytest.mark.skipif(not HAS_ARCH, reason="arch is not installed")
def test_arch_fixed_loglikelihood_and_forecast_parity_garch_normal() -> None:
    from arch import arch_model

    params = np.array([1e-6, 0.08, 0.90], dtype=np.float64)
    data = _legacy_simulate_garch_normal(400, *params, seed=202)

    spec = GARCH(1, 1) + Normal()
    fixed = spec.fix(data, params)

    am = arch_model(data, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    arch_fixed = am.fix(params)
    arch_fc = arch_fixed.forecast(horizon=5, reindex=False)

    assert fixed.log_likelihood == pytest.approx(float(arch_fixed.loglikelihood), rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(fixed.forecast(5).variance, arch_fc.variance.values[-1], rtol=1e-6, atol=1e-10)


@pytest.mark.skipif(not HAS_ARCH, reason="arch is not installed")
def test_arch_fixed_forecast_parity_garch_studentt_and_gjr_normal() -> None:
    from arch import arch_model

    garch_t_params = np.array([1e-6, 0.08, 0.90, 8.0], dtype=np.float64)
    garch_t_data = _legacy_simulate_garch_normal(350, 1e-6, 0.08, 0.90, seed=303)
    spec_t = GARCH(1, 1) + StudentT()
    fc_t = spec_t.fix(garch_t_data, garch_t_params).forecast(4)
    arch_t = arch_model(garch_t_data, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(garch_t_params)
    np.testing.assert_allclose(fc_t.variance, arch_t.forecast(horizon=4, reindex=False).variance.values[-1], rtol=1e-6, atol=1e-10)

    gjr_params = np.array([1e-6, 0.05, 0.06, 0.90], dtype=np.float64)
    gjr_data = _legacy_simulate_gjr_garch_normal(350, *gjr_params, seed=404)
    spec_gjr = GJRGARCH(1, 1) + Normal()
    fc_gjr = spec_gjr.fix(gjr_data, gjr_params).forecast(4)
    arch_gjr = arch_model(gjr_data, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False).fix(gjr_params)
    np.testing.assert_allclose(fc_gjr.variance, arch_gjr.forecast(horizon=4, reindex=False).variance.values[-1], rtol=1e-6, atol=1e-10)
