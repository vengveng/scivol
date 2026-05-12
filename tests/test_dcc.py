"""
DCC-GARCH Tests
===============

Tests for the DCC(p,q) Gaussian correlation model:
  1. C kernel correctness (vs Python prototype)
  2. Gradient validation (vs AD oracle)
  3. Hessian validation (vs AD oracle)
  4. _11 vs _pq consistency
  5. DCC class: convergence and parameter recovery
  6. Sandwich standard errors
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
import numpy as np
import pytest
from numpy.typing import NDArray

from scivol import CCC, DCC, GARCH, Normal
from scivol._devtools.ad_oracle import dcc_value_grad_hess
from scivol._dcc_kernels import (
    dcc_nll, dcc_nll_grad, dcc_nll_grad_hess, compute_qbar,
)


HAS_JAX = importlib.util.find_spec("jax") is not None


# =============================================================================
# DGP simulation
# =============================================================================

def simulate_dcc_garch(
    T: int,
    garch_params: list,
    dcc_a: NDArray[np.float64],
    dcc_b: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    seed: int = 42,
) -> dict:
    """Simulate DCC-GARCH returns and standardised residuals."""
    rng = np.random.default_rng(seed)
    N = len(garch_params)
    p, q = len(dcc_a), len(dcc_b)

    returns = np.zeros((T, N))
    sigma2 = np.zeros((T, N))
    eps = np.zeros((T, N))

    intercept = (1.0 - np.sum(dcc_a) - np.sum(dcc_b)) * Qbar
    Q_buf = [Qbar.copy() for _ in range(max(q, 1))]
    eps_buf = [np.zeros(N) for _ in range(max(p, 1))]

    for i in range(N):
        omega, alpha, beta = garch_params[i]
        sigma2[0, i] = omega / (1.0 - alpha - beta)

    for t in range(T):
        Qt = intercept.copy()
        for i in range(p):
            e = eps_buf[i]
            Qt = Qt + dcc_a[i] * np.outer(e, e)
        for j in range(q):
            Qt = Qt + dcc_b[j] * Q_buf[j]

        s = np.sqrt(np.maximum(np.diag(Qt), 1e-12))
        Rt = Qt * np.outer(1.0 / s, 1.0 / s)

        L = np.linalg.cholesky(Rt)
        eps_t = L @ rng.standard_normal(N)
        eps[t] = eps_t

        if t > 0:
            for i in range(N):
                omega, alpha, beta = garch_params[i]
                sigma2[t, i] = omega + alpha * returns[t - 1, i] ** 2 + beta * sigma2[t - 1, i]
        returns[t] = np.sqrt(sigma2[t]) * eps_t

        if p > 0:
            eps_buf = [eps_t.copy()] + eps_buf[: p - 1]
        if q > 0:
            Q_buf = [Qt.copy()] + Q_buf[: q - 1]

    return {"returns": returns, "sigma2": sigma2, "eps": eps}


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture(scope="module")
def dcc_dgp_data():
    """Generate DCC(1,1) data for all tests."""
    T, N = 5000, 3
    garch_params = [(1e-6, 0.05, 0.93), (2e-6, 0.08, 0.90), (1.5e-6, 0.06, 0.92)]
    dcc_a = np.array([0.05])
    dcc_b = np.array([0.93])
    theta = np.concatenate([dcc_a, dcc_b])
    Qbar_true = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    dgp = simulate_dcc_garch(T, garch_params, dcc_a, dcc_b, Qbar_true, seed=42)
    eps = dgp["eps"]
    Qbar = compute_qbar(eps)
    return {
        "eps": eps, "Qbar": Qbar, "theta_true": theta,
        "T": T, "N": N, "p": 1, "q": 1,
    }


@pytest.fixture(scope="module")
def dcc_oracle_data():
    """Smaller DCC sample for AD-oracle derivative validation."""
    T, N = 400, 3
    garch_params = [(1e-6, 0.05, 0.93), (2e-6, 0.08, 0.90), (1.5e-6, 0.06, 0.92)]
    dcc_a = np.array([0.05])
    dcc_b = np.array([0.93])
    Qbar_true = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    dgp = simulate_dcc_garch(T, garch_params, dcc_a, dcc_b, Qbar_true, seed=123)
    eps = dgp["eps"]
    return {"eps": eps, "Qbar": compute_qbar(eps)}


@pytest.fixture(scope="module")
def fitted_multivariate_results():
    """Shared fitted CCC/DCC results with Normal marginals."""
    T, N = 1200, 3
    garch_params = [(1e-6, 0.05, 0.93), (2e-6, 0.08, 0.90), (1.5e-6, 0.06, 0.92)]
    dcc_a = np.array([0.05])
    dcc_b = np.array([0.93])
    qbar_true = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    dgp = simulate_dcc_garch(T, garch_params, dcc_a, dcc_b, qbar_true, seed=2024)
    returns = dgp["returns"]
    univariate_results = [(GARCH(1, 1) + Normal()).fit(returns[:, i]) for i in range(N)]

    dcc_result = DCC(1, 1).fit(returns, univariate_results=univariate_results)
    ccc_result = CCC().fit(returns, univariate_results=univariate_results)
    return {
        "returns": returns,
        "univariate_results": univariate_results,
        "dcc": dcc_result,
        "ccc": ccc_result,
    }


# =============================================================================
# 1. NLL consistency: _11 vs _pq
# =============================================================================

class TestNLLConsistency:
    """NLL from _11 and _pq must match."""

    def test_nll_11_vs_pq(self, dcc_dgp_data):
        theta = dcc_dgp_data["theta_true"]
        eps, Qbar = dcc_dgp_data["eps"], dcc_dgp_data["Qbar"]
        # _11 call
        nll_11 = dcc_nll(theta, eps, Qbar, 1, 1)
        # force _pq path by passing through the pq interface
        from scivol import _core
        eps_c = np.ascontiguousarray(eps, dtype=np.float64)
        Qbar_c = np.ascontiguousarray(Qbar.ravel(), dtype=np.float64)
        theta_c = np.ascontiguousarray(theta, dtype=np.float64)
        nll_pq = _core._dcc_nll_pq_gaussian(
            theta_c.ctypes.data, eps_c.ctypes.data, Qbar_c.ctypes.data,
            5000, 3, 1, 1)
        assert np.isclose(nll_11, nll_pq, rtol=1e-12)


# =============================================================================
# 2. Gradient validation via AD oracle
# =============================================================================

@pytest.mark.skipif(not HAS_JAX, reason="jax is required for AD-oracle derivative checks")
class TestGradientValidation:
    """Analytical gradient must match the AD oracle."""

    @pytest.mark.parametrize("theta,p,q", [
        (np.array([0.05, 0.93]), 1, 1),
        (np.array([0.10, 0.80]), 1, 1),
        (np.array([0.02, 0.97]), 1, 1),
        (np.array([0.03, 0.02, 0.90]), 2, 1),
        (np.array([0.04, 0.45, 0.45]), 1, 2),
        (np.array([0.02, 0.01, 0.50, 0.40]), 2, 2),
    ])
    def test_gradient_vs_ad(self, dcc_oracle_data, theta, p, q):
        eps, Qbar = dcc_oracle_data["eps"], dcc_oracle_data["Qbar"]
        _, grad_a = dcc_nll_grad(theta, eps, Qbar, p, q)
        _, grad_ref, _ = dcc_value_grad_hess(theta, eps, Qbar, p, q)
        np.testing.assert_allclose(grad_a, grad_ref, rtol=1e-8, atol=1e-10)


# =============================================================================
# 3. Hessian validation via AD oracle
# =============================================================================

@pytest.mark.skipif(not HAS_JAX, reason="jax is required for AD-oracle derivative checks")
class TestHessianValidation:
    """Analytical Hessian must match the AD oracle."""

    @pytest.mark.parametrize("theta,p,q", [
        (np.array([0.05, 0.93]), 1, 1),
        (np.array([0.10, 0.80]), 1, 1),
        (np.array([0.03, 0.02, 0.90]), 2, 1),
        (np.array([0.04, 0.45, 0.45]), 1, 2),
        (np.array([0.02, 0.01, 0.50, 0.40]), 2, 2),
    ])
    def test_hessian_vs_ad(self, dcc_oracle_data, theta, p, q):
        eps, Qbar = dcc_oracle_data["eps"], dcc_oracle_data["Qbar"]
        _, _, hess_a = dcc_nll_grad_hess(theta, eps, Qbar, p, q)
        _, _, hess_ref = dcc_value_grad_hess(theta, eps, Qbar, p, q)
        np.testing.assert_allclose(hess_a, hess_ref, rtol=1e-8, atol=1e-10)


# =============================================================================
# 4. Per-obs scores
# =============================================================================

class TestScores:
    """Per-obs scores must be consistent with total gradient."""

    def test_scores_sum_to_grad(self, dcc_dgp_data):
        theta = dcc_dgp_data["theta_true"]
        eps, Qbar = dcc_dgp_data["eps"], dcc_dgp_data["Qbar"]
        T = dcc_dgp_data["T"]
        nll, grad, scores = dcc_nll_grad(theta, eps, Qbar, 1, 1, return_scores=True)
        grad_from_scores = -scores.sum(axis=0) / T
        np.testing.assert_allclose(grad, grad_from_scores, rtol=1e-10)

    def test_scores_from_hess(self, dcc_dgp_data):
        theta = dcc_dgp_data["theta_true"]
        eps, Qbar = dcc_dgp_data["eps"], dcc_dgp_data["Qbar"]
        T = dcc_dgp_data["T"]
        nll, grad, hess, scores = dcc_nll_grad_hess(
            theta, eps, Qbar, 1, 1, return_scores=True)
        grad_from_scores = -scores.sum(axis=0) / T
        np.testing.assert_allclose(grad, grad_from_scores, rtol=1e-10)


# =============================================================================
# 5. DCC class: convergence and parameter recovery
# =============================================================================

class TestDCCFit:
    """Test the DCC class fits and recovers parameters."""

    def test_convergence(self, dcc_dgp_data):
        eps = dcc_dgp_data["eps"]
        dcc = DCC(1, 1)
        result = dcc.fit_from_residuals(eps)
        assert result.converged

    def test_parameter_recovery(self, dcc_dgp_data):
        eps = dcc_dgp_data["eps"]
        theta_true = dcc_dgp_data["theta_true"]
        dcc = DCC(1, 1)
        result = dcc.fit_from_residuals(eps)
        np.testing.assert_allclose(result.theta, theta_true, rtol=0.15)

    def test_standard_errors_exist(self, dcc_dgp_data):
        eps = dcc_dgp_data["eps"]
        dcc = DCC(1, 1)
        result = dcc.fit_from_residuals(eps)
        assert not np.any(np.isnan(result.std_errors))
        assert not np.any(np.isnan(result.std_errors_robust))
        assert np.all(result.std_errors > 0)
        assert np.all(result.std_errors_robust > 0)

    def test_summary(self, dcc_dgp_data):
        eps = dcc_dgp_data["eps"]
        dcc = DCC(1, 1)
        result = dcc.fit_from_residuals(eps)
        s = result.summary()
        assert "DCC(1,1)" in s
        assert "Method: Gaussian correlation MLE" in s
        assert "AIC:" in s
        assert "BIC:" in s
        assert "a[1]" in s
        assert "b[1]" in s
        assert "Marginals stored: No" in s

    def test_persistence(self, dcc_dgp_data):
        eps = dcc_dgp_data["eps"]
        dcc = DCC(1, 1)
        result = dcc.fit_from_residuals(eps)
        assert 0 < result.params.persistence < 1


class TestMultivariateForecastAndSimulation:
    def test_ccc_summary(self, dcc_dgp_data):
        result = CCC().fit_from_residuals(dcc_dgp_data["eps"])
        s = result.summary()
        assert "CCC Estimation Results" in s
        assert "Method: Closed-form constant correlation" in s
        assert "AIC:" in s
        assert "BIC:" in s
        assert "Marginals stored: No" in s

    def test_ccc_constant_path_from_residuals(self, dcc_dgp_data):
        result = CCC().fit_from_residuals(dcc_dgp_data["eps"])
        expected = _normalize_corr(result.Qbar)
        np.testing.assert_allclose(result.params.corr, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.Rt, np.broadcast_to(expected, result.Rt.shape))
        np.testing.assert_allclose(result.corr(0, 1), np.full(result.T, expected[0, 1]))

    def test_dcc_forecast_reuses_marginal_forecasts(self, fitted_multivariate_results):
        result = fitted_multivariate_results["dcc"]
        forecast = result.forecast(horizon=5)

        assert forecast.mean.shape == (5, 3)
        assert forecast.variance.shape == (5, 3)
        assert forecast.residual_variance.shape == (5, 3)
        assert forecast.correlation.shape == (5, 3, 3)
        assert forecast.residual_covariance.shape == (5, 3, 3)

        for i, marginal in enumerate(result.univariate_results):
            marginal_fc = marginal.forecast(5)
            np.testing.assert_allclose(forecast.mean[:, i], marginal_fc.mean)
            np.testing.assert_allclose(forecast.variance[:, i], marginal_fc.variance)
            np.testing.assert_allclose(forecast.residual_variance[:, i], marginal_fc.residual_variance)

        _ = result.Rt
        q1 = (1.0 - np.sum(result.theta)) * result.Qbar
        q1 = q1 + result.theta[0] * np.outer(result._eps[-1], result._eps[-1]) + result.theta[1] * result._Qt_cache[-1]
        r1 = _normalize_corr(q1)
        np.testing.assert_allclose(forecast.correlation[0], r1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            forecast.residual_covariance[0],
            np.outer(np.sqrt(forecast.residual_variance[0]), np.sqrt(forecast.residual_variance[0])) * forecast.correlation[0],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_ccc_forecast_and_simulate_keep_constant_correlation(self, fitted_multivariate_results):
        result = fitted_multivariate_results["ccc"]
        forecast = result.forecast(horizon=4)
        sim = result.simulate(12, seed=123)

        for h in range(4):
            np.testing.assert_allclose(forecast.correlation[h], result.params.corr, rtol=1e-12, atol=1e-12)
        for t in range(12):
            np.testing.assert_allclose(sim.correlation[t], result.params.corr, rtol=1e-12, atol=1e-12)

    def test_dcc_simulation_uses_one_step_correlation_forecast(self, fitted_multivariate_results):
        result = fitted_multivariate_results["dcc"]
        forecast = result.forecast(horizon=1)
        sim = result.simulate(20, seed=321)

        assert sim.data.shape == (20, 3)
        assert sim.mean.shape == (20, 3)
        assert sim.residuals.shape == (20, 3)
        assert sim.sigma2.shape == (20, 3)
        assert sim.innovations.shape == (20, 3)
        assert sim.correlation.shape == (20, 3, 3)
        assert sim.residual_covariance.shape == (20, 3, 3)

        np.testing.assert_allclose(sim.correlation[0], forecast.correlation[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sim.data, sim.mean + sim.residuals, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            np.diagonal(sim.residual_covariance, axis1=1, axis2=2),
            sim.sigma2,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_forecast_and_simulate_require_stored_marginals(self, dcc_dgp_data):
        dcc_result = DCC(1, 1).fit_from_residuals(dcc_dgp_data["eps"])
        ccc_result = CCC().fit_from_residuals(dcc_dgp_data["eps"])

        with pytest.raises(RuntimeError, match="stored univariate marginal fits"):
            dcc_result.forecast(2)
        with pytest.raises(RuntimeError, match="stored univariate marginal fits"):
            dcc_result.simulate(5)
        with pytest.raises(RuntimeError, match="stored univariate marginal fits"):
            ccc_result.forecast(2)
        with pytest.raises(RuntimeError, match="stored univariate marginal fits"):
            ccc_result.simulate(5)

    def test_simulate_rejects_non_normal_marginals(self, fitted_multivariate_results, monkeypatch):
        result = fitted_multivariate_results["dcc"]
        original_state = result.univariate_results[0].filter()

        def fake_filter():
            return replace(original_state, distribution="StudentT", nu=8.0)

        monkeypatch.setattr(result.univariate_results[0], "filter", fake_filter)
        with pytest.raises(NotImplementedError, match="requires Normal marginals"):
            result.simulate(5, seed=7)


# =============================================================================
# 6. Input validation
# =============================================================================

class TestValidation:
    def test_dcc_requires_positive_pq(self):
        with pytest.raises(ValueError):
            DCC(0, 1)

    def test_fit_requires_2d(self):
        dcc = DCC(1, 1)
        with pytest.raises(ValueError):
            dcc.fit_from_residuals(np.zeros(100))


def _normalize_corr(qmat: NDArray[np.float64]) -> NDArray[np.float64]:
    diag = np.sqrt(np.maximum(np.diag(qmat), 1e-12))
    return qmat / np.outer(diag, diag)
