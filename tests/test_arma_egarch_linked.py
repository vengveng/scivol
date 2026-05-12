from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scivol import ARMA, EGARCH, Normal, SkewT, StudentT, _core
from scivol._devtools.ad_oracle import (
    arma_egarch_logspace_value_grad_hess,
    arma_egarch_value_grad_hess,
)
from scivol._kernels.transforms import (
    jacobian_arma_egarch_normal,
    jacobian_arma_egarch_studentt,
    log_hessian_arma_egarch_normal,
    log_hessian_arma_egarch_studentt,
    unpack_arma_egarch_normal,
    unpack_arma_egarch_studentt,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle model checks", allow_module_level=True)


def _as_cptr(arr: np.ndarray) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _studentt_abs_moment(nu: float) -> float:
    from scipy.special import gammaln

    return (
        2.0
        * np.exp(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu))
        * (nu - 2.0)
        / ((nu - 1.0) * np.sqrt(np.pi * (nu - 2.0)))
    )


def _simulate_arma_egarch(
    n: int,
    c: float,
    phi: list[float],
    theta_ma: list[float],
    omega: float,
    alpha: list[float],
    gamma: list[float],
    beta: list[float],
    *,
    dist: str = "normal",
    nu: float = 8.0,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_ar = len(phi)
    q_ma = len(theta_ma)
    p_arch = len(alpha)
    q_egarch = len(beta)
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)

    y = np.zeros(n, dtype=np.float64)
    resid = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    persistence = float(sum(beta))
    x0 = omega / max(1.0 - persistence, 1e-3)
    sigma2[:max_lag] = np.exp(x0)
    x[:max_lag] = x0

    if dist == "normal":
        abs_moment = np.sqrt(2.0 / np.pi)
        draws = rng.standard_normal(n)
    elif dist == "studentt":
        abs_moment = _studentt_abs_moment(nu)
        draws = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    else:
        raise ValueError(dist)

    for t in range(max_lag):
        resid[t] = np.sqrt(sigma2[t]) * draws[t]
        y[t] = resid[t]

    for t in range(max_lag, n):
        mu_t = c
        for i, phi_i in enumerate(phi, start=1):
            mu_t += phi_i * y[t - i]
        for j, theta_j in enumerate(theta_ma, start=1):
            mu_t += theta_j * resid[t - j]

        x_t = omega
        for i, alpha_i in enumerate(alpha, start=1):
            z_lag = resid[t - i] / np.sqrt(sigma2[t - i])
            x_t += alpha_i * (abs(z_lag) - abs_moment) + gamma[i - 1] * z_lag
        for j, beta_j in enumerate(beta, start=1):
            x_t += beta_j * x[t - j]

        x[t] = x_t
        sigma2[t] = np.exp(x_t)
        resid[t] = np.sqrt(sigma2[t]) * draws[t]
        y[t] = mu_t + resid[t]

    return y


@pytest.fixture(scope="module")
def arma_egarch_series() -> np.ndarray:
    return _simulate_arma_egarch(
        420,
        0.001,
        [0.20],
        [-0.12],
        -0.18,
        [0.10],
        [-0.05],
        [0.92],
        dist="normal",
        seed=20260611,
    )


class TestARMAEGARCHThetaOracle:
    def test_normal_11_matches_ad(self, arma_egarch_series: np.ndarray) -> None:
        params = np.array([0.001, 0.20, -0.12, -0.18, 0.10, -0.05, 0.92], dtype=np.float64)
        resid = np.zeros_like(arma_egarch_series)
        sigma2 = np.zeros_like(arma_egarch_series)
        grad = np.zeros_like(params)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_egarch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_egarch_value_grad_hess(params, arma_egarch_series, 1, 1, 1, 1, "normal")
        value_c = _core._arma_egarch_nll_11_normal(
            _as_cptr(params), _as_cptr(arma_egarch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_egarch_series.size
        )
        value_grad_c = _core._arma_egarch_nll_grad_11_normal(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_egarch_series.size,
        )
        _core._arma_egarch_hess_11_normal(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_egarch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad, grad_ad, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(hess, hess_ad, rtol=5e-4, atol=5e-5)

    def test_studentt_pq_matches_ad(self, arma_egarch_series: np.ndarray) -> None:
        params = np.array([0.001, 0.20, -0.12, -0.22, 0.07, 0.03, -0.05, 0.01, 0.90, 8.0], dtype=np.float64)
        resid = np.zeros_like(arma_egarch_series)
        sigma2 = np.zeros_like(arma_egarch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_egarch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        value_ad, grad_ad, hess_ad = arma_egarch_value_grad_hess(params, arma_egarch_series, 1, 1, 2, 1, "studentt")
        value_c = _core._arma_egarch_nll_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        value_grad_c = _core._arma_egarch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        _core._arma_egarch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=8e-4, atol=3e-5)
        np.testing.assert_allclose(hess[:9, :9], hess_ad[:9, :9], rtol=2e-3, atol=2e-4)


class TestARMAEGARCHLogOracle:
    def test_normal_log_gradient_and_hessian_match_ad(self, arma_egarch_series: np.ndarray) -> None:
        params = np.array([0.001, 0.20, -0.12, -0.22, 0.07, 0.03, -0.05, 0.01, 0.90], dtype=np.float64)
        z = unpack_arma_egarch_normal(params, 1, 1, 2, 1)
        resid = np.zeros_like(arma_egarch_series)
        sigma2 = np.zeros_like(arma_egarch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_egarch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(z)
        grad_theta = np.zeros_like(params)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)

        _, grad_z_ad, hess_z_ad = arma_egarch_logspace_value_grad_hess(z, arma_egarch_series, 1, 1, 2, 1, "normal")
        _core._log_arma_egarch_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        _core._arma_egarch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        _core._arma_egarch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        hess_z = log_hessian_arma_egarch_normal(params, grad_theta, hess_theta, 1, 1, 2, 1)

        np.testing.assert_allclose(grad_z, jacobian_arma_egarch_normal(params, 1, 1, 2, 1).T @ grad_theta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_z, grad_z_ad, rtol=5e-5, atol=2e-6)
        np.testing.assert_allclose(hess_z, hess_z_ad, rtol=2e-3, atol=2e-4)

    def test_studentt_log_gradient_and_hessian_match_ad(self, arma_egarch_series: np.ndarray) -> None:
        params = np.array([0.001, 0.20, -0.12, -0.22, 0.07, 0.03, -0.05, 0.01, 0.90, 8.0], dtype=np.float64)
        z = unpack_arma_egarch_studentt(params, 1, 1, 2, 1)
        resid = np.zeros_like(arma_egarch_series)
        sigma2 = np.zeros_like(arma_egarch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_egarch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(z)
        grad_theta = np.zeros_like(params)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)

        _, grad_z_ad, hess_z_ad = arma_egarch_logspace_value_grad_hess(z, arma_egarch_series, 1, 1, 2, 1, "studentt")
        _core._log_arma_egarch_nll_grad_pq_studentt(
            _as_cptr(z),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        _core._arma_egarch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        _core._arma_egarch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_egarch_series.size,
            1,
            1,
            2,
            1,
        )
        hess_z = log_hessian_arma_egarch_studentt(params, grad_theta, hess_theta, 1, 1, 2, 1)

        np.testing.assert_allclose(grad_z, jacobian_arma_egarch_studentt(params, 1, 1, 2, 1).T @ grad_theta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_z, grad_z_ad, rtol=8e-4, atol=3e-5)
        np.testing.assert_allclose(hess_z[:9, :9], hess_z_ad[:9, :9], rtol=3e-3, atol=3e-4)


class TestARMAEGARCHPublicSurface:
    def test_fit_and_fixed_workflows_run_for_normal(self, arma_egarch_series: np.ndarray) -> None:
        spec = ARMA(1, 1) + EGARCH(1, 1) + Normal()
        result = spec.fit(arma_egarch_series, solver="slsqp", log_mode=True)
        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(spec.loglikelihood(arma_egarch_series, result.params))
        assert np.all(np.isfinite(spec.score(arma_egarch_series, result.params)))
        assert np.all(np.isfinite(spec.hessian(arma_egarch_series, result.params)))
        filtered = spec.filter(arma_egarch_series, result.params)
        assert np.all(np.isfinite(filtered.sigma2))

    def test_fit_runs_for_generic_studentt(self, arma_egarch_series: np.ndarray) -> None:
        spec = ARMA(1, 1) + EGARCH(2, 1) + StudentT()
        result = spec.fit(arma_egarch_series, solver="slsqp", log_mode=True)
        assert result.params is not None
        assert len(result.params) == 10
        assert np.all(np.isfinite(result.params))
