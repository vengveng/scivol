from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from volkit import ARMA, GARCH, GJRGARCH, Normal, SkewT, StudentT, _core
from volkit._devtools.ad_oracle import (
    arma_garch_logspace_value_grad_hess,
    arma_garch_value_grad_hess,
    arma_normal_logspace_value_grad_hess,
    arma_normal_value_grad_hess,
    garch_logspace_value_grad_hess,
    garch_value_grad_hess,
    gjr_garch_logspace_value_grad_hess,
    gjr_garch_value_grad_hess,
)
from volkit._kernels.transforms import (
    jacobian_arma_garch_normal,
    jacobian_arma_garch_skewt,
    jacobian_arma_garch_studentt,
    log_hessian_arma_garch_normal,
    log_hessian_arma_garch_skewt,
    log_hessian_arma_garch_studentt,
    log_hessian_arma_normal,
    log_hessian_garch,
    log_hessian_gjr_garch,
    unpack_arma_normal,
    unpack_arma_garch_normal,
    unpack_arma_garch_skewt,
    unpack_arma_garch_studentt,
    unpack_garch,
    unpack_garch_skewt,
    unpack_garch_studentt,
    unpack_gjr_garch,
    unpack_gjr_garch_skewt,
    unpack_gjr_garch_studentt,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle model checks", allow_module_level=True)


def _as_cptr(arr: np.ndarray) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


@pytest.fixture(scope="module")
def arma_series() -> np.ndarray:
    rng = np.random.default_rng(2026)
    y = np.zeros(320, dtype=np.float64)
    c, phi1, phi2, theta1 = 0.01, 0.25, -0.10, 0.20
    eps_prev = 0.0
    for t in range(2, len(y)):
        shock = rng.normal(scale=0.1)
        mu_t = c + phi1 * y[t - 1] + phi2 * y[t - 2] + theta1 * eps_prev
        y[t] = mu_t + shock
        eps_prev = shock
    return y


@pytest.fixture(scope="module")
def arma_garch_series() -> np.ndarray:
    rng = np.random.default_rng(2027)
    y = np.zeros(360, dtype=np.float64)
    sigma2 = np.zeros_like(y)
    eps = np.zeros_like(y)
    c, phi, theta = 0.0, 0.20, -0.15
    omega, alpha, beta = 2e-6, 0.06, 0.92
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, len(y)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]
    return y


@pytest.fixture(scope="module")
def garch_skewt_series() -> np.ndarray:
    rng = np.random.default_rng(2028)
    y = np.zeros(420, dtype=np.float64)
    sigma2 = np.zeros_like(y)
    omega, alpha, beta = 1.5e-6, 0.05, 0.92
    nu, lam = 8.0, -0.25
    sigma2[0] = omega / (1.0 - alpha - beta)
    t_scale = np.sqrt((nu - 2.0) / nu)

    for t in range(len(y)):
        if t > 0:
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        z_raw = rng.standard_t(nu) * t_scale
        z = z_raw * (1.0 - lam) if z_raw < 0.0 else z_raw * (1.0 + lam)
        y[t] = np.sqrt(sigma2[t]) * z

    return y


class TestARMAOracle:
    def test_arma11_normal_matches_c_nll_and_grad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_value_grad_hess(params, arma_series, 1, 1)
        value_c = _core._arma_nll_11_normal(_as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), arma_series.size)
        _core._arma_nll_grad_11_normal(
            _as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(grad), arma_series.size
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-8, atol=1e-10)
        assert np.all(np.isfinite(hess_ad))

    def test_arma22_normal_matches_pq_nll_and_grad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_value_grad_hess(params, arma_series, 2, 2)
        value_c = _core._arma_nll_pq_normal(
            _as_cptr(params), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(e0), arma_series.size, 2, 2
        )
        _core._arma_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad),
            arma_series.size,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-8, atol=1e-10)
        assert np.all(np.isfinite(hess_ad))

    def test_arma22_normal_theta_hessian_matches_ad(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(hess_c),
            arma_series.size,
            2,
            2,
        )
        _, _, hess_ad = arma_normal_value_grad_hess(params, arma_series, 2, 2)

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-7, atol=1e-8)

    def test_arma22_normal_log_matches_c_grad_and_hess(self, arma_series: np.ndarray) -> None:
        params = np.array([0.01, 0.25, -0.10, 0.20, -0.05], dtype=np.float64)
        z = unpack_arma_normal(params, 2, 2)
        resid = np.zeros_like(arma_series)
        e0 = np.zeros(2, dtype=np.float64)
        grad_z = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_normal_logspace_value_grad_hess(z, arma_series, 2, 2)
        value_c = _core._log_arma_nll_pq_normal(
            _as_cptr(z), _as_cptr(arma_series), _as_cptr(resid), _as_cptr(e0), arma_series.size, 2, 2
        )
        _core._log_arma_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad_z),
            arma_series.size,
            2,
            2,
        )
        grad_theta = np.zeros_like(params)
        _core._arma_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(grad_theta),
            arma_series.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_series),
            _as_cptr(resid),
            _as_cptr(e0),
            _as_cptr(hess_theta),
            arma_series.size,
            2,
            2,
        )
        hess_z = log_hessian_arma_normal(params, grad_theta, hess_theta, 2, 2)

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-7, atol=1e-8)

    def test_arma22_normal_fit_runs_in_log_mode(self, arma_series: np.ndarray) -> None:
        spec = ARMA(2, 2) + Normal()
        result = spec.fit(arma_series, solver="slsqp", log_mode=True, verbose=False)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))

    def test_arma22_normal_fit_runs_in_log_mode_with_trust(self, arma_series: np.ndarray) -> None:
        spec = ARMA(2, 2) + Normal()
        result = spec.fit(arma_series, solver="trust", log_mode=True, verbose=False)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))


class TestARMAGARCHOracle:
    def test_arma_garch11_normal_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "normal")
        value_c = _core._arma_garch_nll_11_normal(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        _core._arma_garch_nll_grad_11_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "normal")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch11_studentt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "studentt")
        value_c = _core._arma_garch_nll_11_studentt(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        _core._arma_garch_nll_grad_11_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "studentt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_normal_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")
        value_c = _core._arma_garch_nll_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")
        value_c = _core._arma_garch_nll_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-6, atol=1e-8)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch11_skewt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        grad = np.zeros_like(params)
        h0 = float(np.mean(arma_garch_series ** 2))

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "skewt")
        value_c = _core._arma_garch_nll_11_skewt(
            _as_cptr(params), _as_cptr(arma_garch_series), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_series.size
        )
        value_grad_c = _core._arma_garch_nll_grad_11_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            h0,
            arma_garch_series.size,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-5, atol=1e-7)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch11_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.20, -0.15, 2e-6, 0.06, 0.92, 8.0, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        hess = np.zeros((params.size, params.size), dtype=np.float64)
        h0 = float(np.mean(arma_garch_series ** 2))

        _core._arma_garch_hess_11_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            h0,
            arma_garch_series.size,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 1, 1, "skewt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_skewt_matches_c_nll_and_grad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad = np.zeros_like(params)

        value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")
        value_c = _core._arma_garch_nll_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        value_grad_c = _core._arma_garch_nll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(grad, grad_ad, rtol=1e-5, atol=1e-7)
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        hess = np.zeros((params.size, params.size), dtype=np.float64)

        _core._arma_garch_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        _, _, hess_ad = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")

        np.testing.assert_allclose(hess, hess_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_normal_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_normal(params, 1, 1, 2, 2)

        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "normal")
        grad_z_ref = jacobian_arma_garch_normal(params, 1, 1, 2, 2).T @ grad_theta

        _core._log_arma_garch_nll_grad_pq_normal(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_normal(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "normal")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_normal(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_studentt(params, 1, 1, 2, 2)

        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "studentt")
        grad_z_ref = jacobian_arma_garch_studentt(params, 1, 1, 2, 2).T @ grad_theta

        _core._log_arma_garch_nll_grad_pq_studentt(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_studentt(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_studentt(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-8)

    def test_arma_garch_pq_skewt_log_gradient_matches_chain_rule(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        grad_z = np.zeros_like(params)
        z = unpack_arma_garch_skewt(params, 1, 1, 2, 2)
        _, grad_theta, _ = arma_garch_value_grad_hess(params, arma_garch_series, 1, 1, 2, 2, "skewt")
        grad_z_ref = jacobian_arma_garch_skewt(params, 1, 1, 2, 2).T @ grad_theta
        _, grad_z_ad, _ = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "skewt")
        _core._log_arma_garch_nll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_z),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )

        np.testing.assert_allclose(grad_z, grad_z_ref, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(grad_z_ref, grad_z_ad, rtol=1e-5, atol=1e-7)

    def test_arma_garch_pq_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5, -0.2], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)
        z = unpack_arma_garch_skewt(params, 1, 1, 2, 2)

        _, _, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_series, 1, 1, 2, 2, "skewt")
        grad_theta = np.zeros_like(params)
        _core._arma_garch_nll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(grad_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._arma_garch_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(arma_garch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            _as_cptr(hess_theta),
            arma_garch_series.size,
            1,
            1,
            2,
            2,
        )
        hess_z = log_hessian_arma_garch_skewt(params, grad_theta, hess_theta, 1, 1, 2, 2)

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-7)

    def test_generic_arma_garch_normal_fit_runs_in_log_mode(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, Normal

        spec = ARMA(1, 1) + GARCH(2, 2) + Normal()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, Normal

        spec = ARMA(1, 1) + GARCH(2, 2) + Normal()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_studentt_fit_runs_in_constrained_mode(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, StudentT

        spec = ARMA(1, 1) + GARCH(2, 2) + StudentT()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=False)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, StudentT

        spec = ARMA(1, 1) + GARCH(2, 2) + StudentT()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_skewt_fit_runs_in_log_mode(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, SkewT

        spec = ARMA(1, 1) + GARCH(2, 2) + SkewT()
        result = spec.fit(arma_garch_series, solver="slsqp", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 10
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_generic_arma_garch_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        from volkit import ARMA, GARCH, SkewT

        spec = ARMA(1, 1) + GARCH(2, 2) + SkewT()
        result = spec.fit(arma_garch_series, solver="trust", log_mode=True)

        assert result.params is not None
        assert len(result.params) == 10
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


class TestLogspaceHessianOracle:
    def test_garch22_normal_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid2.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "normal")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-7, atol=1e-8)

    def test_garch22_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid2.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "studentt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_variance_recursion_matches_manual(self, arma_garch_series: np.ndarray) -> None:
        resid2 = arma_garch_series * arma_garch_series
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        sigma2_c = np.zeros_like(resid2)
        sigma2_c[0] = np.mean(resid2)
        sigma2_py = np.zeros_like(resid2)
        sigma2_py[0] = np.mean(resid2)

        _core._garch_variance_pq(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2_c),
            resid2.size,
            2,
            2,
        )

        for t in range(1, resid2.size):
            sigma2_py[t] = (
                params[0]
                + params[1] * resid2[t - 1]
                + (params[2] * resid2[t - 2] if t >= 2 else 0.0)
                + params[3] * sigma2_py[t - 1]
                + (params[4] * sigma2_py[t - 2] if t >= 2 else 0.0)
            )

        np.testing.assert_allclose(sigma2_c, sigma2_py, rtol=1e-12, atol=1e-12)

    def test_garch22_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15], dtype=np.float64)
        z = unpack_garch(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "normal")
        _, grad_theta, _ = garch_value_grad_hess(params, resid2, 2, 2, "normal")
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid2.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5], dtype=np.float64)
        z = unpack_garch_studentt(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        _core._garch_ll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid2.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid2.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="studentt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = garch_value_grad_hess(params, resid2, 2, 2, "skewt", resid=resid)

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-5, atol=1e-6)

    def test_garch22_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        z = unpack_garch_skewt(params, 2, 2)

        _, _, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "skewt", resid=resid)
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        _core._garch_ll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="skewt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-6)

    def test_garch22_skewt_fused_log_gradient_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        resid2 = resid * resid
        params = np.array([2e-6, 0.05, 0.03, 0.70, 0.15, 8.5, -0.2], dtype=np.float64)
        z = unpack_garch_skewt(params, 2, 2)
        sigma2 = np.zeros_like(resid2)
        sigma2[0] = np.mean(resid2)
        grad_z = np.zeros_like(z)

        value_c = _core._log_garch_ll_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            resid.size,
            2,
            2,
        )
        _core._log_garch_ll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_z),
            resid.size,
            2,
            2,
        )
        value_ad, grad_ad, _ = garch_logspace_value_grad_hess(z, resid2, 2, 2, "skewt", resid=resid)

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_normal_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10], dtype=np.float64)
        z = unpack_gjr_garch(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "normal")
        _, grad_theta, _ = gjr_garch_value_grad_hess(params, resid, 2, 2, "normal")
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_normal(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="normal")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_garch22_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + Normal()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))

    def test_garch22_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + SkewT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))

    def test_gjr22_normal_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + Normal()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))

    def test_garch22_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GARCH(2, 2) + StudentT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 6
        assert np.all(np.isfinite(result.params))

    def test_gjr22_studentt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5], dtype=np.float64)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._gjr_garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = gjr_garch_value_grad_hess(params, resid, 2, 2, "studentt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-6, atol=1e-6)

    def test_gjr22_studentt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5], dtype=np.float64)
        z = unpack_gjr_garch_studentt(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "studentt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        _core._gjr_garch_ll_grad_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_studentt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="studentt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-6, atol=1e-6)

    def test_gjr22_studentt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + StudentT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))

    def test_gjr22_skewt_theta_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        hess_c = np.zeros((params.size, params.size), dtype=np.float64)

        _core._gjr_garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_c),
            resid.size,
            2,
            2,
        )
        _, _, hess_ad = gjr_garch_value_grad_hess(params, resid, 2, 2, "skewt")

        np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_log_hessian_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        z = unpack_gjr_garch_skewt(params, 2, 2)

        _, _, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "skewt")
        grad_theta = np.zeros_like(params)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        _core._gjr_garch_ll_grad_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_theta),
            resid.size,
            2,
            2,
        )
        hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
        _core._gjr_garch_ll_hess_pq_skewt(
            _as_cptr(params),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess_theta),
            resid.size,
            2,
            2,
        )
        hess_z = log_hessian_gjr_garch(params, grad_theta, hess_theta, 2, 2, dist="skewt")

        np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_fused_log_gradient_matches_ad(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        params = np.array([2e-6, 0.04, 0.02, 0.03, 0.01, 0.70, 0.10, 8.5, -0.2], dtype=np.float64)
        z = unpack_gjr_garch_skewt(params, 2, 2)
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid * resid)
        grad_z = np.zeros_like(z)

        value_c = _core._log_gjr_garch_ll_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            resid.size,
            2,
            2,
        )
        _core._log_gjr_garch_ll_grad_pq_skewt(
            _as_cptr(z),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad_z),
            resid.size,
            2,
            2,
        )
        value_ad, grad_ad, _ = gjr_garch_logspace_value_grad_hess(z, resid, 2, 2, "skewt")

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-5, atol=1e-6)

    def test_gjr22_skewt_fit_runs_in_log_mode_with_trust(self, arma_garch_series: np.ndarray) -> None:
        resid = arma_garch_series
        spec = GJRGARCH(2, 2) + SkewT()
        result = spec.fit(resid, solver="trust", log_mode=True, verbose=False)

        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))


class TestGARCHSkewTOracle:
    @pytest.mark.parametrize(
        "params",
        [
            np.array([1.5e-6, 0.05, 0.92, 8.0, -0.25], dtype=np.float64),
            np.array([2.0e-6, 0.07, 0.88, 12.0, 0.20], dtype=np.float64),
        ],
    )
    def test_garch11_skewt_matches_c_objective_and_gradient(
        self,
        garch_skewt_series: np.ndarray,
        params: np.ndarray,
    ) -> None:
        resid2 = garch_skewt_series * garch_skewt_series
        sigma2 = np.zeros_like(garch_skewt_series)
        sigma2[0] = np.mean(resid2)
        grad = np.zeros_like(params)

        _core._garch_variance_11(_as_cptr(params[:3]), _as_cptr(resid2), _as_cptr(sigma2), garch_skewt_series.size)
        value_c = -_core._skewt_ll(
            _as_cptr(garch_skewt_series),
            _as_cptr(sigma2),
            garch_skewt_series.size,
            float(params[3]),
            float(params[4]),
        )
        value_grad_c = _core._garch_ll_grad_11_skewt(
            _as_cptr(params), _as_cptr(garch_skewt_series), _as_cptr(grad), garch_skewt_series.size
        )
        value_ad, grad_ad, hess_ad = garch_value_grad_hess(
            params, resid2, 1, 1, dist="skewt", resid=garch_skewt_series
        )

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(value_grad_c, value_ad, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(grad, grad_ad, rtol=2e-6, atol=1e-4)
        assert np.all(np.isfinite(hess_ad))
