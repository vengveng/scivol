from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from volkit import _core
from volkit._devtools.ad_oracle import (
    arma_garch_value_grad_hess,
    arma_normal_value_grad_hess,
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

    def test_arma_garch_pq_normal_matches_c_nll(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)

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

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(grad_ad))
        assert np.all(np.isfinite(hess_ad))

    def test_arma_garch_pq_studentt_matches_c_nll(self, arma_garch_series: np.ndarray) -> None:
        params = np.array([0.0, 0.15, -0.10, 2e-6, 0.04, 0.02, 0.70, 0.20, 8.5], dtype=np.float64)
        resid = np.zeros_like(arma_garch_series)
        sigma2 = np.zeros_like(arma_garch_series)
        e0 = np.zeros(2, dtype=np.float64)
        h0 = np.full(2, np.mean(arma_garch_series ** 2), dtype=np.float64)

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

        np.testing.assert_allclose(value_c, value_ad, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(grad_ad))
        assert np.all(np.isfinite(hess_ad))
