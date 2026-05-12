from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol import ARMA, GED, GJRGARCH
from scivol._devtools.ad_oracle import arma_gjr_garch_value_grad_hess, gjr_garch_value_grad_hess
from scivol._kernels.transforms import unpack_arma_gjr_garch_ged, unpack_gjr_garch_ged


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _make_gjr_data(n: int = 700, seed: int = 1234) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta = 1e-6, 0.05, 0.03, 0.90
    sigma2 = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    sigma2[0] = omega / max(1.0 - alpha - gamma - beta, 1e-3)
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        e = y[t - 1]
        sigma2[t] = omega + alpha * e * e + gamma * (1.0 if e < 0.0 else 0.0) * e * e + beta * sigma2[t - 1]
        y[t] = np.sqrt(max(sigma2[t], 1e-12)) * rng.standard_normal()
    return np.ascontiguousarray(y, dtype=np.float64)


def _call_standalone_gjr_ged(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    *,
    p: int,
    q: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    y = np.ascontiguousarray(resid, dtype=np.float64)
    features = np.empty((y.size, 0), dtype=np.float64)
    resid_buf = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)

    if p == 1 and q == 1:
        value = float(_c._linear_mean_gjr_garch_nll_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), y.size, 0))
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_nll_grad_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), _as_cptr(grad), y.size, 0)
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_hess_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), _as_cptr(hess), y.size, 0)
    else:
        value = float(_c._linear_mean_gjr_garch_nll_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), y.size, 0, p, q))
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_nll_grad_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), _as_cptr(grad), y.size, 0, p, q)
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_hess_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid_buf), _as_cptr(sigma2), _as_cptr(hess), y.size, 0, p, q)

    return value, grad, hess


def test_gjr11_ged_theta_grad_hess_match_ad() -> None:
    resid = _make_gjr_data()
    theta = np.array([1e-6, 0.05, 0.03, 0.90, 1.6], dtype=np.float64)

    value_c, grad_c, hess_c = _call_standalone_gjr_ged(theta, resid, p=1, q=1)
    value_ad, grad_ad, hess_ad = gjr_garch_value_grad_hess(theta, resid, 1, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(grad_c, grad_ad, rtol=5e-5, atol=6e-6)
    np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-3, atol=2e-2)


def test_gjr_pq_ged_logspace_roundtrip_is_finite() -> None:
    theta = np.array([2e-6, 0.03, 0.02, 0.015, 0.01, 0.88, 1.7], dtype=np.float64)
    z = unpack_gjr_garch_ged(theta, 2, 1)
    assert np.all(np.isfinite(z))


def test_gjr_ged_public_fit_and_filter_surface() -> None:
    resid = _make_gjr_data(320, seed=99)
    spec = GJRGARCH(1, 1) + GED()

    result = spec.fit(resid, solver="slsqp", log_mode=True, verbose=False)
    state = spec.filter(resid, result.params)
    score = spec.score(resid, result.params)
    hessian = spec.hessian(resid, result.params)

    assert np.all(np.isfinite(result.params))
    assert np.isfinite(state.log_likelihood)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))


def test_arma_gjr11_ged_theta_grad_hess_match_ad() -> None:
    y = _make_gjr_data(260, seed=202)
    theta = np.array([0.01, 0.10, 0.05, 2e-6, 0.04, 0.02, 0.90, 1.6], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)
    h0 = float(np.mean(y * y))

    value_c = float(_c._arma_gjr_garch_nll_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(sigma2), h0, y.size))
    _c._arma_gjr_garch_nll_grad_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), h0, y.size)
    _c._arma_gjr_garch_hess_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess), h0, y.size)

    value_ad, grad_ad, hess_ad = arma_gjr_garch_value_grad_hess(theta, y, 1, 1, 1, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=7e-5, atol=8e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=2e-3, atol=2e-2)


def test_arma_gjr_pq_ged_logspace_roundtrip_is_finite() -> None:
    theta = np.array([0.01, 0.08, 0.03, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88, 1.7], dtype=np.float64)
    z = unpack_arma_gjr_garch_ged(theta, 1, 1, 2, 1)
    assert np.all(np.isfinite(z))


def test_arma_gjr_ged_public_fit_and_filter_surface() -> None:
    y = _make_gjr_data(320, seed=404)
    spec = ARMA(1, 1) + GJRGARCH(1, 1) + GED()

    result = spec.fit(y, solver="slsqp", log_mode=True, verbose=False)
    state = spec.filter(y, result.params)
    score = spec.score(y, result.params)
    hessian = spec.hessian(y, result.params)

    assert np.all(np.isfinite(result.params))
    assert np.isfinite(state.log_likelihood)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))
