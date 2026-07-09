from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol import ARMA, GED, GJRGARCH
from scivol._devtools.ad_oracle import (
    arma_gjr_garch_logspace_value_grad_hess,
    arma_gjr_garch_value_grad_hess,
    gjr_garch_logspace_value_grad_hess,
    gjr_garch_value_grad_hess,
)
from scivol._kernels.transforms import (
    log_hessian_arma_gjr_garch_ged,
    log_hessian_gjr_garch,
    pack_arma_gjr_garch_ged,
    pack_gjr_garch_ged,
    unpack_arma_gjr_garch_ged,
    unpack_gjr_garch_ged,
)


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
        args = [
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(features),
            _as_cptr(resid_buf),
            _as_cptr(sigma2),
            y.size,
            0,
        ]
        value = float(_c._linear_mean_gjr_garch_nll_11_ged(*args))
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_nll_grad_11_ged(*args[:5], _as_cptr(grad), *args[5:])
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_hess_11_ged(*args[:5], _as_cptr(hess), *args[5:])
    else:
        args = [
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(features),
            _as_cptr(resid_buf),
            _as_cptr(sigma2),
            y.size,
            0,
            p,
            q,
        ]
        value = float(_c._linear_mean_gjr_garch_nll_pq_ged(*args))
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_nll_grad_pq_ged(*args[:5], _as_cptr(grad), *args[5:])
        sigma2[0] = np.mean(y * y)
        _c._linear_mean_gjr_garch_hess_pq_ged(*args[:5], _as_cptr(hess), *args[5:])

    return value, grad, hess


def test_gjr11_ged_theta_grad_hess_match_ad() -> None:
    resid = _make_gjr_data()
    theta = np.array([1e-6, 0.05, 0.03, 0.90, 1.6], dtype=np.float64)

    value_c, grad_c, hess_c = _call_standalone_gjr_ged(theta, resid, p=1, q=1)
    value_ad, grad_ad, hess_ad = gjr_garch_value_grad_hess(theta, resid, 1, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(grad_c, grad_ad, rtol=5e-5, atol=6e-6)
    np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-3, atol=2e-2)


def test_gjr_pq_ged_theta_grad_hess_match_ad() -> None:
    resid = _make_gjr_data(620, seed=303)
    theta = np.array([2e-6, 0.03, 0.02, 0.015, 0.01, 0.88, 1.7], dtype=np.float64)

    value_c, grad_c, hess_c = _call_standalone_gjr_ged(theta, resid, p=2, q=1)
    value_ad, grad_ad, hess_ad = gjr_garch_value_grad_hess(theta, resid, 2, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(grad_c, grad_ad, rtol=7e-5, atol=8e-6)
    np.testing.assert_allclose(hess_c, hess_ad, rtol=2e-3, atol=2e-2)


def test_gjr_pq_ged_logspace_grad_hess_match_ad() -> None:
    resid = _make_gjr_data(620, seed=304)
    theta = np.array([2e-6, 0.03, 0.02, 0.015, 0.01, 0.88, 1.7], dtype=np.float64)
    z = unpack_gjr_garch_ged(theta, 2, 1)
    theta_z = pack_gjr_garch_ged(z, 2, 1)
    value_c, grad_theta, hess_theta = _call_standalone_gjr_ged(theta_z, resid, p=2, q=1)
    hess_z = log_hessian_gjr_garch(theta_z, grad_theta, hess_theta, 2, 1, dist="ged")
    value_ad, grad_ad, hess_ad = gjr_garch_logspace_value_grad_hess(z, resid, 2, 1, dist="ged")
    from scivol._kernels.transforms import jacobian_gjr_garch_ged

    grad_z = jacobian_gjr_garch_ged(theta_z, 2, 1).T @ grad_theta

    np.testing.assert_allclose(value_c, value_ad, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=8e-5, atol=1e-5)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=3e-3, atol=3e-2)


def test_gjr_ged_public_fit_and_filter_surface() -> None:
    resid = _make_gjr_data(320, seed=99)
    spec = GJRGARCH(1, 1) + GED()

    result = spec.fit(resid, solver="slsqp", log_mode=True, verbose=False)
    state = spec.filter(resid, result.params)
    score = spec.score(resid, result.params)
    hessian = spec.hessian(resid, result.params)

    assert np.all(np.isfinite(result.params))
    np.testing.assert_allclose(result.log_likelihood, state.log_likelihood, rtol=1e-10, atol=1e-8)
    assert np.isfinite(state.log_likelihood)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))


def test_gjr_ged_fixed_hessian_is_finite_with_zero_returns_low_shape() -> None:
    y = _make_gjr_data(420, seed=808)
    y[::19] = 0.0
    theta = np.array([2.96e-5, 0.034, 0.118, 0.750, 1.34], dtype=np.float64)
    spec = GJRGARCH(1, 1) + GED()

    state = spec.filter(y, theta)
    score = spec.score(y, theta)
    hessian = spec.hessian(y, theta)
    value_c, grad_c, hess_c = _call_standalone_gjr_ged(theta, y, p=1, q=1)

    assert np.isfinite(state.log_likelihood)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))
    assert np.isfinite(value_c)
    assert np.all(np.isfinite(grad_c))
    assert np.all(np.isfinite(hess_c))


def test_arma_gjr11_ged_theta_grad_hess_match_ad() -> None:
    y = _make_gjr_data(260, seed=202)
    theta = np.array([0.01, 0.10, 0.05, 2e-6, 0.04, 0.02, 0.90, 1.6], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)
    h0 = float(np.mean(y * y))

    args = [_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(sigma2), h0, y.size]
    value_c = float(_c._arma_gjr_garch_nll_11_ged(*args))
    _c._arma_gjr_garch_nll_grad_11_ged(*args[:4], _as_cptr(grad), *args[4:])
    _c._arma_gjr_garch_hess_11_ged(*args[:4], _as_cptr(hess), *args[4:])

    value_ad, grad_ad, hess_ad = arma_gjr_garch_value_grad_hess(theta, y, 1, 1, 1, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=7e-5, atol=8e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=2e-3, atol=2e-2)


def test_arma_gjr_pq_ged_theta_grad_hess_match_ad() -> None:
    y = _make_gjr_data(300, seed=505)
    theta = np.array([0.01, 0.08, 0.03, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88, 1.7], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    e0 = np.zeros(1, dtype=np.float64)
    h0 = np.full(2, np.mean(y * y), dtype=np.float64)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)

    value_c = float(
        _c._arma_gjr_garch_nll_pq_ged(
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            y.size,
            1,
            1,
            2,
            1,
        )
    )
    _c._arma_gjr_garch_nll_grad_pq_ged(
        _as_cptr(theta),
        _as_cptr(y),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(grad),
        y.size,
        1,
        1,
        2,
        1,
    )
    _c._arma_gjr_garch_hess_pq_ged(
        _as_cptr(theta),
        _as_cptr(y),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(hess),
        y.size,
        1,
        1,
        2,
        1,
    )

    value_ad, grad_ad, hess_ad = arma_gjr_garch_value_grad_hess(theta, y, 1, 1, 2, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(hess, hess_ad, rtol=3e-3, atol=3e-2)


def test_arma_gjr_pq_ged_logspace_grad_hess_match_ad() -> None:
    y = _make_gjr_data(300, seed=506)
    theta = np.array([0.01, 0.08, 0.03, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88, 1.7], dtype=np.float64)
    z = unpack_arma_gjr_garch_ged(theta, 1, 1, 2, 1)
    theta_z = pack_arma_gjr_garch_ged(z, 1, 1, 2, 1)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    e0 = np.zeros(1, dtype=np.float64)
    h0 = np.full(2, np.mean(y * y), dtype=np.float64)
    grad_theta = np.zeros_like(theta_z)
    hess_theta = np.zeros((theta_z.size, theta_z.size), dtype=np.float64)

    value_c = float(
        _c._arma_gjr_garch_nll_pq_ged(
            _as_cptr(theta_z),
            _as_cptr(y),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            y.size,
            1,
            1,
            2,
            1,
        )
    )
    _c._arma_gjr_garch_nll_grad_pq_ged(
        _as_cptr(theta_z),
        _as_cptr(y),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(grad_theta),
        y.size,
        1,
        1,
        2,
        1,
    )
    _c._arma_gjr_garch_hess_pq_ged(
        _as_cptr(theta_z),
        _as_cptr(y),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(hess_theta),
        y.size,
        1,
        1,
        2,
        1,
    )
    from scivol._kernels.transforms import jacobian_arma_gjr_garch_ged

    grad_z = jacobian_arma_gjr_garch_ged(theta_z, 1, 1, 2, 1).T @ grad_theta
    hess_z = log_hessian_arma_gjr_garch_ged(theta_z, grad_theta, hess_theta, 1, 1, 2, 1)
    value_ad, grad_ad, hess_ad = arma_gjr_garch_logspace_value_grad_hess(z, y, 1, 1, 2, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=4e-3, atol=4e-2)


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
