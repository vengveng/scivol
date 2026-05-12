from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import gammaln
from scipy.stats import gennorm

import scivol._core as _c
from scivol import ARMA, GARCH, GED
from scivol._devtools.ad_oracle import (
    arma_garch_logspace_value_grad_hess,
    arma_garch_value_grad_hess,
    arma_ged_logspace_value_grad_hess,
    arma_ged_value_grad_hess,
)
from scivol._kernels.transforms import (
    log_hessian_arma_garch_ged,
    log_hessian_arma_ged,
    unpack_arma_garch_ged,
    unpack_arma_ged,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _ged_scale(nu: float) -> float:
    return float(np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu))))


def _ged_draws(rng: np.random.Generator, n: int, nu: float) -> NDArray[np.float64]:
    return np.asarray(gennorm.rvs(beta=nu, scale=_ged_scale(nu), size=n, random_state=rng), dtype=np.float64)


def _simulate_arma_ged(n: int, *, seed: int, c: float, phi: list[float], theta: list[float], sigma2: float, nu: float) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    burn = 250
    total = n + burn
    y = np.zeros(total, dtype=np.float64)
    eps = np.zeros(total, dtype=np.float64)
    z = _ged_draws(rng, total, nu)
    p = len(phi)
    q = len(theta)

    for t in range(total):
        eps[t] = np.sqrt(sigma2) * z[t]
        mu_t = c
        for i in range(p):
            if t > i:
                mu_t += phi[i] * y[t - 1 - i]
        for j in range(q):
            if t > j:
                mu_t += theta[j] * eps[t - 1 - j]
        y[t] = mu_t + eps[t]
    return np.ascontiguousarray(y[burn:], dtype=np.float64)


def _simulate_arma_garch_ged(
    n: int,
    *,
    seed: int,
    c: float,
    phi: list[float],
    theta: list[float],
    omega: float,
    alpha: list[float],
    beta: list[float],
    nu: float,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    y = np.zeros(total, dtype=np.float64)
    eps = np.zeros(total, dtype=np.float64)
    sigma2 = np.zeros(total, dtype=np.float64)
    z = _ged_draws(rng, total, nu)
    p_ar = len(phi)
    q_ma = len(theta)
    p_arch = len(alpha)
    q_garch = len(beta)
    persistence = float(sum(alpha) + sum(beta))
    sigma2[0] = omega / max(1.0 - persistence, 1e-3)

    for t in range(total):
        if t > 0:
            value = omega
            for i in range(p_arch):
                if t > i:
                    value += alpha[i] * eps[t - 1 - i] ** 2
            for j in range(q_garch):
                if t > j:
                    value += beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(value, 1e-12)
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        mu_t = c
        for i in range(p_ar):
            if t > i:
                mu_t += phi[i] * y[t - 1 - i]
        for j in range(q_ma):
            if t > j:
                mu_t += theta[j] * eps[t - 1 - j]
        y[t] = mu_t + eps[t]
    return np.ascontiguousarray(y[burn:], dtype=np.float64)


@pytest.fixture(scope="module")
def arma_ged_series_11() -> NDArray[np.float64]:
    return _simulate_arma_ged(320, seed=20260511, c=0.01, phi=[0.22], theta=[-0.14], sigma2=0.75, nu=1.7)


@pytest.fixture(scope="module")
def arma_ged_series_pq() -> NDArray[np.float64]:
    return _simulate_arma_ged(340, seed=20260512, c=0.02, phi=[0.25, -0.08], theta=[0.12], sigma2=0.90, nu=1.8)


@pytest.fixture(scope="module")
def arma_garch_ged_series_11() -> NDArray[np.float64]:
    return _simulate_arma_garch_ged(
        340,
        seed=20260513,
        c=0.0,
        phi=[0.18],
        theta=[-0.10],
        omega=2e-6,
        alpha=[0.05],
        beta=[0.92],
        nu=1.7,
    )


@pytest.fixture(scope="module")
def arma_garch_ged_series_pq() -> NDArray[np.float64]:
    return _simulate_arma_garch_ged(
        360,
        seed=20260514,
        c=0.0,
        phi=[0.20],
        theta=[-0.08],
        omega=3e-6,
        alpha=[0.04, 0.02],
        beta=[0.70, 0.15],
        nu=1.8,
    )


def test_arma11_ged_theta_grad_hess_match_ad(arma_ged_series_11: NDArray[np.float64]) -> None:
    params = np.array([0.01, 0.22, -0.14, 0.75, 1.7], dtype=np.float64)
    resid = np.zeros_like(arma_ged_series_11)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _c._arma_nll_11_ged(_as_cptr(params), _as_cptr(arma_ged_series_11), _as_cptr(resid), arma_ged_series_11.size)
    _c._arma_nll_grad_11_ged(_as_cptr(params), _as_cptr(arma_ged_series_11), _as_cptr(resid), _as_cptr(grad), arma_ged_series_11.size)
    _c._arma_hess_11_ged(_as_cptr(params), _as_cptr(arma_ged_series_11), _as_cptr(resid), _as_cptr(hess), arma_ged_series_11.size)
    value_ad, grad_ad, hess_ad = arma_ged_value_grad_hess(params, arma_ged_series_11, 1, 1)

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(hess, hess_ad, rtol=3e-5, atol=2e-6)


def test_arma21_ged_log_matches_ad(arma_ged_series_pq: NDArray[np.float64]) -> None:
    params = np.array([0.02, 0.25, -0.08, 0.12, 0.90, 1.8], dtype=np.float64)
    resid = np.zeros_like(arma_ged_series_pq)
    e0 = np.zeros(1, dtype=np.float64)
    grad_z = np.zeros_like(params)
    z = unpack_arma_ged(params, 2, 1)

    value_ad, grad_ad, hess_ad = arma_ged_logspace_value_grad_hess(z, arma_ged_series_pq, 2, 1)
    value_c = _c._log_arma_nll_pq_ged(
        _as_cptr(z), _as_cptr(arma_ged_series_pq), _as_cptr(resid), _as_cptr(e0), arma_ged_series_pq.size, 2, 1
    )
    _c._log_arma_nll_grad_pq_ged(
        _as_cptr(z), _as_cptr(arma_ged_series_pq), _as_cptr(resid), _as_cptr(e0), _as_cptr(grad_z), arma_ged_series_pq.size, 2, 1
    )

    grad_theta = np.zeros_like(params)
    _c._arma_nll_grad_pq_ged(
        _as_cptr(params), _as_cptr(arma_ged_series_pq), _as_cptr(resid), _as_cptr(e0), _as_cptr(grad_theta), arma_ged_series_pq.size, 2, 1
    )
    hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
    _c._arma_hess_pq_ged(
        _as_cptr(params), _as_cptr(arma_ged_series_pq), _as_cptr(resid), _as_cptr(e0), _as_cptr(hess_theta), arma_ged_series_pq.size, 2, 1
    )
    hess_z = log_hessian_arma_ged(params, grad_theta, hess_theta, 2, 1)

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=6e-6, atol=5e-7)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=5e-5, atol=4e-6)


def test_arma_garch11_ged_theta_grad_hess_match_ad(arma_garch_ged_series_11: NDArray[np.float64]) -> None:
    params = np.array([0.0, 0.18, -0.10, 2e-6, 0.05, 0.92, 1.7], dtype=np.float64)
    resid = np.zeros_like(arma_garch_ged_series_11)
    sigma2 = np.zeros_like(arma_garch_ged_series_11)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)
    h0 = float(np.mean(arma_garch_ged_series_11 ** 2))

    value_c = _c._arma_garch_nll_11_ged(
        _as_cptr(params), _as_cptr(arma_garch_ged_series_11), _as_cptr(resid), _as_cptr(sigma2), h0, arma_garch_ged_series_11.size
    )
    _c._arma_garch_nll_grad_11_ged(
        _as_cptr(params), _as_cptr(arma_garch_ged_series_11), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), h0, arma_garch_ged_series_11.size
    )
    _c._arma_garch_hess_11_ged(
        _as_cptr(params), _as_cptr(arma_garch_ged_series_11), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess), h0, arma_garch_ged_series_11.size
    )
    value_ad, grad_ad, hess_ad = arma_garch_value_grad_hess(params, arma_garch_ged_series_11, 1, 1, 1, 1, "ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=2e-5, atol=1e-2)
    np.testing.assert_allclose(hess, hess_ad, rtol=3e-4, atol=4e3)


def test_arma_garch_pq_ged_log_matches_ad(arma_garch_ged_series_pq: NDArray[np.float64]) -> None:
    params = np.array([0.0, 0.20, -0.08, 3e-6, 0.04, 0.02, 0.70, 0.15, 1.8], dtype=np.float64)
    resid = np.zeros_like(arma_garch_ged_series_pq)
    sigma2 = np.zeros_like(arma_garch_ged_series_pq)
    e0 = np.zeros(2, dtype=np.float64)
    h0 = np.full(2, np.mean(arma_garch_ged_series_pq ** 2), dtype=np.float64)
    grad_z = np.zeros_like(params)
    z = unpack_arma_garch_ged(params, 1, 1, 2, 2)

    value_ad, grad_ad, hess_ad = arma_garch_logspace_value_grad_hess(z, arma_garch_ged_series_pq, 1, 1, 2, 2, "ged")
    value_c = _c._log_arma_garch_nll_pq_ged(
        _as_cptr(z),
        _as_cptr(arma_garch_ged_series_pq),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        arma_garch_ged_series_pq.size,
        1,
        1,
        2,
        2,
    )
    _c._log_arma_garch_nll_grad_pq_ged(
        _as_cptr(z),
        _as_cptr(arma_garch_ged_series_pq),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(grad_z),
        arma_garch_ged_series_pq.size,
        1,
        1,
        2,
        2,
    )

    grad_theta = np.zeros_like(params)
    _c._arma_garch_nll_grad_pq_ged(
        _as_cptr(params),
        _as_cptr(arma_garch_ged_series_pq),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(grad_theta),
        arma_garch_ged_series_pq.size,
        1,
        1,
        2,
        2,
    )
    hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
    _c._arma_garch_hess_pq_ged(
        _as_cptr(params),
        _as_cptr(arma_garch_ged_series_pq),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(hess_theta),
        arma_garch_ged_series_pq.size,
        1,
        1,
        2,
        2,
    )
    hess_z = log_hessian_arma_garch_ged(params, grad_theta, hess_theta, 1, 1, 2, 2)

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=2e-4, atol=2e-3)


def test_arma_ged_public_fit_and_filter_surface(arma_ged_series_11: NDArray[np.float64]) -> None:
    spec = ARMA(1, 1) + GED()
    theta = np.array([0.01, 0.22, -0.14, 0.75, 1.7], dtype=np.float64)

    state = spec.filter(arma_ged_series_11, theta)
    assert state.score is None
    np.testing.assert_allclose(state.sigma2, theta[3])
    assert spec.score(arma_ged_series_11, theta).shape == (5,)
    assert spec.hessian(arma_ged_series_11, theta).shape == (5, 5)

    fit_theta = spec.fit(arma_ged_series_11, solver="slsqp", log_mode=False, verbose=False)
    fit_log = spec.fit(arma_ged_series_11, solver="slsqp", log_mode=True, verbose=False)
    assert fit_theta.params.shape == (5,)
    assert fit_log.params.shape == (5,)
    assert np.all(np.isfinite(fit_theta.params))
    assert np.all(np.isfinite(fit_log.params))


def test_arma_garch_ged_public_fit_and_filter_surface(arma_garch_ged_series_11: NDArray[np.float64]) -> None:
    spec = ARMA(1, 1) + GARCH(1, 1) + GED()
    theta = np.array([0.0, 0.18, -0.10, 2e-6, 0.05, 0.92, 1.7], dtype=np.float64)

    state = spec.filter(arma_garch_ged_series_11, theta)
    assert state.score is None
    assert state.hessian is None
    assert np.all(np.isfinite(spec.score(arma_garch_ged_series_11, theta)))
    assert np.all(np.isfinite(spec.hessian(arma_garch_ged_series_11, theta)))

    fit_theta = spec.fit(arma_garch_ged_series_11, solver="slsqp", log_mode=False, verbose=False)
    fit_log = spec.fit(arma_garch_ged_series_11, solver="slsqp", log_mode=True, verbose=False)
    assert fit_theta.params.shape == (7,)
    assert fit_log.params.shape == (7,)
    assert np.all(np.isfinite(fit_theta.params))
    assert np.all(np.isfinite(fit_log.params))
