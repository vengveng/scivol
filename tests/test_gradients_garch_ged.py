from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import gammaln
from scipy.stats import gennorm

import scivol._core as _c
from scivol import GARCH, GED
from scivol._devtools.ad_oracle import garch_logspace_value_grad_hess, garch_value_grad_hess
from scivol._kernels.transforms import log_hessian_garch, unpack_garch_ged


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _ged_scale(nu: float) -> float:
    return float(np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu))))


def _ged_draws(rng: np.random.Generator, n: int, nu: float) -> NDArray[np.float64]:
    return np.asarray(gennorm.rvs(beta=nu, scale=_ged_scale(nu), size=n, random_state=rng), dtype=np.float64)


def _simulate_garch_ged(
    n: int,
    *,
    seed: int,
    omega: float,
    alpha: list[float],
    beta: list[float],
    nu: float,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    y = np.zeros(total, dtype=np.float64)
    sigma2 = np.zeros(total, dtype=np.float64)
    z = _ged_draws(rng, total, nu)
    p = len(alpha)
    q = len(beta)
    persistence = float(sum(alpha) + sum(beta))
    sigma2[0] = omega / max(1.0 - persistence, 1e-3)

    for t in range(total):
        if t > 0:
            value = omega
            for i in range(p):
                if t > i:
                    value += alpha[i] * y[t - 1 - i] ** 2
            for j in range(q):
                if t > j:
                    value += beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(value, 1e-12)
        y[t] = np.sqrt(sigma2[t]) * z[t]
    return np.ascontiguousarray(y[burn:], dtype=np.float64)


def _sigma_seed(resid: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    sigma2 = np.zeros(len(resid), dtype=np.float64)
    sigma2[:max_lag] = np.mean(resid * resid)
    return sigma2


@pytest.fixture(scope="module")
def garch_ged_series_11() -> NDArray[np.float64]:
    return _simulate_garch_ged(340, seed=20260521, omega=2e-6, alpha=[0.05], beta=[0.92], nu=1.7)


@pytest.fixture(scope="module")
def garch_ged_series_pq() -> NDArray[np.float64]:
    return _simulate_garch_ged(360, seed=20260522, omega=3e-6, alpha=[0.04, 0.02], beta=[0.70, 0.15], nu=1.8)


def test_garch11_ged_theta_grad_hess_match_ad(garch_ged_series_11: NDArray[np.float64]) -> None:
    resid = np.ascontiguousarray(garch_ged_series_11, dtype=np.float64)
    resid2 = resid * resid
    params = np.array([2e-6, 0.05, 0.92, 1.7], dtype=np.float64)
    sigma_g = _sigma_seed(resid, 1)
    sigma_h = _sigma_seed(resid, 1)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    _c._garch_ll_grad_11_ged(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma_g), _as_cptr(grad), resid.size)
    _c._garch_ll_hess_11_ged(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma_h), _as_cptr(hess), resid.size)
    _, grad_ad, hess_ad = garch_value_grad_hess(params, resid2, 1, 1, "ged")

    np.testing.assert_allclose(grad, grad_ad, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=3e-4, atol=3e-3)


def test_garch_pq_ged_log_matches_ad(garch_ged_series_pq: NDArray[np.float64]) -> None:
    resid = np.ascontiguousarray(garch_ged_series_pq, dtype=np.float64)
    resid2 = resid * resid
    params = np.array([3e-6, 0.04, 0.02, 0.70, 0.15, 1.8], dtype=np.float64)
    sigma2 = _sigma_seed(resid, 2)
    grad_z = np.zeros_like(params)
    z = unpack_garch_ged(params, 2, 2)

    value_ad, grad_ad, hess_ad = garch_logspace_value_grad_hess(z, resid2, 2, 2, "ged")
    value_c = _c._log_garch_ll_pq_ged(_as_cptr(z), _as_cptr(resid2), _as_cptr(sigma2), resid.size, 2, 2)
    _c._log_garch_ll_grad_pq_ged(_as_cptr(z), _as_cptr(resid2), _as_cptr(sigma2), _as_cptr(grad_z), resid.size, 2, 2)

    sigma_theta = _sigma_seed(resid, 2)
    grad_theta = np.zeros_like(params)
    _c._garch_ll_grad_pq_ged(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma_theta), _as_cptr(grad_theta), resid.size, 2, 2)
    hess_theta = np.zeros((params.size, params.size), dtype=np.float64)
    sigma_h = _sigma_seed(resid, 2)
    _c._garch_ll_hess_pq_ged(_as_cptr(params), _as_cptr(resid2), _as_cptr(sigma_h), _as_cptr(hess_theta), resid.size, 2, 2)
    hess_z = log_hessian_garch(params, grad_theta, hess_theta, 2, 2, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=4e-4, atol=5e-3)


def test_garch_ged_public_fit_and_filter_surface(garch_ged_series_11: NDArray[np.float64]) -> None:
    spec = GARCH(1, 1) + GED()
    theta = np.array([2e-6, 0.05, 0.92, 1.7], dtype=np.float64)

    state = spec.filter(garch_ged_series_11, theta)
    assert state.distribution == "GED"
    assert np.all(np.isfinite(spec.score(garch_ged_series_11, theta)))
    assert np.all(np.isfinite(spec.hessian(garch_ged_series_11, theta)))

    fit_theta = spec.fit(garch_ged_series_11, solver="slsqp", log_mode=False, verbose=False)
    fit_log = spec.fit(garch_ged_series_11, solver="slsqp", log_mode=True, verbose=False)
    assert fit_theta.params.shape == (4,)
    assert fit_log.params.shape == (4,)
    assert np.all(np.isfinite(fit_theta.params))
    assert np.all(np.isfinite(fit_log.params))
