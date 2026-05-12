from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol import ARX, HARX
from scivol._devtools.ad_oracle import linear_mean_egarch_value_grad_hess
from scivol._kernels.linear_mean_egarch_common import build_linear_mean_features


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _studentt_abs_moment(nu: float) -> float:
    from scipy.special import gammaln

    return (
        2.0
        * np.exp(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu))
        * (nu - 2.0)
        / ((nu - 1.0) * np.sqrt(np.pi * (nu - 2.0)))
    )


def _simulate_meanx_egarch(
    mean_factory,
    mean_params: NDArray[np.float64],
    *,
    n: int,
    seed: int,
    omega: float,
    alpha: list[float],
    gamma: list[float],
    beta: list[float],
    dist: str = "normal",
    nu: float = 8.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    x = rng.standard_normal((total, 1)).astype(np.float64)
    mean = mean_factory()
    mean.set_n_exog(1)

    y = np.zeros(total, dtype=np.float64)
    resid = np.zeros(total, dtype=np.float64)
    sigma2 = np.zeros(total, dtype=np.float64)
    logh = np.zeros(total, dtype=np.float64)

    if dist == "normal":
        abs_moment = np.sqrt(2.0 / np.pi)
        draws = rng.standard_normal(total).astype(np.float64)
    elif dist == "studentt":
        abs_moment = _studentt_abs_moment(nu)
        draws = (rng.standard_t(nu, size=total) * np.sqrt((nu - 2.0) / nu)).astype(np.float64)
    else:
        raise ValueError(dist)

    sigma2[0] = 1.0
    logh[0] = 0.0
    resid[0] = np.sqrt(sigma2[0]) * draws[0]
    features0 = build_linear_mean_features(mean, y[:1], x[:1])
    y[0] = float(features0[0] @ mean_params) + resid[0]

    for t in range(1, total):
        logh_t = omega
        for i, (alpha_i, gamma_i) in enumerate(zip(alpha, gamma), start=1):
            if t >= i:
                z_lag = resid[t - i] / np.sqrt(sigma2[t - i])
                logh_t += alpha_i * (abs(z_lag) - abs_moment) + gamma_i * z_lag
        for j, beta_j in enumerate(beta, start=1):
            if t >= j:
                logh_t += beta_j * logh[t - j]

        logh[t] = logh_t
        sigma2[t] = np.exp(logh_t)
        resid[t] = np.sqrt(sigma2[t]) * draws[t]

        features_t = build_linear_mean_features(mean, y[: t + 1], x[: t + 1])
        y[t] = float(features_t[t] @ mean_params) + resid[t]

    return np.ascontiguousarray(y[burn:], dtype=np.float64), np.ascontiguousarray(x[burn:], dtype=np.float64)


def _call_meanx_egarch_kernel(
    suffix: str,
    specialized: bool,
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    grad: NDArray[np.float64] | None,
    hess: NDArray[np.float64] | None,
    *,
    n_mean: int,
    p: int,
    q: int,
) -> float:
    infix = "11" if specialized else "pq"
    value = getattr(_c, f"_linear_mean_egarch_nll_{infix}_{suffix}")
    args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), y.size, n_mean]
    if not specialized:
        args.extend([p, q])
    nll = float(value(*args))
    if grad is not None:
        sigma2[0] = np.mean(y * y)
        args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), y.size, n_mean]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_egarch_nll_grad_{infix}_{suffix}")(*args)
    if hess is not None:
        sigma2[0] = np.mean(y * y)
        args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess), y.size, n_mean]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_egarch_hess_{infix}_{suffix}")(*args)
    return nll


@pytest.fixture(scope="module")
def arx_egarch_series_11() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _simulate_meanx_egarch(
        lambda: ARX(1),
        np.array([0.02, 0.22, 0.15], dtype=np.float64),
        n=280,
        seed=20260512,
        omega=-0.15,
        alpha=[0.10],
        gamma=[-0.04],
        beta=[0.91],
        dist="normal",
    )


@pytest.fixture(scope="module")
def harx_egarch_series_pq() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _simulate_meanx_egarch(
        lambda: HARX((1, 5)),
        np.array([0.01, 0.30, 0.09, 0.12], dtype=np.float64),
        n=300,
        seed=20260513,
        omega=-0.20,
        alpha=[0.06, 0.03],
        gamma=[0.02, -0.01],
        beta=[0.89],
        dist="normal",
    )


def test_arx_egarch11_theta_grad_hess_match_ad(
    arx_egarch_series_11: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    y, x = arx_egarch_series_11
    mean = ARX(1)
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    params = np.array([0.02, 0.22, 0.15, -0.15, 0.10, -0.04, 0.91], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_egarch_kernel(
        "normal",
        True,
        params,
        y,
        features,
        resid,
        sigma2,
        grad,
        hess,
        n_mean=mean.n_params,
        p=1,
        q=1,
    )
    value_ad, grad_ad, hess_ad = linear_mean_egarch_value_grad_hess(params, y, features, 1, 1, dist="normal")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=4e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=5e-5, atol=8e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=1.5e-3, atol=2e-2)


def test_harx_egarch_pq_theta_grad_hess_match_ad(
    harx_egarch_series_pq: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    y, x = harx_egarch_series_pq
    mean = HARX((1, 5))
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    params = np.array([0.01, 0.30, 0.09, 0.12, -0.20, 0.06, 0.03, 0.02, -0.01, 0.89], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_egarch_kernel(
        "normal",
        False,
        params,
        y,
        features,
        resid,
        sigma2,
        grad,
        hess,
        n_mean=mean.n_params,
        p=2,
        q=1,
    )
    value_ad, grad_ad, hess_ad = linear_mean_egarch_value_grad_hess(params, y, features, 2, 1, dist="normal")

    np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=5e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=5e-5, atol=8e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=1e-3, atol=2e-2)


def test_arx_egarch11_studentt_theta_grad_hess_match_ad() -> None:
    y, x = _simulate_meanx_egarch(
        lambda: ARX(1),
        np.array([0.02, 0.22, 0.15], dtype=np.float64),
        n=260,
        seed=20260514,
        omega=-0.18,
        alpha=[0.09],
        gamma=[-0.03],
        beta=[0.90],
        dist="studentt",
        nu=8.0,
    )
    mean = ARX(1)
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    params = np.array([0.02, 0.22, 0.15, -0.18, 0.09, -0.03, 0.90, 8.0], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_egarch_kernel(
        "studentt",
        True,
        params,
        y,
        features,
        resid,
        sigma2,
        grad,
        hess,
        n_mean=mean.n_params,
        p=1,
        q=1,
    )
    value_ad, grad_ad, hess_ad = linear_mean_egarch_value_grad_hess(params, y, features, 1, 1, dist="studentt")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-5, atol=5e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=5e-5, atol=8e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=1e-3, atol=8e-3)
