from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol import ARX, GED, HARX, GJRGARCH, Normal, SkewT, StudentT
from scivol._devtools.ad_oracle import linear_mean_gjr_garch_value_grad_hess
from scivol._kernels.linear_mean_gjr_garch_common import build_linear_mean_features


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _call_meanx_gjr_kernel(
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
    value = getattr(_c, f"_linear_mean_gjr_garch_nll_{infix}_{suffix}")
    args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), y.size, n_mean]
    if not specialized:
        args.extend([p, q])
    nll = float(value(*args))
    if grad is not None:
        sigma2[0] = np.mean(y * y)
        args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), y.size, n_mean]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_gjr_garch_nll_grad_{infix}_{suffix}")(*args)
    if hess is not None:
        sigma2[0] = np.mean(y * y)
        args = [_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess), y.size, n_mean]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_gjr_garch_hess_{infix}_{suffix}")(*args)
    return nll


def _simulate_arx_gjr_normal(
    n: int,
    *,
    seed: int,
    const: float,
    ar: list[float],
    beta_x: list[float],
    omega: float,
    alpha: list[float],
    gamma: list[float],
    beta: list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    k = len(beta_x)
    x = rng.standard_normal((total, k)).astype(np.float64)
    y = np.zeros(total, dtype=np.float64)
    eps = np.zeros(total, dtype=np.float64)
    sigma2 = np.zeros(total, dtype=np.float64)
    z = rng.standard_normal(total).astype(np.float64)
    persistence = float(sum(alpha) + 0.5 * sum(gamma) + sum(beta))
    sigma2[0] = omega / max(1.0 - persistence, 1e-3)

    for t in range(total):
        if t > 0:
            value = omega
            for i in range(len(alpha)):
                if t > i:
                    e_lag = eps[t - 1 - i]
                    value += alpha[i] * e_lag**2 + gamma[i] * (1.0 if e_lag < 0.0 else 0.0) * e_lag**2
            for j in range(len(beta)):
                if t > j:
                    value += beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(value, 1e-12)
        mu_t = const + float(np.dot(beta_x, x[t]))
        for lag in range(1, len(ar) + 1):
            if t >= lag:
                mu_t += ar[lag - 1] * y[t - lag]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        y[t] = mu_t + eps[t]

    return np.ascontiguousarray(y[burn:], dtype=np.float64), np.ascontiguousarray(x[burn:], dtype=np.float64)


def _simulate_harx_gjr_normal(
    n: int,
    *,
    seed: int,
    const: float,
    har: list[float],
    horizons: tuple[int, ...],
    beta_x: list[float],
    omega: float,
    alpha: list[float],
    gamma: list[float],
    beta: list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    k = len(beta_x)
    x = rng.standard_normal((total, k)).astype(np.float64)
    y = np.zeros(total, dtype=np.float64)
    eps = np.zeros(total, dtype=np.float64)
    sigma2 = np.zeros(total, dtype=np.float64)
    z = rng.standard_normal(total).astype(np.float64)
    persistence = float(sum(alpha) + 0.5 * sum(gamma) + sum(beta))
    sigma2[0] = omega / max(1.0 - persistence, 1e-3)

    for t in range(total):
        if t > 0:
            value = omega
            for i in range(len(alpha)):
                if t > i:
                    e_lag = eps[t - 1 - i]
                    value += alpha[i] * e_lag**2 + gamma[i] * (1.0 if e_lag < 0.0 else 0.0) * e_lag**2
            for j in range(len(beta)):
                if t > j:
                    value += beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(value, 1e-12)

        mu_t = const + float(np.dot(beta_x, x[t]))
        for coeff, horizon in zip(har, horizons):
            width = min(horizon, t)
            if width > 0:
                mu_t += coeff * float(np.mean(y[t - width:t]))
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        y[t] = mu_t + eps[t]

    return np.ascontiguousarray(y[burn:], dtype=np.float64), np.ascontiguousarray(x[burn:], dtype=np.float64)


@pytest.fixture(scope="module")
def arx_gjr_series_11() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _simulate_arx_gjr_normal(
        280,
        seed=20260602,
        const=0.02,
        ar=[0.20],
        beta_x=[0.15],
        omega=2e-6,
        alpha=[0.04],
        gamma=[0.03],
        beta=[0.90],
    )


@pytest.fixture(scope="module")
def harx_gjr_series_pq() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _simulate_harx_gjr_normal(
        300,
        seed=20260603,
        const=0.01,
        har=[0.35, 0.10],
        horizons=(1, 5),
        beta_x=[0.12],
        omega=3e-6,
        alpha=[0.03, 0.015],
        gamma=[0.02, 0.01],
        beta=[0.88],
    )


@pytest.mark.parametrize(
    ("suffix", "dist", "params"),
    [
        ("normal", "normal", np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90], dtype=np.float64)),
        ("studentt", "studentt", np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 8.0], dtype=np.float64)),
        ("skewt", "skewt", np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 8.0, -0.10], dtype=np.float64)),
        ("ged", "ged", np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 1.6], dtype=np.float64)),
    ],
)
def test_arx_gjr11_theta_grad_hess_match_ad(
    arx_gjr_series_11: tuple[NDArray[np.float64], NDArray[np.float64]],
    suffix: str,
    dist: str,
    params: NDArray[np.float64],
) -> None:
    y, x = arx_gjr_series_11
    mean = ARX(1)
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_gjr_kernel(
        suffix,
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
    value_ad, grad_ad, hess_ad = linear_mean_gjr_garch_value_grad_hess(params, y, features, 1, 1, dist=dist)

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=3e-5, atol=4e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=8e-4, atol=1e-2)


def test_harx_gjr_pq_theta_grad_hess_match_ad(
    harx_gjr_series_pq: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    y, x = harx_gjr_series_pq
    mean = HARX((1, 5))
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    params = np.array([0.01, 0.35, 0.10, 0.12, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_gjr_kernel(
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
    value_ad, grad_ad, hess_ad = linear_mean_gjr_garch_value_grad_hess(params, y, features, 2, 1, dist="normal")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=4e-5, atol=5e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=1e-3, atol=1.2e-2)


def test_harx_gjr_pq_ged_theta_grad_hess_match_ad(
    harx_gjr_series_pq: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    y, x = harx_gjr_series_pq
    mean = HARX((1, 5))
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    params = np.array([0.01, 0.35, 0.10, 0.12, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88, 1.7], dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_gjr_kernel(
        "ged",
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
    value_ad, grad_ad, hess_ad = linear_mean_gjr_garch_value_grad_hess(params, y, features, 2, 1, dist="ged")

    np.testing.assert_allclose(value_c, value_ad, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=4e-5, atol=6e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=1e-3, atol=1.5e-2)


@pytest.mark.parametrize(
    ("density_factory", "theta"),
    [
        (Normal, np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90], dtype=np.float64)),
        (StudentT, np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 8.0], dtype=np.float64)),
        (SkewT, np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 8.0, -0.10], dtype=np.float64)),
        (GED, np.array([0.02, 0.20, 0.15, 2e-6, 0.04, 0.03, 0.90, 1.6], dtype=np.float64)),
    ],
)
def test_meanx_gjr_public_fit_paths_are_finite(
    arx_gjr_series_11: tuple[NDArray[np.float64], NDArray[np.float64]],
    density_factory,
    theta: NDArray[np.float64],
) -> None:
    y, x = arx_gjr_series_11
    spec = ARX(1) + GJRGARCH(1, 1) + density_factory()
    assert np.all(np.isfinite(spec.score(y, theta, x=x)))
    assert np.all(np.isfinite(spec.hessian(y, theta, x=x)))
    fit_theta = spec.fit(y, x=x, solver="slsqp", log_mode=False, verbose=False)
    fit_log = spec.fit(y, x=x, solver="slsqp", log_mode=True, verbose=False)
    assert np.all(np.isfinite(fit_theta.params))
    assert np.all(np.isfinite(fit_log.params))
    assert fit_theta.fit_info.optimization_space == "theta-space"
    assert fit_log.fit_info.optimization_space == "z-space"


def test_harx_meanx_gjr_normal_public_fit_paths_are_finite(
    harx_gjr_series_pq: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    y, x = harx_gjr_series_pq
    theta = np.array([0.01, 0.35, 0.10, 0.12, 3e-6, 0.03, 0.015, 0.02, 0.01, 0.88], dtype=np.float64)
    spec = HARX((1, 5)) + GJRGARCH(2, 1) + Normal()
    assert np.all(np.isfinite(spec.score(y, theta, x=x)))
    assert np.all(np.isfinite(spec.hessian(y, theta, x=x)))
