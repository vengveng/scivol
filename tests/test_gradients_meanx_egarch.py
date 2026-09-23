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
        args = [
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(features),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(grad),
            y.size,
            n_mean,
        ]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_egarch_nll_grad_{infix}_{suffix}")(*args)
    if hess is not None:
        sigma2[0] = np.mean(y * y)
        args = [
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(features),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(hess),
            y.size,
            n_mean,
        ]
        if not specialized:
            args.extend([p, q])
        getattr(_c, f"_linear_mean_egarch_hess_{infix}_{suffix}")(*args)
    return nll


@pytest.mark.parametrize("dist,shape", [
    ("normal", []), ("studentt", [7.0]), ("ged", [1.7]),
    ("skewt", [7.0, 0.2]), ("skewt", [7.0, -0.2]),
])
def test_linear_egarch_stable_polynomial_is_not_a_coefficient_box(dist, shape):
    rng = np.random.default_rng(82951)
    y = rng.normal(size=90)
    features = np.ascontiguousarray(np.column_stack([np.ones_like(y), rng.normal(size=y.size)]))
    theta = np.array([0.02, 0.04, -0.02, 0.03, 0.02, -0.01, -0.01, 1.2, -0.4, *shape])
    assert np.max(np.abs(np.roots([1.0, -1.2, 0.4]))) < 1.0
    residuals, variances = np.empty_like(y), np.full_like(y, np.mean(y*y))
    gradient, hessian = np.empty_like(theta), np.empty((theta.size, theta.size))
    value = _call_meanx_egarch_kernel(
        dist, False, theta, y, features, residuals, variances,
        gradient, hessian, n_mean=2, p=2, q=2,
    )
    expected = linear_mean_egarch_value_grad_hess(theta, y, features, 2, 2, dist)
    np.testing.assert_allclose(value, expected[0], rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(gradient, expected[1], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(hessian, expected[2], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rho", [-0.12, 0.0, 0.12])
def test_legacy_vg_absolute_moment_split_matches_independent_ad(rho):
    from scivol._devtools.ad_oracle import _jax_modules, _jax_vg_abs_moment

    jax, jnp, jsp = _jax_modules()
    shape = np.array([rho, 2.0])
    value, gradient, hessian = np.empty(1), np.empty(3), np.empty((3, 3))
    _c._gh_abs_moment_grad_hess(
        shape.ctypes.data, value.ctypes.data, gradient.ctypes.data,
        hessian.ctypes.data, np.inf,
    )
    objective = lambda s: _jax_vg_abs_moment(jnp, jsp, s[0], s[1])
    expected_value, expected_gradient = jax.value_and_grad(objective)(jnp.asarray(shape))
    expected_hessian = jax.hessian(objective)(jnp.asarray(shape))
    np.testing.assert_allclose(value[0], expected_value, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(gradient[:2], expected_gradient, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hessian[:2, :2], expected_hessian, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [
    [-0.08590362405556805, 3.6453348340489273], [0.0, 3.0],
    [-0.7, 1.501], [0.7, 3.0], [0.98, 10.0], [-0.98, 50.0],
    [0.9949, 1.5001], [-0.9949, 50.0],
])
def test_vg_absolute_moment_oracle_has_resolved_shape_derivatives(shape):
    from scivol._devtools.ad_oracle import (
        _jax_modules, _jax_vg_mixture_absolute_moment, _jax_vg_power_moment,
    )

    jax, jnp, jsp = _jax_modules()
    point = jnp.asarray(shape)
    references = []
    for nodes in (256, 512):
        objective = lambda s: _jax_vg_mixture_absolute_moment(
            jnp, jsp, s, nodes_per_interval=nodes
        )
        value, gradient = jax.value_and_grad(objective)(point)
        hessian = jax.hessian(objective)(point)
        references.append(np.r_[value, gradient, np.asarray(hessian).ravel()])
    np.testing.assert_allclose(references[0], references[1], rtol=3e-8, atol=3e-10)
    if abs(shape[0]) <= 0.7:
        density_integral = lambda s: _jax_vg_power_moment(
            jnp, jsp, 0.0, 1.0, *s, nodes_count=256
        )
        value, gradient = jax.value_and_grad(density_integral)(point)
        hessian = jax.hessian(density_integral)(point)
        density_reference = np.r_[value, gradient, np.asarray(hessian).ravel()]
        np.testing.assert_allclose(references[1], density_reference, rtol=3e-6, atol=3e-7)


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


@pytest.mark.parametrize(
    ("suffix", "dist", "params"),
    [
        (
            "skewt",
            "skewt",
            np.array([0.02, 0.22, 0.15, -0.15, 0.10, -0.04, 0.91, 8.0, -0.05], dtype=np.float64),
        ),
        (
            "ged",
            "ged",
            np.array([0.02, 0.22, 0.15, -0.15, 0.10, -0.04, 0.91, 1.6], dtype=np.float64),
        ),
    ],
)
def test_arx_egarch11_extended_theta_grad_hess_match_ad(
    arx_egarch_series_11: tuple[NDArray[np.float64], NDArray[np.float64]],
    suffix: str,
    dist: str,
    params: NDArray[np.float64],
) -> None:
    y, x = arx_egarch_series_11
    mean = ARX(1)
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_egarch_kernel(
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
    value_ad, grad_ad, hess_ad = linear_mean_egarch_value_grad_hess(params, y, features, 1, 1, dist=dist)

    grad_rtol = 2e-2 if dist == "skewt" else 2e-3
    hess_rtol = 1.2e-1 if dist == "skewt" else 7e-2
    hess_atol = 8.0 if dist == "skewt" else 7e-2
    np.testing.assert_allclose(value_c, value_ad, rtol=3e-5, atol=6e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=grad_rtol, atol=2e-4)
    np.testing.assert_allclose(hess, hess_ad, rtol=hess_rtol, atol=hess_atol)


@pytest.mark.parametrize(
    ("suffix", "dist", "params"),
    [
        (
            "studentt",
            "studentt",
            np.array([0.01, 0.30, 0.09, 0.12, -0.20, 0.06, 0.03, 0.02, -0.01, 0.89, 8.0], dtype=np.float64),
        ),
        (
            "skewt",
            "skewt",
            np.array(
                [0.01, 0.30, 0.09, 0.12, -0.20, 0.06, 0.03, 0.02, -0.01, 0.89, 8.0, -0.05],
                dtype=np.float64,
            ),
        ),
        (
            "ged",
            "ged",
            np.array([0.01, 0.30, 0.09, 0.12, -0.20, 0.06, 0.03, 0.02, -0.01, 0.89, 1.6], dtype=np.float64),
        ),
    ],
)
def test_harx_egarch_pq_nonnormal_theta_grad_hess_match_ad(
    harx_egarch_series_pq: tuple[NDArray[np.float64], NDArray[np.float64]],
    suffix: str,
    dist: str,
    params: NDArray[np.float64],
) -> None:
    y, x = harx_egarch_series_pq
    mean = HARX((1, 5))
    mean.set_n_exog(1)
    features = build_linear_mean_features(mean, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(params)
    hess = np.zeros((params.size, params.size), dtype=np.float64)

    value_c = _call_meanx_egarch_kernel(
        suffix,
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
    value_ad, grad_ad, hess_ad = linear_mean_egarch_value_grad_hess(params, y, features, 2, 1, dist=dist)

    grad_rtol = 2e-2 if dist == "skewt" else 2e-3
    hess_rtol = 1.2e-1 if dist == "skewt" else 8e-2
    hess_atol = 8.0 if dist == "skewt" else 8e-2
    np.testing.assert_allclose(value_c, value_ad, rtol=3e-5, atol=6e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=grad_rtol, atol=3e-4)
    np.testing.assert_allclose(hess, hess_ad, rtol=hess_rtol, atol=hess_atol)
