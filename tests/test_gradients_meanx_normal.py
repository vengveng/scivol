from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _c
from scivol import ARX, HARX
from scivol._devtools.ad_oracle import (
    linear_mean_normal_log_value_grad_hess,
    linear_mean_normal_value_grad_hess,
)
from scivol._kernels.linear_mean_normal_common import (
    build_linear_mean_features,
    jacobian_linear_mean_normal,
    log_hessian_linear_mean_normal,
    pack_linear_mean_normal,
    unpack_linear_mean_normal,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _mean_scales(bounds: list[tuple[float, float]]) -> NDArray[np.float64]:
    scales = np.empty(len(bounds), dtype=np.float64)
    for idx, (lo, hi) in enumerate(bounds):
        scales[idx] = max(abs(float(lo)), abs(float(hi)))
    return scales


def _call_meanx_normal_kernel(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    resid = np.zeros_like(y)
    grad = np.zeros(theta.size, dtype=np.float64)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)
    value = float(
        _c._linear_mean_nll_normal(
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(features),
            _as_cptr(resid),
            y.size,
            features.shape[1],
        )
    )
    _c._linear_mean_nll_grad_normal(
        _as_cptr(theta),
        _as_cptr(y),
        _as_cptr(features),
        _as_cptr(resid),
        _as_cptr(grad),
        y.size,
        features.shape[1],
    )
    _c._linear_mean_hess_normal(
        _as_cptr(theta),
        _as_cptr(y),
        _as_cptr(features),
        _as_cptr(resid),
        _as_cptr(hess),
        y.size,
        features.shape[1],
    )
    return value, grad, hess


@pytest.mark.parametrize(
    ("mean", "x", "params"),
    [
        (ARX(1), np.linspace(-0.3, 0.3, 80, dtype=np.float64)[:, None], np.array([0.12, 0.35, -0.60], dtype=np.float64)),
        (HARX((1, 5)), np.linspace(-0.2, 0.2, 80, dtype=np.float64)[:, None], np.array([0.08, 0.40, -0.15, 0.55], dtype=np.float64)),
    ],
)
def test_meanx_normal_c_derivatives_match_ad_oracle(
    mean: ARX | HARX,
    x: NDArray[np.float64],
    params: NDArray[np.float64],
) -> None:
    rng = np.random.default_rng(20260512)
    y = np.ascontiguousarray(rng.standard_normal(x.shape[0]), dtype=np.float64)
    mean.set_n_exog(x.shape[1])
    features = build_linear_mean_features(mean, y, x)

    value_c, grad_c, hess_c = _call_meanx_normal_kernel(params, y, features)
    value_ad, grad_ad, hess_ad = linear_mean_normal_value_grad_hess(params, y, features)

    assert value_c == pytest.approx(value_ad, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(grad_c, grad_ad, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hess_c, hess_ad, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("mean", "x", "theta"),
    [
        (ARX(1), np.linspace(-0.25, 0.25, 90, dtype=np.float64)[:, None], np.array([0.10, 0.28, 0.45], dtype=np.float64)),
        (HARX((1, 5)), np.linspace(-0.15, 0.30, 90, dtype=np.float64)[:, None], np.array([0.06, 0.30, 0.12, -0.35], dtype=np.float64)),
    ],
)
def test_meanx_normal_log_transform_matches_ad_oracle(
    mean: ARX | HARX,
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> None:
    rng = np.random.default_rng(91)
    y = np.ascontiguousarray(rng.standard_normal(x.shape[0]), dtype=np.float64)
    mean.set_n_exog(x.shape[1])
    features = build_linear_mean_features(mean, y, x)

    z = unpack_linear_mean_normal(theta, mean)
    value_theta, grad_theta, hess_theta = _call_meanx_normal_kernel(theta, y, features)

    J = jacobian_linear_mean_normal(theta, mean)
    grad_z = J.T @ grad_theta
    hess_z = log_hessian_linear_mean_normal(theta, grad_theta, hess_theta, mean)
    scales = _mean_scales(list(mean.bounds()))

    value_ad, grad_ad, hess_ad = linear_mean_normal_log_value_grad_hess(z, y, features, scales)
    packed = pack_linear_mean_normal(z, mean)

    np.testing.assert_allclose(packed, theta, rtol=1e-10, atol=1e-10)
    assert value_theta == pytest.approx(value_ad, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(grad_z, grad_ad, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hess_z, hess_ad, rtol=1e-8, atol=1e-8)
