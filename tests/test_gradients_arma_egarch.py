from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _core
from scivol._devtools.ad_oracle import (
    arma_egarch_logspace_value_grad_hess,
    arma_egarch_value_grad_hess,
)
from scivol._kernels import transforms


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


@pytest.fixture(scope="module")
def arma_egarch_series() -> NDArray[np.float64]:
    rng = np.random.default_rng(20260711)
    return np.ascontiguousarray(rng.standard_normal(90) * 0.30, dtype=np.float64)


CASES = (
    ("studentt", np.array([0.01, 0.15, -0.08, -0.30, 0.10, -0.04, 0.90, 8.0], dtype=np.float64)),
    ("skewt", np.array([0.01, 0.15, -0.08, -0.30, 0.10, -0.04, 0.90, 8.0, -0.15], dtype=np.float64)),
    ("ged", np.array([0.01, 0.15, -0.08, -0.30, 0.10, -0.04, 0.90, 1.6], dtype=np.float64)),
)


def _theta_c_derivatives(
    dist: str,
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    e0 = np.zeros(1, dtype=np.float64)
    h0 = np.full(1, np.mean(y * y), dtype=np.float64)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)

    value = float(
        getattr(_core, f"_arma_egarch_nll_pq_{dist}")(
            _as_cptr(theta),
            _as_cptr(y),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            y.size,
            1,
            1,
            1,
            1,
        )
    )
    getattr(_core, f"_arma_egarch_nll_grad_pq_{dist}")(
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
        1,
        1,
    )
    getattr(_core, f"_arma_egarch_hess_pq_{dist}")(
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
        1,
        1,
    )
    return value, grad, hess


@pytest.mark.parametrize(("dist", "theta"), CASES)
def test_arma_egarch_theta_value_grad_hess_match_ad(
    arma_egarch_series: NDArray[np.float64],
    dist: str,
    theta: NDArray[np.float64],
) -> None:
    value_c, grad_c, hess_c = _theta_c_derivatives(dist, theta, arma_egarch_series)
    value_ad, grad_ad, hess_ad = arma_egarch_value_grad_hess(
        theta, arma_egarch_series, 1, 1, 1, 1, dist
    )

    if dist == "skewt":
        np.testing.assert_allclose(value_c, value_ad, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(grad_c, grad_ad, rtol=8e-3, atol=3e-4)
        np.testing.assert_allclose(hess_c, hess_ad, rtol=4e-2, atol=8e-2)
    else:
        np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=3e-7)
        np.testing.assert_allclose(grad_c, grad_ad, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(hess_c, hess_ad, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(("dist", "theta"), CASES)
def test_arma_egarch_log_value_grad_hess_match_ad(
    arma_egarch_series: NDArray[np.float64],
    dist: str,
    theta: NDArray[np.float64],
) -> None:
    unpack = getattr(transforms, f"unpack_arma_egarch_{dist}")
    pack = getattr(transforms, f"pack_arma_egarch_{dist}")
    log_hessian = getattr(transforms, f"log_hessian_arma_egarch_{dist}")
    z = unpack(theta, 1, 1, 1, 1)
    theta_roundtrip = pack(z, 1, 1, 1, 1)

    resid = np.zeros_like(arma_egarch_series)
    sigma2 = np.zeros_like(arma_egarch_series)
    e0 = np.zeros(1, dtype=np.float64)
    h0 = np.full(1, np.mean(arma_egarch_series**2), dtype=np.float64)
    grad_z = np.zeros_like(z)
    value_c = float(
        getattr(_core, f"_log_arma_egarch_nll_pq_{dist}")(
            _as_cptr(z),
            _as_cptr(arma_egarch_series),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0),
            arma_egarch_series.size,
            1,
            1,
            1,
            1,
        )
    )
    getattr(_core, f"_log_arma_egarch_nll_grad_pq_{dist}")(
        _as_cptr(z),
        _as_cptr(arma_egarch_series),
        _as_cptr(resid),
        _as_cptr(sigma2),
        _as_cptr(e0),
        _as_cptr(h0),
        _as_cptr(grad_z),
        arma_egarch_series.size,
        1,
        1,
        1,
        1,
    )
    _, grad_theta, hess_theta = _theta_c_derivatives(dist, theta_roundtrip, arma_egarch_series)
    hess_z = log_hessian(theta_roundtrip, grad_theta, hess_theta, 1, 1, 1, 1)
    value_ad, grad_ad, hess_ad = arma_egarch_logspace_value_grad_hess(
        z, arma_egarch_series, 1, 1, 1, 1, dist
    )

    if dist == "skewt":
        np.testing.assert_allclose(value_c, value_ad, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=8e-3, atol=3e-4)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=4e-2, atol=8e-2)
    else:
        np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=3e-7)
        np.testing.assert_allclose(grad_z, grad_ad, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(hess_z, hess_ad, rtol=3e-3, atol=3e-3)
