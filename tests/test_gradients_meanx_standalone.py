from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _core
from scivol import ARX, HARX
from scivol._devtools.ad_oracle import (
    linear_mean_garch_value_grad_hess,
    linear_mean_logspace_chain_rule_value_grad_hess,
)
from scivol._kernels import linear_mean_garch_common as common


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def _case(mean_kind: str, dist: str):
    rng = np.random.default_rng(20260712 if mean_kind == "arx" else 20260713)
    y = np.ascontiguousarray(rng.standard_normal(80) * 0.25, dtype=np.float64)
    x = np.ascontiguousarray(rng.standard_normal((80, 1)) * 0.10, dtype=np.float64)
    mean = ARX(1) if mean_kind == "arx" else HARX((1, 5))
    mean.set_n_exog(1)
    features = common.build_linear_mean_features(mean, y, x)
    mean_theta = (
        np.array([0.01, 0.12, 0.08], dtype=np.float64)
        if mean_kind == "arx"
        else np.array([0.01, 0.15, 0.05, 0.08], dtype=np.float64)
    )
    tail = {
        "studentt": np.array([0.08, 8.0], dtype=np.float64),
        "skewt": np.array([0.08, 8.0, -0.15], dtype=np.float64),
        "ged": np.array([0.08, 1.6], dtype=np.float64),
    }[dist]
    return mean, y, features, np.concatenate([mean_theta, tail])


@pytest.mark.parametrize("mean_kind", ["arx", "harx"])
@pytest.mark.parametrize("dist", ["studentt", "skewt", "ged"])
def test_meanx_standalone_theta_and_log_derivatives_match_ad(mean_kind: str, dist: str) -> None:
    mean, y, features, theta = _case(mean_kind, dist)
    n_mean = mean.n_params
    resid = np.zeros_like(y)
    sigma2 = np.full_like(y, theta[n_mean])
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)

    value_c = float(
        getattr(_core, f"_linear_mean_garch_nll_pq_{dist}")(
            _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
            _as_cptr(sigma2), y.size, n_mean, 0, 0
        )
    )
    getattr(_core, f"_linear_mean_garch_nll_grad_pq_{dist}")(
        _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
        _as_cptr(sigma2), _as_cptr(grad), y.size, n_mean, 0, 0
    )
    getattr(_core, f"_linear_mean_garch_hess_pq_{dist}")(
        _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
        _as_cptr(sigma2), _as_cptr(hess), y.size, n_mean, 0, 0
    )
    value_ad, grad_ad, hess_ad = linear_mean_garch_value_grad_hess(
        theta, y, features, 0, 0, dist=dist
    )

    np.testing.assert_allclose(value_c, value_ad, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(grad, grad_ad, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(hess, hess_ad, rtol=8e-4, atol=8e-3)

    unpack = getattr(common, f"unpack_linear_mean_garch_{dist}")
    pack = getattr(common, f"pack_linear_mean_garch_{dist}")
    jacobian = getattr(common, f"jacobian_linear_mean_garch_{dist}")
    log_hessian = getattr(common, f"log_hessian_linear_mean_garch_{dist}")
    z = unpack(theta, mean, 0, 0)
    theta_roundtrip = pack(z, mean, 0, 0)
    jac = jacobian(theta_roundtrip, mean, 0, 0)
    grad_z = jac.T @ grad
    hess_z = log_hessian(theta_roundtrip, grad, hess, mean, 0, 0)
    value_log_ad, grad_log_ad, hess_log_ad = linear_mean_logspace_chain_rule_value_grad_hess(
        z,
        theta_roundtrip,
        value_ad,
        grad_ad,
        hess_ad,
        common._mean_scales(mean),
        0,
        0,
        model="garch",
        dist=dist,
    )

    np.testing.assert_allclose(value_c, value_log_ad, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(grad_z, grad_log_ad, rtol=4e-5, atol=8e-6)
    np.testing.assert_allclose(hess_z, hess_log_ad, rtol=1e-3, atol=1e-2)
