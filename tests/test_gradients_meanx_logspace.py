from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

import scivol._core as _core
from scivol import ARX, HARX
from scivol._devtools.ad_oracle import (
    linear_mean_egarch_value_grad_hess,
    linear_mean_garch_value_grad_hess,
    linear_mean_gjr_garch_value_grad_hess,
    linear_mean_logspace_chain_rule_value_grad_hess,
)
from scivol._kernels import (
    linear_mean_egarch_common,
    linear_mean_garch_common,
    linear_mean_gjr_garch_common,
)


if importlib.util.find_spec("jax") is None:
    pytest.skip("jax is required for AD-oracle derivative checks", allow_module_level=True)


COMMONS = {
    "garch": linear_mean_garch_common,
    "gjr": linear_mean_gjr_garch_common,
    "egarch": linear_mean_egarch_common,
}
ORACLES = {
    "garch": linear_mean_garch_value_grad_hess,
    "gjr": linear_mean_gjr_garch_value_grad_hess,
    "egarch": linear_mean_egarch_value_grad_hess,
}
CORE_STEMS = {
    "garch": "linear_mean_garch",
    "gjr": "linear_mean_gjr_garch",
    "egarch": "linear_mean_egarch",
}


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def _parameters(
    model: str,
    mean_kind: str,
    dist: str,
) -> tuple[object, NDArray[np.float64], NDArray[np.float64], int, int, NDArray[np.float64]]:
    rng = np.random.default_rng(20260720 + len(model) + len(mean_kind))
    y = np.ascontiguousarray(rng.standard_normal(70) * 0.25, dtype=np.float64)
    x = np.ascontiguousarray(rng.standard_normal((70, 1)) * 0.10, dtype=np.float64)
    if mean_kind == "arx":
        mean = ARX(1)
        mean_theta = np.array([0.01, 0.12, 0.08], dtype=np.float64)
        p, q = 1, 1
    else:
        mean = HARX((1, 5))
        mean_theta = np.array([0.01, 0.15, 0.05, 0.08], dtype=np.float64)
        p, q = 2, 1
    mean.set_n_exog(1)
    features = COMMONS[model].build_linear_mean_features(mean, y, x)

    if model == "garch":
        model_theta = (
            np.array([0.01, 0.08, 0.85], dtype=np.float64)
            if p == 1
            else np.array([0.01, 0.05, 0.03, 0.85], dtype=np.float64)
        )
    elif model == "gjr":
        model_theta = (
            np.array([0.01, 0.06, 0.02, 0.84], dtype=np.float64)
            if p == 1
            else np.array([0.01, 0.04, 0.02, 0.015, 0.01, 0.84], dtype=np.float64)
        )
    else:
        model_theta = (
            np.array([-0.20, 0.08, -0.04, 0.85], dtype=np.float64)
            if p == 1
            else np.array([-0.20, 0.05, 0.03, -0.04, 0.01, 0.85], dtype=np.float64)
        )

    density_theta = {
        "normal": np.zeros(0, dtype=np.float64),
        "studentt": np.array([8.0], dtype=np.float64),
        "skewt": np.array([8.0, -0.15], dtype=np.float64),
        "ged": np.array([1.6], dtype=np.float64),
    }[dist]
    theta = np.concatenate([mean_theta, model_theta, density_theta])
    return mean, y, features, p, q, theta


@pytest.mark.parametrize("model", ["garch", "gjr", "egarch"])
@pytest.mark.parametrize("mean_kind", ["arx", "harx"])
@pytest.mark.parametrize("dist", ["normal", "studentt", "skewt", "ged"])
def test_meanx_logspace_chain_rule_matches_ad(model: str, mean_kind: str, dist: str) -> None:
    mean, y, features, p, q, theta = _parameters(model, mean_kind, dist)
    n_mean = mean.n_params
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    grad = np.zeros_like(theta)
    hess = np.zeros((theta.size, theta.size), dtype=np.float64)
    stem = CORE_STEMS[model]

    value_c = float(
        getattr(_core, f"_{stem}_nll_pq_{dist}")(
            _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
            _as_cptr(sigma2), y.size, n_mean, p, q
        )
    )
    getattr(_core, f"_{stem}_nll_grad_pq_{dist}")(
        _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
        _as_cptr(sigma2), _as_cptr(grad), y.size, n_mean, p, q
    )
    getattr(_core, f"_{stem}_hess_pq_{dist}")(
        _as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid),
        _as_cptr(sigma2), _as_cptr(hess), y.size, n_mean, p, q
    )
    value_ad, grad_ad, hess_ad = ORACLES[model](theta, y, features, p, q, dist=dist)

    common = COMMONS[model]
    unpack = getattr(common, f"unpack_linear_mean_{'gjr_garch' if model == 'gjr' else model}_{dist}")
    pack = getattr(common, f"pack_linear_mean_{'gjr_garch' if model == 'gjr' else model}_{dist}")
    jacobian = getattr(common, f"jacobian_linear_mean_{'gjr_garch' if model == 'gjr' else model}_{dist}")
    log_hessian = getattr(common, f"log_hessian_linear_mean_{'gjr_garch' if model == 'gjr' else model}_{dist}")
    z = unpack(theta, mean, p, q)
    theta_roundtrip = pack(z, mean, p, q)
    jac = jacobian(theta_roundtrip, mean, p, q)
    grad_z = jac.T @ grad
    hess_z = log_hessian(theta_roundtrip, grad, hess, mean, p, q)
    value_log_ad, grad_log_ad, hess_log_ad = linear_mean_logspace_chain_rule_value_grad_hess(
        z,
        theta_roundtrip,
        value_ad,
        grad_ad,
        hess_ad,
        linear_mean_garch_common._mean_scales(mean),
        p,
        q,
        model=model,
        dist=dist,
    )

    if model == "egarch" and dist == "skewt":
        np.testing.assert_allclose(value_c, value_log_ad, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(grad_z, grad_log_ad, rtol=3e-2, atol=3e-3)
        np.testing.assert_allclose(hess_z, hess_log_ad, rtol=6e-2, atol=2e-1)
    else:
        np.testing.assert_allclose(value_c, value_log_ad, rtol=5e-6, atol=5e-6)
        np.testing.assert_allclose(grad_z, grad_log_ad, rtol=2e-4, atol=3e-5)
        np.testing.assert_allclose(hess_z, hess_log_ad, rtol=4e-3, atol=2e-2)
