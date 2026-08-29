from __future__ import annotations

import numpy as np
import pytest

from scivol import (
    APARCH,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
    _core,
)
from scivol._devtools.ad_oracle import (
    aparch_value_grad_hess,
    gh_surface_value_grad_hess,
    vg_surface_value_grad_hess,
)
from scivol._kernels.aparch_transforms import (
    jacobian_aparch,
    log_hessian_aparch,
    pack_aparch,
    unpack_aparch,
)


Y = np.random.default_rng(714).normal(size=24).astype(np.float64)
VOL = np.array([0.08, 0.07, 0.15, 0.82, 1.7], dtype=np.float64)


def _ptr(array: np.ndarray) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def _native(
    theta: np.ndarray, distribution: int, fixed_lambda: float = np.nan
) -> tuple[float, np.ndarray, np.ndarray]:
    sigma2 = np.empty(Y.size, dtype=np.float64)
    sigma2[0] = float(np.mean(Y * Y))
    gradient = np.empty(theta.size, dtype=np.float64)
    hessian = np.empty((theta.size, theta.size), dtype=np.float64)
    value = _core._aparch_nll_pq(
        _ptr(theta), _ptr(Y), _ptr(sigma2), Y.size, 1, 1,
        distribution, fixed_lambda,
    )
    _core._aparch_nll_grad_pq(
        _ptr(theta), _ptr(Y), _ptr(sigma2), _ptr(gradient), Y.size,
        1, 1, distribution, fixed_lambda,
    )
    _core._aparch_nll_hess_pq(
        _ptr(theta), _ptr(Y), _ptr(sigma2), _ptr(hessian), Y.size,
        1, 1, distribution, fixed_lambda,
    )
    return float(value), gradient, hessian


@pytest.mark.parametrize(
    ("dist", "code", "shape"),
    [
        ("normal", 0, []),
        ("studentt", 1, [8.0]),
        ("skewt", 2, [8.0, 0.12]),
        ("ged", 3, [1.6]),
    ],
)
def test_standard_density_derivatives_match_ad(
    dist: str, code: int, shape: list[float]
) -> None:
    theta = np.concatenate((VOL, shape))
    actual = _native(theta, code)
    expected = aparch_value_grad_hess(theta, Y, 1, 1, dist)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual[2], expected[2], rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize(
    ("fixed_lambda", "shape", "variance_gamma"),
    [
        (np.nan, [0.1, 1.3, 0.4], False),
        (-0.5, [0.1, 1.3], False),
        (1.0, [0.1, 1.3], False),
        (np.inf, [0.1, 3.0], True),
    ],
)
def test_gh_family_derivatives_match_ad(
    fixed_lambda: float, shape: list[float], variance_gamma: bool
) -> None:
    theta = np.concatenate((VOL, shape))
    actual = _native(theta, 4, fixed_lambda)
    if variance_gamma:
        expected = vg_surface_value_grad_hess(
            theta, Y, volatility_kind="aparch", p_vol=1, q_vol=1
        )
    else:
        expected = gh_surface_value_grad_hess(
            theta,
            Y,
            volatility_kind="aparch",
            p_vol=1,
            q_vol=1,
            fixed_lambda=None if np.isnan(fixed_lambda) else fixed_lambda,
        )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize(
    ("density", "theta", "code", "fixed_lambda"),
    [
        ("Normal", VOL, 0, np.nan),
        ("SkewT", np.r_[VOL, 8.0, 0.12], 2, np.nan),
        ("GeneralizedHyperbolic", np.r_[VOL, 0.1, 1.3, 0.4], 4, np.nan),
        ("VarianceGamma", np.r_[VOL, 0.1, 3.0], 4, np.inf),
    ],
)
def test_log_transform_matches_ad_chain_rule(
    density: str, theta: np.ndarray, code: int, fixed_lambda: float
) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    value, gradient, hessian = _native(theta, code, fixed_lambda)
    z = unpack_aparch(theta, 1, 1, density)
    theta_j = jnp.asarray(theta)
    gradient_j = jnp.asarray(gradient)
    hessian_j = jnp.asarray(hessian)

    def pack(z_j):
        omega = jnp.log1p(jnp.exp(z_j[0]))
        dynamic = jax.nn.softmax(jnp.array([z_j[1], z_j[3], 0.0]))[:2]
        gamma = 0.999 * jnp.tanh(z_j[2])
        delta = 0.25 + 3.75 * jax.nn.sigmoid(z_j[4])
        vol = jnp.array([omega, dynamic[0], gamma, dynamic[1], delta])
        if density == "Normal":
            return vol
        if density == "SkewT":
            return jnp.concatenate((vol, jnp.array([
                2.0 + jnp.log1p(jnp.exp(z_j[5])), jnp.tanh(z_j[6])
            ])))
        rho = 0.995 * jnp.tanh(z_j[5])
        if density == "VarianceGamma":
            shape = 1.51 + 48.49 * jax.nn.sigmoid(z_j[6])
            return jnp.concatenate((vol, jnp.array([rho, shape])))
        zeta = 0.1 + 24.9 * jax.nn.sigmoid(z_j[6])
        shape_lambda = -6.0 + 12.0 * jax.nn.sigmoid(z_j[7])
        return jnp.concatenate((vol, jnp.array([rho, zeta, shape_lambda])))

    def local_objective(z_j):
        difference = pack(z_j) - theta_j
        return value + gradient_j @ difference + 0.5 * difference @ hessian_j @ difference

    expected_gradient = np.asarray(jax.grad(local_objective)(jnp.asarray(z)))
    expected_hessian = np.asarray(jax.hessian(local_objective)(jnp.asarray(z)))
    actual_gradient = jacobian_aparch(theta, 1, 1, density).T @ gradient
    actual_hessian = log_hessian_aparch(
        theta, gradient, hessian, 1, 1, density
    )
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(pack_aparch(z, 1, 1, density), theta)


@pytest.mark.parametrize(
    ("density", "shape"),
    [
        (Normal(), []),
        (StudentT(), [8.0]),
        (SkewT(), [8.0, 0.12]),
        (GED(), [1.6]),
        (GeneralizedHyperbolic(), [0.1, 1.3, 0.4]),
        (NormalInverseGaussian(), [0.1, 1.3]),
        (Hyperbolic(), [0.1, 1.3]),
        (VarianceGamma(), [0.1, 3.0]),
    ],
)
def test_public_fixed_workflows_cover_all_densities(
    density: object, shape: list[float]
) -> None:
    spec = APARCH(1, 1) + density
    theta = np.concatenate((VOL, shape))
    fixed = spec.fix(Y, theta)

    assert fixed.filter().sigma2.shape == Y.shape
    assert spec.score(Y, theta).shape == theta.shape
    assert spec.hessian(Y, theta).shape == (theta.size, theta.size)
    assert fixed.forecast(2).variance.shape == (2,)
    assert spec.simulate(12, theta, burn=5, seed=81).data.shape == (12,)


def test_component_contract_and_auto_preview() -> None:
    from scivol import AutoVol

    component = APARCH(2, 1)
    assert component.n_params == 7
    assert component.signature == "APARCH(2,1)"
    assert len(component.bounds()) == component.n_params
    previews = (AutoVol(candidates=["APARCH"], max_p=1, max_q=1) + Normal()).preview_candidates()
    assert [str(spec) for spec in previews] == ["APARCH(1,1)+Normal"]


def test_aparch_auto_includes_genuine_arma_cross_product() -> None:
    from scivol import AutoMean, AutoVol

    spec = (
        AutoMean(max_p=1, max_q=0)
        + AutoVol(candidates=["APARCH"], max_p=1, max_q=1)
        + Normal()
    )
    previews = [str(candidate) for candidate in spec.preview_candidates()]
    assert "APARCH(1,1)+Normal" in previews
    assert "ARMA(1,0)+APARCH(1,1)+Normal" in previews
