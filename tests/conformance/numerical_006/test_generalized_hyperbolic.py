from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import genhyperbolic

pass
from .oracle_ad import (
    gh_standardized_value_grad_hess,
    gh_surface_value_grad_hess,
    vg_standardized_value_grad_hess,
    vg_surface_value_grad_hess,
)
pass
pass
pass
pass



from .transport import DensityTransport, surface
from .oracle_density_values import (standardized_gh_parameters, standardized_gh_logpdf, standardized_vg_logpdf)
_core = DensityTransport()
NIG_LAMBDA = -.5
HYPERBOLIC_LAMBDA = 1.
VG_LAMBDA_MIN = 1.51
VG_LAMBDA_MAX = 50.

Y = np.array([0.22, -0.31, 0.17, 0.41, -0.28, 0.13, -0.19, 0.36, -0.24])
FEATURES = np.column_stack((np.ones(Y.size), np.linspace(-0.4, 0.5, Y.size)))
SHAPE = np.array([0.18, 1.6, -0.35])
VG_SHAPE = np.array([0.18, 2.8])


def _volatility_params(kind: str) -> tuple[np.ndarray, int, int]:
    if kind == "constant":
        return np.array([0.18]), 0, 0
    if kind == "garch":
        return np.array([0.12, 0.08, 0.05, 0.72]), 2, 1
    if kind == "gjr":
        return np.array([0.12, 0.06, 0.04, 0.03, 0.02, 0.70]), 2, 1
    if kind == "egarch":
        return np.array([-0.25, 0.08, 0.04, -0.03, 0.02, 0.65]), 2, 1
    raise AssertionError(kind)


def _surface_theta(
    mean_kind: str,
    volatility_kind: str,
    shape: np.ndarray = SHAPE,
) -> tuple[np.ndarray, int, int]:
    vol, p_vol, q_vol = _volatility_params(volatility_kind)
    if mean_kind == "none":
        mean = np.zeros(0)
    elif mean_kind == "linear":
        mean = np.array([0.02, 0.10])
    else:
        mean = np.array([0.01, 0.15, -0.10])
    return np.concatenate((mean, vol, shape)), p_vol, q_vol


def _native_surface(theta, mean_kind, volatility_kind, p_vol, q_vol, fixed_lambda=None):
    return surface(theta, Y, mean_kind, volatility_kind, p_vol, q_vol, fixed_lambda)







@pytest.mark.parametrize(
    "rho,zeta,shape_lambda",
    [(-0.7, 0.2, -5.5), (0.0, 1.0, -0.5), (0.45, 4.0, 2.0)],
)
def test_standardization_and_scipy_density_parity(
    rho: float, zeta: float, shape_lambda: float
) -> None:
    scipy_params = standardized_gh_parameters(rho, zeta, shape_lambda)
    mean, variance = genhyperbolic.stats(*scipy_params, moments="mv")
    values = np.linspace(-4.0, 4.0, 31)

    assert float(mean) == pytest.approx(0.0, abs=2e-10)
    assert float(variance) == pytest.approx(1.0, rel=2e-10, abs=2e-10)
    np.testing.assert_allclose(
        standardized_gh_logpdf(values, rho, zeta, shape_lambda),
        genhyperbolic.logpdf(values, *scipy_params),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "rho,zeta,shape_lambda",
    [
        (-0.90, 0.10, -5.5),
        (0.90, 0.10, 5.5),
        (-0.75, 0.30, -0.5),
        (0.75, 0.30, 1.0),
        (-0.50, 8.0, 4.0),
        (0.50, 25.0, -4.0),
    ],
)
def test_native_gh_density_matches_scipy_across_bounds_and_tails(
    rho: float, zeta: float, shape_lambda: float
) -> None:
    values = np.array([-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0])
    params = np.array([rho, zeta, shape_lambda])
    actual = _core._gh_standardized_nll(
        params.ctypes.data, values.ctypes.data, values.size
    )
    expected = -standardized_gh_logpdf(values, rho, zeta, shape_lambda).sum()

    assert actual == pytest.approx(expected, rel=3e-10, abs=3e-10)


@pytest.mark.parametrize("rho,zeta", [(-0.9, 0.1), (0.0, 1.0), (0.9, 25.0)])
def test_fast_nig_density_matches_scipy_across_bounds_and_tails(
    rho: float, zeta: float
) -> None:
    values = np.array([-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0])
    params = np.array([rho, zeta])
    actual = _core._nig_standardized_nll(
        params.ctypes.data, values.ctypes.data, values.size
    )
    expected = -standardized_gh_logpdf(values, rho, zeta, NIG_LAMBDA).sum()

    assert actual == pytest.approx(expected, rel=3e-10, abs=3e-10)


@pytest.mark.parametrize("estimate_lambda", [True, False])
def test_standardized_shape_derivatives_match_ad(estimate_lambda: bool) -> None:
    params = SHAPE.copy() if estimate_lambda else SHAPE[:2].copy()
    expected = gh_standardized_value_grad_hess(
        params, Y, fixed_lambda=None if estimate_lambda else NIG_LAMBDA
    )
    value = np.empty(1)
    grad = np.empty(params.size)
    hess = np.empty((params.size, params.size))
    if estimate_lambda:
        _core._gh_standardized_nll_grad_hess(
            params.ctypes.data,
            Y.ctypes.data,
            value.ctypes.data,
            grad.ctypes.data,
            hess.ctypes.data,
            Y.size,
        )
    else:
        _core._nig_standardized_nll_grad_hess(
            params.ctypes.data,
            Y.ctypes.data,
            value.ctypes.data,
            grad.ctypes.data,
            hess.ctypes.data,
            Y.size,
        )

    np.testing.assert_allclose(value[0], expected[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(grad, expected[1], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(hess, expected[2], rtol=2e-7, atol=2e-8)


def test_standardized_variance_gamma_density_and_derivatives() -> None:
    rho, shape_lambda = VG_SHAPE
    density = lambda value: float(
        np.exp(standardized_vg_logpdf(np.array([value]), rho, shape_lambda)[0])
    )
    mass = quad(density, -np.inf, np.inf, epsabs=2e-9)[0]
    mean = quad(lambda value: value * density(value), -np.inf, np.inf, epsabs=2e-9)[0]
    variance = quad(
        lambda value: value * value * density(value),
        -np.inf,
        np.inf,
        epsabs=2e-9,
    )[0]
    expected = vg_standardized_value_grad_hess(VG_SHAPE, Y)
    value = np.empty(1)
    grad = np.empty(2)
    hess = np.empty((2, 2))
    _core._vg_standardized_nll_grad_hess(
        VG_SHAPE.ctypes.data,
        Y.ctypes.data,
        value.ctypes.data,
        grad.ctypes.data,
        hess.ctypes.data,
        Y.size,
    )

    assert mass == pytest.approx(1.0, abs=2e-8)
    assert mean == pytest.approx(0.0, abs=2e-8)
    assert variance == pytest.approx(1.0, abs=2e-8)
    assert _core._vg_standardized_nll(
        VG_SHAPE.ctypes.data, Y.ctypes.data, Y.size
    ) == pytest.approx(-standardized_vg_logpdf(Y, *VG_SHAPE).sum(), rel=2e-11)
    np.testing.assert_allclose(value[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(grad, expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(hess, expected[2], rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize(
    "rho,shape_lambda",
    [(-0.90, VG_LAMBDA_MIN), (0.0, 3.0), (0.90, VG_LAMBDA_MAX)],
)
def test_native_variance_gamma_density_matches_reference_in_tails(
    rho: float, shape_lambda: float
) -> None:
    values = np.array([-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0])
    params = np.array([rho, shape_lambda])
    actual = _core._vg_standardized_nll(
        params.ctypes.data, values.ctypes.data, values.size
    )
    expected = -standardized_vg_logpdf(values, rho, shape_lambda).sum()

    assert actual == pytest.approx(expected, rel=3e-10, abs=3e-10)


def test_variance_gamma_large_argument_shape_curvature() -> None:
    params = np.array([-0.9949103835917059, 47.417166676811])
    values = np.array([-1.9715420758466602])
    nll, gradient, hessian = np.empty(1), np.empty(2), np.empty((2, 2))
    _core._vg_standardized_nll_grad_hess(
        params.ctypes.data,
        values.ctypes.data,
        nll.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
        values.size,
    )
    # Independent MP70 differentiation; retain the existing VG density tolerances.
    np.testing.assert_allclose(nll[0], 2.809912968217987, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(
        gradient,
        [0.00016476975930620224, 0.00028445018668820776],
        rtol=3e-7,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        hessian,
        [
            [0.032751980335175346, -6.21370489040693e-7],
            [-6.21370489040693e-7, -4.485173603012623e-6],
        ],
        rtol=3e-6,
        atol=3e-7,
    )


def test_variance_gamma_center_has_finite_observed_hessian() -> None:
    params = np.array([0.0, 2.8])
    values = np.array([0.0])
    nll = np.empty(1)
    grad = np.empty(2)
    hess = np.empty((2, 2))

    _core._vg_standardized_nll_grad_hess(
        params.ctypes.data,
        values.ctypes.data,
        nll.ctypes.data,
        grad.ctypes.data,
        hess.ctypes.data,
        values.size,
    )

    assert nll[0] == pytest.approx(
        -standardized_vg_logpdf(values, *params)[0], rel=2e-11
    )
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))










def test_variance_gamma_near_center_derivatives_remain_finite() -> None:
    params = np.array([0.0, 3.0])
    values = np.array([-1e-8, 0.0, 1e-10, 1e-8])
    nll = np.empty(1)
    grad = np.empty(2)
    hess = np.empty((2, 2))

    _core._vg_standardized_nll_grad_hess(
        params.ctypes.data,
        values.ctypes.data,
        nll.ctypes.data,
        grad.ctypes.data,
        hess.ctypes.data,
        values.size,
    )

    assert nll[0] == pytest.approx(
        -standardized_vg_logpdf(values, *params).sum(), rel=2e-11
    )
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))


@pytest.mark.parametrize(
    "mean_kind,volatility_kind",
    [
        ("none", "garch"),
        ("none", "gjr"),
        ("none", "egarch"),
        ("linear", "constant"),
        ("linear", "garch"),
        ("linear", "gjr"),
        ("linear", "egarch"),
        ("arma", "constant"),
        ("arma", "garch"),
        ("arma", "gjr"),
        ("arma", "egarch"),
    ],
)
def test_all_shared_gh_tensor_cores_match_ad(
    mean_kind: str, volatility_kind: str
) -> None:
    theta, p_vol, q_vol = _surface_theta(mean_kind, volatility_kind)
    actual = _native_surface(theta, mean_kind, volatility_kind, p_vol, q_vol)
    expected = gh_surface_value_grad_hess(
        theta,
        Y,
        mean_kind=mean_kind,
        p_mean=1 if mean_kind == "arma" else 0,
        q_mean=1 if mean_kind == "arma" else 0,
        features=FEATURES if mean_kind == "linear" else None,
        volatility_kind=volatility_kind,
        p_vol=p_vol,
        q_vol=q_vol,
        fixed_lambda=None, initial_variance=float(np.mean(Y*Y)))

    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize(
    "mean_kind,volatility_kind",
    [
        ("none", "garch"),
        ("none", "gjr"),
        ("none", "egarch"),
        ("linear", "constant"),
        ("linear", "garch"),
        ("linear", "gjr"),
        ("linear", "egarch"),
        ("arma", "constant"),
        ("arma", "garch"),
        ("arma", "gjr"),
        ("arma", "egarch"),
    ],
)
def test_all_shared_variance_gamma_tensor_cores_match_ad(
    mean_kind: str, volatility_kind: str
) -> None:
    theta, p_vol, q_vol = _surface_theta(mean_kind, volatility_kind, VG_SHAPE)
    actual = _native_surface(theta, mean_kind, volatility_kind, p_vol, q_vol, np.inf)
    expected = vg_surface_value_grad_hess(
        theta,
        Y,
        mean_kind=mean_kind,
        p_mean=1 if mean_kind == "arma" else 0,
        q_mean=1 if mean_kind == "arma" else 0,
        features=FEATURES if mean_kind == "linear" else None,
        volatility_kind=volatility_kind,
        p_vol=p_vol,
        q_vol=q_vol, initial_variance=float(np.mean(Y*Y)))

    np.testing.assert_allclose(actual[0], expected[0], rtol=4e-9, atol=4e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=4e-7, atol=4e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=4e-6, atol=4e-7)


def test_nig_garch_surface_matches_ad() -> None:
    theta, p, q = _surface_theta("none", "garch")
    theta = theta[:-1]
    actual = _native_surface(theta, "none", "garch", p, q, NIG_LAMBDA)
    expected = gh_surface_value_grad_hess(
        theta,
        Y,
        volatility_kind="garch",
        p_vol=p,
        q_vol=q,
        fixed_lambda=NIG_LAMBDA, initial_variance=float(np.mean(Y*Y)))

    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-6, atol=3e-7)


def test_hyperbolic_garch_surface_matches_ad() -> None:
    theta, p, q = _surface_theta("none", "garch")
    theta = theta[:-1]
    actual = _native_surface(theta, "none", "garch", p, q, HYPERBOLIC_LAMBDA)
    expected = gh_surface_value_grad_hess(
        theta,
        Y,
        volatility_kind="garch",
        p_vol=p,
        q_vol=q,
        fixed_lambda=HYPERBOLIC_LAMBDA, initial_variance=float(np.mean(Y*Y)))

    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-6, atol=3e-7)












































