from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import genhyperbolic

from scivol import (
    ARMA,
    AutoDensity,
    GARCH,
    GeneralizedHyperbolic,
    Hyperbolic,
    NormalInverseGaussian,
    VarianceGamma,
    _core,
)
from scivol._devtools.ad_oracle import (
    gh_standardized_value_grad_hess,
    gh_surface_value_grad_hess,
    vg_standardized_value_grad_hess,
    vg_surface_value_grad_hess,
)
from scivol._gh import (
    HYPERBOLIC_LAMBDA,
    NIG_LAMBDA,
    standardized_gh_logpdf,
    standardized_gh_parameters,
)
from scivol._autoselect import _per_observation_loglik
from scivol._vg import (
    VG_LAMBDA_MAX,
    VG_LAMBDA_MIN,
    standardized_vg_logpdf,
    standardized_vg_rvs,
)
from scivol._kernels.gh_transforms import (
    jacobian_garch_gh,
    log_hessian_garch_gh,
    pack_garch_gh,
    unpack_garch_gh,
)


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


def _native_surface(
    theta: np.ndarray,
    mean_kind: str,
    volatility_kind: str,
    p_vol: int,
    q_vol: int,
    fixed_lambda: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    native_fixed_lambda = np.nan if fixed_lambda is None else fixed_lambda
    resid = np.empty_like(Y)
    sigma2 = np.empty_like(Y)
    grad = np.empty(theta.size)
    hess = np.empty((theta.size, theta.size))

    if mean_kind == "none":
        resid[:] = Y
        sigma2[0] = np.mean(Y * Y)
        stem = {
            "garch": "garch",
            "gjr": "gjr_garch",
            "egarch": "egarch",
        }[volatility_kind]
        nll = getattr(_core, f"_{stem}_gh_nll_pq")(
            theta.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            Y.size,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        getattr(_core, f"_{stem}_gh_nll_grad_pq")(
            theta.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            grad.ctypes.data,
            Y.size,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        getattr(_core, f"_{stem}_gh_nll_hess_pq")(
            theta.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            hess.ctypes.data,
            Y.size,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        return float(nll), grad, hess

    if mean_kind == "linear":
        n_mean = FEATURES.shape[1]
        sigma2[0] = theta[n_mean] if volatility_kind == "constant" else np.mean(Y * Y)
        stem = {
            "constant": "linear_mean_garch",
            "garch": "linear_mean_garch",
            "gjr": "linear_mean_gjr_garch",
            "egarch": "linear_mean_egarch",
        }[volatility_kind]
        nll = getattr(_core, f"_{stem}_nll_pq_gh")(
            theta.ctypes.data,
            Y.ctypes.data,
            FEATURES.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            Y.size,
            n_mean,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        getattr(_core, f"_{stem}_nll_grad_pq_gh")(
            theta.ctypes.data,
            Y.ctypes.data,
            FEATURES.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            grad.ctypes.data,
            Y.size,
            n_mean,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        getattr(_core, f"_{stem}_hess_pq_gh")(
            theta.ctypes.data,
            Y.ctypes.data,
            FEATURES.ctypes.data,
            resid.ctypes.data,
            sigma2.ctypes.data,
            hess.ctypes.data,
            Y.size,
            n_mean,
            p_vol,
            q_vol,
            native_fixed_lambda,
        )
        return float(nll), grad, hess

    max_lag = max(1, 1, p_vol, q_vol)
    e0 = np.zeros(max_lag)
    h0 = np.full(max_lag, np.mean(Y * Y))
    kind = {"constant": 0, "garch": 0, "gjr": 1, "egarch": 2}[volatility_kind]
    nll = _core._arma_vol_gh_nll_pq(
        theta.ctypes.data,
        Y.ctypes.data,
        resid.ctypes.data,
        sigma2.ctypes.data,
        e0.ctypes.data,
        h0.ctypes.data,
        Y.size,
        1,
        1,
        p_vol,
        q_vol,
        kind,
        native_fixed_lambda,
    )
    _core._arma_vol_gh_nll_grad_pq(
        theta.ctypes.data,
        Y.ctypes.data,
        resid.ctypes.data,
        sigma2.ctypes.data,
        e0.ctypes.data,
        h0.ctypes.data,
        grad.ctypes.data,
        Y.size,
        1,
        1,
        p_vol,
        q_vol,
        kind,
        native_fixed_lambda,
    )
    _core._arma_vol_gh_nll_hess_pq(
        theta.ctypes.data,
        Y.ctypes.data,
        resid.ctypes.data,
        sigma2.ctypes.data,
        e0.ctypes.data,
        h0.ctypes.data,
        hess.ctypes.data,
        Y.size,
        1,
        1,
        p_vol,
        q_vol,
        kind,
        native_fixed_lambda,
    )
    return float(nll), grad, hess


def test_density_classes_are_distinct_public_components() -> None:
    gh = GeneralizedHyperbolic()
    hyp = Hyperbolic()
    nig = NormalInverseGaussian()
    vg = VarianceGamma()

    assert gh.signature == "GeneralizedHyperbolic"
    assert gh.n_params == 3
    assert hyp.signature == "Hyperbolic"
    assert hyp.n_params == 2
    assert nig.signature == "NormalInverseGaussian"
    assert nig.n_params == 2
    assert vg.signature == "VarianceGamma"
    assert vg.n_params == 2
    np.testing.assert_array_equal(gh.default_start(Y), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(hyp.default_start(Y), [0.0, 1.0])
    np.testing.assert_array_equal(nig.default_start(Y), [0.0, 1.0])
    np.testing.assert_array_equal(vg.default_start(Y), [0.0, 3.0])


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


def test_python_vg_logpdf_retains_finite_bessel_product_near_center():
    mp=pytest.importorskip("mpmath")
    values=np.array([[1e-8,-1e-8],[0.,1e-200]])
    actual=standardized_vg_logpdf(values,0.,50.)
    expected=np.empty_like(actual)
    with mp.workdps(80):
        shape=mp.mpf(50); order=shape-mp.mpf(".5")
        constant=mp.log(shape)/2+mp.log(2)-mp.log(2*mp.pi)/2-mp.loggamma(shape)
        for index in np.ndindex(values.shape):
            y=mp.sqrt(shape)*abs(mp.mpf(values[index]))
            kernel=((order-1)*mp.log(2)+mp.loggamma(order)-order*mp.log(2)
                    if y==0 else order*(mp.log(y)-mp.log(2)/2)+mp.log(mp.besselk(order,mp.sqrt(2)*y)))
            expected[index]=float(constant+kernel)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-13)
    np.testing.assert_array_equal(standardized_vg_logpdf(values[:,::-1],0.,50.),actual[:,::-1])
    assert standardized_vg_logpdf(np.array([]),0.,50.).size==0
    assert np.isneginf(standardized_vg_logpdf(np.array([np.inf]),0.,50.)[0])
    assert np.isnan(standardized_vg_logpdf(np.array([np.nan]),0.,50.)[0])


@pytest.mark.parametrize("backend", ["python", "legacy", "native-F", "native-FG", "native-FGH"])
def test_variance_gamma_lower_shape_near_center_matches_exact_density(backend):
    mp = pytest.importorskip("mpmath")
    # The canonical native domain extends below the historical Python fit bound.
    params = np.array([0.0, 1.5000000000000018])
    values = np.array([5.7735026918962544e-6])

    def exact_nll(residual, variance):
        with mp.workdps(70):
            shape = mp.mpf(float(params[1]))
            standardized = mp.mpf(float(residual)) / mp.sqrt(float(variance))
            centered = mp.sqrt(shape) * standardized
            order = shape - mp.mpf("0.5")
            logpdf = (mp.log(shape)/2 + mp.log(2) - mp.log(2*mp.pi)/2
                      - mp.loggamma(shape) + order*(mp.log(centered)-mp.log(2)/2)
                      + mp.log(mp.besselk(order, mp.sqrt(2)*centered)))
            return float(mp.log(float(variance))/2-logpdf)

    expected = exact_nll(values[0], 1.0)
    if backend == "python":
        actual = -standardized_vg_logpdf(values, *params)[0]
    elif backend == "legacy":
        actual = _core._vg_standardized_nll(params.ctypes.data, values.ctypes.data, 1)
    else:
        from scivol import ARX
        from scivol._joint_mean_fit import JointMeanModel

        data = np.repeat(values, 2)
        model = JointMeanModel(ARX(0) + GARCH(1, 1) + VarianceGamma(), data, None)
        theta = np.array([0., 1., 0., 0., *params])
        actual = model.evaluate(theta, {"native-F": 1, "native-FG": 3, "native-FGH": 7}[backend])[0]
        expected += exact_nll(values[0], model.h0)
    assert actual == pytest.approx(expected, rel=3e-10, abs=3e-10)


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
        fixed_lambda=None,
    )

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
        q_vol=q_vol,
    )

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
        fixed_lambda=NIG_LAMBDA,
    )

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
        fixed_lambda=HYPERBOLIC_LAMBDA,
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("estimate_lambda", [True, False])
def test_garch_gh_log_transform_matches_ad_chain_rule(
    estimate_lambda: bool,
) -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    theta, p, q = _surface_theta("none", "garch")
    if not estimate_lambda:
        theta = theta[:-1]
    value, grad, hess = _native_surface(
        theta,
        "none",
        "garch",
        p,
        q,
        None if estimate_lambda else NIG_LAMBDA,
    )
    z = unpack_garch_gh(theta, p, q, estimate_lambda)

    theta_j = jnp.asarray(theta)
    grad_j = jnp.asarray(grad)
    hess_j = jnp.asarray(hess)

    def pack(z_j):
        omega = jnp.log1p(jnp.exp(z_j[0]))
        logits = jnp.concatenate((z_j[1 : 1 + p + q], jnp.zeros(1)))
        dynamic = jax.nn.softmax(logits)[:-1]
        offset = 1 + p + q
        rho = 0.995 * jnp.tanh(z_j[offset])
        zeta = 0.1 + 24.9 * jax.nn.sigmoid(z_j[offset + 1])
        shapes = [rho, zeta]
        if estimate_lambda:
            shapes.append(-6.0 + 12.0 * jax.nn.sigmoid(z_j[offset + 2]))
        return jnp.concatenate((jnp.array([omega]), dynamic, jnp.stack(shapes)))

    def local_objective(z_j):
        delta = pack(z_j) - theta_j
        return value + grad_j @ delta + 0.5 * delta @ hess_j @ delta

    expected_grad = np.asarray(jax.grad(local_objective)(jnp.asarray(z)))
    expected_hess = np.asarray(jax.hessian(local_objective)(jnp.asarray(z)))
    actual_grad = jacobian_garch_gh(theta, p, q, estimate_lambda).T @ grad
    actual_hess = log_hessian_garch_gh(theta, grad, hess, p, q, estimate_lambda)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual_hess, expected_hess, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(pack_garch_gh(z, p, q, estimate_lambda), theta)


def test_garch_variance_gamma_log_transform_matches_ad_chain_rule() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    theta, p, q = _surface_theta("none", "garch", VG_SHAPE)
    value, grad, hess = _native_surface(theta, "none", "garch", p, q, np.inf)
    z = unpack_garch_gh(theta, p, q, False, VG_LAMBDA_MIN, VG_LAMBDA_MAX)
    theta_j = jnp.asarray(theta)
    grad_j = jnp.asarray(grad)
    hess_j = jnp.asarray(hess)

    def pack(z_j):
        omega = jnp.log1p(jnp.exp(z_j[0]))
        logits = jnp.concatenate((z_j[1 : 1 + p + q], jnp.zeros(1)))
        dynamic = jax.nn.softmax(logits)[:-1]
        offset = 1 + p + q
        rho = 0.995 * jnp.tanh(z_j[offset])
        shape_lambda = VG_LAMBDA_MIN + (VG_LAMBDA_MAX - VG_LAMBDA_MIN) * jax.nn.sigmoid(
            z_j[offset + 1]
        )
        return jnp.concatenate(
            (jnp.array([omega]), dynamic, jnp.array([rho, shape_lambda]))
        )

    def local_objective(z_j):
        delta = pack(z_j) - theta_j
        return value + grad_j @ delta + 0.5 * delta @ hess_j @ delta

    expected_grad = np.asarray(jax.grad(local_objective)(jnp.asarray(z)))
    expected_hess = np.asarray(jax.hessian(local_objective)(jnp.asarray(z)))
    actual_grad = (
        jacobian_garch_gh(theta, p, q, False, VG_LAMBDA_MIN, VG_LAMBDA_MAX).T @ grad
    )
    actual_hess = log_hessian_garch_gh(
        theta, grad, hess, p, q, False, VG_LAMBDA_MIN, VG_LAMBDA_MAX
    )
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual_hess, expected_hess, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(
        pack_garch_gh(z, p, q, False, VG_LAMBDA_MIN, VG_LAMBDA_MAX),
        theta,
    )


def _assert_retained_boundary_fit(result, reference_likelihood):
    """A historical solver success is not a strict stationarity certificate."""
    info = result.fit_info
    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood >= reference_likelihood-1e-8
    assert info.maximum_constraint_violation <= 1e-10
    assert info.objective_parity_error <= 1e-8
    if result.converged:
        assert info.kkt_residual <= info.kkt_tolerance
    else:
        assert info.kkt_residual > info.kkt_tolerance
        assert any("Strict credibility check failed" in message for message in info.warnings)
        assert any("Standard errors are unavailable" in message for message in info.warnings)
        assert result.std_errors is None


def test_public_nig_workflow_keeps_lambda_fixed() -> None:
    rng = np.random.default_rng(72)
    data = 0.01 * rng.standard_t(8, 180)
    result = (ARMA(1, 1) + GARCH(1, 1) + NormalInverseGaussian()).fit(data)
    state = result.filter()
    forecast = result.forecast(3)

    _assert_retained_boundary_fit(result, 542.4867630800777)
    assert result.dist_params.shape_lambda == NIG_LAMBDA
    assert result.dist_params.shape_lambda_fixed
    assert result.dist_params.to_array().shape == (2,)
    assert state.shape_lambda == NIG_LAMBDA
    assert np.all(np.isfinite(forecast.quantile(0.05)))


def test_public_hyperbolic_workflow_keeps_lambda_fixed() -> None:
    rng = np.random.default_rng(73)
    data = 0.01 * rng.standard_t(8, 180)
    result = (ARMA(1, 1) + GARCH(1, 1) + Hyperbolic()).fit(data)
    state = result.filter()
    forecast = result.forecast(3)

    _assert_retained_boundary_fit(result, 554.9679076178344)
    assert result.dist_params.shape_lambda == HYPERBOLIC_LAMBDA
    assert result.dist_params.shape_lambda_fixed
    assert result.dist_params.to_array().shape == (2,)
    assert state.shape_lambda == HYPERBOLIC_LAMBDA
    assert np.all(np.isfinite(forecast.quantile(0.05)))


def test_public_variance_gamma_workflow() -> None:
    rng = np.random.default_rng(74)
    innovations = standardized_vg_rvs(rng, 220, 0.2, 3.0)
    data = 0.01 * innovations
    spec = ARMA(1, 1) + GARCH(1, 1) + VarianceGamma()
    result = spec.fit(data)
    state = result.filter()
    score = spec.score(data, result.params)
    hessian = spec.hessian(data, result.params)
    forecast = result.forecast(3)
    simulated = spec.simulate(40, result.params, burn=20, seed=75)

    _assert_retained_boundary_fit(result, 693.0613023072463)
    assert result.dist_params.rho is not None
    assert result.dist_params.shape_lambda is not None
    assert not result.dist_params.shape_lambda_fixed
    assert result.dist_params.to_array().shape == (2,)
    assert state.zeta is None
    assert state.shape_lambda == pytest.approx(result.dist_params.shape_lambda)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hessian))
    assert np.all(np.isfinite(forecast.quantile(0.05)))
    assert simulated.data.shape == (40,)


def test_auto_density_and_oos_log_score_accept_variance_gamma() -> None:
    rng = np.random.default_rng(82)
    data = 0.01 * standardized_vg_rvs(rng, 180, 0.15, 3.0)
    result = (GARCH(1, 1) + AutoDensity(candidates=["VarianceGamma"])).fit(
        data, n_jobs=1
    )
    state = result.filter()
    expected = standardized_vg_logpdf(
        state.std_resid,
        result.dist_params.rho,
        result.dist_params.shape_lambda,
    ) - 0.5 * np.log(state.sigma2)
    actual = _per_observation_loglik(
        "VarianceGamma",
        state.residuals,
        state.sigma2,
        nu=None,
        lam=None,
        rho=result.dist_params.rho,
        shape_lambda=result.dist_params.shape_lambda,
    )

    assert result.density.signature == "VarianceGamma"
    assert result.selection.best is not None
    assert result.selection.best.density == "VarianceGamma"
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_auto_density_accepts_nig_as_an_explicit_candidate() -> None:
    rng = np.random.default_rng(81)
    data = 0.01 * rng.standard_t(8, 160)
    result = (GARCH(1, 1) + AutoDensity(candidates=["NormalInverseGaussian"])).fit(
        data, n_jobs=1
    )

    assert result.density.signature == "NormalInverseGaussian"
    assert result.selection.best is not None
    assert result.selection.best.density == "NormalInverseGaussian"


def test_oos_log_score_uses_the_standardized_gh_density() -> None:
    resid = np.array([-0.2, 0.1, 0.4])
    sigma2 = np.array([0.8, 1.2, 0.9])
    rho, zeta, shape_lambda = 0.2, 1.5, -0.4
    z = resid / np.sqrt(sigma2)
    expected = standardized_gh_logpdf(z, rho, zeta, shape_lambda) - 0.5 * np.log(sigma2)

    actual = _per_observation_loglik(
        "GeneralizedHyperbolic",
        resid,
        sigma2,
        nu=None,
        lam=None,
        rho=rho,
        zeta=zeta,
        shape_lambda=shape_lambda,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_qmle_rejects_unshipped_gh_robust_covariance() -> None:
    with pytest.raises(NotImplementedError, match="QMLE is not shipped"):
        (GARCH(1, 1) + GeneralizedHyperbolic()).fit(Y, method="qmle")


def test_gh_diagnostics_and_serialization_report_shape_parameters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    params = np.array([0.12, 0.08, 0.72, 0.2, 1.5, -0.4])
    result = (GARCH(1, 1) + GeneralizedHyperbolic()).fix(Y, params)
    diagnostics = result.diagnostic_tests(n_cells=3, lags=1, print_results=True)
    serialized = result.to_dict()["dist_params"]

    assert diagnostics["dist_params"]["rho"] == pytest.approx(0.2)
    assert "rho=0.2000" in capsys.readouterr().out
    assert serialized["rho"] == pytest.approx(0.2)
    assert serialized["zeta"] == pytest.approx(1.5)
    assert serialized["lambda"] == pytest.approx(-0.4)
