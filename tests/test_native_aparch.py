from __future__ import annotations

import ctypes
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest

from scivol import (
    APARCH,
    ARMA,
    ARX,
    GED,
    GeneralizedHyperbolic,
    HARX,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
    _native_core,
)
from scivol._devtools.ad_oracle import (
    aparch_power_moment_value_grad_hess,
    aparch_value_grad_hess,
    arma_aparch_value_grad_hess,
    gh_surface_value_grad_hess,
    native_aparch_coordinate_chain_rule_value_grad_hess,
    vg_surface_value_grad_hess,
)
from scivol.roles import Role


DENSITIES = [
    ("normal", 0, []),
    ("studentt", 1, [8.0]),
    ("skewt", 2, [8.0, 0.12]),
    ("ged", 3, [1.6]),
    ("gh", 4, [0.1, 1.3, 0.4]),
    ("nig", 5, [0.1, 1.3]),
    ("hyperbolic", 6, [0.1, 1.3]),
    ("vg", 7, [0.1, 3.0]),
]
VOLATILITY = np.array([0.08, 0.07, 0.15, 0.82, 1.7])
MEAN = np.array([0.03, 0.12, -0.08])
HIGH_POWER_MOMENTS = [
    ("gh", 4, [.9950183009315131, 67.22371758260331,
               .22624282188934747, 24.9999999997434, 5.854544981566564]),
    ("vg", 7, [.9999999999999999, 71.54184776629806,
               .2638396981927625, 49.99999999733749]),
]
RANGE_LIMIT_GH = ("gh", 4, [.9999998652967536, 215.4632640594723,
                           .22911577118769738, 24.999999999999993, 5.999999998267437])


def _workspace(size: int) -> np.ndarray:
    return np.empty(max(size, 1), dtype=np.uint8)


def _theta(shape: list[float], *, joint: bool) -> np.ndarray:
    blocks = [MEAN, VOLATILITY, np.asarray(shape)] if joint else [VOLATILITY, np.asarray(shape)]
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float64)


def _standalone_ad_jet(theta, data, name, *, initial_variance):
    if name == "vg":
        return vg_surface_value_grad_hess(
            theta, data, volatility_kind="aparch", p_vol=1, q_vol=1,
            initial_variance=initial_variance,
        )
    if name in {"gh", "nig", "hyperbolic"}:
        return gh_surface_value_grad_hess(
            theta, data, volatility_kind="aparch", p_vol=1, q_vol=1,
            fixed_lambda={"nig": -0.5, "hyperbolic": 1.0}.get(name),
            initial_variance=initial_variance,
        )
    return aparch_value_grad_hess(theta, data, 1, 1, name,
                                 initial_variance=initial_variance)


def _evaluate(
    parameters: np.ndarray,
    data: np.ndarray,
    density: int,
    request: int = 7,
    *,
    transformed: bool = False,
    orders: tuple[int, int, int, int] | None = None,
    volatility_orders: tuple[int, int] = (1, 1),
    initial_variance: float | None = None,
) -> tuple[int, int, float, np.ndarray, np.ndarray]:
    p_ar, q_ma, p_vol, q_vol = orders or (0, 0, *volatility_orders)
    joint = orders is not None
    initial = float(np.mean(data * data)) if initial_variance is None else initial_variance
    value = np.full(1, np.nan)
    gradient = np.full(parameters.size, np.nan)
    hessian = np.full((parameters.size, parameters.size), np.nan)
    if joint:
        size = _native_core._arma_aparch_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, int(transformed), request
        )
        workspace = _workspace(size)
        status, valid = _native_core._arma_aparch_jet(
            parameters.ctypes.data,
            parameters.size,
            data.ctypes.data,
            data.size,
            initial,
            p_ar,
            q_ma,
            p_vol,
            q_vol,
            density,
            int(transformed),
            request,
            workspace.ctypes.data,
            workspace.nbytes,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        size = _native_core._aparch_workspace_bytes(
            p_vol, q_vol, density, int(transformed), request
        )
        workspace = _workspace(size)
        status, valid = _native_core._aparch_jet(
            parameters.ctypes.data,
            parameters.size,
            data.ctypes.data,
            data.size,
            initial,
            p_vol,
            q_vol,
            density,
            int(transformed),
            request,
            workspace.ctypes.data,
            workspace.nbytes,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    return status, valid, float(value[0]), gradient, hessian


@pytest.mark.parametrize("point_index,transformed", [
    (0, False), (1, False), (2, False), (3, False), (4, False), (0, True), (1, True),
])
def test_joint_aparch_cancellation_matches_independent_anchored_jet(point_index, transformed):
    from scivol._native_fit import _describe_model, _materialize_native_aparch_state

    fixture = json.loads(Path(__file__).with_name("arma_aparch_cancellation_points.json").read_text())
    point = fixture["points"][point_index]
    if transformed:
        point = point["transformed"]
    data, theta = np.array(fixture["data"]), np.array(point["theta"])
    orders, h0 = tuple(fixture["orders"]), fixture["initial_variance"]
    coordinates = np.array(point["coordinates"]) if transformed else theta
    if transformed:
        np.testing.assert_array_equal(_map(coordinates, 0, orders=orders, forward=True), theta)
    expected = point["value"], np.array(point["gradient"]), np.array(point["hessian"])
    for request in range(1, 8):
        status, valid, *actual = _evaluate(coordinates, data, 0, request, orders=orders,
                                         initial_variance=h0, transformed=transformed)
        assert (status, valid) == (0, request)
        runtime = _forced_runtime(coordinates, data, 0, orders=orders, request=request,
                                  initial_variance=h0, transformed=transformed)
        for bit, fixed, general, reference in zip((1, 2, 4), actual, runtime, expected):
            for output in (fixed, general):
                if not request & bit:
                    assert np.isnan(output).all()
                elif bit == 1:
                    assert abs(output-reference) <= 1e-9
                else:
                    np.testing.assert_allclose(output, reference, rtol=3e-10, atol=2e-11)
    model = _describe_model(ARMA(1, 1) + APARCH(1, 1) + Normal())
    residuals, variance = _materialize_native_aparch_state(model, data, theta)
    np.testing.assert_allclose(residuals[177], point["residual177"], rtol=3e-10, atol=0)
    if point_index < 2:
        assert residuals[177] == point["residual177"]
    filtered_value = .5*np.sum(np.log(2*np.pi) + np.log(variance[1:])
                              + residuals[1:]**2/variance[1:])
    assert abs(filtered_value-point["value"]) <= 1e-9


@pytest.mark.parametrize("small,gamma", [(1e-170, 0.15), (0.1, np.nextafter(1.0, 0.0))])
def test_fixed_aparch_shock_derivatives_avoid_cancelled_or_squared_small_bases(small, gamma):
    data = np.array([0.7, small, -0.4, 0.2, -0.5])
    theta = np.array([0.2, 0.07, gamma, 0.6, 1.7])
    expected = aparch_value_grad_hess(theta, data, 1, 1, dist="normal",
                                    initial_variance=float(np.mean(data * data)))
    for request in range(1, 8):
        status, valid, value, gradient, hessian = _evaluate(theta, data, 0, request)
        assert (status, valid) == (0, request)
        for bit, actual, reference in zip((1, 2, 4), (value, gradient, hessian), expected):
            if request & bit:
                np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (2, 2, 4, 3)])
@pytest.mark.parametrize("delta", [.5, 1., np.nextafter(1., 2.), 2., 2.5])
def test_joint_aparch_tiny_shock_preserves_complete_curvature(orders, delta):
    ar, ma, p, q = orders
    data = np.array([1., -.4, .2, .1, 1e-180, .45, -.3, .6, .8, -.7, .25])
    data[max(orders)] = 1e-180
    theta = np.r_[0., np.zeros(ar), np.full(ma, -.2/ma),
                  .13, np.full(p, .08/p), np.full(p, .3), np.full(q, .78/q), delta]
    for transformed in (False, True):
        point = _map(theta, 0, orders=orders) if transformed else theta
        mapped = _map(point, 0, orders=orders, forward=True) if transformed else theta
        expected = arma_aparch_value_grad_hess(
            mapped, data, *orders, initial_variance=float(np.mean(data * data)))
        if transformed:
            expected = native_aparch_coordinate_chain_rule_value_grad_hess(
                point, mapped, *expected, p, q, "normal", p_ar=ar, q_ma=ma)
        assert all(np.isfinite(part).all() for part in map(np.asarray, expected))
        assert np.max(np.abs(expected[2][:1+ar+ma, 1+ar+ma:])) > .01
        for request in range(1, 8):
            status, valid, *outputs = _evaluate(
                point, data, 0, request, orders=orders, transformed=transformed)
            assert (status, valid) == (0, request)
            runtime = _forced_runtime(point, data, 0, orders=orders,
                                      request=request, transformed=transformed)
            for bit, actual, other, reference in zip((1, 2, 4), outputs, runtime, expected):
                if request & bit:
                    assert np.isfinite(actual).all()
                    np.testing.assert_allclose(actual, reference, rtol=3e-10, atol=2e-11)
                    np.testing.assert_allclose(other, reference, rtol=3e-10, atol=2e-11)
                else:
                    assert np.isnan(actual).all() and np.isnan(other).all()


@pytest.mark.parametrize("code,shape", [(1, [8.]), (2, [8., .12]), (3, [1.6])])
def test_aparch_non_normal_log_variance_curvature_range(code, shape):
    from scivol._devtools.ad_oracle import _jax_modules
    from scivol._devtools.joint_mean_oracle import _density_nll

    jax, jnp, jsp = _jax_modules()
    data = np.array([1e100, 1e100, -1e100])
    theta = np.array([1e200, 0., 0., .5, 2., *shape])

    def objective(beta):
        # Fixed delta=2 and alpha=0 give these exact h/h0 values. Differentiate
        # before restoring the constant log(h0), avoiding extreme AD intermediates.
        variance = jnp.array([1., 1.+beta, 1.+beta+beta*beta])
        z = jnp.array([1., 1., -1.]) / jnp.sqrt(variance)
        return .5*jnp.log(variance).sum() + _density_nll(
            jnp, jsp, z, jnp.array(shape), code, np.zeros(3, dtype=bool))

    expected = float(jax.hessian(objective)(jnp.array(.5)))
    status, valid, _, _, hessian = _evaluate(theta, data, code, 4, initial_variance=1e200)
    assert (status, valid) == (0, 4)
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian[3, 3], expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("log_power", [-80., 80.])
def test_aparch_ged_log_density_preserves_variance_representability(log_power):
    data = np.array([1., 0.])
    theta = np.array([np.exp(log_power), 0., 0., 0., .2, 2.])
    # h = power**(2/delta) is unrepresentable even though power and log(h) are finite.
    for mask in range(1, 8):
        status, valid, *_ = _evaluate(theta, data, 3, mask, initial_variance=1.)
        assert status != 0 and valid == 0


def _map(
    theta: np.ndarray,
    density: int,
    *,
    orders: tuple[int, int, int, int] | None = None,
    volatility_orders: tuple[int, int] = (1, 1),
    forward: bool = False,
) -> np.ndarray:
    destination = np.empty_like(theta)
    if orders is None:
        p_vol, q_vol = volatility_orders
        workspace = _workspace(
            _native_core._aparch_map_workspace_bytes(p_vol, q_vol, density)
        )
        _native_core._map_aparch(
            theta.ctypes.data,
            destination.ctypes.data,
            p_vol,
            q_vol,
            density,
            forward,
            workspace.ctypes.data,
            workspace.nbytes,
        )
    else:
        p_ar, q_ma, p_vol, q_vol = orders
        workspace = _workspace(
            _native_core._arma_aparch_map_workspace_bytes(
                p_ar, q_ma, p_vol, q_vol, density
            )
        )
        _native_core._map_arma_aparch(
            theta.ctypes.data,
            destination.ctypes.data,
            p_ar,
            q_ma,
            p_vol,
            q_vol,
            density,
            forward,
            workspace.ctypes.data,
            workspace.nbytes,
        )
    return destination


def test_aparch_large_power_has_representable_complete_hessian():
    # Symbolic derivatives of the audit's three-observation zero-residual NLL.
    sympy = pytest.importorskip("sympy")
    mp = pytest.importorskip("mpmath")
    omega, beta, delta, seed = sympy.symbols("omega beta delta seed", positive=True)
    first = omega + beta * seed ** (delta / 2)
    second = omega + beta * first
    objective = (sympy.log(first) + sympy.log(second)) / delta
    objective += sympy.log(seed) / 2 + sympy.Rational(3, 2) * sympy.log(2 * sympy.pi)
    variables = (omega, beta, delta)
    reference = sympy.lambdify((*variables, seed), (
        objective,
        sympy.Matrix([sympy.diff(objective, v) for v in variables]),
        sympy.hessian(objective, variables),
    ), "mpmath")
    with mp.workdps(80):
        value, gradient, hessian = reference(mp.mpf(1e200), mp.mpf(.5), 2, mp.mpf(1e200))
    expected_gradient, expected_hessian = np.zeros(5), np.zeros((5, 5))
    indices = [0, 3, 4]
    expected_gradient[indices] = np.asarray(gradient, dtype=float).ravel()
    expected_hessian[np.ix_(indices, indices)] = np.asarray(hessian.tolist(), dtype=float)
    expected = (float(value), expected_gradient, expected_hessian)
    theta, data = np.array([1e200, 0., 0., .5, 2.]), np.zeros(3)
    for mask in range(1, 8):
        status, valid, *actual = _evaluate(theta, data, 0, mask, initial_variance=1e200)
        assert (status, valid) == (0, mask)
        for bit, output, target in zip((1, 2, 4), actual, expected):
            if mask & bit:
                assert np.isfinite(output).all()
                np.testing.assert_allclose(output, target, rtol=2e-12, atol=0)
            else:
                assert np.isnan(output).all()


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
@pytest.mark.parametrize("delta", [0.7, 1.7])
def test_aparch_stationarity_moment_matches_ad(
    name: str, code: int, shape: list[float], delta: float,
) -> None:
    shape_array = np.asarray(shape, dtype=np.float64)
    local = np.r_[0.15, delta, shape_array]
    value = np.empty(1)
    gradient = np.empty(local.size)
    hessian = np.empty((local.size, local.size))
    status = _native_core._aparch_asymmetric_power_moment(
        shape_array.ctypes.data,
        code,
        local[0],
        local[1],
        7,
        value.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
    )
    expected = aparch_power_moment_value_grad_hess(local, name)
    assert status == 0
    np.testing.assert_allclose(value[0], expected[0], rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(gradient, expected[1], rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(hessian, expected[2], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("local", (
    [0.06183922416878056, 2.0379819829328323, 0.12034602234581399, 15.272133090357961],
    [0.140235972517651, 2.236495579602217, 0.2857400734797855, 49.999999999999986],
))
def test_vg_power_moment_retained_shape_derivatives(local):
    # The former log-K/product switch made these feasible moments reject G/H.
    point = np.array(local)
    shape = np.ascontiguousarray(point[2:])
    expected = aparch_power_moment_value_grad_hess(point, "vg")
    for request in range(1, 8):
        value, gradient, hessian = np.full(1, 12345.), np.full(4, 12345.), np.full((4, 4), 12345.)
        status = _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, 7, point[0], point[1], request,
            value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data)
        assert status == 0
        for bit, actual, reference in zip((1, 2, 4), (value[0], gradient, hessian), expected):
            if request & bit:
                np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-8)
            else:
                np.testing.assert_array_equal(actual, np.full_like(actual, 12345.))


@pytest.mark.parametrize("name,code,local", HIGH_POWER_MOMENTS)
def test_high_power_moment_derivatives_avoid_separate_factor_overflow(name, code, local):
    point = np.array(local)
    shape = np.ascontiguousarray(point[2:])
    coarse = aparch_power_moment_value_grad_hess(point, name, quadrature_nodes=256)
    reference = aparch_power_moment_value_grad_hess(point, name, quadrature_nodes=512)
    scale = reference[0]
    assert np.isfinite(scale) and scale > 1e60
    for actual, expected in zip(coarse, reference, strict=True):
        np.testing.assert_allclose(actual / scale, expected / scale, rtol=2e-8, atol=2e-9)
    for mask in range(1, 8):
        f = np.full(1, 12345.)
        g = np.full(point.size, 12345.)
        h = np.full((point.size, point.size), 12345.)
        assert _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, code, point[0], point[1], mask,
            f.ctypes.data, g.ctypes.data, h.ctypes.data) == 0
        for bit, actual, expected in zip((1, 2, 4), (f[0], g, h), reference, strict=True):
            if mask & bit:
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual / scale, expected / scale,
                                           rtol=2e-7, atol=2e-8)
            else:
                np.testing.assert_array_equal(actual, np.full_like(actual, 12345.))


def test_power_moment_range_limit_is_request_specific():
    from scivol._devtools.ad_oracle import _jax_modules, _jax_gh_logpdf, _value_grad_hess

    _, jnp, jsp = _jax_modules()
    point = np.array(RANGE_LIMIT_GH[2])
    references = []
    scale = 1e300
    for count in (512, 1024):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        angle = jnp.pi/2*jnp.asarray((nodes+1)/2)
        magnitude = jnp.tan(angle)
        log_weight = jnp.log(jnp.asarray(weights/2)*jnp.pi/2/jnp.cos(angle)**2)-jnp.log(scale)

        def scaled_moment(theta):
            gamma, power, rho, zeta, lam = theta
            positive = _jax_gh_logpdf(jnp, jsp, magnitude, rho, zeta, lam)
            negative = _jax_gh_logpdf(jnp, jsp, -magnitude, rho, zeta, lam)
            positive += log_weight+power*(jnp.log(magnitude)+jnp.log1p(-gamma))
            negative += log_weight+power*(jnp.log(magnitude)+jnp.log1p(gamma))
            return jnp.sum(jnp.exp(positive)+jnp.exp(negative))

        references.append(_value_grad_hess(scaled_moment, point))
    for coarse, fine in zip(*references, strict=True):
        np.testing.assert_allclose(coarse, fine, rtol=2e-8, atol=2e-9)
    expected = references[-1]
    assert np.isfinite(expected[1]*scale).all()
    assert np.count_nonzero(np.abs(expected[2]) > np.finfo(float).max/scale) == 4
    shape = np.ascontiguousarray(point[2:])
    for mask in range(1, 8):
        f, g, h = np.full(1, 12345.), np.full(5, 12345.), np.full((5, 5), 12345.)
        status = _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, 4, point[0], point[1], mask,
            f.ctypes.data, g.ctypes.data, h.ctypes.data)
        assert (status == 0) == (not mask & 4)
        for bit, actual, reference in zip((1, 2, 4), (f[0], g, h), expected, strict=True):
            if status == 0 and mask & bit:
                np.testing.assert_allclose(actual/scale, reference, rtol=2e-8, atol=2e-9)
            else:
                np.testing.assert_array_equal(actual, np.full_like(actual, 12345.))


@pytest.mark.parametrize("name,code,local", HIGH_POWER_MOMENTS + [RANGE_LIMIT_GH,
    ("skewt", 2, [.25, 3.7, 3.700001, 0.]),
    ("skewt", 2, [.2, 3.7, 3.700001, -.8]),
    ("skewt", 2, [.2, 2.0001, 2.0002, .7])])
@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
@pytest.mark.parametrize("fixed", [False, True])
def test_joint_high_power_constraint_uses_finite_moment_derivatives(name, code, local, p, q, fixed):
    from scivol._joint_mean_fit import JointMeanModel, DENSITIES as JOINT_DENSITIES

    rng = np.random.default_rng(214)
    y, x = rng.normal(size=32), rng.normal(size=(32, 4 if p > 1 else 2))
    mean = HARX([1, 3]) if p > 1 else ARX(1)
    model = JointMeanModel(mean + APARCH(p, q) + JOINT_DENSITIES[code](), y, x, fixed=fixed)
    theta = model.initialize()
    alpha, gamma = model.columns + 1, model.columns + 1 + p
    beta, delta = gamma + p, model.shape_start - 1
    theta[gamma:beta] = local[0]
    theta[beta:delta], theta[delta] = .4 / q, local[1]
    theta[model.shape_start:] = local[2:]
    shape = np.ascontiguousarray(theta[model.shape_start:])
    value, gradient = np.empty(1), np.empty(len(local))
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, code, local[0], local[1], 3,
        value.ctypes.data, gradient.ctypes.data, 0) == 0
    theta[alpha:gamma] = .01 / (p * value[0])

    values, jacobian = model.constraints(theta)
    scalar_values, absent = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, scalar_values)
    assert absent is None and np.isfinite(jacobian).all()
    row = 2 * model.domain[0]
    expected = np.zeros_like(theta)
    expected[alpha:gamma] = -value[0]
    expected[gamma:beta] = -theta[alpha:gamma] * gradient[0]
    expected[beta:delta] = -1.
    expected[delta:] = -theta[alpha:gamma].sum() * gradient[1:]
    np.testing.assert_allclose(jacobian[row], expected, rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(values[row], .59,
                               rtol=2e-10, atol=2e-12)
    assert model.assess(theta, None)[0]


@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
@pytest.mark.parametrize("fixed", [False, True])
def test_joint_constraint_recovers_weighted_derivatives_when_raw_moment_overflows(p, q, fixed):
    from scivol._joint_mean_fit import JointMeanModel, DENSITIES as JOINT_DENSITIES

    rng = np.random.default_rng(215)
    y, x = rng.normal(size=32), rng.normal(size=(32, 4 if p > 1 else 2))
    mean = HARX([1, 3]) if p > 1 else ARX(1)
    model = JointMeanModel(mean + APARCH(p, q) + JOINT_DENSITIES[4](), y, x, fixed=fixed)
    theta = model.initialize()
    alpha, gamma = model.columns+1, model.columns+1+p
    beta, delta = gamma+p, model.shape_start-1
    theta[gamma:beta] = .9999998781130509
    theta[beta:delta], theta[delta] = .4/q, 216.965990611158
    theta[model.shape_start:] = [.22917865877731938, 24.999999999999993, 5.9999999985845776]
    shape = np.ascontiguousarray(theta[model.shape_start:])
    value, gradient = np.empty(1), np.full(5, 12345.)
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 4, theta[gamma], theta[delta], 1, value.ctypes.data, 0, 0) == 0
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 4, theta[gamma], theta[delta], 2, 0, gradient.ctypes.data, 0) != 0
    np.testing.assert_array_equal(gradient, 12345.)
    theta[alpha:gamma] = (.1/p)/value[0]
    values, jacobian = model.constraints(theta)
    scalar, _ = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, scalar)
    # Independent 512/1024-node log-integral AD, also checked live in the header suite.
    normalized = np.array([108.48300191691006, 4.104454256438259,
                           -250.95379129965553, -2.293043056268953, -2.2766007032275493])
    expected = np.zeros_like(theta)
    expected[alpha:gamma], expected[beta:delta] = -value[0], -1.
    expected[gamma:beta] = -(.1/p)*normalized[0]
    expected[delta:] = -.1*normalized[1:]
    row = 2*model.domain[0]
    assert np.isfinite(jacobian).all()
    np.testing.assert_allclose(jacobian[row], expected, rtol=2e-8, atol=2e-10)
    assert values[row] == pytest.approx(.5, abs=2e-10)
    assert model.assess(theta, None)[0]


@pytest.mark.parametrize("nu", [128., 1e4, 1e12, 1e15, 1e100])
def test_student_power_moment_high_nu_sparse_requests(nu):
    mp = pytest.importorskip("mpmath")
    shape = np.array([nu])
    outputs = []
    for mask in range(1, 8):
        f, g, h = np.full(1, np.nan), np.full(3, np.nan), np.full((3, 3), np.nan)
        assert _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, 1, .15, 1.7, mask,
            f.ctypes.data, g.ctypes.data, h.ctypes.data) == 0
        outputs.append((f, g, h))
    for mask, parts in enumerate(outputs, 1):
        for bit, part, full in zip((1, 2, 4), parts, outputs[-1], strict=True):
            if mask & bit:
                np.testing.assert_allclose(part, full, rtol=4e-14, atol=0)
            else:
                assert np.isnan(part).all()
    with mp.workdps(400):
        n, d, gamma = mp.mpf(nu), mp.mpf(1.7), mp.mpf(.15)
        value = mp.exp(d/2*mp.log(n-2)+mp.loggamma((d+1)/2)
            +mp.loggamma((n-d)/2)-mp.loggamma(n/2)-mp.log(mp.pi)/2)
        value *= ((1-gamma)**d+(1+gamma)**d)/2
        slope = d/(2*(n-2))+(mp.digamma((n-d)/2)-mp.digamma(n/2))/2
        curvature = -d/(2*(n-2)**2)+(mp.polygamma(1,(n-d)/2)-mp.polygamma(1,n/2))/4
        expected = list(map(float, [value, value*slope, value*(slope*slope+curvature)]))
    f, g, h = outputs[-1]
    np.testing.assert_allclose([f[0], g[2], h[2, 2]], expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("nu", [128., 1e4, 1e15, 1e100])
@pytest.mark.parametrize("delta", [.125, 1.7, 20.])
def test_student_power_moment_high_nu_matches_integral_ad(nu, delta):
    shape = np.array([nu])
    point = np.array([.15, delta, nu])
    f, g, h = np.empty(1), np.empty(3), np.empty((3, 3))
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 1, point[0], delta, 7,
        f.ctypes.data, g.ctypes.data, h.ctypes.data) == 0
    expected = aparch_power_moment_value_grad_hess(point, "studentt")
    scale_g = np.array([1., 1., nu**2])
    scale_h = np.array([[1., 1., nu**2], [1., 1., nu**2], [nu**2, nu**2, nu**3]])
    np.testing.assert_allclose(f[0], expected[0], rtol=2e-11, atol=0)
    np.testing.assert_allclose(g*scale_g, expected[1]*scale_g, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(h*scale_h, expected[2]*scale_h, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("delta", [.125, 200.])
def test_student_gaussian_limit_integral_resolution(monkeypatch, delta):
    import scivol._devtools.ad_oracle as oracle

    point = np.array([.15, delta, 1e15])
    ordinary = oracle.aparch_power_moment_value_grad_hess(point, "studentt")
    monkeypatch.setattr(oracle, "_egarch_skewt_quadrature_rule",
                        lambda: np.polynomial.legendre.leggauss(512))
    refined = oracle.aparch_power_moment_value_grad_hess(point, "studentt")
    for actual, expected in zip(ordinary, refined, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=0)


@pytest.mark.parametrize("gamma", [0.0, 0.25])
@pytest.mark.parametrize("delta", [0.7, 1.7, 2.0, 3.5])
def test_zero_skew_power_moment_oracle_has_finite_exact_reduction(gamma, delta):
    local = np.array([gamma, delta, 8.0, 0.0])
    reference = aparch_power_moment_value_grad_hess(local, "skewt")
    student = aparch_power_moment_value_grad_hess(local[:3], "studentt")
    for value in reference:
        assert np.isfinite(value).all()
    np.testing.assert_allclose(reference[0], student[0], rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(reference[1][:3], student[1], rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(reference[2][:3, :3], student[2], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("skew", [-0.2, 0.2])
@pytest.mark.parametrize("power", [0.2, 0.7])
def test_sublinear_skew_moment_oracle_resolves_the_origin(monkeypatch, skew, power):
    import scivol._devtools.ad_oracle as oracle

    point = np.array([-0.2, power, 8.0, skew])
    ordinary = oracle.aparch_power_moment_value_grad_hess(point, "skewt")
    rule = np.polynomial.legendre.leggauss(512)
    monkeypatch.setattr(oracle, "_egarch_skewt_quadrature_rule", lambda: rule)
    refined = oracle.aparch_power_moment_value_grad_hess(point, "skewt")
    for actual, expected in zip(ordinary, refined):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("delta", [0.7, 1.7, 2.0, 3.5])
def test_zero_skew_power_moment_native_matches_full_oracle(delta):
    gamma = 0.25
    local = np.array([gamma, delta, 8.0, 0.0])
    reference = aparch_power_moment_value_grad_hess(local, "skewt")
    value, gradient, hessian = np.empty(1), np.empty(4), np.empty((4, 4))
    shape = np.ascontiguousarray(local[2:])
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 2, gamma, delta, 7, value.ctypes.data,
        gradient.ctypes.data, hessian.ctypes.data,
    ) == 0
    for actual, expected in zip((value[0], gradient, hessian), reference):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("power", [0.7, 1.7])
def test_zero_skew_vg_moment_derivatives_match_independent_density_integrals(power):
    mp = pytest.importorskip("mpmath")
    local = np.array([0.25, power, 0.0, 3.0])
    value, gradient, hessian = aparch_power_moment_value_grad_hess(local, "vg")
    with mp.workdps(40):
        gamma, delta = mp.mpf(0.25), mp.mpf(power)

        def density(z, rho):
            # At lambda=3, K_(5/2) gives an exact polynomial-times-exponential PDF.
            skew = mp.sqrt(2) * rho / mp.sqrt(1 - rho * rho)
            scale = mp.sqrt(3 * (1 + skew * skew))
            y = scale * z + 3 * skew
            a = mp.sqrt(2 + skew * skew)
            return (
                scale
                / 2
                * mp.exp(skew * y - a * abs(y))
                * (y * y / a**3 + 3 * abs(y) / a**4 + 3 / a**5)
            )

        def moment_derivative(order):
            def integrand(x):
                if x == 0:
                    return mp.mpf(0)
                return x**delta * (
                    (1 - gamma) ** delta
                    * mp.diff(lambda rho: density(x, rho), 0, order)
                    + (1 + gamma) ** delta
                    * mp.diff(lambda rho: density(-x, rho), 0, order)
                )

            return float(mp.quad(integrand, [0, 1, 4, 12, mp.inf]))

        reference = [moment_derivative(order) for order in (0, 1, 2)]
    np.testing.assert_allclose(
        [value, gradient[2], hessian[2, 2]],
        reference,
        rtol=2e-9,
        atol=2e-11,
    )


@pytest.mark.parametrize("rho", [-0.12, 0.0, 0.12])
def test_vg_power_moment_split_oracle_converges(rho):
    from scivol._devtools.ad_oracle import _jax_modules, _jax_vg_power_moment

    jax,jnp,jsp = _jax_modules()
    local = np.array([0.1,1.4,rho,2.0])
    references=[]
    for count in (64,128):
        def objective(theta):
            return _jax_vg_power_moment(jnp,jsp,*theta,nodes_count=count)
        point=jnp.asarray(local)
        references.append((float(objective(point)),np.asarray(jax.grad(objective)(point)),
            np.asarray(jax.hessian(objective)(point))))
    for coarse,fine in zip(*references):
        np.testing.assert_allclose(coarse,fine,rtol=2e-6,atol=2e-8)
    value,gradient,hessian=np.empty(1),np.empty(4),np.empty((4,4))
    shape=np.ascontiguousarray(local[2:])
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data,7,local[0],local[1],7,value.ctypes.data,
        gradient.ctypes.data,hessian.ctypes.data)==0
    for actual,expected in zip((value[0],gradient,hessian),references[-1]):
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_standalone_aparch_theta_jet_matches_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(714).normal(size=24).astype(np.float64)
    theta = _theta(shape, joint=False)
    initial = float(np.mean(data * data))
    status, valid, value, gradient, hessian = _evaluate(
        theta, data, code, initial_variance=initial)
    expected = _standalone_ad_jet(theta, data, name, initial_variance=initial)
    assert (status, valid) == (0, 7)
    np.testing.assert_allclose(value, expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(gradient, expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(hessian, expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_joint_arma_aparch_theta_jet_matches_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(2718).normal(size=28).astype(np.float64)
    theta = _theta(shape, joint=True)
    initial = float(np.mean(data * data))
    actual = _evaluate(theta, data, code, orders=(1, 1, 1, 1), initial_variance=initial)
    expected = arma_aparch_value_grad_hess(
        theta, data, 1, 1, 1, 1, name, initial_variance=initial)
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)
    assert np.max(np.abs(actual[4][:3, 3:])) > 1e-6


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_aparch_oracle_preserves_small_conditioning_and_density_scale(name, code, shape):
    theta = _theta(shape, joint=False)
    scale = 1e-6
    unit = _standalone_ad_jet(theta, np.array([1.]), name, initial_variance=1.)
    data = np.array([scale])
    expected = (unit[0] + np.log(scale), unit[1], unit[2])
    oracle = _standalone_ad_jet(theta, data, name, initial_variance=scale * scale)
    status, valid, *native = _evaluate(theta, data, code, initial_variance=scale * scale)
    assert (status, valid) == (0, 7)
    if name == "normal":
        np.testing.assert_allclose(expected[0], .5 * (np.log(2 * np.pi) + 1.)
                                   + np.log(scale), rtol=0, atol=2e-14)
    for outputs in (oracle, native):
        np.testing.assert_allclose(outputs[0], expected[0], rtol=0, atol=1e-9)
        for part, target in zip(outputs, expected):
            assert np.isfinite(part).all()
            np.testing.assert_allclose(part, target, rtol=4e-7, atol=4e-9)
        np.testing.assert_array_equal(outputs[1][:5], 0.)
        np.testing.assert_array_equal(outputs[2][:5, :], 0.)
        np.testing.assert_array_equal(outputs[2][:, :5], 0.)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_aparch_oracle_retains_explicit_historical_conditioning(name, code, shape):
    theta, data = _theta(shape, joint=False), np.array([1e-6])
    historical = _standalone_ad_jet(theta, data, name, initial_variance=None)
    explicit = _standalone_ad_jet(theta, data, name, initial_variance=1e-8)
    for left, right in zip(historical, explicit):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)


def test_aparch_oracle_uses_supplied_seed_instead_of_recomputing_it():
    theta, data = _theta([], joint=False), np.array([1e-6])
    expected = .5 * (np.log(2 * np.pi) + 1e-12)
    oracle = aparch_value_grad_hess(theta, data, 1, 1, initial_variance=1.)
    native = _evaluate(theta, data, 0, initial_variance=1.)
    assert native[:2] == (0, 7)
    np.testing.assert_allclose([oracle[0], native[2]], expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
@pytest.mark.parametrize("seed", [0., -1., np.nan, np.inf])
def test_aparch_oracles_reject_invalid_explicit_conditioning(name, code, shape, seed):
    data = np.array([.2, -.4, .5])
    with pytest.raises(ValueError, match="initial_variance must be finite and positive"):
        _standalone_ad_jet(_theta(shape, joint=False), data, name, initial_variance=seed)
    with pytest.raises(ValueError, match="initial_variance must be finite and positive"):
        arma_aparch_value_grad_hess(_theta(shape, joint=True), data, 1, 1, 1, 1,
                                   name, initial_variance=seed)


@pytest.mark.parametrize(("name", "code", "shape"), [DENSITIES[i] for i in (0, 4, 7)])
def test_joint_aparch_oracle_uses_the_native_conditioning_plan(name, code, shape):
    data = np.array([.2, -.4, .5, -1., .7, .3])
    theta = _theta(shape, joint=True)
    orders, scale = (1, 1, 1, 1), 1e-5
    unit = _evaluate(theta, data, code, orders=orders)
    theta[0] *= scale
    theta[3] *= scale ** theta[7]
    data *= scale
    initial = float(np.mean(data * data))
    expected = arma_aparch_value_grad_hess(theta, data, *orders, name,
                                         initial_variance=initial)
    native = _evaluate(theta, data, code, orders=orders, initial_variance=initial)
    assert unit[:2] == native[:2] == (0, 7)
    np.testing.assert_allclose(native[2] - unit[2], (data.size - 1) * np.log(scale),
                               rtol=0, atol=2e-12)
    np.testing.assert_allclose(native[2], expected[0], rtol=0, atol=1e-9)
    for actual, reference in zip(native[2:], expected):
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(actual, reference, rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_aparch_transformed_jet_matches_independent_chart_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(99).normal(size=14).astype(np.float64)
    theta = _theta(shape, joint=False)
    coordinates = _map(theta, code)
    mapped = _map(coordinates, code, forward=True)
    base = _evaluate(mapped, data, code)
    actual = _evaluate(coordinates, data, code, transformed=True)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates, mapped, base[2], base[3], base[4], 1, 1, name
    )
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_joint_aparch_transformed_jet_includes_mean_curvature(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(100).normal(size=14).astype(np.float64)
    theta = _theta(shape, joint=True)
    orders = (1, 1, 1, 1)
    coordinates = _map(theta, code, orders=orders)
    mapped = _map(coordinates, code, orders=orders, forward=True)
    base = _evaluate(mapped, data, code, orders=orders)
    actual = _evaluate(coordinates, data, code, transformed=True, orders=orders)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates,
        mapped,
        base[2],
        base[3],
        base[4],
        1,
        1,
        name,
        p_ar=1,
        q_ma=1,
    )
    assert base[:2] == actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_every_sparse_request_is_a_projection_of_the_full_jet(
    joint: bool, transformed: bool, name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(18).normal(size=18).astype(np.float64)
    orders = (1, 1, 1, 1) if joint else None
    theta = _theta(shape, joint=joint)
    point = _map(theta, code, orders=orders) if transformed else theta
    full = _evaluate(point, data, code, transformed=transformed, orders=orders)
    assert full[:2] == (0, 7)
    for request in range(1, 8):
        sparse = _evaluate(
            point,
            data,
            code,
            request,
            transformed=transformed,
            orders=orders,
        )
        assert sparse[:2] == (0, request)
        if request & 1:
            assert np.isfinite(sparse[2])
            np.testing.assert_allclose(sparse[2], full[2], rtol=2e-12, atol=2e-12)
        else:
            assert np.isnan(sparse[2])
        if request & 2:
            assert np.isfinite(sparse[3]).all()
            np.testing.assert_allclose(sparse[3], full[3], rtol=2e-11, atol=2e-11)
        else:
            assert np.isnan(sparse[3]).all()
        if request & 4:
            assert np.isfinite(sparse[4]).all()
            np.testing.assert_allclose(sparse[4], full[4], rtol=2e-10, atol=2e-10)
        else:
            assert np.isnan(sparse[4]).all()


@lru_cache(maxsize=1)
def _runtime_library() -> ctypes.CDLL:
    library = ctypes.CDLL(_native_core.__file__)
    size = ctypes.c_size_t
    unsigned = ctypes.c_uint32
    pointer = ctypes.c_void_p
    library.vlk_runtime_aparch_jet_workspace_bytes.argtypes = [
        size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_aparch_jet_workspace_bytes.restype = size
    library.vlk_runtime_aparch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_aparch_jet_evaluate.restype = ctypes.c_int32
    library.vlk_runtime_arma_aparch_jet_workspace_bytes.argtypes = [
        size, size, size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_arma_aparch_jet_workspace_bytes.restype = size
    library.vlk_runtime_arma_aparch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_arma_aparch_jet_evaluate.restype = ctypes.c_int32
    return library


def _forced_runtime(
    theta: np.ndarray,
    data: np.ndarray,
    code: int,
    *,
    orders: tuple[int, int, int, int] | None,
    request: int = 7,
    transformed: bool = False,
    volatility_orders: tuple[int, int] = (1, 1),
    initial_variance: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    library = _runtime_library()
    value = np.full(1, np.nan)
    gradient = np.full(theta.size, np.nan)
    hessian = np.full((theta.size, theta.size), np.nan)
    valid = ctypes.c_uint32(0)
    initial = float(np.mean(data * data)) if initial_variance is None else initial_variance
    if orders is None:
        p_vol, q_vol = volatility_orders
        size = library.vlk_runtime_aparch_jet_workspace_bytes(
            p_vol, q_vol, code, int(transformed), request)
        workspace = _workspace(size)
        status = library.vlk_runtime_aparch_jet_evaluate(
            theta.ctypes.data, theta.size, data.ctypes.data, data.size, initial,
            p_vol, q_vol, code, int(transformed), request, workspace.ctypes.data, workspace.nbytes,
            value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data,
            ctypes.byref(valid),
        )
    else:
        p_ar, q_ma, p_vol, q_vol = orders
        size = library.vlk_runtime_arma_aparch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, code, int(transformed), request
        )
        workspace = _workspace(size)
        status = library.vlk_runtime_arma_aparch_jet_evaluate(
            theta.ctypes.data, theta.size, data.ctypes.data, data.size, initial,
            p_ar, q_ma, p_vol, q_vol, code, int(transformed), request,
            workspace.ctypes.data, workspace.nbytes, value.ctypes.data,
            gradient.ctypes.data, hessian.ctypes.data, ctypes.byref(valid),
        )
    assert (status, valid.value) == (0, request)
    return float(value[0]), gradient, hessian


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("orders,volatility_orders", [
    pytest.param(None, (1, 1), id="standalone11"),
    pytest.param(None, (1, 2), id="standalone12"),
    pytest.param(None, (2, 1), id="standalone21"),
    pytest.param(None, (2, 2), id="standalone22"),
    pytest.param((1, 1, 1, 1), (1, 1), id="arma1111"),
])
def test_fixed_and_runtime_aparch_specializations_are_identical(
    name: str, code: int, shape: list[float], transformed: bool,
    orders: tuple[int, int, int, int] | None, volatility_orders: tuple[int, int],
) -> None:
    data = np.random.default_rng(42).normal(size=20).astype(np.float64)
    p, q = volatility_orders
    mean = MEAN if orders is not None else np.empty(0)
    theta = np.r_[mean, 0.08, np.full(p, 0.07 / p), np.linspace(0.15, -0.1, p),
                  np.full(q, 0.82 / q), 1.7, shape]
    layout = {"orders": orders, "volatility_orders": volatility_orders}
    point = _map(theta, code, **layout) if transformed else theta
    for request in range(1, 8):
        status, valid, *fixed = _evaluate(
            point, data, code, request, transformed=transformed, **layout)
        runtime = _forced_runtime(
            point, data, code, request=request, transformed=transformed, **layout)
        assert (status, valid) == (0, request)
        for bit, left, right in zip((1, 2, 4), fixed, runtime):
            if request & bit:
                assert np.isfinite(left).all() and np.isfinite(right).all()
                np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)
            else:
                assert np.isnan(left).all() and np.isnan(right).all()


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_arbitrary_joint_orders_match_ad(name: str, code: int, shape: list[float]) -> None:
    data = np.random.default_rng(204).normal(size=18).astype(np.float64)
    orders = (2, 1, 3, 2)
    theta = np.array([
        0.02, 0.10, -0.04, -0.08,
        0.06, 0.02, 0.015, 0.01,
        0.10, -0.12, 0.08,
        0.55, 0.18, 1.8, *shape,
    ])
    actual = _evaluate(theta, data, code, orders=orders)
    expected = arma_aparch_value_grad_hess(
        theta, data, *orders, name, initial_variance=float(np.mean(data * data)))
    assert actual[:2] == (0, 7)
    assert all(np.isfinite(part).all() for part in map(np.asarray, expected))
    np.testing.assert_allclose(actual[2], expected[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual[3], expected[1], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=2e-7, atol=2e-8)
    assert np.max(np.abs(actual[4][:4, 4:])) > 1e-6
    assert _native_core._arma_aparch_dispatch_kind(*orders, code, 0) == 0


@pytest.mark.parametrize("p_ar,q_ma", [(0, 0), (0, 2), (3, 0)])
def test_joint_aparch_empty_reflection_blocks_have_exact_transformed_hessian(
    p_ar, q_ma
):
    data = np.random.default_rng(2087).normal(size=19)
    orders = (p_ar, q_ma, 2, 1)
    theta = np.r_[
        0.03,
        np.full(p_ar, 0.04),
        np.full(q_ma, -0.03),
        0.08,
        0.04,
        0.03,
        0.1,
        -0.1,
        0.6,
        1.7,
    ]
    coordinates = _map(theta, 0, orders=orders)
    theta_jet = _evaluate(theta, data, 0, orders=orders)
    assert theta_jet[:2] == (0, 7)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, *theta_jet[2:], 2, 1, "normal", p_ar=p_ar, q_ma=q_ma
    )
    for mask in (4, 5, 6, 7):
        actual = _evaluate(coordinates, data, 0, mask, transformed=True, orders=orders)
        assert actual[:2] == (0, mask)
        np.testing.assert_allclose(actual[4], expected[2], rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("delta,gamma,alpha,derivative_bits", [
    (.5, .2, .07, 0),
    (1., .2, .07, 0),
    (1.7, .2, .07, 2),
    (2., .2, .07, 2),
    (2., 0., .07, 6),
    (2.5, .2, .07, 6),
    (.5, .2, 0., 2),
    (1.7, .2, 0., 6),
])
def test_aparch_domain_and_zero_residual_derivative_semantics(
    delta, gamma, alpha, derivative_bits,
) -> None:
    data = np.array([0.0, 1.0, 0.0, 1.0])
    theta = np.r_[0.0, 0.0, 0.0, 0.08, alpha, gamma, 0.82, delta]
    for mask in range(1, 8):
        status, valid, *outputs = _evaluate(
            theta, data, 0, mask, orders=(1, 1, 1, 1), initial_variance=1.)
        differentiable = not (mask & 6 & ~derivative_bits)
        assert (status, valid) == ((0, mask) if differentiable else (3, 0))
        for bit, output in zip((1, 2, 4), outputs):
            if not mask & bit:
                assert np.isnan(output).all()
            elif differentiable:
                assert np.isfinite(output).all()

    student = np.r_[VOLATILITY, 4.0]
    student[4] = 4.0
    assert _evaluate(student, data, 1, initial_variance=1.0)[0] == 3


@pytest.mark.parametrize("joint", [False, True])
def test_aparch_zero_volatility_order_is_rejected_before_writing(joint):
    with pytest.raises(ValueError, match="positive integers"):
        APARCH(1, 0)
    data = np.array([.3, -.7, .2, .8])
    theta = np.r_[MEAN if joint else [], .08, .07, .2, 1.7]
    axes = (1, 1, 1, 0) if joint else (1, 0)
    sizes = (_native_core._arma_aparch_workspace_bytes if joint
             else _native_core._aparch_workspace_bytes)
    evaluate = _native_core._arma_aparch_jet if joint else _native_core._aparch_jet
    library = _runtime_library()
    raw_sizes = (library.vlk_runtime_arma_aparch_jet_workspace_bytes if joint
                 else library.vlk_runtime_aparch_jet_workspace_bytes)
    for transformed in (False, True):
        for mask in range(1, 8):
            assert sizes(*axes, 0, transformed, mask) == 1
            assert raw_sizes(*axes, 0, transformed, mask) == 0
            work = np.full(256, 0xA5, dtype=np.uint8)
            outputs = [np.full(1, 12345.), np.full(theta.size, 12345.),
                       np.full((theta.size, theta.size), 12345.)]
            status, valid = evaluate(
                theta.ctypes.data, theta.size, data.ctypes.data, data.size, 1.,
                *axes, 0, transformed, mask, work.ctypes.data, work.nbytes,
                *(output.ctypes.data for output in outputs))
            assert (status, valid) == (1, 0)
            assert np.all(work == 0xA5)
            assert all(np.all(output == 12345.) for output in outputs)


@pytest.mark.parametrize("orders,volatility_orders", [
    (None, (1, 1)), (None, (4, 3)),
    ((1, 1, 1, 1), (1, 1)), ((2, 0, 3, 2), (3, 2)),
])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("code,shape", [(0, []), (1, [7.5]), (3, [1.7])])
def test_aparch_exact_workspace_masks_preserve_guards_and_unused_outputs(
    orders, volatility_orders, transformed, code, shape,
):
    p, q = volatility_orders
    mean = [] if orders is None else [.02, *([.06] * orders[0]), *([-.08] * orders[1])]
    theta = np.r_[mean, .08, np.full(p, .06/p), np.full(p, .1), np.full(q, .6/q), 1.6, shape]
    data = np.random.default_rng(4102).normal(size=9)
    point = _map(theta, code, orders=orders, volatility_orders=volatility_orders) if transformed else theta
    axes = orders or volatility_orders
    sizes = (_native_core._aparch_workspace_bytes if orders is None
             else _native_core._arma_aparch_workspace_bytes)
    evaluate = _native_core._aparch_jet if orders is None else _native_core._arma_aparch_jet
    counts = {mask: sizes(*axes, code, transformed, mask) for mask in range(1, 8)}
    assert counts[4] == counts[5] and counts[6] == counts[7]
    assert counts[6] - counts[4] == (8 * point.size if code == 0 and not transformed else 0)
    full = _evaluate(point, data, code, transformed=transformed, orders=orders,
                     volatility_orders=volatility_orders, initial_variance=1.1)
    assert full[:2] == (0, 7)

    def guarded(count, dtype):
        backing = np.full(count * np.dtype(dtype).itemsize + 192, 0xA5, dtype=np.uint8)
        start = ((backing.ctypes.data + 127) & ~63) - backing.ctypes.data
        output = backing[start:start + count * np.dtype(dtype).itemsize].view(dtype)
        return output, backing, start, start + output.nbytes

    for mask in range(1, 8):
        for short, null_outputs in ((False, False), (False, True), (True, False)):
            count = counts[mask]
            assert count > 0
            work = guarded(count, np.uint8)
            buffers = [guarded(n, np.float64) for n in (1, point.size, point.size**2)]
            for output, *_ in buffers:
                output.fill(12345.)
            pointers = [output.ctypes.data if not null_outputs or mask & bit else 0
                        for bit, (output, *_) in zip((1, 2, 4), buffers)]
            status, valid = evaluate(
                point.ctypes.data, point.size, data.ctypes.data, data.size, 1.1,
                *axes, code, transformed, mask, work[0].ctypes.data, count - int(short), *pointers)
            assert (status, valid) == ((2, 0) if short else (0, mask))
            if short:
                assert np.all(work[0] == 0xA5)
            for output, backing, start, end in [work, *buffers]:
                assert np.all(backing[:start] == 0xA5) and np.all(backing[end:] == 0xA5)
            for bit, (output, *_), reference in zip((1, 2, 4), buffers, full[2:]):
                if short or not mask & bit:
                    assert np.all(output == 12345.)
                else:
                    reference = np.asarray(reference).ravel()
                    assert np.isfinite(output).all()
                    if bit == 1:
                        assert abs(output[0] - reference[0]) <= 1e-9
                    else:
                        np.testing.assert_allclose(output, reference, rtol=3e-10, atol=2e-11)


@pytest.mark.parametrize("omega", [1e-310, np.nextafter(0., 1.)])
def test_aparch_value_and_filter_preserve_subnormal_power(omega):
    from fractions import Fraction

    theta, data = np.array([omega, 0., 0., 0., 2.]), np.zeros(3)
    expected = .5*(data.size*np.log(2*np.pi) + (data.size-1)*np.log(omega))
    inverse_omega = 1 / Fraction.from_float(float(omega))
    maximum = Fraction.from_float(np.finfo(float).max)
    # On this slice F_omega=1/omega and F_omega_omega=-1/omega**2.
    assert inverse_omega > maximum and inverse_omega**2 > maximum
    for mask in range(1, 8):
        status, valid, value, gradient, hessian = _evaluate(theta, data, 0, mask, initial_variance=1.)
        assert (status, valid) == ((0, 1) if mask == 1 else (3, 0))
        if mask == 1:
            assert np.isfinite(value) and abs(value-expected) <= 1e-9
        if not mask & 2:
            assert np.isnan(gradient).all()
        if not mask & 4:
            assert np.isnan(hessian).all()
    runtime = _forced_runtime(theta, data, 0, orders=None, request=1, initial_variance=1.)
    assert np.isfinite(runtime[0]) and abs(runtime[0]-expected) <= 1e-9
    work = _workspace(_native_core._aparch_filter_workspace_bytes(1, 1, 0))
    variance = np.empty_like(data)
    assert _native_core._aparch_filter_state(
        theta.ctypes.data, data.ctypes.data, data.size, 1., 1, 1, 0,
        work.ctypes.data, work.nbytes, variance.ctypes.data) == 0
    np.testing.assert_array_equal(variance, [1., omega, omega])


def test_joint_normal_value_preserves_long_boundary_sum():
    mp = pytest.importorskip("mpmath")
    theta = np.array([0., 0., 0., 1e-310, 0., -.2, .35, 2.])
    pattern = [.8, -.4, 1.1, .3, -1.2, .7, .6, -.2, 1.4, -.9, .5, .2, -.6, .9]
    data = np.resize(pattern, 800) * 1e-160
    references = []
    for digits in (80, 120):
        with mp.workdps(digits):
            # Alpha=0 and delta=2 reduce the exact conditional state to an affine recursion.
            variance, terms = mp.mpf(1), []
            for residual in data[1:]:
                variance = mp.mpf(float(theta[3])) + mp.mpf(float(theta[6])) * variance
                terms.append((mp.log(2 * mp.pi) + mp.log(variance)
                              + mp.mpf(float(residual))**2 / variance) / 2)
            references.append(mp.fsum(terms))
    with mp.workdps(120):
        assert abs(references[0] - references[1]) < mp.mpf("1e-65")
        status, valid, *fixed = _evaluate(
            theta, data, 0, 1, orders=(1, 1, 1, 1), initial_variance=1.)
        assert (status, valid) == (0, 1)
        runtime = _forced_runtime(theta, data, 0, orders=(1, 1, 1, 1),
                                  request=1, initial_variance=1.)
        for value, gradient, hessian in (fixed, runtime):
            assert np.isfinite(value)
            assert abs(mp.mpf(value) - references[1]) <= mp.mpf("2e-11")
            assert np.isnan(gradient).all() and np.isnan(hessian).all()


def test_canonical_filters_match_direct_recursions() -> None:
    data = np.random.default_rng(33).normal(size=16).astype(np.float64)
    theta = _theta([], joint=False)
    initial = float(np.mean(data * data))
    workspace = _workspace(_native_core._aparch_filter_workspace_bytes(1, 1, 0))
    variance = np.empty_like(data)
    status = _native_core._aparch_filter_state(
        theta.ctypes.data, data.ctypes.data, data.size, initial,
        1, 1, 0, workspace.ctypes.data, workspace.nbytes,
        variance.ctypes.data,
    )
    power = np.empty_like(data)
    power[0] = initial ** (0.5 * theta[4])
    expected = np.empty_like(data)
    expected[0] = initial
    for time in range(1, data.size):
        shock = abs(data[time - 1]) - theta[2] * data[time - 1]
        power[time] = theta[0] + theta[1] * shock ** theta[4] + theta[3] * power[time - 1]
        expected[time] = power[time] ** (2.0 / theta[4])
    assert status == 0
    np.testing.assert_allclose(variance, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("p,q", [(1, 1), (4, 3)])
def test_aparch_scaled_seed_preserves_finite_full_jet_and_filter(p, q):
    from scivol._devtools.ad_oracle import _jax_modules

    jax, jnp, _ = _jax_modules()
    theta = np.r_[1., np.zeros(2*p), np.full(q,.1/q), 1500.]
    data = np.zeros(9)

    def log_variances(point):
        delta = point[-1]
        intercept = jnp.exp(jnp.log(point[0])-delta*jnp.log(2.))
        powers, logs = [jnp.array(1.)], [jnp.log(4.)]
        for t in range(1,len(data)):
            power = intercept+sum(point[1+2*p+j]*powers[t-1-j] for j in range(min(t,q)))
            powers.append(power)
            logs.append(2*jnp.log(2.)+2/delta*jnp.log(power))
        return jnp.stack(logs)

    objective = lambda point: .5*jnp.sum(jnp.log(2*jnp.pi)+log_variances(point))
    point = jnp.asarray(theta)
    expected = float(objective(point)), np.asarray(jax.grad(objective)(point)), np.asarray(jax.hessian(objective)(point))
    assert all(np.isfinite(part).all() for part in map(np.asarray,expected))
    for mask in range(1,8):
        work = _workspace(_native_core._aparch_workspace_bytes(p,q,0,False,mask))
        f,g,h = np.full(1,np.nan),np.full(theta.size,np.nan),np.full((theta.size,theta.size),np.nan)
        status,valid = _native_core._aparch_jet(theta.ctypes.data,theta.size,data.ctypes.data,len(data),
            4.,p,q,0,False,mask,work.ctypes.data,work.nbytes,f.ctypes.data,g.ctypes.data,h.ctypes.data)
        assert (status,valid) == (0,mask)
        for bit,actual,reference in zip((1,2,4),(f[0],g,h),expected):
            if mask & bit:
                np.testing.assert_allclose(actual,reference,rtol=3e-12,atol=2e-14)
            else:
                assert np.isnan(actual).all()
    work = _workspace(_native_core._aparch_filter_workspace_bytes(p,q,0))
    variance = np.empty_like(data)
    assert _native_core._aparch_filter_state(theta.ctypes.data,data.ctypes.data,len(data),4.,p,q,0,
        work.ctypes.data,work.nbytes,variance.ctypes.data) == 0
    np.testing.assert_allclose(variance,np.exp(log_variances(point)),rtol=2e-14,atol=0.)


@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (2, 0, 3, 2)])
def test_joint_aparch_scaled_seed_matches_independent_recursion(orders):
    from scivol._devtools.ad_oracle import _jax_modules

    jax, jnp, _ = _jax_modules()
    p_ar, q_ma, p, q = orders
    m = 1 + p_ar + q_ma
    theta = np.r_[.03, np.full(p_ar, .12), np.full(q_ma, -.08),
                  1., np.zeros(p), np.full(p, -.15), np.full(q, .1/q), 1500.]
    data = np.array([.3, -.7, .1, 1.8, -.2, .6, -.4, .15, 1., -.9, .3, .5])
    start = max(orders)

    def conditioned_paths(point):
        delta = point[-1]
        residuals = [jnp.array(0.) for _ in data]
        powers = [jnp.array(1.) for _ in data]
        logs = [jnp.log(4.) for _ in data]
        # Express power states in units of h0**(delta/2), which overflows here.
        intercept = point[m]*jnp.exp(-delta*jnp.log(2.))
        for t in range(start, len(data)):
            residuals[t] = data[t]-point[0]
            for lag in range(p_ar):
                residuals[t] -= point[1+lag]*data[t-1-lag]
            for lag in range(q_ma):
                residuals[t] -= point[1+p_ar+lag]*residuals[t-1-lag]
            power = intercept
            for lag in range(p):
                previous = t-1-lag
                if previous >= start:
                    error = residuals[previous]
                    base = jnp.abs(error)-point[m+1+p+lag]*error
                    power += point[m+1+lag]*jnp.exp(delta*(jnp.log(base)-jnp.log(2.)))
            for lag in range(q):
                power += point[m+1+2*p+lag]*powers[t-1-lag]
            powers[t] = power
            logs[t] = jnp.log(4.)+2/delta*jnp.log(power)
        return jnp.stack(residuals), jnp.stack(logs)

    def objective(point):
        errors, logs = conditioned_paths(point)
        return .5*jnp.sum(jnp.log(2*jnp.pi)+logs[start:]
                         +errors[start:]**2*jnp.exp(-logs[start:]))

    point = jnp.asarray(theta)
    expected = (float(objective(point)), np.asarray(jax.grad(objective)(point)),
                np.asarray(jax.hessian(objective)(point)))
    assert all(np.isfinite(part).all() for part in map(np.asarray, expected))
    assert np.linalg.norm(expected[2][:m, m:]) > 1e-4
    for mask in range(1, 8):
        status, valid, *outputs = _evaluate(
            theta, data, 0, mask, orders=orders, initial_variance=4.)
        runtime = _forced_runtime(theta, data, 0, orders=orders,
                                  request=mask, initial_variance=4.)
        assert (status, valid) == (0, mask)
        for bit, actual, general, reference in zip((1, 2, 4), outputs, runtime, expected):
            if mask & bit:
                np.testing.assert_allclose(actual, reference, rtol=4e-10, atol=2e-12)
                np.testing.assert_allclose(actual, general, rtol=4e-12, atol=2e-12)
            else:
                assert np.isnan(actual).all() and np.isnan(general).all()

    work = _workspace(_native_core._arma_aparch_filter_workspace_bytes(*orders, 0))
    residual, variance = np.empty_like(data), np.empty_like(data)
    status = _native_core._arma_aparch_filter_state(
        theta.ctypes.data, data.ctypes.data, len(data), 4., *orders, 0,
        work.ctypes.data, work.nbytes, residual.ctypes.data, variance.ctypes.data)
    expected_residual, expected_log_variance = conditioned_paths(point)
    assert status == 0
    np.testing.assert_allclose(residual, expected_residual, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(variance, np.exp(expected_log_variance), rtol=2e-14, atol=0.)


def test_aparch_exact_garch_reduction() -> None:
    data = np.random.default_rng(71).normal(size=18).astype(np.float64)
    aparch_theta = np.array([0.08, 0.07, 0.0, 0.82, 2.0])
    garch_theta = aparch_theta[[0, 1, 3]]
    actual = _evaluate(aparch_theta, data, 0)
    squared = np.ascontiguousarray(data * data)
    workspace = _workspace(_native_core._garch_workspace_bytes(1, 1, 0, 0, 7))
    value = np.empty(1)
    gradient = np.empty(3)
    hessian = np.empty((3, 3))
    status, valid = _native_core._garch_jet(
        garch_theta.ctypes.data, data.ctypes.data, squared.ctypes.data,
        data.size, float(np.mean(squared)), 1, 1, 0, 0, 7,
        workspace.ctypes.data, workspace.nbytes, value.ctypes.data,
        gradient.ctypes.data, hessian.ctypes.data,
    )
    common = np.array([0, 1, 3])
    assert (status, valid) == (0, 7)
    np.testing.assert_allclose(actual[2], value[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual[3][common], gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual[4][np.ix_(common, common)], hessian, rtol=2e-11, atol=2e-11
    )


@pytest.mark.parametrize("code", range(8))
@pytest.mark.parametrize("transformed", [False, True])
def test_expected_fixed_dispatches_are_reachable(code: int, transformed: bool) -> None:
    for p in (1, 2):
        for q in (1, 2):
            assert _native_core._aparch_dispatch_kind(p, q, code, int(transformed)) == 1
    assert _native_core._aparch_dispatch_kind(3, 2, code, int(transformed)) == 0
    assert _native_core._arma_aparch_dispatch_kind(1, 1, 1, 1, code, int(transformed)) == 1
    assert _native_core._arma_aparch_dispatch_kind(2, 1, 1, 1, code, int(transformed)) == 0


def _simulate_joint_aparch_studentt() -> np.ndarray:
    rng = np.random.default_rng(91)
    observations = 1_200
    nu = 7.0
    innovations = rng.standard_t(nu, size=observations) * np.sqrt((nu - 2.0) / nu)
    residuals = np.empty(observations)
    powers = np.empty(observations)
    data = np.empty(observations)
    powers[0] = 1.0
    residuals[0] = innovations[0]
    data[0] = residuals[0]
    for index in range(1, observations):
        powers[index] = (
            0.08
            + 0.10 * (abs(residuals[index - 1]) - 0.15 * residuals[index - 1]) ** 1.6
            + 0.80 * powers[index - 1]
        )
        residuals[index] = powers[index] ** (1.0 / 1.6) * innovations[index]
        data[index] = (
            0.03 + 0.18 * data[index - 1]
            - 0.08 * residuals[index - 1] + residuals[index]
        )
    return np.ascontiguousarray(data)


def test_aparch_density_matrix_and_joint_surface_default_to_native() -> None:
    from scivol._native_fit import _uses_native_by_default

    densities = (
        Normal(), StudentT(), SkewT(), GED(),
        GeneralizedHyperbolic(), NormalInverseGaussian(),
        Hyperbolic(), VarianceGamma(),
    )
    for density in densities:
        assert _uses_native_by_default(APARCH(1, 1) + density)
        assert _uses_native_by_default(ARMA(1, 1) + APARCH(1, 1) + density)


def test_joint_aparch_default_is_credible_and_beats_matched_scipy() -> None:
    from scivol._native_fit import (
        _describe_model,
        _evaluate_theta_request,
        _initial_theta,
    )

    data = _simulate_joint_aparch_studentt()
    spec = ARMA(1, 1) + APARCH(1, 1) + StudentT()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    native = spec.fit(data, initial_params=start, maxiter=500)
    scipy = spec.fit(
        data,
        solver="slsqp",
        initial_params=start,
        maxiter=500,
    )
    native_nll = _evaluate_theta_request(
        model, data, np.asarray(native.params), request=1
    )[0]
    scipy_nll = _evaluate_theta_request(
        model, data, np.asarray(scipy.params), request=1
    )[0]

    assert native.converged
    assert native.fit_info.solver == "native-slsqp"
    assert native.fit_info.kkt_residual <= native.fit_info.kkt_tolerance
    assert native.fit_info.maximum_constraint_violation == 0.0
    assert native_nll <= scipy_nll + 1e-8
    filtered = spec.filter(data, native.params)
    np.testing.assert_allclose(filtered.residuals, native.resid)
    np.testing.assert_allclose(filtered.sigma2, native.sigma2)
    assert spec.score(data, native.params).shape == native.params.shape
    assert spec.hessian(data, native.params).shape == (
        native.params.size, native.params.size
    )
    assert native.garch_params is not None
    assert 0.0 < native.garch_params.persistence < 1.0
    assert 0.0 < spec.get_component(Role.VOLATILITY).persistence() < 1.0
