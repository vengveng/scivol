"""Ordinary in-domain Hansen moment recovery and transformed joint consumers."""

from functools import lru_cache
import math

import numpy as np
import pytest


VALUE_RTOL = 2e-10
VALUE_ATOL = 2e-11
DERIVATIVE_RTOL = 2e-8
DERIVATIVE_ATOL = 2e-9
RETAINED = (math.tanh(.15), 2., 3., math.tanh(.4))
REFLECTED = (-RETAINED[0], RETAINED[1], RETAINED[2], -RETAINED[3])
POINTS = (
    pytest.param(RETAINED, id="retained"),
    pytest.param(REFLECTED, id="reflected"),
    pytest.param((RETAINED[0], 1.95, 3., RETAINED[3]), id="lower-power"),
    pytest.param((RETAINED[0], 2.05, 3., RETAINED[3]), id="upper-power"),
)


@lru_cache(maxsize=1)
def _jax_modules():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax, pytest.importorskip("jax.numpy"), pytest.importorskip("jax.scipy")


def _branch_boundary(point):
    nu, skew = point[2:]
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(math.pi * (nu - 2))
    a = 4 * skew * c * (nu - 2) / (nu - 1)
    return abs(a / math.sqrt(1 + 3 * skew * skew - a * a))


@lru_cache(maxsize=4)
def _density_integral(boundary, nodes_count):
    """Direct density integral; no pole extraction, recurrence, floors or clipping."""
    _, jnp, jsp = _jax_modules()
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    unit_np = (nodes + 1) / 2
    assert np.all((unit_np > 0.) & (unit_np < 1.))
    unit, weight = jnp.asarray(unit_np), jnp.asarray(weights / 2)
    bounded_x = boundary * unit**4
    bounded_jacobian = boundary * 4 * unit**3
    unbounded_x = boundary + (unit / (1 - unit))**4
    unbounded_jacobian = 4 * unit**3 / (1 - unit)**5

    def moment(parameters):
        gamma, delta, nu, skew = parameters
        log_c = (jsp.special.gammaln((nu + 1) / 2) - jsp.special.gammaln(nu / 2)
                 - .5 * jnp.log(jnp.pi * (nu - 2)))
        a = 4 * skew * jnp.exp(log_c) * (nu - 2) / (nu - 1)
        b = jnp.sqrt(1 + 3 * skew * skew - a * a)

        def interval(magnitude, jacobian):
            total = 0.
            for sign in (1, -1):
                shifted = sign * b * magnitude + a
                side = 1 + skew * jnp.where(shifted >= 0., 1., -1.)
                adjusted = shifted / side
                log_pdf = (jnp.log(b) + log_c
                           - .5 * (nu + 1) * jnp.log1p(adjusted * adjusted / (nu - 2)))
                log_power = delta * (jnp.log(magnitude) + jnp.log1p(-sign * gamma))
                total += jnp.sum(weight * jnp.exp(log_pdf + log_power + jnp.log(jacobian)))
            return total

        return interval(bounded_x, bounded_jacobian) + interval(unbounded_x, unbounded_jacobian)

    # The density and first parameter derivatives agree at its moving branch.
    # Holding this split fixed therefore preserves the complete second-order jet.
    return moment


@lru_cache(maxsize=4)
def _density_jet_function(boundary, nodes_count):
    jax, _, _ = _jax_modules()
    moment = _density_integral(boundary, nodes_count)
    return jax.jit(lambda point: (moment(point), jax.grad(moment)(point), jax.hessian(moment)(point)))


@lru_cache(maxsize=16)
def _density_reference(point, nodes_count):
    _, jnp, _ = _jax_modules()
    jet = _density_jet_function(_branch_boundary(point), nodes_count)(jnp.asarray(point))
    result = tuple(np.asarray(output, dtype=np.float64) for output in jet)
    assert all(np.isfinite(output).all() for output in result)
    return result


def _assert_jet_close(actual, expected, request=7):
    for bit, output, reference in zip((1, 2, 4), actual, expected, strict=True):
        if request & bit:
            assert np.isfinite(output).all() and np.isfinite(reference).all()
            rtol, atol = (VALUE_RTOL, VALUE_ATOL) if bit == 1 else (DERIVATIVE_RTOL, DERIVATIVE_ATOL)
            np.testing.assert_allclose(output, reference, rtol=rtol, atol=atol)


@pytest.fixture(scope="module", params=POINTS)
def refined_density(request):
    point = request.param
    coarse, fine = (_density_reference(point, count) for count in (256, 512))
    _assert_jet_close(coarse, fine)
    return point, fine


def test_canonical_density_integral_refines(refined_density):
    point, fine = refined_density
    assert abs(point[0]) < 1. and abs(point[3]) < 1.
    assert 0. < point[1] < point[2] and point[2] > 2.
    assert fine[0] > 0.
    np.testing.assert_allclose(fine[2], fine[2].T, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


def test_canonical_density_reflection_preserves_every_cross_block():
    original = _density_reference(RETAINED, 512)
    reflected = _density_reference(REFLECTED, 512)
    signs = np.array([-1., 1., 1., -1.])
    expected = (original[0], signs * original[1], signs[:, None] * original[2] * signs[None, :])
    _assert_jet_close(reflected, expected)


def test_public_skew_moment_all_masks_recover_and_preserve_outputs(refined_density):
    from scivol import _native_core

    point, expected = refined_density
    shape = np.ascontiguousarray(point[2:], dtype=np.float64)
    full = None
    for request in (7, 1, 2, 3, 4, 5, 6):
        value = np.full(1, -313.)
        gradient = np.arange(4, dtype=np.float64) + 91.
        hessian = np.arange(16, dtype=np.float64).reshape(4, 4) + 123.
        untouched = tuple(output.copy() for output in (value, gradient, hessian))
        assert _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, 2, point[0], point[1], request,
            value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data) == 0
        actual = (value[0], gradient, hessian)
        if request == 7:
            full = tuple(np.asarray(output).copy() for output in actual)
        _assert_jet_close(actual, expected, request)
        _assert_jet_close(actual, full, request)
        for bit, output, before in zip((1, 2, 4), (value, gradient, hessian), untouched, strict=True):
            if not request & bit:
                np.testing.assert_array_equal(output, before)


@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2)))
@pytest.mark.parametrize("power_coordinate", (float.fromhex("0x1.62e42fefa39efp-1"), .2),
                         ids=("retained-power", "central-curvature"))
def test_transformed_joint_aparch_recovery(monkeypatch, orders, power_coordinate):
    from scivol._devtools import ad_oracle
    from test_native_aparch import _evaluate, _forced_runtime, _map, _native_core

    p_ar, q_ma, p_vol, q_vol = orders
    mean_width = 1 + p_ar + q_ma
    gamma_index = mean_width + 1 + p_vol
    power_index = gamma_index + p_vol + q_vol
    coordinates = np.r_[np.zeros(mean_width), -2.5, np.full(p_vol, -3.),
                        np.full(p_vol, .15), np.full(q_vol, -1.),
                        power_coordinate,
                        float.fromhex("0x1.15288806261b3p-1"), .4]
    data = np.array([.7, -.8, .35, -.25, .6, -.45, .2, -.7])
    seed = float(np.mean(data * data))
    assert _native_core._arma_aparch_dispatch_kind(*orders, 2, 1) == int(orders == (1, 1, 1, 1))
    theta = _map(coordinates, 2, orders=orders, forward=True)
    local = np.array([RETAINED[0], 3. * (1. / (1. + math.exp(-power_coordinate))), *RETAINED[2:]])
    np.testing.assert_array_equal(theta[[gamma_index, power_index, power_index + 1, power_index + 2]],
                                  local)
    theta_reference = ad_oracle.arma_aparch_value_grad_hess(
        theta, data, *orders, "skewt", initial_variance=seed)
    theta_native = _evaluate(theta, data, 2, orders=orders, initial_variance=seed)
    assert theta_native[:2] == (0, 7)
    _assert_jet_close(theta_native[2:], theta_reference)

    # Keep the existing chart oracle; only its test-local moment quadrature changes.
    jax, jnp, _ = _jax_modules()
    density = jax.jit(_density_integral(_branch_boundary(RETAINED), 512))

    def chart_moment(jnp, jsp, gamma, delta, shape, density_name, quadrature_nodes=128):
        assert density_name == "skewt"
        return density(jnp.stack((gamma, delta, shape[0], shape[1])))

    monkeypatch.setattr(ad_oracle, "_jax_aparch_power_moment", chart_moment)
    expected = ad_oracle.native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, *theta_reference, p_vol, q_vol, "skewt", p_ar=p_ar, q_ma=q_ma)
    full = _evaluate(coordinates, data, 2, 7, transformed=True, orders=orders, initial_variance=seed)
    assert full[:2] == (0, 7)
    _assert_jet_close(full[2:], expected)
    assert np.max(np.abs(full[4][:mean_width, mean_width:])) > 1e-6
    for request in range(1, 8):
        fixed = _evaluate(coordinates, data, 2, request, transformed=True, orders=orders, initial_variance=seed)
        runtime = _forced_runtime(coordinates, data, 2, orders=orders, request=request,
                                  transformed=True, initial_variance=seed)
        assert fixed[:2] == (0, request)
        _assert_jet_close(fixed[2:], runtime, request)
        _assert_jet_close(fixed[2:], full[2:], request)
        _assert_jet_close(runtime, full[2:], request)
        _assert_jet_close(fixed[2:], expected, request)
        for bit, left, right in zip((1, 2, 4), fixed[2:], runtime, strict=True):
            if not request & bit:
                assert np.isnan(left).all() and np.isnan(right).all()
