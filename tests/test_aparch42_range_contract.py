"""APARCH range classification and original-coordinate MP AD regressions."""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import numpy as np
import pytest

from scivol import _native_core
from test_native_anchored_mean_chart import Jet
from test_native_aparch import _evaluate, _forced_runtime, _map
from test_native_aparch_subset_curvature import (
    exp_jet, jet_outputs, log_jet, loggamma_jet, unscaled_theta_oracle,
)


VALUE_ATOL = 2e-11
DERIVATIVE_RTOL = 2e-8
DERIVATIVE_ATOL = 2e-9
DATA = np.array([.7, -.8, .35, -.25, .6, -.45, .2, -.7])
DENSITIES = (("normal", 0), ("studentt", 1), ("skewt", 2))


def anchor(jet, value):
    return Jet(mp.mpf(float(value)), jet.gradient, jet.hessian)


def tanh_jet(jet):
    exponential = exp_jet(2 * jet)
    return (exponential - 1) / (exponential + 1)


def assert_outputs(actual, expected, mask=7):
    for bit, output, reference in zip((1, 2, 4), actual, expected, strict=True):
        if mask & bit:
            assert np.isfinite(output).all() and np.isfinite(reference).all()
            np.testing.assert_allclose(output, reference,
                rtol=0 if bit == 1 else DERIVATIVE_RTOL,
                atol=VALUE_ATOL if bit == 1 else DERIVATIVE_ATOL)
        else:
            assert np.isnan(output).all()


def symmetric_moment(gamma, delta, nu=None):
    if nu is None:
        log_absolute = (delta * mp.log(2) / 2 + loggamma_jet((delta + 1) / 2)
                        - mp.log(mp.pi) / 2)
    else:
        log_absolute = (delta * log_jet(nu - 2) / 2 + loggamma_jet((delta + 1) / 2)
            + loggamma_jet((nu - delta) / 2) - loggamma_jet(nu / 2) - mp.log(mp.pi) / 2)
    return exp_jet(log_absolute) * (
        exp_jet(delta * log_jet(1 - gamma)) + exp_jet(delta * log_jet(1 + gamma))) / 2


@lru_cache(maxsize=24)
def skew_moment_reference(point, nodes_count=128, digits=80):
    """Direct Hansen density integral in y=b*x/sqrt(nu-2), with forward MP AD.

    The integral has no narrow nu->2 mass. Its split is held fixed at the
    evaluation point; density and first derivatives agree across that split.
    """
    with mp.workdps(digits):
        gamma, delta, nu, skew = [Jet.variable(float(x), 4, i) for i, x in enumerate(point)]
        root = exp_jet(log_jet(nu - 2) / 2)
        log_c = loggamma_jet((nu + 1) / 2) - loggamma_jet(nu / 2) - mp.log(mp.pi) / 2
        center = 4 * skew * exp_jet(log_c) / (nu - 1)
        stretch = exp_jet(log_jet(1 + 3 * skew * skew - (root * center) * (root * center)) / 2)
        split = abs(center.value)
        nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
        integral = Jet.constant(0, 4)
        for node, weight in zip(nodes, weights, strict=True):
            unit = (mp.mpf(float(node)) + 1) / 2
            quadrature_weight = mp.mpf(float(weight)) / 2
            intervals = [(split + (unit / (1 - unit))**4, 4 * unit**3 / (1 - unit)**5)]
            if split:
                intervals.append((split * unit**4, 4 * split * unit**3))
            for magnitude, jacobian in intervals:
                for sign in (-1, 1):
                    shifted = center + sign * magnitude
                    side = 1 + skew if shifted.value >= 0 else 1 - skew
                    adjusted = shifted / side
                    log_pdf = log_c - (nu + 1) * log_jet(1 + adjusted * adjusted) / 2
                    log_power = delta * (mp.log(magnitude) + log_jet(1 - sign * gamma))
                    integral += exp_jet(log_pdf + log_power) * (quadrature_weight * jacobian)
        return exp_jet(delta * (log_jet(root) - log_jet(stretch))) * integral


def compose_local_moment(local, arguments):
    width = len(arguments[0].gradient)
    jacobian = mp.matrix([argument.gradient for argument in arguments])
    gradient = jacobian.T * mp.matrix(local.gradient)
    local_hessian = mp.matrix([local.hessian[i*4:(i+1)*4] for i in range(4)])
    hessian = jacobian.T * local_hessian * jacobian
    for coefficient, argument in zip(local.gradient, arguments, strict=True):
        hessian += coefficient * mp.matrix([
            argument.hessian[i*width:(i+1)*width] for i in range(width)])
    return Jet(local.value, tuple(gradient), tuple(hessian[i, j] for i in range(width) for j in range(width)))


def mapped_jets(orders, coordinates, theta, density, digits):
    ar, ma, p, q = orders
    width, omega = len(coordinates), 1 + ar + ma
    alpha, gamma, beta, delta_index = omega + 1, omega + 1 + p, omega + 1 + 2*p, omega + 1 + 2*p + q
    point = [Jet.variable(float(x), width, i) for i, x in enumerate(coordinates)]
    mapped = point.copy()
    for start, order, sign in ((1, ar, 1), (1 + ar, ma, -1)):
        coefficients = []
        for stage in range(order):
            reflection = tanh_jet(point[start + stage])
            previous = coefficients
            coefficients = [previous[i] - reflection * previous[stage - 1 - i]
                            for i in range(stage)] + [reflection]
        mapped[start:start + order] = [anchor(sign * x, theta[start + i])
                                       for i, x in enumerate(coefficients)]
    mapped[omega] = anchor(mp.mpf(float(np.finfo(float).tiny)) + log_jet(1 + exp_jet(point[omega])), theta[omega])
    mapped[gamma:gamma + p] = [anchor(tanh_jet(point[gamma + i]), theta[gamma + i]) for i in range(p)]
    if density == "normal":
        mapped[delta_index] = anchor(mp.mpf(float(np.finfo(float).tiny))
            + log_jet(1 + exp_jet(point[delta_index])), theta[delta_index])
    else:
        nu_index = delta_index + 1
        mapped[nu_index] = anchor(mp.mpf(float(2 + 8*np.finfo(float).eps))
            + log_jet(1 + exp_jet(point[nu_index])), theta[nu_index])
        mapped[delta_index] = anchor(mapped[nu_index] / (1 + exp_jet(-point[delta_index])), theta[delta_index])
        if density == "skewt":
            mapped[-1] = anchor(tanh_jet(point[-1]), theta[-1])
    weights = [exp_jet(x) for x in point[alpha:alpha + p] + point[beta:beta + q]]
    limit = mp.mpf(float(1 - 8*(p + q + 1)*np.finfo(float).eps))
    contributions = [limit * weight / (1 + sum(weights)) for weight in weights]
    for lag in range(p):
        arguments = [mapped[gamma + lag], mapped[delta_index]]
        if density == "normal":
            moment = symmetric_moment(*arguments)
        elif density == "studentt":
            moment = symmetric_moment(*arguments, mapped[-1])
        else:
            arguments += mapped[-2:]
            local = skew_moment_reference(tuple(float(x.value) for x in arguments), 256, digits)
            moment = compose_local_moment(local, arguments)
        mapped[alpha + lag] = anchor(contributions[lag] / moment, theta[alpha + lag])
    mapped[beta:beta + q] = [anchor(x, theta[beta + i]) for i, x in enumerate(contributions[p:])]
    return mapped


def original_z_oracle(orders, coordinates, theta, data, h0, density, digits=80):
    with mp.workdps(digits):
        objective = unscaled_theta_oracle(orders, theta, data, h0, digits=digits, density=density)
        mapped = mapped_jets(orders, coordinates, theta, density, digits)
        width = theta.size
        jacobian = mp.matrix([x.gradient for x in mapped])
        gradient = jacobian.T * mp.matrix(objective.gradient)
        hessian = jacobian.T * mp.matrix([
            objective.hessian[i*width:(i+1)*width] for i in range(width)]) * jacobian
        for coefficient, parameter in zip(objective.gradient, mapped, strict=True):
            hessian += coefficient * mp.matrix([parameter.hessian[i*width:(i+1)*width] for i in range(width)])
        return Jet(objective.value, tuple(gradient), tuple(hessian[i, j] for i in range(width) for j in range(width)))


def coordinate_case(orders, density, *, plateau=False):
    ar, ma, p, q = orders
    mean = np.r_[.03, np.full(ar, .12 / max(ar, 1)), np.full(ma, -.08 / max(ma, 1))]
    shape = [] if density == "normal" else [8.] if density == "studentt" else [8., -.25]
    theta = np.r_[mean, .08, np.full(p, .07 / p), np.full(p, .15), np.full(q, .72 / max(q, 1)), 1.6, shape]
    code = dict(DENSITIES)[density]
    coordinates = _map(theta, code, orders=orders)
    if plateau:
        coordinates[1] = 18.5
        if density != "normal":
            coordinates[-2 if density == "skewt" else -1] = -37.
    return coordinates, _map(coordinates, code, orders=orders, forward=True)


@pytest.mark.parametrize("density,code", DENSITIES)
@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2), (0, 0, 1, 1)))
def test_joint_original_z_all_masks_and_arbitrary_orders(density, code, orders):
    coordinates, theta = coordinate_case(orders, density)
    if density == "skewt":
        for lag in range(orders[2]):
            gamma = theta[1 + orders[0] + orders[1] + 1 + orders[2] + lag]
            point = (gamma, theta[-3], theta[-2], theta[-1])
            coarse, fine = (jet_outputs(skew_moment_reference(point, n, 80)) for n in (128, 256))
            assert_outputs(coarse, fine)
    low, high = [original_z_oracle(orders, coordinates, theta, DATA, .8, density, digits) for digits in (80, 120)]
    expected = jet_outputs(high)
    assert_outputs(jet_outputs(low), expected)
    full = _evaluate(coordinates, DATA, code, 7, transformed=True, orders=orders, initial_variance=.8)
    assert full[:2] == (0, 7)
    assert np.max(np.abs(full[4][:1 + orders[0] + orders[1], 1 + orders[0] + orders[1]:])) > 1e-6
    for mask in range(1, 8):
        actual = _evaluate(coordinates, DATA, code, mask, transformed=True, orders=orders, initial_variance=.8)
        assert actual[:2] == (0, mask)
        runtime = _forced_runtime(coordinates, DATA, code, orders=orders, request=mask,
                                  transformed=True, initial_variance=.8)
        assert_outputs(actual[2:], expected, mask)
        assert_outputs(runtime, expected, mask)
        assert_outputs(actual[2:], full[2:], mask)
    assert _native_core._arma_aparch_dispatch_kind(*orders, code, 1) == int(orders == (1, 1, 1, 1))


@pytest.mark.parametrize("density,code", DENSITIES[:2])
def test_original_z_rounding_plateau_is_not_reconstructed(density, code):
    orders = (1, 1, 1, 1)
    coordinates, theta = coordinate_case(orders, density, plateau=True)
    try:
        recovered = _map(theta, code, orders=orders)
    except ValueError:
        recovered = None
    assert recovered is None or not np.array_equal(recovered, coordinates)
    expected = jet_outputs(original_z_oracle(orders, coordinates, theta, DATA, .8, density, 120))
    for mask in range(1, 8):
        actual = _evaluate(coordinates, DATA, code, mask, transformed=True, orders=orders, initial_variance=.8)
        assert actual[:2] == (0, mask)
        assert_outputs(actual[2:], expected, mask)


@pytest.mark.parametrize("density,code", DENSITIES[1:])
def test_subnormal_power_rejection_has_unrepresentable_additive_value(density, code):
    coordinates = np.array([0., 0., 0., -2.5, -3., .15, -1., -744.,
        float.fromhex("0x1.7fd7567401aafp+2"), *([0.] if density == "skewt" else [])])
    theta = _map(coordinates, code, orders=(1, 1, 1, 1), forward=True)
    assert theta[7] == 12 * np.nextafter(0., 1.)
    with mp.workdps(100):
        power = mp.mpf(float(theta[3])) + mp.mpf(float(theta[6])) * mp.power(mp.mpf(.8), mp.mpf(float(theta[7])) / 2)
        log_variance = 2 * mp.log(power) / mp.mpf(float(theta[7]))
        # The first scored nonzero residual has NLL ~ -nu*log(h)/2.
        lower_bound = -mp.mpf(float(theta[8])) * log_variance / 2 - 1000
        assert lower_bound > mp.mpf(float(np.finfo(float).max))
    for mask in range(1, 8):
        result = _evaluate(coordinates, DATA, code, mask, transformed=True,
                           orders=(1, 1, 1, 1), initial_variance=.8)
        assert result[:2] == (3, 0)
        for bit, output in zip((1, 2, 4), result[2:], strict=True):
            if not mask & bit:
                assert np.isnan(output).all()


@pytest.mark.parametrize("density,code", DENSITIES[1:])
def test_power_rounds_to_nu_outside_the_true_mapped_domain(density, code):
    coordinates, _ = coordinate_case((1, 1, 1, 1), density)
    coordinates[7] = 40.
    with mp.workdps(100):
        nu = mp.mpf(8.)
        exact = nu / (1 + mp.exp(-40))
        assert exact < nu and float(exact) == float(nu)
    with pytest.raises((ValueError, RuntimeError)):
        _map(coordinates, code, orders=(1, 1, 1, 1), forward=True)
    for mask in range(1, 8):
        result = _evaluate(coordinates, DATA, code, mask, transformed=True,
                           orders=(1, 1, 1, 1), initial_variance=.8)
        assert result[:2] == (3, 0)


@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2)))
@pytest.mark.parametrize("skew_coordinate", (-.4, .4))
@pytest.mark.parametrize("nu_coordinate", (-37., -40.))
def test_skew_nu_plateau_requires_finite_original_z_derivatives(orders, skew_coordinate, nu_coordinate):
    """Known core019 failure; this test must pass before integrating the repair."""
    ar, ma, p, q = orders
    coordinates = np.r_[np.zeros(1 + ar + ma), -2.5, np.full(p, -3.),
                         np.full(p, .15), np.full(q, -1.), -1., nu_coordinate, skew_coordinate]
    theta = _map(coordinates, 2, orders=orders, forward=True)
    assert theta[-2] == 2 + 8*np.finfo(float).eps
    point = (theta[2 + ar + ma + p], theta[-3], theta[-2], theta[-1])
    coarse, fine = (jet_outputs(skew_moment_reference(point, n, 100)) for n in (128, 256))
    assert_outputs(coarse, fine)
    low, high = [original_z_oracle(orders, coordinates, theta, DATA, .8, "skewt", digits)
                 for digits in (80, 120)]
    expected = jet_outputs(high)
    assert_outputs(jet_outputs(low), expected)
    theta_reference = jet_outputs(unscaled_theta_oracle(orders, theta, DATA, .8, digits=120, density="skewt"))
    for mask in range(1, 8):
        direct = _evaluate(theta, DATA, 2, mask, orders=orders, initial_variance=.8)
        assert direct[:2] == (0, mask)
        assert_outputs(direct[2:], theta_reference, mask)
        actual = _evaluate(coordinates, DATA, 2, mask, transformed=True, orders=orders, initial_variance=.8)
        assert actual[:2] == (0, mask)
        assert_outputs(actual[2:], expected, mask)
        runtime = _forced_runtime(coordinates, DATA, 2, request=mask, transformed=True,
                                  orders=orders, initial_variance=.8)
        assert_outputs(runtime, expected, mask)


@pytest.mark.parametrize("density,code", DENSITIES)
def test_long_near_boundary_recursive_sensitivities(density, code):
    orders = (1, 1, 1, 1)
    shape = [] if density == "normal" else [8.] if density == "studentt" else [8., -.25]
    theta = np.array([.03, .12, .997, .04, .01, .1, .97, 1.6, *shape])
    data = np.resize(np.array([.8, -.6]), 96)
    coordinates = _map(theta, code, orders=orders)
    theta = _map(coordinates, code, orders=orders, forward=True)
    low, high = [unscaled_theta_oracle(orders, theta, data, .8, digits=digits, density=density)
                 for digits in (80, 120)]
    expected_theta = jet_outputs(high)
    assert_outputs(jet_outputs(low), expected_theta)
    assert np.max(np.abs(expected_theta[2][:3, :3])) > 1e4
    expected_z = jet_outputs(original_z_oracle(orders, coordinates, theta, data, .8, density, 120))
    for transformed, point, expected in ((False, theta, expected_theta), (True, coordinates, expected_z)):
        for mask in range(1, 8):
            actual = _evaluate(point, data, code, mask, transformed=transformed, orders=orders, initial_variance=.8)
            assert actual[:2] == (0, mask)
            runtime = _forced_runtime(point, data, code, request=mask, transformed=transformed,
                                      orders=orders, initial_variance=.8)
            assert_outputs(actual[2:], expected, mask)
            assert_outputs(runtime, expected, mask)


def test_normal_tiny_power_is_a_value_representability_failure():
    coordinates = np.array([0., 0., 0., -2.5, -3., .15, -1., -744.])
    theta = _map(coordinates, 0, orders=(1, 1, 1, 1), forward=True)
    assert theta[-1] == np.finfo(float).tiny + np.exp(-744.)
    with mp.workdps(100):
        power = mp.mpf(float(theta[3])) + mp.mpf(float(theta[6]))
        log_variance = 2 * mp.log(power) / mp.mpf(float(theta[-1]))
        log_standardized_square = 2 * mp.log(abs(mp.mpf(float(DATA[1]))))
        assert log_standardized_square - log_variance > mp.log(mp.mpf(float(np.finfo(float).max)))
    for mask in range(1, 8):
        result = _evaluate(coordinates, DATA, 0, mask, transformed=True,
                           orders=(1, 1, 1, 1), initial_variance=.8)
        assert result[:2] == (3, 0)


@pytest.mark.parametrize("density,code", DENSITIES)
@pytest.mark.parametrize("power,maximum_order", ((.5, 0), (1.5, 1), (2., 1)))
def test_actual_zero_shock_has_request_specific_differentiability(density, code, power, maximum_order):
    shape = [] if density == "normal" else [8.] if density == "studentt" else [8., 0.]
    theta = np.array([0., 0., 0., .08, .04, .15, .6, power, *shape])
    data = np.array([.4, 0., .6, -.5])
    # For alpha>0, (|e|-gamma*e)^power at e=0 has no first
    # derivative below power=1, no second below power=2, and unequal
    # one-sided second derivatives at power=2 with gamma!=0.
    for mask in range(1, 8):
        requested_order = 2 if mask & 4 else 1 if mask & 2 else 0
        result = _evaluate(theta, data, code, mask, orders=(1, 1, 1, 1), initial_variance=.8)
        assert result[:2] == ((0, mask) if requested_order <= maximum_order else (3, 0))
        for bit, output in zip((1, 2, 4), result[2:], strict=True):
            if not mask & bit:
                assert np.isnan(output).all()
