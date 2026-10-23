"""Public Normal/Student-t APARCH adaptive mean precision, without fits or private bridges."""

from __future__ import annotations

from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from scivol import _native_core
from test_native_anchored_mean_chart import Jet, Spec, exact_conditioned_residuals
from test_native_aparch import _evaluate, _forced_runtime, _map
from test_native_aparch_subset_curvature import (
    exp_jet,
    jet_outputs,
    log_jet,
    loggamma_jet,
    sample,
    unscaled_theta_oracle,
)


def cancellation_case():
    # Equal AR and inverse-MA polynomials give epsilon[t] == data[t] exactly.
    # Former DD routing left ~3.6e-35 at t=12, corrupting fractional curvature.
    theta = np.array([0., .123456789, -.123456789, .08, .08, .1, .6, 1.6])
    data = np.array([0., -.4, 1.1, .3, -1.2, .7, .6, -.2,
                     1.4, -.9, .5, .2, 1e-40, .9])
    return (1, 1, 1, 1), theta, data, .8


def filter_reference(orders, theta, data, h0):
    """Exact tracked mean owner plus MP scalar variance recursion."""
    ar, ma, p, q = orders
    start, offset = max(orders), 1 + ar + ma
    spec = Spec.make((ar, ma, start, theta.size - offset), count=data.size)
    rational = exact_conditioned_residuals(spec, theta, data)
    with mp.workdps(120):
        residuals = [mp.mpf(x.numerator) / x.denominator for x in rational]
        point = list(map(lambda x: mp.mpf(float(x)), theta))
        delta = point[offset + 1 + 2*p + q]
        powers = [mp.mpf(float(h0)) ** (delta / 2)] * data.size
        variance = [mp.mpf(float(h0))] * data.size
        for t in range(start, data.size):
            power = point[offset]
            for lag in range(p):
                previous = t - lag - 1
                if previous >= start:
                    error = residuals[previous]
                    base = abs(error) - point[offset + 1 + p + lag] * error
                    power += point[offset + 1 + lag] * base**delta
            for lag in range(q):
                power += point[offset + 1 + 2*p + lag] * powers[t - lag - 1]
            powers[t] = power
            variance[t] = power ** (2 / delta)
        return np.array(list(map(float, residuals))), np.array(list(map(float, variance)))


@pytest.fixture(scope="module", params=[("normal", 0), ("studentt", 1)])
def precision_density(request):
    return request.param


@pytest.fixture(scope="module", params=["cancellation", "general-cancellation", "zero-ma"])
def precision_case(request, precision_density):
    density, code = precision_density
    if request.param == "cancellation":
        orders, theta, data, h0 = cancellation_case()
    elif request.param == "general-cancellation":
        _, base, data, h0 = cancellation_case()
        orders = (2, 1, 3, 2)
        theta = np.array([0., base[1], 0., base[2], .08,
                          .03, .03, .02, .1, .1, .1, .3, .3, 1.6])
        data[:max(orders)] = 0.
    else:
        orders, h0 = (1, 0, 1, 2), .8
        theta, data = sample(orders)
    if code == 1:
        theta = np.r_[theta, 8.]
    # Refine once per point, then project the same oracle onto every request.
    low, high = [unscaled_theta_oracle(orders, theta, data, h0, digits=digits, density=density)
                 for digits in (80, 120)]
    assert isinstance(low, Jet) and isinstance(high, Jet)
    with mp.workdps(120):
        refinement = max(abs(a-b) / (1 + abs(b)) for a, b in zip(
            (low.value, *low.gradient, *low.hessian),
            (high.value, *high.gradient, *high.hessian)))
        assert refinement < mp.mpf("1e-40"), str(refinement)
    return SimpleNamespace(name=request.param, density=density, code=code,
                           orders=orders, theta=theta, data=data,
                           h0=h0, expected=jet_outputs(high), refinement=str(refinement),
                           paths=filter_reference(orders, theta, data, h0))


def assert_requested(actual, expected, mask):
    for bit, output, reference in zip((1, 2, 4), actual, expected):
        if not mask & bit:
            assert np.isnan(output).all()
            continue
        assert np.isfinite(output).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(output, reference, atol=2e-11,
                                   rtol=0 if bit == 1 else 3e-10)


def mapped_parameter_jets(orders, coordinates, density):
    """Independent MP AD of the declared Schur, moment and simplex maps."""
    ar, ma, p, q = orders
    omega = 1 + ar + ma
    alpha, gamma, beta, delta = omega + 1, omega + 1 + p, omega + 1 + 2*p, omega + 1 + 2*p + q
    point = [Jet.variable(float(value), len(coordinates), i) for i, value in enumerate(coordinates)]
    mapped = point.copy()

    def softplus(x):
        return log_jet(1 + exp_jet(x))

    def tanh(x):
        exponential = exp_jet(2*x)
        return (exponential - 1) / (exponential + 1)

    for start, order, sign in ((1, ar, 1), (1 + ar, ma, -1)):
        coefficients = []
        for stage in range(order):
            reflection = tanh(point[start + stage])
            previous = coefficients
            coefficients = [previous[i] - reflection * previous[stage - 1 - i]
                            for i in range(stage)] + [reflection]
        mapped[start:start + order] = [sign * x for x in coefficients]
    mapped[omega] = mp.mpf(float(np.finfo(float).tiny)) + softplus(point[omega])
    mapped[gamma:gamma + p] = [tanh(x) for x in point[gamma:gamma + p]]
    if density == "studentt":
        nu = mapped[-1] = mp.mpf(float(2 + 8*np.finfo(float).eps)) + softplus(point[-1])
        power = mapped[delta] = nu / (1 + exp_jet(-point[delta]))
        absolute_moment = exp_jet(power * log_jet(nu - 2) / 2
            + loggamma_jet((power + 1) / 2) + loggamma_jet((nu - power) / 2)
            - mp.log(mp.pi) / 2 - loggamma_jet(nu / 2))
    else:
        assert density == "normal"
        power = mapped[delta] = mp.mpf(float(np.finfo(float).tiny)) + softplus(point[delta])
        absolute_moment = exp_jet(power * mp.log(2) / 2
            + loggamma_jet((power + 1) / 2) - mp.log(mp.pi) / 2)
    weights = [exp_jet(x) for x in point[alpha:alpha + p] + point[beta:beta + q]]
    limit = mp.mpf(float(1 - 8*(p + q + 1)*np.finfo(float).eps))
    denominator = 1 + sum(weights)
    contributions = [limit * weight / denominator for weight in weights]
    for lag in range(p):
        asymmetry = (exp_jet(power * log_jet(1 - mapped[gamma + lag]))
                     + exp_jet(power * log_jet(1 + mapped[gamma + lag]))) / 2
        mapped[alpha + lag] = contributions[lag] / (absolute_moment * asymmetry)
    mapped[beta:beta + q] = contributions[p:]
    return mapped


def coordinate_oracle(case, coordinates, theta, digits):
    # Evaluate at the actual theta doubles, but differentiate at the original z.
    with mp.workdps(digits):
        objective = unscaled_theta_oracle(case.orders, theta, case.data, case.h0,
                                         digits=digits, density=case.density)
        mapped = mapped_parameter_jets(case.orders, coordinates, case.density)
        width = theta.size
        jacobian = mp.matrix([x.gradient for x in mapped])
        gradient = jacobian.T * mp.matrix(objective.gradient)
        theta_hessian = mp.matrix([objective.hessian[i*width:(i+1)*width] for i in range(width)])
        hessian = jacobian.T * theta_hessian * jacobian
        for coefficient, parameter in zip(objective.gradient, mapped):
            curvature = mp.matrix([parameter.hessian[i*width:(i+1)*width] for i in range(width)])
            hessian += coefficient * curvature
        return Jet(objective.value, tuple(gradient), tuple(hessian[i,j] for i in range(width) for j in range(width)))


@pytest.mark.parametrize("plateau", [False, True])
def test_original_coordinate_sparse_requests(precision_case, plateau):
    case = precision_case
    coordinates = _map(case.theta, case.code, orders=case.orders)
    if plateau:
        coordinates[1] = 18.5
        if case.orders[1]:
            coordinates[1 + case.orders[0]] = 18.5
    theta = _map(coordinates, case.code, orders=case.orders, forward=True)
    if plateau:
        assert _map(theta, case.code, orders=case.orders)[1] != coordinates[1]
    low, high = [coordinate_oracle(case, coordinates, theta, digits) for digits in (80, 120)]
    with mp.workdps(120):
        assert max(abs(a-b)/(1+abs(b)) for a,b in zip(
            (low.value, *low.gradient, *low.hessian),
            (high.value, *high.gradient, *high.hessian))) < mp.mpf("1e-40")
    expected = jet_outputs(high)
    for mask in range(1, 8):
        status, valid, *actual = _evaluate(coordinates, case.data, case.code, mask,
            orders=case.orders, initial_variance=case.h0, transformed=True)
        assert (status, valid) == (0, mask)
        runtime = _forced_runtime(coordinates, case.data, case.code, request=mask,
            orders=case.orders, initial_variance=case.h0, transformed=True)
        assert_requested(actual, expected, mask)
        assert_requested(runtime, expected, mask)


def test_public_full_theta_sparse_requests_and_runtime_identity(precision_case):
    case = precision_case
    before = case.theta.copy(), case.data.copy()
    assert _native_core._arma_aparch_dispatch_kind(*case.orders, case.code, 0) == (
        1 if case.name == "cancellation" else 0)
    status, valid, *full = _evaluate(case.theta, case.data, case.code, orders=case.orders,
                                   initial_variance=case.h0)
    assert (status, valid) == (0, 7)
    assert_requested(full, case.expected, 7)
    for mask in range(1, 8):
        status, valid, *actual = _evaluate(case.theta, case.data, case.code, mask,
            orders=case.orders, initial_variance=case.h0)
        assert (status, valid) == (0, mask)
        runtime = _forced_runtime(case.theta, case.data, case.code, request=mask,
            orders=case.orders, initial_variance=case.h0)
        assert_requested(actual, case.expected, mask)
        assert_requested(runtime, case.expected, mask)
        assert_requested(actual, full, mask)
        for bit, fixed, general in zip((1, 2, 4), actual, runtime):
            if mask & bit:
                np.testing.assert_allclose(fixed, general, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(case.theta, before[0])
    np.testing.assert_array_equal(case.data, before[1])


def test_canonical_filter_residuals_and_variances(precision_case):
    case = precision_case
    before = case.theta.copy(), case.data.copy()
    required = _native_core._arma_aparch_filter_workspace_bytes(*case.orders, case.code)
    assert required > 0
    workspace = np.empty(required, dtype=np.uint8)
    residuals, variance = np.full_like(case.data, np.nan), np.full_like(case.data, np.nan)
    status = _native_core._arma_aparch_filter_state(case.theta.ctypes.data,
        case.data.ctypes.data, case.data.size, case.h0, *case.orders, case.code,
        workspace.ctypes.data, workspace.nbytes, residuals.ctypes.data, variance.ctypes.data)
    assert status == 0
    assert np.isfinite(residuals).all() and np.isfinite(variance).all()
    np.testing.assert_allclose(residuals, case.paths[0], rtol=3e-10, atol=0)
    np.testing.assert_allclose(variance, case.paths[1], rtol=3e-10, atol=2e-11)
    if "cancellation" in case.name:
        assert residuals[12] == case.data[12] != 0
    np.testing.assert_array_equal(case.theta, before[0])
    np.testing.assert_array_equal(case.data, before[1])


@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (2, 1, 3, 2)])
@pytest.mark.parametrize("delta,gamma,alpha,derivative_bits", [
    (.8, .2, .07, 0), (1.7, .2, .07, 2), (2., .2, .07, 2),
    (2., 0., .07, 6), (2.5, .2, .07, 6), (.5, .2, 0., 2), (1.7, .2, 0., 6),
])
def test_student_exact_zero_keeps_request_specific_differentiability(
    orders, delta, gamma, alpha, derivative_bits,
):
    ar, ma, p, q = orders
    theta = np.r_[np.zeros(1 + ar + ma), .08, np.full(p, alpha/p),
                  np.full(p, gamma), np.full(q, .6/q), delta, 8.]
    data = np.r_[np.zeros(max(orders)), 0., 1., 0., 1.]
    for mask in range(1, 8):
        status, valid, *outputs = _evaluate(theta, data, 1, mask,
            orders=orders, initial_variance=1.)
        differentiable = not (mask & 6 & ~derivative_bits)
        assert (status, valid) == ((0, mask) if differentiable else (3, 0))
        for bit, output in zip((1, 2, 4), outputs):
            if not mask & bit:
                assert np.isnan(output).all()
            elif differentiable:
                assert np.isfinite(output).all()
