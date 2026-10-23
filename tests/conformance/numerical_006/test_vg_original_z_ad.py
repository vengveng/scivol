"""Retained VG joint MP likelihood with independent original-z map AD.

The moment AD uses the existing VG density Jet in a normalized MP integral. Native calls
provide only the rounded scalar theta anchor and the requests under test.
"""

from functools import lru_cache

import mpmath as mp
import numpy as np
import pytest

from .oracle_mp import compose_partials
from .oracle_mp import Jet
from .oracle_mp import exp_jet, log_jet
from .test_vg_retained_value_accuracy import (
    RETAINED_THETA, retained_joint_oracle, retained_sample,
)


DERIVATIVE_RTOL = 3e-10
DERIVATIVE_ATOL = 2e-11


@lru_cache(maxsize=12)
def independent_moment(point, digits, outer, inner):
    from .oracle_vg_moment import normalized_vg_moment

    return normalized_vg_moment(point, digits=digits, outer=outer, inner=inner)


def anchor_scalar(parameter, value):
    return Jet(mp.mpf(float(value)), parameter.gradient, parameter.hessian)


def retained_mapped_parameters(model, coordinates, theta, *, digits, outer, inner):
    """Compose the map at original z, using mapped doubles for local moments."""
    width, m, p, q = model.size, model.columns, model.p_vol, model.q_vol
    alpha, gamma, beta = m+1, m+1+p, m+1+2*p
    delta, shape = model.shape_start-1, model.shape_start
    point = [Jet.variable(float(v), width, i) for i, v in enumerate(coordinates)]
    mapped = point.copy()

    def tanh(parameter):
        exponential = exp_jet(2*parameter)
        return (exponential-1)/(exponential+1)

    def bounded(parameter, lower, upper):
        tail = exp_jet(-parameter if parameter.value >= 0 else parameter)
        tail = tail/(1+tail)
        return upper-(upper-lower)*tail if parameter.value >= 0 else lower+(upper-lower)*tail

    tiny = mp.mpf(float(np.finfo(float).tiny))
    for index in (m, delta):
        mapped[index] = anchor_scalar(tiny+log_jet(1+exp_jet(point[index])), theta[index])
    for index in range(gamma, gamma+p):
        mapped[index] = anchor_scalar(tanh(point[index]), theta[index])
    mapped[shape] = anchor_scalar(mp.mpf(.995)*tanh(point[shape]), theta[shape])
    mapped[shape+1] = anchor_scalar(bounded(point[shape+1],
        mp.mpf(1.5000000000000018), mp.mpf(50)), theta[shape+1])
    logits = point[alpha:alpha+p]+point[beta:beta+q]
    maximum = max(mp.mpf(0), *(parameter.value for parameter in logits))
    weights = [exp_jet(parameter-maximum) for parameter in logits]
    denominator = mp.exp(-maximum)+sum(weights)
    limit = mp.mpf(float(1-8*(p+q+1)*np.finfo(float).eps))
    contributions = [limit*weight/denominator for weight in weights]
    for lag in range(p):
        arguments = (mapped[gamma+lag], mapped[delta], mapped[shape], mapped[shape+1])
        local = independent_moment(tuple(float(parameter.value) for parameter in arguments), digits, outer, inner)
        assert all(mp.isfinite(value) for value in (local.value, *local.gradient, *local.hessian))
        moment = compose_partials(local.value, local.gradient, local.hessian, arguments)
        mapped[alpha+lag] = anchor_scalar(contributions[lag]/moment, theta[alpha+lag])
    mapped[beta:beta+q] = [anchor_scalar(value, scalar) for value, scalar in
                            zip(contributions[p:], theta[beta:beta+q], strict=True)]
    return mapped


def retained_original_z_point(model):
    """Use the retained basin and explicit feasible saturation-sensitive tails."""

    seed = RETAINED_THETA.copy()
    seed[-1] = np.nextafter(50., 0.)
    coordinates = model.inverse(seed)
    gamma = model.columns+1+model.p_vol
    coordinates[[gamma, gamma+2, model.shape_start]] = [-18.7, 18.7, -18.7]
    coordinates[-1] = 40.
    # Linear means map identically. The public standalone entry wraps the same
    # volatility map owner; F parity below checks the complete consumer's anchor.
    theta = model.map(coordinates, forward=True)
    assert theta[-1] == 50.
    assert theta[gamma] == -np.nextafter(1., 0.)
    assert theta[gamma+2] == np.nextafter(1., 0.)
    assert theta[model.shape_start] == -np.nextafter(.995, 0.)
    with pytest.raises(ValueError, match="outside the transformed chart"):
        model.inverse(theta)
    return coordinates, theta


def retained_original_z_oracle(model, coordinates, theta, *, digits, count, outer, inner):
    with mp.workdps(digits):
        parameters = retained_mapped_parameters(model, coordinates, theta, digits=digits, outer=outer, inner=inner)
        return retained_joint_oracle(theta, digits=digits, count=count,
            parameters=parameters, model=model, return_jet=True)


def assert_retained_outputs(actual, reference, mask):
    for bit, output, target in zip((1, 2, 4), actual[:3], reference, strict=True):
        if mask & bit:
            assert np.isfinite(output).all() and np.isfinite(target).all()
            np.testing.assert_allclose(output, target,
                rtol=0 if bit == 1 else DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)
        else:
            assert output is None


def saturated_pullback_budget(theta_reference, parameters):
    """Propagate the established theta gates; no measured native errors enter."""
    gradient, hessian = theta_reference[1:3]
    width = len(parameters)
    jacobian = np.array([[float(v) for v in parameter.gradient] for parameter in parameters])
    map_hessians = np.array([[float(v) for v in parameter.hessian] for parameter in parameters]).reshape(width, width, width)
    assert all(np.isfinite(v).all() for v in (gradient, hessian, jacobian, map_hessians))
    absolute_jacobian, absolute_curvature = np.abs(jacobian), np.abs(map_hessians)
    theta_gradient_gate = DERIVATIVE_ATOL+DERIVATIVE_RTOL*np.abs(gradient)
    theta_hessian_gate = DERIVATIVE_ATOL+DERIVATIVE_RTOL*np.abs(hessian)
    gradient_budget = absolute_jacobian.T @ theta_gradient_gate
    hessian_budget = (absolute_jacobian.T @ theta_hessian_gate @ absolute_jacobian
        +np.einsum("i,ijk->jk", theta_gradient_gate, absolute_curvature))
    # Match the existing joint test's absolute-contraction rounding bound.
    # Absolute map Hessians also cover cancellation within the curvature sum.
    rounding = 64*np.finfo(float).eps
    gradient_budget += rounding*(absolute_jacobian.T @ np.abs(gradient))
    hessian_budget += rounding*(absolute_jacobian.T @ np.abs(hessian) @ absolute_jacobian
        +np.einsum("i,ijk->jk", np.abs(gradient), absolute_curvature))
    assert all(np.isfinite(value).all() and np.all(value >= 0) for value in (gradient_budget, hessian_budget))
    return gradient_budget, hessian_budget


def assert_saturated_outputs(actual, reference, mask, budgets):
    for bit, output, target, budget in zip((2, 4), actual[1:3], reference[1:3], budgets, strict=True):
        if mask & bit:
            assert np.isfinite(output).all()
            assert np.all(np.abs(output[-2:]-target[-2:]) <= budget[-2:]), "Saturated pullback exceeds propagated theta verification budget"


def assert_saturated_zeroing_rejected(reference, budgets):
    """Each tiny shape score and diagonal curvature must remain detectable."""
    for bit, index in ((2, 1), (4, 2)):
        for coordinate in (-2, -1):
            altered = [None, reference[1].copy(), reference[2].copy()]
            target = coordinate if bit == 2 else (coordinate, coordinate)
            assert altered[index][target] != 0
            altered[index][target] = 0
            with pytest.raises(AssertionError, match="Saturated pullback"):
                assert_saturated_outputs(altered, reference, bit, budgets)






@pytest.fixture(scope="module")
def retained_original_z_ad():
    from .oracle_mp import jet_outputs

    model = retained_sample()
    coordinates, theta = retained_original_z_point(model)
    references = [jet_outputs(retained_original_z_oracle(model, coordinates, theta,
        digits=digits, count=count, outer=outer, inner=inner))
        for digits, count, outer, inner in ((80, 32, 24, 32), (120, 48, 32, 48))]
    for coarse, refined in zip(*references, strict=True):
        np.testing.assert_allclose(coarse, refined, rtol=3e-11, atol=2e-12)
    with mp.workdps(120):
        theta_jet = retained_joint_oracle(theta, digits=120, count=48, model=model, return_jet=True)
        theta_reference = jet_outputs(theta_jet)
        parameters = retained_mapped_parameters(model, coordinates, theta, digits=120, outer=32, inner=48)
        budgets = saturated_pullback_budget(theta_reference, parameters)
        assert_saturated_zeroing_rejected(references[1], budgets)
    return coordinates, theta, references[1], theta_reference, budgets


@pytest.mark.parametrize("fixed", [False, True])
def test_retained_full_joint_vg_original_z_all_masks_match_ad(retained_original_z_ad, fixed):
    coordinates, theta, reference, theta_reference, budgets = retained_original_z_ad
    model, runtime = retained_sample(fixed=fixed), retained_sample()
    assert model.columns == 7 and (model.p_vol, model.q_vol) == (3, 2)
    mean, volatility, density = slice(0, model.columns), slice(model.columns, model.shape_start), slice(model.shape_start, None)
    for left, right in ((mean, volatility), (mean, density), (volatility, density)):
        assert np.max(np.abs(reference[2][left, right])) > np.finfo(float).tiny
    full = model.evaluate(coordinates, 7, transformed=True)
    assert abs(full[0]-model.evaluate(theta, 1)[0]) <= 2e-11
    assert_retained_outputs(model.evaluate(theta, 7), theta_reference, 7)
    before = coordinates.tobytes(), theta.tobytes(), model.data.tobytes(), model.features.tobytes()
    for mask in range(1, 8):
        actual = model.evaluate(coordinates, mask, transformed=True)
        general = runtime.evaluate(coordinates, mask, transformed=True)
        assert_retained_outputs(actual, reference, mask)
        assert_retained_outputs(actual, full[:3], mask)
        assert_retained_outputs(actual, general[:3], mask)
        for outputs in (actual, general):
            assert_saturated_outputs(outputs, reference, mask, budgets)
    assert before == (coordinates.tobytes(), theta.tobytes(), model.data.tobytes(), model.features.tobytes())
