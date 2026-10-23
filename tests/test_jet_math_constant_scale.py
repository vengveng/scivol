"""Constant-variance jets retain finite standardized density information."""

import mpmath as mp
import numpy as np
import pytest

from scivol._devtools.mean_oracle import mean_value_grad_hess
from test_jet_math_mean_recovery import evaluate_point, fresh_mean_bridge
from test_native_anchored_mean_chart import Jet


_DENSITIES = (
    (0, []), (1, [8.]), (2, [8., .2]), (3, [1.7]),
    (4, [.12, 1.8, .4]), (5, [.12, 1.8]), (6, [.12, 1.8]), (7, [.12, 2.]),
)
_CASES = [
    (backend, density, shape)
    for backend in ("source", "native_abi")
    for density, shape in _DENSITIES
    if backend == "native_abi" or density < 4
]


@pytest.mark.parametrize("backend,density,shape", _CASES)
@pytest.mark.parametrize("columns", [0, 1, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_large_constant_variance_uses_normalized_density(
    request, backend, density, shape, columns, fixed
):
    scale, variance = 1e154, 1e308
    scaled_data = np.array([-10., -5., 4., 12.])
    scaled_design = np.zeros((scaled_data.size, columns))
    if columns == 4:
        scaled_design[:] = np.array([
            [1., -2., .2, 3.], [1., 1., .1, -2.],
            [1., .5, -1., 1.], [1., -.2, 2., .3],
        ])
    theta = np.r_[np.zeros(columns), variance, shape]
    scaled_theta = theta.copy()
    scaled_theta[columns] = 1.
    value, gradient, hessian = mean_value_grad_hess(
        scaled_theta, scaled_data, scaled_design, 0, 0, density
    )
    units = np.ones(theta.size)
    units[columns] = variance
    expected = (
        value + scaled_data.size * np.log(scale),
        gradient / units,
        (hessian / units[:, None]) / units[None, :],
    )
    absolute_tolerances = (
        2e-11,
        3e-12 / units,
        (3e-12 / units[:, None]) / units[None, :],
    )
    data = scaled_data * scale
    design = np.ascontiguousarray(scaled_design * scale)
    if backend == "native_abi":
        from scivol import ARX
        from scivol._mean_fit import DENSITIES, MeanModel
        model = MeanModel(
            ARX(0, constant=False) + DENSITIES[density](), data,
            design if columns else None, fixed=fixed,
        )
        assert model.columns == columns
        evaluate = lambda mask: model.evaluate(theta, mask)
    else:
        bridge = request.getfixturevalue("fresh_mean_bridge")
        evaluate = lambda mask: evaluate_point(
            bridge, density, theta, data, design, fixed=fixed, request=mask
        )
    for mask in range(1, 8):
        actual = evaluate(mask)
        for bit, output, reference, absolute in zip(
            (1, 2, 4), actual, expected, absolute_tolerances
        ):
            if mask & bit:
                assert np.all(np.isfinite(output))
                tolerance = absolute + (0. if bit == 1 else 3e-10 * np.abs(reference))
                assert np.all(np.abs(output - reference) <= tolerance), (mask, bit, output, reference)


def _unary(point, value, first, second):
    width = len(point.gradient)
    return Jet(value, tuple(first * entry for entry in point.gradient), tuple(
        first * point.hessian[i * width + j]
        + second * point.gradient[i] * point.gradient[j]
        for i in range(width) for j in range(width)
    ))


def _log(point):
    return _unary(point, mp.log(point.value), 1 / point.value, -1 / point.value**2)


def _lgamma(point):
    return _unary(point, mp.loggamma(point.value), mp.digamma(point.value),
                  mp.polygamma(1, point.value))


def _ged_oracle(theta, data, digits):
    with mp.workdps(digits):
        mean, variance, shape = [Jet.variable(float(value), 3, i)
                                 for i, value in enumerate(theta)]
        inverse_shape = shape.reciprocal()
        log_scale = (_lgamma(inverse_shape) - _lgamma(inverse_shape * 3)) * mp.mpf('.5')
        constant = -_log(shape) + mp.log(2) + log_scale + _lgamma(inverse_shape)
        log_variance = _log(variance)
        objective = Jet.constant(0, 3)
        for value in data:
            residual = Jet.constant(float(value), 3) - mean
            log_power = shape * (_log(residual * mp.sign(residual.value))
                                 - log_variance * mp.mpf('.5') - log_scale)
            power = mp.exp(log_power.value)
            objective += constant + log_variance * mp.mpf('.5') + _unary(log_power, power, power, power)
        return (float(objective.value), np.asarray(objective.gradient, dtype=float),
                np.asarray(objective.hessian, dtype=float).reshape(3, 3))


@pytest.mark.parametrize('backend', ['source', 'native_abi'])
@pytest.mark.parametrize('fixed', [False, True])
def test_tiny_nonzero_ged_residual_does_not_become_a_kink(request, backend, fixed):
    data = np.array([1e-300, -2e-300])
    design = np.ones((data.size, 1))
    theta = np.array([0., 1e200, 1.7])
    expected = _ged_oracle(theta, data, 100)
    for actual, reference in zip(expected, _ged_oracle(theta, data, 160)):
        np.testing.assert_array_equal(actual, reference)
    assert np.isfinite(expected[2][0, 0]) and expected[2][0, 0] > 1e-90
    if backend == 'native_abi':
        from scivol import ARX, GED
        from scivol._mean_fit import MeanModel
        model = MeanModel(ARX(0, constant=False) + GED(), data, design, fixed=fixed)
        evaluate = lambda mask: model.evaluate(theta, mask)
    else:
        bridge = request.getfixturevalue('fresh_mean_bridge')
        evaluate = lambda mask: evaluate_point(bridge, 3, theta, data, design,
                                               fixed=fixed, request=mask)
    for mask in range(1, 8):
        for bit, actual, reference in zip((1, 2, 4), evaluate(mask), expected):
            if mask & bit:
                assert np.all(np.isfinite(actual))
                np.testing.assert_allclose(actual, reference, rtol=0. if bit == 1 else 3e-10,
                                           atol=2e-11 if bit == 1 else 8e-323)
