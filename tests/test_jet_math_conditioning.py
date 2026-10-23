"""Conditioning is part of the likelihood, not an optimizer regularizer."""

import math

import numpy as np
import pytest

from scivol import GARCH, Normal
from scivol._native_fit import (
    _describe_model,
    _evaluate_theta_request,
    _initial_variance,
)


def test_native_conditioning_preserves_positive_subnormal_seed():
    data = np.full(8, 1e-160)
    expected = float(np.mean(data * data))
    assert 0.0 < expected < np.finfo(float).tiny
    assert _initial_variance(data) == expected


@pytest.mark.parametrize("data", [np.zeros(8), np.full(8, 1e-200), np.full(8, 1e200)])
def test_native_conditioning_rejects_unrepresentable_seed(data):
    with pytest.raises(ValueError, match="conditioning variance must be positive and representable"):
        _initial_variance(data)


@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
def test_native_value_uses_unpadded_subnormal_conditioning(p, q):
    data = np.full(8, 1e-160)
    theta = np.r_[1e-307, 0.1, np.zeros(p - 1), 0.8, np.zeros(q - 1)]
    model = _describe_model(GARCH(p, q) + Normal())
    variances = [float(np.mean(data * data))]
    for time in range(1, data.size):
        variance = theta[0]
        for lag in range(min(p, time)):
            variance += theta[1 + lag] * data[time - lag - 1] ** 2
        for lag in range(min(q, time)):
            variance += theta[1 + p + lag] * variances[time - lag - 1]
        variances.append(variance)
    expected = math.fsum(
        0.5 * (math.log(2.0 * math.pi) + math.log(h) + float(e * e) / h)
        for e, h in zip(data, variances)
    )
    actual, _, _ = _evaluate_theta_request(model, data, theta, request=1)
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
def test_fixed_conditioning_does_not_invalidate_finite_composed_derivatives(p, q):
    mp = pytest.importorskip("mpmath")
    data = np.full(8, 1e-160)
    theta = np.r_[0.1, 0.1, np.zeros(p - 1), 0.8, np.zeros(q - 1)]
    model = _describe_model(GARCH(p, q) + Normal())
    with mp.workdps(400):
        squared = tuple(mp.mpf(float(e2)) for e2 in data * data)
        seed = mp.mpf(_initial_variance(data))

        def objective(*point):
            path = [seed]
            for time in range(1, data.size):
                path.append(
                    point[0]
                    + mp.fsum(point[1 + lag] * squared[time - lag - 1]
                              for lag in range(min(p, time)))
                    + mp.fsum(point[1 + p + lag] * path[time - lag - 1]
                              for lag in range(min(q, time)))
                )
            return mp.fsum((mp.log(2 * mp.pi) + mp.log(h) + e2 / h) / 2
                           for e2, h in zip(squared, path))

        point = tuple(mp.mpf(float(value)) for value in theta)
        unit = np.eye(theta.size, dtype=int)
        expected = (
            float(objective(*point)),
            np.array([float(mp.diff(objective, point, tuple(row))) for row in unit]),
            np.array([[float(mp.diff(objective, point, tuple(row + column)))
                       for column in unit] for row in unit]),
        )
    for request in range(1, 8):
        actual = _evaluate_theta_request(model, data, theta, request=request)
        for bit, observed, reference in zip((1, 2, 4), actual, expected):
            if request & bit:
                assert np.all(np.isfinite(observed))
                np.testing.assert_allclose(observed, reference, rtol=2e-13, atol=1e-12)
