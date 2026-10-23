"""Resolved APARCH persistence value/slack, separate from optimizer success."""

import math

import mpmath as mp
import numpy as np
import pytest

from oracles.power import mp_power_moment
from test_power_transfer_cases import DENSITIES
from test_power_transfer_moment_cache import moment_backend


def persistence(backend, theta, density, p=1, q=1):
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    saved = theta.copy()
    output = np.full(3, 12345.0)
    valid = backend.probe_persistence(theta.ctypes.data, p, q, density, output.ctypes.data)
    np.testing.assert_array_equal(theta, saved)
    return valid, output


@pytest.mark.parametrize("density,shape", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_retained_ordinary_persistence_has_independent_moments(moment_backend, density, shape, p, q):
    theta = np.r_[0.08, np.full(p, 0.06 / p), np.linspace(-0.15, 0.2, p),
                  np.full(q, 0.4 / q), 1.7, shape]
    references = []
    for digits in (50, 70):
        with mp.workdps(digits):
            moments = [mp_power_moment(mp.mpf(float(gamma)), mp.mpf(1.7),
                       tuple(mp.mpf(float(v)) for v in shape), density)
                       for gamma in theta[1 + p:1 + 2 * p]]
            references.append(mp.fsum([*(mp.mpf(float(a)) * m for a, m in zip(theta[1:1 + p], moments)),
                                       *(mp.mpf(float(b)) for b in theta[1 + 2 * p:1 + 2 * p + q])]))
    with mp.workdps(70):
        assert abs(references[0] - references[1]) < mp.mpf("1e-30")
    code = [name for name, _ in DENSITIES].index(density)
    valid, output = persistence(moment_backend, theta, code, p, q)
    assert valid and np.isfinite(output).all()
    np.testing.assert_allclose(output[:2], [float(references[1]), float(1 - references[1])], rtol=2e-12, atol=0)


@pytest.mark.parametrize("density,shape", DENSITIES)
def test_retained_tiny_beta_is_not_reconstructed_from_slack(moment_backend, density, shape):
    theta = [0.1, 0.0, 0.0, 2.0 ** -80, 2.0, *shape]
    code = [name for name, _ in DENSITIES].index(density)
    valid, output = persistence(moment_backend, theta, code)
    assert valid
    assert output[0] == 2.0 ** -80 and output[1] == 1.0
    assert output[0] != 1.0 - output[1]


def test_retained_near_one_positive_slack_survives_value_rounding(moment_backend):
    theta = [0.1, 2.0 ** -54, 0.0, math.nextafter(1.0, 0.0), 2.0]
    valid, output = persistence(moment_backend, theta, 0)
    assert valid
    assert output[0] == 1.0 and output[1] == 2.0 ** -54
    assert 1.0 - output[0] == 0.0 and output[1] > 0.0


@pytest.mark.parametrize("density,theta", (
    (0, [0.1, 0.0, 1.0, 0.4, 2.0]), (0, [0.1, 0.0, 0.0, 0.4, 0.0]),
    (0, [0.1, 0.0, 0.0, 0.4, np.nan]), (1, [0.1, 0.0, 0.0, 0.4, 8.0, 8.0]),
))
def test_retained_invalid_moment_is_not_hidden_by_zero_alpha(moment_backend, density, theta):
    valid, output = persistence(moment_backend, theta, density)
    assert not valid
    np.testing.assert_array_equal(output, 12345.0)


@pytest.mark.parametrize("p,q", ((0, 1), (1, 0), (0, 0),
                                  (int(np.iinfo(np.uintp).max), 1),
                                  (1, int(np.iinfo(np.uintp).max))))
def test_retained_invalid_orders_reject_before_reading_theta(moment_backend, p, q):
    theta = [0.1, 2.0 ** -54, 0.0, math.nextafter(1.0, 0.0), 2.0]
    valid, output = persistence(moment_backend, theta, 0, p, q)
    assert not valid
    np.testing.assert_array_equal(output, 12345.0)


def test_retained_null_persistence_pointer_is_rejected(moment_backend):
    output = np.full(3, 12345.0)
    assert not moment_backend.probe_persistence(None, 1, 1, 0, output.ctypes.data)
    np.testing.assert_array_equal(output, 12345.0)
