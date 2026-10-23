"""Retained local APARCH moment points using the current integral AD owner."""

import math

import numpy as np
import pytest

from oracles.garch import scalar_jet
from oracles.power import mp_power_moment, power_moment
from test_power_transfer_cases import DENSITIES
from test_power_transfer_moment_cache import moment_backend


CODES = {name: i for i, (name, _) in enumerate(DENSITIES)}


def reference(point, density, count):
    return scalar_jet(lambda x: power_moment(x[0], x[1], x[2:], density,
                      count=count, bessel_count=512, quadrature=True), point)


def native_moment(backend, point, density, mask):
    point = np.ascontiguousarray(point, dtype=np.float64)
    outputs = [np.full(1, 12345.0), np.full(len(point), 12345.0),
               np.full((len(point), len(point)), 12345.0)]
    valid = backend.probe_power_moment(point.ctypes.data, CODES[density], mask,
        *(output.ctypes.data for output in outputs))
    for bit, output in zip((1, 2, 4), outputs, strict=True):
        if not mask & bit:
            np.testing.assert_array_equal(output, 12345.0)
    return valid, (outputs[0][0], *outputs[1:])


def check(backend, point, density, tolerances, *, counts=(256, 512)):
    coarse, expected = (reference(point, density, n) for n in counts)
    for index, (left, right) in enumerate(zip(coarse, expected, strict=True)):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=tolerances[index][0], atol=tolerances[index][1])
    for mask in range(1, 8):
        valid, actual = native_moment(backend, point, density, mask)
        assert valid
        for bit, left, right, (relative, absolute) in zip((1, 2, 4), actual, expected, tolerances, strict=True):
            if mask & bit:
                assert np.isfinite(left).all()
                np.testing.assert_allclose(left, right, rtol=relative, atol=absolute)
    return expected


@pytest.mark.parametrize("density,shape", DENSITIES)
@pytest.mark.parametrize("delta", (0.7, 1.7))
def test_retained_stationarity_moment_all_densities(moment_backend, density, shape, delta):
    check(moment_backend, [0.15, delta, *shape], density,
          ((2e-9, 2e-11), (2e-8, 2e-10), (2e-7, 1e-8)))


@pytest.mark.parametrize("point", (
    (0.06183922416878056, 2.0379819829328323, 0.12034602234581399, 15.272133090357961),
    (0.140235972517651, 2.236495579602217, 0.2857400734797855, 49.999999999999986),
))
def test_retained_vg_shape_points(moment_backend, point):
    check(moment_backend, point, "variance_gamma", ((2e-7, 1e-8),) * 3)


@pytest.mark.parametrize("gamma", (0.0, 0.25))
@pytest.mark.parametrize("delta", (0.7, 1.7, 2.0, 3.5))
def test_retained_zero_skew_student_common_parameters(moment_backend, gamma, delta):
    point = [gamma, delta, 8.0, 0.0]
    skew = check(moment_backend, point, "skewt", ((2e-9, 2e-11), (2e-8, 2e-10), (2e-7, 1e-8)))
    student = reference(point[:3], "studentt", 512)
    for actual, expected, relative, absolute in zip(
            (skew[0], skew[1][:3], skew[2][:3, :3]), student,
            (2e-9, 2e-8, 2e-7), (2e-11, 2e-10, 1e-8), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=relative, atol=absolute)


@pytest.mark.parametrize("skew", (-0.2, 0.2))
@pytest.mark.parametrize("power", (0.2, 0.7))
def test_retained_sublinear_origin_integral_refinement(skew, power):
    point = [-0.2, power, 8.0, skew]
    coarse, fine = (reference(point, "skewt", count) for count in (256, 512))
    for actual, expected in zip(coarse, fine, strict=True):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("rho", (-0.12, 0.0, 0.12))
def test_retained_vg_split_integral_refinement(moment_backend, rho):
    check(moment_backend, [0.1, 1.4, rho, 2.0], "variance_gamma", ((2e-6, 2e-8),) * 3,
          counts=(64, 128))


@pytest.mark.parametrize("point", (
    (math.tanh(0.15), 2.0, 3.0, math.tanh(0.4)),
    (-math.tanh(0.15), 2.0, 3.0, -math.tanh(0.4)),
    (math.tanh(0.15), 1.95, 3.0, math.tanh(0.4)),
    (math.tanh(0.15), 2.05, 3.0, math.tanh(0.4)),
))
def test_retained_skew_recovery_all_crossblocks(moment_backend, point):
    expected = check(moment_backend, point, "skewt", ((2e-10, 2e-11), (2e-8, 2e-9), (2e-8, 2e-9)))
    reflected = reference([-point[0], point[1], point[2], -point[3]], "skewt", 512)
    signs = np.array([-1.0, 1.0, 1.0, -1.0])
    for actual, target in zip(reflected, (expected[0], signs * expected[1],
            signs[:, None] * expected[2] * signs[None, :]), strict=True):
        np.testing.assert_allclose(actual, target, rtol=2e-8, atol=2e-9)
