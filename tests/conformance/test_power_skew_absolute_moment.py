"""Focused frozen-owner gate, reusing Euclid's independent density-space oracles.

No JAX initialization at collection. Local owners are checked before a dispatcher
change; the linked regression is a separate post-review gate. No legacy imports.
"""

from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
import pytest

from fixtures.power_skew_moment_cases import (
    ABSOLUTE_POINTS, LOW_NU, MAPPED_SKEW, POWER_POINTS, linked_case,
)
from oracles.garch import scalar_jet
from oracles.power import absolute_moment, mp_power_moment, original_z_jet, power_moment


ROOT = Path(__file__).resolve().parents[2]
REFINEMENT_COUNTS = (512, 1024)
JET_RTOL, JET_ATOL = 2e-8, 2e-9


@pytest.fixture(scope="module")
def moment_binary(tmp_path_factory):
    supplied = os.environ.get("SCIVOL_POWER_SKEW_MOMENT_TEST")
    if supplied:
        return Path(supplied).resolve(strict=True)
    binary = tmp_path_factory.mktemp("power_skew_moment") / "test_power_skew_moments"
    subprocess.run([
        os.environ.get("CXX", "clang++"), "-std=c++20", "-O3", "-fno-fast-math",
        "-ffp-contract=fast", "-fno-math-errno", "-fno-exceptions", "-fno-rtti",
        "-Wframe-larger-than=16384", "-Werror=frame-larger-than", "-Werror=vla",
        "-Werror=alloca", "-I" + str(ROOT / "core" / "include"),
        "-I" + str(ROOT / "core" / "src"),
        "-I" + str(ROOT / "core" / "src" / "math" / "power" / "third_party" / "boost"),
        "-I" + str(ROOT / "core" / "src" / "math" / "power" / "third_party" / "tlfloat"),
        str(ROOT / "tests" / "native" / "test_power_skew_moments.cpp"),
        "-o", str(binary),
    ], check=True, capture_output=True, text=True, timeout=120)
    return binary


def native_orders(binary, mode, point):
    output = subprocess.check_output(
        [str(binary), mode, *(repr(float(value)) for value in point)],
        text=True, timeout=120)
    rows = np.array([[float(value) for value in line.split()] for line in output.splitlines()])
    width = len(point)
    assert rows.shape == (3, 3 + width + width * (width + 1) // 2)
    np.testing.assert_array_equal(rows[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(rows[:, 1], np.ones(3))
    assert np.isfinite(rows).all()
    return rows[:, 2:]


def packed_jet(jet):
    value, gradient, hessian = jet
    return np.r_[value, gradient, hessian[np.triu_indices(len(gradient))]]


def assert_refinement(coarse, fine):
    for actual, expected in zip(coarse, fine, strict=True):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=JET_RTOL, atol=JET_ATOL)


@lru_cache(maxsize=16)
def refined_jet(mode, point):
    def moment(parameters, count):
        if mode == "--absolute":
            return absolute_moment(parameters, "skewt", count=count, quadrature=True)
        return power_moment(
            parameters[0], parameters[1], parameters[2:], "skewt",
            count=count, quadrature=True)
    coarse, fine = (scalar_jet(lambda parameters: moment(parameters, count), point)
                    for count in REFINEMENT_COUNTS)
    assert_refinement(coarse, fine)
    return fine


def assert_mp_value(mode, point, value):
    gamma, delta, *shape = (0.0, 1.0, *point) if mode == "--absolute" else point
    references = []
    for digits in (50, 70):
        with mp.workdps(digits):
            references.append(mp_power_moment(gamma, delta, shape, "skewt", quadrature=True))
    with mp.workdps(70):
        assert all(mp.isfinite(reference) for reference in references)
        assert abs(references[0] - references[1]) < mp.mpf("1e-30")
    assert math.isfinite(value) and abs(value - float(references[1])) < JET_ATOL


def assert_native_orders(actual, expected, width):
    reference = packed_jet(expected)
    for order in range(3):
        active = (1, 1 + width, len(reference))[order]
        np.testing.assert_allclose(actual[order, :active], reference[:active],
                                   rtol=JET_RTOL, atol=JET_ATOL)
        np.testing.assert_array_equal(actual[order, active:], np.zeros(len(reference) - active))
    np.testing.assert_allclose(actual[:, 0], actual[2, 0], rtol=0, atol=3e-14)
    np.testing.assert_allclose(actual[1, :1 + width], actual[2, :1 + width],
                               rtol=2e-12, atol=3e-14)


@pytest.mark.parametrize("point", [point for _, point in ABSOLUTE_POINTS],
                         ids=[name for name, _ in ABSOLUTE_POINTS])
def test_finite_absolute_shape_jet_against_existing_integral_ad(moment_binary, point):
    expected = refined_jet("--absolute", point)
    assert_mp_value("--absolute", point, expected[0])
    assert_native_orders(native_orders(moment_binary, "--absolute", point), expected, 2)


@pytest.mark.parametrize("nu", (LOW_NU, 8.0))
def test_zero_skew_even_curvature_and_student_reduction(moment_binary, nu):
    expected = refined_jet("--absolute", (nu, 0.0))
    student = scalar_jet(lambda shape: absolute_moment(shape, "studentt"), (nu,))
    half = student[0] / 2
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2))
    c /= math.sqrt(math.pi * (nu - 2))
    curvature = -10 * half + 32 * half ** 3 + 32 * c * half ** 2
    np.testing.assert_allclose([expected[0], expected[1][0], expected[2][0, 0]],
                               [student[0], student[1][0], student[2][0, 0]],
                               rtol=JET_RTOL, atol=JET_ATOL)
    np.testing.assert_allclose([expected[1][1], expected[2][0, 1], expected[2][1, 1]],
                               [0, 0, curvature], rtol=JET_RTOL, atol=JET_ATOL)
    assert abs(curvature) > .01
    actual = native_orders(moment_binary, "--absolute", (nu, 0.0))
    assert_native_orders(actual, expected, 2)
    for skew in (-1e-6, 1e-6):
        near = native_orders(moment_binary, "--absolute", (nu, skew))[2]
        assert abs(near[2] / skew - curvature) < 1e-8
        assert abs(near[-1] - curvature) < 1e-8


@pytest.mark.parametrize("point", [point for _, point in POWER_POINTS],
                         ids=[name for name, _ in POWER_POINTS])
def test_aparch_arbitrary_delta_retains_existing_integral_ad(moment_binary, point):
    expected = refined_jet("--power", point)
    assert_mp_value("--power", point, expected[0])
    assert_native_orders(native_orders(moment_binary, "--power", point), expected, 4)


@pytest.mark.parametrize("skew", (MAPPED_SKEW, -MAPPED_SKEW, 0.0, 1e-6, -1e-6))
def test_absolute_equals_aparch_delta_one_common_shape_jet(moment_binary, skew):
    absolute = native_orders(moment_binary, "--absolute", (LOW_NU, skew))[2]
    power = native_orders(moment_binary, "--power", (0.0, 1.0, LOW_NU, skew))[2]
    # APARCH axes are gamma,delta,nu,lambda; compare the common shape block only.
    projection = power[[0, 3, 4, 12, 13, 14]]
    np.testing.assert_allclose(absolute, projection, rtol=JET_RTOL, atol=JET_ATOL)


@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_linked_low_nu_all_masks_after_dispatch_review(native, p, q):
    declared = linked_case(p, q)
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full["reasons"]
    assert full["dispatch"] == int((p, q) == (1, 1))
    np.testing.assert_array_equal(full["scored_rows"], declared["scored_rows"])
    np.testing.assert_array_equal(full["theta"][-2:], [LOW_NU, MAPPED_SKEW])
    coarse, expected = (original_z_jet(declared, full["theta"], moment_count=count)
                        for count in REFINEMENT_COUNTS)
    assert_refinement(coarse, expected)
    for mask in range(1, 8):
        actual = full if mask == 7 else native(declared, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask, actual["reasons"]
        np.testing.assert_array_equal(actual["theta"], full["theta"])
        for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected, strict=True):
            if mask & bit:
                assert np.isfinite(actual[name]).all()
                np.testing.assert_allclose(actual[name], reference, rtol=JET_RTOL, atol=JET_ATOL)
                np.testing.assert_allclose(actual[name], full[name], rtol=2e-12, atol=3e-14)
            else:
                np.testing.assert_array_equal(np.asarray(actual[name]).view(np.uint64),
                    np.full_like(np.asarray(actual[name]), actual["sentinel"]).view(np.uint64))
