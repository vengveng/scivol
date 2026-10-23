"""Exact mathematical cases restored by the chief's record-level phase review."""

from functools import lru_cache
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from oracles.garch import scalar_jet
from oracles.power import power_moment
from oracles.power_mp_ad import density_jet, gh_moment_jet
from numerical_006.oracle_mp import Jet
from numerical_006.test_vg_retained_value_accuracy import log_bessel_jet


FIXTURE = Path(__file__).parent / "fixtures/power_phase_split_math.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def assert_mp_entries_close(actual_entries, expected_entries, tolerance):
    for a, b in zip(actual_entries, expected_entries, strict=True):
        assert mp.isfinite(a) and mp.isfinite(b)
        assert abs(a - b) < tolerance


def flattened(jet):
    return (jet.value, *jet.gradient, *jet.hessian)


@lru_cache(maxsize=8)
def refined_skew(point):
    references = [scalar_jet(lambda x: power_moment(x[0], x[1], x[2:], "skewt",
                   count=count, quadrature=True), point) for count in (256, 512)]
    for coarse, fine, relative, absolute in zip(*references,
            (2e-10, 2e-8, 2e-8), (2e-11, 2e-9, 2e-9), strict=True):
        assert np.isfinite(coarse).all() and np.isfinite(fine).all()
        np.testing.assert_allclose(coarse, fine, rtol=relative, atol=absolute)
    return references[-1]


@pytest.mark.parametrize("case_index", range(4))
def test_canonical_density_integral_refines_exact_retained_cases(case_index):
    point = tuple(float.fromhex(value) for value in fixture()["skew_points_hex"][case_index])
    value, gradient, hessian = refined_skew(point)
    assert abs(point[0]) < 1.0 and abs(point[3]) < 1.0
    assert 0.0 < point[1] < point[2] and point[2] > 2.0
    assert value > 0.0
    np.testing.assert_allclose(hessian, hessian.T, rtol=2e-8, atol=2e-9)


def test_canonical_density_reflection_preserves_every_cross_block_exact_case():
    points = fixture()["skew_points_hex"]
    original, reflected = (refined_skew(tuple(float.fromhex(value) for value in points[index]))
                           for index in (0, 1))
    signs = np.array([-1.0, 1.0, 1.0, -1.0])
    expected = (original[0], signs * original[1], signs[:, None] * original[2] * signs[None, :])
    for actual, target, relative, absolute in zip(reflected, expected,
            (2e-10, 2e-8, 2e-8), (2e-11, 2e-9, 2e-9), strict=True):
        assert np.isfinite(actual).all() and np.isfinite(target).all()
        np.testing.assert_allclose(actual, target, rtol=relative, atol=absolute)


def test_refined_normalized_moment_matches_all_31_independent_targets():
    declared = fixture()
    point = tuple(float.fromhex(value) for value in declared["gh_stress_hex"])
    low = gh_moment_jet(point, digits=60, outer=32, bessel_count=32)
    high = gh_moment_jet(point, digits=80, outer=40, bessel_count=40)
    with mp.workdps(80):
        targets = tuple(mp.mpf(value) for value in declared["gh_moment_target_31"])
        assert len(targets) == len(flattened(low)) == len(flattened(high)) == 31
        for actual in (low, high):
            assert_mp_entries_close(flattened(actual), targets, mp.mpf("1e-20"))
        assert_mp_entries_close(flattened(low), flattened(high), mp.mpf("1e-20"))


@pytest.mark.parametrize("case_index", range(2))
def test_stress_bessel_value_argument_score_and_curvature(case_index):
    declared = fixture()["bessel_pairs"][case_index]
    with mp.workdps(80):
        point = tuple(mp.mpf(value) for value in declared)
        order, argument = (Jet.variable(value, 2, index) for index, value in enumerate(point))
        result = log_bessel_jet(order, argument, 40, 80)
        exact = mp.besselk(*point)
        slope = -(mp.besselk(point[0] - 1, point[1]) + mp.besselk(point[0] + 1, point[1])) / (2 * exact)
        curvature = 1 + point[0] ** 2 / point[1] ** 2 - slope / point[1] - slope ** 2
        assert abs(result.value - mp.log(exact)) < mp.mpf("1e-35")
        assert abs(result.gradient[1] - slope) < mp.mpf("1e-35")
        assert abs(result.hessian[3] - curvature) < mp.mpf("1e-35")


def test_stress_density_exact_nig_reduction_all_local_derivatives():
    point = tuple(float.fromhex(value) for value in fixture()["nig_point_hex"])
    general = density_jet(point, digits=80, bessel_count=40)
    reduced = density_jet(point, reduced=True, digits=80, bessel_count=40)
    with mp.workdps(80):
        assert_mp_entries_close(flattened(general), flattened(reduced), mp.mpf("1e-35"))
