"""Scoped mandatory EGARCH/APARCH conformance, independent of the old package.

Collection initializes neither JAX nor a native build. Required dependencies and
the chief-owned bridge fail when absent, never skip. These tests do not qualify
non-dispatched fixed templates, binary metadata, optimizers or all of Phase 1.
"""

from __future__ import annotations

from copy import deepcopy
import math

import mpmath as mp
import numpy as np
import pytest

from fixtures.garch_cases import case as garch_case
from fixtures.power_cases import (
    DENSITIES, FAMILIES, ORDERS, REQUESTS, SHAPES, case, chart_case, domain_cases,
    long_horizon_case, ordinary_cases, seed_delta_case, small_power_case, tiny_scale_case,
    with_parameters,
)
from oracles.garch import full_jet as garch_jet, jax_modules, local_jet, scalar_jet
from oracles.power import (
    absolute_moment, coordinate_map, full_jet, log_variance_path, mp_coordinate_map,
    mp_log_variances, mp_power_moment, mp_value, original_z_jet, power_moment,
    pullback, state_jet, theta_objective, volatility_width,
)
from oracles.values import mp_value as garch_mp_value


LIGHT = ("normal", "studentt", "skewt", "ged")
ASYMMETRIC = ("skewt", "gh", "nig", "hyperbolic", "variance_gamma")
MATHEMATICAL_DOMAIN, NUMERICAL_RANGE = 4, 5


def assert_jet_close(actual, expected, mask, density, *, value_atol=None):
    light = density in LIGHT
    tolerances = (2e-9, 3e-8, 3e-7) if light else (3e-8, 3e-7, 4e-6)
    if value_atol is not None:
        tolerances = (value_atol,) + tolerances[1:]
    for index, (bit, name, reference) in enumerate(zip((1, 2, 4), ("f", "g", "h"), expected)):
        if mask & bit:
            assert np.isfinite(actual[name]).all() and np.isfinite(reference).all(), name
            np.testing.assert_allclose(actual[name], reference, rtol=0.0 if bit == 1 else tolerances[index],
                                       atol=tolerances[index], err_msg=f"{density}: {name}, mask={mask}")


def assert_success(actual, declared, mask):
    assert actual["status"] == 0, (declared["case_id"], actual["reasons"])
    assert actual["valid_mask"] == mask
    np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
    width = len(declared["theta_names"])
    assert actual["g"].shape == (width,)
    assert actual["h"].shape == (width, width)
    for index, (bit, name) in enumerate(zip((1, 2, 4), ("f", "g", "h"))):
        if mask & bit:
            assert actual["reasons"][index] == 0
        else:
            untouched = np.full_like(np.asarray(actual[name]), actual["sentinel"])
            np.testing.assert_array_equal(np.asarray(actual[name]).view(np.uint64), untouched.view(np.uint64))


def exercise_requests(native, declared, expected, *, value_atol=None):
    full = native(declared, 7)
    assert_success(full, declared, 7)
    for mask in REQUESTS:
        actual = full if mask == 7 else native(declared, mask)
        assert_success(actual, declared, mask)
        assert_jet_close(actual, expected, mask, declared["density"], value_atol=value_atol)
        assert_jet_close(actual, tuple(full[name] for name in ("f", "g", "h")),
                         mask, declared["density"], value_atol=2e-10)
        np.testing.assert_array_equal(actual["theta"], full["theta"])
        if not declared["transformed"]:
            np.testing.assert_array_equal(actual["theta"], declared["parameters"])
    return full


def assert_mp_agreement(declared, actual, *, theta=None, atol=3e-9):
    coarse, fine = mp_value(declared, theta, digits=50), mp_value(declared, theta, digits=70)
    with mp.workdps(70):
        assert mp.isfinite(coarse) and mp.isfinite(fine)
        assert abs(coarse - fine) <= mp.mpf("1e-30")
    assert math.isfinite(actual) and abs(actual - float(fine)) <= atol


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_theta_all_seven_masks_match_independent_ad(native, declared):
    exercise_requests(native, declared, full_jet(declared))


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_dispatch_is_fixed_11_and_runtime_general_pq(native, declared):
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert actual["dispatch"] == int((declared["p"], declared["q"]) == (1, 1))


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_complete_additive_values_match_mp_with_refinement(native, model, density, p, q):
    declared = case(model, density, p, q, first=2, stop=7)
    expected = full_jet(declared)[0]
    assert_mp_agreement(declared, expected)
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert_mp_agreement(declared, actual["f"])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_first_row_is_fixed_variance_and_shape_only(native, model, density, p, q):
    declared = case(model, density, p, q, y=[0.37], h0=1.3)
    width = volatility_width(declared)
    local = local_jet(density, np.r_[0.37, 1.3, SHAPES[density]])
    expected = full_jet(declared)
    np.testing.assert_array_equal(expected[1][:width], np.zeros(width))
    np.testing.assert_array_equal(expected[2][:width], np.zeros_like(expected[2][:width]))
    assert abs(expected[0] - local[0]) < 2e-10
    np.testing.assert_allclose(expected[1][width:], local[1][2:], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(expected[2][width:, width:], local[2][2:, 2:], rtol=2e-11, atol=2e-11)
    actual = exercise_requests(native, declared, expected)
    np.testing.assert_array_equal(actual["g"][:width], np.zeros(width))
    np.testing.assert_array_equal(actual["h"][:width], np.zeros_like(actual["h"][:width]))
    assert_mp_agreement(declared, actual["f"])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_oracle_state_has_exact_seed_and_only_available_lags(model, p, q):
    declared = case(model, "studentt", p, q, h0=2.7)
    state, gradient, hessian = state_jet(declared)
    with mp.workdps(70):
        expected = np.asarray([float(value) for value in mp_log_variances(declared, declared["parameters"])])
    np.testing.assert_allclose(state, expected, rtol=3e-14, atol=3e-14)
    assert state[0] == np.log(declared["h0"])
    np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))
    np.testing.assert_array_equal(hessian[0], np.zeros_like(hessian[0]))
    inactive = list(range(2, 1 + p)) + list(range(2 + p, 1 + 2 * p))
    inactive += list(range(2 + 2 * p, 1 + 2 * p + q))
    np.testing.assert_array_equal(gradient[1, inactive], np.zeros(len(inactive)))
    np.testing.assert_array_equal(hessian[1, inactive], np.zeros_like(hessian[1, inactive]))
    if model == "aparch":
        delta_index = volatility_width(declared) - 1
        power, power_g, power_h = state_jet(declared, return_power=True)
        seed = declared["h0"] ** (declared["parameters"][delta_index] / 2.0)
        slope = np.log(declared["h0"]) / 2.0
        assert power[0] == pytest.approx(seed, rel=3e-15)
        assert power_g[0, delta_index] == pytest.approx(seed * slope, rel=3e-15)
        assert power_h[0, delta_index, delta_index] == pytest.approx(seed * slope * slope, rel=4e-15)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("first,stop,hold_back", ((3, 7, 0), (5, 8, 2), (2, 3, 2)))
def test_scored_suffix_and_physical_holdback_are_independent(native, model, density, first, stop, hold_back):
    declared = case(model, density, 3, 2, first=first, stop=stop, hold_back=hold_back)
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert_mp_agreement(declared, expected[0])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_unscored_prefix_advances_recursion_and_stop_excludes_future(native, model, density):
    declared = case(model, density, 2, 3, first=3, stop=6)
    expected = full_jet(declared)
    actual = exercise_requests(native, declared, expected)
    altered = deepcopy(declared)
    altered["y"][:3] *= -1.7
    changed = native(altered, 7)
    assert_success(changed, altered, 7)
    assert_jet_close(changed, full_jet(altered), 7, density)
    assert abs(changed["f"] - actual["f"]) > 1e-5
    future = deepcopy(declared)
    future["y"][6:] = [1e200, -1e200]
    untouched = native(future, 7)
    assert_success(untouched, future, 7)
    for name in ("f", "g", "h"):
        np.testing.assert_array_equal(untouched[name], actual[name])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_signed_data_are_not_squared_at_the_transport_boundary(native, model, density):
    declared = case(model, density, 4, 3)
    flipped = deepcopy(declared)
    flipped["y"] *= -1.0
    actual, opposite = native(declared, 7), native(flipped, 7)
    for item, result in ((declared, actual), (flipped, opposite)):
        assert_success(result, item, 7)
        assert_jet_close(result, full_jet(item), 7, density)
    assert abs(actual["f"] - opposite["f"]) > 1e-5


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("delta", (0.7, 1.7))
def test_power_moment_ad_has_independent_quadrature_and_mp_refinement(density, delta):
    point = np.r_[-0.16, delta, SHAPES[density]]
    jets = [scalar_jet(lambda parameters: power_moment(
        parameters[0], parameters[1], parameters[2:], density, count=count, quadrature=True), point)
            for count in (128, 256)]
    for actual, expected in zip(jets[0], jets[1]):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
    analytic = scalar_jet(lambda parameters: power_moment(
        parameters[0], parameters[1], parameters[2:], density, count=256), point)
    for actual, expected in zip(analytic, jets[1]):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
    references = []
    for digits in (50, 70):
        with mp.workdps(digits):
            gamma, power, *shape = tuple(mp.mpf(float(value)) for value in point)
            references.append(mp_power_moment(gamma, power, shape, density, quadrature=True))
    with mp.workdps(70):
        assert abs(references[0] - references[1]) < mp.mpf("1e-30")
    assert abs(jets[1][0] - float(references[1])) < 2e-9


@pytest.mark.parametrize("density", DENSITIES)
def test_egarch_absolute_moment_shape_jet_refines_independently(density):
    point = np.asarray(SHAPES[density], dtype=np.float64)
    jets = [scalar_jet(lambda shape: absolute_moment(shape, density, count=count, quadrature=True), point)
            for count in (128, 256)]
    for coarse, fine in zip(*jets):
        assert np.isfinite(coarse).all() and np.isfinite(fine).all()
        np.testing.assert_allclose(coarse, fine, rtol=2e-8, atol=2e-9)
    if point.size:
        assert np.linalg.norm(jets[1][1]) > 1e-5
    with mp.workdps(70):
        reference = mp_power_moment(0, 1, tuple(mp.mpf(float(v)) for v in point), density, quadrature=True)
    assert abs(jets[1][0] - float(reference)) < 2e-9


@pytest.mark.parametrize("density", ("skewt", "variance_gamma"))
def test_zero_skew_moment_has_no_collapsed_interval_singularity(density):
    shape = list(SHAPES[density])
    shape[-1 if density == "skewt" else 0] = 0.0
    point = np.r_[0.0, 0.7, shape]
    jets = [scalar_jet(lambda parameters: power_moment(
        parameters[0], parameters[1], parameters[2:], density, count=count, quadrature=True), point)
            for count in (128, 256)]
    for coarse, fine in zip(*jets):
        assert np.isfinite(coarse).all() and np.isfinite(fine).all()
        np.testing.assert_allclose(coarse, fine, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ORDERS)
def test_original_z_complete_objective_curvature_and_value_map(native, model, density, p, q):
    declared = chart_case(model, density, p, q)
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    _, jnp, _ = jax_modules()
    independent = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], independent, rtol=3e-9, atol=3e-10)
    expected = original_z_jet(declared, actual["theta"])
    exercise_requests(native, declared, expected)
    assert_mp_agreement(declared, actual["f"], theta=actual["theta"])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_chart_value_has_independent_mp_moments(native, model, density):
    declared = chart_case(model, density)
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    reference = np.asarray([float(v) for v in mp_coordinate_map(declared)])
    np.testing.assert_allclose(actual["theta"], reference, rtol=3e-9, atol=3e-10)
    if model == "aparch":
        with mp.workdps(70):
            theta = tuple(mp.mpf(float(v)) for v in actual["theta"])
            moment = mp_power_moment(theta[2], theta[4], theta[5:], density)
            persistence = theta[1] * moment + theta[3]
            z = tuple(mp.mpf(float(v)) for v in declared["original_z"])
            exact = (mp.exp(z[1]) + mp.exp(z[3])) / (1 + mp.exp(z[1]) + mp.exp(z[3]))
        assert abs(float(persistence - exact)) < 3e-9


@pytest.mark.parametrize("coordinate", (-700.0, -36.0, 50.0))
def test_aparch_unshifted_softplus_keeps_original_z_tail_derivatives(native, coordinate):
    declared = chart_case("aparch", "normal")
    declared["parameters"][0] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    exact_map = np.asarray([float(value) for value in mp_coordinate_map(declared)])
    np.testing.assert_allclose(actual["theta"][0], exact_map[0], rtol=3e-15, atol=0.0)
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    exercise_requests(native, declared, expected)
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        slope = float(1 / (1 + mp.exp(-z)))
    assert slope > 0.0
    np.testing.assert_allclose(actual["g"][0] / slope, expected[1][0] / slope, rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(actual["h"][0] / slope, expected[2][0] / slope, rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize("density", ("normal", "skewt", "nig", "variance_gamma"))
def test_aparch_map_curvature_includes_moments_gamma_delta_and_shape(native, density):
    declared = chart_case("aparch", density, 3, 2)
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    composed = original_z_jet(declared, actual["theta"])
    assert_jet_close(dict(zip(("f", "g", "h"), composed)), expected, 7, density)
    assert_jet_close(actual, composed, 7, density)
    jax, jnp, _ = jax_modules()
    jacobian = np.asarray(jax.jacfwd(coordinate_map(declared))(jnp.asarray(declared["original_z"])))
    incomplete = jacobian.T @ theta_jet[2] @ jacobian
    assert np.max(np.abs(expected[2] - incomplete)) > 1e-4
    gamma_index, delta_index = 1 + declared["p"], volatility_width(declared) - 1
    map_h = np.asarray(jax.jacfwd(jax.jacfwd(coordinate_map(declared)))(jnp.asarray(declared["original_z"])))
    assert abs(map_h[1, gamma_index, delta_index]) > 1e-5
    if SHAPES[density]:
        assert np.max(np.abs(map_h[1, delta_index, volatility_width(declared):])) > 1e-5


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("axis,coordinate", (("rho", -18.0), ("rho", 18.0),
                                            ("zeta", -50.0), ("zeta", 50.0)))
def test_original_z_endpoint_rounding_keeps_original_coordinate_curvature(native, model, axis, coordinate):
    declared = chart_case(model, "nig")
    index = volatility_width(declared) + int(axis == "zeta")
    declared["parameters"][index] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta_jet = full_jet(declared, theta=actual["theta"], moment_count=256)
    expected = pullback(declared, theta_jet, moment_count=256)
    exercise_requests(native, declared, expected)
    # Relative row checks expose tiny plateau slopes that absolute tolerances hide.
    scale_g = np.max(np.abs(expected[1][index:index + 1]))
    scale_h = np.max(np.abs(expected[2][index]))
    assert scale_g > 0.0 and scale_h > 0.0
    np.testing.assert_allclose(actual["g"][index] / scale_g, expected[1][index] / scale_g, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(actual["h"][index] / scale_h, expected[2][index] / scale_h, rtol=8e-6, atol=8e-6)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ORDERS)
def test_aparch_delta_two_gamma_zero_reduces_to_garch_same_conditioning(native, density, p, q):
    garch = garch_case(density=density, p=p, q=q, first=4, stop=8, hold_back=2, h0=2.7)
    theta = garch["parameters"]
    aparch = case("aparch", density, p, q, parameters=(list(theta[:1 + p]) + [0.0] * p
                  + list(theta[1 + p:1 + p + q]) + [2.0] + list(SHAPES[density])),
                  first=4, stop=8, hold_back=2, h0=2.7)
    common = np.r_[np.arange(1 + p), np.arange(1 + 2 * p, 1 + 2 * p + q),
                   np.arange(volatility_width(aparch), len(aparch["parameters"]))]
    reference = garch_jet(garch)
    exercise_requests(native, aparch, full_jet(aparch))
    with mp.workdps(70):
        assert abs(mp_value(aparch) - garch_mp_value(garch, digits=70)) < mp.mpf("1e-50")
    for mask in REQUESTS:
        left, right = native(aparch, mask), native(garch, mask)
        assert_success(left, aparch, mask)
        assert_success(right, garch, mask)
        projection = (left["f"], left["g"][common], left["h"][np.ix_(common, common)])
        assert_jet_close(dict(zip(("f", "g", "h"), projection)), reference, mask, density)
        for bit, name, value in zip((1, 2, 4), ("f", "g", "h"), projection):
            if mask & bit:
                # Same mathematical objective; only binary64 arithmetic may differ.
                np.testing.assert_allclose(value, right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_skewt_zero_skew_reduces_to_studentt_common_jet(native, model, p, q):
    student = case(model, "studentt", p, q, first=2, h0=2.7)
    skew = case(model, "skewt", p, q, parameters=np.r_[student["parameters"], 0.0], first=2, h0=2.7)
    expected = full_jet(student)
    with mp.workdps(70):
        assert abs(mp_value(student) - mp_value(skew)) < mp.mpf("1e-40")
    for mask in REQUESTS:
        left, right = native(skew, mask), native(student, mask)
        assert_success(left, skew, mask)
        assert_success(right, student, mask)
        common = dict(f=left["f"], g=left["g"][:-1], h=left["h"][:-1, :-1])
        assert_jet_close(common, expected, mask, "studentt")
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                np.testing.assert_allclose(common[name], right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_ged_shape_two_reduces_to_normal_common_jet(native, model, p, q):
    normal = case(model, "normal", p, q, first=3, h0=2.7)
    ged = case(model, "ged", p, q, parameters=np.r_[normal["parameters"], 2.0], first=3, h0=2.7)
    expected = full_jet(normal)
    with mp.workdps(70):
        assert abs(mp_value(normal) - mp_value(ged)) < mp.mpf("1e-50")
    for mask in REQUESTS:
        left, right = native(ged, mask), native(normal, mask)
        assert_success(left, ged, mask)
        assert_success(right, normal, mask)
        common = dict(f=left["f"], g=left["g"][:-1], h=left["h"][:-1, :-1])
        assert_jet_close(common, expected, mask, "normal")
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                np.testing.assert_allclose(common[name], right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_fixed_zero_observations_have_no_invented_parameter_singularity(native, model, density):
    declared = (small_power_case(density) if model == "aparch"
                else case(model, density, y=[0.0, -0.37, 0.0, 0.81, 0.0]))
    if density == "variance_gamma":
        declared["parameters"][volatility_width(declared)] = 0.0
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert_mp_agreement(declared, expected[0])


@pytest.mark.parametrize("p,q", ((1, 1), (3, 2), (4, 3)))
def test_large_seed_delta_derivatives_are_representable_and_preserved(native, p, q):
    declared = seed_delta_case(p, q)
    expected = full_jet(declared)
    index = volatility_width(declared) - 1
    assert abs(expected[1][index]) > 100.0
    assert abs(expected[2][index, index]) > 100.0
    actual = exercise_requests(native, declared, expected, value_atol=3e-10)
    assert_mp_agreement(declared, actual["f"], atol=3e-10)
    # An independent closed delta calculation retains the seed's delta dependence.
    with mp.workdps(90):
        omega, beta, delta = (mp.mpf(float(declared["parameters"][index]))
                              for index in (0, 1 + 2 * p, volatility_width(declared) - 1))
        k = mp.log(mp.mpf(declared["h0"])) / 2
        seed = mp.exp(delta * k)
        powers = (omega + beta * seed, omega * (1 + beta) + beta * beta * seed)
        derivatives = (beta * seed * k, beta * beta * seed * k)
        second_derivatives = (beta * seed * k * k, beta * beta * seed * k * k)
        exact_g = mp.fsum(-mp.log(v) / delta ** 2 + d / (delta * v)
                         for v, d in zip(powers, derivatives))
        exact_h = mp.fsum(2 * mp.log(v) / delta ** 3 - 2 * d / (delta ** 2 * v)
                         + (dd / v - (d / v) ** 2) / delta
                         for v, d, dd in zip(powers, derivatives, second_derivatives))
    np.testing.assert_allclose([expected[1][index], expected[2][index, index]],
                               [float(exact_g), float(exact_h)], rtol=2e-12, atol=2e-10)


def test_aparch_representable_fg_and_unrepresentable_h_have_request_specific_validity(native):
    declared = case("aparch", parameters=[1e-300, 0.0, 0.0, 0.5, 4.0],
                    y=np.zeros(3), h0=1e-200, case_id="aparch-normal-hessian-range-only")
    declared["power_log_units"] = -150.0
    jax, jnp, _ = jax_modules()
    objective = theta_objective(declared)
    point = jnp.asarray(declared["parameters"])
    expected = float(objective(point)), np.asarray(jax.grad(objective)(point))
    with mp.workdps(90):
        omega, beta, delta = mp.mpf(1e-300), mp.mpf(0.5), mp.mpf(4)
        seed = mp.mpf(1e-200) ** (delta / 2)
        first, second = omega + beta * seed, (1 + beta) * omega + beta ** 2 * seed
        exact_g = (1 / first + (1 + beta) / second) / delta
        exact_h = -(1 / first ** 2 + (1 + beta) ** 2 / second ** 2) / delta
        assert abs(exact_h) > mp.mpf(np.finfo(np.float64).max)
    assert np.isfinite(expected[1]).all()
    np.testing.assert_allclose(expected[1][0], float(exact_g), rtol=2e-12, atol=0.0)
    assert_mp_agreement(declared, expected[0])
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["valid_mask"] == mask & 3
        assert actual["status"] == (NUMERICAL_RANGE if mask & 4 else 0)
        np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                assert actual["reasons"][index] == (NUMERICAL_RANGE if bit == 4 else 0)
        if mask & 1:
            assert abs(actual["f"] - expected[0]) < 3e-9
        if mask & 2:
            np.testing.assert_allclose(actual["g"], expected[1], rtol=3e-8, atol=3e-8)


def test_egarch_nonrepresentable_objective_is_range_failure_not_parameter_domain(native):
    declared = case("egarch", parameters=[-1e300, 0.085, -0.16, 0.38], y=[0.37, -0.81])
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["valid_mask"] == 0
        assert actual["status"] == NUMERICAL_RANGE
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                assert actual["reasons"][index] == NUMERICAL_RANGE


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", ("normal", "studentt", "skewt", "ged"))
def test_long_near_boundary_recursive_sensitivities_match_ad(native, model, density):
    declared = long_horizon_case(model, density)
    expected, ordinary = full_jet(declared), full_jet(case(model, density))
    assert np.max(np.abs(expected[1])) > 100.0 * np.max(np.abs(ordinary[1]))
    assert np.max(np.abs(expected[2])) > 100.0 * np.max(np.abs(ordinary[2]))
    actual = exercise_requests(native, declared, expected, value_atol=5e-8)
    assert_mp_agreement(declared, actual["f"], atol=5e-8)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_values_below_historical_variance_floor_remain_unmodified(native, model, p, q):
    declared = tiny_scale_case(model, p, q)
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert_mp_agreement(declared, expected[0])
    changed = deepcopy(declared)
    changed["h0"] = 1e-12
    assert abs(expected[0] - float(mp_value(changed))) > 1.0


@pytest.mark.parametrize("declared", domain_cases(), ids=lambda item: item["case_id"])
def test_true_parameter_domain_violations_are_not_reported_as_kinks(native, declared):
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] == MATHEMATICAL_DOMAIN
        assert actual["valid_mask"] == 0
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                assert actual["reasons"][index] == MATHEMATICAL_DOMAIN


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", ("gh", "nig", "hyperbolic", "variance_gamma"))
def test_density_numerical_support_is_not_the_mathematical_rho_domain(native, model, density):
    declared = case(model, density)
    point = declared["parameters"].copy()
    point[volatility_width(declared)] = 0.9999
    declared = with_parameters(declared, point)
    assert abs(point[volatility_width(declared)]) < 1.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] == NUMERICAL_RANGE
        assert actual["valid_mask"] == 0
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                assert actual["reasons"][index] == NUMERICAL_RANGE


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((0, 1), (1, 0), (0, 0)))
def test_zero_orders_are_outside_declared_power_scope(native, model, p, q):
    declared = case(model)
    declared.update(p=p, q=q)
    with pytest.raises(ValueError, match="Native preparation failed"):
        native(declared, 1)
