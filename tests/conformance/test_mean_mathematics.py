"""Mandatory independent constant-scale ARMA and external linear mean jets."""

from __future__ import annotations

from copy import deepcopy
import math
import sys

import mpmath as mp
import numpy as np
import pytest

from fixtures.mean_cases import (
    DENSITIES, REQUESTS, SHAPES, arma_case, chart_case, expected_dispatch,
    linear_case, long_ma_case, ordinary_cases, original_z_cases, scored_suffix,
)
from oracles.garch import jax_modules
from oracles.mean import (
    coordinate_map, full_jet, jet, mean_count, mp_residuals, mp_value,
    numpy_residuals, pullback, reflection_coefficients, reflection_mp,
    scipy_value, state_jet,
)
from oracles.values import scipy_logpdf


LIGHT = ("normal", "studentt", "skewt", "ged")


def assert_success(actual, declared, mask):
    assert actual["status"] == 0, actual.get("reasons")
    assert actual["valid_mask"] == mask
    assert "h0" not in declared
    np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
    width = len(declared["theta_names"])
    assert actual["g"].shape == (width,)
    assert actual["h"].shape == (width, width)
    assert np.asarray(actual["theta"]).shape == (width,)
    for bit, name in ((1, "f"), (2, "g"), (4, "h")):
        if not mask & bit:
            output = np.asarray(actual[name])
            sentinel = np.full_like(output, actual["sentinel"])
            np.testing.assert_array_equal(output.view(np.uint64), sentinel.view(np.uint64))


def assert_requested_close(actual, expected, mask, density, *, value_atol=3e-9):
    light = density in LIGHT
    for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
        if not mask & bit:
            continue
        assert np.isfinite(actual[name]).all() and np.isfinite(reference).all(), name
        rtol = 0.0 if bit == 1 else ((3e-8 if bit == 2 else 3e-7) if light
                                    else (3e-7 if bit == 2 else 4e-6))
        atol = value_atol if bit == 1 else rtol
        np.testing.assert_allclose(actual[name], reference, rtol=rtol, atol=atol,
                                   err_msg=f"{density} {name} mask={mask}")


def exercise_requests(native, declared, expected, *, value_atol=3e-9):
    full = native(declared, 7)
    assert_success(full, declared, 7)
    assert_requested_close(full, expected, 7, declared["density"], value_atol=value_atol)
    if not declared["transformed"]:
        np.testing.assert_array_equal(full["theta"], declared["parameters"])
    for mask in REQUESTS:
        actual = full if mask == 7 else native(declared, mask)
        assert_success(actual, declared, mask)
        assert_requested_close(actual, expected, mask, declared["density"], value_atol=value_atol)
        for bit, name in ((1, "f"), (2, "g"), (4, "h")):
            if mask & bit:
                np.testing.assert_allclose(actual[name], full[name], rtol=0.0 if bit == 1 else 2e-12,
                                           atol=2e-10 if bit == 1 else 2e-12)
        np.testing.assert_array_equal(actual["theta"], full["theta"])
    return full


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_mean_theta_all_masks_match_independent_ad(native, declared):
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("declared", original_z_cases(), ids=lambda item: item["case_id"])
def test_mean_original_z_all_masks_include_complete_map_curvature(native, declared):
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    _, jnp, _ = jax_modules()
    expected_map = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], expected_map, rtol=3e-14, atol=3e-15)
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    exercise_requests(native, declared, expected)
    assert abs(expected[0] - float(mp_value(declared, theta=actual["theta"]))) <= 3e-9


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_mean_fixed_and_runtime_dispatch_contract(native, declared):
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert actual["dispatch"] == expected_dispatch(declared), (
        declared["case_id"], actual["dispatch"], expected_dispatch(declared))


@pytest.mark.parametrize("model", ("arma", "linear"))
@pytest.mark.parametrize("density", DENSITIES)
def test_mean_additive_values_have_scale_and_density_constants(native, model, density):
    declared = (arma_case(density, 3, 2, first=6) if model == "arma"
                else linear_case(density, 4, constant_column=True, first=3))
    low, high = mp_value(declared, digits=60), mp_value(declared, digits=90)
    with mp.workdps(90):
        assert abs(low - high) < mp.mpf("1e-48")
    expected = float(high)
    assert abs(scipy_value(declared) - expected) <= 3e-10
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert math.isfinite(actual["f"])
        assert abs(actual["f"] - expected) <= 3e-9


@pytest.mark.parametrize("density", DENSITIES)
def test_scored_suffix_does_not_reset_ma_history(native, density):
    declared = arma_case(density, 0, 2, first=6, stop=11)
    changed = deepcopy(declared)
    changed["y"][3] *= 3.0
    references = [full_jet(item) for item in (declared, changed)]
    assert abs(references[0][0] - references[1][0]) > 1e-6
    for item, expected in zip((declared, changed), references):
        exercise_requests(native, item, expected)
        assert abs(expected[0] - float(mp_value(item))) <= 3e-9


@pytest.mark.parametrize("model", ("arma", "linear"))
@pytest.mark.parametrize("density", DENSITIES)
def test_mean_holdback_and_scoring_use_original_positions(native, model, density):
    base = arma_case(density, 2, 3, intercept=False) if model == "arma" else linear_case(density, 4)
    declared = scored_suffix(base, 8, 11, hold_back=2)
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    changed = deepcopy(declared)
    changed["y"][11] *= 100.0
    if model == "linear":
        changed["design"][11] *= 100.0
    actual = native(changed, 7)
    assert_success(actual, changed, 7)
    assert_requested_close(actual, expected, 7, density)


@pytest.mark.parametrize("p,q,intercept", ((0, 0, False), (1, 1, True), (3, 2, False), (0, 4, False)))
def test_oracle_arma_state_derivatives_respect_conditioning(p, q, intercept):
    declared = arma_case(p=p, q=q, intercept=intercept)
    residuals, gradient, hessian = state_jet(declared)
    begin, count = declared["recursion_begin"], mean_count(declared)
    np.testing.assert_array_equal(residuals[:begin], np.zeros(begin))
    np.testing.assert_array_equal(gradient[:begin], np.zeros_like(gradient[:begin]))
    np.testing.assert_array_equal(hessian[:begin], np.zeros_like(hessian[:begin]))
    np.testing.assert_allclose(residuals, numpy_residuals(declared), rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(gradient[:, count:], np.zeros_like(gradient[:, count:]))
    np.testing.assert_array_equal(hessian[:, count:], np.zeros_like(hessian[:, count:]))
    if p == q == 1 and intercept:
        assert gradient[begin, 2] == 0.0
        assert hessian[begin + 1, 0, 2] == 1.0


@pytest.mark.parametrize("columns,intercept", ((0, False), (0, True), (3, True), (4, False)))
def test_linear_normal_complete_jet_matches_closed_form(native, columns, intercept):
    declared = linear_case(columns=columns, intercept=intercept, constant_column=not intercept)
    x = declared["design"]
    if intercept:
        x = np.column_stack([np.ones(len(x)), x])
    count = mean_count(declared)
    coefficients, variance = declared["parameters"][:count], declared["parameters"][count]
    residuals = declared["y"] - x @ coefficients
    square_sum = float(residuals @ residuals)
    mean_gradient = -x.T @ residuals / variance
    gradient = np.r_[mean_gradient, len(residuals) / (2 * variance) - square_sum / (2 * variance ** 2)]
    hessian = np.empty((count + 1, count + 1))
    hessian[:count, :count] = x.T @ x / variance
    hessian[:count, count] = x.T @ residuals / variance ** 2
    hessian[count, :count] = hessian[:count, count]
    hessian[count, count] = -len(residuals) / (2 * variance ** 2) + square_sum / variance ** 3
    exercise_requests(native, declared, (float(mp_value(declared)), gradient, hessian))


def reduction_bound(native, left, right, common_right):
    masses = []
    for declared, common in ((left, np.arange(len(left["parameters"]))), (right, common_right)):
        index = mean_count(declared)
        variance, shape = declared["parameters"][index], declared["parameters"][index + 1:]
        rows = np.asarray(declared["scored_rows"], dtype=int)
        residuals = numpy_residuals(declared)[rows - declared["hold_back"]]
        logs = scipy_logpdf(residuals / np.sqrt(variance), declared["density"], shape)
        base = float(scipy_logpdf(0.0, declared["density"], shape))
        value_mass = math.fsum(map(float, abs(np.log(variance) / 2) + abs(base) + np.abs(logs - base)))
        derivative_mass = np.zeros(2)
        for row in rows:
            singleton = scored_suffix(declared, int(row), int(row + 1))
            output = native(singleton, 7)
            assert_success(output, singleton, 7)
            g, h = output["g"][common], output["h"][np.ix_(common, common)]
            assert np.isfinite(g).all() and np.isfinite(h).all()
            derivative_mass += [np.max(np.abs(g)), np.max(np.abs(h))]
        masses.append(np.r_[value_mass, derivative_mass])
    count = 8 * len(left["scored_rows"]) + 64
    unit_roundoff = np.finfo(float).eps / 2
    gamma = count * unit_roundoff / (1 - count * unit_roundoff)
    return 2 * gamma / (1 - gamma) * np.maximum(1.0, np.maximum(*masses))


def exercise_reduction(native, left, right, common_right):
    bound = reduction_bound(native, left, right, common_right)
    for mask in REQUESTS:
        a, b = native(left, mask), native(right, mask)
        assert_success(a, left, mask)
        assert_success(b, right, mask)
        common = (b["f"], b["g"][common_right], b["h"][np.ix_(common_right, common_right)])
        for index, (bit, name, other) in enumerate(zip((1, 2, 4), ("f", "g", "h"), common)):
            if mask & bit:
                assert np.isfinite(a[name]).all() and np.isfinite(other).all()
                difference = float(np.max(np.abs(a[name] - other)))
                assert difference <= bound[index], (name, mask, difference, bound[index])


@pytest.mark.parametrize("model", ("arma", "linear"))
def test_constant_mean_hansen_zero_skew_reduces_to_student_t(native, model):
    left = arma_case("studentt", y=[0.0, 0.0, 0.37, -0.81]) if model == "arma" else linear_case("studentt")
    if model == "arma":
        left["parameters"][:mean_count(left)] = 0.0
    right = deepcopy(left)
    right.update(density="skewt", parameters=np.r_[left["parameters"], 0.0],
                 theta_names=left["theta_names"] + ("lambda",))
    exercise_reduction(native, left, right, np.arange(len(left["parameters"])))


@pytest.mark.parametrize("model", ("arma", "linear"))
@pytest.mark.parametrize("density,lam", (("nig", -0.5), ("hyperbolic", 1.0)))
def test_constant_mean_fixed_lambda_gh_reductions(native, model, density, lam):
    left = arma_case(density) if model == "arma" else linear_case(density)
    right = deepcopy(left)
    right.update(density="gh", parameters=np.r_[left["parameters"], lam],
                 theta_names=left["theta_names"] + ("lambda",))
    exercise_reduction(native, left, right, np.arange(len(left["parameters"])))


@pytest.mark.parametrize("model", ("arma", "linear"))
def test_constant_mean_ged_two_reduces_to_normal_including_active_zero(native, model):
    left = arma_case(y=[0.0, 0.0, 0.37, -0.81]) if model == "arma" else linear_case()
    if model == "arma":
        left["parameters"][:mean_count(left)] = 0.0
    right = deepcopy(left)
    right.update(density="ged", parameters=np.r_[left["parameters"], 2.0],
                 theta_names=left["theta_names"] + ("nu",))
    exercise_reduction(native, left, right, np.arange(len(left["parameters"])))
    exercise_requests(native, right, full_jet(right))


@pytest.mark.parametrize("density", DENSITIES)
def test_implicit_linear_intercept_matches_explicit_constant_column(native, density):
    left = linear_case(density, 3, intercept=True)
    right = linear_case(density, design=np.column_stack([np.ones(len(left["y"])), left["design"]]),
                        parameters=left["parameters"], intercept=False)
    exercise_reduction(native, left, right, np.arange(len(left["parameters"])))


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("columns", (0, 4))
def test_large_constant_scale_keeps_normalized_density_and_derivatives(native, density, columns):
    scaled = linear_case(density, columns, constant_column=True)
    count = mean_count(scaled)
    scaled["parameters"][:count] = 0.0
    scaled["parameters"][count] = 1.0
    expected = full_jet(scaled)
    factor, variance_factor = 1e150, 1e300
    declared = deepcopy(scaled)
    declared["y"] *= factor
    declared["design"] *= factor
    declared["parameters"][count] *= variance_factor
    units = np.ones(len(declared["parameters"]))
    units[count] = variance_factor
    target = (expected[0] + len(declared["scored_rows"]) * math.log(factor),
              expected[1] / units, (expected[2] / units[:, None]) / units[None, :])
    absolute = (2e-11, 3e-12 / units, (3e-12 / units[:, None]) / units[None, :])
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        for bit, name, reference, allowance in zip((1, 2, 4), ("f", "g", "h"), target, absolute):
            if mask & bit:
                assert np.isfinite(actual[name]).all() and np.isfinite(reference).all()
                tolerance = allowance + (0.0 if bit == 1 else 3e-10 * np.abs(reference))
                assert np.all(np.abs(actual[name] - reference) <= tolerance), (mask, name)


@pytest.mark.parametrize("model", ("arma", "linear"))
@pytest.mark.parametrize("shape,valid", ((0.8, (1,)), (1.0, (1,)), (1.5, (1, 2, 3)), (2.0, REQUESTS), (3.0, REQUESTS)))
def test_active_ged_centre_has_request_specific_validity(native, model, shape, valid):
    declared = (arma_case("ged", 0, 0, parameters=[0.0, 1.3, shape], y=[0.0, 0.0, 0.0])
                if model == "arma" else linear_case("ged", 0, intercept=True,
                                                     parameters=[0.0, 1.3, shape], y=[0.0, 0.0]))
    value = float(mp_value(declared))
    for mask in REQUESTS:
        actual = native(declared, mask)
        if mask in valid:
            assert_success(actual, declared, mask)
            expected = jet(declared, request=mask)
            assert_requested_close(actual, expected, mask, "ged")
            if mask & 1:
                assert abs(actual["f"] - value) <= 3e-9
        else:
            missing = 6 if shape <= 1.0 else 4
            assert actual["status"] != 0
            assert not actual["valid_mask"] & missing
            assert not actual["valid_mask"] & ~mask


@pytest.mark.parametrize("model", ("arma", "linear"))
def test_inactive_zero_ged_residual_does_not_invent_a_kink(native, model):
    declared = (arma_case("ged", 0, 0, intercept=False, parameters=[1.3, 0.8], y=[0.0, 0.0, 0.37])
                if model == "arma" else linear_case("ged", 1, intercept=False,
                                                     design=np.zeros((3, 1)), parameters=[0.2, 1.3, 0.8],
                                                     y=[0.0, 0.0, 0.37]))
    exercise_requests(native, declared, full_jet(declared))


@pytest.mark.parametrize("density", DENSITIES)
def test_constant_scale_density_domain_rejections_have_no_valid_jet(native, density):
    declared = linear_case(density, 4)
    index = mean_count(declared)
    if density == "normal":
        declared["parameters"][index] = 0.0
    elif density == "studentt":
        declared["parameters"][-1] = 2.0
    elif density == "skewt":
        declared["parameters"][-1] = 1.0
    elif density == "ged":
        declared["parameters"][-1] = 0.0
    else:
        declared["parameters"][index + 1] = 1.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] != 0
        assert actual["valid_mask"] == 0


@pytest.mark.parametrize("part", ("ar", "ma"))
def test_arma_polynomial_domain_is_not_individual_coefficient_clipping(native, part):
    declared = (arma_case(p=2, q=0, intercept=False, parameters=[0.7, 0.6, 1.3]) if part == "ar"
                else arma_case(p=0, q=2, intercept=False, parameters=[0.7, -0.6, 1.3]))
    polynomial = np.r_[1.0, -declared["parameters"][:2] if part == "ar" else declared["parameters"][:2]]
    assert np.max(np.abs(np.roots(polynomial))) > 1.0
    assert np.max(np.abs(declared["parameters"][:2])) < 1.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] != 0
        assert actual["valid_mask"] == 0


@pytest.mark.parametrize("order", (0, 1, 3, 5))
def test_oracle_reflection_values_match_mp_and_stable_polynomials(order):
    _, jnp, _ = jax_modules()
    coordinates = np.linspace(-0.23, 0.31, order)
    for sign in (1.0, -1.0):
        actual = np.asarray(reflection_coefficients(jnp.asarray(coordinates), sign))
        with mp.workdps(90):
            expected = np.array(list(map(float, reflection_mp(coordinates, sign))))
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=3e-15)
        if order:
            polynomial = np.r_[1.0, -actual if sign > 0 else actual]
            assert np.max(np.abs(np.roots(polynomial))) < 1.0


@pytest.mark.parametrize("part,coordinate", (("ar", -18.0), ("ar", 18.0), ("ma", -18.0), ("ma", 18.0)))
def test_original_z_reflection_tail_keeps_map_curvature(native, part, coordinate):
    declared = chart_case(arma_case(p=int(part == "ar"), q=int(part == "ma"), intercept=False))
    declared["parameters"][0] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta_jet = full_jet(declared, theta=actual["theta"])
    exercise_requests(native, declared, pullback(declared, theta_jet))
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        sign = 1 if part == "ar" else -1
        slope = float(sign / mp.cosh(z) ** 2)
        curvature = float(-2 * sign * mp.tanh(z) / mp.cosh(z) ** 2)
    assert slope != 0.0
    np.testing.assert_allclose(actual["g"][0] / slope, theta_jet[1][0], rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(actual["h"][0, 0] / slope,
                               slope * theta_jet[2][0, 0] + curvature / slope * theta_jet[1][0],
                               rtol=3e-7, atol=3e-7)


def test_long_ma_recursion_near_invertibility_preserves_large_sensitivities(native):
    declared = long_ma_case()
    residuals, gradient, hessian = state_jet(declared)
    assert np.isfinite(residuals).all() and np.isfinite(gradient).all() and np.isfinite(hessian).all()
    assert abs(gradient[-1, 0]) > 250.0
    assert abs(hessian[-1, 0, 1]) > 1e5
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_requests(native, declared, expected)


def test_mean_constant_scale_normal_finite_score_survives_inverse_square_overflow(native):
    variance, residual = 2.0 ** -540, 2.0 ** -269
    declared = linear_case(columns=0, intercept=True, parameters=[0.0, variance], y=[residual, residual])
    expected = np.array([-2 * residual / variance, -3 / variance])
    for mask in (2, 3):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert np.isfinite(actual["g"]).all()
        np.testing.assert_allclose(actual["g"], expected, rtol=3e-15, atol=0.0)
    for mask in (4, 5, 6, 7):
        actual = native(declared, mask)
        assert actual["status"] != 0 and not actual["valid_mask"] & 4


def test_mean_scalar_recovery_preserves_an_exact_small_nonzero_residual(native):
    small = 2.0 ** -27
    design = np.zeros((5, 4))
    design[:, 0] = 1.0 - small
    declared = linear_case(design=design, parameters=[1.0 + small, 0.0, 0.0, 0.0, 1e-32], y=np.ones(5))
    with mp.workdps(90):
        residuals = mp_residuals(declared, declared["parameters"])
        assert all(e == mp.mpf(2) ** -54 for e in residuals)
    value = float(mp_value(declared))
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert abs(actual["f"] - value) <= 2e-11


def test_mean_conformance_does_not_import_legacy_package():
    assert not any(name == "scivol" or name.startswith("scivol.") for name in sys.modules)
