"""Mandatory independent joint ARMA/external LinearMean + GARCH/GJR evidence.

Collection initializes neither JAX nor a native bridge. Chief-owned sequential
execution is required; a collected or scalar-only subset is not qualification.
"""

from __future__ import annotations

from copy import deepcopy
import math
import sys

import mpmath as mp
import numpy as np
import pytest

from fixtures.joint_cases import (
    DENSITIES, FAMILIES, REQUESTS, SHAPES, case, chart_case, expected_dispatch,
    long_horizon_case, mean_width, ordinary_cases, original_z_cases, scaled_case,
    scored_suffix, volatility_width,
)
from oracles.garch import jax_modules
from oracles.joint import (
    coordinate_map, full_jet, jet, map_jet, mp_states, mp_value, numpy_states,
    pullback, scipy_value, shape_start, state_jet,
)
from oracles.mean import reflection_mp
from oracles.values import scipy_logpdf


LIGHT = ("normal", "studentt", "skewt", "ged")
SCALAR_CASES = tuple(case(model, density, p_ar=2, q_ma=3, p_vol=3, q_vol=2,
                         columns=4, first=8, stop=11, hold_back=2)
                     for model in FAMILIES for density in DENSITIES)


def assert_success(actual, declared, mask):
    assert actual["status"] == 0, actual.get("reasons")
    assert actual["valid_mask"] == mask
    np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
    width = len(declared["theta_names"])
    assert actual["g"].shape == (width,)
    assert actual["h"].shape == (width, width)
    assert actual["theta"].shape == (width,)
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
        np.testing.assert_allclose(actual[name], reference, rtol=rtol,
                                   atol=value_atol if bit == 1 else rtol,
                                   err_msg=f"{density} {name} mask={mask}")


def exercise_requests(native, declared, expected, *, value_atol=3e-9, derivative_units=None):
    full = native(declared, 7)
    assert_success(full, declared, 7)
    assert_requested_close(full, expected, 7, declared["density"], value_atol=value_atol)
    if not declared["transformed"]:
        np.testing.assert_array_equal(full["theta"], declared["parameters"])
    for mask in REQUESTS:
        actual = full if mask == 7 else native(declared, mask)
        assert_success(actual, declared, mask)
        assert_requested_close(actual, expected, mask, declared["density"], value_atol=value_atol)
        if derivative_units is not None:
            weights = derivative_units
            normalized = dict(g=actual["g"] * weights,
                              h=actual["h"] * weights[:, None] * weights[None, :])
            reference = (None, expected[1] * weights,
                         expected[2] * weights[:, None] * weights[None, :])
            assert_requested_close(normalized, reference, mask & 6, declared["density"])
        for bit, name in ((1, "f"), (2, "g"), (4, "h")):
            if mask & bit:
                np.testing.assert_allclose(actual[name], full[name], rtol=0.0 if bit == 1 else 2e-12,
                                           atol=2e-10 if bit == 1 else 2e-12)
        np.testing.assert_array_equal(actual["theta"], full["theta"])
    return full


def assert_cross_blocks(actual, expected, declared):
    width, shape = mean_width(declared), shape_start(declared)
    blocks = [np.arange(width), np.arange(width, shape)]
    if shape < len(declared["theta_names"]):
        blocks.append(np.arange(shape, len(declared["theta_names"])))
    for left in blocks:
        for right in blocks:
            if not len(left) or not len(right):
                continue
            index = np.ix_(left, right)
            assert_requested_close(dict(h=actual["h"][index]),
                                   (None, None, expected[2][index]), 4, declared["density"])
    np.testing.assert_allclose(actual["h"], actual["h"].T, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_joint_theta_all_masks_and_full_crossblocks_match_independent_ad(native, declared):
    expected = full_jet(declared)
    actual = exercise_requests(native, declared, expected)
    assert_cross_blocks(actual, expected, declared)


@pytest.mark.parametrize("declared", original_z_cases(), ids=lambda item: item["case_id"])
def test_joint_original_z_all_masks_keep_actual_theta_and_map_curvature(native, declared):
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    assert actual["dispatch"] == expected_dispatch(declared)
    _, jnp, _ = jax_modules()
    independently_mapped = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], independently_mapped, rtol=3e-12, atol=3e-13)
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    exercise_requests(native, declared, expected)
    assert_cross_blocks(actual, expected, declared)
    assert abs(expected[0] - float(mp_value(declared, theta=actual["theta"]))) <= 3e-9


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_joint_fixed_and_general_dispatch_contract(native, declared):
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert actual["dispatch"] == expected_dispatch(declared), declared["case_id"]


@pytest.mark.parametrize("declared", SCALAR_CASES, ids=lambda item: item["case_id"])
def test_joint_oracle_scalar_mp_and_scipy_are_independent(declared):
    low, high = mp_value(declared, digits=60), mp_value(declared, digits=90)
    with mp.workdps(90):
        assert abs(low - high) < mp.mpf("1e-48")
    assert math.isfinite(float(high))
    assert abs(scipy_value(declared) - float(high)) <= 3e-10


@pytest.mark.parametrize("declared", SCALAR_CASES, ids=lambda item: item["case_id"])
def test_joint_additive_values_include_every_constant(native, declared):
    expected = float(mp_value(declared, digits=90))
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert math.isfinite(actual["f"])
        assert abs(actual["f"] - expected) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_holdback_score_suffix_and_stop_are_independent(native, model, density):
    declared = case(model, density, p_ar=2, q_ma=3, p_vol=3, q_vol=2, columns=4,
                    first=8, stop=11, hold_back=2)
    # Nontrivial IDs distinguish score positions from labels and calendar gaps.
    declared["rows"] = np.arange(len(declared["y"]), dtype=np.uint64) * 7 + 101
    declared["scored_rows"] = tuple(declared["rows"][8:11])
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    changed = deepcopy(declared)
    changed["y"][:2] *= 100.0
    changed["y"][11:] *= 100.0
    changed["design"][:2] *= 100.0
    changed["design"][11:] *= 100.0
    exercise_requests(native, changed, expected)
    prefix = deepcopy(declared)
    prefix["y"][6] *= 3.0
    prefix_jet = full_jet(prefix)
    assert abs(prefix_jet[0] - expected[0]) > 1e-6
    exercise_requests(native, prefix, prefix_jet)
    assert abs(prefix_jet[0] - float(mp_value(prefix))) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
def test_joint_oracle_recursive_state_seed_sensitivities_and_cross_terms(model):
    declared = case(model, columns=3)
    states, first, second = state_jet(declared)
    e, h = numpy_states(declared)
    np.testing.assert_allclose(states, np.column_stack([e, h]), rtol=3e-14, atol=3e-15)
    with mp.workdps(90):
        mp_e, mp_h = mp_states(declared, declared["parameters"])
    np.testing.assert_allclose(states, np.column_stack([list(map(float, mp_e)), list(map(float, mp_h))]),
                               rtol=3e-14, atol=3e-15)
    begin, mean = declared["recursion_begin"], mean_width(declared)
    if model.startswith("arma_"):
        np.testing.assert_array_equal(states[:begin, 0], np.zeros(begin))
        np.testing.assert_array_equal(states[:begin, 1], np.full(begin, declared["h0"]))
        np.testing.assert_array_equal(first[:begin], np.zeros_like(first[:begin]))
        np.testing.assert_array_equal(second[:begin], np.zeros_like(second[:begin]))
        beta = mean + 1 + (2 if model.endswith("gjr") else 1) * declared["p_vol"]
        assert states[begin, 1] == pytest.approx(declared["parameters"][mean]
                                               + declared["parameters"][beta] * declared["h0"])
        assert first[begin, 1, mean] == 1.0
        assert first[begin, 1, beta] == declared["h0"]
        assert first[begin, 0, mean - 1] == 0.0
        assert second[begin + 1, 0, 0, mean - 1] == 1.0
    else:
        assert states[0, 1] == declared["h0"]
        np.testing.assert_array_equal(first[0, 1], np.zeros_like(first[0, 1]))
        np.testing.assert_array_equal(second[0, 1], np.zeros_like(second[0, 1]))
    assert second[begin + 1, 1, 0, mean + 1] == pytest.approx(-2 * states[begin, 0])
    shape = shape_start(declared)
    np.testing.assert_array_equal(first[:, :, shape:], np.zeros_like(first[:, :, shape:]))
    np.testing.assert_array_equal(second[:, :, shape:], np.zeros_like(second[:, :, shape:]))


def reduction_bound(native, left, right, common_right):
    masses = []
    for declared, common in ((left, np.arange(len(left["parameters"]))), (right, common_right)):
        rows = np.asarray(declared["scored_positions"]) - declared["hold_back"]
        e, h = numpy_states(declared)
        shape = declared["parameters"][shape_start(declared):]
        logpdf = scipy_logpdf(e[rows] / np.sqrt(h[rows]), declared["density"], shape)
        base = float(scipy_logpdf(0.0, declared["density"], shape))
        value_mass = math.fsum(map(float, abs(np.log(h[rows]) / 2) + abs(base) + abs(logpdf - base)))
        derivative_mass = np.zeros(2)
        for row in declared["scored_positions"]:
            singleton = scored_suffix(declared, row, row + 1)
            actual = native(singleton, 7)
            assert_success(actual, singleton, 7)
            g, hessian = actual["g"][common], actual["h"][np.ix_(common, common)]
            assert np.isfinite(g).all() and np.isfinite(hessian).all()
            derivative_mass += [np.max(abs(g)), np.max(abs(hessian))]
        masses.append(np.r_[value_mass, derivative_mass])
    k = 8 * len(left["scored_positions"]) + 64
    u = np.finfo(float).eps / 2
    gamma = k * u / (1 - k * u)
    return 2 * gamma / (1 - gamma) * np.maximum(1.0, np.maximum(*masses))


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("reduction", ("skewt-zero", "ged-two", "nig-lambda", "hyperbolic-lambda"))
def test_joint_exact_density_reductions_all_common_derivatives(native, model, reduction):
    density, added, target = {
        "skewt-zero": ("studentt", 0.0, "skewt"),
        "ged-two": ("normal", 2.0, "ged"),
        "nig-lambda": ("nig", -0.5, "gh"),
        "hyperbolic-lambda": ("hyperbolic", 1.0, "gh"),
    }[reduction]
    left = case(model, density)
    right = deepcopy(left)
    right.update(density=target, parameters=np.r_[left["parameters"], added],
                 theta_names=left["theta_names"] + ("nu" if target == "ged" else "lambda",))
    common = np.arange(len(left["parameters"]))
    bound = reduction_bound(native, left, right, common)
    for mask in REQUESTS:
        a, b = native(left, mask), native(right, mask)
        assert_success(a, left, mask)
        assert_success(b, right, mask)
        for index, (bit, name, other) in enumerate(zip((1, 2, 4), ("f", "g", "h"),
                (b["f"], b["g"][common], b["h"][np.ix_(common, common)]))):
            if mask & bit:
                assert np.isfinite(a[name]).all() and np.isfinite(other).all()
                difference = float(np.max(abs(np.asarray(a[name]) - other)))
                assert difference <= bound[index], (model, reduction, mask, name, difference, bound[index])


@pytest.mark.parametrize("model", FAMILIES)
def test_joint_external_constant_is_not_an_implicit_intercept(native, model):
    if model.startswith("arma_"):
        declared = case(model, p_ar=0, q_ma=0, intercept=False)
        assert mean_width(declared) == 0
        exercise_requests(native, declared, full_jet(declared))
        return
    left = case(model, columns=3, intercept=True)
    right = case(model, columns=4, intercept=False,
                 design=np.column_stack([np.ones(len(left["y"])), left["design"]]),
                 parameters=left["parameters"])
    expected = full_jet(left)
    exercise_requests(native, left, expected)
    exercise_requests(native, right, expected)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("factor", (2.0 ** -34, 2.0 ** 200), ids=("tiny", "large"))
def test_joint_units_are_exact_without_variance_floor(native, model, density, factor):
    base = case(model, density)
    declared = scaled_case(base, factor)
    f, g, h = full_jet(base)
    weights = np.ones(len(base["parameters"]))
    if model.startswith("arma_"):
        if base["intercept"]:
            weights[0] = factor
    else:
        weights[:mean_width(base)] = factor
    weights[mean_width(base)] = factor * factor
    expected = f + len(base["scored_positions"]) * math.log(factor), g / weights, h / weights[:, None] / weights[None, :]
    # Normalize every mask so an absolute tolerance cannot hide tiny derivatives.
    actual = exercise_requests(native, declared, expected, derivative_units=weights)
    assert abs(actual["f"] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", LIGHT)
def test_joint_long_horizon_sensitivities_near_two_boundaries(native, model, density):
    declared = long_horizon_case(model, density)
    states, first, second = state_jet(declared)
    assert np.isfinite(states).all() and np.isfinite(first).all() and np.isfinite(second).all()
    assert np.max(abs(first[:, 1, mean_width(declared)])) > 200.0
    if model.startswith("arma_"):
        assert abs(first[-1, 0, 0]) > 150.0
        assert abs(second[-1, 0, 0, 1]) > 1e4
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("shape,valid", ((0.8, 1), (1.0, 1), (1.5, 3), (2.0, 7), (3.0, 7)))
def test_joint_active_ged_zero_has_request_specific_validity(native, model, shape, valid):
    declared = case(model, "ged", p_ar=0, q_ma=0, columns=0, y=[0.0, 0.0, 0.0])
    declared["parameters"][0] = 0.0
    declared["parameters"][-1] = shape
    if model.endswith("gjr"):
        declared["parameters"][mean_width(declared) + 2] = 0.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        if mask & ~valid:
            assert actual["status"] != 0
            assert actual["valid_mask"] == mask & valid
            if actual["valid_mask"]:
                expected = jet(declared, request=actual["valid_mask"])
                assert_requested_close(actual, expected, actual["valid_mask"], "ged")
        else:
            assert_success(actual, declared, mask)
            assert_requested_close(actual, jet(declared, request=mask), mask, "ged")


@pytest.mark.parametrize("model", ("arma_gjr", "linear_gjr"))
def test_joint_gjr_active_zero_does_not_manufacture_a_hessian(native, model):
    y = [0.1, 0.0, 0.6, -0.2] if model.startswith("arma_") else [0.0, 0.6, -0.2]
    declared = case(model, p_ar=0, q_ma=0, columns=0, y=y)
    declared["parameters"][0] = 0.0
    with pytest.raises(ValueError, match="one-sided H"):
        full_jet(declared)
    for mask in REQUESTS:
        actual = native(declared, mask)
        if mask & 4:
            assert actual["status"] != 0
            assert actual["valid_mask"] == mask & 3
            # Exact reason classification remains a separately recorded backend debt.
            assert actual["reasons"][2] in (5, 6)
            if mask & 3:
                assert_requested_close(actual, jet(declared, request=mask & 3), mask & 3, "normal")
        else:
            assert_success(actual, declared, mask)
            assert_requested_close(actual, jet(declared, request=mask), mask, "normal")


@pytest.mark.parametrize("model", FAMILIES)
def test_joint_inactive_zero_has_no_mean_or_recursion_kink(native, model):
    declared = case(model, "ged", p_ar=0, q_ma=0, columns=1, intercept=False,
                    y=[0.0, 0.0, 0.37, -0.41], design=np.zeros((4, 1)))
    exercise_requests(native, declared, full_jet(declared))


@pytest.mark.parametrize("model", FAMILIES)
def test_joint_vg_exact_symmetric_centre_keeps_mean_shape_crossblocks(native, model):
    declared = case(model, "variance_gamma", p_ar=0, q_ma=0, columns=0,
                    y=[0.0, 0.0, 0.37, -0.41])
    declared["parameters"][0] = 0.0
    declared["parameters"][-2] = 0.0
    if model.endswith("gjr"):
        declared["parameters"][mean_width(declared) + 2] = 0.0
    expected = full_jet(declared)
    actual = exercise_requests(native, declared, expected)
    assert_cross_blocks(actual, expected, declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_true_density_domain_violations_have_no_jet(native, model, density):
    declared = case(model, density)
    point = declared["parameters"]
    if density == "normal":
        point[mean_width(declared)] = 0.0
    elif density == "studentt":
        point[-1] = 2.0
    elif density == "skewt":
        point[-1] = 1.0
    elif density == "ged":
        point[-1] = 0.0
    else:
        point[shape_start(declared)] = 1.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] == 4
        assert actual["valid_mask"] == 0


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density,shape", (("studentt", (2.0 + 2.0 ** -30,)),
    ("skewt", (7.4, 1.0 - 2.0 ** -20)), ("ged", (0.2,))))
def test_joint_theoretical_shape_interior_is_not_historical_padding(native, model, density, shape):
    declared = case(model, density)
    declared["parameters"][shape_start(declared):] = shape
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", ("gh", "nig", "hyperbolic", "variance_gamma"))
def test_joint_numerical_support_is_distinct_from_rho_theoretical_domain(native, model, density):
    declared = case(model, density)
    declared["parameters"][shape_start(declared)] = 0.997
    assert abs(declared["parameters"][shape_start(declared)]) < 1.0
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] == 5
        assert actual["valid_mask"] == 0


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((0, 1), (1, 0), (0, 0)))
def test_joint_zero_volatility_orders_are_outside_declared_scope(native, model, p, q):
    declared = case(model, p_vol=p, q_vol=q)
    with pytest.raises(ValueError, match="Native preparation failed"):
        native(declared, 1)


@pytest.mark.parametrize("model", ("arma_garch", "arma_gjr"))
@pytest.mark.parametrize("part", ("ar", "ma"))
@pytest.mark.parametrize("stable", (False, True))
def test_joint_polynomial_domain_is_not_coefficient_clipping(native, model, part, stable):
    declared = case(model, p_ar=2 if part == "ar" else 0,
                    q_ma=2 if part == "ma" else 0, intercept=False)
    coefficients = np.array([1.2, -0.5] if stable else [0.7, 0.6])
    if part == "ma":
        coefficients = -coefficients
    declared["parameters"][:2] = coefficients
    roots = np.roots(np.r_[1.0, -coefficients if part == "ar" else coefficients])
    assert bool(np.max(abs(roots)) < 1.0) == stable
    if stable:
        exercise_requests(native, declared, full_jet(declared))
    else:
        for mask in REQUESTS:
            actual = native(declared, mask)
            assert actual["status"] != 0 and actual["valid_mask"] == 0


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("kind", ("negative-alpha", "negative-beta", "persistence", "combined-shock"))
def test_joint_volatility_domain_is_not_silently_clipped(native, model, kind):
    declared = case(model)
    mean = mean_width(declared)
    if kind == "negative-alpha":
        declared["parameters"][mean + 1] = -0.01
    elif kind == "negative-beta":
        declared["parameters"][shape_start(declared) - 1] = -0.01
    elif kind == "persistence":
        declared["parameters"][shape_start(declared) - 1] = 1.1
    elif model.endswith("gjr"):
        declared["parameters"][mean + 2] = -declared["parameters"][mean + 1] - 0.01
    else:
        declared["parameters"][mean + 1] = -0.01
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] == 4 and actual["valid_mask"] == 0


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_near_stationarity_interior_has_no_hidden_margin(native, model, density):
    declared = case(model, density, p_ar=0, q_ma=0)
    mean = mean_width(declared)
    declared["parameters"][mean] = 2.0 ** -60
    declared["parameters"][mean + 1] = 0.0
    if model.endswith("gjr"):
        declared["parameters"][mean + 2] = 0.0
    declared["parameters"][shape_start(declared) - 1] = 1.0 - 2.0 ** -48
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("part,coordinate", (("ar", -18.0), ("ar", 18.0), ("ma", -18.0), ("ma", 18.0)))
def test_joint_original_z_reflection_tail_keeps_original_slopes(native, part, coordinate):
    declared = chart_case(case("arma_garch", p_ar=int(part == "ar"),
                                q_ma=int(part == "ma"), intercept=False))
    declared["parameters"][0] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta_jet = full_jet(declared, theta=actual["theta"])
    exercise_requests(native, declared, pullback(declared, theta_jet))
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        sign = 1 if part == "ar" else -1
        expected_theta = float(reflection_mp([coordinate], sign)[0])
        slope = float(sign / mp.cosh(z) ** 2)
        curvature = float(-2 * sign * mp.tanh(z) / mp.cosh(z) ** 2)
    assert abs(actual["theta"][0] - expected_theta) <= np.finfo(float).eps
    assert slope != 0.0
    np.testing.assert_allclose(actual["g"][0] / slope, theta_jet[1][0], rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(actual["h"][0, 0] / slope,
                               slope * theta_jet[2][0, 0] + curvature / slope * theta_jet[1][0],
                               rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("coordinate", (36.0, 40.0))
def test_joint_original_z_rounded_shape_plateau_keeps_curvature(native, model, coordinate):
    declared = chart_case(case(model, "nig"))
    declared["parameters"][-1] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    # The scalar map may round to its upper endpoint; derivative slopes remain nonzero.
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        exponential = mp.exp(-z)
        expected_zeta = float(mp.mpf(25) - mp.mpf("24.9") * exponential / (1 + exponential))
        slope = float(mp.mpf("24.9") * exponential / (1 + exponential) ** 2)
        curvature = float(-mp.mpf("24.9") * exponential * (1 - exponential) / (1 + exponential) ** 3)
    assert abs(actual["theta"][-1] - expected_zeta) <= np.spacing(25.0)
    if coordinate == 40.0:
        assert actual["theta"][-1] == 25.0
    theta_jet = full_jet(declared, theta=actual["theta"])
    exercise_requests(native, declared, pullback(declared, theta_jet))
    if model.endswith("garch"):
        np.testing.assert_allclose(actual["g"][-1] / slope, theta_jet[1][-1], rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(actual["h"][-1, -1] / slope,
            slope * theta_jet[2][-1, -1] + curvature / slope * theta_jet[1][-1], rtol=4e-6, atol=4e-6)
    else:
        # GJR shape also moves shock coefficients through the negative moment.
        _, jacobian, map_hessian = map_jet(declared)
        expected_g = jacobian[:, -1] @ theta_jet[1]
        expected_h = jacobian[:, -1] @ theta_jet[2] @ jacobian[:, -1] + theta_jet[1] @ map_hessian[:, -1, -1]
        np.testing.assert_allclose(actual["g"][-1] / slope, expected_g / slope, rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(actual["h"][-1, -1] / slope, expected_h / slope, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("density", ("skewt", "gh", "nig", "hyperbolic", "variance_gamma"))
def test_joint_gjr_original_z_moment_and_pullback_refinement(density):
    declared = chart_case(case("arma_gjr", density, p_ar=2, q_ma=1, p_vol=2, q_vol=3))
    coarse, fine = map_jet(declared, moment_count=32), map_jet(declared, moment_count=64)
    for left, right in zip(coarse, fine):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-9)
    point = fine[0]
    theta_jet = full_jet(declared, theta=point)
    left, right = pullback(declared, theta_jet, moment_count=32), pullback(declared, theta_jet, moment_count=64)
    assert_requested_close(dict(f=left[0], g=left[1], h=left[2]), right, 7, density)
    mean, p, q = mean_width(declared), declared["p_vol"], declared["q_vol"]
    # Nonzero shape-to-volatility derivatives and curvature must enter the full pullback.
    assert np.max(abs(fine[1][mean + 1:mean + 1 + 2 * p, mean + 1 + 2 * p + q:])) > 1e-6
    _, jacobian, curvature = fine
    curvature_term = np.einsum("i,ijk->jk", theta_jet[1], curvature)
    assert np.max(abs(curvature_term)) > 1e-5


@pytest.mark.parametrize("model", ("linear_garch", "linear_gjr"))
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_mp_scalar_retains_small_nonzero_design_cancellation(native, model, density):
    small = 2.0 ** -27
    declared = case(model, density, columns=1, intercept=False,
                    design=np.full((4, 1), 1.0 - small), y=np.ones(4), h0=1e-32)
    declared["parameters"][0] = 1.0 + small
    declared["parameters"][1] = 1e-33
    with mp.workdps(90):
        residuals, variances = mp_states(declared, declared["parameters"])
        assert all(value == mp.mpf(2) ** -54 for value in residuals)
        assert all(value > 0 for value in variances)
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert abs(actual["f"] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("model", ("arma_gjr", "linear_gjr"))
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_mp_scalar_combines_negative_shock_before_raw_square_overflow(native, model, density):
    arma = model.startswith("arma_")
    y = [0.1, -1e155, 0.37, -0.81] if arma else [-1e155, 0.37, -0.81]
    declared = case(model, density, p_ar=0, q_ma=0, columns=0,
                    y=y, first=2 if arma else 1, h0=1e308)
    declared["parameters"][0] = 0.0
    offset = mean_width(declared)
    declared["parameters"][offset:offset + 4] = [1e290, 0.085, -0.085 + 2.0 ** -52, 0.0]
    with mp.workdps(90):
        residuals, variances = mp_states(declared, declared["parameters"])
        assert residuals[int(arma)] ** 2 > mp.mpf(np.finfo(float).max)
        assert all(0 < value <= mp.mpf(np.finfo(float).max) for value in variances)
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert abs(actual["f"] - float(mp_value(declared))) <= 3e-9


def test_joint_conformance_does_not_import_legacy_package():
    assert not any(name == "scivol" or name.startswith("scivol.") for name in sys.modules)
