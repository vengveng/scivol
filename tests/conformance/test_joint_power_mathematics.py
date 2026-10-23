"""Mandatory scoped joint-power AD/MP conformance for chief-owned execution.

No import starts JAX, native evaluation or a build. Missing required dependencies
fail instead of skipping. The independent oracles never import production code.
"""

from __future__ import annotations

from copy import deepcopy
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.signal import lfilter

from fixtures.joint_cases import case as joint_garch_case
from fixtures.joint_power_cases import (
    DENSITIES, FAMILIES, LAYOUTS, REQUESTS, SHAPES, case, chart_case, expected_dispatch,
    fixed_zero_row_case, historical_aparch_small_unit_case, layout_case, long_horizon_case, mean_width, ordinary_cases,
    original_z_cases, scored_suffix, volatility_width,
)
from fixtures.power_cases import case as zero_power_case
from oracles.garch import jax_modules
from oracles.joint import full_jet as joint_garch_jet, mp_value as joint_garch_mp_value
from oracles.joint_power import (
    coordinate_map, full_jet, jet, mean_case, mp_coordinate_map, mp_states, mp_value,
    original_z_jet, pullback, shape_start, state_jet, theta_objective,
)
from oracles.mean import numpy_residuals
from oracles.power import full_jet as zero_power_jet, mp_power_moment, mp_value as zero_power_mp_value


LIGHT = ("normal", "studentt", "skewt", "ged")


def assert_success(actual, declared, mask):
    assert actual["status"] == 0, (declared["case_id"], actual.get("reasons"))
    assert actual["valid_mask"] == mask
    np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
    width = len(declared["theta_names"])
    assert actual["g"].shape == (width,) and actual["h"].shape == (width, width)
    assert actual["theta"].shape == (width,)
    for index, (bit, name) in enumerate(zip((1, 2, 4), ("f", "g", "h"))):
        if mask & bit:
            assert actual["reasons"][index] == 0
        else:
            value = np.asarray(actual[name])
            expected = np.full_like(value, actual["sentinel"])
            np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))


def assert_requested_close(actual, expected, mask, density, *, value_atol=3e-9):
    for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
        if mask & bit:
            assert np.isfinite(actual[name]).all() and np.isfinite(reference).all(), name
            tolerance = (3e-8 if bit == 2 else 3e-7) if density in LIGHT else (3e-7 if bit == 2 else 4e-6)
            np.testing.assert_allclose(actual[name], reference, rtol=0.0 if bit == 1 else tolerance,
                                       atol=value_atol if bit == 1 else tolerance,
                                       err_msg=f"{density}, {name}, mask={mask}")


def assert_cross_blocks(actual, expected, declared):
    mean, shape, width = mean_width(declared), shape_start(declared), len(declared["theta_names"])
    blocks = (np.arange(mean), np.arange(mean, shape), np.arange(shape, width))
    for left in blocks:
        for right in blocks:
            if len(left) and len(right):
                indices = np.ix_(left, right)
                assert_requested_close(dict(h=actual["h"][indices]), (None, None, expected[2][indices]),
                                       4, declared["density"])
    np.testing.assert_allclose(actual["h"], actual["h"].T, rtol=2e-12, atol=2e-12)


def exercise_requests(native, declared, expected, *, value_atol=3e-9):
    full = native(declared, 7)
    assert_success(full, declared, 7)
    for mask in REQUESTS:
        actual = full if mask == 7 else native(declared, mask)
        assert_success(actual, declared, mask)
        assert_requested_close(actual, expected, mask, declared["density"], value_atol=value_atol)
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                np.testing.assert_allclose(actual[name], full[name], rtol=0.0 if bit == 1 else 2e-12,
                                           atol=2e-10 if bit == 1 else 2e-12)
        np.testing.assert_array_equal(actual["theta"], full["theta"])
    if not declared["transformed"]:
        np.testing.assert_array_equal(full["theta"], declared["parameters"])
    assert_cross_blocks(full, expected, declared)
    return full


def assert_mp_value(declared, value, *, theta=None, atol=3e-9):
    coarse, fine = mp_value(declared, theta, digits=50), mp_value(declared, theta, digits=70)
    with mp.workdps(70):
        assert mp.isfinite(coarse) and mp.isfinite(fine)
        assert abs(coarse - fine) < mp.mpf("1e-30")
    assert math.isfinite(value) and abs(value - float(fine)) < atol


@pytest.mark.parametrize("density", ("normal", "gh", "variance_gamma"))
def test_historical_conditioned_aparch_small_units_all_masks(native, density):
    unit = historical_aparch_small_unit_case(density)
    scaled = historical_aparch_small_unit_case(density, scaled=True)
    np.testing.assert_array_equal(scaled["y"], unit["y"] * 1e-5)
    np.testing.assert_array_equal(unit["scored_rows"], (1, 2, 3, 4, 5))
    np.testing.assert_array_equal(scaled["scored_rows"], unit["scored_rows"])
    assert scaled["recursion_begin"] == unit["recursion_begin"] == 1
    assert scaled["h0"] == float(np.mean(scaled["y"] * scaled["y"]))
    unit_expected, scaled_expected = full_jet(unit), full_jet(scaled)
    unit_actual = exercise_requests(native, unit, unit_expected)
    scaled_actual = exercise_requests(native, scaled, scaled_expected, value_atol=1e-9)
    np.testing.assert_allclose(scaled_actual["f"] - unit_actual["f"], 5 * np.log(1e-5),
                               rtol=0, atol=2e-12)
    for field, reference in zip(("f", "g", "h"), scaled_expected):
        assert np.isfinite(scaled_actual[field]).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(scaled_actual[field], reference, rtol=4e-6, atol=4e-8)

    jax, jnp, _ = jax_modules()
    point = jnp.asarray(unit["parameters"])

    def change_units(theta):
        return theta.at[0].set(theta[0] * 1e-5).at[3].set(theta[3] * jnp.exp(theta[7] * jnp.log(1e-5)))

    jacobian = np.asarray(jax.jacfwd(change_units)(point))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(change_units))(point))
    np.testing.assert_allclose(change_units(point), scaled["parameters"], rtol=4e-15, atol=0)
    assert curvature[3, 3, 7] != 0 and curvature[3, 7, 7] != 0
    for result, reference in ((scaled_actual, unit_actual),
                              (dict(zip(("f", "g", "h"), scaled_expected)),
                               dict(zip(("f", "g", "h"), unit_expected)))):
        gradient = jacobian.T @ result["g"]
        hessian = jacobian.T @ result["h"] @ jacobian + np.einsum("i,ijk->jk", result["g"], curvature)
        np.testing.assert_allclose(gradient, reference["g"], rtol=4e-6, atol=4e-8)
        np.testing.assert_allclose(hessian, reference["h"], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_joint_theta_all_masks_and_every_crossblock_match_independent_ad(native, declared):
    exercise_requests(native, declared, full_jet(declared))


@pytest.mark.parametrize("declared", original_z_cases(), ids=lambda item: item["case_id"])
def test_joint_original_z_all_masks_include_mean_and_power_map_curvature(native, declared):
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    _, jnp, _ = jax_modules()
    mapped = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-9, atol=3e-10)
    expected = original_z_jet(declared, actual["theta"])
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
@pytest.mark.parametrize("transformed", (False, True))
def test_shipped_joint_power_fixed_candidates_and_runtime_general_orders(native, declared, transformed):
    declared = chart_case(declared) if transformed else declared
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert actual["dispatch"] == expected_dispatch(declared), declared["case_id"]


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_additive_constants_selected_sample_and_mp_refinement(native, model, density):
    declared = case(model, density, p_ar=2 if model.startswith("arma_") else 0,
                    q_ma=3 if model.startswith("arma_") else 0,
                    p_vol=3, q_vol=2, columns=5, intercept=False, first=8, stop=11, hold_back=2)
    reference = float(mp_value(declared))
    assert_mp_value(declared, reference)
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert abs(actual["f"] - reference) < 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_chart_has_independent_mp_mean_reflections_and_power_moments(native, model, density):
    declared = chart_case(case(model, density, p_ar=2 if model.startswith("arma_") else 0,
                               q_ma=3 if model.startswith("arma_") else 0,
                               p_vol=3, q_vol=2, columns=5, intercept=False))
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    mapped = np.asarray([float(value) for value in mp_coordinate_map(declared)])
    np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-9, atol=3e-10)
    assert_mp_value(declared, actual["f"], theta=actual["theta"])


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda item: item[0])
def test_mean_state_against_independent_scipy_filter_or_design_identity(layout):
    declared = layout_case(layout, "aparch", "normal")
    states, gradient, hessian = state_jet(declared)
    offset, begin, width = declared["hold_back"], declared["recursion_begin"], mean_width(declared)
    theta, y = declared["parameters"], declared["y"][offset:declared["stop"]]
    intercept = int(declared["intercept"])
    if declared["model"].startswith("arma_"):
        ar = theta[intercept:intercept + declared["p_ar"]]
        ma = theta[intercept + declared["p_ar"]:width]
        forcing = np.array([y[t] - (theta[0] if intercept else 0.0)
                            - math.fsum(float(a) * float(y[t - lag - 1]) for lag, a in enumerate(ar))
                            for t in range(begin, len(y))])
        expected = np.r_[np.zeros(begin), lfilter([1.0], np.r_[1.0, ma], forcing)]
        first_gradient = np.r_[[-1.0] if intercept else [],
                               [-y[begin - lag - 1] for lag in range(declared["p_ar"])],
                               np.zeros(declared["q_ma"])]
        np.testing.assert_allclose(gradient[begin, 0, :width], first_gradient, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(hessian[begin, 0], np.zeros_like(hessian[begin, 0]))
        if declared["q_ma"] and begin + 1 < len(y):
            ma_index = intercept + declared["p_ar"]
            assert gradient[begin + 1, 0, ma_index] == pytest.approx(-expected[begin], rel=3e-14, abs=3e-14)
            np.testing.assert_allclose(hessian[begin + 1, 0, :ma_index, ma_index],
                                       -first_gradient[:ma_index], rtol=0.0, atol=0.0)
    else:
        features = declared["design"][offset:declared["stop"]]
        expected = y - (theta[0] if intercept else 0.0) - features @ theta[intercept:width]
        expected_gradient = np.column_stack([np.full(len(y), -1.0), -features]) if intercept else -features
        np.testing.assert_allclose(gradient[:, 0, :width], expected_gradient, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(hessian[:, 0], np.zeros_like(hessian[:, 0]))
    np.testing.assert_allclose(states[:, 0], expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_allclose(states[:, 0], numpy_residuals(mean_case(declared), theta), rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(gradient[:, 0, width:], np.zeros_like(gradient[:, 0, width:]))
    np.testing.assert_array_equal(hessian[:, 0, width:], np.zeros_like(hessian[:, 0, width:]))


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", ("normal", "studentt", "skewt", "nig"))
def test_declared_first_variance_distinguishes_arma_recurrence_and_linear_seed(model, density):
    declared = case(model, density, p_ar=2 if model.startswith("arma_") else 0,
                    q_ma=3 if model.startswith("arma_") else 0,
                    p_vol=3, q_vol=2, columns=5, intercept=False, h0=2.7)
    states, gradient, hessian = state_jet(declared)
    begin, width, shape = declared["recursion_begin"], mean_width(declared), shape_start(declared)
    with mp.workdps(70):
        residuals, log_h = mp_states(declared, declared["parameters"])
        independent = np.column_stack([list(map(float, residuals)), list(map(float, log_h))])
    np.testing.assert_allclose(states, independent, rtol=3e-13, atol=3e-13)
    if model.startswith("linear_"):
        assert states[0, 1] == np.log(declared["h0"])
        np.testing.assert_array_equal(gradient[0, 1], np.zeros_like(gradient[0, 1]))
        np.testing.assert_array_equal(hessian[0, 1], np.zeros_like(hessian[0, 1]))
    else:
        np.testing.assert_array_equal(states[:begin, 0], np.zeros(begin))
        np.testing.assert_array_equal(gradient[:begin], np.zeros_like(gradient[:begin]))
        np.testing.assert_array_equal(hessian[:begin], np.zeros_like(hessian[:begin]))
        np.testing.assert_array_equal(gradient[begin, 1, :width], np.zeros(width))
        np.testing.assert_array_equal(hessian[begin, 1, :width], np.zeros_like(hessian[begin, 1, :width]))
        with mp.workdps(70):
            point = tuple(mp.mpf(float(v)) for v in declared["parameters"])
            p, q = declared["p_vol"], declared["q_vol"]
            beta = mp.fsum(point[width + 1 + 2 * p:width + 1 + 2 * p + q])
            log_seed = mp.log(mp.mpf(float(declared["h0"])))
            if model.endswith("egarch"):
                moment = mp_power_moment(0, 1, point[shape:], density)
                expected = point[width] - moment * mp.fsum(point[width + 1:width + 1 + p]) + beta * log_seed
                assert gradient[begin, 1, width] == 1.0
            else:
                delta = point[shape - 1]
                seed, k = mp.exp(delta * log_seed / 2), log_seed / 2
                power = point[width] + beta * seed
                expected = 2 * mp.log(power) / delta
                d, dd = beta * seed * k, beta * seed * k * k
                delta_g = -2 * mp.log(power) / delta ** 2 + 2 * d / (delta * power)
                delta_h = (4 * mp.log(power) / delta ** 3 - 4 * d / (delta ** 2 * power)
                           + 2 * (dd / power - (d / power) ** 2) / delta)
                np.testing.assert_allclose([gradient[begin, 1, shape - 1], hessian[begin, 1, shape - 1, shape - 1]],
                                           [float(delta_g), float(delta_h)], rtol=3e-12, atol=3e-12)
        assert states[begin, 1] == pytest.approx(float(expected), rel=3e-14, abs=3e-14)
        assert abs(states[begin, 1] - np.log(declared["h0"])) > 1e-3


@pytest.mark.parametrize("model", ("linear_egarch", "linear_aparch"))
@pytest.mark.parametrize("density", DENSITIES)
def test_linear_first_likelihood_row_keeps_seed_fixed_but_mean_active(native, model, density):
    declared = case(model, density, p_vol=4, q_vol=3, columns=3, y=[0.37], h0=2.7)
    expected = full_jet(declared)
    actual = exercise_requests(native, declared, expected)
    mean, shape = mean_width(declared), shape_start(declared)
    assert np.max(np.abs(actual["g"][:mean])) > 1e-5
    np.testing.assert_array_equal(actual["g"][mean:shape], np.zeros(shape - mean))
    np.testing.assert_array_equal(actual["h"][mean:shape], np.zeros_like(actual["h"][mean:shape]))
    assert_mp_value(declared, actual["f"])


@pytest.mark.parametrize("model", ("arma_egarch", "arma_aparch"))
@pytest.mark.parametrize("density", DENSITIES)
def test_first_arma_likelihood_row_scores_the_complete_conditioned_recurrence(native, model, density):
    declared = case(model, density, p_ar=0, q_ma=2, p_vol=3, q_vol=2, intercept=False, h0=2.7)
    first = declared["recursion_begin"]
    declared = scored_suffix(declared, first, first + 1)
    expected = full_jet(declared)
    actual = exercise_requests(native, declared, expected)
    assert abs(actual["g"][mean_width(declared)]) > 1e-5
    assert_mp_value(declared, actual["f"])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_warmup_mean_variance_sensitivities_holdback_and_stop_are_not_scoring_resets(native, model, density):
    declared = case(model, density, p_ar=2 if model.startswith("arma_") else 0,
                    q_ma=3 if model.startswith("arma_") else 0,
                    p_vol=3, q_vol=2, columns=5, intercept=False, first=8, stop=11, hold_back=2)
    declared["rows"] = np.arange(len(declared["y"]), dtype=np.uint64) * 7 + 101
    declared["scored_rows"] = tuple(declared["rows"][8:11])
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    outside = deepcopy(declared)
    outside["y"][:2] *= 100.0
    outside["y"][11:] *= 100.0
    outside["design"][:2] *= 100.0
    outside["design"][11:] *= 100.0
    exercise_requests(native, outside, expected)
    warmup = deepcopy(declared)
    warmup["y"][6] *= -1.7
    changed = full_jet(warmup)
    assert abs(changed[0] - expected[0]) > 1e-6
    assert np.max(np.abs(changed[1] - expected[1])) > 1e-5
    exercise_requests(native, warmup, changed)
    assert_mp_value(warmup, changed[0])


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", ("normal", "skewt", "nig"))
def test_original_z_direct_composition_equals_full_pullback_and_has_nonzero_map_curvature(native, model, density):
    declared = chart_case(case(model, density, p_ar=2 if model.startswith("arma_") else 0,
                               q_ma=3 if model.startswith("arma_") else 0,
                               p_vol=3, q_vol=2, columns=5, intercept=False))
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta = full_jet(declared, theta=actual["theta"])
    expected = original_z_jet(declared, actual["theta"])
    explicit = pullback(declared, theta)
    assert_requested_close(dict(zip(("f", "g", "h"), expected)), explicit, 7, density)
    assert_cross_blocks(actual, expected, declared)
    assert_requested_close(actual, expected, 7, density)
    jax, jnp, _ = jax_modules()
    jacobian = np.asarray(jax.jacfwd(coordinate_map(declared))(jnp.asarray(declared["original_z"])))
    assert np.max(np.abs(expected[2] - jacobian.T @ theta[2] @ jacobian)) > 1e-4


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("axis,coordinate", (("rho", 18.0), ("zeta", 50.0)))
def test_joint_shape_endpoint_keeps_original_z_curvature_and_all_crossblock_rows(native, model, axis, coordinate):
    declared = chart_case(case(model, "nig", p_ar=0, q_ma=1 if model.startswith("arma_") else 0,
                               p_vol=3, q_vol=2, columns=5, intercept=False))
    index = shape_start(declared) + int(axis == "zeta")
    declared["parameters"][index] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta = full_jet(declared, theta=actual["theta"], moment_count=256)
    expected = pullback(declared, theta, moment_count=256)
    exercise_requests(native, declared, expected)
    scale_g, scale_h = abs(expected[1][index]), np.max(np.abs(expected[2][index]))
    assert scale_g > 0.0 and scale_h > 0.0
    np.testing.assert_allclose(actual["g"][index] / scale_g, expected[1][index] / scale_g, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(actual["h"][index] / scale_h, expected[2][index] / scale_h, rtol=8e-6, atol=8e-6)


@pytest.mark.parametrize("volatility", ("egarch", "aparch"))
@pytest.mark.parametrize("coordinate", (-18.0, 18.0))
def test_joint_mean_reflection_tail_keeps_original_z_mean_volatility_curvature(native, volatility, coordinate):
    declared = chart_case(case(f"arma_{volatility}", p_ar=0, q_ma=1, p_vol=3, q_vol=2, intercept=False))
    declared["parameters"][0] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta)
    exercise_requests(native, declared, expected)
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        slope = float(-(1 - mp.tanh(z) ** 2))
    assert slope != 0.0
    np.testing.assert_allclose(actual["g"][0] / slope, expected[1][0] / slope, rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(actual["h"][0] / abs(slope), expected[2][0] / abs(slope), rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_representative_likelihood_contains_mean_volatility_and_shape_crossblocks(native, model, density):
    declared = case(model, density, p_ar=0,
                    q_ma=2 if model.startswith("arma_") else 0,
                    p_vol=3, q_vol=2, columns=5, intercept=False)
    expected = full_jet(declared)
    mean, shape = mean_width(declared), shape_start(declared)
    assert np.max(np.abs(expected[2][:mean, mean:shape])) > 1e-6
    if SHAPES[density]:
        assert np.max(np.abs(expected[2][:mean, shape:])) > 1e-6
        assert np.max(np.abs(expected[2][mean:shape, shape:])) > 1e-6
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    assert_cross_blocks(actual, expected, declared)


@pytest.mark.parametrize("model", ("linear_egarch", "linear_aparch"))
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("empty_mean", (False, True))
def test_structural_zero_rows_have_exact_derivatives_not_active_residual_kinks(native, model, density, empty_mean):
    declared = fixed_zero_row_case(model, density, empty_mean=empty_mean)
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert_mp_value(declared, expected[0])


@pytest.mark.parametrize("model", ("linear_egarch", "linear_aparch"))
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_empty_linear_mean_reduces_to_zero_mean_power_same_conditioning(native, model, density, p, q):
    declared = case(model, density, p_vol=p, q_vol=q, columns=0, intercept=False,
                    first=4, stop=11, hold_back=2, h0=2.7)
    zero = zero_power_case(model.split("_", 1)[1], density, p, q, parameters=declared["parameters"],
                           y=declared["y"], first=4, stop=11, hold_back=2, h0=2.7)
    expected = zero_power_jet(zero)
    exercise_requests(native, declared, expected)
    with mp.workdps(70):
        assert abs(mp_value(declared) - zero_power_mp_value(zero)) < mp.mpf("1e-40")
    for mask in REQUESTS:
        left, right = native(declared, mask), native(zero, mask)
        assert_success(left, declared, mask)
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                assert right["status"] == 0 and right["valid_mask"] == mask
                np.testing.assert_allclose(left[name], right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("mean", ("arma", "linear"))
@pytest.mark.parametrize("density", DENSITIES)
def test_joint_aparch_delta_two_gamma_zero_reduces_to_joint_garch_all_common_blocks(native, mean, density):
    declared = case(f"{mean}_aparch", density, p_ar=0,
                    q_ma=2 if mean == "arma" else 0, p_vol=3, q_vol=2, columns=5,
                    intercept=False, first=7, stop=11, hold_back=2, h0=2.7)
    width, p, q, shape = mean_width(declared), declared["p_vol"], declared["q_vol"], shape_start(declared)
    declared["parameters"][width + 1 + p:width + 1 + 2 * p] = 0.0
    declared["parameters"][shape - 1] = 2.0
    common = np.r_[np.arange(width + 1 + p), np.arange(width + 1 + 2 * p, width + 1 + 2 * p + q),
                   np.arange(shape, len(declared["parameters"]))]
    garch = joint_garch_case(f"{mean}_garch", density, p_ar=declared["p_ar"], q_ma=declared["q_ma"],
                             p_vol=p, q_vol=q, columns=5, intercept=False, design=declared["design"],
                             parameters=declared["parameters"][common], y=declared["y"],
                             first=7, stop=11, hold_back=2, h0=2.7)
    reference = joint_garch_jet(garch)
    with mp.workdps(70):
        assert abs(mp_value(declared) - joint_garch_mp_value(garch, digits=70)) < mp.mpf("1e-40")
    exercise_requests(native, declared, full_jet(declared))
    for mask in REQUESTS:
        left, right = native(declared, mask), native(garch, mask)
        assert_success(left, declared, mask)
        assert right["status"] == 0 and right["valid_mask"] == mask
        projected = dict(f=left["f"], g=left["g"][common], h=left["h"][np.ix_(common, common)])
        assert_requested_close(projected, reference, mask, density)
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                np.testing.assert_allclose(projected[name], right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("model", FAMILIES)
def test_joint_skewt_zero_skew_reduces_to_studentt_with_mean_crossblocks(native, model):
    student = case(model, "studentt", p_ar=0,
                    q_ma=2 if model.startswith("arma_") else 0,
                    p_vol=3, q_vol=2, columns=5, intercept=False)
    skew = deepcopy(student)
    skew.update(density="skewt", parameters=np.r_[student["parameters"], 0.0],
                theta_names=student["theta_names"] + ("lambda",), case_id=student["case_id"] + "-zero-skew")
    expected = full_jet(student)
    with mp.workdps(70):
        assert abs(mp_value(student) - mp_value(skew)) < mp.mpf("1e-40")
    for mask in REQUESTS:
        left, right = native(skew, mask), native(student, mask)
        assert_success(left, skew, mask)
        assert_success(right, student, mask)
        common = dict(f=left["f"], g=left["g"][:-1], h=left["h"][:-1, :-1])
        assert_requested_close(common, expected, mask, "studentt")
        for bit, name in zip((1, 2, 4), ("f", "g", "h")):
            if mask & bit:
                np.testing.assert_allclose(common[name], right[name], rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("volatility", ("egarch", "aparch"))
def test_unrestricted_linear_coefficients_do_not_inherit_arma_stability_constraints(native, volatility):
    declared = case(f"linear_{volatility}", columns=5, intercept=False, p_vol=3, q_vol=2)
    declared["parameters"][:5] = [1.7, -2.1, 0.3, -0.4, 1.2]
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert_mp_value(declared, expected[0])


@pytest.mark.parametrize("model", FAMILIES)
def test_long_joint_mean_power_sensitivities_are_large_and_match_ad(native, model):
    declared = long_horizon_case(model)
    expected = full_jet(declared)
    ordinary = full_jet(case(model, columns=1, intercept=False))
    assert np.max(np.abs(expected[1])) > 50.0 * np.max(np.abs(ordinary[1]))
    assert np.max(np.abs(expected[2])) > 50.0 * np.max(np.abs(ordinary[2]))
    exercise_requests(native, declared, expected, value_atol=5e-8)
    assert_mp_value(declared, expected[0], atol=5e-8)


@pytest.mark.parametrize("model,delta,request_mask", (("linear_egarch", None, 2),
                                               ("linear_egarch", None, 4),
                                               ("linear_aparch", 0.7, 2),
                                               ("linear_aparch", 1.7, 4)))
def test_oracle_declines_active_recursion_kinks_without_smoothing(model, delta, request_mask):
    declared = case(model, columns=0, intercept=True, y=[0.0, 0.0, 0.0])
    declared["parameters"][0] = 0.0
    if delta is not None:
        declared["parameters"][shape_start(declared) - 1] = delta
    assert_mp_value(declared, float(jet(declared, request=1)[0]))
    with pytest.raises(ValueError, match="Active .*zero"):
        theta_objective(declared, declared["parameters"], request=request_mask)


def test_active_aparch_delta_two_gamma_zero_has_exact_mean_curvature_and_garch_reduction(native):
    declared = case("linear_aparch", columns=0, intercept=True, y=np.zeros(4), h0=2.7)
    declared["parameters"][0] = 0.0
    declared["parameters"][3] = 0.0
    declared["parameters"][5] = 2.0
    garch = joint_garch_case("linear_garch", columns=0, intercept=True, y=np.zeros(4), h0=2.7,
                             parameters=declared["parameters"][[0, 1, 2, 4]])
    expected = full_jet(declared)
    common = [0, 1, 2, 4]
    reference = joint_garch_jet(garch)
    assert_requested_close(dict(f=expected[0], g=expected[1][common], h=expected[2][np.ix_(common, common)]),
                           reference, 7, "normal")
    assert expected[2][0, 0] > 0.0
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", ("linear_egarch", "linear_aparch"))
def test_active_ged_shape_two_centre_uses_exact_density_curvature_at_fixed_seed(native, model):
    # No later row lets an EGARCH/APARCH zero shock create a recursion kink.
    declared = case(model, "ged", columns=0, intercept=True, y=[0.0], h0=2.7)
    declared["parameters"][0] = 0.0
    declared["parameters"][-1] = 2.0
    expected = full_jet(declared)
    assert expected[2][0, 0] == pytest.approx(1.0 / declared["h0"], rel=3e-15)
    exercise_requests(native, declared, expected)
