"""Mandatory, legacy-free GARCH/GJR mathematical conformance.

Collection performs no JAX work. Missing JAX, MP, SciPy or native bridge support
fails the requested test, never removes a required case through a skip.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
import pytest

from fixtures.garch_cases import (
    DENSITIES, FAMILIES, REQUESTS, SHAPES, case, chart_case, long_horizon_case,
    ordinary_cases, rejection_cases, tiny_scale_case, with_parameters,
)
from oracles.charts import bounded_logistic, coordinate_map, negative_second_moment, pullback
from oracles.garch import (
    full_jet, jax_modules, local_jet, log_bessel_k, scalar_jet, state_jet,
    volatility_width,
)
from oracles.values import (
    density_moments, mp_logpdf, mp_value, numpy_variances, scipy_logpdf, scipy_value,
)


LIGHT = ("normal", "studentt", "skewt", "ged")
GH_FAMILY = ("gh", "nig", "hyperbolic", "variance_gamma")


def assert_jet_close(actual, expected, mask, density, *, value_atol=None):
    light = density in LIGHT
    value_atol = value_atol if value_atol is not None else (2e-9 if light else 3e-8)
    for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
        if not mask & bit:
            continue
        output = np.asarray(actual[name])
        assert np.isfinite(output).all() and np.isfinite(reference).all(), name
        rtol = 0.0 if bit == 1 else ((3e-8 if bit == 2 else 3e-7) if light
                                    else (3e-7 if bit == 2 else 4e-6))
        atol = value_atol if bit == 1 else ((3e-8 if bit == 2 else 3e-7) if light
                                          else (3e-7 if bit == 2 else 4e-6))
        np.testing.assert_allclose(output, reference, rtol=rtol, atol=atol,
                                   err_msg=f"{density} {name} request={mask}")


def assert_success(actual, declared, mask):
    assert actual["status"] == 0, actual.get("reasons")
    assert actual["valid_mask"] == mask
    np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
    width = len(declared["theta_names"])
    assert np.asarray(actual["g"]).shape == (width,)
    assert np.asarray(actual["h"]).shape == (width, width)
    for bit, name in ((1, "f"), (2, "g"), (4, "h")):
        if not mask & bit:
            expected = np.full_like(np.asarray(actual[name]), actual["sentinel"])
            np.testing.assert_array_equal(np.asarray(actual[name]).view(np.uint64),
                                          expected.view(np.uint64))


def exercise_requests(native, declared, expected, *, value_atol=None):
    full = native(declared, 7)
    assert_success(full, declared, 7)
    assert_jet_close(full, expected, 7, declared["density"], value_atol=value_atol)
    if not declared["transformed"]:
        np.testing.assert_array_equal(full["theta"], declared["parameters"])
    for mask in REQUESTS:
        actual = full if mask == 7 else native(declared, mask)
        assert_success(actual, declared, mask)
        assert_jet_close(actual, expected, mask, declared["density"], value_atol=value_atol)
        projection = tuple(full[name] for name in ("f", "g", "h"))
        assert_jet_close(actual, projection, mask, declared["density"], value_atol=2e-10)
        np.testing.assert_array_equal(actual["theta"], full["theta"])
    return full


def reduction_roundoff_bound(native, left, right, common_right):
    """Bound reduction roundoff using sample terms, not a statistical tolerance.

    Two evaluations receive gamma_k rounding budgets, k=8*n+64. The allowance
    covers local arithmetic and distributed accumulation. Absolute contribution
    magnitudes protect against cancellation. Native row jets size G/H budgets
    only; they are neither AD expectations nor alternative statistical models.
    """
    masses = []
    for declared, common in ((left, np.arange(len(left["parameters"]))), (right, common_right)):
        n_vol = volatility_width(declared)
        rows = np.asarray(declared["scored_rows"], dtype=int)
        h = numpy_variances(declared)[rows - declared["hold_back"]]
        shape = declared["parameters"][n_vol:]
        logpdf = scipy_logpdf(declared["y"][rows] / np.sqrt(h), declared["density"], shape)
        base = float(scipy_logpdf(0.0, declared["density"], shape))
        value_mass = math.fsum(map(float, np.abs(np.log(h) / 2.0) + abs(base) + np.abs(logpdf - base)))
        derivative_mass = np.zeros(2)
        for row in rows:
            singleton = deepcopy(declared)
            singleton.update(first=int(row), stop=int(row + 1), scored_rows=(int(row),))
            contribution = native(singleton, 7)
            assert_success(contribution, singleton, 7)
            g = contribution["g"][common]
            hessian = contribution["h"][np.ix_(common, common)]
            assert np.isfinite(g).all() and np.isfinite(hessian).all()
            derivative_mass += [np.max(np.abs(g)), np.max(np.abs(hessian))]
        masses.append(np.r_[value_mass, derivative_mass])
    count = 8 * len(left["scored_rows"]) + 64
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma = count * unit_roundoff / (1.0 - count * unit_roundoff)
    return 2.0 * gamma / (1.0 - gamma) * np.maximum(1.0, np.maximum(*masses))


def assert_reduction_roundoff(actual, expected, mask, bound):
    for index, (bit, name, reference) in enumerate(zip((1, 2, 4), ("f", "g", "h"), expected)):
        if mask & bit:
            assert np.isfinite(actual[name]).all() and np.isfinite(reference).all()
            difference = float(np.max(np.abs(np.asarray(actual[name]) - reference)))
            assert difference <= bound[index], (name, difference, bound[index], mask)


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_theta_all_masks_match_independent_ad(native, declared):
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("declared", ordinary_cases(), ids=lambda item: item["case_id"])
def test_shipped_common_order_dispatch_contract(native, declared):
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    expected = 1 if declared["p"] <= 2 and declared["q"] <= 2 else 0
    assert actual["dispatch"] == expected, (
        declared["case_id"], "missing common fixed specialization" if expected else "unexpected runtime route",
        actual["dispatch"], expected,
    )


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_independent_scalar_values_include_all_constants(native, model, density):
    declared = case(model, density, p=3, q=2)
    coarse, fine = mp_value(declared, digits=60), mp_value(declared, digits=90)
    with mp.workdps(90):
        assert abs(coarse - fine) < mp.mpf("1e-50")
    expected = float(fine)
    assert abs(scipy_value(declared) - expected) <= 3e-10
    for mask in (1, 3, 5, 7):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        assert math.isfinite(actual["f"])
        assert abs(actual["f"] - expected) <= 3e-9


@pytest.mark.parametrize("density", DENSITIES)
def test_oracle_density_normalization_and_standardization(density):
    moments, errors = density_moments(density, SHAPES[density])
    assert np.isfinite(moments).all() and np.isfinite(errors).all()
    assert max(errors) < 2e-8
    np.testing.assert_allclose(moments, [1.0, 0.0, 1.0], rtol=0.0, atol=3e-8)


@pytest.mark.parametrize("density", DENSITIES)
def test_local_density_derivatives_via_one_seeded_observation(native, density):
    # At n=1 the variance is fixed. Vary only shape axes and compare their local AD jet.
    declared = case(density=density, y=[0.37], h0=1.3)
    n_vol = volatility_width(declared)
    expected = local_jet(density, np.r_[0.37, 1.3, SHAPES[density]])
    full = native(declared, 7)
    assert_success(full, declared, 7)
    assert abs(full["f"] - expected[0]) <= 3e-9
    np.testing.assert_array_equal(full["g"][:n_vol], np.zeros(n_vol))
    np.testing.assert_array_equal(full["h"][:n_vol], np.zeros((n_vol, len(full["g"]))))
    if SHAPES[density]:
        local = {"f": full["f"], "g": full["g"][n_vol:], "h": full["h"][n_vol:, n_vol:]}
        assert_jet_close(local, (expected[0], expected[1][2:], expected[2][2:, 2:]), 7, density)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_oracle_recursive_state_has_exact_seed_and_available_lags(model, p, q):
    declared = case(model, p=p, q=q)
    values, gradients, hessians = state_jet(declared)
    np.testing.assert_allclose(values, numpy_variances(declared), rtol=3e-15, atol=3e-15)
    assert values[0] == declared["h0"]
    np.testing.assert_array_equal(gradients[0], np.zeros_like(gradients[0]))
    np.testing.assert_array_equal(hessians[0], np.zeros_like(hessians[0]))
    e0, theta = declared["y"][0], declared["parameters"]
    beta_start = 1 + (2 if model == "gjr" else 1) * p
    expected = theta[0] + theta[1] * e0 * e0 + theta[beta_start] * declared["h0"]
    if model == "gjr" and e0 < 0:
        expected += theta[1 + p] * e0 * e0
    assert values[1] == pytest.approx(expected, rel=3e-15, abs=3e-15)
    active = [0, 1, beta_start]
    inactive = sorted(set(range(len(theta))) - set(active))
    np.testing.assert_array_equal(gradients[1, inactive], np.zeros(len(inactive)))
    np.testing.assert_array_equal(hessians[1], np.zeros_like(hessians[1]))


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("first,stop,hold_back", ((0, 1, 0), (3, 7, 0), (5, 8, 2), (2, 3, 2)))
def test_declared_scored_suffix_and_physical_holdback(native, model, density, first, stop, hold_back):
    declared = case(model, density, p=3, q=2, first=first, stop=stop, hold_back=hold_back)
    expected = full_jet(declared)
    exercise_requests(native, declared, expected)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_unscored_prefix_still_advances_state(native, model, density):
    declared = case(model, density, first=4)
    changed = deepcopy(declared)
    changed["y"][1] *= 3.0
    expected, other = full_jet(declared), full_jet(changed)
    assert abs(expected[0] - other[0]) > 1e-4
    exercise_requests(native, declared, expected)
    exercise_requests(native, changed, other)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_original_z_all_masks_include_map_curvature(native, model, density, p, q):
    declared = chart_case(model, density, p, q)
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    _, jnp, _ = jax_modules()
    mapped = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-8, atol=3e-10)
    # Native derivatives never enter the expected jet. Only actual map doubles anchor F.
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    exercise_requests(native, declared, expected)
    if model == "gjr" and density not in ("normal", "studentt", "ged"):
        refined = pullback(declared, theta_jet, moment_count=64)
        refinement = {"f": refined[0], "g": refined[1], "h": refined[2]}
        assert_jet_close(refinement, expected, 7, density, value_atol=0.0)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("at_join", (False, True))
def test_hansen_zero_skew_exactly_reduces_to_student_t(native, model, at_join):
    y = [0.0, 0.0, 0.37, -0.81] if at_join else None
    student = case(model, "studentt", **({"y": y} if y is not None else {}))
    skew = case(model, "skewt", parameters=np.r_[student["parameters"], 0.0], y=student["y"])
    bound = reduction_roundoff_bound(native, student, skew, np.arange(len(student["parameters"])))
    for mask in REQUESTS:
        a, b = native(student, mask), native(skew, mask)
        assert_success(a, student, mask)
        assert_success(b, skew, mask)
        common = (a["f"], a["g"], a["h"])
        actual = {"f": b["f"], "g": b["g"][:-1], "h": b["h"][:-1, :-1]}
        assert_reduction_roundoff(actual, common, mask, bound)


@pytest.mark.parametrize("model", FAMILIES)
def test_ged_shape_two_reduces_to_normal_common_parameters(native, model):
    normal = case(model)
    ged = case(model, "ged", parameters=np.r_[normal["parameters"], 2.0])
    for mask in REQUESTS:
        a, b = native(normal, mask), native(ged, mask)
        assert_success(a, normal, mask)
        assert_success(b, ged, mask)
        assert_jet_close({"f": b["f"], "g": b["g"][:-1], "h": b["h"][:-1, :-1]},
                         (a["f"], a["g"], a["h"]), mask, "normal", value_atol=2e-12)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density,lam", (("nig", -0.5), ("hyperbolic", 1.0)))
def test_fixed_lambda_gh_reductions_are_exact(native, model, density, lam):
    reduced = case(model, density)
    gh = case(model, "gh", parameters=np.r_[reduced["parameters"], lam])
    bound = reduction_roundoff_bound(native, reduced, gh, np.arange(len(reduced["parameters"])))
    for mask in REQUESTS:
        a, b = native(reduced, mask), native(gh, mask)
        assert_success(a, reduced, mask)
        assert_success(b, gh, mask)
        assert_reduction_roundoff({"f": b["f"], "g": b["g"][:-1], "h": b["h"][:-1, :-1]},
                                  (a["f"], a["g"], a["h"]), mask, bound)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_zero_gamma_gjr_reduces_to_garch(native, density, p, q):
    garch = case("garch", density, p, q)
    theta = garch["parameters"]
    gjr = case("gjr", density, p, q, parameters=np.r_[theta[:1 + p], np.zeros(p), theta[1 + p:]])
    common = np.r_[np.arange(1 + p), np.arange(1 + 2 * p, len(gjr["parameters"]))]
    for mask in REQUESTS:
        a, b = native(garch, mask), native(gjr, mask)
        assert_success(a, garch, mask)
        assert_success(b, gjr, mask)
        assert_jet_close({"f": b["f"], "g": b["g"][common], "h": b["h"][np.ix_(common, common)]},
                         (a["f"], a["g"], a["h"]), mask, density, value_atol=2e-10)


@pytest.mark.parametrize("declared", rejection_cases(), ids=lambda item: item["case_id"])
def test_theoretical_domain_rejections_have_no_valid_outputs(native, declared):
    for mask in REQUESTS:
        actual = native(declared, mask)
        assert actual["status"] != 0
        assert actual["valid_mask"] == 0


@pytest.mark.parametrize("model", FAMILIES)
def test_second_order_overflow_does_not_manufacture_a_valid_hessian(native, model):
    theta = [1e-200, 0.0] + ([0.0] if model == "gjr" else []) + [0.0]
    declared = case(model, parameters=theta, y=[0.0, 0.0], h0=1.0)
    value = float(mp_value(declared, digits=250))
    for mask in (1, 2, 3):
        actual = native(declared, mask)
        assert_success(actual, declared, mask)
        if mask & 1:
            assert abs(actual["f"] - value) <= 3e-9
        if mask & 2:
            assert np.isfinite(actual["g"]).all()
    for mask in (4, 5, 6, 7):
        actual = native(declared, mask)
        assert actual["status"] != 0
        assert not actual["valid_mask"] & 4
        assert not actual["valid_mask"] & ~mask


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density,shape", (("studentt", 2.01), ("ged", 0.08)))
def test_shape_domain_is_not_narrowed_by_legacy_padding(native, model, density, shape):
    template = case(model, density)
    theta = template["parameters"].copy()
    theta[-1] = shape
    declared = with_parameters(template, theta)
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density,shape", (
    ("gh", (0.99, 0.1, -6.0)), ("gh", (-0.99, 25.0, 6.0)),
    ("nig", (0.99, 0.1)), ("hyperbolic", (-0.99, 25.0)),
    ("variance_gamma", (0.95, 20.0)),
))
def test_gh_family_support_edge_values_match_high_precision(native, model, density, shape):
    declared = case(model, density)
    theta = declared["parameters"].copy()
    theta[-len(shape):] = shape
    declared = with_parameters(declared, theta)
    low, high = mp_value(declared, digits=70), mp_value(declared, digits=100)
    with mp.workdps(100):
        assert abs(low - high) < mp.mpf("1e-55")
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert math.isfinite(actual["f"])
    assert abs(actual["f"] - float(high)) <= 3e-9


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", DENSITIES)
def test_small_variance_is_not_floored(native, model, density):
    declared = tiny_scale_case(model, density)
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("density", LIGHT)
def test_long_horizon_boundary_sensitivities_all_masks(native, model, density):
    declared = long_horizon_case(model, density)
    path, state_gradient, state_hessian = state_jet(declared)
    assert np.isfinite(path).all()
    assert state_gradient[-1, 0] > 500.0
    assert abs(state_hessian[-1, 0, -1 if density == "normal" else volatility_width(declared) - 1]) > 1e5
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 1e-8
    exercise_requests(native, declared, expected, value_atol=1e-8)


@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_cancelling_gjr_shocks_combine_coefficients_before_squaring(native, p, q):
    parameters = np.r_[0.1, 0.7, np.zeros(p - 1), np.nextafter(-0.7, 0.0),
                       np.zeros(p - 1), np.zeros(q)]
    declared = case("gjr", p=p, q=q, parameters=parameters, y=[-1e8, 0.0, 1.0], h0=1e16 / 3)
    expected = full_jet(declared)
    assert abs(expected[0] - float(mp_value(declared))) <= 3e-9
    exercise_requests(native, declared, expected)


def test_finite_gjr_value_survives_individually_overflowing_shock_terms(native):
    declared = case("gjr", parameters=[0.1, 1.9, -1.9, 0.02],
                    y=[-1e154, 0.0, 0.0], h0=1e308 / 3)
    expected = float(mp_value(declared, digits=100))
    actual = native(declared, 1)
    difference = float(actual["f"]) - expected
    evidence = {
        "case": "gjr11-overflowing-cancelling-shock", "request": 1,
        "theta_hex": [float(value).hex() for value in declared["parameters"]],
        "y_hex": [float(value).hex() for value in declared["y"]],
        "h0_hex": float(declared["h0"]).hex(), "mp_digits": 100,
        "actual_f_hex": float(actual["f"]).hex(), "reference_f_hex": expected.hex(),
        "difference_hex": difference.hex(), "absolute_tolerance": 5e-12,
        "absolute_difference": abs(difference) if math.isfinite(difference) else None,
        "status": int(actual["status"]), "valid_mask": int(actual["valid_mask"]),
    }
    print("GJR_OVERFLOW_EVIDENCE " + json.dumps(evidence, sort_keys=True, allow_nan=False))
    assert_success(actual, declared, 1)
    assert math.isfinite(actual["f"])
    assert abs(difference) <= 5e-12, evidence


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_positive_subnormal_seed_value_has_no_padding(native, model, p, q):
    theta = [2.0 ** -1000] + [0.125] + [0.0] * (p - 1)
    if model == "gjr":
        theta += [0.0] * p
    theta += [0.5] + [0.0] * (q - 1)
    declared = case(model, p=p, q=q, parameters=theta, y=np.full(8, 2.0 ** -530), h0=2.0 ** -1060)
    expected = float(mp_value(declared, digits=400))
    actual = native(declared, 1)
    assert_success(actual, declared, 1)
    assert abs(actual["f"] - expected) <= 3e-9
    # Power-of-two residuals have exact subnormal squares. No oracle/native
    # discrepancy can be attributed to an independently rounded square here.
    floored = deepcopy(declared)
    floored["h0"] = 1e-12
    assert abs(expected - float(mp_value(floored))) > 100.0


@pytest.mark.parametrize("density", ("ged", "variance_gamma"))
@pytest.mark.parametrize("model", FAMILIES)
def test_fixed_zero_residual_has_exact_shape_derivatives(native, model, density):
    template = case(model, density, y=[0.0, 0.37, -0.81, 0.0])
    if density == "variance_gamma":
        theta = template["parameters"].copy()
        theta[-2] = 0.0
        template = with_parameters(template, theta)
    expected = full_jet(template)
    exercise_requests(native, template, expected)


@pytest.mark.parametrize("density", GH_FAMILY)
def test_oracle_bessel_quadrature_refines_against_mp_value(density):
    point = np.r_[0.37, 1.3, SHAPES[density]]
    references = [local_jet(density, point, count=count) for count in (128, 256)]
    assert_jet_close(dict(zip(("f", "g", "h"), references[1])), references[0], 7, density,
                     value_atol=1e-10)
    with mp.workdps(90):
        e, h, *shape = (mp.mpf(float(value)) for value in point)
        exact = mp.log(h) / 2 - mp_logpdf(e / mp.sqrt(h), density, shape)
    assert abs(references[-1][0] - float(exact)) <= 1e-10


@pytest.mark.parametrize("order,argument", ((-6.0, 0.1), (0.35, 2.1), (5.5, 25.0)))
def test_oracle_bessel_local_derivatives_have_independent_value_and_refinement(order, argument):
    _, jnp, _ = jax_modules()
    reference = []
    for count in (128, 256):
        reference.append(scalar_jet(lambda point: log_bessel_k(point[0], point[1], count=count),
                                    jnp.array([order, argument])))
    assert_jet_close(dict(zip(("f", "g", "h"), reference[1])), reference[0], 7, "gh", value_atol=2e-10)
    with mp.workdps(80):
        expected = float(mp.log(mp.besselk(mp.mpf(order), mp.mpf(argument))))
    assert abs(reference[-1][0] - expected) < 2e-10


@pytest.mark.parametrize("coordinate", (-700.0, -50.0, -36.0, -18.0, 0.0, 18.0, 36.0, 50.0, 700.0))
def test_oracle_bounded_map_keeps_original_z_tail_curvature(coordinate):
    jax, jnp, _ = jax_modules()
    function = lambda z: bounded_logistic(z, 0.1, 25.0)
    point = jnp.asarray(coordinate)
    actual = [float(function(point)), float(jax.grad(function)(point)), float(jax.grad(jax.grad(function))(point))]
    with mp.workdps(90):
        z, lower, upper = mp.mpf(coordinate), mp.mpf(0.1), mp.mpf(25.0)
        t, width = mp.exp(-abs(z)), upper - lower
        value = upper - width * t / (1 + t) if z >= 0 else lower + width * t / (1 + t)
        slope = width * t / (1 + t) ** 2
        curvature = -mp.sign(z) * width * t * (1 - t) / (1 + t) ** 3
        expected = [float(value), float(slope), float(curvature)]
    np.testing.assert_allclose(actual[:1], expected[:1], rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=5e-14, atol=0.0)


@pytest.mark.parametrize("axis,coordinate", (("rho", -18.0), ("rho", 18.0), ("zeta", -50.0), ("zeta", 50.0)))
def test_original_z_endpoint_pullback_anchors_actual_theta(native, axis, coordinate):
    declared = chart_case("garch", "nig")
    index = volatility_width(declared) + (axis == "zeta")
    declared["parameters"][index] = coordinate
    declared["original_z"] = declared["parameters"].copy()
    actual = native(declared, 7)
    assert_success(actual, declared, 7)
    theta_jet = full_jet(declared, theta=actual["theta"])
    expected = pullback(declared, theta_jet)
    exercise_requests(native, declared, expected)
    with mp.workdps(90):
        z = mp.mpf(coordinate)
        function = ((lambda x: mp.mpf(0.995) * mp.tanh(x)) if axis == "rho"
                    else (lambda x: mp.mpf(0.1) + (mp.mpf(25.0) - mp.mpf(0.1)) / (1 + mp.exp(-x))))
        first, second = float(mp.diff(function, z)), float(mp.diff(function, z, 2))
    assert first > 0.0
    np.testing.assert_allclose(actual["g"][index] / first, theta_jet[1][index], rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(actual["h"][index, index] / first,
                               first * theta_jet[2][index, index] + second / first * theta_jet[1][index],
                               rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("density", ("skewt", "gh", "nig", "hyperbolic", "variance_gamma"))
@pytest.mark.parametrize("at_symmetry", (False, True), ids=("ordinary", "symmetric"))
def test_oracle_gjr_chart_negative_moment_has_quadrature_refinement(density, at_symmetry):
    # Explicit chief-built executable only. No implicit compile or legacy fallback.
    binary = Path(os.environ["SCIVOL_GJR_HALF_IDENTITY_TEST"]).resolve(strict=True) if at_symmetry else None
    jax, jnp, _ = jax_modules()
    shape = np.asarray(SHAPES[density]).copy()
    skew = 1 if density == "skewt" else 0
    if at_symmetry:
        shape[skew] = 0.0
    point = jnp.asarray(shape)
    jets = [scalar_jet(lambda shape: negative_second_moment(shape, density, count=count), point)
            for count in (32, 64)]
    assert_jet_close(dict(zip(("f", "g", "h"), jets[1])), jets[0], 7, density, value_atol=1e-9)
    # Density-space integration has a different variable and independent special functions.
    from scipy.integrate import quad

    expected, error = quad(lambda z: z * z * np.exp(scipy_logpdf(z, density, shape)),
                           -np.inf, 0.0, epsabs=1e-9, epsrel=1e-9, limit=200)
    assert error < 2e-8
    assert abs(jets[-1][0] - expected) < 2e-8
    if at_symmetry:
        # Keep the full asymmetric oracle at zero; a constant-half branch loses these derivatives.
        gtol, htol = (3e-8, 3e-7) if density in LIGHT else (3e-7, 4e-6)
        tangent = [i for i in range(shape.size) if i != skew]
        assert abs(jets[-1][1][skew]) > 32 * gtol
        assert np.max(np.abs(jets[-1][2][skew, tangent])) > 32 * htol
        assert abs(jets[-1][0] - 0.5) < 1e-9
        output = subprocess.check_output([str(binary), "--moments", density,
                                         *(repr(float(x)) for x in shape)], text=True, timeout=60)
        rows = np.array([[float(x) for x in line.split()] for line in output.splitlines()])
        width = shape.size
        assert rows.shape == (3, 3 + width + width * (width + 1) // 2)
        np.testing.assert_array_equal(rows[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(rows[:, 1], [1, 1, 1])
        np.testing.assert_array_equal(rows[:, 2], np.full(3, 0.5))
        upper = np.triu_indices(width)
        for order, row in enumerate(rows):
            hessian = np.zeros((width, width))
            hessian[upper] = row[3 + width:]
            hessian[(upper[1], upper[0])] = row[3 + width:]
            actual = dict(f=row[2], g=row[3:3 + width], h=hessian)
            assert_jet_close(actual, jets[-1], (1, 3, 7)[order], density, value_atol=1e-9)
            if order == 0:
                np.testing.assert_array_equal(actual["g"], np.zeros(width))
            if order < 2:
                np.testing.assert_array_equal(hessian, np.zeros_like(hessian))
