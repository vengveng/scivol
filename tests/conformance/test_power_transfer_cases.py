"""Retained APARCH cases missing from the ordinary conformance matrix.

Unapplied source proposal. Expectations come from the existing independent AD
and MP owners. These fixed-point tests make no optimizer or fit-reporting claim.
"""

from copy import deepcopy
import math

import numpy as np
import pytest

from fixtures.power_cases import case as power_case
from fixtures.joint_power_cases import case as joint_case
from oracles.garch import jax_modules, scalar_jet
from fixtures.garch_cases import case as garch_case
from oracles.power import full_jet as power_jet, mp_value as power_mp, theta_objective
from oracles.joint_power import (
    full_jet as joint_jet, jet as joint_requested_jet, mp_value as joint_mp,
    original_z_jet, shape_start,
)


DENSITIES = (
    ("normal", ()), ("studentt", (8.0,)), ("skewt", (8.0, 0.12)),
    ("ged", (1.6,)), ("gh", (0.1, 1.3, 0.4)), ("nig", (0.1, 1.3)),
    ("hyperbolic", (0.1, 1.3)), ("variance_gamma", (0.1, 3.0)),
)
VOL = (0.08, 0.07, 0.15, 0.82, 1.7)
MEAN = (0.03, 0.12, -0.08)
DATA = (0.7, -0.8, 0.35, -0.25, 0.6, -0.45, 0.2, -0.7)


def compare(actual, expected, mask, *, f_rtol=3e-9, f_atol=3e-9,
            g_rtol=4e-7, g_atol=4e-9, h_rtol=4e-6, h_atol=4e-8):
    assert actual["status"] == 0, actual["reasons"]
    assert actual["valid_mask"] == mask
    for bit, name, target, relative, absolute in zip(
            (1, 2, 4), ("f", "g", "h"), expected,
            (f_rtol, g_rtol, h_rtol), (f_atol, g_atol, h_atol), strict=True):
        value = np.asarray(actual[name])
        if mask & bit:
            assert np.isfinite(value).all() and np.isfinite(target).all()
            np.testing.assert_allclose(value, target, rtol=relative, atol=absolute)
        else:
            np.testing.assert_array_equal(value.view(np.uint64),
                np.full_like(value, actual["sentinel"]).view(np.uint64))


def all_masks(native, declared, expected, **tolerances):
    saved = {name: declared[name].copy() for name in ("parameters", "y", "design")
             if name in declared}
    for mask in range(1, 8):
        actual = native(declared, mask)
        np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
        compare(actual, expected, mask, **tolerances)
    for name, before in saved.items():
        np.testing.assert_array_equal(declared[name], before)


@pytest.mark.parametrize("density,shape", DENSITIES)
@pytest.mark.parametrize("joint", (False, True))
def test_retained_theta_density_points(native, density, shape, joint):
    y = np.random.default_rng(2718 if joint else 714).normal(size=28 if joint else 24)
    h0 = float(np.mean(y * y))
    declared = (joint_case("arma_aparch", density, parameters=(*MEAN, *VOL, *shape), y=y, h0=h0)
                if joint else power_case("aparch", density, parameters=(*VOL, *shape), y=y, h0=h0))
    expected = joint_jet(declared) if joint else power_jet(declared)
    all_masks(native, declared, expected)
    if joint:
        assert np.max(np.abs(expected[2][:3, 3:])) > 1e-6


@pytest.mark.parametrize("density,shape", DENSITIES[:4])
def test_retained_legacy_standard_density_tolerances(native, density, shape):
    y = np.random.default_rng(714).normal(size=24)
    declared = power_case("aparch", density, parameters=(*VOL, *shape), y=y,
                          h0=float(np.mean(y * y)))
    all_masks(native, declared, power_jet(declared), f_rtol=2e-10, f_atol=2e-10,
              g_rtol=2e-8, g_atol=2e-9, h_rtol=2e-7, h_atol=2e-8)


@pytest.mark.parametrize("density,shape", DENSITIES)
def test_retained_small_seed_density_scale(native, density, shape):
    scale = 1e-6
    declared = power_case("aparch", density, parameters=(*VOL, *shape),
                          y=[scale], h0=scale * scale)
    unit = power_case("aparch", density, parameters=(*VOL, *shape), y=[1.0], h0=1.0)
    baseline = power_jet(unit)
    expected = (baseline[0] + np.log(scale), baseline[1], baseline[2])
    independently_scaled = power_jet(declared)
    for actual, target in zip(independently_scaled, expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=4e-7, atol=4e-9)
    if density == "normal":
        np.testing.assert_allclose(expected[0],
            0.5 * (np.log(2 * np.pi) + 1.0) + np.log(scale), rtol=0, atol=2e-14)
    all_masks(native, declared, expected, f_rtol=0, f_atol=1e-9,
              g_rtol=4e-7, g_atol=4e-9, h_rtol=4e-7, h_atol=4e-9)
    for mask in (2, 4, 7):
        actual = native(declared, mask)
        if mask & 2:
            np.testing.assert_array_equal(actual["g"][:5], 0.0)
        if mask & 4:
            np.testing.assert_array_equal(actual["h"][:5, :], 0.0)
            np.testing.assert_array_equal(actual["h"][:, :5], 0.0)


def test_retained_supplied_seed_is_not_recomputed(native):
    declared = power_case("aparch", parameters=VOL, y=[1e-6], h0=1.0)
    expected = power_jet(declared)
    np.testing.assert_allclose(expected[0], 0.5 * (np.log(2 * np.pi) + 1e-12), rtol=0, atol=2e-14)
    all_masks(native, declared, expected, f_rtol=0, f_atol=2e-14)


@pytest.mark.parametrize("small,gamma", ((1e-170, 0.15), (0.1, math.nextafter(1.0, 0.0))))
def test_retained_small_shock_base(native, small, gamma):
    y = np.array([0.7, small, -0.4, 0.2, -0.5])
    declared = power_case("aparch", parameters=[0.2, 0.07, gamma, 0.6, 1.7],
                          y=y, h0=float(np.mean(y * y)))
    all_masks(native, declared, power_jet(declared), f_rtol=2e-6, f_atol=2e-7,
              g_rtol=2e-6, g_atol=2e-7, h_rtol=2e-6, h_atol=2e-7)


@pytest.mark.parametrize("label,orders", (
    ("standalone11", None), ("standalone32", (3, 2)),
    ("arma1111", (1, 1, 1, 1)), ("arma2243", (2, 2, 4, 3)),
))
def test_retained_ged_two_reduction(native, label, orders):
    joint = label.startswith("arma")
    p, q = orders[-2:] if orders is not None else (1, 1)
    mean = [0.03, *([0.06] * orders[0]), *([-0.04] * orders[1])] if joint else []
    theta = np.r_[mean, 0.08, np.full(p, 0.07 / p), np.linspace(0.15, -0.10, p),
                  np.full(q, 0.72 / q), 1.7, 2.0]
    y = np.random.default_rng(421203).normal(size=12)
    declared = (joint_case("arma_aparch", "ged", *orders, parameters=theta, y=y, h0=0.73)
                if joint else power_case("aparch", "ged", p, q, parameters=theta, y=y, h0=0.73))
    normal = deepcopy(declared)
    normal.update(density="normal", parameters=theta[:-1].copy(), theta_names=declared["theta_names"][:-1])
    expected = joint_jet(declared) if joint else power_jet(declared)
    full = native(declared, 7)
    compare(full, expected, 7)
    for mask in range(1, 8):
        left, right = native(declared, mask), native(normal, mask)
        assert left["status"] == right["status"] == 0
        assert left["valid_mask"] == right["valid_mask"] == mask
        for bit, name in zip((1, 2, 4), ("f", "g", "h"), strict=True):
            if mask & bit:
                projected = left[name] if bit == 1 else left[name][:-1] if bit == 2 else left[name][:-1, :-1]
                np.testing.assert_allclose(projected, right[name], rtol=0 if bit == 1 else 2e-11, atol=2e-11)
                np.testing.assert_allclose(left[name], full[name], rtol=0 if bit == 1 else 2e-11, atol=2e-11)


@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2)))
@pytest.mark.parametrize("density", ("normal", "studentt", "skewt"))
@pytest.mark.parametrize("delta,gamma,alpha,derivative_bits", (
    (0.8, 0.2, 0.07, 0), (1.7, 0.2, 0.07, 2), (2.0, 0.2, 0.07, 2),
    (2.0, 0.0, 0.07, 6), (2.5, 0.2, 0.07, 6), (0.5, 0.2, 0.0, 2),
    (1.7, 0.2, 0.0, 6),
))
def test_retained_zero_shock_request_validity(native, orders, density, delta, gamma, alpha, derivative_bits):
    ar, ma, p, q = orders
    shape = [] if density == "normal" else [8.0] if density == "studentt" else [8.0, 0.0]
    theta = np.r_[np.zeros(1 + ar + ma), 0.08, np.full(p, alpha / p),
                  np.full(p, gamma), np.full(q, 0.6 / q), delta, shape]
    y = np.r_[np.zeros(max(orders)), 0.0, 1.0, 0.0, 1.0]
    declared = joint_case("arma_aparch", density, *orders, parameters=theta, y=y, h0=1.0)
    for mask in range(1, 8):
        actual = native(declared, mask)
        differentiable = not (mask & 6 & ~derivative_bits)
        if differentiable:
            compare(actual, joint_requested_jet(declared, request=mask), mask)
        else:
            assert actual["status"] != 0 and actual["valid_mask"] & mask & ~derivative_bits & 6 == 0
        for bit, name in zip((1, 2, 4), ("f", "g", "h"), strict=True):
            if not mask & bit:
                value = np.asarray(actual[name])
                np.testing.assert_array_equal(value.view(np.uint64),
                    np.full_like(value, actual["sentinel"]).view(np.uint64))


@pytest.mark.parametrize("density,shape", DENSITIES[1:4])
def test_retained_log_variance_beta_curvature(native, density, shape):
    declared = power_case("aparch", density, parameters=[1e200, 0.0, 0.0, 0.5, 2.0, *shape],
                          y=[1e100, 1e100, -1e100], h0=1e200)
    unit = power_case("aparch", density, parameters=[1.0, 0.0, 0.0, 0.5, 2.0, *shape],
                      y=[1.0, 1.0, -1.0], h0=1.0)
    _, jnp, _ = jax_modules()
    objective = theta_objective(unit)
    def beta_objective(point):
        return objective(jnp.asarray(unit["parameters"]).at[3].set(point[0]))
    expected = scalar_jet(beta_objective, [0.5])[2][0, 0]
    actual = native(declared, 4)
    assert actual["status"] == 0 and actual["valid_mask"] == 4
    assert np.isfinite(actual["h"]).all()
    np.testing.assert_allclose(actual["h"][3, 3], expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("omega", (1e-310, math.nextafter(0.0, 1.0)))
def test_retained_subnormal_power_value(native, omega):
    declared = power_case("aparch", parameters=[omega, 0.0, 0.0, 0.0, 2.0], y=np.zeros(3), h0=1.0)
    coarse, fine = (power_mp(declared, digits=digits) for digits in (80, 120))
    assert math.isfinite(float(fine))
    assert abs(float(coarse) - float(fine)) <= 2e-12
    actual = native(declared, 1)
    assert actual["status"] == 0 and actual["valid_mask"] == 1
    np.testing.assert_allclose(actual["f"], float(fine), rtol=0, atol=1e-9)


@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2)))
@pytest.mark.parametrize("power_coordinate", (float.fromhex("0x1.62e42fefa39efp-1"), 0.2))
def test_retained_transformed_skew_moment_consumer(native, orders, power_coordinate):
    ar, ma, p, q = orders
    z = np.r_[np.zeros(1 + ar + ma), -2.5, np.full(p, -3.0),
              np.full(p, 0.15), np.full(q, -1.0), power_coordinate,
              float.fromhex("0x1.15288806261b3p-1"), 0.4]
    declared = joint_case("arma_aparch", "skewt", *orders, parameters=z,
                          y=DATA, h0=float(np.mean(np.array(DATA) ** 2)))
    declared.update(transformed=True, original_z=z.copy())
    actual = native(declared, 7)
    assert actual["status"] == 0 and actual["valid_mask"] == 7
    expected = original_z_jet(declared, actual["theta"], moment_count=512)
    coarse = original_z_jet(declared, actual["theta"], moment_count=256)
    for left, right in zip(coarse, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-9)
    width = 1 + ar + ma
    assert np.max(np.abs(expected[2][:width, width:])) > 1e-6
    all_masks(native, declared, expected, f_rtol=2e-10, f_atol=2e-11,
              g_rtol=2e-8, g_atol=2e-9, h_rtol=2e-8, h_atol=2e-9)


@pytest.mark.parametrize("density,shape", DENSITIES[:3])
def test_retained_96_row_near_boundary_theta(native, density, shape):
    theta = np.array([0.03, 0.12, 0.997, 0.04, 0.01, 0.1, 0.97, 1.6, *shape])
    y = np.resize(np.array([0.8, -0.6]), 96)
    declared = joint_case("arma_aparch", density, parameters=theta, y=y, h0=0.8)
    expected = joint_jet(declared)
    assert np.max(np.abs(expected[2][:3, :3])) > 1e4
    all_masks(native, declared, expected, f_rtol=0, f_atol=2e-11,
              g_rtol=2e-8, g_atol=2e-9, h_rtol=2e-8, h_atol=2e-9)


@pytest.mark.parametrize("density,theta", (
    ("normal", [0.1, 0.0, 1.0, 0.4, 2.0]),
    ("normal", [0.1, 0.0, 0.0, 0.4, 0.0]),
    ("normal", [0.1, 0.0, 0.0, 0.4, np.nan]),
    ("studentt", [0.1, 0.0, 0.0, 0.4, 8.0, 8.0]),
))
def test_retained_zero_alpha_does_not_hide_invalid_moment_domain(native, density, theta):
    declared = power_case("aparch", density, parameters=theta)
    for mask in range(1, 8):
        actual = native(declared, mask)
        assert actual["status"] == 4 and actual["valid_mask"] == 0


@pytest.mark.parametrize("density,shape", DENSITIES[4:])
def test_retained_legacy_gh_density_tolerances(native, density, shape):
    y = np.random.default_rng(714).normal(size=24)
    declared = power_case("aparch", density, parameters=(*VOL, *shape), y=y,
                          h0=float(np.mean(y * y)))
    all_masks(native, declared, power_jet(declared), f_atol=3e-9,
              g_rtol=3e-7, g_atol=3e-8, h_rtol=3e-6, h_atol=3e-7)


@pytest.mark.parametrize("density,shape", DENSITIES)
def test_retained_arbitrary_2132_joint_theta(native, density, shape):
    y = np.random.default_rng(204).normal(size=18)
    theta = [0.02, 0.10, -0.04, -0.08, 0.06, 0.02, 0.015, 0.01,
             0.10, -0.12, 0.08, 0.55, 0.18, 1.8, *shape]
    declared = joint_case("arma_aparch", density, 2, 1, 3, 2, parameters=theta,
                          y=y, h0=float(np.mean(y * y)))
    expected = joint_jet(declared)
    assert np.max(np.abs(expected[2][:4, 4:])) > 1e-6
    all_masks(native, declared, expected, f_rtol=2e-10, f_atol=2e-10,
              g_rtol=2e-8, g_atol=2e-9, h_rtol=2e-7, h_atol=2e-8)
    assert native(declared, 1)["dispatch"] == 0


@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_retained_delta_1500_scaled_seed_theta(native, p, q):
    declared = power_case("aparch", p=p, q=q,
        parameters=np.r_[1.0, np.zeros(2 * p), np.full(q, 0.1 / q), 1500.0],
        y=np.zeros(9), h0=4.0)
    declared["power_log_units"] = math.log(2.0)
    all_masks(native, declared, power_jet(declared), f_rtol=3e-12, f_atol=2e-14,
              g_rtol=3e-12, g_atol=2e-14, h_rtol=3e-12, h_atol=2e-14)


@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 0, 3, 2)))
def test_retained_joint_delta_1500_scaled_seed_theta(native, orders):
    ar, ma, p, q = orders
    theta = np.r_[0.03, np.full(ar, 0.12), np.full(ma, -0.08), 1.0,
                  np.zeros(p), np.full(p, -0.15), np.full(q, 0.1 / q), 1500.0]
    y = [0.3, -0.7, 0.1, 1.8, -0.2, 0.6, -0.4, 0.15, 1.0, -0.9, 0.3, 0.5]
    declared = joint_case("arma_aparch", p_ar=ar, q_ma=ma, p_vol=p, q_vol=q,
                          parameters=theta, y=y, h0=4.0)
    declared["power_log_units"] = math.log(2.0)
    expected = joint_jet(declared)
    width = 1 + ar + ma
    assert np.linalg.norm(expected[2][:width, width:]) > 1e-4
    all_masks(native, declared, expected, f_rtol=4e-10, f_atol=2e-12,
              g_rtol=4e-10, g_atol=2e-12, h_rtol=4e-10, h_atol=2e-12)


def test_retained_large_power_seed_complete_hessian(native):
    declared = power_case("aparch", parameters=[1e200, 0.0, 0.0, 0.5, 2.0],
                          y=np.zeros(3), h0=1e200)
    declared["power_log_units"] = math.log(declared["h0"]) / 2
    all_masks(native, declared, power_jet(declared), f_rtol=2e-12, f_atol=0.0,
              g_rtol=2e-12, g_atol=0.0, h_rtol=2e-12, h_atol=0.0)


def test_retained_800_row_subnormal_boundary_sum(native):
    pattern = [0.8, -0.4, 1.1, 0.3, -1.2, 0.7, 0.6, -0.2, 1.4, -0.9, 0.5, 0.2, -0.6, 0.9]
    y = np.resize(pattern, 800) * 1e-160
    declared = joint_case("arma_aparch", parameters=[0.0, 0.0, 0.0, 1e-310, 0.0, -0.2, 0.35, 2.0],
                          y=y, h0=1.0)
    coarse, fine = (joint_mp(declared, digits=digits) for digits in (80, 120))
    import mpmath as mp
    with mp.workdps(120):
        assert abs(coarse - fine) < mp.mpf("1e-65")
        actual = native(declared, 1)
        assert actual["status"] == 0 and actual["valid_mask"] == 1
        assert abs(mp.mpf(actual["f"]) - fine) <= mp.mpf("2e-11")
    for name in ("g", "h"):
        np.testing.assert_array_equal(actual[name], actual["sentinel"])


def test_retained_normal_garch_reduction_tolerances(native):
    y = np.random.default_rng(71).normal(size=18)
    h0 = float(np.mean(y * y))
    aparch = power_case("aparch", parameters=[0.08, 0.07, 0.0, 0.82, 2.0], y=y, h0=h0)
    garch = garch_case("garch", parameters=[0.08, 0.07, 0.82], y=y, h0=h0)
    common = [0, 1, 3]
    for mask in range(1, 8):
        left, right = native(aparch, mask), native(garch, mask)
        assert left["status"] == right["status"] == 0
        assert left["valid_mask"] == right["valid_mask"] == mask
        for bit, name, relative, absolute in zip((1, 2, 4), ("f", "g", "h"),
                (2e-13, 2e-12, 2e-11), (2e-13, 2e-12, 2e-11), strict=True):
            if mask & bit:
                projection = left[name] if bit == 1 else left[name][common] if bit == 2 else left[name][np.ix_(common, common)]
                np.testing.assert_allclose(projection, right[name], rtol=relative, atol=absolute)
