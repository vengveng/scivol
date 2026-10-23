"""Deferred independent-AD checks for the plain-GARCH map consolidation.

Run only against the chief-approved rebuilt standalone bridge. No legacy import
or implicit build is allowed. Collection does not initialize JAX.
"""

from copy import deepcopy

import mpmath as mp
import numpy as np
import pytest

from fixtures.garch_cases import DENSITIES, chart_case
from fixtures.joint_cases import case as joint_case, chart_case as joint_chart
from oracles.charts import coordinate_map, pullback
from oracles.garch import full_jet, jax_modules, scalar_jet, theta_objective
from oracles.joint import full_jet as joint_jet, pullback as joint_pullback


def assert_requests(native, declared, expected, *, full=None):
    full = native(declared, 7) if full is None else full
    density = declared["density"]
    light = density in {"normal", "studentt", "skewt", "ged"}
    value_atol = 2e-9 if light else 3e-8
    gtol, htol = (3e-8, 3e-7) if light else (3e-7, 4e-6)
    for mask in range(1, 8):
        actual = full if mask == 7 else native(declared, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask, actual.get("reasons")
        np.testing.assert_array_equal(actual["theta"], full["theta"])
        np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
        for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected):
            output = np.asarray(actual[name])
            if mask & bit:
                assert np.isfinite(output).all() and np.isfinite(reference).all()
                tolerance = gtol if bit == 2 else htol
                np.testing.assert_allclose(output, reference, rtol=0 if bit == 1 else tolerance,
                                           atol=value_atol if bit == 1 else tolerance)
                np.testing.assert_allclose(output, full[name], rtol=0 if bit == 1 else 2e-12,
                                           atol=2e-10 if bit == 1 else 2e-12)
            else:
                untouched = np.full_like(output, actual["sentinel"])
                np.testing.assert_array_equal(output.view(np.uint64), untouched.view(np.uint64))
    return full


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (4, 3)))
def test_shared_map_all_seven_requests_against_independent_ad(native, density, p, q):
    declared = chart_case("garch", density, p, q)
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full.get("reasons")
    assert full["dispatch"] == int((p, q) == (1, 1))
    _, jnp, _ = jax_modules()
    mapped = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(full["theta"], mapped, rtol=3e-14, atol=3e-14)
    expected = pullback(declared, full_jet(declared, theta=full["theta"]))
    assert_requests(native, declared, expected, full=full)


@pytest.mark.parametrize("model", ("arma_garch", "linear_garch"))
@pytest.mark.parametrize("density", ("normal", "studentt"))
def test_joint_plain_map_keeps_mean_and_shape_cross_blocks(native, model, density):
    declared = joint_chart(joint_case(model, density, p_ar=2, q_ma=1,
                                     p_vol=3, q_vol=2, columns=5, intercept=False))
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full.get("reasons")
    expected = joint_pullback(declared, joint_jet(declared, theta=full["theta"]))
    assert_requests(native, declared, expected, full=full)


def stable_tail_map(z):
    """Independent two-logit chart with no 1-rounded_weight differentiation."""
    jax, jnp, _ = jax_modules()
    slack_ratio, other_ratio = jnp.exp(-z[1]), jnp.exp(z[2] - z[1])
    denominator = 1.0 + slack_ratio + other_ratio
    tail = (slack_ratio + other_ratio) / denominator
    omega = jax.lax.cond(z[0] >= 0.0,
                         lambda: z[0] + jnp.log1p(jnp.exp(-z[0])),
                         lambda: jnp.log1p(jnp.exp(z[0])))
    return jnp.array([omega, 1.0 - tail, other_ratio / denominator])


@pytest.mark.parametrize("omega,dominant", ((-0.8, 37.0), (-700.0, 37.0), (50.0, 37.0)))
def test_complete_endpoint_jet_uses_original_z_and_actual_theta(native, omega, dominant):
    declared = chart_case("garch", "normal")
    declared["parameters"] = np.array([omega, dominant, 0.0])
    declared["original_z"] = declared["parameters"].copy()
    full = native(declared, 7)
    assert full["status"] == 0 and full["valid_mask"] == 7, full.get("reasons")
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(declared["original_z"])
    mapped = stable_tail_map(z)
    anchor = jnp.asarray(full["theta"])
    objective = theta_objective(declared)
    expected = scalar_jet(lambda point: objective(anchor + (stable_tail_map(point) - mapped)), z)
    assert_requests(native, declared, expected, full=full)

    # Audit the tail AD itself at much higher precision. Ordinary absolute
    # tolerances would accept a vanished dominant derivative.
    with mp.workdps(100):
        point = [mp.mpf(float(value)) for value in declared["original_z"]]

        def alpha(left, right):
            return mp.exp(left) / (1 + mp.exp(left) + mp.exp(right))

        first = float(mp.diff(lambda left: alpha(left, point[2]), point[1]))
        second = float(mp.diff(lambda left: alpha(left, point[2]), point[1], 2))
        omega_slope = float(1 / (1 + mp.exp(-point[0])))
    jacobian = np.asarray(jax.jacfwd(stable_tail_map)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(stable_tail_map))(z))
    np.testing.assert_allclose(jacobian[1, 1], first, rtol=5e-14, atol=0)
    np.testing.assert_allclose(curvature[1, 1, 1], second, rtol=5e-14, atol=0)
    np.testing.assert_allclose(full["theta"], np.asarray(mapped), rtol=3e-15, atol=0)
    for mask in (2, 3, 4, 5, 6, 7):
        actual = full if mask == 7 else native(declared, mask)
        if mask & 2:
            np.testing.assert_allclose(actual["g"][1] / first, expected[1][1] / first,
                                       rtol=3e-8, atol=3e-8)
            np.testing.assert_allclose(actual["g"][0] / omega_slope, expected[1][0] / omega_slope,
                                       rtol=3e-8, atol=3e-8)
        if mask & 4:
            np.testing.assert_allclose(actual["h"][1, 1] / first, expected[2][1, 1] / first,
                                       rtol=3e-7, atol=3e-7)
            np.testing.assert_allclose(actual["h"][0, :] / omega_slope, expected[2][0, :] / omega_slope,
                                       rtol=3e-7, atol=3e-7)
            np.testing.assert_allclose(actual["h"][:, 0] / omega_slope, expected[2][:, 0] / omega_slope,
                                       rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize("z", ((-0.8, 40.0, 0.0), (-750.0, -2.0, -1.0)))
def test_unrepresentable_endpoint_rejection_preserves_every_output(native, z):
    declared = deepcopy(chart_case("garch", "normal"))
    declared["parameters"] = np.asarray(z)
    declared["original_z"] = declared["parameters"].copy()
    for mask in range(1, 8):
        actual = native(declared, mask)
        assert actual["status"] != 0 and actual["valid_mask"] == 0
        for name in ("f", "g", "h"):
            output = np.asarray(actual[name])
            untouched = np.full_like(output, actual["sentinel"])
            np.testing.assert_array_equal(output.view(np.uint64), untouched.view(np.uint64))
