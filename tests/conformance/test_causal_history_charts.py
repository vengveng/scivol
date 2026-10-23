"""Independent causal chart composition; execution belongs to chief's schedule."""

import mpmath as mp
import numpy as np
import pytest

from fixtures.history_cases import (
    CAUSAL_MODELS, DENSITIES, as_chart, case, causal_ar_plateau_case, declarations,
)
from fixtures.mean_cases import arma_case, chart_case, linear_case
from fixtures.profile_arx_cases import reflection_values
from oracles.garch import jax_modules, scalar_jet
from oracles.mean import mean_coordinate_map, mean_count, reflection_mp
from oracles.profile import coordinate_map as profile_mean_map
from test_history_mathematics import owners, supplied_mode


def anchored_objective_jet(oracle, declared, theta_anchor):
    _, jnp, _ = jax_modules()
    z = jnp.asarray(declared["original_z"])
    mapping = oracle.coordinate_map(declared)
    mapped = mapping(z)
    objective = oracle.theta_objective(declared, theta_anchor)
    anchor = jnp.asarray(theta_anchor)
    return scalar_jet(lambda point: objective(anchor + (mapping(point) - mapped)), z)


@pytest.mark.parametrize("model,density,mode", declarations(causal=True))
@pytest.mark.parametrize("intercept", (False, True))
def test_causal_original_z_all_masks_with_declared_histories(native, model, density, mode, intercept):
    # Constant-scale ARX has no residual recurrence to carry unscored history.
    conditioned = case(model, density, mode, causal_intercept=intercept,
                       causal_first_offset=0 if model.endswith("_constant") else 2)
    declared = as_chart(conditioned)
    for field in ("y", "design", "input_exogenous", "observation_history", "innovation_history",
                  "recursion_begin", "hold_back", "first", "stop", "scored_rows"):
        np.testing.assert_array_equal(declared[field], conditioned[field])
    if "h0" in conditioned:
        assert declared["h0"] == conditioned["h0"]
    oracle, checks = owners(declared)
    _, jnp, _ = jax_modules()
    mapped = np.asarray(oracle.coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    actual = native(declared, 7)
    checks.assert_success(actual, declared, 7)
    # Native theta supplies only the rounding anchor, never chart expectations.
    np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-9, atol=3e-10)
    expected = anchored_objective_jet(oracle, declared, actual["theta"])
    checks.exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", CAUSAL_MODELS)
@pytest.mark.parametrize("intercept", (False, True))
def test_causal_mean_chart_blocks_compose_without_touching_other_axes(model, intercept):
    declared = as_chart(case(model, "ged", supplied_mode(model), causal_intercept=intercept))
    oracle, _ = owners(declared)
    width = int(intercept) + declared["design"].shape[1]
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(declared["original_z"])
    mean_mapping = lambda point: oracle.coordinate_map(declared)(point)[:width]
    mapped = np.asarray(mean_mapping(z))
    jacobian = np.asarray(jax.jacfwd(mean_mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mean_mapping))(z))
    expected = np.asarray(z[:width]).copy()
    if declared["causal_kind"] == "arx":
        start, end = int(intercept), int(intercept) + declared["causal_order"]
        expected[start:end] = reflection_values(np.asarray(z[start:end]))
        assert expected[start] > 1.0
        assert np.max(np.abs(curvature[start:end, start:end, start:end])) > 1e-3
    else:
        np.testing.assert_array_equal(jacobian[:, :width], np.eye(width))
        np.testing.assert_array_equal(curvature, np.zeros_like(curvature))
    np.testing.assert_allclose(mapped, expected, rtol=4e-14, atol=4e-15)
    np.testing.assert_array_equal(jacobian[:, width:], np.zeros_like(jacobian[:, width:]))
    np.testing.assert_array_equal(curvature[:, width:, :], np.zeros_like(curvature[:, width:, :]))
    np.testing.assert_array_equal(curvature[:, :, width:], np.zeros_like(curvature[:, :, width:]))
    for row in (([0] if intercept else []) + [width - 1]):
        np.testing.assert_array_equal(jacobian[row], np.eye(len(z))[row])
        np.testing.assert_array_equal(curvature[row], np.zeros_like(curvature[row]))


@pytest.mark.parametrize("model", CAUSAL_MODELS)
@pytest.mark.parametrize("density", DENSITIES)
def test_causal_independent_complete_objective_includes_mean_chart_curvature(model, density):
    declared = as_chart(case(model, density, supplied_mode(model), causal_intercept=False,
                             causal_first_offset=0 if model.endswith("_constant") else 2))
    oracle, checks = owners(declared)
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(declared["original_z"])
    mapping = oracle.coordinate_map(declared)
    theta = np.asarray(mapping(z))
    theta_jet = oracle.full_jet(declared, theta=theta)
    expected = anchored_objective_jet(oracle, declared, theta)
    pulled = oracle.pullback(declared, theta_jet)
    checks.assert_requested_close(dict(zip(("f", "g", "h"), expected)), pulled, 7, density)
    width = declared["design"].shape[1]
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    contraction = np.einsum("i,ijk->jk", theta_jet[1], curvature)
    naive = jacobian.T @ theta_jet[2] @ jacobian
    np.testing.assert_allclose(pulled[2] - naive, contraction, rtol=3e-11, atol=3e-11)
    if declared["causal_kind"] == "arx":
        assert np.max(np.abs(contraction[:width, :width])) > 1e-8
    else:
        np.testing.assert_array_equal(contraction[:width, :width], np.zeros((width, width)))


@pytest.mark.parametrize("model,p,q", (("arma", 0, 0), ("arma", 0, 3),
                                      ("arma", 3, 0), ("arma", 2, 3), ("linear", 0, 0)))
@pytest.mark.parametrize("intercept", (False, True))
def test_shared_mean_chart_preserves_existing_noncausal_reflection_owner(model, p, q, intercept):
    template = (arma_case("ged", p, q, intercept=intercept) if model == "arma"
                else linear_case("ged", columns=3, intercept=intercept))
    declared = chart_case(template)
    jax, jnp, _ = jax_modules()
    z, width = jnp.asarray(declared["original_z"]), mean_count(declared)
    mapping = mean_coordinate_map(declared)
    reference = lambda point: profile_mean_map(declared)(point[:width])
    for derivative in (lambda f: f, jax.jacfwd, lambda f: jax.jacfwd(jax.jacfwd(f))):
        np.testing.assert_array_equal(np.asarray(derivative(mapping)(z)), np.asarray(derivative(reference)(z)))


def ar1_mp_pullback_row(oracle, declared, theta_jet, *, digits):
    """MP AR-map contraction of independent theta AD, not an MP likelihood jet."""
    jax, jnp, _ = jax_modules()
    z, ar = jnp.asarray(declared["original_z"]), int(declared["intercept"])
    mapping = oracle.coordinate_map(declared)
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    other = np.delete(np.arange(len(z)), ar)
    np.testing.assert_array_equal(jacobian[other, ar], np.zeros(len(other)))
    np.testing.assert_array_equal(jacobian[ar, other], np.zeros(len(other)))
    np.testing.assert_array_equal(curvature[:, ar, other], np.zeros((len(z), len(other))))
    with mp.workdps(digits):
        coordinate = mp.mpf(float(declared["original_z"][ar]))
        component = lambda point: reflection_mp((point,))[0]
        kappa = component(coordinate)
        slope = mp.diff(component, coordinate)
        second = mp.diff(component, coordinate, 2)
        assert abs(kappa) < 1 and slope > 0 and second != 0
        tail = mp.exp(-2 * abs(coordinate))
        assert mp.almosteq(slope, 4 * tail / (1 + tail)**2)
        assert mp.almosteq(second, -2 * kappa * slope)
        np.testing.assert_allclose(jacobian[ar, ar] / float(slope), 1, rtol=4e-13, atol=0)
        np.testing.assert_allclose(curvature[ar, ar, ar] / float(second), 1, rtol=4e-13, atol=0)
        gradient = mp.mpf(float(theta_jet[1][ar]))
        hessian_row = [mp.mpf(float(value)) for value in theta_jet[2][ar]]
        mapped_row = []
        for column in range(len(z)):
            value = (slope * slope * hessian_row[ar] + gradient * second if column == ar
                     else slope * mp.fsum(hessian_row[row] * mp.mpf(float(jacobian[row, column]))
                                          for row in other))
            mapped_row.append(float(value))
        return float(slope), float(slope * gradient), np.asarray(mapped_row)


@pytest.mark.parametrize("volatility", ("constant", "gjr", "egarch", "aparch"))
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("sign", (-1, 1))
def test_causal_ar_original_z_positive_plateau_all_masks(native, volatility, density, sign):
    points = [causal_ar_plateau_case(volatility, density, sign, coordinate)
              for coordinate in (18.0, 18.05)]
    actuals, targets = [], []
    for declared in points:
        start = declared["hold_back"]
        assert declared["first"] == start and declared["scored_rows"][0] == start
        assert declared["design"][start, 0] == declared["observation_history"][0]
        assert bool(declared["innovation_history"]) == (volatility != "constant")
        oracle, checks = owners(declared)
        actual = native(declared, 7)
        checks.assert_success(actual, declared, 7)
        assert 0 < abs(actual["theta"][1]) < 1
        _, jnp, _ = jax_modules()
        mapped = np.asarray(oracle.coordinate_map(declared)(jnp.asarray(declared["original_z"])))
        np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-9, atol=3e-10)
        expected = anchored_objective_jet(oracle, declared, actual["theta"])
        checks.exercise_requests(native, declared, expected)
        theta_jet = oracle.full_jet(declared, theta=actual["theta"])
        coarse = ar1_mp_pullback_row(oracle, declared, theta_jet, digits=80)
        target = ar1_mp_pullback_row(oracle, declared, theta_jet, digits=120)
        for left, right in zip(coarse, target):
            np.testing.assert_allclose(left, right, rtol=2e-14, atol=0)
        slope, gradient, row = target
        assert gradient != 0 and row[1] != 0
        shape = mean_count(declared) + 1 if volatility == "constant" else oracle.shape_start(declared)
        for block in ([0, 2], range(3, shape), range(shape, len(row))):
            assert np.max(np.abs(row[list(block)])) > 0
        gradient_tolerance = 3e-8 if density in ("skewt", "ged") else 3e-7
        hessian_tolerance = 3e-7 if density in ("skewt", "ged") else 4e-6
        np.testing.assert_allclose(expected[1][1] / slope, gradient / slope,
                                   rtol=gradient_tolerance, atol=gradient_tolerance)
        np.testing.assert_allclose(expected[2][1] / slope, row / slope,
                                   rtol=hessian_tolerance, atol=hessian_tolerance)
        for mask in range(1, 8):
            output = actual if mask == 7 else native(declared, mask)
            checks.assert_success(output, declared, mask)
            if mask & 2:
                assert output["g"][1] != 0
                np.testing.assert_allclose(output["g"][1] / slope, gradient / slope,
                                           rtol=gradient_tolerance, atol=gradient_tolerance)
            if mask & 4:
                assert output["h"][1, 1] != 0
                np.testing.assert_allclose(output["h"][1] / slope, row / slope,
                                           rtol=hessian_tolerance, atol=hessian_tolerance)
                np.testing.assert_allclose(output["h"][:, 1] / slope, row / slope,
                                           rtol=hessian_tolerance, atol=hessian_tolerance)
        actuals.append(actual)
        targets.append(target)
    np.testing.assert_array_equal(actuals[0]["theta"], actuals[1]["theta"])
    assert actuals[0]["f"] == actuals[1]["f"]
    assert targets[0][0] != targets[1][0] and targets[0][1] != targets[1][1]
    assert actuals[0]["g"][1] != actuals[1]["g"][1]


@pytest.mark.parametrize("volatility", ("constant", "gjr", "egarch", "aparch"))
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("sign", (-1, 1))
def test_causal_ar_original_z_rounded_boundary_rejects_all_masks(native, volatility, density, sign):
    declared = causal_ar_plateau_case(volatility, density, sign, 20.0)
    with mp.workdps(100):
        kappa = reflection_mp((mp.mpf(sign * 20),))[0]
        assert abs(kappa) < 1 and float(kappa) == float(sign)
    for mask in range(1, 8):
        actual = native(declared, mask)
        assert actual["status"] == 5 and actual["valid_mask"] == 0
        np.testing.assert_array_equal(actual["scored_rows"], declared["scored_rows"])
        # Point preparation fails before the private bridge invokes the jet.
        np.testing.assert_array_equal(actual["reasons"], (999, 999, 999))
        assert np.isnan(actual["theta"]).all()
        for field in ("f", "g", "h"):
            value = np.asarray(actual[field])
            np.testing.assert_array_equal(value.view(np.uint64),
                                           np.full_like(value, actual["sentinel"]).view(np.uint64))
