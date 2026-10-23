"""Six-density history/unit regressions for chief-scheduled execution."""

import math

import mpmath as mp
import numpy as np
import pytest

from fixtures.history_cases import DENSITIES
from fixtures.history_units_cases import (
    DECLARATIONS, MODELS, VOLATILITIES, design_divisors, fitting_case, internal_case, internal_theta_map, unit_case,
)
from fixtures.joint_power_cases import mean_width
from oracles.charts import coordinate_map as variance_chart
from oracles.garch import jax_modules, scalar_jet
from test_history_mathematics import owners, supplied_mode


def objective(oracle, declared, anchor):
    return (oracle.theta_objective(declared) if declared["model"] in VOLATILITIES
            else oracle.theta_objective(declared, anchor))


def log_states(oracle, declared, theta):
    _, jnp, _ = jax_modules()
    if declared["model"] in ("egarch", "aparch"):
        return oracle.log_variance_path(theta, declared)
    if declared["model"] in ("garch", "gjr"):
        return jnp.log(oracle.variance_path(theta, declared))
    state = oracle.state_path(theta, declared)[1]
    return state if declared["model"].endswith(("egarch", "aparch")) else jnp.log(state)


def coordinate_map(oracle, declared):
    return (variance_chart(declared) if declared["model"] in ("garch", "gjr")
            else oracle.coordinate_map(declared))


def public_point(oracle, declared, anchor):
    _, jnp, _ = jax_modules()
    if not declared["transformed"]:
        return lambda point: point
    mapping = coordinate_map(oracle, declared)
    mapped = mapping(jnp.asarray(declared["original_z"]))
    np.testing.assert_allclose(anchor, np.asarray(mapped), rtol=3e-9, atol=3e-10)
    # Native theta anchors rounding only. Derivatives remain at the original z.
    return lambda point: jnp.asarray(anchor) + (mapping(point) - mapped)


@pytest.mark.parametrize("model,density,mode", DECLARATIONS)
@pytest.mark.parametrize("chart", (False, True))
@pytest.mark.parametrize("divisor", (4.0, 0.25))
def test_history_units_complete_objective_all_masks(native, model, density, mode, chart, divisor):
    declared = unit_case(model, density, mode, chart=chart, divisor=divisor)
    baseline = dict(declared, internal_response_scale=1.0, internal_design_scales=())
    oracle, checks = owners(declared)
    actual = native(declared, 7)
    checks.assert_success(actual, declared, 7)
    anchor = actual["theta"] if chart else declared["parameters"]
    _, jnp, _ = jax_modules()
    point = public_point(oracle, declared, anchor)
    convert = internal_theta_map(declared)
    fitting, internal = fitting_case(declared), internal_case(declared)
    fitting_objective = objective(oracle, fitting, anchor)
    internal_objective = objective(oracle, internal, np.asarray(convert(jnp.asarray(anchor))))
    correction = len(declared["scored_rows"]) * math.log(divisor)
    expected = scalar_jet(lambda source: fitting_objective(point(source)), declared["parameters"])
    converted = scalar_jet(lambda source: internal_objective(convert(point(source))) + correction,
                           declared["parameters"])
    for left, right in zip(expected, converted):
        np.testing.assert_allclose(left, right, rtol=3e-10, atol=3e-10)
    checks.exercise_requests(native, baseline, expected)
    checks.exercise_requests(native, declared, converted)
    # History arrays and score/trim declarations are exogenous, not fitted axes.
    assert len(anchor) == len(declared["theta_names"])
    assert fitting["recursion_begin"] == internal["recursion_begin"] == declared["recursion_begin"]
    np.testing.assert_array_equal(internal["scored_rows"], declared["scored_rows"])


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", ("empty", "innovations"))
def test_history_unit_state_ad_and_fixed_row_zero(density, model, mode):
    declared = unit_case(model, density, mode)
    oracle, _ = owners(declared)
    jax, jnp, _ = jax_modules()
    theta = jnp.asarray(declared["parameters"])
    fitting, internal = fitting_case(declared), internal_case(declared)
    convert = internal_theta_map(declared)
    public = lambda point: log_states(oracle, fitting, point)
    scaled = lambda point: log_states(oracle, internal, convert(point))
    shift = -2 * math.log(declared["internal_response_scale"])
    for derivative in (lambda fn: fn, jax.jacfwd, lambda fn: jax.jacfwd(jax.jacfwd(fn))):
        left, right = np.asarray(derivative(public)(theta)), np.asarray(derivative(scaled)(theta))
        if left.ndim == 1:
            left = left + shift
        np.testing.assert_allclose(right, left, rtol=3e-10, atol=3e-10)
    if not model.startswith("arma_"):
        assert float(scaled(theta)[0]) == pytest.approx(math.log(internal["h0"]), rel=0, abs=2e-15)
        np.testing.assert_array_equal(np.asarray(jax.jacfwd(scaled)(theta))[0], np.zeros(len(theta)))
        np.testing.assert_array_equal(np.asarray(jax.jacfwd(jax.jacfwd(scaled))(theta))[0],
                                      np.zeros((len(theta), len(theta))))
    # Only empty-innovation, non-ARMA EGARCH uses the implicit log-state field.
    ignored = dict(internal, implicit_log_variance=17.0)
    changed = np.asarray(log_states(oracle, ignored, convert(theta)))
    if not model.endswith("egarch") or mode == "innovations" or model == "arma_egarch":
        np.testing.assert_array_equal(changed, np.asarray(scaled(theta)))
    else:
        assert np.max(np.abs(changed[1:] - np.asarray(scaled(theta))[1:])) > 1e-3


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", ("egarch", "linear_egarch", "arx_egarch", "harx_egarch"))
def test_missing_persistence_seed_keeps_beta_sensitivity_and_propagated_hessian(density, model):
    declared = unit_case(model, density, "empty")
    oracle, _ = owners(declared)
    jax, jnp, _ = jax_modules()
    internal = internal_case(declared)
    theta = internal_theta_map(declared)(jnp.asarray(declared["parameters"]))
    wrong = dict(internal, implicit_log_variance=0.0)
    difference = lambda point: log_states(oracle, internal, point) - log_states(oracle, wrong, point)
    width = 0 if model == "egarch" else mean_width(declared)
    p = declared["p"] if model == "egarch" else declared["p_vol"]
    beta = width + 1 + 2 * p
    value = np.asarray(difference(theta))
    gradient = np.asarray(jax.jacfwd(difference)(theta))
    hessian = np.asarray(jax.jacfwd(jax.jacfwd(difference))(theta))
    shift = internal["implicit_log_variance"]
    # q=2: only the second persistence lag is missing at t=1.
    assert value[1] == pytest.approx(float(theta[beta + 1]) * shift, rel=2e-13, abs=2e-14)
    expected = np.zeros(len(theta)); expected[beta + 1] = shift
    np.testing.assert_allclose(gradient[1], expected, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(hessian[1], np.zeros_like(hessian[1]))
    assert np.max(np.abs(hessian[2:])) > 1e-5
    correct_objective, stale_objective = objective(oracle, internal, theta), objective(oracle, wrong, theta)
    assert abs(float(correct_objective(theta) - stale_objective(theta))) > 1e-6


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", ("egarch", "linear_egarch", "arma_egarch"))
def test_mp_state_owner_also_converts_implicit_history(density, model):
    declared = unit_case(model, density, "empty")
    oracle, _ = owners(declared)
    _, jnp, _ = jax_modules()
    fitting, internal = fitting_case(declared), internal_case(declared)
    theta = np.asarray(internal_theta_map(declared)(jnp.asarray(declared["parameters"])))
    with mp.workdps(70):
        if model == "egarch":
            before = oracle.mp_log_variances(fitting, declared["parameters"])
            after = oracle.mp_log_variances(internal, theta)
        else:
            before = oracle.mp_states(fitting, declared["parameters"])[1]
            after = oracle.mp_states(internal, theta)[1]
        shift = mp.mpf(internal["implicit_log_variance"])
        # The independent theta unit map is rounded once to binary64 here.
        for left, right in zip(before, after):
            assert abs(right - left - shift) < mp.mpf("3e-13")


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", tuple(model for model in MODELS if model.endswith("aparch")))
@pytest.mark.parametrize("chart", (False, True))
@pytest.mark.parametrize("divisor", (4.0, 0.25))
def test_aparch_units_keep_delta_crosscurvature(density, model, chart, divisor):
    declared = unit_case(model, density, supplied_mode(model), chart=chart, divisor=divisor)
    oracle, _ = owners(declared)
    jax, jnp, _ = jax_modules()
    source = jnp.asarray(declared["parameters"])
    anchor = coordinate_map(oracle, declared)(source) if chart else source
    point = public_point(oracle, declared, anchor)
    convert = internal_theta_map(declared)
    composed_map = lambda coordinates: convert(point(coordinates))
    internal = internal_case(declared)
    internal_anchor = np.asarray(convert(anchor))
    internal_objective = objective(oracle, internal, internal_anchor)
    inner = scalar_jet(internal_objective, internal_anchor)
    correction = len(declared["scored_rows"]) * math.log(divisor)
    complete = scalar_jet(lambda coordinates: internal_objective(composed_map(coordinates)) + correction, source)
    jacobian = np.asarray(jax.jacfwd(composed_map)(source))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(composed_map))(source))
    contraction = np.einsum("i,ijk->jk", inner[1], curvature)
    naive = jacobian.T @ inner[2] @ jacobian
    np.testing.assert_allclose(complete[2], naive + contraction, rtol=3e-10, atol=3e-10)
    delta, omega = (declared["theta_names"].index(name) for name in ("delta", "omega"))
    assert np.max(np.abs(contraction[delta])) > 1e-8

    # These are the unit map's theta derivatives, before any original-z chart.
    unit_j = np.asarray(jax.jacfwd(convert)(anchor))
    unit_h = np.asarray(jax.jacfwd(jax.jacfwd(convert))(anchor))
    logarithm, factor = math.log(divisor), math.exp(-float(anchor[delta]) * math.log(divisor))
    np.testing.assert_allclose(
        [unit_j[omega, omega], unit_j[omega, delta],
         unit_h[omega, omega, delta], unit_h[omega, delta, omega], unit_h[omega, delta, delta]],
        [factor, -logarithm * internal_anchor[omega], -logarithm * factor,
         -logarithm * factor, logarithm**2 * internal_anchor[omega]], rtol=3e-14, atol=3e-14)
    # Freezing delta gives the same point value but the wrong derivatives.
    def frozen_delta_map(theta):
        return convert(theta).at[omega].set(theta[omega] * factor)
    np.testing.assert_allclose(np.asarray(frozen_delta_map(anchor)), internal_anchor, rtol=3e-14, atol=3e-14)
    frozen = scalar_jet(lambda coordinates: internal_objective(frozen_delta_map(point(coordinates))) + correction,
                        source)
    assert abs(complete[1][delta] - frozen[1][delta]) > 1e-8
    assert np.max(np.abs(complete[2][delta] - frozen[2][delta])) > 1e-8


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("divisor", (4.0, 0.25))
def test_aparch_fixed_raw_history_retains_seed_and_shock_delta_derivatives(density, divisor):
    declared = unit_case("aparch", density, "innovations", divisor=divisor)
    oracle, _ = owners(declared)
    _, jnp, _ = jax_modules()
    internal = internal_case(declared)
    theta = np.asarray(internal_theta_map(declared)(jnp.asarray(declared["parameters"])))
    internal["parameters"] = theta
    power, gradient, hessian = oracle.state_jet(internal, return_power=True)
    names = declared["theta_names"]
    delta = names.index("delta")
    slope, exponent = .5 * math.log(internal["h0"]), theta[delta]
    seed = math.exp(exponent * slope)
    np.testing.assert_allclose([power[0], gradient[0, delta], hessian[0, delta, delta]],
                               [seed, slope * seed, slope**2 * seed], rtol=3e-13, atol=3e-13)
    # At t=1 both q=2 persistence slots still contain the declared h0 power.
    for name in ("beta[1]", "beta[2]"):
        np.testing.assert_allclose(hessian[1, names.index(name), delta], slope * seed,
                                   rtol=3e-13, atol=3e-13)
    alpha, gamma = names.index("alpha[2]"), names.index("gamma[2]")
    raw = internal["innovation_history"][0]
    base = abs(raw) - theta[gamma] * raw
    expected = -theta[alpha] * raw * base**(exponent - 1) * (1 + exponent * math.log(base))
    assert abs(expected) > 1e-8
    np.testing.assert_allclose(hessian[1, gamma, delta], expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("volatility", VOLATILITIES)
@pytest.mark.parametrize("mean", ("arx", "harx"))
@pytest.mark.parametrize("intercept", (False, True))
@pytest.mark.parametrize("divisor", (4.0, 0.25))
def test_causal_unit_roles_preserve_histories_and_generated_coefficients(volatility, mean, intercept, divisor):
    declared = unit_case(f"{mean}_{volatility}", "ged", "both", divisor=divisor, causal_intercept=intercept)
    fitting, internal = fitting_case(declared), internal_case(declared)
    jax, jnp, _ = jax_modules()
    theta = jnp.asarray(declared["parameters"])
    convert = internal_theta_map(declared)
    mapped = np.asarray(convert(theta))
    start, generated = int(intercept), 2
    width = declared["theta_names"].index("omega")
    factors = design_divisors(declared)
    np.testing.assert_array_equal(mapped[start:start + generated], np.asarray(theta[start:start + generated]))
    np.testing.assert_allclose(mapped[start + generated:width],
                               np.asarray(theta[start + generated:width]) * factors[generated:] / divisor,
                               rtol=2e-15, atol=0)
    if intercept:
        assert mapped[0] == float(theta[0]) / divisor
    jacobian = np.asarray(jax.jacfwd(convert)(theta))
    np.testing.assert_array_equal(jacobian[start:start + generated], np.eye(len(theta))[start:start + generated])
    np.testing.assert_array_equal(np.asarray(jax.jacfwd(jax.jacfwd(convert))(theta))[:width],
                                  np.zeros((width, len(theta), len(theta))))
    np.testing.assert_allclose(internal["design"], fitting["design"] / factors, rtol=2e-15, atol=0)
    for field in ("observation_history", "innovation_history"):
        np.testing.assert_array_equal(internal[field], np.asarray(declared[field]) * 2 / divisor)
    assert internal["h0"] == declared["h0"] * 4 / divisor**2
    for field in ("recursion_begin", "hold_back", "first", "stop", "scored_rows"):
        np.testing.assert_array_equal(internal[field], declared[field])
