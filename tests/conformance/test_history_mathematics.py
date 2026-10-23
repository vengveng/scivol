"""Unexecuted history conformance using the established oracle and assertion owners."""

from copy import deepcopy
import importlib

import numpy as np
import pytest

from fixtures.history_cases import (
    DENSITIES, HISTORICAL_MEANX_CASES, MODELS, as_chart, case, declarations,
    historical_meanx_case, modes,
)
from fixtures.profile_arx_cases import causal_design
from oracles.garch import jax_modules
from oracles.mean import numpy_residuals
from oracles.power import absolute_moment


def owners(declared):
    model = declared["model"]
    family = ("garch" if model in ("garch", "gjr") else "power" if model in ("egarch", "aparch")
              else "mean" if model in ("arma", "linear") else "joint"
              if model.endswith(("_garch", "_gjr")) else "joint_power")
    return importlib.import_module("oracles." + family), importlib.import_module("test_" + family + "_mathematics")


def supplied_mode(model):
    available = modes(model)
    return next(mode for mode in ("both", "innovations", "observations", "empty") if mode in available)


@pytest.mark.parametrize("model,density,mode", declarations() + declarations(causal=True))
def test_history_theta_all_masks(native, model, density, mode):
    declared = case(model, density, mode)
    oracle, checks = owners(declared)
    checks.exercise_requests(native, declared, oracle.full_jet(declared))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("density", DENSITIES)
def test_history_original_z_uses_original_point_and_map_curvature(native, model, density):
    declared = as_chart(case(model, density, supplied_mode(model)))
    oracle, checks = owners(declared)
    actual = native(declared, 7)
    checks.assert_success(actual, declared, 7)
    if model in ("garch", "gjr"):
        from oracles.charts import coordinate_map, pullback
    else:
        coordinate_map, pullback = oracle.coordinate_map, oracle.pullback
    _, jnp, _ = jax_modules()
    mapped = np.asarray(coordinate_map(declared)(jnp.asarray(declared["original_z"])))
    np.testing.assert_allclose(actual["theta"], mapped, rtol=3e-9, atol=3e-10)
    expected = (oracle.original_z_jet(declared, actual["theta"]) if model.endswith(("egarch", "aparch"))
                else pullback(declared, oracle.full_jet(declared, theta=actual["theta"])))
    checks.exercise_requests(native, declared, expected)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("density", DENSITIES)
def test_empty_histories_preserve_old_bridge_and_default_recipe(native, model, density):
    declared = case(model, density, "empty")
    legacy = deepcopy(declared)
    legacy.pop("observation_history")
    legacy.pop("innovation_history")
    before, after = native(legacy, 7), native(declared, 7)
    _, checks = owners(declared)
    checks.assert_success(before, legacy, 7)
    checks.assert_success(after, declared, 7)
    for field in ("f", "g", "h", "theta", "scored_rows", "reasons"):
        np.testing.assert_array_equal(after[field], before[field])


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("field", ("observation_history", "innovation_history"))
@pytest.mark.parametrize("length", (1, 4))
def test_partial_or_extra_history_is_not_silently_trimmed(native, density, field, length):
    declared = case("arma_garch", density, "both")
    declared[field] = tuple(.25 for _ in range(length))
    with pytest.raises(ValueError, match="Native preparation failed"):
        native(declared, 1)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", ("arma", "arma_garch", "arma_gjr", "arma_egarch", "arma_aparch"))
def test_common_orders_also_forward_supplied_histories(native, density, model):
    declared = case(model, density, "both", "common")
    oracle, checks = owners(declared)
    checks.exercise_requests(native, declared, oracle.full_jet(declared))


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", ("arma_garch", "arma_gjr", "arma_egarch", "arma_aparch"))
@pytest.mark.parametrize("layout", ("ma_long", "vol_long"))
def test_shared_innovation_history_serves_unequal_mean_and_volatility_lags(native, density, model, layout):
    declared = case(model, density, "both", layout)
    oracle, checks = owners(declared)
    checks.exercise_requests(native, declared, oracle.full_jet(declared))


@pytest.mark.parametrize("density", DENSITIES)
def test_nearest_first_ar_and_ma_boundary_uses_actual_next_row(density):
    declared = case("arma", density, "both")
    residuals = numpy_residuals(declared)
    theta, begin = declared["parameters"], declared["recursion_begin"]
    y = declared["y"][declared["hold_back"]:]
    yh, eh = declared["observation_history"], declared["innovation_history"]
    first = y[begin] - theta[0] - theta[1] * yh[0] - theta[2] * yh[1] - np.dot(theta[3:6], eh)
    second = y[begin + 1] - theta[0] - theta[1] * y[begin] - theta[2] * yh[0] - np.dot(theta[3:6], (first, eh[0], eh[1]))
    np.testing.assert_allclose(residuals[begin:begin + 2], (first, second), rtol=0, atol=2e-15)


@pytest.mark.parametrize("field", ("observation_history", "innovation_history"))
def test_history_forcing_is_not_misclassified_as_structural_zero(field):
    from fixtures.mean_cases import arma_case
    from oracles.mean import _fixed_zero_row
    declared = arma_case("ged", 1, 1, intercept=False, y=np.zeros(4))
    assert _fixed_zero_row(declared, 1)
    declared[field] = (1.0,)
    assert not _fixed_zero_row(declared, 1)


@pytest.mark.parametrize("model", ("garch", "gjr"))
def test_current_nu6_splice_report_survives_history_transport(native, model):
    from fixtures.garch_cases import case as garch_case
    # Exact fixture owned by test_skew_splice_public_status.cpp; no AD H at a kink.
    declared = garch_case(model, "skewt", 3, 2, y=[-3.], h0=19.)
    declared["parameters"][-2:] = (6., .5)
    conditioned = dict(declared, observation_history=(), innovation_history=(-.375, .625, -.875))
    for mask in range(1, 8):
        before, after = native(declared, mask), native(conditioned, mask)
        assert after["status"] == before["status"] == (6 if mask & 4 else 0)
        assert after["valid_mask"] == before["valid_mask"] == mask & 3
        np.testing.assert_array_equal(after["reasons"], before["reasons"])
        np.testing.assert_array_equal(after["scored_rows"], (0,))
        if mask & 1:
            assert after["f"] == pytest.approx(np.log(128. / 15.), rel=0, abs=2e-12)
        if mask & 2:
            np.testing.assert_allclose(after["g"][-2:], ((10. / 19.) * np.log(2.) - 92. / 285., -6. / 19.),
                                       rtol=0, atol=2e-12)
            np.testing.assert_array_equal(after["g"][:-2], np.zeros(len(after["g"]) - 2))
        if mask & 4:
            assert after["reasons"][2] == 6
        for bit, field in ((1, "f"), (2, "g"), (4, "h")):
            if not mask & bit:
                np.testing.assert_array_equal(after[field], np.full_like(after[field], after["sentinel"]))


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("model", ("garch", "gjr", "egarch", "aparch",
                                   "linear_garch", "linear_gjr", "linear_egarch", "linear_aparch"))
def test_fixed_row_zero_h0_survives_supplied_histories(density, model):
    declared = case(model, density, "innovations")
    oracle, _ = owners(declared)
    values, gradient, hessian = oracle.state_jet(declared)
    if model.startswith("linear_"):
        values, gradient, hessian = values[:, 1], gradient[:, 1], hessian[:, 1]
    expected = np.log(declared["h0"]) if model.endswith(("egarch", "aparch")) else declared["h0"]
    assert values[0] == expected
    np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))
    np.testing.assert_array_equal(hessian[0], np.zeros_like(hessian[0]))


@pytest.mark.parametrize("density", DENSITIES)
def test_aparch_seed_power_keeps_delta_derivatives(density):
    declared = case("aparch", density, "innovations")
    oracle, _ = owners(declared)
    value, gradient, hessian = oracle.state_jet(declared, return_power=True)
    delta_index = oracle.volatility_width(declared) - 1
    log_scale = .5 * np.log(declared["h0"])
    assert gradient[0, delta_index] == pytest.approx(log_scale * value[0], rel=2e-14)
    assert hessian[0, delta_index, delta_index] == pytest.approx(log_scale**2 * value[0], rel=2e-14)
    assert abs(gradient[1, 1 + declared["p"]]) > 1e-12


@pytest.mark.parametrize("density", DENSITIES)
def test_egarch_fixed_raw_shocks_do_not_freeze_density_moment(density):
    declared = case("egarch", density, "innovations")
    oracle, _ = owners(declared)
    _, gradient, hessian = oracle.state_jet(declared)
    jax, jnp, _ = jax_modules()
    width, p = oracle.volatility_width(declared), declared["p"]
    shape = jnp.asarray(declared["parameters"][width:])
    derivative = np.asarray(jax.jacfwd(lambda point: absolute_moment(point, density))(shape))
    np.testing.assert_allclose(gradient[1, width:], -sum(declared["parameters"][1:1 + p]) * derivative,
                               rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(hessian[1, 1, width:], -derivative, rtol=3e-11, atol=3e-12)


def test_har_history_has_literal_full_windows_and_data_takeover():
    y, external = np.asarray([7., 11., 13., 17.]), np.zeros((4, 1))
    actual = causal_design(y, external, 2, 0, observation_history=(3., -2., 5.), horizons=(1, 3))
    np.testing.assert_array_equal(actual[:, 0], (3., 7., 11., 13.))
    np.testing.assert_allclose(actual[:, 1], (2., 8 / 3, 7., 31 / 3), rtol=0, atol=2e-15)
    empty = causal_design(y, external, 2, 0, horizons=(1, 3))
    np.testing.assert_array_equal(empty[:3, 1], (0., 7., 9.))


@pytest.mark.parametrize("recipe,density", HISTORICAL_MEANX_CASES)
@pytest.mark.parametrize("causal", (False, True), ids=("historical-linear", "causal-retained"))
def test_historical_meanx_cases_all_masks(native, recipe, density, causal):
    declared = historical_meanx_case(recipe, density, causal=causal)
    offset = declared["historical_row_offset"]
    np.testing.assert_array_equal(declared["y"][offset:], declared["historical_y"])
    np.testing.assert_array_equal(declared["design"][offset:], declared["historical_features"][:, 1:])
    np.testing.assert_array_equal(np.asarray(declared["scored_rows"]) - offset,
                                  np.arange(len(declared["historical_y"])))
    assert declared["recursion_begin"] == 0
    assert declared["h0"] == float(np.mean(declared["historical_y"] ** 2))
    assert not declared["observation_history"] and not declared["innovation_history"]
    if causal:
        np.testing.assert_array_equal(declared["input_exogenous"][offset:], declared["historical_exogenous"])
        assert offset == (5 if declared["causal_kind"] == "harx" else 1)
    oracle, checks = owners(declared)
    expected = oracle.full_jet(declared)
    actual = checks.exercise_requests(native, declared, expected)
    # Keep the old numerical assertions as well as the stricter shared checks.
    for field, reference, (rtol, atol) in zip(("f", "g", "h"), expected, declared["historical_tolerances"]):
        np.testing.assert_allclose(actual[field], reference, rtol=rtol, atol=atol)
