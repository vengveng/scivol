"""Regression guards for independent jet validation, not solver qualification."""

from __future__ import annotations

import ctypes
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scivol import _core, _kernels
from scivol._devtools import ad_oracle, derivcheck, diagnostic


@pytest.fixture(scope="module")
def jax_modules():
    pytest.importorskip("jax")
    return ad_oracle._jax_modules()


def _chart_case(name, jax, jnp, jsp):
    """Small independent chart equations, including non-default order layouts."""
    def persistence(logits):
        count = logits.size
        limit = 1.0 - 8.0 * (count + 1) * np.finfo(float).eps
        return limit * jax.nn.softmax(jnp.r_[logits, 0.0])[:-1]

    if name == "garch":
        coordinates = np.linspace(-0.4, 0.7, 7)

        def chart(z):
            return jnp.r_[z[0], jax.nn.softplus(z[1]), persistence(z[2:])]

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.native_coordinate_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, 3, 2, "normal",
                linear_mean_parameters=1,
            )
    elif name == "gjr":
        coordinates = np.linspace(-0.4, 0.7, 9)

        def chart(z):
            contributions = persistence(z[4:])
            alpha = 2.0 * contributions[:2]
            gamma = 2.0 * contributions[2:4] - alpha
            return jnp.r_[z[0], jnp.tanh(z[1]), -jnp.tanh(z[2]),
                          jax.nn.softplus(z[3]), alpha, gamma, contributions[4:]]

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.native_coordinate_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, 2, 1, "normal",
                p_ar=1, q_ma=1, leverage=True,
            )
    elif name == "egarch":
        coordinates = np.array([0.3, 0.08, 0.7])

        def chart(z):
            return jnp.r_[z[:2], jnp.tanh(z[2])]

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.native_egarch_coordinate_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, 0, 1, "normal",
                linear_mean_parameters=1,
            )
    elif name == "arma_egarch":
        coordinates = np.array([0.1, 0.2, -0.3, 0.08, 0.7])

        def chart(z):
            return jnp.array([z[0], jnp.tanh(z[1]), -jnp.tanh(z[2]),
                              z[3], jnp.tanh(z[4])])

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.native_arma_egarch_coordinate_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, 1, 1, 0, 1, "normal",
            )
    elif name == "aparch":
        coordinates = np.array([-0.4, -0.7, 0.3, 0.2, 0.8])

        def chart(z):
            gamma = jnp.tanh(z[2])
            delta = jax.nn.softplus(z[4])
            absolute_moment = jnp.exp(
                0.5 * delta * jnp.log(2.0)
                + jsp.special.gammaln(0.5 * (delta + 1.0))
                - 0.5 * jnp.log(jnp.pi)
            )
            moment = 0.5 * ((1.0 - gamma)**delta + (1.0 + gamma)**delta) * absolute_moment
            contributions = persistence(z[jnp.array([1, 3])])
            return jnp.array([jax.nn.softplus(z[0]), contributions[0] / moment,
                              gamma, contributions[1], delta])

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.native_aparch_coordinate_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, 1, 1, "normal",
            )
    elif name == "legacy_linear":
        coordinates = np.array([0.3, 0.08, 0.7])

        def chart(z):
            return jnp.array([1.7 * jnp.tanh(z[0]), z[1], jnp.tanh(z[2])])

        def evaluate(theta, value, gradient, hessian):
            return ad_oracle.linear_mean_logspace_chain_rule_value_grad_hess(
                coordinates, theta, value, gradient, hessian, np.array([1.7]),
                0, 1, model="egarch", dist="normal",
            )
    else:
        raise AssertionError(f"Unknown chart case: {name}")
    return coordinates, chart, evaluate


@pytest.mark.parametrize("name", ["garch", "gjr", "egarch", "arma_egarch", "aparch", "legacy_linear"])
@pytest.mark.parametrize("one_ulp_anchor", [False, True])
def test_chart_oracles_anchor_the_supplied_theta_jet(name, one_ulp_anchor, jax_modules):
    jax, jnp, jsp = jax_modules
    coordinates, chart, evaluate = _chart_case(name, jax, jnp, jsp)
    point = jnp.asarray(coordinates)
    theta = np.asarray(chart(point)).copy()
    if one_ulp_anchor:
        theta = np.nextafter(theta, np.inf)
    gradient = np.linspace(0.25, 1.0, theta.size)
    hessian = np.eye(theta.size) + np.outer(gradient, gradient)
    hessian[-1, -1] = float(2**54)
    jacobian = np.asarray(jax.jacfwd(chart)(point))
    map_hessians = np.asarray(jax.jacfwd(jax.jacrev(chart))(point))
    expected_gradient = jacobian.T @ gradient
    expected_hessian = (
        jacobian.T @ hessian @ jacobian
        + np.einsum("i,ijk->jk", gradient, map_hessians)
    )

    value, actual_gradient, actual_hessian = evaluate(theta, 2.0, gradient, hessian)

    assert value == 2.0
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=2e-14, atol=1e-14)


def test_one_ulp_anchor_cannot_cancel_the_transformed_score(jax_modules):
    _, jnp, _ = jax_modules
    coordinates = np.array([0.08, 0.7])
    mapped_beta = float(jnp.tanh(coordinates[1]))
    theta = np.array([coordinates[0], np.nextafter(mapped_beta, np.inf)])
    slope = 1.0 - mapped_beta * mapped_beta
    gradient = np.array([0.25, 1.0])
    hessian = np.diag([1.0, float(2**54)])
    unanchored_score = slope * (gradient[1] + hessian[1, 1] * (mapped_beta - theta[1]))
    assert unanchored_score < 0.0 < slope

    value, actual_gradient, actual_hessian = ad_oracle.native_egarch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, 2.0, gradient, hessian, 0, 1, "normal",
    )

    assert value == 2.0
    np.testing.assert_allclose(actual_gradient, [0.25, slope], rtol=2e-14, atol=0.0)
    expected_hessian = np.diag([1.0, hessian[1, 1] * slope**2 - 2.0 * mapped_beta * slope])
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=2e-14, atol=0.0)


@pytest.mark.parametrize("coordinate", [0.7, 1.0, 2.0, 3.0, -18.7, 18.7])
def test_libc_tanh_mapped_point_controls(coordinate, jax_modules):
    jax, jnp, _ = jax_modules
    coordinates = np.array([0.08, coordinate])
    theta = np.array([coordinates[0], math.tanh(coordinate)])
    gradient = np.array([0.25, 1.0])
    hessian = np.diag([1.0, float(2**54)])
    slope = float(jax.grad(jnp.tanh)(coordinate))
    curvature = float(jax.grad(jax.grad(jnp.tanh))(coordinate))

    value, actual_gradient, actual_hessian = ad_oracle.native_egarch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, 2.0, gradient, hessian, 0, 1, "normal",
    )

    assert value == 2.0
    np.testing.assert_allclose(actual_gradient, [0.25, slope], rtol=2e-14, atol=0.0)
    expected_hessian = np.diag([1.0, hessian[1, 1] * slope**2 + curvature])
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=2e-14, atol=0.0)


def test_rounded_endpoint_keeps_original_coordinate_derivatives(jax_modules):
    jax, jnp, _ = jax_modules
    coordinates = np.array([-0.4, 0.2, -0.3, 0.1, 40.0])
    limit = 1.0 - 24.0 * np.finfo(float).eps
    theta = np.r_[float(jax.nn.softplus(coordinates[0])),
                  limit * np.asarray(jax.nn.softmax(jnp.r_[coordinates[1:3], 0.0]))[:-1],
                  0.995 * float(jnp.tanh(coordinates[3])), 50.0]
    gradient = np.zeros(5)
    gradient[-1] = 1.0
    exponential = np.exp(-coordinates[-1])
    width = 50.0 - ad_oracle.NATIVE_VG_LAMBDA_BOUNDS[0]
    expected_gradient = np.zeros(5)
    expected_gradient[-1] = width * exponential / (1.0 + exponential)**2
    expected_hessian = np.zeros((5, 5))
    expected_hessian[-1, -1] = -width * exponential * (1.0 - exponential) / (1.0 + exponential)**3

    value, actual_gradient, actual_hessian = ad_oracle.native_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, 7.0, gradient, np.zeros((5, 5)), 1, 1, "vg",
    )

    assert value == 7.0
    assert actual_gradient[-1] > 0.0
    assert actual_hessian[-1, -1] < 0.0
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=3e-14, atol=0.0)


@pytest.mark.parametrize("bad_shape", ["theta", "gradient", "hessian"])
def test_chart_rejects_inconsistent_supplied_jet_shapes(bad_shape, jax_modules):
    coordinates = np.array([0.08, 0.7])
    theta = np.array([0.08, math.tanh(0.7)])
    gradient = np.ones(2)
    hessian = np.eye(2)
    if bad_shape == "theta":
        theta = theta[:1]
    elif bad_shape == "gradient":
        gradient = gradient[:1]
    else:
        hessian = hessian[:1]
    with pytest.raises(ValueError, match="matching shapes"):
        ad_oracle.native_egarch_coordinate_chain_rule_value_grad_hess(
            coordinates, theta, 2.0, gradient, hessian, 0, 1, "normal",
        )


@pytest.fixture
def diagnostic_producers(monkeypatch):
    state = {
        "analytical_gradient": np.array([1.0, 2.0, 3.0]),
        "oracle_gradient": np.array([1.0, 2.0, 3.0]),
        "analytical_hessian": np.eye(3),
        "oracle_hessian": np.eye(3),
        "calls": [],
    }

    def get_routine(uid):
        state["calls"].append(("routine", uid))
        return SimpleNamespace(start=lambda data: np.ones(state["analytical_gradient"].size))

    def fill_output(name):
        def producer(theta, resid2, sigma2, output, n, *orders):
            state["calls"].append((name, orders))
            if state.get("reject_callback", False):
                raise AssertionError("Invalid layout reached a derivative callback")
            values = state[f"analytical_{name}"].ravel()
            target = np.ctypeslib.as_array((ctypes.c_double * values.size).from_address(output))
            target[:] = values
        return producer

    def oracle(theta, resid2, p, q, *, dist):
        state["calls"].append(("oracle", p, q, dist))
        return 0.0, state["oracle_gradient"].copy(), state["oracle_hessian"].copy()

    monkeypatch.setattr(_kernels, "get_routine", get_routine)
    monkeypatch.setattr(ad_oracle, "garch_value_grad_hess", oracle)
    monkeypatch.setattr(derivcheck, "garch_value_grad_hess", oracle)
    routine_core = SimpleNamespace()
    for density in ("normal", "studentt"):
        for order in ("11", "pq"):
            gradient_name = f"_garch_ll_grad_{order}_{density}"
            hessian_name = f"_garch_ll_hess_{order}_{density}"
            gradient_producer = fill_output("gradient")
            hessian_producer = fill_output("hessian")
            monkeypatch.setattr(_core, gradient_name, gradient_producer)
            monkeypatch.setattr(_core, hessian_name, hessian_producer)
            setattr(routine_core, gradient_name, gradient_producer)
            setattr(routine_core, hessian_name, hessian_producer)
    monkeypatch.setattr(derivcheck, "_c", routine_core)
    return state


class _Spec:
    def __init__(self, uid="GARCH(1,1)+Normal"):
        self.uid = uid

    def __str__(self):
        return self.uid


@pytest.mark.parametrize("source", ["analytical", "oracle"])
@pytest.mark.parametrize("derivative", ["gradient", "hessian"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("all_entries", [False, True])
def test_nonfinite_derivatives_never_validate(source, derivative, nonfinite, all_entries, diagnostic_producers):
    state = diagnostic_producers
    values = state[f"{source}_{derivative}"]
    if all_entries:
        values[:] = nonfinite
    else:
        values.flat[0] = nonfinite

    report = diagnostic.validate_derivatives(_Spec(), np.array([1.0, 2.0, 3.0]))

    assert not report.passed
    assert not getattr(report, f"{derivative}_passed")
    assert getattr(report, f"{derivative}_max_abs_error") == np.inf
    label = "AD-oracle" if source == "oracle" else "analytical"
    assert f"non-finite {label} entries" in report.message
    other_label = "analytical" if source == "oracle" else "AD-oracle"
    assert f"non-finite {other_label} entries" not in report.message
    assert not diagnostic.quick_check(_Spec(), np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_identical_nonfinite_producers_do_not_establish_agreement(nonfinite, diagnostic_producers):
    for source in ("analytical", "oracle"):
        diagnostic_producers[f"{source}_gradient"][0] = nonfinite
        diagnostic_producers[f"{source}_hessian"][0, 0] = nonfinite
    report = diagnostic.validate_derivatives(_Spec(), np.array([1.0, 2.0, 3.0]))
    assert not report.passed
    assert not report.gradient_passed
    assert not report.hessian_passed
    assert "non-finite analytical entries" in report.message
    assert "non-finite AD-oracle entries" in report.message


@pytest.mark.parametrize("density", ["Normal", "StudentT"])
@pytest.mark.parametrize("orders", [(1, 1), (3, 2), (0, 1)])
def test_exact_supported_diagnostic_layouts(density, orders, diagnostic_producers):
    state = diagnostic_producers
    p, q = orders
    size = 1 + p + q + int(density == "StudentT")
    state["analytical_gradient"] = state["oracle_gradient"] = np.arange(1.0, size + 1.0)
    state["analytical_hessian"] = state["oracle_hessian"] = np.eye(size)

    report = diagnostic.validate_derivatives(_Spec(f"GARCH({p},{q})+{density}"), np.array([1.0, 2.0, 3.0]))

    assert report.passed
    assert ("oracle", p, q, density.lower()) in state["calls"]
    assert ("gradient", () if orders == (1, 1) else orders) in state["calls"]


UNSUPPORTED_VALIDATION_UIDS = [
    "GARCH(1,1)+NormalInverseGaussian", "GARCH(1,1)+SkewT", "GARCH(1,1)+GED",
    "GJR-GARCH(1,1)+Normal", "EGARCH(1,1)+Normal",
    "ARMA(1,1)+GARCH(1,1)+Normal", "GARCH(1,1)+Normal+StudentT",
]


@pytest.mark.parametrize("uid", UNSUPPORTED_VALIDATION_UIDS)
def test_unsupported_diagnostics_reject_before_dispatch(uid, diagnostic_producers):
    with pytest.raises(ValueError, match="supports only"):
        diagnostic.validate_derivatives(_Spec(uid), np.array([1.0, 2.0, 3.0]))
    assert not diagnostic_producers["calls"]


@pytest.mark.parametrize("params", [
    np.ones(1), np.ones(4), np.ones((1, 3)),
    np.array([0.1, np.nan, 0.8]), np.array([0.1, 0.1, np.inf]),
])
def test_invalid_parameter_buffers_reject_before_derivative_calls(params, diagnostic_producers):
    diagnostic_producers["reject_callback"] = True
    with pytest.raises(ValueError, match="finite 1-D array of length 3"):
        diagnostic.validate_derivatives(_Spec(), np.array([1.0, 2.0, 3.0]), params=params)
    assert all(call[0] == "routine" for call in diagnostic_producers["calls"])


def test_finite_derivative_disagreement_still_fails(diagnostic_producers):
    diagnostic_producers["oracle_gradient"][0] = 2.0
    diagnostic_producers["oracle_hessian"][0, 0] = 2.0
    report = diagnostic.validate_derivatives(_Spec(), np.array([1.0, 2.0, 3.0]))
    assert not report.passed
    assert not report.gradient_passed
    assert not report.hessian_passed


def _mock_validation_routine(uid, params, state):
    def start(data):
        state["calls"].append(("start",))
        return params.copy()

    return SimpleNamespace(uid=uid, start=start)


@pytest.mark.parametrize("uid", UNSUPPORTED_VALIDATION_UIDS)
def test_check_routine_rejects_unsupported_models_before_start(uid, diagnostic_producers):
    state = diagnostic_producers
    state["reject_callback"] = True
    routine = _mock_validation_routine(uid, np.ones(3), state)

    with pytest.raises(ValueError, match="supports only"):
        derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))

    assert not state["calls"]


@pytest.mark.parametrize("uid,size", [
    ("GARCH(1,1)+Normal", 3), ("GARCH(1,1)+StudentT", 4),
    ("GARCH(3,2)+Normal", 6),
])
@pytest.mark.parametrize("invalid", ["short", "long", "matrix", "nan", "inf"])
def test_check_routine_rejects_invalid_start_layout_before_callbacks(uid, size, invalid, diagnostic_producers):
    state = diagnostic_producers
    state["reject_callback"] = True
    if invalid == "short":
        params = np.ones(size - 1)
    elif invalid == "long":
        params = np.ones(size + 1)
    elif invalid == "matrix":
        params = np.ones((1, size))
    else:
        params = np.ones(size)
        params[-1] = np.nan if invalid == "nan" else np.inf
    routine = _mock_validation_routine(uid, params, state)

    with pytest.raises(ValueError, match=f"finite 1-D array of length {size}"):
        derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))

    assert state["calls"] == [("start",)]


@pytest.mark.parametrize("density", ["Normal", "StudentT"])
@pytest.mark.parametrize("orders", [(1, 1), (3, 2), (0, 1)])
def test_check_routine_preserves_supported_dispatch_and_cardinality(density, orders, diagnostic_producers, capsys):
    state = diagnostic_producers
    p, q = orders
    size = 1 + p + q + int(density == "StudentT")
    state["analytical_gradient"] = state["oracle_gradient"] = np.arange(1.0, size + 1.0)
    state["analytical_hessian"] = state["oracle_hessian"] = np.eye(size)
    routine = _mock_validation_routine(f"GARCH({p},{q})+{density}", np.ones(size), state)

    derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))

    assert ("oracle", p, q, density.lower()) in state["calls"]
    assert ("gradient", () if orders == (1, 1) else orders) in state["calls"]
    assert ("hessian", () if orders == (1, 1) else orders) in state["calls"]
    assert "All derivatives validated successfully" in capsys.readouterr().out


@pytest.fixture
def verbose_linalg_calls(monkeypatch):
    calls = []

    def condition(matrix):
        calls.append("cond")
        return 1.0

    def eigenvalues(matrix):
        calls.append("eigvalsh")
        return np.ones(matrix.shape[0])

    monkeypatch.setattr(np.linalg, "cond", condition)
    monkeypatch.setattr(np.linalg, "eigvalsh", eigenvalues)
    return calls


@pytest.mark.parametrize("source", ["analytical", "oracle"])
@pytest.mark.parametrize("derivative", ["gradient", "hessian"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_check_routine_rejects_partial_nonfinite_before_diagnostics(
    source, derivative, nonfinite, diagnostic_producers, verbose_linalg_calls,
):
    state = diagnostic_producers
    state[f"{source}_{derivative}"].flat[0] = nonfinite
    routine = _mock_validation_routine("GARCH(1,1)+Normal", np.ones(3), state)
    label = "analytical" if source == "analytical" else "AD-oracle"

    with np.errstate(invalid="raise", divide="raise", over="raise"):
        with pytest.raises(AssertionError, match=f"Non-finite {label} {derivative}"):
            derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))

    assert not verbose_linalg_calls


@pytest.mark.parametrize("derivative", ["gradient", "hessian"])
@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_check_routine_rejects_same_infinity_before_diagnostics(
    derivative, infinity, diagnostic_producers, verbose_linalg_calls,
):
    state = diagnostic_producers
    for source in ("analytical", "oracle"):
        state[f"{source}_{derivative}"].flat[0] = infinity
    assert np.allclose(state[f"analytical_{derivative}"], state[f"oracle_{derivative}"])
    routine = _mock_validation_routine("GARCH(1,1)+Normal", np.ones(3), state)

    with pytest.raises(AssertionError, match=f"Non-finite analytical {derivative}"):
        derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))

    assert not verbose_linalg_calls


@pytest.mark.parametrize("derivative", ["gradient", "hessian"])
def test_check_routine_finite_mismatch_uses_original_tolerances(derivative, diagnostic_producers):
    state = diagnostic_producers
    state[f"oracle_{derivative}"].flat[0] = 2.0
    routine = _mock_validation_routine("GARCH(1,1)+Normal", np.ones(3), state)
    label = "Gradient" if derivative == "gradient" else "Hessian"

    with pytest.raises(AssertionError, match=f"{label} mismatch"):
        derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))


def test_check_routine_keeps_finite_comparison_tolerances(diagnostic_producers):
    state = diagnostic_producers
    state["oracle_gradient"][0] += 5e-9
    state["oracle_hessian"][0, 0] += 5e-9
    routine = _mock_validation_routine("GARCH(1,1)+Normal", np.ones(3), state)

    derivcheck.check_routine(routine, np.array([1.0, 2.0, 3.0]))
