"""Persistence value/slack regressions at fixed points, without optimizer solves."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import json
import math
import time

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from scivol import (
    APARCH, ARX, HARX, GED, GeneralizedHyperbolic, Hyperbolic, Normal,
    NormalInverseGaussian, SkewT, StudentT, VarianceGamma, _native_core,
)
from scivol import _native_fit as native
from scivol._evaluation import _filter_aparch, simulate_spec
from scivol._joint_mean_fit import JointMeanModel, assemble_joint_fit
from scivol.result import APARCHParams, EstimationResult
from scivol.roles import Role


DENSITIES = (
    (0, Normal, []),
    (1, StudentT, [8.0]),
    (2, SkewT, [8.0, 0.12]),
    (3, GED, [1.6]),
    (4, GeneralizedHyperbolic, [0.1, 1.3, 0.4]),
    (5, NormalInverseGaussian, [0.1, 1.3]),
    (6, Hyperbolic, [0.1, 1.3]),
    (7, VarianceGamma, [0.1, 3.0]),
)
GH_OVERFLOW = np.array([
    0.1, np.nextafter(0.0, 1.0), 0.9999998781130509, 0.0,
    217.965990611158, 0.22917865877731938, 24.999999999999993,
    5.9999999985845776,
], dtype=np.float64)
GH_WEIGHTED_REFERENCE = 1.502524401994032e-14


def _ordinary_theta(p, q, shape):
    return np.ascontiguousarray(np.r_[
        0.08, np.full(p, 0.06 / p), np.linspace(-0.15, 0.2, p),
        np.full(q, 0.4 / q), 1.7, shape,
    ], dtype=np.float64)


def _near_one_theta():
    return np.array([0.1, 2.0**-54, 0.0, np.nextafter(1.0, 0.0), 2.0])


def _raw_moment(shape, density, gamma, delta):
    shape = np.ascontiguousarray(shape, dtype=np.float64)
    value = np.full(1, 12345.0, dtype=np.float64)
    status = _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, density, float(gamma), float(delta), 1,
        value.ctypes.data, 0, 0,
    )
    return status, value[0]


@pytest.mark.parametrize("density,density_class,shape", DENSITIES)
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_ordinary_persistence_matches_raw_moment_owner(density, density_class, shape, p, q):
    theta = _ordinary_theta(p, q, shape)
    saved = theta.copy()
    moments = []
    for gamma in theta[1+p:1+2*p]:
        status, moment = _raw_moment(shape, density, gamma, theta[1+2*p+q])
        assert status == 0 and np.isfinite(moment) and moment > 0.0
        moments.append(moment)
    expected = math.fsum([
        *(theta[1:1+p] * moments), *theta[1+2*p:1+2*p+q],
    ])
    evidence = _native_core._aparch_persistence(theta.ctypes.data, p, q, density)
    assert isinstance(evidence, tuple) and len(evidence) == 2
    value, slack = evidence
    assert value == pytest.approx(expected, rel=2e-12, abs=0.0)
    assert slack == pytest.approx(1.0-expected, rel=2e-12, abs=0.0)
    model = native._describe_model(APARCH(p, q) + density_class())
    assert native._aparch_persistence_evidence(model, theta) == evidence
    violation, feasible, reported = native._aparch_volatility_domain(model, theta)
    assert violation == 0.0 and feasible and reported == value
    np.testing.assert_array_equal(theta, saved)


@pytest.mark.parametrize("density,density_class,shape", DENSITIES)
def test_tiny_beta_value_is_not_reconstructed_from_slack(density, density_class, shape):
    theta = np.ascontiguousarray(np.r_[0.1, 0.0, 0.0, 2.0**-80, 2.0, shape])
    value, slack = _native_core._aparch_persistence(theta.ctypes.data, 1, 1, density)
    assert value == 2.0**-80 and slack == 1.0
    assert value != 1.0-slack
    model = native._describe_model(APARCH(1, 1) + density_class())
    assert native._aparch_persistence_evidence(model, theta) == (value, slack)


def test_near_one_retains_resolved_positive_slack():
    theta = _near_one_theta()
    status, moment = _raw_moment([], 0, 0.0, 2.0)
    assert status == 0
    assert np.nextafter(1.0, 0.0) <= moment <= np.nextafter(1.0, np.inf)
    value, slack = _native_core._aparch_persistence(theta.ctypes.data, 1, 1, 0)
    assert value == 1.0 and slack == 2.0**-54
    assert 1.0-value == 0.0 and slack > 0.0
    model = native._describe_model(APARCH(1, 1) + Normal())
    assert native._aparch_volatility_domain(model, theta) == (0.0, True, 1.0)


def test_python_evidence_delegates_contiguous_volatility_block(monkeypatch):
    spec = native._describe_model(APARCH(3, 2) + GeneralizedHyperbolic())
    volatility = _ordinary_theta(3, 2, [0.1, 1.3, 0.4])
    model = replace(spec, n_mean=3, n_params=3+volatility.size)
    storage = np.empty(2*model.n_params)
    theta = storage[::2]
    theta[:] = np.r_[0.2, 0.3, 0.4, volatility]
    saved = theta.copy()
    calls = []

    def persistence(pointer, p, q, density):
        actual = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)),
            shape=(volatility.size,),
        )
        np.testing.assert_array_equal(actual, volatility)
        calls.append((p, q, density))
        return 2.0**-80, 1.0

    monkeypatch.setattr(_native_core, "_aparch_persistence", persistence)
    monkeypatch.setattr(native, "_aparch_moment_jets", lambda *a, **k: pytest.fail("Raw-moment fallback"))
    assert native._aparch_persistence_evidence(model, theta) == (2.0**-80, 1.0)
    assert calls == [(3, 2, 4)]
    np.testing.assert_array_equal(theta, saved)


@pytest.mark.parametrize("slack,feasible,violation", (
    (2.0**-54, True, 0.0), (0.0, False, 0.0), (-0.125, False, 0.125),
))
def test_domain_uses_supplied_slack_and_preserves_three_tuple(monkeypatch, slack, feasible, violation):
    model = native._describe_model(APARCH(1, 1) + Normal())
    monkeypatch.setattr(native, "_aparch_persistence_evidence",
                        lambda *a: pytest.fail("Precomputed evidence was discarded"))
    actual = native._aparch_volatility_domain(
        model, _near_one_theta(), persistence_evidence=(1.0, slack),
    )
    assert actual == (violation, feasible, 1.0) and len(actual) == 3


@pytest.mark.parametrize("density,error", ((-1, OverflowError), (8, ValueError), (2**32, ValueError)))
def test_native_persistence_rejects_invalid_density_without_uint_wrap(density, error):
    theta = _near_one_theta()
    with pytest.raises(error):
        _native_core._aparch_persistence(theta.ctypes.data, 1, 1, density)


def test_native_persistence_rejects_null_pointer():
    with pytest.raises(RuntimeError, match="APARCH persistence rejected"):
        _native_core._aparch_persistence(0, 1, 1, 0)


@pytest.mark.parametrize("p,q,error", (
    (0, 1, RuntimeError), (1, 0, RuntimeError), (0, 0, RuntimeError),
    (-1, 1, OverflowError), (1, -1, OverflowError),
    (int(np.iinfo(np.uintp).max), 1, RuntimeError),
    (1, int(np.iinfo(np.uintp).max), RuntimeError),
))
def test_native_persistence_rejects_invalid_orders_before_reading_theta(p, q, error):
    theta = _near_one_theta()
    with pytest.raises(error):
        _native_core._aparch_persistence(theta.ctypes.data, p, q, 0)


@pytest.mark.parametrize("density,theta", (
    (0, [0.1, 0.0, 1.0, 0.4, 2.0]),
    (0, [0.1, 0.0, 0.0, 0.4, 0.0]),
    (0, [0.1, 0.0, 0.0, 0.4, np.nan]),
    (1, [0.1, 0.0, 0.0, 0.4, 8.0, 8.0]),
))
def test_zero_alpha_does_not_hide_invalid_moment_domain(density, theta):
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    with pytest.raises(RuntimeError, match="APARCH persistence rejected"):
        _native_core._aparch_persistence(theta.ctypes.data, 1, 1, density)


def test_gh_raw_overflow_keeps_independently_checked_weighted_persistence():
    theta = GH_OVERFLOW.copy()
    status, raw = _raw_moment(theta[5:], 4, theta[2], theta[4])
    assert status != 0 and raw == 12345.0
    evidence = _native_core._aparch_persistence(theta.ctypes.data, 1, 1, 4)
    value, slack = evidence
    assert value == pytest.approx(GH_WEIGHTED_REFERENCE, rel=2e-9, abs=0.0)
    assert np.isfinite(value) and value > 0.0 and 0.0 < slack <= 1.0
    model = native._describe_model(APARCH(1, 1) + GeneralizedHyperbolic())
    assert native._aparch_persistence_evidence(model, theta) == evidence
    assert native._aparch_volatility_domain(model, theta) == (0.0, True, value)
    np.testing.assert_array_equal(theta, GH_OVERFLOW)


@pytest.mark.parametrize("fixed", (False, True))
def test_gh_constraint_value_recovery_does_not_invent_raw_alpha_derivative(fixed):
    data = np.linspace(0.17, 0.63, 32)
    x = np.c_[np.sin(np.arange(32)), np.cos(np.arange(32))]
    model = JointMeanModel(ARX(1) + APARCH(1, 1) + GeneralizedHyperbolic(), data, x, fixed=fixed)
    theta = np.ascontiguousarray(np.r_[np.zeros(model.columns), GH_OVERFLOW])
    saved = theta.copy()
    values, absent = model.constraints(theta, derivatives=False)
    assert absent is None and np.isfinite(values).all()
    _, slack = native._aparch_persistence_evidence(model.vol, theta[model.columns:])
    assert values[2*model.domain[0]] == slack and slack > 0.0
    with pytest.raises(ValueError, match="Joint domain evaluation failed"):
        model.constraints(theta)
    with pytest.raises(RuntimeError, match="stationarity moment rejected"):
        native._aparch_moment_jets(model.vol, theta[model.columns:], gradient=True)
    np.testing.assert_array_equal(theta, saved)


@pytest.mark.parametrize("persistence,slack,stationary", (
    (1.0, 2.0**-54, True), (1.0, 0.0, False), (0.9, -0.1, False),
    (0.9, None, True), (1.0, None, False),
))
def test_aparch_params_optional_slack_controls_stationarity(persistence, slack, stationary):
    params = APARCHParams(0.1, np.array([0.1]), np.array([0.4]), np.array([0.0]), 2.0,
                          density_adjusted_persistence=persistence, stationarity_slack=slack)
    assert params.persistence == persistence and params.is_stationary is stationary


def test_near_one_filter_snapshot_preserves_warnings_and_unavailable_inference(capsys):
    spec, theta = APARCH(1, 1) + Normal(), _near_one_theta()
    volatility = spec.get_component(Role.VOLATILITY)
    volatility.unpack(theta)
    spec.get_component(Role.DENSITY).unpack(np.empty(0))
    data = np.linspace(0.1, 0.3, 16)
    state = _filter_aparch(spec, data, theta, compute_score=False, compute_hessian=False)
    assert volatility.persistence() == 1.0 and volatility.is_stationary()
    assert volatility.fitted_params["stationarity_slack"] == 2.0**-54
    warning = "Finite feasible estimate; convergence is not certified. Parameters are unchanged."
    point = OptimizeResult(x=theta.copy(), fun=-state.log_likelihood, success=False,
                           status=9, nit=0, message=warning)
    fit = EstimationResult(spec, point, data, sigma2=state.sigma2, fit_info=dict(
        optimizer_terminated=False, stationarity_status="uncertified",
        inference_status="unavailable", warnings=(warning,),
    ))
    assert fit.garch_params.persistence == 1.0 and fit.garch_params.is_stationary
    assert fit.garch_params.stationarity_slack == 2.0**-54
    assert not fit.converged and not fit.fit_info.optimizer_terminated
    assert fit.fit_info.stationarity_status == "uncertified"
    assert fit.fit_info.inference_status == "unavailable" and fit.std_errors is None
    assert fit.fit_info.warnings == (warning,)
    assert warning in fit.warnings and "APARCH persistence is very close to 1" in fit.warnings
    assert "Non-stationary APARCH persistence detected" not in fit.warnings
    exported = fit.to_dict()
    json.dumps(exported)
    assert exported["garch_params"]["stationarity_slack"] == 2.0**-54
    assert exported["parameters"][volatility.signature]["stationarity_slack"] == 2.0**-54
    assert exported["garch_params"]["persistence"] == 1.0
    assert exported["fit_info"]["warnings"] == [warning]
    assert exported["fit_info"]["inference_status"] == "unavailable"
    warnings_before = fit.warnings
    volatility.fitted_params.update(persistence=0.5, stationarity_slack=-0.5)
    assert not volatility.is_stationary()
    exported["parameters"][volatility.signature]["stationarity_slack"] = -1.0
    assert fit.garch_params.is_stationary and fit.garch_params.stationarity_slack == 2.0**-54
    assert fit.warnings == warnings_before
    np.testing.assert_array_equal(fit.params, theta)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mean", (ARX(1), HARX([1, 3])), ids=("arx", "harx"))
def test_joint_assembly_exports_both_near_one_evidence_values_without_certification(mean):
    data = np.linspace(0.17, 0.63, 32)
    x = np.c_[np.sin(np.arange(32)), np.cos(np.arange(32))]
    model = JointMeanModel(mean + APARCH(1, 1) + Normal(), data, x)
    theta = np.ascontiguousarray(np.r_[np.zeros(model.columns), _near_one_theta()])
    value = model.evaluate(theta, 1)[0]
    point = OptimizeResult(x=theta.copy(), fun=value, success=False, status=9, nit=0,
                           phases=(), selected_space="theta", stationarity_bound=np.inf)
    fit = assemble_joint_fit(model, point, False, 1e-12, 1, time.perf_counter())
    assert fit.garch_params.persistence == 1.0
    assert fit.garch_params.stationarity_slack == 2.0**-54 and fit.garch_params.is_stationary
    assert model.volatility.is_stationary()
    assert not fit.converged and not fit.fit_info.optimizer_terminated
    assert fit.fit_info.stationarity_status == "uncertified"
    assert fit.fit_info.inference_status == "unavailable" and fit.std_errors is None
    assert fit.fit_info.warnings
    exported = fit.to_dict()
    assert exported["garch_params"]["persistence"] == 1.0
    assert exported["garch_params"]["stationarity_slack"] == 2.0**-54
    np.testing.assert_array_equal(fit.params, theta)


def test_near_one_simulation_uses_resolved_slack_for_initial_power():
    theta = _near_one_theta()
    simulation = simulate_spec(APARCH(1, 1) + Normal(), 4, theta, burn=0, seed=314)
    assert simulation.sigma2[0] == theta[0] / (2.0**-54)
    assert np.isfinite(simulation.sigma2).all() and np.all(simulation.sigma2 > 0.0)
    assert np.isfinite(simulation.data).all()
