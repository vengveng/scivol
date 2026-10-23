"""Pointwise sparse-request regression under the installed compiler arithmetic."""

import json
from pathlib import Path

import numpy as np
import pytest

from scivol import _native_core
from scivol._fit_credibility import jet_agreement


@pytest.fixture(scope="module")
def retained_point():
    fixture = json.loads(Path(__file__).with_name("aparch_skewt_precision_point.json").read_text())
    for name in ("theta", "data", "features", "units"):
        fixture[name] = np.ascontiguousarray(fixture[name], dtype=np.float64)
    return fixture


def evaluate_point(point, mask, fixed, paths):
    theta, data, features = (point[name] for name in ("theta", "data", "features"))
    plan = (point["family"], point["density"], point["columns"],
            point["p"], point["q"], len(data), fixed)
    size, _ = _native_core._linear_joint_sizes(plan, mask, False, (0, 0, 0, 0))
    owner = np.empty(size + 63, dtype=np.uint8)
    address = (owner.ctypes.data + 63) & ~63
    value = np.full(1, np.nan)
    gradient = np.full(theta.size, np.nan)
    hessian = np.full((theta.size, theta.size), np.nan)
    residuals, variance = np.full_like(data, np.nan), np.full_like(data, np.nan)
    assert _native_core._linear_joint_jet(
        plan, theta.ctypes.data, data.ctypes.data, features.ctypes.data, point["h0"],
        mask, False, address, size, value.ctypes.data if mask & 1 else 0,
        gradient.ctypes.data if mask & 2 else 0, hessian.ctypes.data if mask & 4 else 0,
        residuals.ctypes.data if paths else 0, variance.ctypes.data if paths else 0)
    for bit, output in ((1, value), (2, gradient), (4, hessian)):
        assert np.isfinite(output).all() if mask & bit else np.isnan(output).all()
    if paths:
        assert np.isfinite(residuals).all() and np.isfinite(variance).all()
        assert np.all(variance > 0)
        assert 0 < np.min(np.abs(residuals)) < 1e-18
    return value[0], gradient, hessian


@pytest.mark.parametrize("fixed", (True, False))
@pytest.mark.parametrize("paths", (False, True))
def test_near_boundary_skewt_requests_preserve_gradient_consistency(retained_point, fixed, paths):
    point = retained_point
    baseline = evaluate_point(point, 7, fixed, paths)
    assert abs(baseline[0] - float(point["precise"][1]["nll"])) <= 1e-9
    gamma = point["columns"] + 1 + point["p"]
    # The independently differentiated score is cancellation-sensitive, not zero.
    assert baseline[1][gamma] == pytest.approx(
        float(point["precise"][1]["gamma_score"]), rel=1e-11, abs=0)
    for mask in range(1, 8):
        value, gradient, hessian = evaluate_point(point, mask, fixed, paths)
        if mask & 1:
            assert abs(value - baseline[0]) <= 1e-9
        if mask & 2:
            agreement = jet_agreement(baseline[0], baseline[1], baseline[0], gradient,
                                      point["units"], len(point["data"]), 1e-12)
            assert agreement["status"] == "consistent", (mask, agreement)
        if mask & 4:
            np.testing.assert_allclose(hessian, baseline[2], rtol=1e-12, atol=1e-9)


def test_near_boundary_skewt_fixed_and_runtime_agree(retained_point):
    fixed = evaluate_point(retained_point, 7, True, False)
    runtime = evaluate_point(retained_point, 7, False, False)
    assert abs(fixed[0] - runtime[0]) <= 1e-9
    for actual, expected in zip(fixed[1:], runtime[1:]):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)


def test_original_start_curvature_retry_recovers_validated_likelihood(retained_point):
    import time
    from benchmark_native_joint import comparison_sample
    from scivol import APARCH, HARX, SkewT
    from scivol._joint_mean_fit import JointMeanModel, assemble_joint_fit, solve_joint_with_retry

    data, x = comparison_sample(2785, True, "persistent")
    model = JointMeanModel(HARX([1, 3]) + APARCH(3, 2) + SkewT(), data, x)
    np.testing.assert_array_equal(model.data, retained_point["data"])
    np.testing.assert_array_equal(model.features, retained_point["features"])
    assert model.h0 == retained_point["h0"]
    initial = model.initialize()
    unchanged = initial.copy()
    fitted = solve_joint_with_retry(model, initial, accuracy=1e-12, maxiter=2000)
    np.testing.assert_array_equal(initial, unchanged)
    assert fitted.fun <= float(retained_point["precise"][1]["nll"]) + 1e-9
    assert model.assess(fitted.x, fitted.jac)[0]
    assert fitted.nit <= 2000 and fitted.route_budgets[0] == 1000
    retry = next(p for p in fitted.phases if p.get("purpose") == "same-start-metric-retry")
    assert retry["refinement_phases"][0]["curvature_scaling"]
    for field, request in (("nfev", "F"), ("njev", "G"), ("nhev", "H")):
        assert fitted[field] == fitted.jet_requests[request]
    theta = fitted.x.copy()
    result = assemble_joint_fit(model, fitted, False, 1e-12, 2000, time.perf_counter())
    np.testing.assert_array_equal(result.params, theta)
    assert result.log_likelihood == pytest.approx(-fitted.fun, abs=1e-9, rel=0)
    assert result.fit_info.jet_consistency["status"] == "consistent"
    if fitted.stationarity_bound > 1e-6:
        assert not result.converged
        assert result.fit_info.inference_status == "unavailable"
        assert result.warnings and result.cov_matrix is None and result.std_errors is None
