"""Mocked inference and status regressions for the native and canonical adapters."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from scivol import APARCH, GARCH, Normal
from scivol import _native_fit as native
from scivol._fit_credibility import observed_covariance
from scivol.result import FitMetadata


INFERENCE_WARNING = (
    "Standard errors are unavailable: convergence or positive, numerically resolved observed information is not certified."
)


@pytest.mark.parametrize("backend", ["native", "slsqp", "lbfgsb"])
@pytest.mark.parametrize(
    "case",
    [
        "uncertified-mode-zero",
        "stationary-nonzero-mode",
        "certified-positive-information",
        "indefinite-positive-inverse-diagonal",
        "unresolved-information",
        "unavailable-information",
        "infeasible",
        "constraint-violation",
        "objective-disagreement",
        "nonfinite-gradient",
    ],
)
def test_fit_inference_uses_current_credibility_and_resolved_information(
    monkeypatch, capsys, recwarn, backend, case
):
    spec = APARCH(1, 1) + Normal()
    model = native._describe_model(spec)
    data = np.array([1.0, -0.5, 0.25, -1.25])
    initial = np.arange(model.n_params, dtype=np.float64) + 1.0
    returned = initial + 0.125
    nll = 123.5
    gradient = np.zeros(model.n_params)
    hessian = np.diag(np.arange(model.n_params, dtype=np.float64) + 1.0)
    mode, residual, violation, feasible, parity_error = 0, 0.0, 0.0, True, 0.0
    if case == "uncertified-mode-zero":
        residual = 0.25
    elif case == "stationary-nonzero-mode":
        mode = 9
    elif case == "indefinite-positive-inverse-diagonal":
        hessian[:2, :2] = np.linalg.inv(np.array([[1.0, 2.0], [2.0, 1.0]]))
        assert np.all(np.linalg.inv(hessian).diagonal() > 0.0)
        assert np.linalg.eigvalsh(hessian)[0] < 0.0
    elif case == "unresolved-information":
        correlation = np.nextafter(1.0, 0.0)
        hessian[:2, :2] = [[1.0, correlation], [correlation, 1.0]]
        assert np.all(np.linalg.inv(hessian).diagonal() > 0.0)
    elif case == "unavailable-information":
        hessian = None
    elif case == "infeasible":
        feasible = False
    elif case == "constraint-violation":
        violation = 0.25
    elif case == "objective-disagreement":
        parity_error = 1.0
    elif case == "nonfinite-gradient":
        gradient[0] = np.nan

    stationary = case not in {
        "uncertified-mode-zero", "infeasible", "constraint-violation",
        "objective-disagreement", "nonfinite-gradient",
    }
    credible = mode == 0 and stationary
    inference_available = case == "certified-positive-information"
    assessment = native.ThetaAssessment(violation, residual, feasible, False)
    state = SimpleNamespace(sigma2=np.ones_like(data), residuals=data.copy())
    monkeypatch.setattr(native, "_initial_theta", lambda *args: (initial.copy(), True))
    monkeypatch.setattr(native, "_theta_to_coordinates", lambda model, theta: theta.copy())
    monkeypatch.setattr(native, "_coordinates_to_theta", lambda model, point: point.copy())
    monkeypatch.setattr(native, "_coordinate_bounds", lambda model: (
        np.full(model.n_params, np.nan), np.full(model.n_params, np.nan)))
    monkeypatch.setattr(native, "_assess_theta_solution", lambda *args, **kwargs: assessment)
    monkeypatch.setattr(native, "_boundary_warning", lambda *args: None)
    monkeypatch.setattr(native, "_cross_check_canonical_filter", lambda *args: (state, parity_error))

    def evaluate_solution(model, prepared_data, theta):
        np.testing.assert_array_equal(theta, returned)
        return nll, gradient.copy(), None if hessian is None else hessian.copy(), None

    def evaluate_candidate(model, prepared_data, theta, *, request):
        assert request == 2
        np.testing.assert_array_equal(theta, returned)
        return np.nan, gradient.copy(), None

    def unexpected_coordinate_request(*args, **kwargs):
        pytest.fail("Mocked fits must not call a native objective.")

    monkeypatch.setattr(native, "_evaluate_theta_solution", evaluate_solution)
    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate_candidate)
    monkeypatch.setattr(native, "_evaluate_coordinate_request", unexpected_coordinate_request)
    covariance_calls = []

    def observe_covariance(information, *, stationary):
        covariance_calls.append(stationary)
        return observed_covariance(information, stationary=stationary)

    monkeypatch.setattr(native, "observed_covariance", observe_covariance)
    calls = []

    if backend == "native":
        def solve_coordinates(model, prepared_data, coordinates, *, accuracy, maximum_iterations):
            assert accuracy == 1e-10 and maximum_iterations == 10
            np.testing.assert_array_equal(coordinates, initial)
            coordinates[:] = returned
            calls.append("native-z-slsqp")
            return native.NativeSolve(mode, nll / len(data), violation, residual,
                                      2, 3, 4, 0, 0, 0)

        def no_refinement(model, prepared_data, theta, *, accuracy, remaining_iterations):
            assert accuracy == 1e-10 and remaining_iterations == 8
            np.testing.assert_array_equal(theta, returned)
            return None

        monkeypatch.setattr(native, "_solve_coordinates", solve_coordinates)
        monkeypatch.setattr(native, "_refine_garch_theta_if_needed", no_refinement)
        if case == "infeasible":
            with pytest.raises(RuntimeError, match="final native parameter point is infeasible"):
                native.fit_native_spec(spec, data, maxiter=10)
            assert covariance_calls == []
            assert not recwarn
            assert capsys.readouterr() == ("", "")
            return
        result = native.fit_native_spec(spec, data, maxiter=10)
    else:
        def minimize(fun, coordinates, *, method, jac, bounds, tol, options):
            assert method == ("SLSQP" if backend == "slsqp" else "L-BFGS-B")
            assert jac is True and bounds == [(None, None)] * model.n_params
            assert options["maxiter"] == 10 - 2 * len(calls)
            assert options["disp"] is False
            assert options["ftol"] == (1e-12 if backend == "slsqp" else 1e-15)
            assert tol == (1e-12 if backend == "slsqp" else None)
            if backend == "lbfgsb":
                assert options["gtol"] == 1e-12 and options["maxls"] == 50
                assert options["maxfun"] == 100 - 3 * len(calls)
            calls.append(f"scipy-z-{backend}")
            return OptimizeResult(x=returned.copy(), fun=nll / len(data), success=mode == 0,
                                  status=mode, nit=2, nfev=3, njev=4, message="Mock termination.")

        monkeypatch.setattr("scipy.optimize.minimize", minimize)
        if case == "infeasible":
            with pytest.raises(RuntimeError, match="final canonical parameter point is infeasible"):
                native.fit_scipy_canonical_spec(spec, data, solver=backend, maxiter=10, maxfev=100)
            assert covariance_calls == []
            assert not recwarn
            assert capsys.readouterr() == ("", "")
            return
        result = native.fit_scipy_canonical_spec(spec, data, solver=backend, maxiter=10, maxfev=100)

    assert covariance_calls == [credible]
    assert len(calls) == (3 if backend == "lbfgsb" and (
        mode != 0 or not feasible or violation > 1e-5 or residual > 1e-5
    ) else 1)
    np.testing.assert_array_equal(initial, np.arange(model.n_params) + 1.0)
    np.testing.assert_array_equal(result.params, returned)
    np.testing.assert_array_equal(result._resid, state.residuals)
    np.testing.assert_array_equal(result.sigma2, state.sigma2)
    assert result.log_likelihood == -nll
    assert result.converged == credible
    assert result.fit_info.optimizer_mode == mode
    assert result.fit_info.optimizer_terminated is (mode == 0)
    assert result.fit_info.stationarity_status == (
        "unavailable" if case == "nonfinite-gradient"
        else "certified" if stationary else "uncertified"
    )
    assert result.fit_info.inference_status == ("available" if inference_available else "unavailable")
    assert result.n_iter == 2 * len(calls)
    assert result.fit_info.function_evaluations == 3 * len(calls)
    assert result.fit_info.gradient_evaluations == 4 * len(calls)
    assert result.fit_info.optimizer_phase_modes == (mode,) * len(calls)
    expected_phases = (calls[0],) + (f"{calls[0]}-restart",) * (len(calls) - 1)
    assert result.fit_info.optimizer_phases == expected_phases
    assert result.fit_info.refinement_restarts == len(calls) - 1
    if inference_available:
        np.testing.assert_allclose(result.cov_matrix, np.linalg.inv(hessian))
        np.testing.assert_allclose(result.std_errors, np.sqrt(np.linalg.inv(hessian).diagonal()))
        assert INFERENCE_WARNING not in result.warnings
    else:
        assert result.cov_matrix is None and result.std_errors is None
        assert result.fit_info.warnings.count(INFERENCE_WARNING) == 1
        assert result.warnings.count(INFERENCE_WARNING) == 1
        assert result.warnings == result.warnings
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.optimizer_terminated == result.fit_info.optimizer_terminated
    assert restored.stationarity_status == result.fit_info.stationarity_status
    assert restored.inference_status == result.fit_info.inference_status
    assert restored.warnings == result.fit_info.warnings
    assert not recwarn
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.fixture
def final_endpoint(monkeypatch):
    spec = APARCH(1, 1) + Normal()
    model = native._describe_model(spec)
    data = np.array([1.0, -0.5, 0.25, -1.25])
    theta = np.array([0.125, 0.05, 0.1, 0.8, 2.0])
    state = SimpleNamespace(sigma2=np.ones_like(data), residuals=data.copy())
    endpoint = SimpleNamespace(
        spec=spec, model=model, data=data, theta=theta, state=state,
        nll=123.5, mode=-105, filter_calls=[],
    )
    monkeypatch.setattr(native, "_initial_theta", lambda *args: (theta.copy(), True))
    monkeypatch.setattr(native, "_theta_to_coordinates", lambda model, point: point.copy())
    monkeypatch.setattr(native, "_coordinates_to_theta", lambda model, point: point.copy())
    monkeypatch.setattr(native, "_coordinate_bounds", lambda model: (
        np.full(model.n_params, np.nan), np.full(model.n_params, np.nan)))
    monkeypatch.setattr(native, "_boundary_warning", lambda *args: None)
    monkeypatch.setattr(native, "_aparch_moment_jets", lambda *args, **kwargs: (
        np.ones(model.p_vol), np.zeros((model.p_vol, 2))))
    monkeypatch.setattr(native, "_aparch_kkt_residual", lambda *args: 0.0)
    monkeypatch.setattr(native, "_refine_garch_theta_if_needed", lambda *args, **kwargs: None)

    def canonical_filter(spec, prepared_data, point, nll):
        assert spec is endpoint.spec
        np.testing.assert_array_equal(prepared_data, data)
        np.testing.assert_array_equal(point, theta)
        assert nll == endpoint.nll
        endpoint.filter_calls.append(nll)
        return state, 0.0

    def solve_coordinates(model, prepared_data, point, **kwargs):
        point[:] = theta
        return native.NativeSolve(endpoint.mode, endpoint.nll / len(data), 0.0,
                                  0.0, 2, 3, 4, 0, 0, 0)

    def minimize(fun, point, **kwargs):
        return OptimizeResult(
            x=theta.copy(), fun=endpoint.nll / len(data), success=endpoint.mode == 0,
            status=endpoint.mode, nit=2, nfev=3, njev=4, message="Mock termination.",
        )

    def unexpected_coordinate_request(*args, **kwargs):
        pytest.fail("Mocked endpoint checks must not call a native objective.")

    monkeypatch.setattr(native, "_cross_check_canonical_filter", canonical_filter)
    monkeypatch.setattr(native, "_solve_coordinates", solve_coordinates)
    monkeypatch.setattr(native, "_evaluate_coordinate_request", unexpected_coordinate_request)
    monkeypatch.setattr("scipy.optimize.minimize", minimize)
    return endpoint


def fit_mocked_endpoint(endpoint, backend):
    if backend == "native":
        return native.fit_native_spec(endpoint.spec, endpoint.data, maxiter=10)
    return native.fit_scipy_canonical_spec(
        endpoint.spec, endpoint.data, solver=backend, maxiter=10, maxfev=100
    )


@pytest.mark.parametrize("failure", ["rejected", "nonfinite-value", "nonfinite-gradient", "wrong-shape"])
def test_final_fg_plan_recovers_separate_requests(monkeypatch, final_endpoint, failure):
    endpoint = final_endpoint
    gradient = np.arange(endpoint.model.n_params, dtype=np.float64)
    hessian = np.arange(endpoint.model.n_params ** 2, dtype=np.float64).reshape(
        endpoint.model.n_params, endpoint.model.n_params
    )
    requests = []

    def evaluate(model, data, theta, *, request):
        assert model is endpoint.model and data is endpoint.data and theta is endpoint.theta
        requests.append(request)
        if request == 3:
            if failure == "rejected":
                raise RuntimeError("Combined request failed with valid_mask=1.")
            return (
                np.nan if failure == "nonfinite-value" else endpoint.nll,
                np.full(model.n_params, np.nan) if failure == "nonfinite-gradient"
                else gradient[:-1] if failure == "wrong-shape" else gradient,
                None,
            )
        if request == 1:
            return endpoint.nll, np.full(model.n_params, np.nan), None
        if request == 2:
            return np.nan, gradient.copy(), None
        assert request == 4
        return np.nan, np.full(model.n_params, np.nan), hessian.copy()

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    value, actual_gradient, actual_hessian, warning = native._evaluate_theta_solution(
        endpoint.model, endpoint.data, endpoint.theta
    )
    assert requests == [3, 1, 2, 4]
    assert value == endpoint.nll and warning is None
    np.testing.assert_array_equal(actual_gradient, gradient)
    np.testing.assert_array_equal(actual_hessian, 0.5 * (hessian + hessian.T))


@pytest.mark.parametrize("backend", ["native", "slsqp", "lbfgsb"])
def test_recovered_fg_endpoint_keeps_certified_inference(monkeypatch, final_endpoint, backend):
    endpoint = final_endpoint
    endpoint.mode = 0
    requests = []

    def evaluate(model, data, theta, *, request):
        np.testing.assert_array_equal(theta, endpoint.theta)
        requests.append(request)
        if request == 3:
            raise RuntimeError("Combined request plan failed.")
        return (
            endpoint.nll if request == 1 else np.nan,
            np.zeros(model.n_params) if request == 2 else np.full(model.n_params, np.nan),
            np.eye(model.n_params) if request == 4 else None,
        )

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    result = fit_mocked_endpoint(endpoint, backend)
    assert requests == ([2] if backend != "native" else []) + [3, 1, 2, 4]
    assert result.converged
    assert result.fit_info.stationarity_status == "certified"
    assert result.fit_info.inference_status == "available"
    np.testing.assert_array_equal(result.std_errors, np.ones(endpoint.model.n_params))
    assert endpoint.filter_calls == [endpoint.nll]
    assert native._GRADIENT_UNAVAILABLE_WARNING not in result.warnings


@pytest.mark.parametrize("backend", ["native", "slsqp", "lbfgsb"])
@pytest.mark.parametrize("failure", ["rejected", "nonfinite"])
@pytest.mark.parametrize("mode", [0, -105])
def test_f_only_endpoint_is_retained_with_quiet_unavailable_inference(
    monkeypatch, capsys, recwarn, final_endpoint, backend, failure, mode
):
    endpoint = final_endpoint
    endpoint.mode = mode
    requests = []

    def evaluate(model, data, theta, *, request):
        np.testing.assert_array_equal(theta, endpoint.theta)
        requests.append(request)
        if request == 1:
            return endpoint.nll, np.full(model.n_params, np.nan), None
        if request == 3 or (request == 2 and failure == "rejected"):
            raise RuntimeError("The requested gradient is unavailable.")
        assert request == 2, "An unavailable gradient must not lead to Hessian inference."
        return np.nan, np.full(model.n_params, np.nan), None

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    result = fit_mocked_endpoint(endpoint, backend)
    assert requests[-3:] == [3, 1, 2] and 4 not in requests
    assert endpoint.filter_calls == [endpoint.nll]
    np.testing.assert_array_equal(result.params, endpoint.theta)
    np.testing.assert_array_equal(result.sigma2, endpoint.state.sigma2)
    np.testing.assert_array_equal(result._resid, endpoint.state.residuals)
    assert result.log_likelihood == -endpoint.nll
    assert not result.converged
    assert result.fit_info.optimizer_mode == mode
    assert result.fit_info.optimizer_terminated is (mode == 0)
    assert result.fit_info.maximum_constraint_violation == 0.0
    assert result.fit_info.stationarity_status == "unavailable"
    assert result.fit_info.kkt_residual is None
    assert result.fit_info.inference_status == "unavailable"
    assert result.hessian is None and result.cov_matrix is None and result.std_errors is None
    for warning in (native._GRADIENT_UNAVAILABLE_WARNING, INFERENCE_WARNING):
        assert result.warnings.count(warning) == 1
        assert result.fit_info.warnings.count(warning) == 1
    assert result.warnings == result.warnings
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.stationarity_status == "unavailable"
    assert restored.inference_status == "unavailable"
    assert restored.kkt_residual is None
    assert restored.warnings == result.fit_info.warnings
    assert not recwarn
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("backend", ["native", "slsqp", "lbfgsb"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "rejected"])
def test_unavailable_final_value_is_actual_fit_failure(
    monkeypatch, final_endpoint, backend, value
):
    endpoint = final_endpoint
    endpoint.mode = 0

    def evaluate(model, data, theta, *, request):
        if request == 2:
            return np.nan, np.zeros(model.n_params), None
        if request == 3 or value == "rejected":
            raise RuntimeError("Final value request is invalid.")
        assert request == 1
        return value, np.full(model.n_params, np.nan), None

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    with pytest.raises(RuntimeError, match="Final value request is invalid|final objective is nonfinite"):
        fit_mocked_endpoint(endpoint, backend)
    assert endpoint.filter_calls == []


@pytest.mark.parametrize("backend", ["native", "slsqp", "lbfgsb"])
@pytest.mark.parametrize("gradient_available", [True, False])
def test_infeasible_finite_endpoint_is_actual_fit_failure(
    monkeypatch, final_endpoint, backend, gradient_available
):
    endpoint = final_endpoint
    endpoint.theta[0] = -0.125

    def evaluate(model, data, theta, *, request):
        if request == 3 and not gradient_available:
            raise RuntimeError("Combined request failed.")
        return (
            endpoint.nll if request & 1 else np.nan,
            np.zeros(model.n_params) if gradient_available else np.full(model.n_params, np.nan),
            np.eye(model.n_params) if request == 4 else None,
        )

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    with pytest.raises(RuntimeError, match="final .* parameter point is infeasible"):
        fit_mocked_endpoint(endpoint, backend)


@pytest.mark.parametrize("theta", [[0.1, 0.05, 0.8], [0.0, 0.05, 0.8], [0.1, 0.5, 0.8]])
@pytest.mark.parametrize("gradient", [[np.nan] * 3, [np.inf] * 3, [], [[0.0] * 3]])
def test_feasibility_is_independent_of_gradient_availability(monkeypatch, theta, gradient):
    model = native._describe_model(GARCH(1, 1) + Normal())

    def unexpected_kkt(*args):
        pytest.fail("Unavailable gradients cannot support a KKT assessment.")

    monkeypatch.setattr(native._native_core, "_garch_projected_kkt", unexpected_kkt)
    assessment = native._assess_theta_solution(
        model, np.array(theta), np.array(gradient), np.ones(4), 1e-10,
        boundary_reached=True,
    )
    assert assessment.model_feasible is (theta[0] > 0.0 and sum(theta[1:]) < 1.0)
    assert assessment.maximum_violation == pytest.approx(max(0.0, sum(theta[1:]) - 1.0))
    assert np.isinf(assessment.kkt_residual)
    assert assessment.boundary_reached


def test_garch_refinement_does_not_require_unavailable_final_gradient(monkeypatch):
    model = native._describe_model(GARCH(1, 1) + Normal())
    theta = np.array([0.1, 0.05, 0.8])
    requests = []

    def evaluate(model, data, theta, *, request):
        requests.append(request)
        if request != 1:
            raise RuntimeError("Gradient request is unavailable.")
        return 123.5, np.full(model.n_params, np.nan), None

    monkeypatch.setattr(native, "_evaluate_theta_request", evaluate)
    assert native._refine_garch_theta_if_needed(
        model, np.ones(4), theta, accuracy=1e-10, remaining_iterations=8
    ) is None
    assert requests == [3, 1, 2]
