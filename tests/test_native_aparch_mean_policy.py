"""Controller contracts for the gated conditional ARMA-APARCH mean phase."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scivol import APARCH, ARMA, StudentT
from scivol import _native_fit as native
from scivol._mean_fit import StabilityBlock
from scivol.result import FitMetadata


ACCURACY = 1e-10
INFERENCE_WARNING = (
    "Standard errors are unavailable: convergence or positive, numerically "
    "resolved observed information is not certified."
)


def _pointer_array(address, count):
    return np.ctypeslib.as_array((ctypes.c_double * count).from_address(address))


@pytest.fixture
def policy(monkeypatch):
    spec = ARMA(1, 1) + APARCH(1, 1) + StudentT()
    model = native._describe_model(spec)
    theta = np.array([0.03, 0.2, 0.1, 0.08, 0.07, 0.15, 0.82, 1.7, 8.0])
    data = np.array([1.0, -0.5, 0.25, -1.25, 0.75, -0.125])
    value = 123.5
    gradient = np.ones(model.n_params)
    solve = native.NativeSolve(9, value / model.scored_observations(data.size),
                               0.0, 0.25, 4, 10, 5, 2, 1, 0)
    best = theta.copy()
    best[:model.n_mean] = [0.02, 0.22, 0.08]
    continuation = theta.copy()
    continuation[:model.n_mean] = [0.04, 0.21, 0.09]
    case = SimpleNamespace(
        spec=spec, model=model, theta=theta, data=data, value=value,
        gradient=gradient, solve=solve, best=best, continuation=continuation,
        best_value=122.5, continuation_value=123.0, actual_value=122.5,
        candidate_gradient=np.ones(model.n_params), evaluation="normal",
        hessian_unavailable=False,
        stop=0, phase_iterations=2, accepted=0, values=7, curvatures=2,
        workspace_size=256, workspace_calls=[], phase_calls=[], jet_calls=[],
    )

    # Keep domain and KKT assembly real; only the density-moment primitive is mocked.
    monkeypatch.setattr(native, "_aparch_moment_jets", lambda model, *args, **kwargs: (
        np.ones(model.p_vol), np.zeros((model.p_vol, 3))))

    def workspace_bytes(*arguments):
        case.workspace_calls.append(arguments)
        assert arguments == (
            case.model.p_ar, case.model.q_ma, case.model.p_vol,
            case.model.q_vol, case.model.density_code, False,
        )
        return case.workspace_size

    def run_phase(parameters, count, observations, n, h0, p_ar, q_ma, p, q,
                  density, maximum_steps, radius, workspace, workspace_size,
                  continuation, continuation_count, best, best_count, runtime):
        np.testing.assert_array_equal(_pointer_array(parameters, count), case.theta)
        np.testing.assert_array_equal(_pointer_array(observations, n), case.data)
        assert count == continuation_count == best_count == case.model.n_params
        assert (p_ar, q_ma, p, q, density, runtime) == (
            case.model.p_ar, case.model.q_ma, case.model.p_vol,
            case.model.q_vol, case.model.density_code, False,
        )
        assert h0 == native._initial_variance(case.data)
        assert workspace % 64 == 0 and workspace_size == case.workspace_size
        assert len({parameters, continuation, best}) == 3
        case.phase_calls.append((maximum_steps, radius))
        _pointer_array(continuation, continuation_count)[:] = case.continuation
        _pointer_array(best, best_count)[:] = case.best
        iterations = min(case.phase_iterations, maximum_steps)
        return (case.stop, case.best_value, case.continuation_value, iterations,
                case.accepted, case.values, case.curvatures, 11, 5, 3)

    def request(model, data, point, *, request):
        assert model == case.model
        np.testing.assert_array_equal(data, case.data)
        case.jet_calls.append((request, point.copy()))
        candidate = np.array_equal(point, case.best)
        assert candidate or np.array_equal(point, case.theta), (
            "The continuation point must not replace the retained best point."
        )
        if candidate:
            if case.evaluation == "unavailable":
                raise RuntimeError("candidate objective unavailable")
            if case.evaluation == "f-only" and request in (2, 3):
                raise RuntimeError("candidate exact gradient unavailable")
            if case.evaluation == "separate" and request == 3:
                raise RuntimeError("candidate fused request unavailable")
            value, gradient = case.actual_value, case.candidate_gradient
            if request == 3 and case.evaluation == "fused-nonfinite-gradient":
                gradient = np.full(model.n_params, np.nan)
            if request == 3 and case.evaluation == "fused-nonfinite-value":
                value = np.inf
        else:
            value, gradient = case.value, case.gradient
        if request == 4 and case.hessian_unavailable:
            raise RuntimeError("selected-point Hessian unavailable")
        return (value if request & 1 else np.nan,
                gradient.copy() if request & 2 else np.full(model.n_params, np.nan),
                np.eye(model.n_params) if request & 4 else None)

    monkeypatch.setattr(native._native_core, "_arma_aparch_mean_refinement_workspace_bytes",
                        workspace_bytes, raising=False)
    monkeypatch.setattr(native._native_core, "_arma_aparch_mean_refine",
                        run_phase, raising=False)
    monkeypatch.setattr(native, "_evaluate_theta_request", request)
    return case


def _refine(case, *, maximum_iterations=20, requests=None):
    if requests is None:
        requests = {}
    return native._refine_aparch_mean_if_needed(
        case.model, case.data, case.theta, case.solve, case.value, case.gradient,
        accuracy=ACCURACY, maximum_iterations=maximum_iterations, requests=requests,
    )


def _assert_accounting(case, combined, diagnostics):
    assert combined.iterations == case.solve.iterations + diagnostics["iterations"]
    assert combined.function_evaluations == (
        case.solve.function_evaluations + case.values + case.curvatures
    )
    assert combined.gradient_evaluations == case.solve.gradient_evaluations + case.curvatures
    assert combined.curvature_refinement_steps == (
        case.solve.curvature_refinement_steps + diagnostics["iterations"]
    )
    assert combined.refinement_restarts == case.solve.refinement_restarts
    assert combined.scientific_projection_steps == case.solve.scientific_projection_steps
    assert diagnostics["requests"] == {"F": case.values, "F|G|H": case.curvatures}
    assert diagnostics["certificate"] == "none"


def _assert_baseline(case, outcome):
    theta, combined, value, gradient, diagnostics = outcome
    np.testing.assert_array_equal(theta, case.theta)
    np.testing.assert_array_equal(gradient, case.gradient)
    assert value == case.value
    assert combined.mode == case.solve.mode
    assert combined.scaled_value == case.solve.scaled_value
    assert combined.kkt_residual == case.solve.kkt_residual
    assert combined.maximum_violation == case.solve.maximum_violation
    assert diagnostics is not None and not diagnostics["retained"]
    _assert_accounting(case, combined, diagnostics)


@pytest.mark.parametrize("reason", [
    "certified", "stationary-with-nonzero-mode", "nonjoint", "no-ar",
    "no-budget", "over-budget", "infeasible-baseline", "no-workspace",
])
def test_phase_skips_inapplicable_or_already_stationary_points(policy, reason):
    case = policy
    budget = 20
    if reason in {"certified", "stationary-with-nonzero-mode"}:
        case.gradient.fill(0.0)
        if reason == "certified":
            case.solve = replace(case.solve, mode=0, kkt_residual=0.0)
    elif reason == "nonjoint":
        case.model = replace(case.model, kind="aparch")
    elif reason == "no-ar":
        case.model = native._describe_model(ARMA(0, 1) + APARCH(1, 1) + StudentT())
    elif reason in {"no-budget", "over-budget"}:
        budget = case.solve.iterations - (reason == "over-budget")
    elif reason == "infeasible-baseline":
        case.theta[case.model.n_mean] = -0.08
    else:
        case.workspace_size = 0
    requests = {}
    theta, solve, value, gradient, diagnostics = _refine(
        case, maximum_iterations=budget, requests=requests)
    assert theta is case.theta and solve is case.solve and gradient is case.gradient
    assert value == case.value and diagnostics is None
    assert case.phase_calls == case.jet_calls == [] and requests == {}
    assert len(case.workspace_calls) == (reason == "no-workspace")


@pytest.mark.parametrize("remaining", [1, 3, 6, 20])
def test_phase_caps_iterations_to_remaining_joint_budget(policy, remaining):
    case = policy
    case.best_value = case.value
    case.phase_iterations = 100
    outcome = _refine(case, maximum_iterations=case.solve.iterations + remaining)
    _assert_baseline(case, outcome)
    assert len(case.phase_calls) == 1
    assert case.phase_calls[0][0] == min(6, remaining)
    assert outcome[4]["maximum_steps"] == min(6, remaining)
    assert outcome[1].iterations <= case.solve.iterations + remaining


@pytest.mark.parametrize("continuation_value", [123.0, 124.0, np.inf])
def test_best_point_is_retained_without_replacing_phase_continuation(policy, continuation_value):
    case = policy
    case.continuation_value = continuation_value
    baseline, continuation = case.theta.copy(), case.continuation.copy()
    requests = {}
    theta, solve, value, gradient, diagnostics = _refine(case, requests=requests)
    np.testing.assert_array_equal(theta, case.best)
    np.testing.assert_array_equal(gradient, case.candidate_gradient)
    np.testing.assert_array_equal(case.theta, baseline)
    np.testing.assert_array_equal(case.continuation, continuation)
    assert value == case.best_value and diagnostics["retained"]
    assert diagnostics["accepted_steps"] == 0
    assert diagnostics["continuation_value"] == continuation_value
    assert diagnostics["validation"] == "finite-feasible-improvement"
    assert solve.mode == -105 and np.isposinf(solve.kkt_residual)
    assert solve.scaled_value == value / case.model.scored_observations(case.data.size)
    assert requests == {"F|G": 1}
    assert [request for request, _ in case.jet_calls] == [3]
    _assert_accounting(case, solve, diagnostics)


@pytest.mark.parametrize("index", [3, 4, 5, 6, 7, 8])
def test_phase_rejects_any_changed_volatility_or_density_parameter(policy, index):
    policy.best[index] = np.nextafter(policy.theta[index], np.inf)
    outcome = _refine(policy)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "frozen-parameters-changed"
    assert policy.jet_calls == []


@pytest.mark.parametrize("invalid", ["equal", "worse", "nan", "inf", "negative-inf", "nan-theta", "inf-theta"])
def test_phase_rejects_invalid_or_nonimproving_raw_best_evidence(policy, invalid):
    if invalid.endswith("theta"):
        policy.best[0] = np.nan if invalid == "nan-theta" else np.inf
    else:
        policy.best_value = {
            "equal": policy.value, "worse": policy.value + 1.0,
            "nan": np.nan, "inf": np.inf, "negative-inf": -np.inf,
        }[invalid]
    outcome = _refine(policy)
    _assert_baseline(policy, outcome)
    assert policy.jet_calls == []


@pytest.mark.parametrize("candidate_value", [122.0, 123.0, 123.5, 124.0])
def test_phase_rejects_worse_or_nonmatching_rescored_value(policy, candidate_value):
    policy.actual_value = candidate_value
    requests = {}
    outcome = _refine(policy, requests=requests)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "rejected"
    assert requests == {"F|G": 1}


@pytest.mark.parametrize("error,retained", [(0.0, True), (1e-11, True), (1e-9, False)])
def test_candidate_value_agreement_uses_absolute_additive_likelihood_tolerance(policy, error, retained):
    policy.actual_value = policy.best_value + error
    outcome = _refine(policy)
    assert outcome[4]["retained"] is retained
    assert outcome[2] == (policy.actual_value if retained else policy.value)
    np.testing.assert_array_equal(outcome[0], policy.best if retained else policy.theta)
    _assert_accounting(policy, outcome[1], outcome[4])


@pytest.mark.parametrize("evaluation", ["separate", "fused-nonfinite-gradient", "fused-nonfinite-value"])
def test_candidate_fused_failure_can_recover_through_separate_f_and_g(policy, evaluation):
    policy.evaluation = evaluation
    requests = {}
    theta, solve, value, gradient, diagnostics = _refine(policy, requests=requests)
    np.testing.assert_array_equal(theta, policy.best)
    np.testing.assert_array_equal(gradient, policy.candidate_gradient)
    assert value == policy.actual_value and diagnostics["retained"]
    assert requests == {"F|G": 1, "F": 1, "G": 1}
    assert [request for request, _ in policy.jet_calls] == [3, 1, 2]
    _assert_accounting(policy, solve, diagnostics)


@pytest.mark.parametrize("candidate_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidate_value_preserves_baseline_and_phase_work(policy, candidate_value):
    policy.actual_value = candidate_value
    requests = {}
    outcome = _refine(policy, requests=requests)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "objective-unavailable"
    assert requests == {"F|G": 1, "F": 1}


def test_unavailable_candidate_objective_preserves_baseline_and_failed_requests(policy):
    policy.evaluation = "unavailable"
    requests = {}
    outcome = _refine(policy, requests=requests)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "objective-unavailable"
    assert "candidate objective unavailable" in outcome[4]["error"]
    assert requests == {"F|G": 1, "F": 1}


@pytest.mark.parametrize("index,coefficient", [(1, 1.0), (1, 1.1), (2, -1.0), (2, 1.1)])
def test_phase_rejects_infeasible_mean_before_requesting_candidate_jet(policy, index, coefficient):
    policy.best[index] = coefficient
    outcome = _refine(policy)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "infeasible-mean"
    assert policy.jet_calls == []


def test_phase_respects_independent_full_candidate_feasibility_assessment(policy, monkeypatch):
    assess = native._assess_theta_solution

    def assess_candidate(model, theta, *args, **kwargs):
        assessment = assess(model, theta, *args, **kwargs)
        return replace(assessment, model_feasible=False, maximum_violation=1.0) if (
            np.array_equal(theta, policy.best)
        ) else assessment

    monkeypatch.setattr(native, "_assess_theta_solution", assess_candidate)
    outcome = _refine(policy)
    _assert_baseline(policy, outcome)
    assert outcome[4]["validation"] == "rejected"


@pytest.mark.parametrize("stop", [-3, -2, -1, 0, 1])
def test_unproductive_phase_stop_preserves_baseline_and_counts(policy, stop):
    policy.stop = stop
    policy.best_value = np.inf
    policy.best.fill(np.nan)
    outcome = _refine(policy)
    _assert_baseline(policy, outcome)
    assert outcome[4]["stop"] == stop
    assert policy.jet_calls == []


def test_lower_finite_feasible_f_only_point_is_retained_with_unavailable_gradient(policy):
    policy.evaluation = "f-only"
    requests = {}
    theta, solve, value, gradient, diagnostics = _refine(policy, requests=requests)
    np.testing.assert_array_equal(theta, policy.best)
    assert value == policy.actual_value and np.isnan(gradient).all()
    assert diagnostics["retained"] and solve.mode == -105
    assert requests == {"F|G": 1, "F": 1, "G": 1}
    _assert_accounting(policy, solve, diagnostics)


@pytest.fixture
def fitting_policy(policy, monkeypatch):
    case = policy
    case.filter_calls = []
    case.inverse_calls = []
    monkeypatch.setattr(native, "_APARCH_MEAN_REFINEMENT_ENABLED", True)
    monkeypatch.setattr(native, "_initial_theta", lambda *args: (case.theta.copy(), True))
    monkeypatch.setattr(native, "_boundary_warning", lambda *args: None)

    def map_parameters(model, source, destination, *, forward):
        assert model == case.model
        if not forward:
            case.inverse_calls.append(source.copy())
        destination[:] = source

    def solve(model, data, coordinates, *, accuracy, maximum_iterations):
        assert model == case.model and accuracy == ACCURACY and maximum_iterations == 20
        np.testing.assert_array_equal(data, case.data)
        np.testing.assert_array_equal(coordinates, case.theta)
        return case.solve

    def filter_solution(spec, data, theta, value):
        assert spec is case.spec
        np.testing.assert_array_equal(data, case.data)
        case.filter_calls.append((theta.copy(), value))
        return SimpleNamespace(sigma2=np.ones_like(data), residuals=data.copy()), 0.0

    monkeypatch.setattr(native, "_map_parameters", map_parameters)
    monkeypatch.setattr(native, "_solve_coordinates", solve)
    monkeypatch.setattr(native, "_cross_check_canonical_filter", filter_solution)
    return case


@pytest.mark.parametrize("selection", [
    "best", "rejected", "failed-phase", "failed-objective", "f-only", "hessian-unavailable",
])
def test_fit_handoff_requests_final_hessian_only_at_selected_point(
    fitting_policy, selection, capsys, recwarn
):
    case = fitting_policy
    retained = selection in {"best", "f-only", "hessian-unavailable"}
    if selection == "rejected":
        case.actual_value = case.value + 1.0
    elif selection == "failed-phase":
        case.stop, case.best_value = -3, np.inf
    elif selection == "failed-objective":
        case.evaluation = "unavailable"
    elif selection == "f-only":
        case.evaluation = "f-only"
    elif selection == "hessian-unavailable":
        case.hessian_unavailable = True
    result = native.fit_native_spec(case.spec, case.data, maxiter=20)
    selected = case.best if retained else case.theta
    value = case.best_value if retained else case.value
    np.testing.assert_array_equal(result.params, selected)
    assert result.log_likelihood == -value
    assert len(case.filter_calls) == 1
    np.testing.assert_array_equal(case.filter_calls[0][0], selected)
    assert case.filter_calls[0][1] == value
    hessian_calls = [point for request, point in case.jet_calls if request & 4]
    assert len(hessian_calls) == (selection != "f-only")
    if hessian_calls:
        np.testing.assert_array_equal(hessian_calls[0], selected)
        assert case.jet_calls[-1][0] == 4
    expected = {"F|G": 1 if selection == "failed-phase" else 2}
    if selection in {"failed-objective", "f-only"}:
        expected["F"] = 1
    if selection == "f-only":
        expected["G"] = 1
    else:
        expected["H"] = 1
    assert result.fit_info.postfit_jet_requests == expected
    diagnostics, = result.fit_info.phase_diagnostics
    assert diagnostics["retained"] is retained
    assert diagnostics["certificate"] == "none"
    assert result.n_iter == case.solve.iterations + case.phase_iterations
    assert result.fit_info.function_evaluations == case.solve.function_evaluations + case.values + case.curvatures
    assert result.fit_info.gradient_evaluations == case.solve.gradient_evaluations + case.curvatures
    assert result.fit_info.curvature_refinement_steps == case.solve.curvature_refinement_steps + case.phase_iterations
    assert result.fit_info.optimizer_phases == ("native-z-slsqp", "native-theta-anchored-mean")
    assert result.fit_info.optimizer_phase_modes == (case.solve.mode, case.stop)
    assert result.fit_info.optimizer_mode == (-105 if retained else case.solve.mode)
    assert result.fit_info.coordinate_kkt_residual == case.solve.kkt_residual
    assert len(case.inverse_calls) == 1 + retained
    np.testing.assert_array_equal(case.inverse_calls[-1], selected)
    assert not result.converged and not result.fit_info.optimizer_terminated
    assert result.fit_info.inference_status == "unavailable"
    assert result.cov_matrix is None and result.std_errors is None
    assert INFERENCE_WARNING in result.warnings
    assert "conditional phase requests" in result.fit_info.request_count_scope
    if selection == "f-only":
        assert result.fit_info.stationarity_status == "unavailable"
        assert result.fit_info.kkt_residual is None
        assert native._GRADIENT_UNAVAILABLE_WARNING in result.warnings
    if selection == "hessian-unavailable":
        assert any("selected-point Hessian unavailable" in warning for warning in result.warnings)
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.postfit_jet_requests == expected
    assert tuple(restored.phase_diagnostics) == result.fit_info.phase_diagnostics
    assert not recwarn and capsys.readouterr() == ("", "")


@pytest.mark.parametrize("stop", [0, 1])
@pytest.mark.parametrize("joint_stationary", [False, True])
def test_conditional_stop_never_certifies_whole_joint_fit(fitting_policy, stop, joint_stationary):
    case = fitting_policy
    case.stop = stop
    case.solve = replace(case.solve, mode=0)
    if joint_stationary:
        case.candidate_gradient.fill(0.0)
    result = native.fit_native_spec(case.spec, case.data, maxiter=20)
    assert result.fit_info.optimizer_phase_modes == (0, stop)
    assert result.fit_info.optimizer_mode == -105
    assert not result.converged and not result.fit_info.optimizer_terminated
    assert result.fit_info.stationarity_status == (
        "certified" if joint_stationary else "uncertified"
    )
    assert result.fit_info.inference_status == "unavailable"
    assert result.fit_info.phase_diagnostics[0]["certificate"] == "none"


def test_disabled_gate_never_runs_automatic_mean_handoff(fitting_policy, monkeypatch):
    case = fitting_policy
    monkeypatch.setattr(native, "_APARCH_MEAN_REFINEMENT_ENABLED", False)
    result = native.fit_native_spec(case.spec, case.data, maxiter=20)
    assert case.workspace_calls == case.phase_calls == []
    np.testing.assert_array_equal(result.params, case.theta)
    assert [request for request, _ in case.jet_calls] == [3, 4]
    assert result.n_iter == case.solve.iterations
    assert result.fit_info.function_evaluations == case.solve.function_evaluations
    assert result.fit_info.gradient_evaluations == case.solve.gradient_evaluations
    assert result.fit_info.phase_diagnostics == ()
    assert result.fit_info.optimizer_phases == ("native-z-slsqp",)


@pytest.mark.parametrize("reason", ["certified", "no-budget"])
def test_enabled_gate_skips_phase_without_extra_final_hessian_requests(fitting_policy, reason):
    case = fitting_policy
    if reason == "certified":
        case.gradient.fill(0.0)
        case.solve = replace(case.solve, mode=0, kkt_residual=0.0)
    else:
        case.solve = replace(case.solve, iterations=20)
    result = native.fit_native_spec(case.spec, case.data, maxiter=20)
    assert case.workspace_calls == case.phase_calls == []
    assert [request for request, _ in case.jet_calls] == [3, 4]
    np.testing.assert_array_equal(result.params, case.theta)
    assert result.n_iter == case.solve.iterations
    assert result.fit_info.function_evaluations == case.solve.function_evaluations
    assert result.fit_info.gradient_evaluations == case.solve.gradient_evaluations
    assert result.fit_info.postfit_jet_requests == {"F|G": 1, "H": 1}
    assert result.fit_info.phase_diagnostics == ()
    assert result.fit_info.optimizer_phases == ("native-z-slsqp",)
    assert result.converged is (reason == "certified")


@pytest.mark.parametrize("ar,ma,feasible", [
    ([], [], True),
    ([0.2], [0.1], True),
    ([np.nextafter(1.0, 0.0)], [], True),
    ([1.0], [], False),
    ([1.1], [0.1], False),
    ([0.2], [-1.0], False),
    ([0.2], [1.1], False),
    ([1.2, -0.4], [0.3, 0.2], True),
    ([0.7, 0.5], [0.1, 0.1], False),
    ([0.2, 0.1], [-0.7, -0.5], False),
    ([0.2, 0.1, -0.05], [0.1, -0.05, 0.02], True),
    ([0.2, 0.1, -0.05], [0.7, -0.5, 0.1], False),
    ([np.nan], [0.1], False),
    ([0.2], [np.inf], False),
])
def test_mean_feasibility_uses_shared_ar_and_ma_polynomial_domains(ar, ma, feasible):
    model = native._describe_model(ARMA(len(ar), len(ma)) + APARCH(1, 1) + StudentT())
    theta = np.r_[0.03, ar, ma, 0.08, 0.07, 0.15, 0.82, 1.7, 8.0]
    before = theta.copy()
    shared_feasible = True
    for offset, coefficients, sign in ((1, ar, 1.0), (1 + len(ar), ma, -1.0)):
        if coefficients:
            block = StabilityBlock(offset, sign * np.eye(len(coefficients)))
            try:
                block.evaluate(theta, derivatives=False, stable=True)
            except ValueError:
                shared_feasible = False
    assert shared_feasible is feasible
    assert native._mean_parameters_are_feasible(model, theta) is feasible
    np.testing.assert_array_equal(theta, before)


def test_mean_feasibility_does_not_swallow_shared_owner_constructor_errors(monkeypatch):
    model = native._describe_model(ARMA(1, 1) + APARCH(1, 1) + StudentT())

    def invalid_constructor(*args, **kwargs):
        raise ValueError("invalid stability order programming error")

    monkeypatch.setattr("scivol._mean_fit.StabilityBlock", invalid_constructor)
    with pytest.raises(ValueError, match="programming error"):
        native._mean_parameters_are_feasible(model, np.zeros(model.n_params))
