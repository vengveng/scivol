"""APARCH phase/controller regressions against real scivol wiring."""

from dataclasses import replace

import numpy as np
import pytest

from scivol import APARCH, ARMA, Normal
from scivol import _aparch_fit_adapters as adapters
from scivol import _aparch_fit_policy as policy
from scivol import _native_fit as native

def make_point(value, *, theta=None, power=2.0, mode=9, certified=False,
               feasible=True, missing_gradient=False, coordinates=None, residual=0.1):
    theta = np.array([0.2, -0.0, power, 8.0]) if theta is None else theta
    gradient = np.full(theta.size, np.nan if missing_gradient else 0.0)
    solve = native.NativeSolve(mode, value, 0.0, residual, 1, 2, 1, 0, 0, 0)
    assessment = native.ThetaAssessment(0.0, 0.0 if certified else residual, feasible, False)
    return policy.PhasePoint(theta, value, gradient, solve, assessment,
                             coordinates=coordinates, certified=certified)


def outcome(point, *, continuation=True, work=None, mode=None, failure=None, evidence=None,
            equality_feasible=True):
    return policy.PhaseOutcome(
        points=(point,), continuation=point if continuation else None,
        work=work or policy.PhaseWork(1, 2, 1, 0, {"F|G": 1}),
        mode=point.solve.mode if mode is None else mode,
        failure=failure, evidence=evidence or {}, equality_feasible=equality_feasible,
    )


def run_policy(baseline, *, phases=None, released=None, initial=None, density=0, clock=None):
    return policy.run_aparch_controller(
        initial or policy.PhaseStart(np.array([0.1, -0.0, 1.5, 8.0]), np.array([0.3, -0.0, 0.2, 2.0])),
        baseline=baseline, released=released or (lambda request: outcome(make_point(20.0))),
        phases=phases or policy.ControllerPhases(), baseline_budget=10,
        density_code=density, power_index=2, accuracy=1e-10, clock=clock,
    )


def test_explicit_certified_fast_path_attempts_only_baseline():
    calls = []

    def unexpected(request):
        pytest.fail("Certified-only trigger must skip every optional preparation callback.")

    point = make_point(10.0, mode=0, certified=True)
    selected = run_policy(lambda request: calls.append(request) or outcome(point),
        phases=policy.ControllerPhases(power=unexpected,
            complementary=(("chart", unexpected),), qualified_power_densities=(0,),
            trigger_policy="uncertified-only"))
    assert len(calls) == len(selected.diagnostics) == 1
    assert selected.point is point and selected.total_work().validation_requests == {"F|G": 1}
    assert selected.trigger_policy == "uncertified-only"


def test_qualification_default_retains_better_smooth_uncertified_point_and_baseline_certificate():
    baseline = make_point(590.3597225201628, mode=0, certified=True)
    better = make_point(583.5475409676595, mode=-105, residual=2.654137288023802e12)
    phases = policy.ControllerPhases(complementary=(("chart", lambda request: outcome(better)),))
    selected = run_policy(lambda request: outcome(baseline), phases=phases)
    assert selected.point is better and not selected.point.certified
    assert selected.point.solve.mode == -105 and selected.trigger_policy == "always"
    assert len(selected.diagnostics) == 2
    assert selected.diagnostics[0]["points"][0]["certified"]
    assert not selected.diagnostics[1]["points"][0]["certified"]
    assert baseline.value - selected.point.value == 6.812181552503262


@pytest.mark.parametrize("density,mode", [(0, "fixed-coordinate-slsqp"),
                                          (1, "physical-equality-nu-free"),
                                          (2, "physical-equality-nu-free")])
def test_descending_physical_power_releases_all_parameters_and_preserves_original_z(density, mode):
    initial_theta = np.array([0.1, -0.0, 1.5, 8.0])
    initial_z = np.array([0.31, -0.0, 0.21, 2.01])
    initial = policy.PhaseStart(initial_theta, initial_z)
    stages, release_requests = [], []

    def power(request):
        index = len(stages)
        stages.append(request)
        assert request.power_mode == mode and request.accuracy == 1e-10
        assert request.free_parameters == ((0, 1, 3) if density == 0 else (0, 1, 2, 3))
        if index == 0:
            assert request.start.coordinates.tobytes() == initial_z.tobytes()
            assert request.start.theta.tobytes() == initial_theta.tobytes()
        else:
            assert request.start.coordinates.tobytes() == prior_z[index-1].tobytes()
        theta = np.array([0.11, -0.0, request.physical_power, 8.1 + index])
        z = np.array([0.32, -0.0, request.physical_power + 0.01, 2.1 + index])
        prior_z.append(z)
        return outcome(make_point(9.0-index, theta=theta, mode=-105, coordinates=z), mode=0)

    def release(request):
        release_requests.append(request)
        assert request.role == "released" and request.physical_power is None and request.power_mode is None
        assert request.free_parameters == (0, 1, 2, 3) and request.budget == 8
        assert request.start.coordinates.tobytes() == prior_z[-1].tobytes()
        return outcome(make_point(1.0, mode=9))

    prior_z = []
    selected = run_policy(lambda request: outcome(make_point(12.0)), initial=initial, density=density,
        released=release, phases=policy.ControllerPhases(power=power,
            qualified_power_densities=(density,), power_budget=3, released_budget=8))
    assert [request.physical_power for request in stages] == [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert len(release_requests) == 1 and selected.selected_phase == "native-released-joint"
    assert initial_z.tobytes() == initial.coordinates.tobytes()


def test_retained_trial_never_replaces_accepted_continuation_or_release_seed():
    requests = []
    accepted_z = np.array([1.0, -0.0, 2.0, 8.0])
    retained_z = accepted_z + 0.25

    def power(request):
        requests.append(request)
        if len(requests) == 1:
            accepted = make_point(9.0, mode=-105, coordinates=accepted_z)
            retained = make_point(1.0, mode=-105, coordinates=retained_z)
            return policy.PhaseOutcome((retained,), accepted, policy.PhaseWork(1, 3, 2), mode=0)
        assert request.start.coordinates.tobytes() == accepted_z.tobytes()
        return outcome(make_point(10.0, power=1.0, mode=-105))

    def release(request):
        assert request.start.coordinates.tobytes() == accepted_z.tobytes()
        return outcome(make_point(8.0))

    selected = run_policy(lambda request: outcome(make_point(12.0)), released=release,
        phases=policy.ControllerPhases(power=power, qualified_power_densities=(0,)))
    assert selected.point.value == 1.0 and selected.point.coordinates.tobytes() == retained_z.tobytes()
    assert selected.preparation_stop == "first profile increase"


@pytest.mark.parametrize("miss,expected_count", [(False, 3), (True, 2)])
def test_first_profile_increase_and_equality_miss_keep_actual_lower_theta(miss, expected_count):
    calls = []

    def power(request):
        index = len(calls)
        calls.append(request)
        value = (10.0, 9.0, 9.0 + 3e-11)[index]
        actual = request.physical_power + (1e-9 if miss and index == 1 else 0.0)
        return outcome(make_point(value, power=actual, mode=-105), mode=0)

    selected = run_policy(lambda request: outcome(make_point(12.0)),
        phases=policy.ControllerPhases(power=power, qualified_power_densities=(0,)))
    assert len(calls) == expected_count and selected.point.value == 9.0
    if miss:
        assert selected.point.theta[2] == 1.0 + 1e-9
        assert not selected.diagnostics[2]["eligible_for_continuation"]
    else:
        assert selected.preparation_stop == "first profile increase"


def test_value_only_kink_is_retained_without_becoming_algorithmic_continuation():
    point = make_point(1.0, mode=-105, missing_gradient=True)
    selected = run_policy(lambda request: outcome(make_point(10.0)),
        phases=policy.ControllerPhases(power=lambda request: outcome(point),
            qualified_power_densities=(0,)),
        released=lambda request: pytest.fail("A missing-gradient initialization cannot continue."))
    assert selected.point is point and selected.point.finite_feasible
    assert not selected.point.can_continue and len(selected.diagnostics) == 2
    assert not selected.diagnostics[-1]["eligible_for_continuation"]


def test_complementary_callback_owns_derivative_validity_at_a_value_only_origin():
    origin = make_point(5.0, missing_gradient=True)
    calls = []

    def qualified_chart(request):
        calls.append(request)
        assert request.start.theta.tobytes() == origin.theta.tobytes()
        # A callback's chart derivative domain does not supply canonical theta G.
        return outcome(make_point(4.0, mode=-105, missing_gradient=True),
                       continuation=False, evidence={"chart_GH_valid": True})

    selected = run_policy(lambda request: outcome(origin),
        phases=policy.ControllerPhases(complementary=(("qualified-chart", qualified_chart),)))
    assert len(calls) == 1 and selected.point.value == 4.0
    assert np.isnan(selected.point.gradient).all() and not selected.point.certified


def test_preparation_can_salvage_failed_baseline_without_restarting_from_a_hidden_point():
    starts = []

    def power(request):
        starts.append(request.start)
        return outcome(make_point(5.0, power=request.physical_power, mode=-105,
                                  missing_gradient=True), mode=-101)

    selected = run_policy(lambda request: policy.PhaseOutcome(
        work=policy.PhaseWork(2, 5, 3, None), mode=-101, failure="baseline objective unavailable"),
        phases=policy.ControllerPhases(power=power, qualified_power_densities=(0,)))
    assert starts[0].theta.tobytes() == np.array([0.1, -0.0, 1.5, 8.0]).tobytes()
    assert selected.point.value == 5.0 and selected.point.solve.mode == -105
    assert selected.total_work().function_evaluations == 7


def test_exact_likelihood_tie_preserves_first_point_and_its_status():
    baseline = make_point(5.0, mode=9)
    other = make_point(5.0, mode=-105)
    selected = run_policy(lambda request: outcome(baseline),
        phases=policy.ControllerPhases(complementary=(("chart", lambda request: outcome(other)),)))
    assert selected.point is baseline and selected.point.solve.mode == 9


@pytest.mark.parametrize("increase,expected_stages", [(1e-11, 3), (3e-11, 2)])
def test_profile_uses_unchanged_absolute_increase_tolerance(increase, expected_stages):
    calls = []

    def power(request):
        index = len(calls)
        calls.append(request)
        value = 1.0 if index == 0 else 1.0 + increase
        return outcome(make_point(value, power=request.physical_power, mode=-105,
                                  missing_gradient=index == 2))

    selected = run_policy(lambda request: outcome(make_point(10.0)),
        phases=policy.ControllerPhases(power=power, qualified_power_densities=(0,)))
    assert len(calls) == expected_stages and selected.point.value == 1.0


def test_reported_equality_failure_stops_even_if_rounded_power_matches():
    selected = run_policy(lambda request: outcome(make_point(10.0)),
        phases=policy.ControllerPhases(power=lambda request: outcome(
            make_point(5.0, power=2.0, mode=-105), equality_feasible=False),
            qualified_power_densities=(1,)), density=1,
        released=lambda request: pytest.fail("A failed physical equality cannot seed release."))
    assert selected.point.value == 5.0
    assert selected.diagnostics[-1]["power_residual"] == 0.0
    assert not selected.diagnostics[-1]["eligible_for_continuation"]


def test_complementary_phases_share_joint_origin_and_preserve_lower_baseline():
    baseline = make_point(5.0)
    origin_bytes = baseline.theta.tobytes()
    requests = []

    def chart(value):
        def solve(request):
            requests.append(request)
            assert request.start.theta.tobytes() == origin_bytes
            return outcome(make_point(value, mode=-105))
        return solve

    selected = run_policy(lambda request: outcome(baseline),
        phases=policy.ControllerPhases(complementary=(("single", chart(6.0)), ("selected", chart(7.0)))))
    assert selected.point is baseline and len(requests) == 2
    assert [row["retained"] for row in selected.diagnostics] == [True, False, False]


def test_work_counts_include_rejected_and_unavailable_attempts_without_status_transfer():
    baseline = make_point(5.0, mode=0)
    work = policy.PhaseWork(2, 7, 4, 3, {"F": 1, "F|G": 1}, {"F": 2, "G": 3})

    def unavailable(request):
        raise policy.PhaseUnavailable("rank unavailable", policy.PhaseOutcome(
            work=work, mode=-3, failure="rank unavailable", evidence={"selection_evaluations": 1}))

    selected = run_policy(lambda request: outcome(baseline,
        work=policy.PhaseWork(3, 10, 6, None, {"F|G": 1})),
        phases=policy.ControllerPhases(complementary=(("unavailable", unavailable),)))
    totals = selected.total_work()
    assert (totals.iterations, totals.function_evaluations, totals.gradient_evaluations) == (5, 17, 10)
    assert totals.hessian_evaluations is None
    assert totals.validation_requests == {"F": 1, "F|G": 2}
    assert totals.constraint_requests == {"F": 2, "G": 3}
    assert selected.point.solve.mode == 0 and not selected.point.certified
    assert selected.diagnostics[-1]["failure"] == "rank unavailable"
    assert sum(row["budget"] for row in selected.diagnostics) == 22


@pytest.mark.parametrize("exception", [ValueError, RuntimeError, AssertionError, TimeoutError, KeyboardInterrupt])
def test_unexpected_errors_ownership_failures_and_interrupts_propagate(exception):
    def bad(request):
        raise exception("do not convert caller defects into a warned fit")

    with pytest.raises(exception):
        run_policy(lambda request: outcome(make_point(5.0)),
            phases=policy.ControllerPhases(complementary=(("bad", bad),)))


@pytest.mark.parametrize("case", ["nonfinite", "infeasible"])
def test_no_finite_feasible_estimate_fails_with_attempted_history(case):
    point = make_point(np.inf if case == "nonfinite" else 5.0, feasible=case != "infeasible")
    with pytest.raises(policy.ControllerFitFailure) as caught:
        run_policy(lambda request: outcome(point))
    assert len(caught.value.diagnostics) == 1
    assert caught.value.diagnostics[0]["work"]["function_evaluations"] == 2


def test_unqualified_density_does_not_call_power_provider():
    selected = run_policy(lambda request: outcome(make_point(5.0)), density=3,
        phases=policy.ControllerPhases(power=lambda request: pytest.fail("Unqualified density invoked."),
                                      qualified_power_densities=(0, 1, 2)))
    assert len(selected.diagnostics) == 1


def test_nondefault_qualified_density_requires_explicit_interface_mode():
    with pytest.raises(ValueError, match="needs a mode"):
        run_policy(lambda request: outcome(make_point(5.0)), density=3,
            phases=policy.ControllerPhases(power=lambda request: None, qualified_power_densities=(3,)))


def test_points_are_owned_readonly_and_preserve_signed_zero_bits():
    theta = np.array([0.1, -0.0, 2.0, 8.0])
    point = make_point(5.0, theta=theta)
    before = point.theta.tobytes()
    theta[:] = 99.0
    assert point.theta.tobytes() == before and np.signbit(point.theta[1])
    with pytest.raises(ValueError):
        point.theta[0] = 99.0


@pytest.mark.parametrize("bad", ["budget", "certificate", "restricted-status", "nan-certificate"])
def test_contract_failures_do_not_become_warnings(bad):
    point = make_point(5.0, mode=-105)
    if bad == "budget":
        work = policy.PhaseWork(20, 1, 1)
    else:
        work = policy.PhaseWork(1, 1, 1)
    if bad == "certificate":
        point = make_point(5.0, mode=9, certified=True)
    elif bad == "restricted-status":
        point = make_point(5.0, mode=0)
    elif bad == "nan-certificate":
        point = replace(make_point(5.0, mode=0, certified=True),
                        assessment=native.ThetaAssessment(0.0, np.nan, True, False))
    with pytest.raises(ValueError):
        run_policy(lambda request: outcome(make_point(10.0)),
            phases=policy.ControllerPhases(complementary=(("bad", lambda request: outcome(point, work=work)),)))




@pytest.mark.parametrize("missing_gradient,certified", [(False, False), (True, False), (False, True)])
def test_integration_filters_and_infers_once_at_selected_actual_theta(
    monkeypatch, missing_gradient, certified
):
    candidate = native
    spec = ARMA(1, 1) + APARCH(1, 1) + Normal()
    model = candidate._describe_model(spec)
    data = np.array([1.0, -0.5, 0.25, -1.25])
    initial = np.arange(model.n_params, dtype=np.float64) + 1.0
    warm_z = initial + 0.001
    baseline_theta, actual_theta = initial + 0.1, initial + 0.2
    actual_theta[0] = np.nextafter(actual_theta[0], np.inf)
    requests, covariance_calls, filter_calls, solves, inverse_calls = [], [], [], [], []
    monkeypatch.setattr(candidate, "_initial_theta", lambda *args: (initial.copy(), True))

    def inverse(*args):
        inverse_calls.append(args)
        pytest.fail("Supplied original z and a theta-chart endpoint must not be inverted.")

    def forward(model, coordinates):
        return initial.copy() if coordinates.tobytes() == warm_z.tobytes() else coordinates.copy()

    def solve(model, data, coordinates, *, accuracy, maximum_iterations):
        assert accuracy == 1e-10 and maximum_iterations == 10
        assert coordinates.tobytes() == warm_z.tobytes()
        solves.append(coordinates.copy())
        coordinates[:] = baseline_theta
        return candidate.NativeSolve(0, 10.0, 0.0, 0.0, 2, 3, 4, 0, 0, 0)

    def first_order(model, data, theta, *, requests=None):
        assert theta.tobytes() == baseline_theta.tobytes()
        if requests is not None:
            requests["F|G"] = requests.get("F|G", 0) + 1
        return 10.0, np.zeros(model.n_params)

    def hessian(model, data, theta, *, request):
        assert request == 4
        requests.append(request)
        expected = baseline_theta if certified else actual_theta
        assert theta.tobytes() == expected.tobytes()
        return np.nan, np.full(model.n_params, np.nan), np.eye(model.n_params)

    assessment = candidate.ThetaAssessment(0.0, 0.0 if certified else 0.1, True, False)
    monkeypatch.setattr(candidate, "_theta_to_coordinates", inverse)
    monkeypatch.setattr(candidate, "_coordinates_to_theta", forward)
    monkeypatch.setattr(candidate, "_solve_coordinates", solve)
    monkeypatch.setattr(candidate, "_refine_garch_theta_if_needed", lambda *args, **kwargs: None)
    monkeypatch.setattr(candidate, "_evaluate_theta_first_order", first_order)
    monkeypatch.setattr(candidate, "_evaluate_theta_request", hessian)
    monkeypatch.setattr(candidate, "_assess_theta_solution", lambda *args, **kwargs: assessment)
    monkeypatch.setattr(candidate, "_mean_parameters_are_feasible", lambda *args: True)
    monkeypatch.setattr(candidate, "_boundary_warning", lambda *args: None)

    def filter_state(spec, data, theta, value):
        filter_calls.append(theta.copy())
        from types import SimpleNamespace
        return SimpleNamespace(sigma2=np.ones_like(data), residuals=data.copy()), 0.0

    monkeypatch.setattr(candidate, "_cross_check_canonical_filter", filter_state)
    monkeypatch.setattr(candidate, "observed_covariance", lambda information, *, stationary:
                        covariance_calls.append(stationary) or (np.eye(model.n_params) if stationary else None))

    def chart(request):
        if certified:
            pytest.fail("Certified-only trigger attempted a chart.")
        point = make_point(9.0, theta=actual_theta, mode=-105,
                           missing_gradient=missing_gradient, residual=0.2)
        return outcome(point, work=policy.PhaseWork(1, 5, 6, 1, {"F|G": 1}), mode=7)

    fitted = candidate.fit_native_spec(spec, data, maxiter=10,
        _aparch_initial_coordinates=warm_z,
        _aparch_phases=policy.ControllerPhases(complementary=(("single-chart", chart),),
            trigger_policy="uncertified-only" if certified else "always"))
    expected = baseline_theta if certified else actual_theta
    assert fitted.params.tobytes() == expected.tobytes()
    assert len(filter_calls) == len(covariance_calls) == len(solves) == 1
    assert not inverse_calls and requests == ([] if missing_gradient else [4])
    assert covariance_calls == [certified]
    assert fitted.fit_info.optimizer_mode == (0 if certified else -105)
    assert fitted.fit_info.stationarity_status == (
        "certified" if certified else "unavailable" if missing_gradient else "uncertified")
    assert fitted.fit_info.inference_status == ("available" if certified else "unavailable")
    assert fitted.fit_info.function_evaluations == fitted.optimization_result.nfev == (3 if certified else 8)
    assert fitted.fit_info.gradient_evaluations == fitted.optimization_result.njev == (4 if certified else 10)
    assert fitted.n_iter == (2 if certified else 3)
    assert fitted.fit_info.maxiter == (10 if certified else 22)
    assert fitted.fit_info.postfit_jet_requests == (
        {"F|G": 1, "H": 1} if certified else {"F|G": 2} if missing_gradient else {"F|G": 2, "H": 1})
    if missing_gradient:
        assert native._GRADIENT_UNAVAILABLE_WARNING in fitted.fit_info.warnings




@pytest.mark.parametrize("mode,propagates", [(-101, False), (0, True)])
def test_coordinate_endpoint_failure_policy_is_limited_to_explicit_native_rejection(
    monkeypatch, mode, propagates
):
    candidate = native
    spec = ARMA(1, 1) + APARCH(1, 1) + Normal()
    model = candidate._describe_model(spec)
    data = np.ones(4)
    theta = np.ones(model.n_params)
    monkeypatch.setattr(candidate, "_solve_coordinates", lambda *args, **kwargs:
                        candidate.NativeSolve(mode, np.inf, np.inf, np.inf, 2, 3, 4, 0, 0, 0))
    monkeypatch.setattr(candidate, "_coordinates_to_theta", lambda model, coordinates: theta.copy())

    def rejected(*args, requests=None, **kwargs):
        requests["F|G"], requests["F"] = 1, 1
        raise RuntimeError("objective unavailable")

    monkeypatch.setattr(candidate, "_assess_aparch_phase_point", rejected)
    exception = RuntimeError if propagates else policy.ControllerFitFailure
    with pytest.raises(exception) as caught:
        candidate._run_aparch_fit_controller(model, data, theta, theta,
            policy.ControllerPhases(), 1e-10, 10)
    if not propagates:
        row = caught.value.diagnostics[0]
        assert row["mode"] == -101 and row["work"]["function_evaluations"] == 3
        assert row["work"]["validation_requests"] == {"F|G": 1, "F": 1}


def test_integrated_phase_order_selects_before_one_final_reporting_pass(monkeypatch):
    candidate = native
    spec = ARMA(1, 1) + APARCH(1, 1) + Normal()
    model = candidate._describe_model(spec)
    data = np.array([1.0, -0.5, 0.25, -1.25])
    initial = np.arange(model.n_params, dtype=np.float64) + 1.0
    baseline, released, final = (initial + shift for shift in (0.1, 0.3, 0.4))
    power_index = model.n_mean + model.n_vol - 1
    trace, power_z = [], []
    monkeypatch.setattr(candidate, "_initial_theta", lambda *args: (initial.copy(), True))
    monkeypatch.setattr(candidate, "_theta_to_coordinates", lambda *args:
                        pytest.fail("Every coordinate stage has its original warm z."))
    monkeypatch.setattr(candidate, "_coordinates_to_theta", lambda model, coordinates: coordinates.copy())
    monkeypatch.setattr(candidate, "_mean_parameters_are_feasible", lambda *args: True)
    monkeypatch.setattr(candidate, "_refine_garch_theta_if_needed", lambda *args, **kwargs: None)
    monkeypatch.setattr(candidate, "_boundary_warning", lambda *args: None)
    monkeypatch.setattr(candidate, "_assess_theta_solution", lambda *args, **kwargs:
                        candidate.ThetaAssessment(0.0, 0.1, True, False))

    def solve(model, data, coordinates, *, accuracy, maximum_iterations):
        if not trace:
            trace.append("baseline")
            assert maximum_iterations == 10 and coordinates.tobytes() == initial.tobytes()
            coordinates[:] = baseline
            return candidate.NativeSolve(9, 10.0, 0.0, 0.1, 2, 3, 4, 0, 0, 0)
        trace.append("released")
        assert maximum_iterations == 8 and coordinates.tobytes() == power_z[0].tobytes()
        coordinates[:] = released
        return candidate.NativeSolve(9, 9.5, 0.0, 0.1, 2, 7, 6, 0, 0, 0)

    def first_order(model, data, theta, *, requests):
        requests["F|G"] = requests.get("F|G", 0) + 1
        assert theta.tobytes() in (baseline.tobytes(), released.tobytes())
        return (10.0 if theta.tobytes() == baseline.tobytes() else 9.5), np.zeros(model.n_params)

    def chart(request):
        label = "baseline-chart" if request.phase.startswith("native-baseline-") else "released-chart"
        trace.append(label)
        assert request.start.theta.tobytes() == (baseline if label == "baseline-chart" else released).tobytes()
        theta = initial + 0.2 if label == "baseline-chart" else final
        return outcome(make_point(9.0 if label == "baseline-chart" else 8.0, theta=theta, mode=-105),
                       work=policy.PhaseWork(1, 5, 6, 1, {"F|G": 1}), mode=7)

    def power(request):
        trace.append(f"power-{request.physical_power:g}")
        assert request.start.coordinates.tobytes() == (initial if not power_z else power_z[-1]).tobytes()
        theta = initial.copy()
        theta[power_index] = request.physical_power
        power_z.append(theta.copy())
        return outcome(make_point(10.0 if len(power_z) == 1 else 10.1, theta=theta,
                                  coordinates=theta, mode=-105), mode=0)

    def final_hessian(model, data, theta, *, request):
        trace.append("H")
        assert request == 4 and theta.tobytes() == final.tobytes()
        return np.nan, np.full(model.n_params, np.nan), np.eye(model.n_params)

    def filtering(spec, data, theta, value):
        from types import SimpleNamespace
        trace.append("filter")
        assert theta.tobytes() == final.tobytes() and value == 8.0
        return SimpleNamespace(sigma2=np.ones_like(data), residuals=data.copy()), 0.0

    def covariance(information, *, stationary):
        trace.append("covariance")
        assert not stationary
        return None

    monkeypatch.setattr(candidate, "_solve_coordinates", solve)
    monkeypatch.setattr(candidate, "_evaluate_theta_first_order", first_order)
    monkeypatch.setattr(candidate, "_evaluate_theta_request", final_hessian)
    monkeypatch.setattr(candidate, "_cross_check_canonical_filter", filtering)
    monkeypatch.setattr(candidate, "observed_covariance", covariance)
    fitted = candidate.fit_native_spec(spec, data, maxiter=10, _aparch_initial_coordinates=initial,
        _aparch_phases=policy.ControllerPhases(power=power, qualified_power_densities=(0,),
            complementary=(("chart", chart),), complementary_budget=3, power_budget=3, released_budget=8))
    assert trace == ["baseline", "baseline-chart", "power-2", "power-1", "released", "released-chart",
                     "H", "filter", "covariance"]
    assert fitted.params.tobytes() == final.tobytes() and fitted.fit_info.optimizer_mode == -105
    assert fitted.fit_info.maxiter == 30 and fitted.n_iter == 8
    assert fitted.fit_info.function_evaluations == fitted.optimization_result.nfev == 24
    assert fitted.fit_info.gradient_evaluations == fitted.optimization_result.njev == 24
    assert fitted.fit_info.postfit_jet_requests == {"F|G": 6, "H": 1}


def preparation_fixture(density, *, status=0, eligible=1, missing_best_gradient=False):
    import ctypes
    from types import SimpleNamespace

    model = SimpleNamespace(kind="arma_aparch", density_code=density, n_params=4,
        n_mean=1, n_vol=2, p_ar=1, q_ma=1, p_vol=1, q_vol=1,
        scored_observations=lambda n: n-1)
    data = np.array([0.5, -0.7, 0.1, -0.4])
    calls = []

    def array(address, width):
        return np.ctypeslib.as_array((ctypes.c_double * width).from_address(address))

    def prepare(*args):
        assert len(args) == 16
        specification, incoming, width, observations, count, h0, lower, upper, arena, size, jet, jet_size, continuation, continuation_n, best, best_n = args
        assert len(specification) == 9 and type(specification[4]) is int and specification[4] == density
        assert continuation_n == best_n == width == 4 and size > 0 and jet_size > 0
        np.testing.assert_array_equal(array(observations, count), data)
        assert len({incoming, observations, lower, upper, arena, jet, continuation, best}) == 8
        start = array(incoming, width).copy()
        calls.append((specification, start, arena))
        array(continuation, width)[:] = start
        array(continuation, width)[2] = specification[6]
        array(best, width)[:] = start
        array(best, width)[0] = 0.33
        array(best, width)[2] = 1.5  # Retained trial need not obey the held equality.
        if status == -1:
            return (-1, np.inf, np.inf, np.inf, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0)
        if status == -2:
            array(continuation, width)[:] = np.nan
        return (status, 10.0 if status == 0 else np.inf, 7.0, 0.0, 9, 2,
                11, 7, 0, 3, 2, 0 if specification[5] else 1, 1,
                int(density not in (1, 2)), eligible, 1)

    core = SimpleNamespace(_arma_aparch_preparation_workspace_bytes=lambda request: 64,
                           _arma_aparch_prepare=prepare,
                           _arma_aparch_workspace_bytes=lambda ar, ma, p, q, resolved, transformed, mask:
                           64 if (resolved, transformed, mask) == (density, 1, 3) else 0)

    def assess(model, data, theta, solve, accuracy, *, coordinates, requests):
        requests["F|G"] = requests.get("F|G", 0) + 1
        best = theta[0] == 0.33
        return make_point(7.0 if best else 10.0, theta=theta, mode=-105,
                          coordinates=coordinates, missing_gradient=best and missing_best_gradient)

    provider = SimpleNamespace(_native_core=core, _initial_variance=lambda data: 1.0,
        _coordinate_bounds=lambda model: (np.full(4, -np.inf), np.full(4, np.inf)),
        _jet_workspace=lambda *args, **kwargs: native._aligned_workspace(64),
        _aligned_workspace=native._aligned_workspace, NativeSolve=native.NativeSolve,
        _coordinates_to_theta=lambda model, z: z.copy(), _assess_aparch_phase_point=assess,
        _theta_to_coordinates=lambda *args: pytest.fail("An original coordinate start must be reused."))
    return model, data, provider, calls


def full_preparation_fixture(density):
    model, data, provider, calls = preparation_fixture(density)
    original = provider._native_core._arma_aparch_prepare
    plans = []

    def prepare(specification, *arguments):
        assert len(specification) == 10 and specification[-1] == 1
        report = list(original(specification[:-1], *arguments))
        report[4], report[8] = -105, 3
        return tuple(report), dict(strategy=1, held_power_certified=False,
            solver_diagnostics_unrestricted=(0, 7.0 / 3, 0.0, 0.0, 2, 11, 7, 1, 3, 0),
            reuse_queries=18, reuse_hits=4, constraint_evaluations=0,
            jacobian_evaluations=0, constraint_curvature_resets=0,
            checkpoint_restarts=0, stop_reason=0, derivatives_current=True)

    def jet_bytes(*plan):
        plans.append(plan)
        return 128 if plan[-1] == 7 else 0

    provider._native_core._arma_aparch_prepare = prepare
    provider._native_core._arma_aparch_workspace_bytes = jet_bytes
    return model, data, provider, calls, plans


@pytest.mark.parametrize("density", [0, 3, 4, 5, 6, 7])
def test_full_preparation_keeps_counts_and_unrestricted_diagnostics_separate(density):
    model, data, provider, calls, plans = full_preparation_fixture(density)
    callback = adapters.NativePowerPreparation(model, data, provider, full_controller=True)
    z = np.array([0.2, -0.0, 1.7, 8.5])
    request = policy.PhaseRequest("held-full", "power", policy.PhaseStart(z, z),
        1e-12, 19, (0, 1, 3), 2.0, "fixed-coordinate-slsqp")
    outcome = callback(request)
    assert plans == [(1, 1, 1, 1, density, 1, 7)]
    assert calls[0][0][-2:] == (1e-12, 19)
    assert calls[0][1].tobytes() == z.tobytes()
    assert outcome.evidence["specification"][-1] == 1
    assert outcome.evidence["full_controller"] is True
    assert outcome.evidence["controller_diagnostics"]["solver_diagnostics_unrestricted"][0] == 0
    assert outcome.mode == -105 and all(point.solve.mode == -105 for point in outcome.points)
    assert outcome.continuation.theta[2] == 2.0 and outcome.points[-1].theta[2] == 1.5
    assert outcome.work.hessian_evaluations == 3
    assert (outcome.work.function_evaluations, outcome.work.gradient_evaluations) == (11, 7)
    callback(request)
    assert calls[0][2] == calls[1][2]


@pytest.mark.parametrize("density", [1, 2])
def test_full_preparation_rejects_coupled_power(density):
    model, data, provider, _ = preparation_fixture(density)
    with pytest.raises(ValueError, match="physical-power equality"):
        adapters.NativePowerPreparation(model, data, provider, full_controller=True)


@pytest.mark.parametrize("change", [
    {"strategy": 0}, {"held_power_certified": True}, {"reuse_hits": 999},
    {"reuse_queries": -1}, {"derivatives_current": 1},
    {"solver_diagnostics_unrestricted": (0,)},
])
def test_full_preparation_rejects_malformed_diagnostics(change):
    model, data, provider, _, _ = full_preparation_fixture(0)
    original = provider._native_core._arma_aparch_prepare

    def changed(*args):
        report, diagnostics = original(*args)
        return report, dict(diagnostics, **change)

    provider._native_core._arma_aparch_prepare = changed
    callback = adapters.NativePowerPreparation(model, data, provider, full_controller=True)
    z = np.array([0.2, -0.0, 1.7, 8.5])
    request = policy.PhaseRequest("held-full", "power", policy.PhaseStart(z, z),
        1e-12, 19, (0, 1, 3), 2.0, "fixed-coordinate-slsqp")
    with pytest.raises(RuntimeError):
        callback(request)


@pytest.mark.parametrize("density", range(8))
def test_native_preparation_uses_declared_tuple_resolved_density_and_separate_slots(density):
    model, data, provider, calls = preparation_fixture(density, missing_best_gradient=True)
    phases = adapters.bind_native_preparation(model, data, policy.ControllerPhases(
        qualified_power_densities=(density,), force_runtime_map=True), native=provider)
    z = np.array([0.2, -0.0, 1.7, 8.5])
    free = tuple(i for i in range(4) if density in (1, 2) or i != 2)
    mode = "physical-equality-nu-free" if density in (1, 2) else "fixed-coordinate-slsqp"
    request = policy.PhaseRequest("native-physical-power-2", "power", policy.PhaseStart(z, z),
                                  1e-10, 5, free, 2.0, mode)
    result = phases.power(request)
    assert calls[0][0] == (1, 1, 1, 1, density, True, 2.0, 1e-10, 5)
    assert calls[0][1].tobytes() == z.tobytes()
    assert result.continuation.value == 10.0 and result.continuation.theta[2] == 2.0
    assert result.points[-1].value == 7.0 and result.points[-1].theta[2] == 1.5
    assert np.isnan(result.points[-1].gradient).all() and result.points[-1].solve.mode == -105
    assert result.mode == 9 and result.equality_feasible and result.work.hessian_evaluations == 0
    assert result.evidence["map_dispatch"] == 0 and result.evidence["objective_dispatch"] == 1
    assert result.evidence["force_runtime_map"] is True
    assert result.work.validation_requests == {"F|G": 2}
    assert result.work.constraint_requests == {"F": 3, "G": 2}
    assert (result.work.iterations, result.work.function_evaluations, result.work.gradient_evaluations) == (2, 11, 7)
    assert phases.complementary == () and phases.trigger_policy == "always"
    phases.power(request)
    assert calls[0][2] == calls[1][2]  # Reuse the caller-owned stage arena.


@pytest.mark.parametrize("status", [-1, -2])
def test_native_preparation_rejection_is_not_a_success_or_lost_best(status):
    model, data, provider, calls = preparation_fixture(1, status=status, eligible=0)
    phases = adapters.bind_native_preparation(model, data,
        policy.ControllerPhases(qualified_power_densities=(1,)), native=provider)
    z = np.array([0.2, 0.0, 1.7, 8.5])
    request = policy.PhaseRequest("power", "power", policy.PhaseStart(z, z),
                                 1e-10, 5, (0, 1, 2, 3), 2.0, "physical-equality-nu-free")
    if status == -1:
        with pytest.raises(ValueError, match="ABI/layout"):
            phases.power(request)
    else:
        result = phases.power(request)
        assert result.failure and result.continuation is None and not result.equality_feasible
        assert result.points[0].value == 7.0 and result.work.function_evaluations == 11


def test_native_preparation_requires_qualification_and_does_not_coerce_density_to_bool():
    model, data, provider, _ = preparation_fixture(4)
    with pytest.raises(ValueError, match="qualified"):
        adapters.bind_native_preparation(model, data, policy.ControllerPhases(), native=provider)
    model.density_code = True
    with pytest.raises(ValueError, match="density code"):
        adapters.NativePowerPreparation(model, data, provider)


def test_native_preparation_malformed_result_is_a_binding_failure():
    with pytest.raises(RuntimeError, match="result tuple"):
        adapters._parse_preparation_result((0, 1.0))
    with pytest.raises(RuntimeError, match="result tuple"):
        adapters._parse_preparation_result((0,) * 15)


def test_fit_wires_only_explicitly_qualified_native_preparation(monkeypatch):
    candidate = native
    spec = ARMA(1, 1) + APARCH(1, 1) + Normal()
    model = candidate._describe_model(spec)
    initial = np.ones(model.n_params)
    data = np.array([0.5, -0.4, 0.8, -0.1])
    chart = lambda request: None
    phases = policy.ControllerPhases(qualified_power_densities=(0,), complementary=(("injectable", chart),))
    calls = []
    monkeypatch.setattr(candidate, "_initial_theta", lambda *args: (initial.copy(), True))
    monkeypatch.setattr(candidate, "_theta_to_coordinates", lambda *args: initial.copy())

    def bind(model, prepared, supplied, *, native):
        calls.append("binding")
        assert native is candidate and supplied is phases
        np.testing.assert_array_equal(prepared, data)
        return replace(supplied, power=lambda request: None)

    def controller(model, prepared, theta, coordinates, supplied, accuracy, maximum_iterations):
        calls.append("controller")
        assert supplied.power is not None and supplied.complementary == (("injectable", chart),)
        raise RuntimeError("stop before running numerical phases")

    monkeypatch.setattr(candidate, "bind_native_preparation", bind)
    monkeypatch.setattr(candidate, "_run_aparch_fit_controller", controller)
    with pytest.raises(RuntimeError, match="stop before"):
        candidate.fit_native_spec(spec, data, maxiter=10, _aparch_phases=phases)
    assert calls == ["binding", "controller"]


def test_phase_wall_time_includes_unavailable_retained_callback_and_assessment():
    ticks = iter([100.0, 101.0, 104.0, 105.0, 111.0])
    point = make_point(4.0, mode=-105, missing_gradient=True)

    def unavailable(request):
        raise policy.PhaseUnavailable("chart derivative unavailable", policy.PhaseOutcome(
            points=(point,), work=policy.PhaseWork(1, 5, 2), mode=-3,
            failure="chart derivative unavailable"))

    selected = run_policy(lambda request: outcome(make_point(5.0)),
        phases=policy.ControllerPhases(complementary=(("chart", unavailable),)), clock=lambda: next(ticks))
    assert selected.point is point
    assert [row["elapsed_seconds"] for row in selected.diagnostics] == [3.0, 6.0]
    assert [row["controller_checkpoint_elapsed_seconds"] for row in selected.diagnostics] == [4.0, 11.0]
    assert selected.diagnostics[-1]["failure"] == "chart derivative unavailable"
    assert "not first attainment" in selected.diagnostics[-1]["time_scope"]
    with pytest.raises(StopIteration):
        next(ticks)


def multi_fixture(density=0, *, status=0, stop=1, missing_gradient=False,
                  unchanged=False, identical=False, mutate=False, summary_change=None):
    import ctypes
    from types import SimpleNamespace

    width, k = 8, 2
    model = SimpleNamespace(kind="arma_aparch", density_code=density, n_params=width,
        n_mean=3, n_vol=5, p_ar=1, q_ma=1, p_vol=1, q_vol=1, maximum_lag=1,
        scored_observations=lambda n: n-1)
    data = np.array([0.5, -0.7, 0.1, -0.4, 0.3])
    calls, assessments, workspace_calls = [], [], []

    def array(address, count, dtype=ctypes.c_double):
        return np.ctypeslib.as_array((dtype * count).from_address(address))

    def workspace(*specification):
        workspace_calls.append(specification)
        assert specification == (1, 1, 1, 1, density, k, True)
        return 256

    def phase(*args):
        assert len(args) == 27
        (incoming, size, observations, n, h0, ar, ma, p, q, resolved, rank, steps,
         arena, arena_size, anchors, anchor_n, columns, column_n,
         continuation, continuation_n, best, best_n, summary, summary_n,
         trace, trace_n, force_runtime) = args
        assert (ar, ma, p, q, resolved, rank, steps) == (1, 1, 1, 1, density, k, 5)
        assert type(resolved) is int and type(force_runtime) is bool and force_runtime
        assert size == continuation_n == best_n == width
        assert anchor_n == column_n == k and summary_n == 12
        assert trace_n == steps*(12+2*width) and arena_size == 256 and arena % 16 == 0
        buffers = [(incoming, size*8), (observations, n*8), (arena, arena_size),
                   (anchors, anchor_n*np.dtype(np.uintp).itemsize),
                   (columns, column_n*np.dtype(np.uintp).itemsize),
                   (continuation, continuation_n*8), (best, best_n*8),
                   (summary, summary_n*8), (trace, trace_n*8)]
        for index, (address, byte_count) in enumerate(buffers):
            assert address % 8 == 0
            assert all(address+byte_count <= other or other+other_n <= address
                       for other, other_n in buffers[:index])
        np.testing.assert_array_equal(array(observations, n), data)
        theta = array(incoming, size).copy()
        calls.append((theta, args))
        if mutate:
            array(incoming, size)[0] += 1.0
        # Poison every non-summary output on failure; the adapter must ignore it.
        array(continuation, width)[:] = theta
        array(best, width)[:] = theta
        array(anchors, k, ctypes.c_size_t)[:] = [1, 2]
        array(columns, k, ctypes.c_size_t)[:] = [0, 1]
        if status != 0:
            array(continuation, width)[0] = 0.31
            array(best, width)[0] = np.nextafter(0.33, 1.0)
            array(anchors, k, ctypes.c_size_t)[:] = np.iinfo(np.uintp).max
            array(columns, k, ctypes.c_size_t)[:] = np.iinfo(np.uintp).max
            array(trace, trace_n)[:] = 123.0
            if status == -2:
                array(summary, summary_n)[:] = [-2, np.inf, np.inf, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            return status
        if not unchanged:
            array(continuation, width)[0] = 0.31
            array(best, width)[0] = np.nextafter(0.33, 1.0)
        if identical:
            array(best, width)[:] = array(continuation, width)
        array(summary, summary_n)[:] = [stop, 10.0 if identical or unchanged else 7.0,
                                       10.0, 2, 1, 11, 3, 2, 6, 9, 7, 2]
        if summary_change is not None:
            summary_change(array(summary, summary_n))
        array(trace, trace_n)[:2*(12+2*width)] = np.arange(2*(12+2*width))
        return status

    def assess(model, data, theta, solve, accuracy, *, coordinates, requests):
        assert solve.mode == -105 and not np.isfinite(solve.kkt_residual)
        requests["F|G"] = requests.get("F|G", 0) + 1
        assessments.append((theta.copy(), coordinates, solve))
        return make_point(7.0 if theta[0] == np.nextafter(0.33, 1.0) else 10.0,
                          theta=theta, mode=-105, coordinates=coordinates,
                          missing_gradient=missing_gradient)

    core = SimpleNamespace(_arma_aparch_multi_phase_workspace_bytes=workspace,
                           _arma_aparch_multi_phase_run=phase)
    provider = SimpleNamespace(_native_core=core, _initial_variance=lambda data: 1.0,
        _aligned_workspace=native._aligned_workspace, NativeSolve=native.NativeSolve,
        _assess_aparch_phase_point=assess,
        _coordinates_to_theta=lambda *args: pytest.fail("Charts return actual theta, not z."),
        _theta_to_coordinates=lambda *args: pytest.fail("Chart theta must not be round-tripped."))
    return model, data, provider, calls, assessments, workspace_calls


def multi_request(model, *, budget=5):
    theta = np.array([0.2, -0.0, 0.1, 0.08, 0.1, 0.1, 0.7, 1.4])
    z = np.array([0.4, -0.0, 0.2, -2.0, 0.3, 0.3, 0.6, 0.1])
    return policy.PhaseRequest("native-baseline-selected-multi", "complementary",
        policy.PhaseStart(theta, z), 1e-10, budget, tuple(range(model.n_params)))


def multi_callback(model, data, provider):
    return adapters.NativeMultiPhase(model, data, provider, k=2, maximum_steps=5,
                                     qualified_densities=(model.density_code,), force_runtime=True)


@pytest.mark.parametrize("density", range(8))
def test_native_multi_uses_real_27arg_contract_and_retains_actual_theta(density):
    model, data, provider, calls, assessments, workspace_calls = multi_fixture(density)
    callback = multi_callback(model, data, provider)
    request = multi_request(model)
    result = callback(request)
    assert calls[0][0].tobytes() == request.start.theta.tobytes()
    assert len(workspace_calls) == 1 and request.start.theta[0] == 0.2
    assert result.continuation.value == 10.0 and result.points[-1].value == 7.0
    assert result.points[-1].theta[0] == np.nextafter(0.33, 1.0)
    assert result.points[-1].coordinates is None
    assert result.points[-1].solve.mode == -105 and not result.points[-1].certified
    assert result.work == policy.PhaseWork(2, 11, 3, 3, {"F|G": 2})
    assert result.evidence["anchors"] == [1, 2] and result.evidence["columns"] == [0, 1]
    assert len(result.evidence["trace"]) == 2 and len(result.evidence["trace"][0]) == 28
    assert result.evidence["selection_attempts"] == 1
    assert result.evidence["dispatch_identity"] == "not exposed by this private Python ABI"
    callback(request)
    assert calls[0][1][12] == calls[1][1][12] and len(workspace_calls) == 1
    assert len(assessments) == 4


def test_native_multi_unchanged_theta_preserves_original_z_and_deduplicates_assessment():
    model, data, provider, _, assessments, _ = multi_fixture(unchanged=True)
    request = multi_request(model)
    result = multi_callback(model, data, provider)(request)
    assert len(assessments) == len(result.points) == 1
    assert result.points[0] is result.continuation
    assert result.points[0].theta.tobytes() == request.start.theta.tobytes()
    assert result.points[0].coordinates.tobytes() == request.start.coordinates.tobytes()
    assert result.work.validation_requests == {"F|G": 1}


@pytest.mark.parametrize("stop", [-3, -4, -5])
def test_native_multi_failed_refinement_retains_finite_F_without_canonical_G(stop):
    model, data, provider, _, _, _ = multi_fixture(stop=stop, missing_gradient=True)
    callback = multi_callback(model, data, provider)
    baseline = make_point(11.0, theta=multi_request(model).start.theta, missing_gradient=True)
    selected = policy.run_aparch_controller(multi_request(model).start,
        baseline=lambda request: outcome(baseline), released=lambda request: pytest.fail("No power phase"),
        phases=policy.ControllerPhases(complementary=(("selected-multi", callback),), complementary_budget=5),
        baseline_budget=10, density_code=0, power_index=7, accuracy=1e-10)
    assert selected.point.value == 7.0 and not selected.point.can_continue
    assert selected.point.solve.mode == -105 and not selected.point.certified
    assert len(selected.diagnostics) == 2 and selected.diagnostics[-1]["failure"]
    assert selected.diagnostics[-1]["mode"] == stop
    assert selected.diagnostics[-1]["work"]["hessian_evaluations"] == 3
    assert selected.total_work().function_evaluations == 13
    assert selected.total_work().validation_requests == {"F|G": 3}
    assert selected.diagnostics[-1]["elapsed_seconds"] >= 0


def test_native_multi_minus2_consumes_counts_only_never_poisoned_output_slots():
    model, data, provider, _, assessments, _ = multi_fixture(status=-2)
    result = multi_callback(model, data, provider)(multi_request(model))
    assert result.points == () and result.continuation is None and not assessments
    assert result.mode == -2 and result.failure and result.work == policy.PhaseWork(0, 1, 0, 0)
    assert "anchors" not in result.evidence and "trace" not in result.evidence
    assert result.evidence["summary"]["values"] == 1


def test_native_multi_minus2_counted_failed_phase_keeps_baseline_and_real_checkpoint_time():
    model, data, provider, _, assessments, _ = multi_fixture(status=-2)
    baseline = make_point(11.0, theta=multi_request(model).start.theta)
    ticks = iter([100.0, 101.0, 103.0, 104.0, 109.0])
    selected = policy.run_aparch_controller(multi_request(model).start,
        baseline=lambda request: outcome(baseline), released=lambda request: pytest.fail("No power phase"),
        phases=policy.ControllerPhases(complementary=(("selected-multi", multi_callback(model, data, provider)),),
                                       complementary_budget=5),
        baseline_budget=10, density_code=0, power_index=7, accuracy=1e-10, clock=lambda: next(ticks))
    assert selected.point is baseline and not assessments
    assert selected.total_work().function_evaluations == 3
    assert selected.total_work().gradient_evaluations == 1
    assert selected.diagnostics[-1]["mode"] == -2 and selected.diagnostics[-1]["failure"]
    assert selected.diagnostics[-1]["elapsed_seconds"] == 5.0
    assert selected.diagnostics[-1]["controller_checkpoint_elapsed_seconds"] == 9.0
    assert "not first attainment" in selected.diagnostics[-1]["time_scope"]


@pytest.mark.parametrize("values", [0, 1])
def test_native_multi_minus2_accepts_only_actual_initial_value_attempt_count(values):
    raw = np.array([-2, np.inf, np.inf, 0, 0, values, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    assert adapters._parse_multi_phase_summary(raw, status=-2, budget=5).values == values
    raw[5] = 2
    with pytest.raises(RuntimeError, match="unavailable summary"):
        adapters._parse_multi_phase_summary(raw, status=-2, budget=5)
    raw[5], raw[6] = values, 1
    with pytest.raises(RuntimeError, match="unavailable summary"):
        adapters._parse_multi_phase_summary(raw, status=-2, budget=5)


@pytest.mark.parametrize("status", [-11, -12, -13, -14, -15])
def test_native_multi_selector_failure_has_no_fabricated_phase_requests(status):
    model, data, provider, _, assessments, _ = multi_fixture(status=status)
    result = multi_callback(model, data, provider)(multi_request(model))
    assert result.mode == status and result.failure and not assessments
    assert result.points == () and result.continuation is None and result.work == policy.PhaseWork()
    assert "summary" not in result.evidence and result.evidence["selection_attempts"] == 1


@pytest.mark.parametrize("status,error", [(-1, ValueError), (42, RuntimeError), (True, RuntimeError)])
def test_native_multi_programming_or_unknown_status_propagates(status, error):
    model, data, provider, _, assessments, _ = multi_fixture(status=status)
    with pytest.raises(error):
        multi_callback(model, data, provider)(multi_request(model))
    assert not assessments


@pytest.mark.parametrize("field,value", [(3, -1), (3, 6), (4, 3), (5, np.nan), (6, 1.5),
                                        (7, 2**53), (0, -9), (1, np.inf), (1, 11.0)])
def test_native_multi_malformed_summary_is_not_numeric_unavailability(field, value):
    model, data, provider, _, assessments, _ = multi_fixture(
        summary_change=lambda summary: summary.__setitem__(field, value))
    with pytest.raises(RuntimeError):
        multi_callback(model, data, provider)(multi_request(model))
    assert not assessments


def test_native_multi_canonical_value_disagreement_and_mutation_are_not_swallowed():
    model, data, provider, _, _, _ = multi_fixture(summary_change=lambda summary: summary.__setitem__(1, 6.0))
    with pytest.raises(RuntimeError, match="canonical theta NLL"):
        multi_callback(model, data, provider)(multi_request(model))
    model, data, provider, _, _, _ = multi_fixture(mutate=True)
    with pytest.raises(RuntimeError, match="mutated"):
        multi_callback(model, data, provider)(multi_request(model))


def test_native_multi_requires_explicit_qualification_and_declared_budget():
    model, data, provider, calls, _, _ = multi_fixture()
    with pytest.raises(ValueError, match="qualified"):
        adapters.NativeMultiPhase(model, data, provider, k=2, maximum_steps=5, qualified_densities=())
    callback = multi_callback(model, data, provider)
    request = multi_request(model)
    for changed in (replace(request, budget=6), replace(request, budget=0),
                    replace(request, role="power"), replace(request, free_parameters=(0, 1))):
        with pytest.raises(ValueError):
            callback(changed)
    assert not calls
    model.density_code = True
    with pytest.raises(ValueError, match="density code"):
        multi_callback(model, data, provider)


def test_native_multi_missing_real_binding_and_overflow_fail_before_allocation():
    model, data, provider, _, _, _ = multi_fixture()
    provider._native_core._arma_aparch_multi_phase_run = None
    with pytest.raises(NotImplementedError, match="image"):
        multi_callback(model, data, provider)
    model, data, provider, _, _, _ = multi_fixture()
    provider._native_core._arma_aparch_multi_phase_workspace_bytes = lambda *args: np.iinfo(np.intp).max
    with pytest.raises(ValueError, match="overflows"):
        multi_callback(model, data, provider)

