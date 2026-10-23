"""Explicit route budgets, best-point retention, and canonical-jet controls."""

from types import SimpleNamespace
import json
import time
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import scivol._joint_mean_fit as joint
from scivol.result import FitMetadata


def result(point, value, iterations, *, success=False, z=None):
    return OptimizeResult(x=point.copy(), fun=value, success=success, stationarity_bound=0. if success else 1.,
        status=0 if success else 9, nit=iterations, nfev=0, njev=0, nhev=0, phases=(),
        selected_space="theta" if z is None else "z", optimizer_coordinates=z,
        route_budgets=[iterations])


@pytest.mark.parametrize("total", (800, 1600, 2400, 3200))
@pytest.mark.parametrize("outcome", ("better", "worse-certified", "failed"))
def test_explicit_routes_preserve_primary_and_best_point(monkeypatch, total, outcome):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int), data=np.zeros(5))
    initial, calls = np.array([1., 2.]), []
    saved_z = np.array([31., 41.])

    def primary(model, theta, **options):
        calls.append(("primary", theta.copy(), options))
        model.request_counts[3] += 7
        fitted = result(theta+1., 100., 600)
        fitted.nfev = fitted.njev = 7
        return fitted

    def alternate(model, theta, **options):
        continuation = "full_first_phase" in options
        label = "continuation" if continuation else "scaled" if options["initial_curvature"] else "unit"
        calls.append((label, theta.copy(), options))
        model.request_counts[1] += 3
        model.request_counts[2] += 2
        model.request_counts[4] += 1
        if outcome == "failed":
            raise ValueError("Unavailable route")
        value = {"scaled": 90., "unit": 80., "continuation": 70.}[label]
        fitted = result(theta+2., value if outcome == "better" else 110., 100,
            success=outcome == "worse-certified", z=saved_z.copy())
        fitted.nfev, fitted.njev, fitted.nhev = 3, 2, 1
        return fitted

    monkeypatch.setattr(joint, "solve_joint_with_retry", primary)
    monkeypatch.setattr(joint, "solve_joint", alternate)
    fitted = joint.solve_joint_with_metric_portfolio(model, initial, accuracy=1e-12, maxiter=total)
    assert calls[0][2] == dict(accuracy=1e-12, maxiter=800, transformed=False, allow_chart_refinement=True)
    np.testing.assert_array_equal(fitted.primary_route["theta"], initial+1.)
    assert fitted.primary_route["nll"] == 100. and fitted.primary_route["iterations"] == 600
    expected_calls = 1+(total-800)//800
    assert len(calls) == expected_calls
    for label, theta, options in calls[1:]:
        assert options["maxiter"] == 800 and not options["speculate"]
        assert not options["allow_chart_refinement"]
        if label != "continuation":
            np.testing.assert_array_equal(theta, initial)
            assert options["transformed"] and options["initial_curvature"] == (label == "scaled")
        else:
            assert options["full_first_phase"] and options["invalid_trial_policy"] == "minimum-step"
            assert not options["curvature_reset"] and not options["initial_curvature"]
            if outcome == "better":
                np.testing.assert_array_equal(options["initial_coordinates"], saved_z)
                np.testing.assert_array_equal(theta, initial+2.)
    expected_value = 100. if total == 800 or outcome != "better" else 100.-10.*(expected_calls-1)
    assert fitted.fun == expected_value
    assert not fitted.success
    if outcome == "better" and total > 800:
        np.testing.assert_array_equal(fitted.optimizer_coordinates, saved_z)
    assert fitted.route_budgets == [800]*expected_calls
    assert fitted.nit <= sum(fitted.route_budgets) == total
    assert fitted.jet_requests["F"] == fitted.nfev == 7+3*(expected_calls-1)
    assert fitted.jet_requests["G"] == fitted.njev == 7+2*(expected_calls-1)
    assert fitted.jet_requests["H"] == fitted.nhev == expected_calls-1
    np.testing.assert_array_equal(initial, [1., 2.])


@pytest.mark.parametrize("field", ("maxiter", "primary_iteration_limit", "alternate_iteration_limit", "continuation_iteration_limit"))
@pytest.mark.parametrize("bad", (True, 0, 1.5, np.inf))
def test_invalid_work_limits_are_rejected(field, bad):
    options = dict(maxiter=3200)
    options[field] = bad
    with pytest.raises(ValueError, match="positive integer"):
        joint.solve_joint_with_metric_portfolio(None, [1.], accuracy=1e-12, **options)


@pytest.mark.parametrize("accuracy", (0., -1., np.nan, np.inf))
def test_invalid_accuracy_is_rejected(accuracy):
    with pytest.raises(ValueError, match="positive and finite"):
        joint.solve_joint_with_metric_portfolio(None, [1.], accuracy=accuracy, maxiter=3200)


@pytest.mark.parametrize("transformed", (False, True))
def test_explicit_single_chart_disables_all_additional_work(monkeypatch, transformed):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    calls = []
    initial, z = np.array([1., 2.]), np.array([123., 456.])

    def primary(model, theta, **options):
        calls.append(options)
        return result(theta, 3., 5, z=z.copy() if transformed else None)

    monkeypatch.setattr(joint, "solve_joint_with_retry", primary)
    monkeypatch.setattr(joint, "solve_joint", lambda *a, **k: pytest.fail("Unexpected alternate"))
    fitted = joint.solve_joint_with_metric_portfolio(model, initial, accuracy=1e-12, maxiter=3200,
        transformed=transformed, allow_chart_refinement=False, initial_coordinates=z)
    assert len(calls) == 1 and calls[0]["maxiter"] == 800
    assert fitted.route_budgets == [800] and not fitted.portfolio_routes
    if transformed:
        np.testing.assert_array_equal(calls[0]["initial_coordinates"], z)


def test_certified_selected_point_skips_continuation(monkeypatch):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    calls = []
    monkeypatch.setattr(joint, "solve_joint_with_retry", lambda m, t, **k: result(t, 1., 4, success=True))

    def alternate(model, theta, **options):
        calls.append(options)
        return result(theta, 2., 3)

    monkeypatch.setattr(joint, "solve_joint", alternate)
    fitted = joint.solve_joint_with_metric_portfolio(model, [1.], accuracy=1e-12, maxiter=3200)
    assert len(calls) == 2 and fitted.success and fitted.fun == 1.
    assert fitted.route_budgets == [800, 800, 800]


@pytest.mark.parametrize("total", (1, 599, 601, 1100, 1800, 2501, 4000))
def test_custom_route_caps_reserve_without_transferring_unused_work(monkeypatch, total):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    budgets = []

    def route(model, theta, **options):
        budgets.append(options["maxiter"])
        return result(theta, 10.-len(budgets), min(3, options["maxiter"]))

    monkeypatch.setattr(joint, "solve_joint_with_retry", route)
    monkeypatch.setattr(joint, "solve_joint", route)
    fitted = joint.solve_joint_with_metric_portfolio(model, [1.], accuracy=1e-12,
        maxiter=total, primary_iteration_limit=600, alternate_iteration_limit=700,
        continuation_iteration_limit=800)
    expected, remaining = [], total
    for cap in (600, 700, 700, 800):
        if not remaining:
            break
        allocation = min(cap, remaining)
        expected.append(allocation)
        remaining -= allocation
    assert budgets == fitted.route_budgets == expected
    assert fitted.nit == sum(min(3, budget) for budget in budgets)
    assert fitted.nit <= sum(budgets) <= total


def test_failed_scaled_route_does_not_suppress_valid_unit_route(monkeypatch):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    calls = []
    monkeypatch.setattr(joint, "solve_joint_with_retry", lambda m, t, **k: result(t, 10., 2))

    def alternate(model, theta, **options):
        calls.append(options["initial_curvature"])
        model.request_counts[1] += 1
        if options["initial_curvature"]:
            raise ValueError("Initial curvature unavailable")
        fitted = result(theta+1, 9., 3)
        fitted.nfev = 1
        return fitted

    monkeypatch.setattr(joint, "solve_joint", alternate)
    fitted = joint.solve_joint_with_metric_portfolio(model, [1.], accuracy=1e-12, maxiter=2400)
    assert calls == [True, False] and fitted.fun == 9.
    assert fitted.portfolio_routes[0]["error"] == "Initial curvature unavailable"
    assert fitted.portfolio_routes[1]["error"] is None
    assert fitted.nfev == fitted.jet_requests["F"] == 2


def test_transformed_primary_preserves_original_z_when_alternates_are_worse(monkeypatch):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    z, calls = np.array([37., 41.]), []

    def primary(model, theta, **options):
        original = options["initial_coordinates"]
        np.testing.assert_array_equal(original, z)
        assert not np.shares_memory(original, z)
        return result(theta, 10., 2, success=True, z=original.copy())

    def alternate(model, theta, **options):
        calls.append(options)
        assert not options["transformed"] and "initial_coordinates" not in options
        return result(theta+1, 11., 3)

    monkeypatch.setattr(joint, "solve_joint_with_retry", primary)
    monkeypatch.setattr(joint, "solve_joint", alternate)
    fitted = joint.solve_joint_with_metric_portfolio(model, [1., 2.], accuracy=1e-12,
        maxiter=3200, transformed=True, initial_coordinates=z)
    assert len(calls) == 2 and fitted.fun == 10. and fitted.success
    np.testing.assert_array_equal(fitted.optimizer_coordinates, z)


def test_primary_route_metadata_export_is_mutation_isolated():
    info = FitMetadata(primary_route=dict(theta=[1., 2.], coordinates=[37., 41.],
        nll=100., success=True, optimizer_terminated=True, stationarity_bound=1e-8,
        kkt_tolerance=1e-6, inference_status="not-assessed", phases=[dict(mode=0)]),
        optimizer_terminated=False, stationarity_status="uncertified",
        inference_status="unavailable", warnings=("Uncertified estimate unchanged.",))
    exported = info.to_dict()
    json.dumps(exported)
    exported["primary_route"]["theta"][0] = 99.
    exported["primary_route"]["coordinates"][0] = 99.
    exported["primary_route"]["phases"][0]["mode"] = 9
    assert info.primary_route["theta"] == [1., 2.]
    assert info.primary_route["coordinates"] == [37., 41.]
    assert info.primary_route["phases"] == [dict(mode=0)]
    assert info.primary_route["success"] and info.primary_route["optimizer_terminated"]
    assert not info.optimizer_terminated and info.stationarity_status == "uncertified"
    assert info.inference_status == "unavailable" and info.warnings
    assert FitMetadata().to_dict()["primary_route"] is None


@pytest.fixture
def agreeing_routes():
    model = SimpleNamespace(data=np.zeros(4), original_n_obs=400,
        assess=lambda theta, gradient: (bool(theta[0] > 0), 0., 0.))
    primary = dict(theta=np.array([1.]), nll=1., success=True,
        optimizer_terminated=True, stationarity_bound=0.)
    scaled = dict(primary, name="scaled", theta=np.array([2.]))
    return model, primary, scaled


@pytest.mark.parametrize("route", ("primary", "scaled"))
@pytest.mark.parametrize("field,bad", (
    ("success", False), ("optimizer_terminated", False),
    ("nll", None), ("nll", np.nan), ("nll", np.inf),
    ("stationarity_bound", None), ("stationarity_bound", np.nan),
    ("stationarity_bound", np.inf), ("stationarity_bound", .01),
    ("theta", np.array([-1.])),
))
def test_portfolio_agreement_rejects_missing_credibility(agreeing_routes, route, field, bad):
    model, primary, scaled = agreeing_routes
    (primary if route == "primary" else scaled)[field] = bad
    assert joint._certified_portfolio_agreement(model, primary, [scaled], 1e-12) is None


def test_portfolio_agreement_uses_scored_sample_and_inclusive_thresholds(agreeing_routes):
    model, primary, scaled = agreeing_routes
    accuracy = 2.**-20
    scaled["nll"] += accuracy*len(model.data)
    primary["stationarity_bound"] = scaled["stationarity_bound"] = np.sqrt(accuracy)
    agreement = joint._certified_portfolio_agreement(model, primary, [scaled], accuracy)
    assert agreement["likelihood_tolerance"] == accuracy*4
    assert agreement["stationarity_tolerance"] == np.sqrt(accuracy)
    assert agreement["skipped_route"] == "unit"
    scaled["nll"] = np.nextafter(scaled["nll"], np.inf)
    assert joint._certified_portfolio_agreement(model, primary, [scaled], accuracy) is None
    scaled["nll"] = primary["nll"]
    scaled["stationarity_bound"] = np.nextafter(np.sqrt(accuracy), np.inf)
    assert joint._certified_portfolio_agreement(model, primary, [scaled], accuracy) is None


def test_portfolio_agreement_requires_completed_scaled_route(agreeing_routes):
    model, primary, scaled = agreeing_routes
    assert joint._certified_portfolio_agreement(model, primary, [], 1e-12) is None
    scaled["name"] = "unit"
    assert joint._certified_portfolio_agreement(model, primary, [scaled], 1e-12) is None


@pytest.mark.parametrize("scaled_better", (False, True))
@pytest.mark.parametrize("total", (1600, 3200))
def test_agreement_skips_only_an_eligible_unit_route(monkeypatch, scaled_better, total):
    calls = []
    model = SimpleNamespace(data=np.zeros(5), request_counts=np.zeros(8, dtype=int),
        assess=lambda theta, gradient: (True, 0., 0.))

    def primary(model, theta, **options):
        fitted = result(theta+1., 100., 4, success=True)
        model.request_counts[3] += 2
        fitted.nfev = fitted.njev = 2
        return fitted

    def alternate(model, theta, **options):
        calls.append(options)
        assert options["initial_curvature"] and "full_first_phase" not in options
        fitted = result(theta+2., 100.-(4e-12 if scaled_better else 0.), 3, success=True)
        model.request_counts[3] += 1
        fitted.nfev = fitted.njev = 1
        return fitted

    monkeypatch.setattr(joint, "solve_joint_with_retry", primary)
    monkeypatch.setattr(joint, "solve_joint", alternate)
    fitted = joint.solve_joint_with_metric_portfolio(model, [1.], accuracy=1e-12, maxiter=total)
    assert len(calls) == 1 and fitted.success and fitted.nit == 7
    np.testing.assert_array_equal(fitted.x, [3. if scaled_better else 2.])
    assert fitted.fun == 100.-(4e-12 if scaled_better else 0.)
    assert fitted.route_budgets == [800, 800]
    assert len(fitted.portfolio_routes) == 1
    assert fitted.nfev == fitted.njev == 3 and fitted.nhev == 0
    assert fitted.jet_requests["F"] == fitted.jet_requests["G"] == 3
    assert (fitted.get("portfolio_agreement") is not None) == (total == 3200)


def test_agreement_metadata_cannot_certify_the_returned_point():
    from scivol import APARCH, ARX, Normal

    model = joint.JointMeanModel(ARX(1)+APARCH(1, 1)+Normal(), np.linspace(.17, .63, 32), None)
    theta = model.initialize()
    value, gradient = model.evaluate(theta, 3)[:2]
    point = result(theta, value, 0, success=True)
    point.jac, point.stationarity_bound = gradient, np.inf
    point.portfolio_agreement = dict(reason="two-certified-routes-agree", skipped_route="unit")
    report = joint.assemble_joint_fit(model, point, False, 1e-12, 3200, time.perf_counter())
    np.testing.assert_array_equal(report.params, theta)
    assert report.fit_info.portfolio_agreement == point.portfolio_agreement
    assert not report.converged and report.fit_info.stationarity_status == "uncertified"
    assert report.fit_info.warnings and report.fit_info.inference_status == "unavailable"
    assert report.cov_matrix is None
    point.portfolio_agreement["reason"] = "mutated-source"
    exported = report.fit_info.to_dict()
    exported["portfolio_agreement"]["reason"] = "mutated-export"
    assert report.fit_info.portfolio_agreement["reason"] == "two-certified-routes-agree"
    assert FitMetadata().to_dict()["portfolio_agreement"] is None


@pytest.mark.parametrize("har", (False, True), ids=("arx", "harx"))
@pytest.mark.parametrize("case", ("inactive-overflow", "tiny-beta", "large-moment", "tiny-weight"))
def test_aparch_assembly_exports_density_persistence_across_moment_ranges(har, case, capsys):
    from scivol import APARCH, ARX, HARX
    from scivol import _native_fit as native

    p, q = (3, 2) if har else (1, 1)
    data = np.linspace(.17, .63, 32)
    x = np.c_[np.sin(np.arange(32)), np.cos(np.arange(32))]
    density = 3 if case in ("inactive-overflow", "tiny-beta") else 4
    spec = (HARX([1, 3]) if har else ARX(1))+APARCH(p, q)+joint.DENSITIES[density]()
    model = joint.JointMeanModel(spec, data, x)
    theta = model.initialize()
    theta[:model.columns], theta[model.columns] = 0., .1
    alpha, gamma, delta = model.columns+1, model.columns+1+p, model.shape_start-1
    beta = gamma+p
    theta[alpha:gamma] = 0.
    raw = np.empty(1)
    if density == 3:
        theta[gamma:beta], theta[delta:] = -.999999981596122, [231.20175044099793, 1.839478852061437]
        theta[beta:delta] = (2.**-80 if case == "tiny-beta" else .4)/q
        expected = float(np.sum(theta[beta:delta]))
    else:
        theta[gamma:beta], theta[delta] = .9999998781130509, 216.965990611158
        theta[model.shape_start:] = [.22917865877731938, 24.999999999999993, 5.9999999985845776]
    shape = np.ascontiguousarray(theta[model.shape_start:])
    raw_status = native._native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, density, theta[gamma], theta[delta], 1, raw.ctypes.data, 0, 0)
    if density == 3:
        assert raw_status != 0
        # Full raw derivative requests must still reject this unrepresentable moment.
        with pytest.raises(RuntimeError, match="stationarity moment rejected"):
            native._aparch_moment_jets(model.vol, theta[model.columns:], gradient=True)
    else:
        assert raw_status == 0 and raw[0] > 1e290
        theta[alpha:gamma] = np.nextafter(0., 1.) if case == "tiny-weight" else (.1/p)/raw[0]
        theta[beta:delta] = 0. if case == "tiny-weight" else .4/q
        expected = float(np.sum(theta[alpha:gamma]*raw[0])+np.sum(theta[beta:delta]))
    slack, _ = model.constraints(theta, derivatives=False)
    assert model.assess(theta, None)[0] and expected > 0.
    if case == "tiny-beta":
        assert slack[2*model.domain[0]] == 1. and 1.-slack[2*model.domain[0]] == 0.
    elif case == "tiny-weight":
        assert expected < 1e-12 and 1.-slack[2*model.domain[0]] != expected
    scalar = model.evaluate(theta, 1)[0]
    try:
        value = model.evaluate(theta, 7)[0]
    except ValueError:
        value = model.evaluate(theta, 3)[0]
    assert abs(value-scalar) <= 2e-11
    point = result(theta, value, 0)
    point.stationarity_bound = np.inf
    fit = joint.assemble_joint_fit(model, point, False, 1e-12, 1, time.perf_counter())
    np.testing.assert_array_equal(fit.params, theta)
    assert -fit.log_likelihood == value
    assert not fit.converged and not fit.fit_info.optimizer_terminated
    assert fit.fit_info.stationarity_status == "uncertified"
    assert fit.fit_info.inference_status == "unavailable" and fit.std_errors is None
    assert fit.fit_info.warnings
    persistence, evidence_slack = native._aparch_persistence_evidence(
        model.vol, theta[model.columns:])
    # Batched half-moment integration and a scalar moment can round differently.
    np.testing.assert_allclose(persistence, expected, rtol=32*np.finfo(float).eps, atol=0.)
    assert fit.garch_params.persistence == persistence
    assert fit.garch_params.stationarity_slack == evidence_slack
    whole = fit.to_dict()
    json.dumps(whole)
    assert whole["garch_params"]["persistence"] == persistence
    assert whole["garch_params"]["stationarity_slack"] == evidence_slack
    fit._print_diagnostics_formatted(75)
    text = capsys.readouterr().out
    assert "Persistence (density-aware):" in text
    assert "Power-state stationary:" in text and "alpha + beta" not in text
    from scivol._evaluation import _filter_aparch
    from scivol.roles import Role

    standalone = APARCH(p, q)+joint.DENSITIES[density]()
    volatility = standalone.get_component(Role.VOLATILITY)
    volatility.unpack(theta[model.columns:model.shape_start])
    standalone.get_component(Role.DENSITY).unpack(theta[model.shape_start:])
    _filter_aparch(standalone, model.data, theta[model.columns:], compute_score=False, compute_hessian=False)
    assert volatility.fitted_params["persistence"] == persistence
    assert volatility.fitted_params["stationarity_slack"] == evidence_slack


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("har", (False, True), ids=("arx-narrow", "harx-wide"))
def test_matched_linear_portfolio_keeps_primary_and_crossscores_controls(density, har):
    import benchmark_native_joint as benchmark
    from scivol import APARCH, ARX, HARX
    from scivol._validation import prepare_series_input

    exog, p, q = (4, 3, 2) if har else (2, 1, 1)
    rng = np.random.default_rng(431+int(har))
    x = rng.normal(size=(96, exog))
    y = rng.standard_t(7, size=96)*.65+x@np.linspace(.18, -.12, exog)
    for i in range(1, len(y)):
        y[i] += .18*y[i-1]
    spec = (HARX([1, 3]) if har else ARX(1))+APARCH(p, q)+joint.DENSITIES[density]()
    sample = prepare_series_input(spec, y, x=x)
    model = joint.JointMeanModel(spec, sample.data, sample.x)
    shape = ([], [7.], [7., .2], [1.7], [.12, 1.8, .4], [.12, 1.8], [.12, 1.8], [.12, 2.])[density]
    initial = model.initialize(np.r_[np.full(model.columns, .025), .15, np.full(p, .03),
        np.full(p, .12), np.full(q, .45/q), 1.7, shape])
    primary_model = joint.JointMeanModel(spec, sample.data, sample.x)
    primary = joint.solve_joint_with_retry(primary_model, initial.copy(), accuracy=1e-12, maxiter=800)
    started = time.monotonic()
    fitted = joint.solve_joint_with_metric_portfolio(model, initial.copy(), accuracy=1e-12, maxiter=3200)
    np.testing.assert_array_equal(fitted.primary_route["theta"], primary.x)
    assert fitted.primary_route["nll"] == primary.fun
    assert fitted.primary_route["phases"] == primary.phases
    runtime = joint.JointMeanModel(spec, sample.data, sample.x, fixed=False)
    for transformed in (False, True):
        reference_model = joint.JointMeanModel(spec, sample.data, sample.x)
        reference = benchmark.independent_fit(reference_model, initial.copy(), 3200, transformed)
        values = [model.evaluate(point, 1)[0] for point in (fitted.x, reference.x)]
        runtime_values = [runtime.evaluate(point, 1)[0] for point in (fitted.x, reference.x)]
        np.testing.assert_allclose(values, runtime_values, rtol=0., atol=2e-11)
        assert values[0] <= values[1]+2e-11
        assert model.assess(reference.x, None)[0]
    assert fitted.fun <= min([primary.fun]+[r["nll"] for r in fitted.portfolio_routes if r["nll"] is not None])
    assert fitted.nit <= sum(fitted.route_budgets) <= 3200
    if fitted.selected_space == "z":
        assert fitted.optimizer_coordinates is not None
        joint._initial_joint_coordinates(model, fitted.x, True, fitted.optimizer_coordinates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = joint.assemble_joint_fit(model, fitted, False, 1e-12, 3200, started)
    np.testing.assert_array_equal(report.params, fitted.x)
    assert model.assess(report.params, None)[0]
    preserved = report.fit_info.primary_route
    np.testing.assert_array_equal(preserved["theta"], primary.x)
    assert preserved["nll"] == primary.fun and preserved["success"] == bool(primary.success)
    assert preserved["stationarity_bound"] == primary.stationarity_bound
    assert preserved["optimizer_terminated"] == bool(primary.get("termination_success", primary.status == 0))
    assert preserved["kkt_tolerance"] == 1e-6 and preserved["inference_status"] == "not-assessed"
    if primary.get("optimizer_coordinates") is None:
        assert preserved["coordinates"] is None
    else:
        np.testing.assert_array_equal(preserved["coordinates"], primary.optimizer_coordinates)
    exported = report.fit_info.to_dict()
    json.dumps(exported)
    exported["primary_route"]["theta"][0] += 1.
    fitted.primary_route["theta"][0] += 1.
    np.testing.assert_array_equal(preserved["theta"], primary.x)
    _, _, persistence = joint._aparch_volatility_domain(model.vol, report.params[model.columns:])
    assert report.garch_params.persistence == persistence
    whole = report.to_dict()
    json.dumps(whole)
    assert whole["garch_params"]["persistence"] == persistence
    if not report.converged:
        assert report.fit_info.warnings and report.fit_info.inference_status == "unavailable"
        assert report.cov_matrix is None
