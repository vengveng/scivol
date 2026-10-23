"""Primary-preserving explicit additional-route regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import scivol._joint_mean_fit as joint

policy = joint
pytestmark = pytest.mark.skipif(
    not hasattr(joint, "solve_joint_with_additional_route"),
    reason="Additional-route production patch is not integrated yet.",
)


def fit(point, value, iterations, success=False, bound=1.0):
    phase = dict(space="theta", mode=9, iterations=iterations, nll=value,
        function_evaluations=3, gradient_evaluations=2, hessian_evaluations=1)
    return OptimizeResult(x=point, fun=value, success=success, stationarity_bound=bound,
        status=9, nit=iterations, nfev=3, njev=2, nhev=1, phases=(phase,),
        selected_space="theta", route_budgets=[iterations])


@pytest.mark.parametrize("extra", (400, 800))
@pytest.mark.parametrize("outcome", ("better", "worse", "failed", "gain-cert-loss", "credible"))
def test_preserves_complete_primary_and_accounts_extra(monkeypatch, extra, outcome):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int), data=np.zeros(10))
    original, calls = np.array([1.0, 2.0]), []
    primary = fit(np.array([3.0, 4.0]), 100.0, 800,
        success=outcome in ("credible", "gain-cert-loss"),
        bound=0.0 if outcome in ("credible", "gain-cert-loss") else 1.0)

    def run_primary(model, initial, **options):
        calls.append(("primary", initial.copy(), options))
        model.request_counts[1] += 3
        model.request_counts[2] += 2
        model.request_counts[4] += 1
        return primary

    def alternate(model, initial, **options):
        calls.append(("alternate", initial.copy(), options))
        model.request_counts[1] += 5
        model.request_counts[2] += 4
        if outcome == "failed":
            raise ValueError("Alternate chart unavailable")
        result = fit(np.array([5.0, 6.0]), 90.0 if outcome in ("better", "gain-cert-loss") else 110.0, extra)
        result.nfev, result.njev, result.nhev = 5, 4, 0
        result.phases = (dict(space="z", mode=9, iterations=extra, nll=result.fun,
            function_evaluations=5, gradient_evaluations=4, hessian_evaluations=0),)
        result.selected_space = "z"
        result.optimizer_coordinates = np.array([7.0, 8.0])
        return result

    monkeypatch.setattr(policy, "solve_joint_with_retry", run_primary)
    monkeypatch.setattr(policy, "solve_joint", alternate)
    result = policy.solve_joint_with_additional_route(model, original, accuracy=1e-12, maxiter=800+extra)
    assert calls[0][2] == dict(accuracy=1e-12, maxiter=800, transformed=False, allow_chart_refinement=True)
    np.testing.assert_array_equal(original, [1.0, 2.0])
    np.testing.assert_array_equal(result.primary_route["theta"], [3.0, 4.0])
    assert result.primary_route["nll"] == 100.0 and result.primary_route["iterations"] == 800
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][1], original)
    assert calls[1][2] == dict(accuracy=1e-12, maxiter=extra, transformed=True,
        allow_chart_refinement=False, initial_curvature=True, speculate=False)
    assert result.nfev == 8 and result.njev == 6 and result.nhev == 1
    assert result.route_budgets == [800, extra]
    assert result.total_iteration_limit == 800+extra and result.nit <= 800+extra
    assert result.fun == (90.0 if outcome in ("better", "gain-cert-loss") else 100.0)
    assert result.optimizer_controller == "joint-primary-preserved-additional-route"
    assert result.jet_requests["F"] == result.nfev
    if outcome == "gain-cert-loss":
        assert not result.success
        np.testing.assert_array_equal(result.optimizer_coordinates, [7.0, 8.0])
    if outcome == "failed":
        assert result.phases[-1]["error"] == "Alternate chart unavailable"
        assert result.phases[-1]["jet_requests"]["F"] == 5


@pytest.mark.parametrize("transformed", (False, True))
def test_original_coordinates_forwarded_without_roundtrip(monkeypatch, transformed):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int), data=np.zeros(10))
    z, theta, calls = np.array([15.0, 17.0]), np.array([1.0, 2.0]), []

    def route(model, initial, **options):
        calls.append(options)
        return fit(initial.copy(), 10.0, options["maxiter"])

    monkeypatch.setattr(policy, "solve_joint_with_retry", route)
    monkeypatch.setattr(policy, "solve_joint", route)
    result = policy.solve_joint_with_additional_route(model, theta, accuracy=1e-12,
        maxiter=1200, transformed=transformed, initial_coordinates=z)
    target = calls[0] if transformed else calls[1]
    np.testing.assert_array_equal(target["initial_coordinates"], z)
    assert not np.shares_memory(target["initial_coordinates"], z)
    assert "initial_coordinates" not in calls[1 if transformed else 0]
    assert result.nit <= 1200


@pytest.mark.parametrize("budget", (0, -1, 1.5, np.nan, np.inf, True))
@pytest.mark.parametrize("field", ("maxiter", "primary_iteration_limit"))
def test_bad_budget_rejected_before_primary(monkeypatch, budget, field):
    calls = []
    monkeypatch.setattr(policy, "solve_joint_with_retry", lambda *a, **kw: calls.append(kw))
    options = dict(maxiter=1200, primary_iteration_limit=800)
    options[field] = budget
    with pytest.raises(ValueError, match="iteration limits"):
        policy.solve_joint_with_additional_route(None, np.ones(2), accuracy=1e-12, **options)
    assert not calls


@pytest.mark.parametrize("total,permission,reason", ((500, True, "no-extra-budget"),
    (800, True, "no-extra-budget"), (1200, False, "explicit-single-chart")))
def test_no_hidden_extra_work(monkeypatch, total, permission, reason):
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int), data=np.zeros(10))
    calls = []

    def primary(model, initial, **options):
        calls.append(options)
        return fit(initial.copy(), 10.0, options["maxiter"])

    monkeypatch.setattr(policy, "solve_joint_with_retry", primary)
    monkeypatch.setattr(policy, "solve_joint", lambda *a, **kw: pytest.fail("Unexpected alternate work"))
    result = policy.solve_joint_with_additional_route(model, np.ones(2), accuracy=1e-12,
        maxiter=total, allow_chart_refinement=permission)
    assert calls[0]["maxiter"] == min(total, 800)
    assert result.additional_route_skipped == reason
    assert result.nit <= total and result.total_iteration_limit == total


def test_harx_skew_keeps_complete_800_primary_then_uses_explicit_400():
    import time
    import warnings
    from scivol import APARCH, HARX, SkewT
    from scivol._validation import prepare_series_input

    rng = np.random.default_rng(432)
    x = rng.normal(size=(96, 4))
    data = rng.standard_t(7, size=96)*0.65+x@np.linspace(0.18, -0.12, 4)
    for i in range(1, len(data)):
        data[i] += 0.18*data[i-1]
    spec = HARX([1, 3])+APARCH(3, 2)+SkewT()
    prepared = prepare_series_input(spec, data, x=x)
    initial = np.r_[np.full(7, 0.025), 0.15, np.full(3, 0.03), np.full(3, 0.12),
                    [0.225, 0.225, 1.7, 7.0, 0.2]]
    baseline_model = joint.JointMeanModel(spec, prepared.data, prepared.x)
    baseline = joint.solve_joint_with_retry(baseline_model, baseline_model.initialize(initial),
        accuracy=1e-12, maxiter=800)
    model = joint.JointMeanModel(spec, prepared.data, prepared.x)
    started = time.perf_counter()
    result = joint.solve_joint_with_additional_route(model, model.initialize(initial),
        accuracy=1e-12, maxiter=1200, primary_iteration_limit=800)
    np.testing.assert_array_equal(result.primary_route["theta"], baseline.x)
    assert result.primary_route["nll"] == baseline.fun
    assert result.primary_route["phases"] == baseline.phases
    assert result.fun <= 40.21622179847018+2e-11
    assert result.nit <= 1200 and result.route_budgets == [800, 400]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = joint.assemble_joint_fit(model, result, False, 1e-12, 1200, started)
    assert model.assess(fitted.params, None)[0]
    assert not fitted.converged and fitted.fit_info.inference_status == "unavailable"
    assert fitted.fit_info.warnings and fitted.cov_matrix is None
    assert result.optimizer_coordinates is not None
    np.testing.assert_array_equal(fitted.params, result.x)
