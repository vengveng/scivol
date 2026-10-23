"""Native model adapters, best-point retention, accounting and controller budgets."""

import time

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from scivol import ARX, HARX
from scivol._active_face_fit import apply_refinement, native_refinement
from scivol._joint_mean_fit import (
    DENSITIES, FAMILIES, JointMeanModel, assemble_joint_fit, solve_joint,
)


def prepared(family=3, density=0, *, wide=False, p=None, q=None, fixed=True):
    rng = np.random.default_rng(214)
    n = 64
    data = rng.normal(size=n)
    features = rng.normal(size=(n, 4 if wide else 2))
    mean = HARX([1, 3]) if wide else ARX(1)
    p, q = (3 if wide else 1) if p is None else p, (2 if wide else 1) if q is None else q
    return JointMeanModel(mean + FAMILIES[family](p, q) + DENSITIES[density](),
                          data, features, fixed=fixed)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("chart_available", [False, True])
@pytest.mark.parametrize("path", ["auto", "theta", "z"])
def test_early_refinement_preserves_qualified_routes_and_chart_handoffs(
    monkeypatch, density, terminated, chart_available, path
):
    from types import SimpleNamespace
    import scivol._joint_mean_fit as joint

    calls, refinements = [], []
    model = SimpleNamespace(family=3, p_vol=1, code=density, request_counts=[0]*8,
        assess=lambda *args: (True, 1., 0.))
    initial = np.array([1., 2.])
    accepted = np.array([1., 1.])

    class HandoffObserved(RuntimeError):
        pass

    def phase(model, point, **options):
        calls.append((point.copy(), options))
        if len(calls) == 2:
            raise HandoffObserved
        return OptimizeResult(x=point.copy(), fun=10., jac=np.zeros(2), nit=1,
            success=terminated, status=0 if terminated else 9,
            curvature_prepared=False, kernel_dispatch=2)

    def refine(model, best, phases, remaining, accuracy):
        refinements.append((best.x.copy(), remaining, accuracy))
        return OptimizeResult(best, x=accepted.copy(), fun=9.), phases, remaining-2

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(joint, "refine_before_restart", refine)
    monkeypatch.setattr(joint, "refinement_chart_available", lambda *args: chart_available)
    with pytest.raises(HandoffObserved):
        joint.solve_joint(model, initial, transformed=path == "z", accuracy=1e-12,
            maxiter=20, terminal_reserve=3, allow_chart_refinement=path == "auto")

    expected_refinement = path == "theta" or (path == "auto" and (
        not chart_available or (density in (1, 3) and not terminated)))
    assert len(refinements) == int(expected_refinement)
    assert calls[0][1]["maxiter"] == 10
    if expected_refinement:
        np.testing.assert_array_equal(refinements[0][0], initial)
        assert refinements[0][1:] == (16, 1e-12)
    np.testing.assert_array_equal(calls[1][0], accepted if expected_refinement else initial)
    assert calls[1][1]["maxiter"] == (8 if expected_refinement else 9)
    assert calls[1][1]["transformed"] == (path != "theta")
    np.testing.assert_array_equal(initial, [1., 2.])


@pytest.mark.parametrize("accepted", [False, True])
def test_policy_retains_uncertified_improvement_and_reports_work(accepted, monkeypatch, capsys):
    model = prepared(family=0)
    initial = model.initialize()
    value, gradient, *_ = model.evaluate(initial, 3)
    direction = model.units * gradient
    improved = initial - 1e-4 * model.units * direction / np.linalg.norm(direction)
    improved_value = model.evaluate(improved, 1)[0]
    assert improved_value < value and model.assess(improved, None)[0]
    old = OptimizeResult(x=initial, fun=value, status=9, success=False,
        stationarity_bound=1., selected_space="theta", predicted_coordinate_gain=None,
        curvature_hessian=np.eye(model.size)*999,
        phases=(dict(space="theta", mode=9, iterations=1, function_evaluations=2,
                     gradient_evaluations=1, hessian_evaluations=1),))
    if accepted:
        monkeypatch.setattr(model, "assess", lambda *args: (True, 0., 0.))

    def phase(model, point, *, target):
        counts = np.zeros(16, dtype=np.uintp)
        counts[[1, 2, 4]] = 1
        counts[15] = 2
        for mask, count in enumerate(counts[:8]):
            model.request_counts[mask] += int(count)
        return dict(status=7 if accepted else 6, theta=improved if accepted else initial,
            nll=improved_value if accepted else value, kkt=0. if accepted else 1.,
            best_theta=improved, best_nll=improved_value, best_kkt=0. if accepted else np.inf,
            best_valid=True, dispatch=2, counts=counts)

    result = apply_refinement(model, old, phase=phase)
    np.testing.assert_array_equal(result.x, improved)
    assert "curvature_hessian" not in result and "curvature_hessian" in old
    assert result.curvature_prediction_pending
    assert result.success == accepted
    # Identical accepted and retained slots need only one independent assessment.
    assert result.phases[-1]["function_evaluations"] == (2 if accepted else 3)
    assert result.phases[-1]["gradient_evaluations"] == (2 if accepted else 3)
    assert result.nit == 2 and result.nhev == 2
    monkeypatch.setattr(model, "predicted_coordinate_gain", lambda *args: 0.)
    fit = assemble_joint_fit(model, result, False, 1e-12, 10, time.perf_counter())
    np.testing.assert_array_equal(fit.params, improved)
    assert fit.fit_info.optimizer_status_kind == "native-active-face"
    assert fit.fit_info.optimizer_terminated == accepted
    assert fit.converged == accepted
    if not accepted:
        assert np.isinf(fit.fit_info.kkt_residual)
        assert fit.std_errors is None and fit.warnings
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("family", [2, 3])
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("wide,p,q", [(False, 1, 1), (True, 4, 3)])
def test_native_phase_preserves_canonical_general_order_math(family, density, wide, p, q):
    model = prepared(family, density, wide=wide, p=p, q=q)
    initial = model.initialize()
    original, lower, upper = initial.copy(), model.lower.copy(), model.upper.copy()
    value = model.evaluate(initial, 1)[0]
    before = np.array(model.request_counts)
    result = native_refinement(model, initial)
    np.testing.assert_array_equal(np.array(model.request_counts)-before, result["counts"][:8])
    assert result["best_valid"] and result["best_nll"] <= result["nll"] <= value + 1e-12
    for point, reported in ((result["theta"], result["nll"]),
                            (result["best_theta"], result["best_nll"])):
        assert model.assess(point, None)[0]
        np.testing.assert_allclose(model.evaluate(point, 1)[0], reported, rtol=0, atol=2e-12)
    assert result["counts"][4] <= 1
    assert not result["counts"][[0, 3, 5, 6, 7]].any()
    assert result["dispatch"] == (0 if wide else 2)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(model.lower, lower)
    np.testing.assert_array_equal(model.upper, upper)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("density", [0, 1, 3])
def test_fixed_and_runtime_refinement_agree_at_shared_point(wide, density):
    fixed = prepared(density=density, wide=wide)
    runtime = prepared(density=density, wide=wide, fixed=False)
    initial = fixed.initialize()
    left = native_refinement(fixed, initial, backtracks=0)
    right = native_refinement(runtime, initial, backtracks=0)
    assert left["dispatch"] == 2 and right["dispatch"] == 0
    for key in ("nll", "best_nll"):
        np.testing.assert_allclose(left[key], right[key], rtol=3e-12, atol=2e-12)
    fixed_gradient = fixed.evaluate(initial, 2)[1]
    runtime_gradient = runtime.evaluate(initial, 2)[1]
    np.testing.assert_allclose(fixed_gradient, runtime_gradient, rtol=3e-12, atol=2e-12)
    # A conservative primal-dual bound includes sqrt(complementarity error).
    # Its rounding resolution is not the likelihood/gradient agreement tolerance.
    resolution = np.sqrt(np.finfo(float).eps)*max(
        1., np.linalg.norm(fixed_gradient*fixed.units)/len(fixed.data))
    for key in ("kkt", "best_kkt"):
        np.testing.assert_allclose(left[key], right[key], rtol=0, atol=resolution)
    reference = fixed.assess(initial, fixed_gradient)[1]
    for result in (left, right):
        assert abs(result["kkt"]-reference) <= 8*np.sqrt(fixed.size)*resolution
        assert (result["kkt"] <= 1e-6) == (reference <= 1e-6)
    np.testing.assert_array_equal(left["counts"][:15], right["counts"][:15])


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("density", range(8))
def test_fixed_refinement_exercises_objective_and_constraint_curvature(wide, density):
    fixed = prepared(density=density, wide=wide)
    runtime = prepared(density=density, wide=wide, fixed=False)
    initial = fixed.initialize()
    results = [native_refinement(model, initial, target=0., backtracks=2)
               for model in (fixed, runtime)]
    for model, result in zip((fixed, runtime), results):
        assert result["counts"][4] == 1
        assert result["counts"][11] == 1
        assert result["best_valid"] and model.assess(result["best_theta"], None)[0]
        values = [provider.evaluate(result["best_theta"], 1)[0] for provider in (fixed, runtime)]
        np.testing.assert_allclose(values, result["best_nll"], rtol=0, atol=1e-9)
    assert results[0]["dispatch"] == 2 and results[1]["dispatch"] == 0
    np.testing.assert_allclose(results[0]["best_nll"], results[1]["best_nll"], rtol=0, atol=1e-9)


def test_refinement_reuses_scratch_without_aliasing_returned_points():
    model = prepared(density=1)
    initial = model.initialize()
    first = native_refinement(model, initial, backtracks=0)
    work = model.workspaces[("active-face",)]
    kept = first["theta"].copy()
    initial[0] += .01
    native_refinement(model, initial, backtracks=0)
    assert model.workspaces[("active-face",)] is work
    np.testing.assert_array_equal(first["theta"], kept)
    assert not np.shares_memory(first["theta"], work.point)
    assert not np.shares_memory(first["best_theta"], work.best)


@pytest.mark.parametrize("old_bound,new_bound", [(5e-7, 6e-7), (6e-7, 5e-7)])
def test_identical_point_keeps_new_termination_and_conservative_passing_bound(
    monkeypatch, old_bound, new_bound
):
    model = prepared()
    point = model.initialize()
    value = model.evaluate(point, 1)[0]
    old = OptimizeResult(x=point, fun=value, status=9, success=False,
        stationarity_bound=old_bound, predicted_coordinate_gain=None, phases=())
    monkeypatch.setattr(model, "assess", lambda *args: (True, new_bound, 0.))

    def phase(model, point, **options):
        return dict(status=3, theta=point, nll=value, kkt=new_bound,
            best_theta=point, best_nll=value, best_kkt=new_bound, best_valid=True,
            dispatch=2, counts=np.zeros(16, dtype=np.uintp))

    result = apply_refinement(model, old, phase=phase)
    np.testing.assert_array_equal(result.x, point)
    assert result.fun == value and result.success and result.termination_success
    assert result.stationarity_bound == max(old_bound, new_bound)
    assert result.status == 3 and result.status_kind == "native-active-face"


def test_short_refinement_arena_fails_before_native_requests():
    model = prepared()
    initial = model.initialize()
    native_refinement(model, initial, backtracks=0)
    work = model.workspaces[("active-face",)]
    work.arena = work.arena[:-1]
    before = model.request_counts.copy()
    with pytest.raises(ValueError, match="rejected"):
        native_refinement(model, initial)
    assert model.request_counts == before


def test_selected_point_precision_is_resolved_before_controller_stopping(monkeypatch):
    model = prepared()
    point = model.initialize()
    value = model.evaluate(point, 1)[0]
    old = OptimizeResult(x=point, fun=value, status=9, success=False,
        stationarity_bound=5e-7, predicted_coordinate_gain=None, phases=())
    monkeypatch.setattr(model, "assess", lambda *args: (True, 6e-7, 0.))
    predictions = []

    def prediction(theta, gradient, hessian):
        predictions.append((theta.copy(), hessian.copy()))
        return .1

    def phase(model, point, **options):
        return dict(status=3, theta=point, nll=value, kkt=6e-7,
            best_theta=point, best_nll=value, best_kkt=6e-7, best_valid=True,
            dispatch=2, counts=np.zeros(16, dtype=np.uintp))

    monkeypatch.setattr(model, "predicted_coordinate_gain", prediction)
    result = apply_refinement(model, old, phase=phase, check_precision=True)
    assert result.success and result.predicted_coordinate_gain == .1
    assert not result.curvature_prediction_pending and len(predictions) == 1
    np.testing.assert_array_equal(predictions[0][0], result.x)
    np.testing.assert_array_equal(predictions[0][1], result.curvature_hessian)
    assert result.phases[-1]["hessian_evaluations"] == result.nhev == 1
    assert result.phases[-1]["native_jet_requests"]["H"] == 0


def test_passing_first_order_check_does_not_skip_requested_precision_refinement(monkeypatch):
    from scivol import _active_face_fit as policy

    model = prepared()
    old = OptimizeResult(x=model.initialize(), fun=1., success=True,
        stationarity_bound=0., predicted_coordinate_gain=.1)
    calls = []

    def phase(model, previous, **options):
        calls.append(options)
        return OptimizeResult(previous, phases=(*previous.phases,
            dict(iterations=1, selected_slot="previous", function_evaluations=1,
                 gradient_evaluations=1, hessian_evaluations=1)))

    monkeypatch.setattr(policy, "apply_refinement", phase)
    result, phases, remaining = policy.refine_before_restart(model, old, [], 3, 1e-12)
    assert calls == [dict(accuracy=1e-12, check_precision=True)]
    assert remaining == 2 and result.predicted_coordinate_gain == .1


@pytest.mark.parametrize("second_value", [7., 8.5])
def test_refinement_follows_accepted_points_but_returns_best_trial(monkeypatch, second_value):
    from scivol import _active_face_fit as policy

    model = prepared()
    initial = OptimizeResult(x=np.array([0.]), fun=10., success=False,
        stationarity_bound=3., predicted_coordinate_gain=None)
    visited = []

    def propose(model, current, **options):
        visited.append(current.x[0])
        record = dict(iterations=1, selected_slot="retained", function_evaluations=2,
                      gradient_evaluations=3, hessian_evaluations=1)
        phases = (*current.phases, record)
        if len(visited) == 1:
            accepted = OptimizeResult(current, x=np.array([1.]), fun=9.,
                stationarity_bound=2., phases=phases)
            return OptimizeResult(current, x=np.array([2.]), fun=8.,
                stationarity_bound=5., phases=phases, refinement_accepted=accepted)
        accepted = OptimizeResult(current, x=np.array([3.]), fun=second_value,
            stationarity_bound=1., phases=phases)
        return OptimizeResult(accepted, refinement_accepted=OptimizeResult(accepted))

    monkeypatch.setattr(policy, "apply_refinement", propose)
    result, phases, remaining = policy.refine_before_restart(model, initial, [], 2, 1e-12)
    assert visited == [0., 1.]
    assert result.fun == min(8., second_value)
    assert result.x[0] == (3. if second_value < 8. else 2.)
    assert remaining == 0 and len(phases) == result.nit == 2
    assert (result.nfev, result.njev, result.nhev) == (4, 6, 2)
    assert "refinement_accepted" not in result and not result.success


def test_native_proposal_exposes_accepted_point_without_discarding_better_trial(monkeypatch):
    model = prepared(family=0)
    initial = model.initialize()
    value, gradient, *_ = model.evaluate(initial, 3)
    direction = model.units**2 * gradient / np.linalg.norm(model.units*gradient)
    accepted, retained = initial-1e-4*direction, initial-2e-4*direction
    accepted_value, retained_value = [model.evaluate(point, 1)[0] for point in (accepted, retained)]
    assert retained_value < accepted_value < value
    monkeypatch.setattr(model, "assess", lambda point, gradient:
        (True, 1. if np.array_equal(point, accepted) else 3., 0.))
    old = OptimizeResult(x=initial, fun=value, success=False, stationarity_bound=2.,
        predicted_coordinate_gain=None, phases=(), curvature_hessian=np.eye(model.size))

    def phase(model, point, **options):
        return dict(status=7, theta=accepted, nll=accepted_value, kkt=1.,
            best_theta=retained, best_nll=retained_value, best_kkt=3., best_valid=True,
            dispatch=2, counts=np.zeros(16, dtype=np.uintp))

    result = apply_refinement(model, old, phase=phase)
    np.testing.assert_array_equal(result.x, retained)
    continuation = result.refinement_accepted
    np.testing.assert_array_equal(continuation.x, accepted)
    assert continuation.fun == accepted_value and continuation.stationarity_bound == 1.
    assert continuation.termination_success and not continuation.success
    assert "curvature_hessian" not in continuation and continuation.curvature_prediction_pending
    assert continuation.phases == result.phases
    assert "refinement_accepted" not in continuation


def test_refinement_chain_keeps_same_point_success_with_conservative_bound(monkeypatch):
    from scivol import _active_face_fit as policy

    old = OptimizeResult(x=np.array([0.]), fun=1., success=False,
        stationarity_bound=5e-7, predicted_coordinate_gain=None)

    def propose(model, previous, **options):
        return OptimizeResult(previous, success=True, stationarity_bound=6e-7,
            phases=(*previous.phases, dict(iterations=0, function_evaluations=1,
                gradient_evaluations=1, hessian_evaluations=0)))

    monkeypatch.setattr(policy, "apply_refinement", propose)
    result, phases, remaining = policy.refine_before_restart(prepared(), old, [], 2, 1e-12)
    assert result.success and result.stationarity_bound == 6e-7
    assert result.fun == old.fun and np.array_equal(result.x, old.x)
    assert len(phases) == 1 and remaining == 1


@pytest.mark.parametrize("gain", [0., .1])
def test_incoming_pending_precision_is_checked_before_early_return(monkeypatch, gain):
    model = prepared()
    point = model.initialize()
    value, gradient, *_ = model.evaluate(point, 3)
    old = OptimizeResult(x=point, fun=value, jac=gradient, success=True, status=0,
        stationarity_bound=0., predicted_coordinate_gain=None,
        curvature_prediction_pending=True, phases=())
    monkeypatch.setattr(model, "predicted_coordinate_gain", lambda *args: gain)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 0., 0.))
    calls = []

    def phase(model, point, **options):
        calls.append(options)
        return dict(status=6, theta=point, nll=value, kkt=0.,
            best_theta=point, best_nll=value, best_kkt=0., best_valid=True,
            dispatch=2, counts=np.zeros(16, dtype=np.uintp))

    result = apply_refinement(model, old, phase=phase, check_precision=True)
    assert calls == ([dict(target=0.)] if gain else [])
    assert result.predicted_coordinate_gain == gain and not result.curvature_prediction_pending
    assert result.nhev == 1 and result.phases[0]["iterations"] == 0
    assert result.phases[0]["algorithm"] == "native-curvature-assessment"
    assert old.curvature_prediction_pending and "curvature_hessian" not in old


def test_chain_retains_completed_precision_assessment_at_unchanged_point(monkeypatch):
    from scivol import _active_face_fit as policy

    model = prepared()
    point = model.initialize()
    value, gradient, *_ = model.evaluate(point, 3)
    old = OptimizeResult(x=point, fun=value, jac=gradient, success=True, status=0,
        stationarity_bound=0., predicted_coordinate_gain=None,
        curvature_prediction_pending=True, phases=())
    monkeypatch.setattr(model, "predicted_coordinate_gain", lambda *args: 0.)
    result, phases, remaining = policy.refine_before_restart(model, old, [], 2, 1e-12)
    assert result.success and not result.curvature_prediction_pending
    assert result.predicted_coordinate_gain == 0. and result.nhev == 1 and result.nit == 0
    assert "curvature_hessian" in result and len(phases) == 1 and remaining == 1


@pytest.mark.parametrize("options", [dict(target=-1), dict(target=np.nan), dict(backtracks=1025)])
def test_refinement_rejects_bad_options_before_objective_requests(options):
    model = prepared()
    initial = model.initialize()
    before = model.request_counts.copy()
    with pytest.raises(ValueError):
        native_refinement(model, initial, **options)
    assert model.request_counts == before


@pytest.mark.parametrize("budget", [1, 2, 7, 30])
@pytest.mark.parametrize("density", [1, 3])
@pytest.mark.parametrize("transformed", [False, True])
def test_controller_respects_unspent_budget_and_explicit_chart(budget, density, transformed):
    model = prepared(density=density, wide=True)
    initial = model.initialize()
    before = tuple(model.request_counts)
    result = solve_joint(model, initial, transformed=transformed, accuracy=1e-12,
                         maxiter=budget, allow_chart_refinement=False)
    assert result.nit <= budget
    assert result.nit == sum(phase["iterations"] for phase in result.phases)
    for field, bit in (("F", 1), ("G", 2), ("H", 4)):
        reported = sum(phase["jet_requests"][field] for phase in result.phases)
        actual = sum(count-old for mask, (count, old) in enumerate(
            zip(model.request_counts, before)) if mask & bit)
        assert reported == actual
    if transformed:
        assert all(phase["space"] == "z" for phase in result.phases)
        assert not any(phase.get("algorithm") == "native-active-face" for phase in result.phases)
    assert model.assess(result.x, None)[0]
    np.testing.assert_allclose(model.evaluate(result.x, 1)[0], result.fun, rtol=0, atol=1e-9)


@pytest.mark.parametrize("seed,density,target,certified", [
    (2785, 3, 509.99026154559645, False),
    (2786, 1, 543.8115763746505, True),
    (2786, 3, 543.0680893733575, True),
])
def test_joint_refinement_preserves_matched_quality_and_honest_status(seed, density, target, certified):
    from benchmark_native_joint import comparison_sample
    from scivol import APARCH
    from scivol._joint_mean_fit import solve_joint_with_retry

    data, features = comparison_sample(seed, True, "persistent")
    model = JointMeanModel(HARX([1, 3]) + APARCH(3, 2) + DENSITIES[density](), data, features)
    result = solve_joint_with_retry(model, model.initialize(), accuracy=1e-12,
        maxiter=2000, allow_chart_refinement=True)
    fit = assemble_joint_fit(model, result, False, 1e-12, 2000, time.perf_counter())
    assert -fit.log_likelihood <= target + 1e-9
    assert model.assess(fit.params, None)[0]
    assert result.nit <= 2000 and result.nit == sum(p["iterations"] for p in result.phases)
    assert fit.converged == certified
    assert any(p.get("algorithm") == "native-active-face" for p in result.phases)
    if not certified:
        assert fit.std_errors is None and fit.warnings
        assert fit.fit_info.kkt_residual > 1e-6
