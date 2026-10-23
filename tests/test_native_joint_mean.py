"""Public-boundary regressions for the joint linear-mean native integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from scivol import ARX, HARX, Normal
from scivol._joint_mean_fit import (
    JointMeanModel,
    FAMILIES,
    DENSITIES,
    fit_joint_mean_spec,
    filter_joint_mean_spec,
    _solve_joint_phase,
)
from scivol._fit_credibility import projected_stationarity_bound, observed_covariance
from scivol.result import FitMetadata


def prepared(family=0, density=0, p=1, q=1, *, n=32, har=False, fixed=True):
    rng = np.random.default_rng(214)
    y = rng.normal(size=n)
    x = rng.normal(size=(n, 4 if har else 2))
    spec = (
        (HARX([1, 3]) if har else ARX(1))
        + FAMILIES[family](p, q)
        + DENSITIES[density]()
    )
    return JointMeanModel(spec, y, x, fixed=fixed), x


@pytest.mark.parametrize("outcome", ["certified", "uncertified", "worse", "infeasible", "error"])
def test_active_bound_refinement_preserves_quality_and_charges_work(monkeypatch, outcome):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    model = SimpleNamespace(family=3, lower=np.array([0., np.nan]),
        upper=np.full(2, np.nan), request_counts=[0]*8)
    model.assess = lambda point, gradient: (
        outcome != "infeasible", 0. if outcome == "certified" else 2.,
        1. if outcome == "infeasible" else 0.)
    fitted = OptimizeResult(x=np.array([0., 1.]), jac=np.array([1e195, -1.]),
        fun=10., success=False, stationarity_bound=1., nit=55, nfev=80, njev=60,
        nhev=2, phases=(), status=4, selected_space="theta")

    def refine(model, point, **options):
        np.testing.assert_array_equal(point, fitted.x)
        np.testing.assert_array_equal(options["free_coordinates"], [1])
        assert options["maxiter"] == 45 and not options["transformed"]
        assert not options["use_curvature"] and not options["curvature_reset"]
        model.request_counts[3] += 3
        if outcome == "error":
            raise ValueError("Unavailable requested derivative")
        return OptimizeResult(x=np.array([0., .8]), jac=np.array([1e195, 0.]),
            fun=11. if outcome == "worse" else 9., success=True, status=0, nit=4)

    monkeypatch.setattr(joint, "_solve_joint_phase", refine)
    result = joint._refine_joint_active_bounds(model, fitted, accuracy=1e-12, maxiter=100)
    accepted = outcome in ("certified", "uncertified")
    assert result.fun == (9. if accepted else 10.)
    assert result.success == (outcome == "certified")
    assert result.nit == (55 if outcome == "error" else 59)
    assert (result.nfev, result.njev, result.nhev) == (83, 63, 2)
    assert result.phases[-1]["fixed_coordinates"] == [0]
    assert result.phases[-1]["iteration_budget"] == 45
    assert bool(result.phases[-1]["error"]) == (outcome == "error")
    np.testing.assert_array_equal(fitted.x, [0., 1.])
    assert model.lower[0] == 0 and np.isnan(model.upper).all()


@pytest.mark.parametrize("reason", ["certified", "exhausted", "interior", "inward", "invalid-gradient"])
def test_active_bound_refinement_skips_inapplicable_points(monkeypatch, reason):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    def forbidden(*args, **kwargs):
        raise AssertionError("No restricted solve should run")

    monkeypatch.setattr(joint, "_solve_joint_phase", forbidden)
    model = SimpleNamespace(family=3, lower=np.array([0., np.nan]),
        upper=np.full(2, np.nan), request_counts=[0]*8)
    fitted = OptimizeResult(x=np.array([1. if reason == "interior" else 0., 1.]),
        jac=None if reason == "invalid-gradient" else np.array([-1. if reason == "inward" else 1., 1.]),
        success=reason == "certified", stationarity_bound=0., nit=100 if reason == "exhausted" else 55)
    assert joint._refine_joint_active_bounds(model, fitted, accuracy=1e-12, maxiter=100) is fitted
    assert not any(model.request_counts)


@pytest.mark.parametrize("family", (0, 1, 2, 3))
@pytest.mark.parametrize("transformed,automatic", ((False, False), (True, False), (True, True)))
def test_terminal_bound_policy_respects_family_chart_and_reserve(
    monkeypatch, family, transformed, automatic
):
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    model, _ = prepared(family)
    calls = []

    def phase(model, point, **options):
        return OptimizeResult(x=point.copy(), jac=np.ones(model.size), fun=10.,
            nit=1, success=False, status=4, curvature_prepared=False, kernel_dispatch=2)

    def refine(model, fitted, **options):
        calls.append(options)
        assert fitted.nit == 2
        return fitted

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(joint, "_refine_joint_active_bounds", refine)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 1., 0.))
    joint.solve_joint(model, model.initialize(), transformed=transformed,
        allow_chart_refinement=automatic, accuracy=1e-12, maxiter=80,
        terminal_reserve=10, speculate=False, invalid_trial_policy="minimum-step")
    enabled = family == 3 and (not transformed or automatic)
    assert calls == ([dict(accuracy=1e-12, maxiter=70, speculate=False,
        invalid_trial_policy="minimum-step")] if enabled else [])


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("outcome", ["better", "worse", "invalid"])
@pytest.mark.parametrize("speculate", [False, True])
def test_continuation_retains_best_and_charges_all_work(monkeypatch, outcome, transformed, speculate):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    model = SimpleNamespace(request_counts=[0]*8)
    fitted = OptimizeResult(x=np.array([1.]), fun=10., success=False, stationarity_bound=1.,
        nit=55, nfev=80, njev=60, nhev=2, phases=(), route_budgets=[30, 30],
        total_iteration_limit=100, status=9)
    def continuation(model, initial, **options):
        assert options["maxiter"] == 45 and options["transformed"] == transformed
        assert options["invalid_trial_policy"] == "minimum-step"
        assert options["speculate"] == speculate and not options["allow_chart_refinement"]
        np.testing.assert_array_equal(initial, [1.])
        initial[0] = 42.
        model.request_counts[3] += 3
        if outcome == "invalid":
            raise ValueError("Unavailable derivative")
        return OptimizeResult(x=initial, fun=9. if outcome == "better" else 11.,
            success=False, stationarity_bound=2., status=9, nit=4, phases=())
    monkeypatch.setattr(joint, "solve_joint", continuation)
    result = joint._continue_joint_estimate(model, fitted, maxiter=100,
        accuracy=1e-12, transformed=transformed, speculate=speculate)
    np.testing.assert_array_equal(fitted.x, [1.])
    assert result.fun == (9. if outcome == "better" else 10.)
    assert result.nit == (55 if outcome == "invalid" else 59)
    assert (result.nfev, result.njev, result.nhev) == (83, 63, 2)
    assert result.total_iteration_limit == 100
    assert result.route_budgets == [30, 30] and result.route_budgets is not fitted.route_budgets
    assert result.phases[-1]["iteration_budget"] == 45
    assert result.phases[-1]["space"] == ("z" if transformed else "theta")
    assert bool(result.phases[-1]["error"]) == (outcome == "invalid")


@pytest.mark.parametrize("stop", ["certified", "kink", "exhausted"])
def test_continuation_preserves_existing_stops(monkeypatch, stop):
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    def fail(*args, **kwargs):
        raise AssertionError("No continuation should run.")
    monkeypatch.setattr(joint, "solve_joint", fail)
    fitted = OptimizeResult(success=stop == "certified", stationarity_bound=0.,
        nit=100 if stop == "exhausted" else 20,
        nearby_kink_evidence=dict(supports_kink_stop=stop == "kink"))
    assert joint._continue_joint_estimate(None, fitted, maxiter=100, accuracy=1e-12,
                                         transformed=False) is fitted


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, np.nan, np.inf])
def test_continuation_rejects_bad_budgets_before_work(budget):
    from scivol._joint_mean_fit import solve_joint_egarch

    with pytest.raises(ValueError):
        solve_joint_egarch(None, None, maxiter=budget, accuracy=1e-12)


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("speculate", [False, True])
def test_neighboring_region_refinement_respects_budget_and_coordinate_choice(monkeypatch, transformed, invalid, speculate):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint
    import scivol._nonsmooth_fit as probes

    model = SimpleNamespace(family=2, columns=1, p_vol=1, data=np.zeros(10), request_counts=[0]*8)
    fitted = OptimizeResult(x=np.array([1.]), fun=10., success=True, stationarity_bound=0.,
        nit=90, nfev=30, njev=20, nhev=1, phases=(), route_budgets=[100], total_iteration_limit=100)
    def propose(*args, **options):
        model.request_counts[1] += 4
        return dict(proposals=[dict(theta=[2.], nll=9.)])
    def solve(model, initial, **options):
        assert options["maxiter"] == 10 and options["transformed"] == transformed
        assert options["initial_curvature"] and not options["allow_chart_refinement"]
        assert options["speculate"] == speculate
        np.testing.assert_array_equal(initial, [2.])
        model.request_counts[3] += 2
        if invalid:
            raise ValueError("Unavailable coordinate map")
        return OptimizeResult(x=np.array([3.]), fun=8., success=True, stationarity_bound=0.,
            nit=3, phases=(), status=0)
    monkeypatch.setattr(probes, "neighboring_residual_proposals", propose)
    monkeypatch.setattr(joint, "solve_joint", solve)
    result = joint._refine_neighboring_regions(model, fitted, maxiter=100,
        accuracy=1e-12, transformed=transformed, speculate=speculate)
    assert result.nit == (90 if invalid else 93)
    assert (result.nfev, result.njev, result.nhev) == (36, 22, 1)
    assert result.total_iteration_limit == 100 and result.fun == (10. if invalid else 8.)
    assert result.phases[-1]["iteration_budget"] == 10
    assert bool(result.phases[-1]["error"]) == invalid
    np.testing.assert_array_equal(fitted.x, [1.])


def test_exhausted_budget_does_not_probe_regions(monkeypatch):
    from types import SimpleNamespace
    import scivol._joint_mean_fit as joint
    import scivol._nonsmooth_fit as probes

    def fail(*args, **options):
        raise AssertionError("An exhausted fit must not probe.")
    monkeypatch.setattr(probes, "neighboring_residual_proposals", fail)
    fitted = SimpleNamespace(nit=100)
    assert joint._refine_neighboring_regions(None, fitted, maxiter=100, accuracy=1e-12,
                                            transformed=False) is fitted


def test_unknown_invalid_trial_policy_fails_before_work():
    with pytest.raises(ValueError, match="invalid-trial"):
        _solve_joint_phase(None, None, transformed=False, accuracy=1e-12,
                           maxiter=10, invalid_trial_policy="unknown")


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("layout", [(4, 1, 1), (7, 3, 2), (4, 2, 1), (3, 1, 1)])
@pytest.mark.parametrize("transformed", [False, True])
def test_egarch_controller_limits_speculation_to_measured_layout(monkeypatch, density, layout, transformed):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    model = SimpleNamespace(code=density, columns=layout[0], p_vol=layout[1], q_vol=layout[2])
    fitted, calls = OptimizeResult(nit=0), []
    def phase(model, point, **options):
        calls.append(options)
        return fitted
    for name in ("solve_joint_two_routes", "_continue_joint_estimate", "_refine_neighboring_regions"):
        monkeypatch.setattr(joint, name, phase)
    result = joint.solve_joint_egarch(model, np.ones(1), accuracy=1e-12, maxiter=1000,
        transformed=transformed, allow_chart_refinement=False)
    expected = not transformed and density >= 4 and layout == (4, 1, 1)
    assert len(calls) == 3 and all(call["speculate"] == expected for call in calls)
    assert all(call["transformed"] == transformed for call in calls)
    assert result.total_iteration_limit == 1000


@pytest.mark.parametrize(
    "family,axis", [(i, "mean") for i in range(4)] + [(2, "volatility")]
)
@pytest.mark.parametrize(
    "last", [-1.0, np.nextafter(-1.0, 0.0), np.nextafter(-1.0, -2.0)]
)
def test_joint_schur_boundary_is_evaluable_but_not_silently_admitted(
    family, axis, last
):
    data = np.random.default_rng(721).normal(size=32)
    model = JointMeanModel(ARX(2) + FAMILIES[family](3, 2) + Normal(), data, None)
    theta = model.initialize()
    if axis == "mean":
        offset, row = model.blocks[0].offset, 0
    else:
        offset = model.shape_start - model.q_vol
        row = 2 * model.domain[0]
    theta[offset : offset + 2] = [0.7, last]
    denominator = 1 - last
    expected = np.array(
        [1 - 0.7 / denominator, 1 + 0.7 / denominator, 1 - last, 1 + last]
    )
    expected_jacobian = np.array(
        [
            [-1 / denominator, -0.7 / denominator**2],
            [1 / denominator, 0.7 / denominator**2],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    values, jacobian = model.constraints(theta)
    values_only, absent = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values_only, values)
    assert absent is None
    np.testing.assert_allclose(values[row : row + 4], expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        jacobian[row : row + 4, offset : offset + 2],
        expected_jacobian,
        rtol=2e-15,
        atol=2e-15,
    )
    assert model.assess(theta, None)[0] == (last > -1)


@pytest.mark.parametrize(
    "reserve,expected", [(0, [40, 20, 10, 10]), (1, [40, 20, 10, 9])]
)
def test_terminal_reserve_preserves_earlier_phase_budgets(
    monkeypatch, reserve, expected
):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model, _ = prepared()
    initial = model.initialize()
    budgets = []

    def phase(model, point, **options):
        budgets.append(options["maxiter"])
        return OptimizeResult(
            x=point.copy(),
            jac=np.ones(model.size),
            fun=-float(len(budgets)),
            nit=options["maxiter"],
            success=False,
            status=9,
            curvature_prepared=False,
            kernel_dispatch=2,
        )

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 1.0, 0.0))
    result = joint.solve_joint(
        model,
        initial,
        transformed=False,
        accuracy=1e-12,
        maxiter=80,
        terminal_reserve=reserve,
    )
    assert budgets == expected
    assert [phase["iteration_budget"] for phase in result.phases] == expected
    assert result.nit + reserve == 80


@pytest.mark.parametrize("reserve", [-1, 0.5, 80, 81, np.nan])
def test_invalid_terminal_reserve_is_rejected_before_fitting(reserve):
    from scivol._joint_mean_fit import solve_joint

    with pytest.raises(ValueError, match="terminal reserve"):
        solve_joint(
            None,
            None,
            transformed=False,
            accuracy=1e-12,
            maxiter=80,
            terminal_reserve=reserve,
        )


@pytest.mark.parametrize("family,override,reserve,first_spent,expected", [
    (2, None, 0, None, [80]),
    (2, None, 10, None, [70]),
    (2, None, 0, 12, [80, 34, 17, 17]),
    (2, None, 10, 12, [70, 34, 17, 7]),
    (2, False, 0, None, [40, 20, 10, 10]),
    (0, None, 0, None, [40, 20, 10, 10]),
    (1, None, 0, None, [40, 20, 10, 10]),
    (3, None, 0, None, [40, 20, 10, 10]),
])
def test_joint_first_phase_preserves_unspent_budget_and_terminal_reserve(
    monkeypatch, family, override, reserve, first_spent, expected
):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model, _ = prepared(family)
    budgets = []

    def phase(model, point, **options):
        budgets.append(options["maxiter"])
        spent = first_spent if len(budgets) == 1 and first_spent is not None else budgets[-1]
        return OptimizeResult(x=point.copy(), jac=np.ones(model.size),
            fun=-float(len(budgets)), nit=spent, success=False, status=9,
            curvature_prepared=False, kernel_dispatch=2)

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 1.0, 0.0))
    # This test isolates SQP budget splitting; refinement charging has separate coverage.
    monkeypatch.setattr(joint, "refine_before_restart",
        lambda model, best, phases, remaining, accuracy: (best, phases, remaining))
    result = joint.solve_joint(model, model.initialize(), transformed=False,
        accuracy=1e-12, maxiter=80, terminal_reserve=reserve, full_first_phase=override)
    assert budgets == expected
    assert [row["iteration_budget"] for row in result.phases] == expected
    assert result.nit == sum(row["iterations"] for row in result.phases) == 80-reserve


def test_egarch_early_certificate_never_forces_a_restart(monkeypatch):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model, _ = prepared(2)
    initial = model.initialize()
    calls = []

    def phase(model, point, **options):
        calls.append(options["maxiter"])
        return OptimizeResult(x=point.copy(), jac=np.zeros(model.size), fun=1.0,
            nit=12, success=True, status=0, curvature_prepared=False, kernel_dispatch=2)

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 0.0, 0.0))
    monkeypatch.setattr(model, "predicted_coordinate_gain", lambda *args: 0.0)
    results = [joint.solve_joint(model, initial, transformed=False, accuracy=1e-12,
        maxiter=80, full_first_phase=full) for full in (False, True)]
    assert calls == [40, 80]
    for result in results:
        np.testing.assert_array_equal(result.x, initial)
        assert result.nit == 12 and len(result.phases) == 1 and result.success


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("with_gradient", [False, True])
def test_joint_feasibility_is_independent_of_jacobian_availability(
    family, with_gradient, monkeypatch
):
    import ctypes
    from scivol import _native_core

    model, _ = prepared(family, p=3, q=2)
    theta = model.initialize()
    expected = model.constraints(theta)[0]
    calls = []

    def values(plan, domain, point, output, work, size):
        calls.append("values")
        target = np.ctypeslib.as_array(
            ctypes.cast(output, ctypes.POINTER(ctypes.c_double)), shape=(model.rows,)
        )
        target[:] = expected
        return True

    def failed_jacobian(*args):
        calls.append("jacobian")
        return False

    monkeypatch.setattr(
        _native_core, "_linear_joint_constraint_values", values, raising=False
    )
    monkeypatch.setattr(_native_core, "_linear_joint_constraints", failed_jacobian)
    feasible, bound, violation = model.assess(
        theta, np.zeros(model.size) if with_gradient else None
    )
    # The complete zero gradient certifies stationarity without a Jacobian.
    assert feasible and violation == 0 and bound == (0. if with_gradient else np.inf)
    assert calls == (["jacobian", "values"] if with_gradient else ["values"])
    expected[0] = -1.0
    assert not model.assess(theta, None)[0]


@pytest.mark.parametrize("family", range(4))
def test_joint_value_request_supports_older_compiled_extension(family, monkeypatch):
    from scivol import _native_core

    monkeypatch.delattr(_native_core, "_linear_joint_constraint_values", raising=False)
    model, _ = prepared(family)
    theta = model.initialize()
    expected = model.constraints(theta)[0]
    values, jacobian = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, expected)
    assert jacobian is None
    assert model.assess(theta, None) == (True, np.inf, 0.0)


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
def test_compiled_joint_constraint_value_entry(family, p, q, monkeypatch):
    from scivol import _native_core

    if not hasattr(_native_core, "_linear_joint_constraint_values"):
        pytest.skip("Rebuilt native extension required for the new value-only ABI")
    model, _ = prepared(family, p=p, q=q)
    theta = model.initialize()
    expected = model.constraints(theta)[0]
    monkeypatch.setattr(_native_core, "_linear_joint_constraints", lambda *args: False)
    values, jacobian = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, expected)
    assert jacobian is None and model.assess(theta, None)[0]


def test_joint_assessment_overflow_does_not_change_feasibility():
    import warnings

    model, _ = prepared()
    theta = model.initialize()
    model.units[0] = 1e308
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert model.assess(theta, np.full(model.size, 1e308)) == (True, np.inf, 0.0)


@pytest.mark.parametrize("family", range(4))
def test_joint_assessment_backend_uses_original_domain_and_point_units(family):
    model, _ = prepared(family, p=3, q=2)
    theta = model.initialize()
    gradient = np.linspace(-0.2, 0.4, model.size)
    captured = []

    def backend(*geometry):
        captured.append(tuple(value.copy() for value in geometry))
        return 0.125

    for power in (1.7, 0.9):
        if family == 3:
            theta[model.shape_start - 1] = power
        feasible, bound, violation = model.assess(
            theta, gradient, stationarity_solver=backend
        )
        assert feasible and bound == 0.125 and violation == 0
        units = model.units.copy()
        if family != 2:
            units[model.columns] = model.h0 ** (power / 2 if family == 3 else 1)
        slack, jacobian = model.constraints(theta)
        lower = np.where(np.isnan(model.lower), -np.inf, model.lower)
        upper = np.where(np.isnan(model.upper), np.inf, model.upper)
        expected = (
            gradient * units / len(model.data),
            (lower - theta) / units,
            (upper - theta) / units,
            jacobian * units,
            slack,
        )
        for actual, value in zip(captured[-1], expected):
            np.testing.assert_array_equal(actual, value)
    calls = len(captured)
    assert model.assess(theta, None, stationarity_solver=backend) == (True, np.inf, 0.0)
    infeasible = theta.copy()
    infeasible[0] = np.inf
    assert not model.assess(infeasible, gradient, stationarity_solver=backend)[0]
    assert len(captured) == calls


@pytest.mark.parametrize("failure", [np.nan, np.inf, -1.0])
def test_joint_assessment_backend_cannot_return_an_invalid_certificate(failure):
    model, _ = prepared()
    assert model.assess(
        model.initialize(),
        np.zeros(model.size),
        stationarity_solver=lambda *args: failure,
    ) == (True, np.inf, 0.0)


@pytest.mark.parametrize("free_gradient", [0., 1e-5, np.nan, np.inf])
def test_joint_assessment_without_constraint_jacobian_keeps_full_feasibility(
    monkeypatch, free_gradient,
):
    model, _ = prepared()
    initial = model.initialize()
    original = model.constraints

    def unavailable(theta, derivatives=True):
        if derivatives:
            raise ValueError("Unrepresentable constraint Jacobian")
        return original(theta, derivatives=False)

    monkeypatch.setattr(model, "constraints", unavailable)
    gradient = np.zeros(model.size)
    gradient[0] = free_gradient*len(model.data)/model.units[0]
    feasible, bound, violation = model.assess(initial, gradient)
    assert feasible and violation == 0
    if np.isfinite(free_gradient):
        assert free_gradient <= bound <= free_gradient*(1+1e-12)
    else:
        assert bound == np.inf
    assert model.assess(initial, None)[1] == np.inf
    # An external backend is not given fabricated nonlinear derivatives.
    assert model.assess(initial, gradient, stationarity_solver=lambda *_: 0.)[1] == np.inf
    initial[model.columns] = -1.
    assert not model.assess(initial, gradient)[0]


def test_joint_curvature_reset_preserves_likelihood_and_reports_each_phase():
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint

    y, x = comparison_sample(2785, True, "normal")
    spec = HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[6]()
    results = []
    for reset in (False, True):
        model = JointMeanModel(spec, y, x)
        result = solve_joint(
            model,
            model.initialize(),
            transformed=False,
            accuracy=1e-12,
            maxiter=1000,
            allow_chart_refinement=True,
            curvature_reset=reset,
        )
        gradient = model.evaluate(result.x, 2)[1]
        feasible, residual, _ = model.assess(result.x, gradient)
        assert feasible and residual < 1e-6
        assert all(p["constraint_curvature_reset"] == reset for p in result.phases)
        assert all(0 <= p["constraint_curvature_resets"] <= 1 for p in result.phases)
        resets = sum(p["constraint_curvature_resets"] for p in result.phases)
        assert (resets > 0) == reset
        results.append(result)
    assert results[1].fun <= results[0].fun + 2e-7


@pytest.mark.parametrize("family", range(4))
def test_joint_reset_default_is_explicit_and_serializable(family):
    model, x = prepared(family, n=60)
    result = fit_joint_mean_spec(
        model.spec, np.random.default_rng(214).normal(size=60), x=x, maxiter=12
    )
    phases = result.fit_info.phase_diagnostics
    assert phases
    assert all(p["constraint_curvature_reset"] is False for p in phases)
    assert all(isinstance(p["constraint_curvature_resets"], int) for p in phases)
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.to_dict() == result.fit_info.to_dict()


def test_joint_curvature_reset_state_is_per_solve():
    from concurrent.futures import ThreadPoolExecutor
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint

    y, x = comparison_sample(2785, True, "normal")

    def run():
        spec = HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[6]()
        model = JointMeanModel(spec, y, x)
        return solve_joint(
            model,
            model.initialize(),
            transformed=False,
            accuracy=1e-12,
            maxiter=1000,
            allow_chart_refinement=True,
            curvature_reset=True,
        )

    expected = run()
    assert any(p["constraint_curvature_resets"] for p in expected.phases)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(4)]
        results = [future.result() for future in futures]
    for actual in results:
        np.testing.assert_array_equal(actual.x, expected.x)
        assert actual.fun == expected.fun and actual.phases == expected.phases


def test_aparch_default_preserves_skewt_quality_without_discarding_sqp_state():
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint

    y, x = comparison_sample(2786, True, "persistent")
    spec = HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[2]()
    results = []
    for reset in (None, True):
        model = JointMeanModel(spec, y, x)
        fit = solve_joint(model, model.initialize(), transformed=False,
            accuracy=1e-12, maxiter=1000, allow_chart_refinement=True,
            curvature_reset=reset)
        value, gradient = model.evaluate(fit.x, 3)[:2]
        feasible, residual, violation = model.assess(fit.x, gradient)
        assert fit.success and feasible and residual <= 1e-6 and violation == 0.
        assert abs(fit.fun-value) <= 1e-9
        results.append(fit)
    ordinary, reset = results
    assert ordinary.fun <= reset.fun + 1e-9
    assert ordinary.nfev < reset.nfev and ordinary.njev < reset.njev
    assert all(not phase['constraint_curvature_reset'] for phase in ordinary.phases)
    assert any(phase['constraint_curvature_resets'] for phase in reset.phases)


@pytest.mark.parametrize("failure", ("invalid", "nonfinite", "unselected"))
@pytest.mark.parametrize("transformed", (False, True))
def test_egarch_phase_does_not_reuse_unavailable_or_unselected_gradient(monkeypatch, failure, transformed):
    import ctypes
    from scivol import _native_core

    model, _ = prepared(2)
    native = _native_core._linear_joint_fit
    def damaged(*args):
        valid = native(*args)
        if failure == "invalid":
            return False
        index = 12 if failure == "nonfinite" else 11
        output = np.ctypeslib.as_array(ctypes.cast(args[index],
            ctypes.POINTER(ctypes.c_double)),shape=(model.size,))
        output[0] = np.nan
        return valid
    monkeypatch.setattr(_native_core, "_linear_joint_fit", damaged)
    result = _solve_joint_phase(
        model,
        model.initialize(),
        transformed=transformed,
        accuracy=1e-12,
        maxiter=1,
        use_curvature=False,
    )
    assert result.jac is None


@pytest.mark.parametrize("invalid", [0., -1., np.nan, np.inf])
def test_joint_phase_rejects_invalid_stationarity_tolerance(invalid):
    with pytest.raises(ValueError, match="finite positive stationarity tolerance"):
        _solve_joint_phase(None, None, transformed=False, accuracy=1e-12,
                           maxiter=1, stationarity_tolerance=invalid)


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("accuracy,stationarity_tolerance,credible", [
    (1e-12, None, True), (np.finfo(float).eps, None, False),
    (np.finfo(float).eps, 1e-6, True),
])
@pytest.mark.parametrize("outcome", [
    "certified", "uncertified", "infeasible", "invalid-gradient", "nonfinite-gradient",
    "invalid-solve", "failed-solve",
])
def test_retained_seed_uses_its_own_stationarity_evidence(
    monkeypatch, transformed, outcome, accuracy, stationarity_tolerance, credible
):
    import ctypes
    from scivol import _native_core

    model, _ = prepared(3)
    initial = model.initialize()
    returned = initial.copy()
    returned[0] += .1
    own_gradient = np.arange(model.size, dtype=float)
    derivative_points = []

    def evaluate(point, request, **options):
        model.request_counts[request] += 1
        if request == 1:
            return (11. if np.array_equal(point, returned) else 10., None, None)
        assert request == 2
        derivative_points.append(point.copy())
        if outcome == "invalid-gradient":
            raise ValueError("Unavailable retained-point derivative")
        derivative = (np.full(model.size, np.nan) if outcome == "nonfinite-gradient"
                      else own_gradient.copy())
        return None, derivative, None

    def assess(point, derivative):
        if derivative is None:
            return True, np.inf, 0.
        np.testing.assert_array_equal(point, initial)
        np.testing.assert_array_equal(derivative, own_gradient)
        return outcome != "infeasible", 1. if outcome == "uncertified" else 1e-7, 0.

    def solve(*args):
        for index, values in ((11, returned), (12, np.full(model.size, -99.))):
            output = np.ctypeslib.as_array(ctypes.cast(args[index],
                ctypes.POINTER(ctypes.c_double)), shape=(model.size,))
            output[:] = values
        stats = np.ctypeslib.as_array(ctypes.cast(args[14],
            ctypes.POINTER(ctypes.c_int64)), shape=(6,))
        stats[:] = [9 if outcome == "failed-solve" else 0, 1, 1, 1, 0, 2]
        return outcome != "invalid-solve"

    monkeypatch.setattr(model, "evaluate", evaluate)
    monkeypatch.setattr(model, "assess", assess)
    monkeypatch.setattr(_native_core, "_linear_joint_fit", solve)
    result = _solve_joint_phase(model, initial, transformed=transformed,
        accuracy=accuracy, maxiter=10, use_curvature=False,
        stationarity_tolerance=stationarity_tolerance)
    checked = outcome not in ("invalid-solve", "failed-solve")
    valid_gradient = checked and outcome not in ("invalid-gradient", "nonfinite-gradient")
    np.testing.assert_array_equal(result.x, initial)
    assert result.fun == 10. and result.selected == "initial"
    assert result.success == (outcome == "certified" and credible)
    assert result.stationarity_tolerance == (
        max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
        if stationarity_tolerance is None else stationarity_tolerance)
    assert result.njev == len(derivative_points) == int(checked)
    if valid_gradient:
        np.testing.assert_array_equal(result.jac, own_gradient)
    else:
        assert result.jac is None


@pytest.mark.parametrize("difference", ["none", "point", "value", "stationarity", "failed"])
def test_refinement_success_tie_requires_identical_estimate(difference):
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import _retain_joint_refinement

    fitted = OptimizeResult(x=np.array([1.]), fun=10., stationarity_bound=1e-8,
        success=False, status=-105, phases=(), nit=2, nfev=3, njev=4, nhev=5)
    child = OptimizeResult(x=np.array([2. if difference == "point" else 1.]),
        fun=11. if difference == "value" else 10.,
        stationarity_bound=2e-8 if difference == "stationarity" else 1e-8,
        success=difference != "failed", status=0)
    phase = dict(iterations=1, function_evaluations=2, gradient_evaluations=3,
                 hessian_evaluations=4)
    result = _retain_joint_refinement(fitted, child, phase)
    assert result.success == (difference == "none")
    assert result.status == (0 if difference == "none" else -105)
    assert result.nit == 3 and result.nfev == 5 and result.njev == 7 and result.nhev == 9
    np.testing.assert_array_equal(result.x, fitted.x)
    assert not fitted.success and fitted.status == -105


def test_joint_controller_retains_clean_termination_at_identical_point(monkeypatch):
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    model, _ = prepared()
    calls = []

    def phase(model, point, **options):
        assert options["stationarity_tolerance"] == 1e-6
        calls.append(options)
        return OptimizeResult(x=point.copy(), jac=np.zeros(model.size), fun=10.,
            nit=1, success=len(calls) > 1, status=0 if len(calls) > 1 else -105,
            curvature_prepared=False, kernel_dispatch=2,
            jet_requests=joint._describe_jet_requests([0] * 8))

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 1e-8, 0.))
    monkeypatch.setattr(model, "predicted_coordinate_gain", lambda *args: 0.)
    initial = model.initialize()
    model.evaluate(initial, 7)  # Prior work must not enter the controller's totals.
    result = joint.solve_joint(model, initial, transformed=False,
        accuracy=1e-12, maxiter=20)
    assert len(calls) == 2
    assert result.success and result.status == 0 and result.stationarity_bound == 1e-8
    assert result.fun == 10. and result.nit == 2
    assert result.jet_requests["H"] == result.nhev == 1
    assert result.jet_requests["F"] == result.nfev == 0
    assert result.jet_requests["G"] == result.njev == 0
    assert result.jet_requests["masks"]["H"] == 1
    assert result.jet_requests["masks"]["F|G|H"] == 0


def test_box_limited_scales_preserve_dyadic_roundtrips_and_unbounded_axes():
    from scivol._joint_mean_fit import _limit_scales_to_box

    below_two = np.nextafter(2., 0.)
    tiny = np.nextafter(0., 1.)
    lower = np.array([0., 0., -1., 0., np.nan, -np.inf, -1e308, 0., 0.])
    upper = np.array([below_two, 2., 1., tiny, 1., np.inf, 1e308, 0., .75])
    scales = np.array([8., 8., .25, 8., 8., 8., 8., 8., 8.])
    expected = np.array([1., 2., .25, tiny, 8., 8., 8., 8., .5])
    limited = _limit_scales_to_box(scales, lower, upper)
    np.testing.assert_array_equal(limited, expected)
    np.testing.assert_array_equal(scales, [8., 8., .25, 8., 8., 8., 8., 8., 8.])
    assert np.all(limited > 0)
    # Exact powers of two avoid changing an active bound on scaling round-trip.
    for point in (lower, upper):
        finite = np.isfinite(point)
        np.testing.assert_array_equal(
            point[finite] / limited[finite] * limited[finite], point[finite]
        )


@pytest.mark.parametrize("family,p,density", [
    (3, 3, 2), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7),
    (3, 3, 0), (3, 3, 1), (3, 3, 3), (3, 1, 4), (2, 3, 4),
])
@pytest.mark.parametrize("status", ["certified", "uncertified", "failed"])
def test_box_metric_is_only_used_for_validated_precision_refinement(
    monkeypatch, family, p, density, status
):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint

    calls, assessments = [], []
    model = SimpleNamespace(family=family, p_vol=p, code=density, columns=0,
        size=2, data=np.ones(10), request_counts=[0]*8)
    model.evaluate = lambda *args: (None, None, np.eye(2))
    model.predicted_coordinate_gain = lambda *args: 1. if len(calls) == 1 else 0.

    def assess(point, gradient):
        assessments.append(point.copy())
        return True, 1. if status == "uncertified" and len(calls) == 1 else 0., 0.

    def phase(model, point, **options):
        calls.append(options)
        return OptimizeResult(x=point.copy(), jac=np.zeros(2), fun=10.-len(calls),
            nit=1, success=status != "failed" or len(calls) > 1, status=0,
            curvature_prepared=options["use_curvature"], kernel_dispatch=2,
            curvature_scale_limits=[0] if options["limit_curvature_scale"] else [])

    model.assess = assess
    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(joint, "refinement_chart_available", lambda *args: True)
    monkeypatch.setattr(joint, "refine_before_restart",
                        lambda model, best, phases, remaining, accuracy: (best, phases, remaining))
    fitted = joint.solve_joint(model, np.ones(2), transformed=True,
        accuracy=1e-12, maxiter=20, allow_chart_refinement=True)
    enabled = (family == 3 and p > 1 and density in (2, 4, 5, 6, 7)
               and status == "certified")
    assert len(calls) == len(assessments) == 2
    assert not calls[0]["limit_curvature_scale"]
    assert calls[1]["limit_curvature_scale"] == enabled
    assert fitted.phases[1]["curvature_scale_limits"] == ([0] if enabled else [])
    assert all(call["stationarity_tolerance"] == 1e-6 for call in calls)
    assert calls[1]["accuracy"] == np.finfo(float).eps


@pytest.mark.parametrize("density,retained_nll", [
    (4, 535.7871140848002), (7, 535.7870515530456),
])
def test_box_limited_aparch_refinement_preserves_likelihood_and_reporting(
    density, retained_nll
):
    import time
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint, assemble_joint_fit

    data, x = comparison_sample(2785, True, "normal")
    model = JointMeanModel(HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[density](), data, x)
    bounds = (model.lower.copy(), model.upper.copy())
    started = time.perf_counter()
    fitted = solve_joint(model, model.initialize(), transformed=True,
        accuracy=1e-12, maxiter=1000, allow_chart_refinement=True)
    assert fitted.success and fitted.stationarity_bound <= 1e-6
    assert fitted.fun <= retained_nll + 1e-9
    assert not fitted.phases[0]["curvature_scale_limits"]
    assert any(phase["curvature_scale_limits"] for phase in fitted.phases[1:])
    reported = assemble_joint_fit(model, fitted, True, 1e-12, 1000, started)
    assert reported.converged and reported.fit_info.stationarity_status == "certified"
    assert reported.fit_info.maximum_constraint_violation == 0
    assert reported.fit_info.jet_consistency["status"] == "consistent"
    assert reported.fit_info.phase_diagnostics == fitted.phases
    assert -reported.log_likelihood <= retained_nll + 1e-9
    for actual, original in zip((model.lower, model.upper), bounds):
        np.testing.assert_array_equal(actual, original)


def test_aparch_z_refinement_certifies_unchanged_retained_point():
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint

    data, x = comparison_sample(2785, True, "normal")
    model = JointMeanModel(HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[6](), data, x)
    fitted = solve_joint(model, model.initialize(), transformed=True,
        initial_curvature=True, allow_chart_refinement=True, accuracy=1e-12, maxiter=1900)
    feasible, residual, _ = model.assess(fitted.x, fitted.jac)
    assert fitted.success and fitted.status == 0 and feasible and residual <= 1e-6
    assert fitted.fun <= 535.9488652917788 + 1e-9


@pytest.mark.parametrize("density", (4, 7))
@pytest.mark.parametrize("wide", (False, True))
def test_aparch_prepared_moments_are_owned_by_each_solve(density, wide):
    from concurrent.futures import ThreadPoolExecutor

    def run():
        model, _ = prepared(
            3, density, p=3 if wide else 1, q=2 if wide else 1, har=wide
        )
        return _solve_joint_phase(
            model,
            model.initialize(),
            transformed=False,
            accuracy=1e-12,
            maxiter=8,
            use_curvature=False,
        )

    expected = run()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(4)))
    for result in results:
        np.testing.assert_array_equal(result.x, expected.x)
        np.testing.assert_array_equal(result.jac, expected.jac)
        assert (
            result.fun == expected.fun and result.jet_requests == expected.jet_requests
        )


APARCH_PERSISTENCE_POINTS = json.loads(
    Path(__file__).with_name("aparch_persistence_points.json").read_text()
)["cases"]


@pytest.mark.parametrize("point", APARCH_PERSISTENCE_POINTS, ids=lambda point: point["id"])
@pytest.mark.parametrize("fixed", [False, True])
def test_aparch_boundary_domain_uses_one_resolved_value(point, fixed):
    model, _ = prepared(3, point["density"], p=3 if point["wide"] else 1,
                        q=2 if point["wide"] else 1, har=point["wide"], fixed=fixed)
    theta = np.array(point["theta"])
    original = theta.copy()
    expected = float(point["independent"][-1]["slack"])
    coarse_reference = float(point["independent"][0]["slack"])
    assert abs(expected - coarse_reference) < 1e-35
    row = 2 * model.domain[0]
    values, jacobian = model.constraints(theta)
    values_only, absent = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, values_only)
    assert absent is None and np.isfinite(jacobian).all()
    assert abs(values[row] - expected) < 3e-27
    assert model.assess(theta, None)[0] == (expected > 0)
    for mask in range(1, 8):
        if expected > 0:
            outputs = model.evaluate(theta, mask)
            for bit, output in zip((1, 2, 4), outputs[:3]):
                if mask & bit:
                    assert np.isfinite(output).all()
                else:
                    assert output is None
        else:
            with pytest.raises(ValueError, match="Joint jet request"):
                model.evaluate(theta, mask)
    # The inverse chart has its own declared representability margin; it may
    # not turn one of these boundary points into a different feasible estimate.
    with pytest.raises(ValueError, match="outside the transformed chart"):
        model.inverse(theta)
    np.testing.assert_array_equal(theta, original)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("p,q", [(1, 1), (3, 2), (4, 3)])
@pytest.mark.parametrize("inside", [False, True])
def test_aparch_unit_variance_boundary_has_no_density_padding(density, p, q, inside):
    model, _ = prepared(3, density, p=p, q=q)
    theta = model.initialize()
    alpha, gamma = model.columns + 1, model.columns + 1 + p
    beta, delta = gamma + p, model.shape_start - 1
    theta[alpha:gamma] = 0.
    theta[alpha] = .25
    theta[gamma:beta] = 0.
    theta[beta:delta] = 0.
    theta[beta] = np.nextafter(.75, 0. if inside else 1.)
    theta[delta] = 2.
    expected = .75 - theta[beta]
    row = 2 * model.domain[0]
    values, _ = model.constraints(theta, derivatives=False)
    assert values[row] == expected
    assert model.assess(theta, None)[0] == inside
    if inside:
        assert np.isfinite(model.evaluate(theta, 1)[0])
    else:
        with pytest.raises(ValueError, match="Joint jet request"):
            model.evaluate(theta, 1)


@pytest.mark.parametrize("nu", [1e8, 1e15, 1e100])
@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("inside", [False, True])
def test_aparch_student_gaussian_limit_strict_boundary(nu, p, q, fixed, inside):
    mp = pytest.importorskip("mpmath")
    model, _ = prepared(3, 1, p=p, q=q, fixed=fixed)
    theta = model.initialize()
    alpha, gamma = model.columns + 1, model.columns + 1 + p
    beta, delta = gamma + p, model.shape_start - 1
    theta[alpha:gamma] = 0.
    theta[alpha] = .25
    theta[gamma:beta] = .15
    theta[beta:delta] = 0.
    theta[delta], theta[model.shape_start] = 1.7, nu
    with mp.workdps(160):
        n, d, asymmetry = mp.mpf(nu), mp.mpf(1.7), mp.mpf(.15)
        moment = mp.exp(d/2*mp.log(n-2)+mp.loggamma((d+1)/2)
            +mp.loggamma((n-d)/2)-mp.loggamma(n/2)-mp.log(mp.pi)/2)
        moment *= ((1-asymmetry)**d+(1+asymmetry)**d)/2
        boundary = 1-mp.mpf(.25)*moment
        theta[beta] = np.nextafter(float(boundary), 0. if inside else 1.)
        expected = float(boundary-mp.mpf(theta[beta]))
    assert (expected > 0) == inside
    row = 2 * model.domain[0]
    values, jacobian = model.constraints(theta)
    scalar, _ = model.constraints(theta, derivatives=False)
    np.testing.assert_array_equal(values, scalar)
    assert np.isfinite(jacobian).all() and abs(values[row]-expected) < 3e-27
    assert model.assess(theta, None)[0] == inside
    for mask in range(1, 8):
        if inside:
            outputs = model.evaluate(theta, mask)
            assert all(output is None or np.isfinite(output).all() for output in outputs[:3])
        else:
            with pytest.raises(ValueError, match="Joint jet request"):
                model.evaluate(theta, mask)


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
def test_phase_retains_checked_original_parameter_gradient(
    family, density, transformed
):
    model, _ = prepared(family, density, p=3, q=2, n=24, har=True)
    phase = _solve_joint_phase(
        model,
        model.initialize(),
        transformed=transformed,
        accuracy=1e-12,
        maxiter=1,
        use_curvature=False,
    )
    assert phase.selected == "returned" and phase.jac is not None
    expected = model.evaluate(phase.x, 2)[1]
    np.testing.assert_allclose(phase.jac, expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("family", range(4))
def test_joint_reuses_phase_gradient_without_an_extra_python_request(
    family, monkeypatch
):
    from scivol._joint_mean_fit import solve_joint

    model, _ = prepared(family, n=40)
    original = model.evaluate
    masks = []

    def record(theta, request=7, **kwargs):
        masks.append(request)
        return original(theta, request, **kwargs)

    monkeypatch.setattr(model, "evaluate", record)
    result = solve_joint(
        model,
        model.initialize(),
        transformed=False,
        accuracy=1e-12,
        maxiter=8,
        allow_chart_refinement=True,
    )
    assert result.jac is not None and 2 not in masks


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("transformed", (False, True))
def test_phase_request_counts_reconcile_with_independent_boundary_spies(
    family, transformed, monkeypatch
):
    import ctypes
    from scivol import _native_core

    model, _ = prepared(family)
    initial = model.initialize()
    requests, native = [], []
    jet, solve = _native_core._linear_joint_prevalidated, _native_core._linear_joint_fit

    def record_request(*args):
        requests.append(args[5])
        return jet(*args)

    def record_solve(*args):
        result = solve(*args)
        counts = np.ctypeslib.as_array(
            ctypes.cast(args[21], ctypes.POINTER(ctypes.c_int64)), shape=(3,)
        ).copy()
        stats = np.ctypeslib.as_array(
            ctypes.cast(args[14], ctypes.POINTER(ctypes.c_int64)), shape=(6,)
        ).copy()
        assert counts[0] + counts[2] == stats[2]
        assert counts[1] + counts[2] == stats[3]
        native.append(counts)
        return result

    monkeypatch.setattr(_native_core, "_linear_joint_prevalidated", record_request)
    monkeypatch.setattr(_native_core, "_linear_joint_fit", record_solve)
    phase = _solve_joint_phase(
        model,
        initial,
        transformed=transformed,
        accuracy=1e-12,
        maxiter=3,
        use_curvature=True,
    )
    expected = np.bincount(requests, minlength=8)
    expected[1:4] += native[0]
    assert list(phase.jet_requests["masks"].values()) == list(expected[1:])
    assert phase.nfev == sum(expected[mask] for mask in range(8) if mask & 1)
    assert phase.njev == sum(expected[mask] for mask in range(8) if mask & 2)
    assert phase.nhev == 1
    assert phase.native_jet_requests["F|G"] >= 2


def test_rejected_joint_request_is_counted_but_shape_validation_is_not():
    model, _ = prepared()
    initial = model.initialize()
    before = tuple(model.request_counts)
    with pytest.raises(ValueError, match="number"):
        model.evaluate(initial[:-1], 7)
    assert tuple(model.request_counts) == before
    initial[0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        model.evaluate(initial, 7)
    assert model.request_counts[7] == before[7] + 1


def test_fit_request_counts_include_initialization_phases_and_inference():
    model, x = prepared()
    result = fit_joint_mean_spec(model.spec, model.data, x=x, maxiter=3)
    info = result.fit_info
    masks = dict.fromkeys(info.jet_requests["masks"], 0)
    masks["F"] = 1  # Initial point validation precedes optimizer phases.
    for stage in (*info.phase_diagnostics, {"jet_requests": info.postfit_jet_requests}):
        for key, value in stage["jet_requests"]["masks"].items():
            masks[key] += value
    assert masks == info.jet_requests["masks"]
    assert info.function_evaluations == info.jet_requests["F"]
    assert info.gradient_evaluations == info.jet_requests["G"]
    assert info.hessian_evaluations == info.jet_requests["H"] >= 1
    import json

    serialized = json.loads(json.dumps(info.to_dict()))
    restored = FitMetadata(**serialized)
    assert restored.jet_requests == info.jet_requests
    assert restored.postfit_jet_requests == info.postfit_jet_requests
    assert restored.request_count_scope == info.request_count_scope


@pytest.mark.parametrize("disagreement", ["value", "gradient"])
def test_joint_fit_does_not_certify_inconsistent_requests_at_active_bounds(
    monkeypatch, capsys, disagreement
):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared()
    theta = model.initialize()
    alpha = model.columns + 1
    theta[alpha] = 0.0
    value, _, _, residuals, variance = model.evaluate(theta, 7, paths=True)
    retained_gradient = np.zeros(model.size)
    retained_gradient[alpha] = 1.0
    reporting_gradient = retained_gradient.copy()
    if disagreement == "gradient":
        reporting_gradient[alpha] = 2.0
    # Both projected gradients vanish; their scalar bounds conceal disagreement.
    assert model.assess(theta, retained_gradient)[1] == 0.0
    assert model.assess(theta, reporting_gradient)[1] == 0.0
    reported_value = value + (0.1 if disagreement == "value" else 0.0)
    evaluated = []

    def evidence(point, request, **kwargs):
        evaluated.append(request)
        np.testing.assert_array_equal(point, theta)
        return (
            reported_value,
            reporting_gradient,
            np.eye(model.size),
            residuals,
            variance,
        )

    monkeypatch.setattr(model, "evaluate", evidence)
    optimized = OptimizeResult(
        x=theta.copy(),
        fun=value,
        jac=retained_gradient,
        status=0,
        success=True,
        stationarity_bound=0.0,
        phases=(),
        selected_space="theta",
    )
    fitted = assemble_joint_fit(model, optimized, False, 1e-12, 1, time.perf_counter())
    np.testing.assert_array_equal(fitted.params, theta)
    assert evaluated == [7]
    assert fitted.fit_info.optimizer_terminated
    assert not fitted.converged
    assert fitted.fit_info.stationarity_status == "uncertified"
    assert fitted.fit_info.inference_status == "unavailable"
    assert fitted.fit_info.jet_consistency["status"] == "inconsistent"
    assert fitted.std_errors is None
    assert any("Jet requests disagree" in warning for warning in fitted.warnings)
    assert capsys.readouterr().out == ""
    restored = FitMetadata(**fitted.fit_info.to_dict())
    assert restored.jet_consistency == fitted.fit_info.jet_consistency


def test_jet_agreement_uses_optimizer_units_and_preserves_inputs():
    from scivol._fit_credibility import jet_agreement

    before, after, units = (
        np.array([2.0, 3.0]),
        np.array([2.1, 3.2]),
        np.array([1.0, 2.0]),
    )
    report = jet_agreement(4.0, before, 4.01, after, units, 10, 1e-12)
    assert report["status"] == "inconsistent"
    assert report["value_error"] == pytest.approx(0.001)
    assert report["gradient_error"] == pytest.approx(np.hypot(0.01, 0.04))
    np.testing.assert_array_equal(before, [2, 3])
    np.testing.assert_array_equal(after, [2.1, 3.2])
    report = jet_agreement(4.0, before, 4.0, before, units, 10, 1e-12)
    assert report["status"] == "consistent"
    report = jet_agreement(4.0, None, 4.0, before, units, 10, 1e-12)
    assert report["status"] == "unavailable" and report["gradient_error"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
def test_jet_agreement_rejects_unresolved_arithmetic_quietly(bad):
    import warnings
    from scivol._fit_credibility import jet_agreement

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = jet_agreement(
            1.0, np.array([-1e308]), bad, np.array([bad]), np.ones(1), 1, 1e-12
        )
    assert report["status"] == "inconsistent"


@pytest.mark.parametrize("magnitude", [1e100, 1e152, 1e300])
@pytest.mark.parametrize("sign", [-1, 1])
def test_jet_agreement_separates_roundoff_from_component_errors(magnitude, sign):
    from scivol._fit_credibility import jet_agreement

    before = np.array([sign*magnitude, 1.0, 0.0])
    after = before.copy()
    for _ in range(3):
        after[0] = np.nextafter(after[0], np.inf)
    after[1] += 1e-10
    units = np.array([1.0, 2.0, 3.0])
    compare = lambda g: jet_agreement(4.0, before, 4.0, g, units, 100, 1e-12)
    report = compare(after)
    assert report["status"] == "consistent"
    assert report["gradient_error"] > report["gradient_tolerance"]
    assert report["gradient_error_above_roundoff"] < report["gradient_tolerance"]
    assert report["gradient_roundoff_rtol"] == 8*np.finfo(float).eps
    for component, error in ((0, -2*sign*magnitude), (0, 1e-10*magnitude), (1, .1), (2, .1)):
        defective = after.copy()
        defective[component] += error
        assert compare(defective)["status"] == "inconsistent"
    np.testing.assert_array_equal(before, [sign*magnitude, 1.0, 0.0])
    np.testing.assert_array_equal(units, [1.0, 2.0, 3.0])


def test_jet_agreement_roundoff_does_not_relax_value_agreement():
    from scivol._fit_credibility import jet_agreement

    g = np.array([1e152])
    report = jet_agreement(4.0, g, 4.0+1e-6, g, np.ones(1), 100, 1e-12)
    assert report["status"] == "inconsistent"
    assert report["value_error"] > report["value_tolerance"]
    assert report["gradient_error_above_roundoff"] == 0


@pytest.mark.parametrize("steps,status", [(8, "consistent"), (9, "inconsistent")])
def test_jet_agreement_roundoff_boundary_is_explicit(steps, status):
    from scivol._fit_credibility import jet_agreement

    before = np.array([np.ldexp(1.0, 500)])
    after = before.copy()
    for _ in range(steps):
        after = np.nextafter(after, np.inf)
    before.flags.writeable = after.flags.writeable = False
    snapshots = before.tobytes(), after.tobytes()
    report = jet_agreement(1.0, before, 1.0, after, np.ones(1), 1, 1e-12)
    assert report["status"] == status
    assert report["gradient_error"] == after[0]-before[0]
    assert (before.tobytes(), after.tobytes()) == snapshots
    restored = FitMetadata(**json.loads(json.dumps(
        FitMetadata(jet_consistency=report).to_dict(), allow_nan=False)))
    assert restored.jet_consistency == report


@pytest.mark.parametrize("before,after,units,n_eff", [
    ([0., 0.], [8e-7, 8e-7], [1., 1.], 1),
    ([1e152, 0.], [1e152, 1e-12], [1., 1e8], 1),
    ([1e300], [1e300], [1e30], 1),
])
def test_jet_agreement_rejects_excess_norm_and_allowance_overflow(before, after, units, n_eff):
    import warnings
    from scivol._fit_credibility import jet_agreement

    before, after, units = map(np.array, (before, after, units))
    raw_error = np.hypot.reduce((after-before)*units/n_eff, initial=0.)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = jet_agreement(1., before, 1., after, units, n_eff, 1e-12)
    assert report["status"] == "inconsistent"
    assert report["gradient_error"] == raw_error
    assert report["gradient_error_above_roundoff"] > report["gradient_tolerance"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("previous", [False, True])
def test_jet_agreement_rejects_nonfinite_gradient_with_equal_values(bad, previous):
    import warnings
    from scivol._fit_credibility import jet_agreement

    before, after = np.array([bad]), np.array([1e152])
    if previous:
        before, after = after, before
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = jet_agreement(1., before, 1., after, np.ones(1), 1, 1e-12)
    assert report["status"] == "inconsistent"


@pytest.mark.parametrize("retained,reported", [(1.2e-6, .4e-6), (.4e-6, 1.2e-6), (.4e-6, .5e-6)])
@pytest.mark.parametrize("cached_hessian", [False, True])
def test_roundoff_agreement_cannot_bypass_either_stationarity_check(
    monkeypatch, retained, reported, cached_hessian
):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared(3)
    theta = model.initialize()
    alpha = model.columns+1
    theta[alpha] = 0.
    value, _, _, residuals, variance = model.evaluate(theta, 7, paths=True)
    before, after = np.zeros(model.size), np.zeros(model.size)
    before[0], after[0] = np.array([retained, reported])*len(model.data)/model.units[0]
    before[alpha] = after[alpha] = 1e152
    for _ in range(3):
        after[alpha] = np.nextafter(after[alpha], np.inf)
    prior_bound = model.assess(theta, before)[1]
    current_bound = model.assess(theta, after)[1]
    np.testing.assert_allclose([prior_bound, current_bound], [retained, reported], rtol=1e-12)
    optimized = OptimizeResult(x=theta.copy(), fun=value, jac=before,
        status=0, success=True, stationarity_bound=prior_bound,
        phases=(), selected_space="theta")
    if cached_hessian:
        optimized.curvature_hessian = np.eye(model.size)
    requests = []

    def synthetic_report(point, request, *, paths=False):
        np.testing.assert_array_equal(point, theta)
        requests.append(request)
        return value, after, np.eye(model.size) if request & 4 else None, residuals, variance

    monkeypatch.setattr(model, "evaluate", synthetic_report)
    fitted = assemble_joint_fit(model, optimized, False, 1e-12, 1, time.perf_counter())
    assert requests == ([3] if cached_hessian else [7])
    assert fitted.fit_info.jet_consistency["status"] == "consistent"
    assert fitted.fit_info.kkt_residual >= max(prior_bound, current_bound)
    credible = max(retained, reported) <= 1e-6
    assert fitted.converged == credible
    assert (fitted.std_errors is not None) == credible
    assert fitted.fit_info.inference_status == ("available" if credible else "unavailable")
    np.testing.assert_array_equal(fitted.params, theta)


def test_joint_retains_selected_gradient_without_a_reporting_request():
    from scivol._joint_mean_fit import solve_joint

    model, _ = prepared(2, n=40)
    optimized = solve_joint(
        model, model.initialize(), transformed=False, accuracy=1e-12, maxiter=2
    )
    assert optimized.jac is not None
    saved = optimized.jac.copy()
    np.testing.assert_array_equal(saved, model.evaluate(optimized.x, 2)[1])
    changed = optimized.x.copy()
    changed[0] += 0.01
    model.evaluate(changed, 2)
    np.testing.assert_array_equal(optimized.jac, saved)


@pytest.mark.parametrize("family", (0, 2))
def test_joint_reuses_selected_point_curvature_for_inference(family):
    import time
    from scivol._joint_mean_fit import solve_joint, assemble_joint_fit

    model, _ = prepared(family, 1, n=400)
    if family == 2:
        from benchmark_native_joint import comparison_sample
        y, x = comparison_sample(2785, False, "normal")
        model = JointMeanModel(ARX(1) + FAMILIES[2]() + DENSITIES[1](), y, x)
    optimized = solve_joint(
        model,
        model.initialize(),
        transformed=False,
        accuracy=1e-12,
        maxiter=1000,
        allow_chart_refinement=True,
        speculate=False,
    )
    assert optimized.curvature_hessian is not None
    expected = model.evaluate(optimized.x, 7, paths=True)
    result = assemble_joint_fit(
        model, optimized, False, 1e-12, 1000, time.perf_counter()
    )
    assert result.fit_info.postfit_jet_requests["H"] == 0
    assert result.fit_info.postfit_jet_requests["masks"]["F|G"] == 1
    np.testing.assert_allclose(result._hessian, expected[2], rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(result.sigma2, expected[4], rtol=2e-12)
    assert result.log_likelihood == pytest.approx(-expected[0], abs=1e-10, rel=0)


@pytest.mark.parametrize("first_space,last_space", (("theta", "z"), ("z", "theta")))
def test_joint_reporting_includes_completed_nested_charts(first_space, last_space):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared(3)
    theta = model.initialize()
    value, gradient, hessian = model.evaluate(theta, 7)[:3]
    leaf = lambda space, mode: dict(space=space, mode=mode, iterations=1)
    phases = (
        leaf(first_space, 9),
        dict(space=first_space, mode=9, algorithm="aggregate-retry", refinement_phases=(
            leaf(last_space, None),
            leaf(first_space, 0),
            dict(space=first_space, mode=0, algorithm="aggregate-continuation",
                 refinement_phases=(leaf(last_space, 0),)),
        )),
    )
    optimized = OptimizeResult(x=theta.copy(), fun=value, jac=gradient.copy(),
        curvature_hessian=hessian.copy(), success=False, status=9,
        selected_space=last_space, phases=phases, nit=3)
    result = assemble_joint_fit(model, optimized, False, 1e-12, 10, time.perf_counter())
    assert result.fit_info.solver == "native-slsqp-hybrid"
    assert result.fit_info.optimizer_phases == (
        f"native-slsqp-{first_space}", f"native-slsqp-{first_space}",
        f"native-slsqp-{last_space}")
    assert result.fit_info.optimizer_phase_modes == (9, 0, 0)
    assert result.fit_info.phase_diagnostics == phases
    assert result.fit_info.postfit_jet_requests["H"] == 0
    assert (result.fit_info.function_evaluations, result.fit_info.gradient_evaluations,
            result.fit_info.hessian_evaluations) == (3, 2, 1)
    np.testing.assert_array_equal(result.params, theta)


@pytest.mark.parametrize("selected", ("primary", "child", "child-without-hessian"))
def test_aparch_retry_assembly_uses_only_selected_point_derivatives(selected):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import _retain_joint_refinement, assemble_joint_fit

    model, _ = prepared(3, 1)
    first = model.initialize()
    second = first.copy()
    second[0] += 0.2
    estimates = []
    for theta in (first, second):
        value, gradient, hessian = model.evaluate(theta, 7)[:3]
        estimates.append(OptimizeResult(x=theta.copy(), fun=value, jac=gradient.copy(),
            curvature_hessian=hessian.copy(), stationarity_bound=np.inf,
            phases=(), selected_space="theta", status=9, success=False,
            nit=1, nfev=1, njev=1, nhev=1))
    estimates.sort(key=lambda result: result.fun)
    assert estimates[0].fun < estimates[1].fun
    assert not np.array_equal(estimates[0].curvature_hessian, estimates[1].curvature_hessian)
    primary, child = estimates if selected == "primary" else estimates[::-1]
    expected = estimates[0].curvature_hessian.copy()
    if selected == "child-without-hessian":
        del child.curvature_hessian
    phase = dict(space="theta", mode=9, iterations=1,
        function_evaluations=1, gradient_evaluations=1, hessian_evaluations=1)
    retained = _retain_joint_refinement(primary, child, phase)
    request = 7 if selected == "child-without-hessian" else 3
    expected_gradient = model.evaluate(retained.x, request, paths=True)[1].copy()
    result = assemble_joint_fit(model, retained, False, 1e-12, 10, time.perf_counter())
    np.testing.assert_array_equal(result.params, estimates[0].x)
    np.testing.assert_array_equal(result._hessian, expected)
    np.testing.assert_array_equal(result._filtered_state_cache.score, -expected_gradient)
    assert result.fit_info.jet_consistency["status"] == "consistent"
    assert result.fit_info.postfit_jet_requests["H"] == int(selected == "child-without-hessian")


def test_inference_request_fallbacks_remain_in_the_work_record(monkeypatch):
    import time
    from scivol import _native_core
    from scivol._joint_mean_fit import solve_joint, assemble_joint_fit

    model, _ = prepared()
    result = solve_joint(
        model, model.initialize(), transformed=False, accuracy=1e-12, maxiter=1
    )
    jet = _native_core._linear_joint_prevalidated

    def fail_derivatives(*args):
        return False if args[5] in (7, 3) else jet(*args)

    monkeypatch.setattr(_native_core, "_linear_joint_prevalidated", fail_derivatives)
    fitted = assemble_joint_fit(model, result, False, 1e-12, 1, time.perf_counter())
    requests = fitted.fit_info.postfit_jet_requests
    assert requests["F"] == 3 and requests["G"] == 2 and requests["H"] == 1
    assert (
        requests["masks"]["F|G|H"]
        == requests["masks"]["F|G"]
        == requests["masks"]["F"]
        == 1
    )
    assert fitted.fit_info.inference_status == "unavailable"


@pytest.mark.parametrize("family", range(3))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True)])
def test_promotion_request_adapters_use_the_same_point(family, density, p, q, har):
    from benchmark_native_joint import (
        assess_request_agreement, prepare_native_request, prepare_legacy_request,
    )

    model, _ = prepared(family, density, p, q, har=har)
    theta = model.initialize()
    expected = model.evaluate(theta, 7)
    for mask, index in ((1, 0), (2, 1), (3, None), (4, 2)):
        native, native_output = prepare_native_request(model, theta, mask)
        legacy, legacy_output = prepare_legacy_request(model, theta, mask)
        native()
        legacy()
        np.testing.assert_allclose(
            native_output,
            np.r_[expected[0], expected[1]] if mask == 3 else expected[index],
            rtol=2e-10,
            atol=2e-10,
        )
        agreement = assess_request_agreement(mask, native_output, legacy_output)
        assert agreement["parity"], agreement


@pytest.mark.parametrize("mask", (1, 3))
def test_request_timing_uses_absolute_likelihood_agreement(mask):
    from benchmark_native_joint import assess_request_agreement

    expected = np.array([500.0] if mask == 1 else [500.0, 10000.0])
    actual = expected.copy()
    actual[0] += 1e-4
    assert np.allclose(actual, expected, rtol=2e-6, atol=3e-6)
    evidence = assess_request_agreement(mask, actual, expected)
    assert not evidence["parity"]
    assert evidence["timing_status"] == "diagnostic_only"
    assert evidence["value_absolute_error"] > evidence["value_absolute_tolerance"]
    actual[0] = expected[0] + 5e-10
    if mask == 3:
        actual[1] += 1e-3
    assert assess_request_agreement(mask, actual, expected)["parity"]


@pytest.mark.parametrize("mask", (1, 2, 3, 4))
@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_nonfinite_request_timing_is_never_accuracy_matched(mask, invalid):
    from benchmark_native_joint import assess_request_agreement

    shape = {1: (1,), 2: (2,), 3: (3,), 4: (2, 2)}[mask]
    output = np.full(shape, invalid)
    evidence = assess_request_agreement(mask, output, output)
    assert not evidence["parity"] and not evidence["finite_outputs"]
    assert evidence["timing_status"] == "diagnostic_only"
    assert evidence["maximum_absolute_error"] is None


@pytest.mark.parametrize("mask", (2, 3, 4))
def test_request_timing_checks_derivatives_separately(mask):
    from benchmark_native_joint import assess_request_agreement

    expected = np.full((2, 2) if mask == 4 else (3,), 10000.0)
    actual = expected.copy()
    actual.flat[-1] += 1e-3
    evidence = assess_request_agreement(mask, actual, expected)
    assert evidence["parity"] and evidence["timing_status"] == "same_point_agreement"
    assert evidence["derivative_maximum_absolute_error"] > 1e-9
    actual.flat[-1] += 1.0
    assert not assess_request_agreement(mask, actual, expected)["parity"]


@pytest.mark.parametrize("mask,output", (
    (1, np.ones(2)), (2, np.ones((2, 1))), (3, np.ones(1)),
    (4, np.ones((2, 3))), (1, np.array([])), (7, np.ones(2)),
))
def test_request_agreement_rejects_malformed_outputs(mask, output):
    from benchmark_native_joint import assess_request_agreement

    with pytest.raises(ValueError):
        assess_request_agreement(mask, output, output)
    with pytest.raises(ValueError):
        assess_request_agreement(2, np.ones(2), np.ones(3))


def test_request_timing_retains_incompatible_measurements_as_diagnostics(monkeypatch):
    import benchmark_native_joint as benchmark
    from types import SimpleNamespace

    def prepare(model, theta, mask, *, discrepancy):
        output = np.ones({1: (1,), 2: (2,), 3: (3,), 4: (2, 2)}[mask])
        if mask & 1:
            output[0] += discrepancy
        return lambda: None, output

    monkeypatch.setattr(benchmark, "prepare_native_request",
                        lambda *args: prepare(*args, discrepancy=1e-4))
    monkeypatch.setattr(benchmark, "prepare_legacy_request",
                        lambda *args: prepare(*args, discrepancy=0.0))
    monkeypatch.setattr(benchmark, "paired_times", lambda *args, **kwargs: (2.0, 1.0, 0.5))
    records = benchmark.measure_requests(SimpleNamespace(family=2), np.ones(2), 3, None)
    assert [row["parity"] for row in records] == [False, True, False, True]
    assert all(row["speedup"] == 0.5 for row in records)
    assert records[0]["timing_status"] == "diagnostic_only"


def test_promotion_benchmark_retains_native_evidence_when_reference_fails(monkeypatch):
    import benchmark_native_joint as benchmark
    from types import SimpleNamespace

    def fail_reference(*args, **kwargs):
        raise RuntimeError("reference sentinel")

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    monkeypatch.setattr(
        benchmark, "get_routine", lambda *args: SimpleNamespace(fit=fail_reference)
    )
    result = benchmark.compare(0, 0, False, pairs=3, maxiter=1, native_maxiter=2)
    assert "reference sentinel" in result["error"]
    assert np.isfinite(result["native_nll"])
    assert result["theta"]
    assert result["native_fit_info"]["phase_diagnostics"]
    assert result["iteration_limits"] == {"native": 2, "reference": 1}


def test_complete_fit_timings_record_cpu_and_alternate_order(monkeypatch):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    result = benchmark.compare(
        0, 0, False, pairs=3, maxiter=1, reference_engine="canonical"
    )
    timings = result["timings"]
    assert [pair["first"] for pair in timings] == ["native", "reference", "native"]
    for pair in timings:
        for engine in ("native", "reference"):
            assert np.isfinite(pair[engine]) and pair[engine] > 0
            assert np.isfinite(pair[engine + "_cpu"]) and pair[engine + "_cpu"] >= 0
    # CPU time diagnoses scheduling noise; elapsed time still defines speedup.
    assert result["speedup"] == sorted(
        pair["reference"] / pair["native"] for pair in timings
    )[1]


@pytest.mark.parametrize("initial_curvature", (False, True))
def test_comparison_dispatches_and_reports_separate_iteration_budgets(monkeypatch, initial_curvature):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    observed = []
    original = benchmark.independent_complete_fit

    def reference(spec, y, x, initial, maxiter, *options):
        observed.append(maxiter)
        return original(spec, y, x, initial, maxiter, *options)

    monkeypatch.setattr(benchmark, "independent_complete_fit", reference)
    result = benchmark.compare(0, 0, False, pairs=3, maxiter=1, native_maxiter=2,
        reference_engine="canonical", initial_curvature=initial_curvature)
    assert result["iteration_limits"] == {"native": 2, "reference": 1}
    assert result["native_fit_info"]["maxiter"] == 2
    assert observed == [1] * 4


@pytest.mark.parametrize("budget", (0, -1, 1.5, np.inf, np.nan))
def test_comparison_rejects_invalid_native_budget_before_preparation(monkeypatch, budget):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "comparison_sample", lambda *args: pytest.fail("unexpected preparation"))
    with pytest.raises(ValueError, match="positive integers"):
        benchmark.compare(0, 0, False, pairs=3, maxiter=1, native_maxiter=budget)


@pytest.mark.parametrize("transformed", (None, False, True))
def test_aparch_retry_comparison_uses_existing_controller_and_full_assembly(monkeypatch, transformed):
    import benchmark_native_joint as benchmark

    calls = []
    original = benchmark.solve_joint_with_retry
    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])

    def retry(model, initial, **options):
        calls.append((initial.copy(), options.copy()))
        return original(model, initial, **options)

    def unexpected_public(*args, **kwargs):
        pytest.fail("The retry comparison silently used the public primary controller.")

    monkeypatch.setattr(benchmark, "solve_joint_with_retry", retry)
    monkeypatch.setattr(benchmark, "fit_joint_mean_spec", unexpected_public)
    result = benchmark.compare(3, 0, False, 3, 1, native_maxiter=2,
        native_log=transformed, reference_engine="canonical", aparch_retry=True)
    assert len(calls) == 4
    for initial, options in calls:
        np.testing.assert_array_equal(initial, result["initial"])
        assert options == dict(transformed=bool(transformed), accuracy=1e-12,
                              maxiter=2, allow_chart_refinement=transformed is None)
    assert result["iteration_limits"] == dict(native=2, reference=1)
    info = result["native_fit_info"]
    assert info["optimizer_controller"] == "joint-same-start-retry"
    assert info["maxiter"] == 2 and info["phase_diagnostics"]
    assert info["warnings"] and not result["native_certified"]
    assert result["native_feasible"] and result["comparison_domain"] == "identical"
    assert result["native_nll"] == result["legacy_at_native"]
    assert result["reference_nll"] == result["legacy_at_reference"]


@pytest.mark.parametrize("options", (
    dict(family=2), dict(initial_curvature=True), dict(native_speculate=False),
    dict(native_curvature_reset=True),
))
def test_aparch_retry_comparison_rejects_incompatible_controls_before_work(monkeypatch, options):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "comparison_sample", lambda *args: pytest.fail("unexpected work"))
    with pytest.raises(ValueError, match="APARCH retry control"):
        benchmark.compare(**(dict(family=3, density=0, har=False, pairs=3,
                                  maxiter=1, aparch_retry=True) | options))


@pytest.mark.parametrize("arguments", (
    [], ["--family", "2"], ["--family", "3", "--initial-curvature"],
    ["--family", "3", "--no-speculate-trials"],
    ["--family", "3", "--egarch-acceptance-shard", "0"],
))
def test_aparch_retry_cli_rejects_conflicting_modes(monkeypatch, arguments):
    import sys
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(sys, "argv", ["benchmark_native_joint.py", "--aparch-retry", *arguments])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ("jet", "mean", "constraint"))
def test_canonical_control_never_substitutes_zero_derivatives(failure, monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(3, n=100)
    initial = model.initialize()

    def reject(*args, **kwargs):
        raise ValueError("invalid request sentinel")

    def inspect_requests(objective, point, **options):
        if failure == "constraint":
            constraint = options["constraints"][0]
            assert np.all(constraint["fun"](point) < 0)
            assert np.isnan(constraint["jac"](point)).all()
        else:
            value, gradient = objective(point)
            assert value == np.inf
            assert np.isnan(gradient).all()
        return OptimizeResult(x=point.copy(), status=9, success=False)

    monkeypatch.setattr(benchmark, "minimize", inspect_requests)
    with monkeypatch.context() as requests:
        if failure == "jet":
            requests.setattr(
                benchmark._native_core,
                "_linear_joint_prevalidated",
                lambda *args: False,
            )
            # The final scalar rescore is independent of the rejected callback.
            requests.setattr(model, "evaluate", lambda *args, **kwargs: (123.0,))
        elif failure == "mean":
            from types import SimpleNamespace

            requests.setattr(model, "blocks", [SimpleNamespace(evaluate=reject)])
        else:
            requests.setattr(model, "constraints", reject)
            # Isolate the callback's invalid request from final-point feasibility.
            requests.setattr(model, "assess", lambda *args: (True, np.inf, 0.0))
        result = benchmark.independent_fit(model, initial, maxiter=1)
    phase = result.phases[0]
    assert phase["invalid_objective_requests"] == int(failure != "constraint")
    assert phase["invalid_constraint_requests"] == 2 * int(failure == "constraint")
    assert phase["mean_domain_rejections"] == int(failure == "mean")


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("transformed", (False, True))
def test_canonical_control_preserves_constraint_request_intentions(monkeypatch, family, transformed):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult
    from types import SimpleNamespace

    model, _ = prepared(family, n=12)
    initial = model.initialize()
    expected_values, expected_jacobian = model.constraints(initial)
    calls = []
    original = model.constraints

    def constraints(point, *, derivatives=True):
        calls.append(derivatives)
        return original(point, derivatives=derivatives)

    def recorded_block(block):
        def evaluate(point, *, derivatives=True, **options):
            calls.append(derivatives)
            return block.evaluate(point, derivatives=derivatives, **options)
        return SimpleNamespace(evaluate=evaluate, offset=block.offset, width=block.width)

    if transformed:
        model.blocks = [recorded_block(block) for block in model.blocks]
        expected_values = expected_values[:2 * model.domain[0]]
        expected_jacobian = expected_jacobian[:2 * model.domain[0]]
    else:
        monkeypatch.setattr(model, "constraints", constraints)

    def inspect_requests(objective, start, **options):
        constraint = options["constraints"][0]
        for name, derivative, expected in (
            ("fun", False, expected_values), ("jac", True, expected_jacobian)
        ):
            calls.clear()
            np.testing.assert_array_equal(constraint[name](start), expected)
            assert calls == [derivative] * (len(model.blocks) if transformed else 1)
        return OptimizeResult(x=start.copy(), status=9, success=False)

    monkeypatch.setattr(benchmark, "minimize", inspect_requests)
    benchmark.independent_fit(model, initial, 1, transformed)


def test_canonical_constraint_value_does_not_require_available_jacobian(monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(3, n=12)
    initial = model.initialize()
    expected = model.constraints(initial, derivatives=False)[0]
    monkeypatch.setattr(benchmark._native_core, "_linear_joint_constraints", lambda *args: False)

    def inspect_requests(objective, start, **options):
        constraint = options["constraints"][0]
        np.testing.assert_array_equal(constraint["fun"](start), expected)
        assert np.isnan(constraint["jac"](start)).all()
        return OptimizeResult(x=start.copy(), status=9, success=False)

    monkeypatch.setattr(benchmark, "minimize", inspect_requests)
    result = benchmark.independent_fit(model, initial, 1)
    assert result.phases[0]["invalid_constraint_requests"] == 1
    np.testing.assert_array_equal(result.x, initial)


def test_promotion_benchmark_keeps_finite_reference_with_invalid_gradient(monkeypatch):
    from benchmark_native_joint import assess_comparison_point

    model, _ = prepared()
    theta = model.initialize()
    expected = model.evaluate(theta, 1)[0]
    original = model.evaluate

    def reject_derivatives(point, request=7, **kwargs):
        if request & 6:
            raise ValueError("derivative sentinel")
        return original(point, request, **kwargs)

    monkeypatch.setattr(model, "evaluate", reject_derivatives)
    evidence = assess_comparison_point(model, theta)
    assert evidence["nll"] == expected
    assert evidence["feasible"]
    assert evidence["kkt_bound"] == np.inf
    assert evidence["violation"] == 0
    assert evidence["evaluation_errors"] == {"gradient": "derivative sentinel"}


def test_promotion_benchmark_does_not_turn_value_failure_into_finite_evidence(
    monkeypatch,
):
    from benchmark_native_joint import assess_comparison_point

    model, _ = prepared()
    theta = model.initialize()
    original = model.evaluate

    def reject_value(point, request=7, **kwargs):
        if request & 1:
            raise ValueError("value sentinel")
        return original(point, request, **kwargs)

    monkeypatch.setattr(model, "evaluate", reject_value)
    evidence = assess_comparison_point(model, theta)
    assert evidence["nll"] == np.inf
    assert evidence["feasible"]
    assert evidence["evaluation_errors"] == {"value": "value sentinel"}


@pytest.mark.parametrize("family", range(4))
def test_canonical_control_counts_its_direct_objective_requests(family, monkeypatch):
    import benchmark_native_joint as benchmark
    from scivol import _native_core

    model, x = prepared(family, n=100)
    initial = model.initialize()
    calls = [0] * 8
    original = _native_core._linear_joint_prevalidated

    def record(*args):
        calls[args[5]] += 1
        return original(*args)

    monkeypatch.setattr(_native_core, "_linear_joint_prevalidated", record)
    result = benchmark.independent_complete_fit(model.spec, model.data, x, initial, 1)
    assert result.fit_info.jet_requests["masks"]["F|G"] > 0
    assert tuple(result.fit_info.jet_requests["masks"].values()) == tuple(calls[1:])
    assert result.fit_info.solver == "scipy-slsqp"


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("transformed", (False, True))
def test_canonical_control_rejects_unstable_mean_trials_before_the_jet(
    family, har, transformed, monkeypatch
):
    import benchmark_native_joint as benchmark
    from scivol import _native_core
    from scipy.optimize import OptimizeResult

    model, _ = prepared(family, har=har)
    initial = model.initialize()
    calls = []
    original = _native_core._linear_joint_prevalidated

    def record(*args):
        calls.append(args[5])
        return original(*args)

    def inspect_trials(objective, start, **kwargs):
        for coefficient in (-1.2, -1.0, 1.0, 1.2):
            trial = start.copy()
            block = model.blocks[0]
            trial[block.offset : block.offset + block.width] = 0
            trial[block.offset] = coefficient
            before = len(calls)
            value, gradient = objective(trial)
            assert value == np.inf
            assert np.isnan(gradient).all()
            assert len(calls) == before
            # Violated but evaluable constraints still guide the independent solver.
            slack = kwargs["constraints"][0]["fun"](trial)
            assert np.isfinite(slack).all()
            assert np.min(slack) <= 0
        value, gradient = objective(start)
        assert np.isfinite(value) and np.isfinite(gradient).all()
        assert calls == [3]
        assert model.request_counts[3] == 5
        return OptimizeResult(x=start.copy(), status=9, success=False)

    monkeypatch.setattr(_native_core, "_linear_joint_prevalidated", record)
    monkeypatch.setattr(benchmark, "minimize", inspect_trials)
    result = benchmark.independent_fit(model, initial, 1, transformed)
    assert result.phases[0]["mean_domain_rejections"] == 4
    assert calls == [3, 1]


def test_canonical_comparison_keeps_actual_legacy_cross_scores(monkeypatch):
    import benchmark_native_joint as benchmark

    def unexpected_fit(*args):
        pytest.fail("An identical-jet solver control must not select a legacy fitter.")

    legacy_filter = benchmark._filter_meanx_garch
    points = []

    def record_filter(spec, data, theta, **kwargs):
        points.append(theta.copy())
        return legacy_filter(spec, data, theta, **kwargs)

    monkeypatch.setattr(benchmark, "get_routine", unexpected_fit)
    monkeypatch.setattr(benchmark, "_filter_meanx_garch", record_filter)
    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    result = benchmark.compare(0, 0, False, 3, 1, reference_engine="canonical")
    assert result["comparison_domain"] == "identical"
    assert result["reference_engine"] == "canonical"
    assert "scipy-on-identical-jet" in result["reference"]
    outcomes = [result["warmup"], *(pair["outcome"] for pair in result["timings"])]
    assert len(points) == 2 * len(outcomes) == 8
    for pair, outcome in enumerate(outcomes):
        np.testing.assert_array_equal(points[2 * pair], outcome["theta"])
        np.testing.assert_array_equal(points[2 * pair + 1], outcome["reference_theta"])
    np.testing.assert_allclose(
        result["legacy_at_native"], result["native_nll"], rtol=1e-12
    )
    info = result["reference_fit_info"]
    assert info["jet_requests"]["masks"]["F|G"] > 0
    assert result["reference_terminated"] == info["optimizer_terminated"]


@pytest.mark.parametrize("density", (0, 5, 7))
def test_matched_legacy_control_preserves_callbacks_and_both_import_styles(monkeypatch, density):
    import scipy.optimize
    from scipy.optimize import OptimizeResult
    import benchmark_native_joint as benchmark

    model, x = prepared(2, density, p=3, q=2, n=32, har=True)
    initial = model.initialize()
    initial[model.shape_start - 2 : model.shape_start] = [1.2, -0.4]
    assert model.assess(initial, None)[0]
    routine = benchmark.get_routine(str(model.spec))
    old_bound = routine.fit.__globals__.get("minimize")
    old_scipy = scipy.optimize.minimize
    value_call, value = benchmark.prepare_legacy_request(model, initial, 1)
    gradient_call, gradient = benchmark.prepare_legacy_request(model, initial, 2)
    value_call()
    gradient_call()

    def inspect_historical_requests(function, start, **options):
        np.testing.assert_array_equal(start, initial)
        np.testing.assert_allclose(function(start), value[0] / len(model.data), rtol=1e-13)
        np.testing.assert_allclose(options["jac"](start), gradient / len(model.data), rtol=1e-13)
        lower = np.where(np.isnan(model.lower), -np.inf, model.lower)
        upper = np.where(np.isnan(model.upper), np.inf, model.upper)
        np.testing.assert_array_equal(options["bounds"].lb, lower)
        np.testing.assert_array_equal(options["bounds"].ub, upper)
        restrictions = options["constraints"][0]
        expected, jacobian = model.constraints(start)
        np.testing.assert_array_equal(restrictions["fun"](start), expected)
        np.testing.assert_array_equal(restrictions["jac"](start), jacobian)
        return OptimizeResult(x=start.copy(), fun=function(start), success=False,
                              status=9, nit=1, message="test budget")

    monkeypatch.setattr(benchmark, "minimize", inspect_historical_requests)
    result, evidence = benchmark.matched_legacy_egarch_fit(model, initial, x, 1)
    np.testing.assert_array_equal(result.params, initial)
    assert evidence["calls"] == 1
    assert evidence["requests"] == dict(F=3, G=1, invalid=0)
    assert not evidence["retained_feasible_trial"]
    assert scipy.optimize.minimize is old_scipy
    assert routine.fit.__globals__.get("minimize") is old_bound


@pytest.mark.parametrize("failed", (False, True))
def test_matched_legacy_control_normalizes_all_callbacks_and_retains_finite_trial(monkeypatch, failed):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scipy.optimize
    import benchmark_native_joint as benchmark

    seen = []
    initial = np.array([2.0])

    def constraints(point, *, derivatives=True):
        seen.append(("constraints", point.copy()))
        if point[0] == 3:
            raise ValueError("Unavailable constraint request")
        return np.array([point[0]]), np.ones((1, 1)) if derivatives else None

    model = SimpleNamespace(family=2, lower=np.array([0.0]), upper=np.array([4.0]),
        size=1, rows=1, blocks=(), data=np.zeros(3), spec="matched-control-fixture",
        constraints=constraints, assess=lambda point, gradient: (point[0] > 0, np.inf, 0.0))

    def value(point):
        seen.append(("value", point.copy()))
        return np.nan if point[0] == 4 else float(point[0] ** 2)

    def gradient(point):
        seen.append(("gradient", point.copy()))
        return 2 * point

    def historical(data, *, x, initial_params, maxiter, solver, log_mode):
        return scipy.optimize.minimize(value, initial_params, jac=gradient,
            method="SLSQP", options=dict(maxiter=maxiter, ftol=1e-12))

    def solve(function, start, **options):
        restrictions = options["constraints"][0]
        assert function(start) == 4
        assert function(np.array([1.0])) == 1
        # A finite infeasible minimum must not replace the feasible trial.
        assert function(np.array([-1.0])) == 0
        options["jac"](np.array([-1.0]))
        restrictions["fun"](np.array([-1.0]))
        restrictions["jac"](np.array([-1.0]))
        assert restrictions["fun"](np.array([3.0]))[0] < 0
        assert np.isnan(restrictions["jac"](np.array([3.0]))).all()
        if failed:
            raise ValueError("Interrupted historical solve")
        return OptimizeResult(x=np.array([5.0]), fun=np.nan, status=5,
                              success=False, nit=3)

    original = scipy.optimize.minimize
    monkeypatch.setattr(benchmark, "get_routine", lambda spec: SimpleNamespace(fit=historical))
    monkeypatch.setattr(benchmark, "minimize", solve)
    if failed:
        with pytest.raises(ValueError, match="Interrupted"):
            benchmark.matched_legacy_egarch_fit(model, initial, None, 3)
    else:
        result, evidence = benchmark.matched_legacy_egarch_fit(model, initial, None, 3)
        np.testing.assert_array_equal(result.x, [1])
        assert result.fun == 1 and not result.success
        assert evidence["retained_feasible_trial"]
        assert evidence["raw"]["theta"] == [4]
        assert np.isnan(evidence["raw"]["value"])
    assert scipy.optimize.minimize is original
    assert all(0 <= point[0] <= 4 for _, point in seen)
    assert {kind for kind, point in seen if point[0] == 0} == {"value", "gradient", "constraints"}
    np.testing.assert_array_equal(initial, [2])


def test_matched_legacy_comparison_records_domain_and_cross_scores(monkeypatch):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    result = benchmark.compare(2, 5, False, 3, 1, reference_engine="legacy-matched")
    assert result["reference_engine"] == "legacy-matched"
    assert result["comparison_domain"] == "identical"
    assert "historical numerical validity remains" in result["comparison_trial_validity"]
    assert result["reference_fit_info"]["comparison_control"]["calls"] == 1
    assert result["reference_feasible"]
    np.testing.assert_allclose(result["legacy_at_native"], result["native_nll"], rtol=1e-12)
    np.testing.assert_allclose(result["legacy_at_reference"], result["reference_nll"], rtol=1e-12)
    for options in (dict(family=0), dict(family=2, legacy_log=True), dict(family=2, reference_log=True)):
        with pytest.raises(ValueError, match="theta space"):
            benchmark.compare(density=5, har=False, pairs=3, maxiter=1,
                              reference_engine="legacy-matched", **options)


def test_comparison_value_failure_cannot_look_like_infinite_native_improvement(monkeypatch):
    import benchmark_native_joint as benchmark

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    assess = benchmark.assess_comparison_point

    def missing_value(model, theta):
        evidence = assess(model, theta)
        evidence["nll"] = np.inf
        evidence["evaluation_errors"]["value"] = "Value request rejected"
        return evidence

    monkeypatch.setattr(benchmark, "assess_comparison_point", missing_value)
    result = benchmark.compare(2, 5, False, 3, 1, reference_engine="legacy-matched")
    assert result["nll_loss"] is None
    assert result["reference_evaluation_errors"]["value"] == "Value request rejected"
    assert np.isfinite(result["legacy_at_reference"])


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("transformed", (False, True))
def test_vg_lower_shape_boundary_keeps_value_request_valid(fixed, transformed):
    from benchmark_native_joint import comparison_sample

    y, x = comparison_sample(2785, True, "normal")
    model = JointMeanModel(HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[7](),
                           y, x, fixed=fixed)
    # Finite historical endpoint: lambda-.5 is close to one, but is not one.
    theta = np.array([
        -.09303138049839323, -.10253850481749467, .14749409185564247,
        .032749274072862135, -.02266483782830796, -.0830076168060174,
        .08215564729331713, .04749303130059332, -.5143733267701783,
        .6932133434516138, -.32893852267113244, -.40903672925664536,
        .5947833667717658, -.3410507211095869, 1.4828946561168896,
        -.9981991259070052, -.2735357500618175, 1.5000000000000018,
    ])
    if transformed:
        # The open coordinate chart excludes the exact numerical endpoint.
        theta[-1] = np.nextafter(theta[-1], np.inf)
    assert model.assess(theta, None)[0]
    point = model.inverse(theta) if transformed else theta
    full = model.evaluate(point, 7, transformed=transformed)
    for mask in range(1, 8):
        result = model.evaluate(point, mask, transformed=transformed)
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                np.testing.assert_allclose(result[index], full[index], rtol=2e-9, atol=3e-9)
            else:
                assert result[index] is None


@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("stale", (False, True))
def test_canonical_control_retains_only_additive_same_point_theta_gradient(
    transformed, stale, monkeypatch
):
    import ctypes
    import benchmark_native_joint as benchmark
    from scivol import _native_core
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, 1, n=100)
    initial = model.initialize()
    expected = model.evaluate(initial, 2)[1]
    assert np.linalg.norm(expected) > 1
    recorded = []
    evaluate = _native_core._linear_joint_prevalidated

    def record_gradient(*args):
        valid = evaluate(*args)
        if valid and args[5] & 2:
            recorded.append(np.ctypeslib.as_array(
                ctypes.cast(args[10], ctypes.POINTER(ctypes.c_double)),
                shape=(model.size,),
            ).copy())
        return valid

    def finish(objective, point, **options):
        trial = point.copy()
        if stale:
            trial[0] += 0.01 * np.sign(expected[0])
        value, averaged_gradient = objective(trial)
        return OptimizeResult(
            x=point.copy(), fun=value, jac=averaged_gradient, status=9, success=False
        )

    monkeypatch.setattr(benchmark, "minimize", finish)
    monkeypatch.setattr(_native_core, "_linear_joint_prevalidated", record_gradient)
    result = benchmark.independent_fit(model, initial, 1, transformed)
    assert len(recorded) == 1
    if transformed or stale:
        assert result.jac is None
    else:
        np.testing.assert_array_equal(result.jac, recorded[0])
        np.testing.assert_allclose(result.jac, expected, rtol=1e-13, atol=1e-13)


def test_canonical_control_normalizes_every_callback_at_the_same_bounds(monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, 1, n=100)
    initial = model.initialize()
    # Use an interior bound in this adapter test so all four requests are valid.
    model.lower[-1] = initial[-1]

    def finish(objective, point, **options):
        trial = point.copy()
        trial[-1] -= 1
        value, gradient = objective(trial)
        expected_value, expected_gradient = objective(point)
        assert value == expected_value
        np.testing.assert_array_equal(gradient, expected_gradient)
        constraints = options["constraints"][0]
        for request in ("fun", "jac"):
            np.testing.assert_array_equal(
                constraints[request](trial), constraints[request](point)
            )
        return OptimizeResult(x=trial, status=9, success=False)

    monkeypatch.setattr(benchmark, "minimize", finish)
    result = benchmark.independent_fit(model, initial, 1)
    np.testing.assert_array_equal(result.x, initial)


def test_canonical_control_retains_best_feasible_trial_after_invalid_endpoint(monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, n=100)
    initial = model.initialize()

    def finish(objective, point, **options):
        value, gradient = objective(point)
        invalid = point.copy()
        invalid[0] = np.nan
        return OptimizeResult(
            x=invalid, fun=value, jac=gradient, status=0, success=True
        )

    monkeypatch.setattr(benchmark, "minimize", finish)
    result = benchmark.independent_fit(model, initial, 1)
    np.testing.assert_array_equal(result.x, initial)
    np.testing.assert_array_equal(result.jac, model.evaluate(initial, 2)[1])
    assert result.fun == model.evaluate(initial, 1)[0]
    assert not result.success
    assert result.termination_success
    assert result.phases[0]["retained_best_trial"]
    assert result.phases[0]["endpoint_nll"] == np.inf
    assert result.phases[0]["endpoint_error"]


def test_canonical_control_fails_without_a_finite_estimate(monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, n=100)
    initial = model.initialize()

    def finish(objective, point, **options):
        invalid = point.copy()
        invalid[0] = np.nan
        objective(invalid)
        return OptimizeResult(x=invalid, status=9, success=False)

    monkeypatch.setattr(benchmark, "minimize", finish)
    with pytest.raises(ValueError, match="no finite feasible estimate"):
        benchmark.independent_fit(model, initial, 1)


def test_canonical_control_keeps_value_evidence_after_gradient_rejection(monkeypatch):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, n=100)
    initial = model.initialize()
    original = benchmark._native_core._linear_joint_prevalidated

    def reject_gradient(*args):
        return False if args[5] & 2 else original(*args)

    def finish(objective, point, **options):
        value, gradient = objective(point)
        assert value == np.inf
        assert np.isnan(gradient).all()
        invalid = point.copy()
        invalid[0] = np.nan
        return OptimizeResult(x=invalid, status=9, success=False)

    monkeypatch.setattr(benchmark._native_core, "_linear_joint_prevalidated", reject_gradient)
    monkeypatch.setattr(benchmark, "minimize", finish)
    result = benchmark.independent_fit(model, initial, 1)
    np.testing.assert_array_equal(result.x, initial)
    assert result.fun == model.evaluate(initial, 1)[0]
    assert result.jac is None
    assert result.phases[0]["retained_best_trial"]
    assert not result.success


@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("gradient_valid", (False, True))
def test_canonical_control_observes_feasible_trial_before_interruption(
    transformed, gradient_valid, monkeypatch
):
    import benchmark_native_joint as benchmark

    model, _ = prepared(2, n=100)
    initial = model.initialize()
    observed = []
    original = benchmark._native_core._linear_joint_prevalidated

    def evaluate(*args):
        return False if not gradient_valid and args[5] & 2 else original(*args)

    def interrupt(objective, point, **options):
        objective(point)
        objective(point)
        invalid = point.copy()
        invalid[0] = np.nan
        objective(invalid)
        raise TimeoutError("bounded control")

    monkeypatch.setattr(benchmark._native_core, "_linear_joint_prevalidated", evaluate)
    monkeypatch.setattr(benchmark, "minimize", interrupt)
    with pytest.raises(TimeoutError, match="bounded control"):
        benchmark.independent_fit(
            model, initial, 1000, transformed,
            on_best_trial=lambda theta, value: observed.append((theta, value)),
        )
    assert len(observed) == 1
    theta, value = observed[0]
    np.testing.assert_allclose(theta, initial, rtol=1e-14, atol=1e-14)
    assert model.assess(theta, None)[0]
    assert value == pytest.approx(model.evaluate(theta, 1)[0], abs=1e-10, rel=0)


@pytest.mark.parametrize("transformed", (False, True))
def test_canonical_control_observer_cannot_mutate_the_fit_or_add_requests(
    transformed, monkeypatch
):
    import benchmark_native_joint as benchmark
    from scipy.optimize import OptimizeResult

    model, _ = prepared(2, n=100)
    initial = model.initialize()
    observed = []

    def finish(objective, point, **options):
        objective(point)
        invalid = point.copy()
        invalid[0] = np.nan
        return OptimizeResult(x=invalid, status=9, success=False)

    def observe(theta, value):
        observed.append((theta.copy(), value))
        theta[:] = np.nan

    monkeypatch.setattr(benchmark, "minimize", finish)
    start_counts = model.request_counts.copy()
    baseline = benchmark.independent_fit(model, initial, 1, transformed)
    baseline_counts = [count - start for count, start
                       in zip(model.request_counts, start_counts)]
    start_counts = model.request_counts.copy()
    candidate = benchmark.independent_fit(
        model, initial, 1, transformed, on_best_trial=observe
    )
    candidate_counts = [count - start for count, start
                        in zip(model.request_counts, start_counts)]
    np.testing.assert_array_equal(candidate.x, baseline.x)
    assert candidate.fun == baseline.fun
    assert candidate.phases == baseline.phases
    assert candidate_counts == baseline_counts
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0][0], candidate.x)


@pytest.mark.parametrize("family", (0, 1))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("har", (False, True))
def test_garch_trial_request_policy_uses_the_shared_optimizer(
    family, density, transformed, har
):
    p, q = (3, 2) if har else (1, 1)
    model, _ = prepared(family, density, p, q, har=har)
    initial = model.initialize()
    saved = initial.copy()
    for speculate in (None, True):
        result = _solve_joint_phase(
            model,
            initial,
            transformed=transformed,
            accuracy=1e-12,
            maxiter=3,
            use_curvature=False,
            speculate=speculate,
        )
        if speculate is None:
            assert result.native_jet_requests["F|G"] == 2
            assert result.speculative_trial_gradients is False
        else:
            assert result.native_jet_requests["F|G"] > 2
            assert result.speculative_trial_gradients is True
        assert result.fun == model.evaluate(result.x, 1)[0]
        assert result.fun <= model.evaluate(initial, 1)[0]
        assert model.assess(result.x, None)[0]
    np.testing.assert_array_equal(initial, saved)


def ill_conditioned_egarch_vg_point():
    rng = np.random.default_rng(2785)
    data = rng.normal(size=400)
    exogenous = rng.normal(size=(400, 4))
    model = JointMeanModel(
        HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[7](), data, exogenous
    )
    theta = np.array(
        [
            0.031026334893510595,
            0.023831807625488977,
            0.019653398498764466,
            0.0446132150831285,
            0.01880571942892474,
            -0.052817969881623815,
            0.012258324127208267,
            -0.01578933178786574,
            -0.7253780875831427,
            -0.029108241737557244,
            0.37162024032084245,
            -0.4123837753978291,
            0.24820527744847998,
            0.21035796524234396,
            0.3680657838841479,
            0.5588540512089964,
            -0.08590362405556805,
            3.6453348340489273,
        ]
    )
    return model, theta


@pytest.mark.parametrize("fixed", (False, True))
def test_joint_skewt_large_nu_derivative_sign(fixed):
    from scivol import EGARCH, SkewT

    rng = np.random.default_rng(2786)
    data, exogenous = rng.normal(size=400), rng.normal(size=(400, 2))
    model = JointMeanModel(ARX(1)+EGARCH(1,1)+SkewT(),data,exogenous,fixed=fixed)
    theta = np.array([-0.059571874919141315,-0.022824269175321397,
                      0.10029003929763204,0.0010990694092704405,
                      -0.02918094883808402,-0.3117233967648301,
                      -0.03761827517859957,0.3834992753363415,
                      19958224.45994093,0.08728423370553595])
    # Independent MP50/70 complete likelihood, including an integrated E|z|.
    # Five-point steps 1 and 0.1 agree beyond binary64; AD tests separately
    # cover every joint derivative in this Gaussian-limit regime.
    for mask in range(1,8):
        value,gradient,hessian,*_ = model.evaluate(theta,mask)
        if mask & 1:
            assert value == pytest.approx(557.985782736418,rel=0,abs=3e-12)
        if mask & 2:
            np.testing.assert_allclose(gradient[-2],-7.22271841067742e-15,rtol=2e-10,atol=0)
        if mask & 4:
            np.testing.assert_allclose(hessian[-2,-2],7.237865942725097e-22,rtol=2e-10,atol=0)


def egarch_precision_points(section):
    return json.loads(Path(__file__).with_name("egarch_precision_points.json").read_text())[section]


def egarch_precision_model(point, fixed):
    rng = np.random.default_rng(point["seed"])
    data, exogenous = rng.normal(size=400), rng.normal(size=(400, 4))
    return JointMeanModel(HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[point["density"]](),
                          data, exogenous, fixed=fixed)


@pytest.mark.parametrize("point", egarch_precision_points("endpoints"))
@pytest.mark.parametrize("fixed", (False, True))
def test_egarch_disputed_endpoints_match_independent_precision(point, fixed):
    # MP35/50 normalized densities and recursions agree at the saved doubles.
    # These cover both engines' estimates, not just the preferred native point.
    model = egarch_precision_model(point, fixed)
    full = model.evaluate(point["theta"], 7)
    for mask in range(1, 8):
        actual = model.evaluate(point["theta"], mask)
        if mask & 1:
            np.testing.assert_allclose(actual[0], point["value"], rtol=0, atol=1e-9)
        if mask & 2:
            np.testing.assert_allclose(actual[1][:model.columns], point["mean_gradient"],
                                       rtol=2e-6, atol=3e-6)
        for flag, value, expected in zip((1, 2, 4), actual[:3], full[:3]):
            if mask & flag:
                np.testing.assert_allclose(value, expected, rtol=2e-6, atol=3e-6)
            else:
                assert value is None


@pytest.mark.parametrize("point", egarch_precision_points("joint_derivatives"))
@pytest.mark.parametrize("fixed", (False, True))
def test_egarch_stressed_joint_derivatives_match_independent_ad(point, fixed):
    # Independent JAX moment/recursion/map AD, with MP50 primal anchors only.
    # Retained curvature includes mean, omega, alpha, beta and every shape axis.
    model = egarch_precision_model(point, fixed)
    indices = point["hessian_indices"]
    for mask in range(1, 8):
        value, gradient, hessian, *_ = model.evaluate(
            point["parameters"], mask, transformed=point["transformed"])
        if mask & 1:
            np.testing.assert_allclose(value, point["value"], rtol=0, atol=1e-9)
        if mask & 2:
            np.testing.assert_allclose(gradient, point["gradient"], rtol=2e-6, atol=3e-6)
        if mask & 4:
            np.testing.assert_allclose(hessian[np.ix_(indices, indices)], np.asarray(point["hessian"], dtype=float),
                                       rtol=2e-6, atol=3e-6)


def test_ill_conditioned_egarch_preserves_sparse_request_projection():
    model, theta = ill_conditioned_egarch_vg_point()
    full = model.evaluate(theta, 7, paths=True)
    for mask in (1, 2, 3, 4, 5, 6):
        sparse = model.evaluate(theta, mask, paths=True)
        np.testing.assert_allclose(sparse[4], full[4], rtol=2e-10, atol=2e-12)
        for bit, index in ((1, 0), (2, 1), (4, 2)):
            if mask & bit:
                np.testing.assert_allclose(
                    sparse[index], full[index], rtol=2e-6, atol=3e-6
                )


def test_ill_conditioned_egarch_mean_gradient_matches_high_precision():
    model, theta = ill_conditioned_egarch_vg_point()
    # Independent 50-digit Normal-Gamma moment and recursive sensitivities;
    # the 30-digit rerun differs by 3.2e-8 relatively. No fitted point is moved.
    expected = np.array(
        [
            -75554.03542104011,
            779488.8993761473,
            173952.5729646946,
            -269710.0531870778,
            164942.98426454794,
            136883.37277113786,
            -604428.871506683,
        ]
    )
    gradient = model.evaluate(theta, 2)[1][: model.columns]
    np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=3e-6)


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("mask", (1, 2, 3, 7))
def test_egarch_vg_trial_mean_volatility_gradient_matches_high_precision(fixed, mask):
    rng = np.random.default_rng(2785)
    data, exogenous = rng.normal(size=400), rng.normal(size=(400, 4))
    model = JointMeanModel(
        HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[7](),
        data, exogenous, fixed=fixed,
    )
    theta = np.array([
        0.03355217166212355, 0.014457210348681707, 0.015413909800445812,
        0.0059013749564691634, 0.044393861474468346, -0.06249497846322767,
        0.002455736451456424, -0.01041680090710724, -0.4964649776952126,
        -0.04310083797657849, 0.22751475028732637, -0.29403736249775764,
        0.2283016454963375, 0.12465588489557516, 0.6765542270123973,
        0.2545666438595033, -0.08360297299502924, 2.9731367270117977,
    ])
    # Independent Normal-Gamma moment/density and recursive sensitivities at
    # 45/65 decimal digits agree after rounding to doubles. Seed amplification
    # is 4.76e8. This fixture covers mean/volatility, not direct shape derivatives.
    expected = np.array([
        4288642729.7462277, 22922201320.367634, 5139739694.028379,
        -7018050312.436018, 4628827791.6630535, 3471600872.0088763,
        -16684632657.682806, -255973117290.76025, 29536278642.033215,
        31893061370.999523, 31121796010.789143, 967061859.4830453,
        4594432673.801553, 3257974427.1781187, -51831785409.84005,
        -48889430663.4245,
    ])
    actual = model.evaluate(theta, mask)
    if mask & 1:
        np.testing.assert_allclose(actual[0], 559.0778153709489, rtol=0, atol=3e-9)
    if mask & 2:
        np.testing.assert_allclose(
            actual[1][:model.shape_start], expected, rtol=2e-10, atol=3e-6
        )


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("mask", (2, 7))
def test_ill_conditioned_egarch_nig_gradient_matches_high_precision(fixed, mask):
    rng = np.random.default_rng(2785)
    data = rng.normal(size=400)
    exogenous = rng.normal(size=(400, 4))
    model = JointMeanModel(
        HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[5](),
        data,
        exogenous,
        fixed=fixed,
    )
    theta = np.array([
        -0.0020545949452343225, -0.10223551454974344, 0.09605192014540645,
        -0.06668650670900252, -0.015793947894350544, -0.1209983501295424,
        0.04471121254762102, -0.06829786774056487, -0.3887992017456314,
        0.5173763546569434, -0.19381518793983943, -0.180401919627783,
        0.2746914614761186, -0.11680411683743605, 1.543648963909011,
        -0.9857210637940174, -0.562245593839108, 24.999999999999993,
    ])
    # Independent Normal-GIG integral and recursive sensitivities agree at 30/45
    # decimal digits. Density integration at 30 digits gives the same reference.
    expected = np.array([
        26.77880408520906, -4.453902219577813, -20.78123938456524,
        153.75284445291115, 22.936729474832212, 21.39112059154665,
        -39.48305892811388,
    ])
    gradient = model.evaluate(theta, mask)[1][: model.columns]
    np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=3e-6)


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("mask", (1, 3, 5, 7))
def test_ill_conditioned_egarch_gh_value_matches_high_precision(fixed, mask):
    rng = np.random.default_rng(2786)
    data, exogenous = rng.normal(size=400), rng.normal(size=(400, 4))
    model = JointMeanModel(
        HARX([1, 3]) + FAMILIES[2](3, 2) + DENSITIES[4](),
        data, exogenous, fixed=fixed,
    )
    theta = np.array([
        1.8853894318719973, -1.2367603019675264, 0.7460672063965066,
        0.3247246841588669, 0.851918144367071, -0.02581064158003645,
        -0.23310120724711264, -1.3785001086357498, 0.29212952145108617,
        3.883405050631268, 1.6609955045466307, -0.5505276084920737,
        -0.9197445587649881, -0.8568021863015526, -0.09628001529522923,
        0.6481113045422939, 0.6444097543374807, 5.200617273321904,
        3.6992358598512918,
    ])
    # Independent log-GIG integration and recursion agree at 45/70 digits.
    # Binary64 moment/recursion errors previously changed this finite NLL by 935.
    actual = model.evaluate(theta, mask)
    np.testing.assert_allclose(actual[0], 4795.219269986339, rtol=0, atol=1e-9)


@pytest.mark.parametrize("density", range(8))
def test_egarch_reports_scalar_precision_and_serializes_evidence(density):
    model, x = prepared(2, density, n=60)
    result = fit_joint_mean_spec(model.spec, model.data, x=x, maxiter=2)
    expected = (
        "double-double primal state; binary64 derivatives"
        if density != 0
        else "binary64"
    )
    assert result.fit_info.scalar_precision == expected
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.scalar_precision == expected
    assert restored.to_dict() == result.fit_info.to_dict()


@pytest.mark.parametrize("density,shape_order", ((5, -0.5), (6, 1.0)))
@pytest.mark.parametrize("p,q,har", ((1, 1, False), (3, 2, True), (4, 3, False)))
def test_joint_egarch_gh_exact_reductions(density, shape_order, p, q, har):
    general, x = prepared(2, 4, p, q, har=har, n=20)
    reduced = JointMeanModel(
        general.mean + FAMILIES[2](p, q) + DENSITIES[density](), general.data, x
    )
    theta = general.initialize()
    theta[:general.columns] = 0.025
    theta[-3:] = (0.12, 1.8, shape_order)
    full = general.evaluate(theta, 7, paths=True)
    special = reduced.evaluate(theta[:-1].copy(), 7, paths=True)
    for actual, expected in ((special[0], full[0]), (special[1], full[1][:-1]),
                             (special[2], full[2][:-1, :-1])):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(special[4], full[4], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("extra_columns", (0, 1, 5))
@pytest.mark.parametrize(
    "theta,expected_gradient",
    [
        (
            [
                -0.007873098550010591,
                0.04163515113467865,
                0.016617425080098454,
                0.008111460601629254,
                -0.006818393783877042,
                -0.05304266118671755,
                0.020047320462004067,
                -0.013745097547291673,
                -0.4296462468735065,
                0.0591623551247326,
                0.14684579081605548,
                -0.2237330899604478,
                0.06750805963378748,
                0.2060220042888797,
                0.5568923933506515,
                0.4106887733565826,
            ],
            [
                -16522.76584211095,
                -127498.96113077948,
                -34796.03434145808,
                44379.5699246518,
                -20147.233131137822,
                -33211.90248768698,
                89881.27133420417,
                1878263.280939353,
                -206576.65727578348,
                -221126.00547059104,
                -211784.35182238545,
                50966.446917870584,
                33605.91156925018,
                27966.173234772337,
                168486.44033291982,
                156879.10966124205,
            ],
        ),
        (
            [
                -0.005088609524830868,
                0.0324532916037479,
                0.03859810928967342,
                -0.025057931979625775,
                -0.023654564724780867,
                -0.06010181837578299,
                0.011190740681321035,
                -0.017924548568031322,
                -0.3965111199202637,
                0.010626031950956905,
                0.11621243117292689,
                -0.16105490583997048,
                0.03420105546153657,
                0.19006092443122657,
                0.4433162847010734,
                0.514093453965909,
            ],
            [
                43483.76088759937,
                184091.6973349966,
                51486.69104764359,
                -63759.91555899952,
                29202.752635404933,
                48975.85898073846,
                -129913.20201976466,
                -2357424.6030321103,
                262227.39138750825,
                282846.10413253703,
                269183.6452141691,
                -94390.230423671,
                -68821.16113570501,
                -58650.956706618985,
                -230221.7492380002,
                -213460.6987924458,
            ],
        ),
    ],
    ids=("ordinary_endpoint", "accurate_endpoint"),
)
def test_egarch_normal_stress_accuracy_is_independent_of_mean_width(
    theta, expected_gradient, extra_columns
):
    # Independent MP50/70 recursion at unchanged doubles. Added zero regressors
    # alter reduction widths, not residuals or the statistical objective.
    rng = np.random.default_rng(2785)
    data, x = rng.normal(size=400), rng.normal(size=(400, 4))
    x = np.column_stack((x, np.zeros((len(data), extra_columns))))
    model = JointMeanModel(HARX([1, 3]) + FAMILIES[2](3, 2), data, x)
    theta = np.r_[theta[:7], np.zeros(extra_columns), theta[7:]]
    expected = np.r_[
        expected_gradient[:7], np.zeros(extra_columns), expected_gradient[7:]
    ]
    full = model.evaluate(theta, 7, paths=True)
    for mask in range(1, 8):
        actual = model.evaluate(theta, mask, paths=True)
        np.testing.assert_allclose(actual[4], full[4], rtol=2e-10, atol=2e-12)
        for bit, index in ((1, 0), (2, 1), (4, 2)):
            if mask & bit:
                np.testing.assert_allclose(
                    actual[index], full[index], rtol=2e-6, atol=3e-6
                )
        if mask & 2:
            np.testing.assert_allclose(actual[1], expected, rtol=2e-6, atol=3e-6)


def test_overflowing_multiplier_search_is_quiet_and_preserves_feasibility():
    import warnings

    model, _ = prepared()
    theta = model.initialize()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        feasible, bound, violation = model.assess(theta, np.full(model.size, 1e200))
    assert feasible
    # A representable conservative bound is preferable to an overflowing search.
    # The huge free force must still prevent a stationarity claim.
    assert np.isfinite(bound) and bound > 1e190
    assert violation == 0


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", range(8))
def test_joint_domain_report_matches_the_actual_plan(family, density):
    import json

    model, _ = prepared(family, density, p=3, q=2, har=True)
    domain = json.loads(json.dumps(model.describe_domain()))
    assert domain["mean_polynomial_order"] == 3
    assert domain["mean_coefficient_count"] == 2
    assert domain["volatility_orders"] == [3, 2]
    assert domain["density"] == model.density.signature
    assert domain["first_scored"] == 0
    assert domain["scored_observations"] == len(model.data)
    assert domain["initial_variance"] == model.h0
    for key, actual in (
        ("numerical_lower", model.lower),
        ("numerical_upper", model.upper),
    ):
        assert len(domain[key]) == model.size
        np.testing.assert_equal(
            np.array([np.nan if v is None else v for v in domain[key]]), actual
        )
    assert ("delta < nu" in domain["volatility"]) == (family == 3 and density in (1, 2))


def test_joint_fit_exposes_the_model_owned_domain():
    model, x = prepared()
    fit = fit_joint_mean_spec(model.spec, model.data, x=x, maxiter=1)
    assert fit.fit_info.parameter_domain == model.describe_domain()


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_legacy_joint_gjr_uses_density_weighted_domain(density, har, p, q, monkeypatch):
    import importlib
    import scipy.optimize
    from scivol._distribution_moments import negative_half_second_moment
    from scivol._kernels import get_routine

    model, x = prepared(1, density, p, q, n=64, har=har)
    theta = model.initialize()
    shape = theta[model.shape_start :]
    arguments = {}
    if density == 2:
        shape[1] = 0.35
        arguments = dict(nu=shape[0], lam=shape[1])
    elif density >= 4:
        shape[0] = 0.25
        arguments = dict(
            rho=shape[0],
            shape_lambda=(
                shape[1]
                if density == 7
                else shape[2] if density == 4 else -0.5 if density == 5 else 1.0
            ),
        )
        if density != 7:
            arguments["zeta"] = shape[1]
    moment = negative_half_second_moment(model.density.signature, **arguments)
    alpha = slice(model.columns + 1, model.columns + 1 + p)
    gamma = slice(alpha.stop, alpha.stop + p)
    beta = slice(gamma.stop, gamma.stop + q)
    theta[alpha], theta[gamma] = 0.04 / p, 0.2 / p
    theta[beta] = (0.97 - 0.04 - 0.2 * moment) / q
    assert theta[alpha].sum() + theta[gamma].sum() + theta[beta].sum() > 1

    class ConstraintCaptured(Exception):
        pass

    def inspect_minimize(objective, initial, **kwargs):
        constraint = kwargs["constraints"]
        values, jacobian = constraint["fun"](theta), constraint["jac"](theta)
        assert np.all(values > 0)
        np.testing.assert_allclose(values[-1], 0.03, rtol=0, atol=1e-10)
        np.testing.assert_allclose(jacobian[-1, gamma], -moment, rtol=0, atol=3e-10)
        if density in (2, 4, 5, 6, 7):
            assert np.linalg.norm(jacobian[-1, model.shape_start :]) > 1e-8
        assert np.isfinite(objective(theta))
        assert np.isfinite(kwargs["jac"](theta)).all()
        raise ConstraintCaptured

    def forbid_native_likelihood(*args, **kwargs):
        raise AssertionError("A legacy likelihood comparison must stay independent.")

    routine = get_routine(str(model.spec))
    module = importlib.import_module(routine.fit.__module__)
    monkeypatch.setattr(scipy.optimize, "minimize", inspect_minimize)
    if hasattr(module, "minimize"):
        monkeypatch.setattr(module, "minimize", inspect_minimize)
    monkeypatch.setattr(JointMeanModel, "evaluate", forbid_native_likelihood)
    with pytest.raises(ConstraintCaptured):
        routine.fit(model.data, x=x, initial_params=theta, maxiter=5, log_mode=False)


@pytest.mark.parametrize("family", (2, 3))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (2, 4, False)])
def test_affine_joint_phase_keeps_orders_density_and_coordinates(
    family, density, transformed, p, q, har
):
    model, _ = prepared(family, density, p, q, har=har)
    initial = model.initialize()
    normal = np.zeros(model.columns)
    normal[0] = 1.0
    normal[-1] = 0.25
    target = float(normal @ initial[: model.columns])
    result = _solve_joint_phase(
        model,
        initial,
        transformed=transformed,
        accuracy=1e-12,
        maxiter=3,
        use_curvature=True,
        affine=(normal, target),
    )
    assert result.kernel_dispatch == (0 if (p, q) == (2, 4) else 2)
    assert result.nit <= 3 and result.nfev > 0 and result.njev > 0
    assert result.affine_equality["normal"] == normal.tolist()
    assert result.affine_equality["target"] == target
    assert result.affine_equality["violation"] < 2e-10
    assert model.assess(result.x, None)[0]
    assert result.fun <= model.evaluate(initial, 1)[0]
    np.testing.assert_allclose(
        result.fun, model.evaluate(result.x, 1)[0], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("family", (2, 3))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (2, 4, False)])
def test_coordinate_subset_keeps_joint_jet_dispatch_and_held_parameters(
    family, density, transformed, p, q, har,
):
    model, _ = prepared(family, density, p, q, har=har)
    initial = model.initialize()
    saved = initial.copy()
    held = model.columns-1
    free = np.delete(np.arange(model.size), held)
    result = _solve_joint_phase(model, initial, transformed=transformed,
        accuracy=1e-12, maxiter=3, use_curvature=True, free_coordinates=free)
    np.testing.assert_array_equal(initial, saved)
    assert result.x[held] == initial[held]
    assert result.coordinate_subset == dict(space="z" if transformed else "theta", free=free.tolist())
    assert result.nit <= 3 and result.nfev > 0 and result.njev > 0
    assert result.kernel_dispatch == (0 if (p, q) == (2, 4) else 2)
    assert model.assess(result.x, None)[0]
    value, gradient = model.evaluate(result.x, 3)[:2]
    assert abs(value-result.fun) < 1e-9
    np.testing.assert_allclose(result.jac, gradient, rtol=2e-9, atol=2e-9)
    assert result.fun <= model.evaluate(initial, 1)[0]


@pytest.mark.parametrize("family", (2, 3))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (2, 4, False)])
def test_coordinate_subset_composes_with_affine_constraints(
    family, transformed, fixed, p, q, har,
):
    model, _ = prepared(family, 1, p, q, har=har, fixed=fixed)
    initial = model.initialize()
    normal = np.eye(model.columns)[:2]
    result = _solve_joint_phase(model, initial, transformed=transformed,
        accuracy=1e-12, maxiter=5, use_curvature=False,
        affine=(normal, initial[:2]), free_coordinates=np.arange(model.size-1))
    np.testing.assert_allclose(result.x[:2], initial[:2], atol=1e-14, rtol=0)
    if not transformed:
        assert result.x[-1] == initial[-1]
    else:
        np.testing.assert_allclose(model.inverse(result.x)[-1], model.inverse(initial)[-1], atol=1e-12)
    assert result.nit <= 5 and result.affine_equality["violation"] < 1e-14
    assert result.kernel_dispatch == (2 if fixed and (p, q) != (2, 4) else 0)
    assert model.assess(result.x, None)[0]


@pytest.mark.parametrize("indices", ([], [0, 0], [1, 0], [-1], [100], [1.5], [[1]]))
def test_coordinate_subset_rejects_invalid_indices_before_evaluation(indices):
    model, _ = prepared(3)
    initial = model.initialize()
    before = tuple(model.request_counts)
    with pytest.raises(ValueError, match="coordinate indices"):
        _solve_joint_phase(model, initial, transformed=False, accuracy=1e-12,
            maxiter=1, free_coordinates=indices)
    assert tuple(model.request_counts) == before


@pytest.mark.parametrize("failure", ["short", "unaligned", "index", "count"])
def test_native_coordinate_subset_checks_its_workspace_and_indices(monkeypatch, failure):
    from scivol import _native_core

    model, _ = prepared(3, 3)
    initial = model.initialize()
    invalid_indices = np.array([model.size], dtype=np.uintp)
    native = _native_core._linear_joint_fit
    called = []

    def invalid(*arguments):
        args = list(arguments)
        if failure == "short":
            args[-1] -= 1
        elif failure == "unaligned":
            args[-2] += 1
        elif failure == "index":
            args[-4], args[-3] = invalid_indices.ctypes.data, 1
        else:
            args[-3] = model.size+1
        valid = native(*args)
        called.append(valid)
        return valid

    monkeypatch.setattr(_native_core, "_linear_joint_fit", invalid)
    result = _solve_joint_phase(model, initial, transformed=False, accuracy=1e-12,
        maxiter=3, use_curvature=False, free_coordinates=np.arange(model.size-1))
    assert called == [False]
    assert result.native_jet_requests == {"F": 0, "G": 0, "F|G": 0}
    assert not result.success
    np.testing.assert_array_equal(result.x, initial)


@pytest.mark.parametrize("fixed", (False, True))
def test_aparch_active_bound_refinement_restores_full_domain_stationarity(fixed):
    from benchmark_native_joint import comparison_sample

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3]() + DENSITIES[3](), data, x, fixed=fixed)
    initial = np.array([
        -.06285050083178957, -.02605951153478455, .04473336130019931,
        .024516780208625678, .002434558802213593, 0., -.9999999829025641,
        .863570334440023, 230.33185647943873, 1.8456850857445533,
    ])
    value, gradient = model.evaluate(initial, 3)[:2]
    assert gradient[5] > 1e190 and model.assess(initial, gradient)[1] > 1e-4
    result = _solve_joint_phase(model, initial, transformed=False, accuracy=1e-12,
        maxiter=100, use_curvature=False, speculate=True, curvature_reset=False,
        free_coordinates=np.delete(np.arange(model.size), 5))
    assert result.fun < value-.0008 and result.nit <= 100 and result.x[5] == 0
    feasible, residual, violation = model.assess(result.x, result.jac)
    assert result.success and feasible and violation == 0 and residual <= 1e-6
    assert result.jac[5] > 1e190
    # Independent 40/60-digit scalar values and the same-coordinate SciPy control.
    assert abs(result.fun-561.4753720807962) < 1e-9


def test_public_aparch_active_bound_refinement_records_full_domain_evidence(capsys):
    from benchmark_native_joint import comparison_sample

    data, x = comparison_sample(2786, False, "normal")
    spec = ARX(1) + FAMILIES[3]() + DENSITIES[3]()
    result = fit_joint_mean_spec(spec, data, x=x, maxiter=1000)
    assert result.log_likelihood >= -561.4753720807962-1e-9
    assert result.converged and result.fit_info.stationarity_status == "certified"
    assert result.fit_info.maximum_constraint_violation == 0
    phase = result.fit_info.phase_diagnostics[-1]
    assert phase["purpose"] == "active-bound-refinement"
    assert phase["fixed_coordinates"] == [5]
    assert phase["iterations"] == 17 and phase["iteration_budget"] == 100
    assert sum(p["iterations"] for p in result.fit_info.phase_diagnostics) <= 1000
    assert phase["kernel_dispatch"] == 2
    # Certification of the estimate cannot manufacture finite observed information.
    assert result.fit_info.inference_status == "unavailable"
    assert result.std_errors is None and result.warnings
    assert capsys.readouterr().out == ""
    model = JointMeanModel(spec, data, x)
    with pytest.raises(ValueError, match="Joint domain evaluation failed"):
        model.constraints(result.params)
    value, gradient = model.evaluate(result.params, 3)[:2]
    feasible, bound, violation = model.assess(result.params, gradient)
    assert feasible and bound <= 1e-6 and violation == 0
    assert abs(value+result.log_likelihood) <= 1e-9


@pytest.mark.parametrize("fixed", (False, True))
def test_aparch_ged_unrepresentable_constraint_moment_keeps_exact_fg(fixed):
    pytest.importorskip("jax")
    from benchmark_native_joint import comparison_sample
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3]() + DENSITIES[3](), data, x, fixed=fixed)
    theta = np.array([
        -.06331738402043866, -.026508086222618483, .044348769202867094,
        .02510588299035908, .0019223818062048428, 0., -.999999981596122,
        .8783086203578239, 231.20175044099793, 1.839478852061437,
    ])
    value, gradient = model.evaluate(theta, 3)[:2]
    # Independent 40/60-digit recurrences agree at these exact input doubles.
    assert abs(value-561.4753650756267) <= 1e-9
    expected = joint_value_grad_hess(theta, model.data, model.features, model.h0,
        1, 1, 3, 3, request=3, exact_mean_anchor=True)
    np.testing.assert_allclose(gradient, expected[1], rtol=1e-8, atol=1e-8)
    assert abs(value-expected[0]) <= 1e-9
    assert model.evaluate(theta, 1)[0] == value
    np.testing.assert_array_equal(model.evaluate(theta, 2)[1], gradient)
    feasible, bound, violation = model.assess(theta, gradient)
    assert feasible and bound <= 1e-6 and violation == 0
    with pytest.raises(ValueError, match="Joint domain evaluation failed"):
        model.constraints(theta)
    for request in (4, 5, 6, 7):
        with pytest.raises(ValueError, match="invalid at these parameters"):
            model.evaluate(theta, request)


@pytest.mark.parametrize("fixed", (False, True))
def test_general_native_kink_phase_recovers_likelihood_without_snapping(fixed):
    rng = np.random.default_rng(2785)
    data, x = rng.normal(size=400), rng.normal(size=(400, 2))
    model = JointMeanModel(
        ARX(1) + FAMILIES[2](1, 1) + DENSITIES[7](), data, x, fixed=fixed
    )
    initial = np.array(
        [
            0.020708708287308312,
            -0.060204662592039895,
            -0.04213069067569303,
            0.09418706857884712,
            -0.13384290026634943,
            -0.05064499529787504,
            -0.16786302124383154,
            0.009127659299055695,
            -0.11827069892710707,
            4.8103260409111614,
        ]
    )
    result = _solve_joint_phase(
        model,
        initial,
        transformed=False,
        accuracy=np.finfo(float).eps,
        maxiter=100,
        use_curvature=False,
        affine=(model.features[212], model.data[212]),
    )
    assert result.fun < model.evaluate(initial, 1)[0] - 1e-7
    assert result.fun <= 536.6166985052951
    assert result.kernel_dispatch == (2 if fixed else 0)
    assert result.affine_equality["violation"] < 1e-15
    value, gradient, _, residual, _ = model.evaluate(result.x, 3, paths=True)
    assert model.evaluate(result.x, 1)[0] == result.fun
    # Separate request plans may round differently; the phase's own F rescore is exact.
    np.testing.assert_array_max_ulp(value, result.fun, maxulp=2)
    assert residual[212] != 0
    assert model.assess(result.x, gradient)[1] > 1e-6


@pytest.mark.parametrize(
    "normal,target",
    [([0, 0, 0, 0], 0), ([1, 0], 0), ([1, 0, 0, np.nan], 0), ([1, 0, 0, 0], np.inf)],
)
def test_invalid_affine_phase_is_rejected_before_solving(normal, target):
    model, _ = prepared(2)
    with pytest.raises(ValueError, match="affine"):
        _solve_joint_phase(
            model,
            model.initialize(),
            transformed=False,
            accuracy=1e-12,
            maxiter=1,
            affine=(normal, target),
        )


@pytest.mark.parametrize("short_arena", (False, True))
@pytest.mark.parametrize("family,density", ((2, 0), (3, 4)))
def test_affine_phase_requires_a_valid_normal_and_planned_arena(
    short_arena, family, density
):
    from scivol import _native_core
    from scivol._mean_fit import _pointer

    model, _ = prepared(family, density)
    initial = model.initialize()
    normal = np.eye(1, model.columns)[0]
    if not short_arena:
        normal[:] = 0
    work = model.workspace(3, solve=True, affine=True)
    assert work.nbytes > model.workspace(3, solve=True).nbytes
    outputs = [
        np.full(model.size, 42.0),
        np.full(model.size, 42.0),
        np.full(1, 42.0),
        np.full(6, 42, np.int64),
    ]
    scales = np.ones(model.size)
    valid = _native_core._linear_joint_fit(
        model.descriptor,
        model.domain,
        _pointer(initial),
        _pointer(model.data),
        _pointer(model.features),
        model.h0,
        _pointer(model.lower),
        _pointer(model.upper),
        _pointer(scales),
        _pointer(work),
        work.nbytes - int(short_arena),
        *[_pointer(array) for array in outputs],
        1e-12,
        3,
        False,
        True,
        _pointer(normal),
        0.0,
    )
    assert not valid
    for array in outputs:
        np.testing.assert_equal(array, 42)


@pytest.mark.parametrize("density,theta", [
    (7, [
        0.020710762751471757, -0.06020333139468085, -0.04212794215375844,
        0.0941897416011332, -0.13385337831682317, -0.05068053603263031,
        -0.16784771939987161, 0.009119991210401284, -0.1182976959610899,
        4.8114439489744925,
    ]),
    (2, [
        0.017144658059546406, -0.060554757697977225, -0.03699221525512182,
        0.0924935354539109, -0.1421607090545204, -0.04837893967032467,
        -0.1641805646710932, 0.0017759608725398796, 23.09666821241392,
        -0.09937913646971636,
    ]),
])
def test_egarch_refinement_has_limiting_not_smooth_stationarity(density, theta):
    from fractions import Fraction
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    rng = np.random.default_rng(2785)
    data, x = rng.normal(size=400), rng.normal(size=(400, 2))
    model = JointMeanModel(ARX(1) + FAMILIES[2](1, 1) + DENSITIES[density](), data, x)
    theta = np.array(theta)
    row = 212
    value, gradient, _, residuals, _ = model.evaluate(theta, 3, paths=True)
    assert residuals[row] != 0
    assert model.assess(theta, gradient)[1] > 1e-6
    args = (theta, model.data, model.features, model.h0, 1, 1, 2, density)
    left_value, left, _ = joint_value_grad_hess(*args, request=3, kink_limit=(row, -1))
    right_value, right, _ = joint_value_grad_hess(*args, request=3, kink_limit=(row, 1))
    left, right = left * model.units / len(data), right * model.units / len(data)
    jump = right - left
    weight = -(left @ jump) / (jump @ jump)
    assert 0 < weight < 1
    assert max(abs(left)) > 1e-6 and max(abs(right)) > 1e-6
    assert max(abs(left + weight * jump)) < 1e-7
    np.testing.assert_allclose([left_value, right_value], value, rtol=0, atol=2e-10)
    exact_intercept = Fraction(float(data[row])) - sum(
        Fraction(float(a)) * Fraction(float(b))
        for a, b in zip(model.features[row, 1:], theta[1 : model.columns])
    )
    backward_error = float(abs(exact_intercept - Fraction(float(theta[0]))))
    assert backward_error < np.spacing(theta[0])


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("transformed", (False, True))
def test_optimizer_records_the_dispatch_it_actually_executes(
    family, fixed, transformed
):
    model, _ = prepared(family, fixed=fixed)
    result = _solve_joint_phase(
        model,
        model.initialize(),
        transformed=transformed,
        accuracy=1e-12,
        maxiter=1,
        use_curvature=False,
    )
    assert result.kernel_dispatch == (2 if fixed else 0)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True)])
@pytest.mark.parametrize("log_mode", (None, False, True))
def test_aparch_public_controller_executes_expected_fixed_dispatch(
    monkeypatch, density, p, q, har, log_mode
):
    import ctypes
    from scivol import _native_core
    import scivol._joint_mean_fit as joint

    model, x = prepared(3, density, p, q, har=har)
    native = _native_core._linear_joint_fit
    portfolio = joint.solve_joint_with_metric_portfolio
    dispatches, portfolio_calls = [], []
    expected_layout = (7, 3, 2) if har else (4, 1, 1)

    def observe_native(*args):
        assert args[0][:2] == (3, density)
        assert args[0][2:5] == expected_layout and args[0][6]
        valid = native(*args)
        stats = np.ctypeslib.as_array(
            ctypes.cast(args[14], ctypes.POINTER(ctypes.c_int64)), shape=(6,)
        ).copy()
        assert stats[5] == 2
        assert stats[2] > 1 and stats[3] > 1
        dispatches.append(bool(args[17]))
        return valid

    def observe_portfolio(*args, **options):
        portfolio_calls.append(options.copy())
        return portfolio(*args, **options)

    monkeypatch.setattr(_native_core, "_linear_joint_fit", observe_native)
    monkeypatch.setattr(joint, "solve_joint_with_metric_portfolio", observe_portfolio)
    fit_joint_mean_spec(model.spec, model.data, x=x, log_mode=log_mode, maxiter=1)
    assert dispatches
    if log_mode is None:
        assert len(portfolio_calls) == 1
        assert portfolio_calls[0]["maxiter"] == 1
        assert portfolio_calls[0]["allow_chart_refinement"]
        before = len(dispatches)
        portfolio(model, model.initialize(), accuracy=1e-12, maxiter=4,
            primary_iteration_limit=1, alternate_iteration_limit=1,
            continuation_iteration_limit=1)
        assert set(dispatches[before:]) == {False, True}
    else:
        assert not portfolio_calls
        assert all(space == log_mode for space in dispatches)


@pytest.mark.parametrize("log_mode", (False, True), ids=("theta-rejected", "z-solver"))
@pytest.mark.parametrize("code,shape", [
    (0, []), (1, [8.0]), (2, [8.0, 0.12]), (3, [1.6]),
    (4, [0.1, 1.3, 0.4]), (5, [0.1, 1.3]), (6, [0.1, 1.3]), (7, [0.1, 3.0]),
])
@pytest.mark.parametrize("orders,volatility_orders,expected", [
    pytest.param(None, (1, 1), 2, id="standalone11"),
    pytest.param(None, (1, 2), 2, id="standalone12"),
    pytest.param(None, (2, 1), 2, id="standalone21"),
    pytest.param(None, (2, 2), 2, id="standalone22"),
    pytest.param(None, (3, 2), 1, id="standalone-runtime"),
    pytest.param((1, 1, 1, 1), (1, 1), 2, id="arma1111"),
    pytest.param((2, 1, 1, 1), (1, 1), 1, id="arma-runtime"),
])
def test_optimizer_executes_expected_aparch_dispatch(
    monkeypatch, log_mode, code, shape, orders, volatility_orders, expected
):
    from scivol import APARCH, ARMA, _native_core
    from scivol import _native_fit as native

    p, q = volatility_orders
    spec = APARCH(p, q) + DENSITIES[code]()
    mean = np.empty(0)
    if orders is not None:
        ar, ma, p, q = orders
        spec = ARMA(ar, ma) + spec
        mean = np.r_[0.03, np.full(ar, 0.12 / ar), np.full(ma, -0.08 / ma)]
    model = native._describe_model(spec)
    data = np.random.default_rng(42).normal(size=32).astype(np.float64)
    theta = np.r_[mean, 0.08, np.full(p, 0.07 / p), np.linspace(0.15, -0.1, p),
                  np.full(q, 0.82 / q), 1.7, shape]
    entry = "_fit_arma_aparch" if orders is not None else "_fit_aparch"
    actual = getattr(_native_core, entry)
    calls = []

    def observe(*args):
        raw = actual(*args)
        calls.append(raw)
        return raw

    monkeypatch.setattr(_native_core, entry, observe)
    if not log_mode:
        with pytest.raises(ValueError, match="APARCH no longer exposes log_mode=False"):
            spec.fit(data, solver="native", log_mode=False, maxiter=1)
        with pytest.raises(ValueError, match="transformed-coordinate path"):
            native.fit_native_spec(spec, data, log_mode=False, maxiter=1)
        with pytest.raises(ValueError, match="canonical SciPy jet requires log_mode=True"):
            native.fit_scipy_canonical_spec(spec, data, log_mode=False, maxiter=1)
        assert not calls
        return
    coordinates = native._theta_to_coordinates(model, theta)
    solve = native._solve_coordinates(model, data, coordinates, accuracy=1e-10,
                                      maximum_iterations=1)
    assert len(calls) == 1 and len(calls[0]) == 11
    assert calls[0][5] > 0 and calls[0][6] > 0
    assert calls[0][10] == solve.execution_dispatch == expected
    assert solve.kernel_dispatch == (1 if expected == 2 else 0)


@pytest.mark.parametrize("execution,kind", [(0, None), (1, 0), (2, 1), (3, None)])
def test_aparch_execution_proof_parses_without_inventing_dispatch(execution, kind):
    from scivol import _native_fit as native

    raw = (9, 1.0, 0.0, 1.0, 1, 1, 1, 0, 0, 0)
    legacy = native._native_solve(raw)
    assert legacy.execution_dispatch is legacy.kernel_dispatch is None
    solve = native._native_solve((*raw, execution))
    assert solve.execution_dispatch == execution and solve.kernel_dispatch == kind
    assert native._native_solve((*raw, np.int64(execution))).execution_dispatch == execution
    combined = native._combine_native_solves(native._native_solve((*raw, 1)),
                                            native._native_solve((*raw, 2)))
    assert combined.execution_dispatch == 3 and combined.kernel_dispatch is None


@pytest.mark.parametrize("execution", (2.9, 2.0, "2", True, np.bool_(False), None, 4, -1))
def test_aparch_execution_proof_rejects_noninteger_or_invalid_bits(execution):
    from scivol import _native_fit as native

    raw = (9, 1.0, 0.0, 1.0, 1, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="execution-dispatch bitmask"):
        native._native_solve((*raw, execution))


@pytest.mark.parametrize("first,second,expected", [
    (None, 2, None), (2, None, None), (0, 2, None), (2, 0, None),
    (None, None, None), (2, 2, 2), (1, 1, 1), (1, 2, 3), (None, 3, 3),
])
def test_aparch_execution_proof_keeps_unobserved_phases_unknown(first, second, expected):
    from types import SimpleNamespace
    from scivol import _native_fit as native

    raw = (9, 1.0, 0.0, 1.0, 1, 1, 1, 0, 0, 0)
    solves = [native._native_solve(raw if bits is None else (*raw, bits))
              for bits in (first, second)]
    combined = native._combine_native_solves(*solves)
    assert combined.execution_dispatch == expected
    kind = {1: 0, 2: 1}.get(expected)
    assert combined.kernel_dispatch == kind
    diagnostics = tuple(dict(phase=str(index), budget=1, mode=9, failure=None,
        evidence={} if bits is None else {"execution_dispatch": bits})
        for index, bits in enumerate((first, second)))
    controller = SimpleNamespace(diagnostics=diagnostics, trigger_policy="always",
        selected_phase="0", total_work=lambda: native.PhaseWork(),
        point=SimpleNamespace(coordinates=np.zeros(1), warnings=()))
    info = FitMetadata(jet_request_plan="canonical")
    native._apply_aparch_controller_metadata(info, controller)
    assert info.kernel_dispatch == kind
    assert info.phase_diagnostics == diagnostics


@pytest.mark.parametrize("unobserved_mean_phase", (False, True), ids=("canonical-only", "mean-phase"))
@pytest.mark.parametrize("log_mode", (None, True), ids=("default", "explicit-z"))
@pytest.mark.parametrize("retain_execution", (False, True), ids=("missing", "witnessed"))
def test_aparch_execution_proof_survives_final_filter_and_reporting(
    monkeypatch, unobserved_mean_phase, log_mode, retain_execution
):
    from scivol import APARCH, ARMA
    from scivol import _native_fit as native

    if not retain_execution:
        decode = native._native_solve
        monkeypatch.setattr(native, "_native_solve", lambda raw: decode(raw[:10]))

    spec = APARCH(1, 1) + Normal()
    if unobserved_mean_phase:
        spec = ARMA(1, 1) + spec
        monkeypatch.setattr(native, "_APARCH_MEAN_REFINEMENT_ENABLED", True)
        monkeypatch.setattr(native, "_refine_aparch_mean_if_needed",
            lambda model, data, theta, solve, value, gradient, **options:
                (theta, solve, value, gradient, {"phase": "unobserved-mean",
                    "scope": "conditional", "stop": 0, "retained": False}))
    data = np.random.default_rng(42).normal(size=32).astype(np.float64)
    result = spec.fit(data, solver="native", log_mode=log_mode, maxiter=1)
    assert result.fit_info.function_evaluations > 0
    assert result.fit_info.gradient_evaluations > 0
    kind = 1 if retain_execution and not unobserved_mean_phase else None
    assert result.fit_info.kernel_dispatch == kind
    phases = [phase for phase in result.fit_info.phase_diagnostics if "execution_dispatch" in phase]
    if retain_execution:
        assert len(phases) == 1
        assert phases[0]["execution_dispatch"] == 2 and phases[0]["kernel_dispatch"] == 1
        assert "excludes KKT/filter/inference" in phases[0]["scope"]
        assert phases[0] in result.fit_info.to_dict()["phase_diagnostics"]
    else:
        assert not phases


@pytest.mark.parametrize("family", (2, 3))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (2, 4, False)])
@pytest.mark.parametrize("restricted", (False, True))
def test_transformed_solver_reports_the_evaluated_point(
    monkeypatch, family, density, p, q, har, restricted,
):
    import ctypes
    from scivol import _native_core

    model, x = prepared(family, density, p, q, har=har)
    runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    initial = model.initialize()
    np.testing.assert_allclose(
        model.inverse(initial), runtime.inverse(initial), rtol=2e-14, atol=1e-13,
    )
    native = _native_core._linear_joint_fit
    returned = []

    def capture(*args):
        valid = native(*args)
        assert valid
        coordinates, theta, gradient, value = (
            np.ctypeslib.as_array(
                ctypes.cast(args[index], ctypes.POINTER(ctypes.c_double)),
                shape=(size,),
            ).copy()
            for index, size in ((2, model.size), (11, model.size),
                                (12, model.size), (13, 1))
        )
        returned.append((coordinates, theta, gradient, value[0]))
        return valid

    monkeypatch.setattr(_native_core, "_linear_joint_fit", capture)
    result = _solve_joint_phase(
        model, initial, transformed=True, accuracy=1e-12, maxiter=2,
        use_curvature=False,
        free_coordinates=np.arange(model.size-1) if restricted else None,
    )
    assert result.kernel_dispatch == (0 if (p, q) == (2, 4) else 2)
    assert len(returned) == 1
    coordinates, theta, gradient, value = returned[0]
    coordinate_value = model.evaluate(coordinates, 1, transformed=True)[0]
    theta_value, theta_gradient = model.evaluate(theta, 3)[:2]
    assert abs(coordinate_value-theta_value) < 1e-9
    assert value == theta_value
    np.testing.assert_array_equal(gradient, theta_gradient)


@pytest.mark.parametrize("family", (2, 3))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
def test_solver_retention_is_planned_only_for_wide_egarch(family, density, transformed):
    from scivol import _native_core

    model, _ = prepared(family, density)
    longer = list(model.descriptor)
    extra_observations = 17
    longer[5] += extra_observations
    first = _native_core._linear_joint_sizes(model.descriptor, 3, transformed, model.domain)
    second = _native_core._linear_joint_sizes(tuple(longer), 3, transformed, model.domain)
    assert first[0] == second[0]  # Standalone jets retain no sample-length paths.
    bytes_per_observation = 2 * 4 * 2 * np.dtype(np.float64).itemsize
    expected = bytes_per_observation * extra_observations if family == 2 and density else 0
    assert second[1] - first[1] == expected


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True)])
def test_checked_joint_sparse_requests_and_runtime_agree(family, density, p, q, har):
    model, x = prepared(family, density, p, q, har=har)
    runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    theta = model.initialize()
    theta[: model.columns] = 0.025
    for transformed in (False, True):
        point = model.inverse(theta) if transformed else theta
        expected = model.evaluate(point, 7, transformed=transformed)
        reference = runtime.evaluate(point, 7, transformed=transformed)
        for actual, other in zip(expected[:3], reference[:3]):
            np.testing.assert_allclose(actual, other, rtol=2e-8, atol=2e-8)
        for mask in range(1, 8):
            actual = model.evaluate(point, mask, transformed=transformed)
            for bit, value, full in zip((1, 2, 4), actual[:3], expected[:3]):
                if mask & bit:
                    np.testing.assert_allclose(value, full, rtol=2e-8, atol=2e-8)
                else:
                    assert value is None


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("coordinate", (21.0, 40.0))
def test_joint_student_coordinates_preserve_the_full_softplus_map(family, coordinate):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    model, _ = prepared(family, 1, n=9)
    theta = model.initialize()
    theta[-1] = 2.0 + np.logaddexp(0.0, coordinate)
    point = model.inverse(theta)
    np.testing.assert_allclose(point[-1], coordinate, rtol=0, atol=8e-15)
    actual = model.evaluate(point, 7, transformed=True)
    expected = joint_value_grad_hess(
        theta,
        model.data,
        model.features,
        model.h0,
        1,
        1,
        family,
        1,
        coordinates=point,
    )
    for value, reference in zip(actual[:3], expected):
        np.testing.assert_allclose(value, reference, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("density", [2, 4, 5, 6, 7])
@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (4, 3, False)])
def test_sublinear_joint_aparch_preserves_requests_maps_and_general_orders(
    density, p, q, har
):
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    model, x = prepared(3, density, p, q, har=har, n=9)
    runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    theta = model.initialize()
    theta[: model.columns] = 0.025
    theta[model.shape_start - 1] = 0.7
    for transformed in (False, True):
        point = model.inverse(theta) if transformed else theta
        expected = model.evaluate(point, 7, transformed=transformed)
        reference = runtime.evaluate(point, 7, transformed=transformed)
        for actual, other in zip(expected[:3], reference[:3]):
            np.testing.assert_allclose(actual, other, rtol=2e-8, atol=2e-8)
        for mask in range(1, 8):
            actual = model.evaluate(point, mask, transformed=transformed)
            for bit, value, full in zip((1, 2, 4), actual[:3], expected[:3]):
                if mask & bit:
                    np.testing.assert_allclose(value, full, rtol=2e-8, atol=2e-8)
                else:
                    assert value is None
        if (p, q) == (4, 3):
            independent = joint_value_grad_hess(
                theta,
                model.data,
                model.features,
                model.h0,
                p,
                q,
                3,
                density,
                coordinates=point if transformed else None,
            )
            for actual, other in zip(expected[:3], independent):
                np.testing.assert_allclose(actual, other, rtol=4e-6, atol=2e-7)
            assert np.linalg.norm(expected[2][: model.columns, model.columns :]) > 1e-4


@pytest.mark.parametrize("p,q,har", [(1, 1, False), (3, 2, True), (4, 3, False)])
@pytest.mark.parametrize("density", [0, 1, 2, 3])
def test_joint_aparch_large_power_hessian_matches_rescaled_ad(p, q, har, density):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import jax.scipy as jsp
    jax.config.update("jax_enable_x64", True)
    from scivol._devtools.ad_oracle import _jax_aparch_power_moment, _native_shape_coordinates
    from scivol._devtools.joint_mean_oracle import _density_nll, NAMES

    model, x = prepared(3, density, p, q, har=har, n=9)
    runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    theta = model.initialize()
    theta[:model.columns] = .025
    theta[model.columns] = model.h0 = runtime.h0 = 1e200
    theta[model.shape_start - 1] = 2.
    m, beta_start = model.columns, model.columns + 1 + 2*p
    theta[m+1+p:beta_start] = np.linspace(-.15, .2, p)
    units = np.ones(theta.size)
    units[m] = model.h0

    def objective(scaled):
        # Differentiate P / h0 and omega / h0, then undo only that linear scaling.
        # Forming the raw AD Hessian first would recreate the kernel's overflow.
        residuals = jnp.asarray(model.data) - jnp.asarray(model.features) @ scaled[:m]
        delta = scaled[model.shape_start-1]
        log_scale = jnp.log(model.h0)
        powers = [jnp.exp((delta/2 - 1)*log_scale)]
        logs = [log_scale]
        for t in range(1, len(model.data)):
            power = scaled[m]
            for lag in range(min(t, p)):
                e = residuals[t-1-lag]
                shock = jnp.abs(e) - scaled[m+1+p+lag]*e
                power += scaled[m+1+lag] * jnp.exp(delta*jnp.log(shock)-log_scale)
            for lag in range(min(t, q)):
                power += scaled[beta_start+lag]*powers[t-1-lag]
            powers.append(power)
            logs.append((2/delta)*(jnp.log(power)+log_scale))
        log_variance = jnp.stack(logs)
        z = residuals*jnp.exp(-.5*log_variance)
        return .5*jnp.sum(log_variance)+_density_nll(
            jnp,jsp,z,scaled[model.shape_start:],density,np.zeros(len(model.data),dtype=bool))

    point = jnp.asarray(theta / units)
    expected = (float(objective(point)),
        np.asarray(jax.grad(objective)(point)) / units,
        np.asarray(jax.hessian(objective)(point)) / units[:, None] / units[None, :])
    assert all(np.isfinite(part).all() for part in map(np.asarray, expected))
    assert np.max(np.abs(expected[2][m:, m:])) > 1.
    assert np.any(expected[2][:m, m:] != 0.)

    def coordinate_map(z):
        shape = _native_shape_coordinates(jax,jnp,z[model.shape_start:],NAMES[density])
        gamma = jnp.tanh(z[m+1+p:beta_start])
        delta = shape[0]*jax.nn.sigmoid(z[model.shape_start-1]) if density in (1,2) \
            else jax.nn.softplus(z[model.shape_start-1])
        moment = jnp.stack([_jax_aparch_power_moment(jnp,jsp,gamma[i],delta,shape,NAMES[density])
                            for i in range(p)])
        logits = jnp.r_[z[m+1:m+1+p], z[beta_start:beta_start+q], 0.]
        contributions = (1-8*(p+q+1)*np.finfo(float).eps)*jax.nn.softmax(logits)[:-1]
        return jnp.r_[z[:m], jax.nn.softplus(z[m]), contributions[:p]/moment,
                      gamma, contributions[p:], delta, shape]

    z = model.inverse(theta)
    jacobian = np.asarray(jax.jacfwd(coordinate_map)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(
        lambda z: jnp.dot(coordinate_map(z), jnp.asarray(expected[1])))(jnp.asarray(z)))
    transformed_expected = (expected[0], jacobian.T @ expected[1],
                            jacobian.T @ expected[2] @ jacobian + curvature)
    # Bound pullback rounding by the terms being contracted, including entries
    # where the Hessian and map-curvature contributions nearly cancel.
    pullback_roundoff = 64*np.finfo(float).eps*(
        np.abs(jacobian).T @ np.abs(expected[2]) @ np.abs(jacobian) + np.abs(curvature))
    if density == 3:
        # E_GED[(|Z|-gamma*Z)^2] = 1+gamma^2, independent of nu.
        # The generic Gamma formula cancels its shape score internally. Bound
        # that cancellation before multiplying by the tiny alpha likelihood
        # score, not by a fixed absolute tolerance for the whole Hessian.
        for lag in range(p):
            gamma_index, alpha_index = m+1+p+lag, m+1+lag
            bound = 64*np.finfo(float).eps*abs(expected[1][alpha_index]*theta[alpha_index])
            pullback_roundoff[gamma_index,model.shape_start] += bound
            pullback_roundoff[model.shape_start,gamma_index] += bound
    for transformed, point, reference in ((False, theta, expected), (True, z, transformed_expected)):
        for engine in (model, runtime):
            for mask in range(1, 8):
                actual = engine.evaluate(point, mask, transformed=transformed)
                for bit, output, target in zip((1, 2, 4), actual[:3], reference):
                    if mask & bit:
                        assert np.isfinite(output).all()
                        if transformed and bit == 4:
                            assert np.all(np.abs(output-target) <=
                                3e-10*np.abs(target)+pullback_roundoff)
                        else:
                            np.testing.assert_allclose(output, target,
                                rtol=0 if bit == 1 else 3e-10, atol=2e-11 if bit == 1 else 0)
                    else:
                        assert output is None




@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("columns", [1, 4])
def test_aparch_conditioned_normal_omits_only_inactive_residual_curvature(fixed, active, columns):
    from scivol import _native_core

    plan, domain = (3, 0, columns, 1, 1, 1, fixed), (0, 0, 0, 0)
    theta = np.r_[np.zeros(columns), .08, .07, .15, .82, 1.7]
    data, features = np.zeros(1), np.full((1, columns), float(active))
    for mask in range(1, 8):
        size = _native_core._linear_joint_sizes(plan, mask, False, domain)[0]
        workspace = np.empty(max(size, 1), dtype=np.uint8)
        value, gradient, hessian = (
            np.full(1, 71.), np.full(theta.size, 71.), np.full((theta.size, theta.size), 71.))
        valid = _native_core._linear_joint_jet(
            plan, theta.ctypes.data, data.ctypes.data, features.ctypes.data,
            1e-310, mask, False, workspace.ctypes.data, workspace.nbytes,
            value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data, 0, 0)
        # Active mean curvature is 1/h0 and genuinely unrepresentable. A zero
        # linear design row makes both residual derivative orders exactly zero.
        assert bool(valid) == (not active or not mask & 4)
        for bit, output in zip((1, 2, 4), (value, gradient, hessian)):
            if not mask & bit:
                assert np.all(output == 71.)
            elif valid:
                expected = .5*(np.log(2*np.pi) + np.log(1e-310)) if bit == 1 else 0.
                np.testing.assert_allclose(output, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("p,q", [(1,1),(4,3)])
def test_joint_normal_egarch_extreme_log_variance_matches_log_state_ad(p,q):
    from scivol._devtools.ad_oracle import _jax_modules, _native_reflection_coefficients

    jax,jnp,_ = _jax_modules()
    data = np.r_[1.,np.full(7,np.exp(-200.))]
    spec = ARX(0)+FAMILIES[2](p,q)+Normal()
    engines = [JointMeanModel(spec,data,None,fixed=fixed) for fixed in (True,False)]
    model = engines[0]
    theta = np.r_[0.,-400.,np.zeros(2*p+q)]

    def objective(point):
        residuals = jnp.asarray(model.data)-point[0]
        logs = [jnp.log(model.h0)]
        for t in range(1,len(model.data)):
            state = point[1]
            for lag in range(min(t,p)):
                standardized = residuals[t-1-lag]*jnp.exp(-.5*logs[t-1-lag])
                state += point[2+lag]*(jnp.abs(standardized)-jnp.sqrt(2/jnp.pi))
                state += point[2+p+lag]*standardized
            for lag in range(min(t,q)):
                state += point[2+2*p+lag]*logs[t-1-lag]
            logs.append(state)
        logs = jnp.stack(logs)
        z = residuals*jnp.exp(-.5*logs)
        return .5*jnp.sum(jnp.log(2*jnp.pi)+logs+z*z)

    point = jnp.asarray(theta)
    expected = float(objective(point)),np.asarray(jax.grad(objective)(point)),np.asarray(jax.hessian(objective)(point))
    assert all(np.isfinite(part).all() for part in map(np.asarray,expected))
    beta_start = 2+2*p
    transformed_objective = lambda z: objective(jnp.r_[
        z[:beta_start],_native_reflection_coefficients(jnp,z[beta_start:],1.)])
    z = model.inverse(theta)
    transformed_expected = (float(transformed_objective(jnp.asarray(z))),
        np.asarray(jax.grad(transformed_objective)(jnp.asarray(z))),
        np.asarray(jax.hessian(transformed_objective)(jnp.asarray(z))))
    for transformed,point,references in ((False,theta,expected),(True,z,transformed_expected)):
        for engine in engines:
            for mask in range(1,8):
                actual = engine.evaluate(point,mask,transformed=transformed)
                for bit,value,reference in zip((1,2,4),actual[:3],references):
                    if mask & bit:
                        np.testing.assert_allclose(value,reference,rtol=2e-11,atol=2e-11)
                    else:
                        assert value is None


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("log_variance", [-800., 800.])
@pytest.mark.parametrize("density,shape", [(0, []), (3, [2.])])
def test_egarch_density_rescaling_preserves_variance_representability(fixed, log_variance, density, shape):
    model = JointMeanModel(ARX(0)+FAMILIES[2]()+DENSITIES[density](), np.array([1.,0.]), fixed=fixed)
    theta = np.r_[0.,log_variance,0.,0.,0.,shape]
    # A local log-density value can be finite even though the declared variance
    # state is not representable. Range recovery must not enlarge that domain.
    for transformed in (False,True):
        point = model.inverse(theta) if transformed else theta
        for mask in range(1,8):
            with pytest.raises(ValueError,match="invalid"):
                model.evaluate(point,mask,transformed=transformed)
        with pytest.raises(ValueError,match="invalid"):
            model.evaluate(point,1,transformed=transformed,paths=True)


@pytest.mark.parametrize("density,shape", [(0,[]),(1,[7.]),(2,[7.,0.]),(2,[7.,.2]),(3,[1.5])])
def test_extreme_egarch_log_curvature_does_not_cancel_to_zero(density,shape):
    from scivol._devtools.ad_oracle import _jax_modules
    from scivol._devtools.joint_mean_oracle import _density_nll

    jax,jnp,jsp = _jax_modules()
    data = np.array([1.,1.,2.])
    spec = ARX(0)+FAMILIES[2]()+DENSITIES[density]()
    model = JointMeanModel(spec,data)
    theta = np.r_[0.,460.,0.,0.,0.,shape]
    def objective(omega):
        logs = jnp.array([jnp.log(model.h0),omega,omega])
        z = jnp.asarray(data)*jnp.exp(-.5*logs)
        return .5*jnp.sum(logs)+_density_nll(jnp,jsp,z,jnp.asarray(shape),density,np.zeros(3,bool))
    expected = (float(objective(460.)),float(jax.grad(objective)(460.)),
                float(jax.hessian(objective)(460.)))
    assert np.isfinite(expected).all() and expected[2] != 0.
    for fixed in (False,True):
        engine = JointMeanModel(spec,data,fixed=fixed)
        for mask in range(1,8):
            f,g,h = engine.evaluate(theta,mask)[:3]
            for bit,value,reference in zip((1,2,4),(f,None if g is None else g[1],
                None if h is None else h[1,1]),expected):
                if mask & bit:
                    np.testing.assert_allclose(value,reference,rtol=3e-12,atol=0.)
                else:
                    assert value is None


def test_egarch_range_repair_preserves_retained_terminal_likelihood():
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint_egarch

    data,x = comparison_sample(2785,True,"normal")
    model = JointMeanModel(HARX([1,3])+FAMILIES[2](3,2)+Normal(),data,x)
    reference = np.array([
        .005718546378212949,-.10925940085134228,.06055139746979874,-.14869877075375482,
        .03663757468683274,-.10704355319385181,-.011246026486717582,-.0981873761559446,
        -.4509200767407483,.5278804982084012,-.15014199533913203,-.1891464497360185,
        .3019533571732595,-.13335054945423144,1.5433174433082564,-.948322409962865])
    target,reference_gradient = model.evaluate(reference,3)[:2]
    assert model.assess(reference,reference_gradient)[0]
    fitted = solve_joint_egarch(model,model.initialize(),transformed=False,
        accuracy=1e-12,maxiter=1800,allow_chart_refinement=True)
    value,gradient = model.evaluate(fitted.x,3)[:2]
    assert value <= target+1e-9
    assert model.assess(fitted.x,gradient)[0]


@pytest.mark.parametrize("orders", [(1, 1), (4, 3)])
def test_linear_aparch_underflowing_shock_matches_full_ad(orders):
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    rng = np.random.default_rng(621)
    data, features = rng.normal(size=9), rng.normal(size=(9, 3))
    data[0] = 1e-180
    spec = ARX(0) + FAMILIES[3](*orders) + Normal()
    engines = [JointMeanModel(spec, data, features, fixed=fixed) for fixed in (True, False)]
    model = engines[0]
    theta = model.initialize()
    theta[:model.columns] = 0.
    theta[-1] = 2.
    for transformed in (False, True):
        point = model.inverse(theta) if transformed else theta
        expected = joint_value_grad_hess(
            theta, model.data, model.features, model.h0, *orders, 3, 0,
            coordinates=point if transformed else None)
        assert all(np.isfinite(part).all() for part in map(np.asarray, expected))
        for engine in engines:
            for mask in range(1, 8):
                actual = engine.evaluate(point, mask, transformed=transformed)
                for bit, value, reference in zip((1, 2, 4), actual[:3], expected):
                    if mask & bit:
                        np.testing.assert_allclose(value, reference, rtol=3e-10, atol=2e-10)
                    else:
                        assert value is None


@pytest.mark.parametrize("density", [1, 3])
@pytest.mark.parametrize("power", [0.3, 1.3])
@pytest.mark.parametrize("orders", [(1, 1), (4, 3)])
def test_joint_aparch_cancelled_residual_matches_exact_mean_ad(density, power, orders):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    rng = np.random.default_rng(42)
    data, x = rng.normal(size=9), rng.normal(size=(9, 3))
    data[0], x[0] = 0., [1., 1., 0.]
    spec = ARX(0) + FAMILIES[3](*orders) + DENSITIES[density]()
    model = JointMeanModel(spec, data, x)
    runtime = JointMeanModel(spec, data, x, fixed=False)
    theta = model.initialize()
    # The exact first residual is -2**-60, although a binary64 dot can erase it.
    theta[:model.columns] = [1., -1., 2.**-60, .05]
    theta[model.shape_start-1] = power
    reference = joint_value_grad_hess(theta, model.data, model.features, model.h0,
        *orders, 3, density, exact_mean_anchor=True)
    assert all(np.isfinite(value).all() for value in map(np.asarray, reference))
    for engine in (model, runtime):
        for mask in range(1, 8):
            actual = engine.evaluate(theta, mask)
            for bit, value, expected in zip((1, 2, 4), actual[:3], reference):
                if mask & bit:
                    np.testing.assert_allclose(value, expected, rtol=2e-7, atol=2e-8)
                else:
                    assert value is None


@pytest.mark.parametrize("conflict", ["coordinates", "kink_limit"])
def test_exact_mean_oracle_anchor_does_not_change_the_evaluation_point(conflict):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    model, _ = prepared(2)
    theta = model.initialize()
    option = {conflict: model.inverse(theta) if conflict == "coordinates" else (0, 1)}
    with pytest.raises(ValueError, match="theta space without a kink limit"):
        joint_value_grad_hess(theta, model.data, model.features, model.h0,
            1, 1, 2, 0, exact_mean_anchor=True, **option)


@pytest.fixture
def retained_aparch_ged_basin():
    from benchmark_native_joint import comparison_sample

    data, x = comparison_sample(2785, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3](1, 1) + DENSITIES[3](), data, x)
    theta = np.array([
        .0367341514620848, -.044255002309452424, -.028617542198394314,
        .10405829363968366, .0906379466348576, 6.621263990114937e-8,
        .8784046556880225, .7632811923074256, 10.808336299039802, 1.643526465689418,
    ])
    # Independent scalar recurrences at 40 and 60 decimal digits agree here.
    return model, theta, 536.8635178817824


def test_retained_aparch_ged_basin_preserves_independent_value(retained_aparch_ged_basin):
    model, theta, expected = retained_aparch_ged_basin
    value, gradient = model.evaluate(theta, 3)[:2]
    feasible, stationarity, violation = model.assess(theta, gradient)
    assert abs(value-expected) <= 1e-9
    assert feasible and violation == 0 and stationarity <= 1e-6


@pytest.mark.parametrize("transformed", [False, True])
def test_curvature_scaled_route_reaches_retained_aparch_ged_basin(
    retained_aparch_ged_basin, transformed,
):
    from scivol._joint_mean_fit import solve_joint

    model, _, expected = retained_aparch_ged_basin
    initial = model.initialize()
    unchanged = initial.copy()
    fitted = solve_joint(model, initial, transformed=transformed, accuracy=1e-12,
        maxiter=1000, initial_curvature=True, allow_chart_refinement=True)
    value, gradient = model.evaluate(fitted.x, 3)[:2]
    feasible, stationarity, violation = model.assess(fitted.x, gradient)
    np.testing.assert_array_equal(initial, unchanged)
    assert value <= expected+1e-9
    assert fitted.success and feasible and violation == 0 and stationarity <= 1e-6
    assert fitted.phases[0]["curvature_scaling"]
    assert fitted.nit == sum(phase["iterations"] for phase in fitted.phases) <= 1000


@pytest.mark.parametrize("density", [2, 4, 5, 6, 7])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_sublinear_joint_aparch_start_fits_without_hiding_uncertainty(
    density, wide, transformed, capsys
):
    model, x = prepared(3, density, 3 if wide else 1, 2 if wide else 1, har=wide, n=96)
    initial = model.initialize()
    initial[model.shape_start - 1] = 0.7
    unchanged = initial.copy()
    initial_nll = model.evaluate(initial, 1)[0]
    fitted = fit_joint_mean_spec(
        model.spec,
        model.data,
        x=x,
        initial_params=initial,
        log_mode=transformed,
        maxiter=60,
    )
    np.testing.assert_array_equal(initial, unchanged)
    nll = model.evaluate(fitted.params, 1)[0]
    assert np.isfinite(nll) and nll < initial_nll
    assert model.assess(fitted.params, None)[0]
    np.testing.assert_allclose(-fitted.log_likelihood, nll, rtol=0, atol=1e-10)
    phases = fitted.fit_info.phase_diagnostics
    assert sum(phase["iterations"] for phase in phases) <= 60
    assert all(phase["space"] == ("z" if transformed else "theta") for phase in phases)
    if not fitted.converged:
        assert fitted.fit_info.inference_status == "unavailable"
        assert fitted.std_errors is None and fitted.warnings
    assert not capsys.readouterr().out


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("transformed", (False, True))
def test_general_joint_density_and_cross_blocks_match_independent_ad(
    family, density, transformed
):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    model, _ = prepared(family, density, 3, 2, n=9)
    theta = model.initialize()
    m, p, q = model.columns, model.p_vol, model.q_vol
    beta = m + 1 + (1 if family == 0 else 2) * p
    theta[:m], theta[m], theta[m + 1 : m + 1 + p] = 0.025, 0.15, 0.03
    if family:
        theta[m + 1 + p : beta] = 0.12
    theta[beta : beta + q] = 0.2
    if family == 3:
        theta[model.shape_start - 1] = 1.7
    theta[model.shape_start :] = (
        [],
        [7.0],
        [7.0, 0.2],
        [1.7],
        [0.12, 1.8, 0.4],
        [0.12, 1.8],
        [0.12, 1.8],
        [0.12, 2.0],
    )[density]
    point = model.inverse(theta) if transformed else theta
    actual = model.evaluate(point, 7, transformed=transformed)
    expected = joint_value_grad_hess(
        theta,
        model.data,
        model.features,
        model.h0,
        p,
        q,
        family,
        density,
        coordinates=point if transformed else None,
    )
    for value, reference in zip(actual[:3], expected):
        np.testing.assert_allclose(value, reference, rtol=4e-6, atol=2e-7)
    assert np.linalg.norm(actual[2][:m, m:]) > 1e-4


@pytest.mark.parametrize("family", range(4))
def test_public_joint_fit_preserves_full_vector_and_canonical_paths(family):
    model, x = prepared(family, n=250)
    initial = model.initialize()
    result = model.spec.fit(
        model.data, x=x, solver="native", initial_params=initial, maxiter=200
    )
    fitted_model = JointMeanModel(result.spec, result.data, result._fit_x)
    state = filter_joint_mean_spec(
        result.spec, result.data, result.params, x=result._fit_x
    )
    np.testing.assert_allclose(result.sigma2, state.sigma2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.log_likelihood, state.log_likelihood, rtol=1e-12)
    assert len(result.params) == model.size
    assert result.fit_info.engine == "native-joint-linear-jet"
    assert fitted_model.assess(result.params, None)[0]
    assert result.log_likelihood >= -fitted_model.evaluate(initial, 1)[0] - 1e-10


@pytest.mark.parametrize("density", (0, 2, 7))
@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("log_mode", (None, False, True))
def test_public_egarch_native_uses_the_validated_controller(density, har, log_mode):
    from scivol._joint_mean_fit import solve_joint_egarch
    from scivol._validation import prepare_series_input

    model, x = prepared(2, density, p=4 if har else 1, q=3 if har else 1,
                        n=60, har=har)
    initial = model.initialize()
    sample = prepare_series_input(model.spec, model.data, x=x)
    prepared_model = JointMeanModel(model.spec, sample.data, sample.x)
    expected = solve_joint_egarch(prepared_model, initial, accuracy=1e-12, maxiter=12,
        transformed=bool(log_mode), allow_chart_refinement=log_mode is None)
    actual = model.spec.fit(model.data, x=x, initial_params=initial,
                           solver="native", maxiter=12, log_mode=log_mode)
    np.testing.assert_array_equal(actual.params, expected.x)
    assert actual.fit_info.optimizer_controller == "joint-continuation"
    assert actual.fit_info.route_iteration_budgets == expected.route_budgets
    assert sum(phase["iterations"] for phase in actual.fit_info.phase_diagnostics) <= 12
    if log_mode is not None:
        assert {phase["space"] for phase in actual.fit_info.phase_diagnostics} == {
            "z" if log_mode else "theta"}
        assert actual.fit_info.route_iteration_budgets == [12]


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("log_mode", (None, False, True))
def test_promoted_joint_default_matches_explicit_native(family, density, har, log_mode):
    model, x = prepared(
        family, density, p=3 if har else 1, q=2 if har else 1, n=250, har=har
    )
    options = dict(x=x, initial_params=model.initialize(), maxiter=100)
    if log_mode is not None:
        options["log_mode"] = log_mode
    default = model.spec.fit(model.data, **options)
    explicit = model.spec.fit(model.data, solver="native", **options)
    assert default.fit_info.engine == "native-joint-linear-jet"
    assert default.fit_info.used_default_solver
    assert not explicit.fit_info.used_default_solver
    np.testing.assert_array_equal(default.params, explicit.params)
    assert default.log_likelihood == explicit.log_likelihood
    np.testing.assert_array_equal(default.sigma2, explicit.sigma2)
    assert default.fit_info.stationarity_status == explicit.fit_info.stationarity_status
    assert default.fit_info.inference_status == explicit.fit_info.inference_status


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("budget", (None, 1, 23))
def test_public_joint_default_budget_matches_validated_policy(monkeypatch, family, budget):
    import benchmark_native_joint as benchmark
    import scivol._joint_mean_fit as joint

    model, x = prepared(family, n=60)
    name = ("solve_joint", "solve_joint", "solve_joint_egarch", "solve_joint_with_metric_portfolio")[family]
    controller = getattr(joint, name)
    requested = []

    def observe_budget(*args, **options):
        requested.append(options["maxiter"])
        return controller(*args, **{**options, "maxiter": 1})

    monkeypatch.setattr(joint, name, observe_budget)
    options = {} if budget is None else dict(maxiter=budget)
    result = model.spec.fit(model.data, x=x, solver="native", **options)
    expected = budget if budget is not None else (1000, 1000, 1800, 3200)[family]
    assert requested == [expected]
    assert result.fit_info.maxiter == expected
    assert joint.EGARCH_DEFAULT_MAXITER == benchmark.EGARCH_ACCEPTANCE_NATIVE_MAXITER
    assert joint.APARCH_PORTFOLIO_MAXITER == 3200


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("budget", (None, 1, 23, 2000))
@pytest.mark.parametrize("log_mode", (None, False, True))
def test_public_joint_controller_respects_requested_path_and_budget(
    monkeypatch, family, budget, log_mode
):
    from unittest.mock import Mock, sentinel
    import scivol._joint_mean_fit as joint

    model, x = prepared(family, n=60)
    initial = model.initialize()
    unchanged = initial.copy()
    names = ("solve_joint", "solve_joint_egarch", "solve_joint_with_retry", "solve_joint_with_metric_portfolio")
    controllers = {name: Mock(return_value=sentinel.optimized) for name in names}
    for name, controller in controllers.items():
        monkeypatch.setattr(joint, name, controller)
    assembly = Mock(return_value=sentinel.reported)
    monkeypatch.setattr(joint, "assemble_joint_fit", assembly)

    options = {} if budget is None else dict(maxiter=budget)
    result = fit_joint_mean_spec(
        model.spec, model.data, x=x, initial_params=initial,
        log_mode=log_mode, native_accuracy=1e-8, **options
    )
    expected = budget if budget is not None else (1800 if family == 2 else 1000)
    name = ("solve_joint", "solve_joint", "solve_joint_egarch", "solve_joint_with_retry")[family]
    if family == 3 and log_mode is None:
        name = "solve_joint_with_metric_portfolio"
        expected = budget if budget is not None else 3200
    selected = controllers[name]
    selected.assert_called_once()
    fitted_model, start = selected.call_args.args
    assert fitted_model.family == family
    np.testing.assert_array_equal(start, initial)
    np.testing.assert_array_equal(initial, unchanged)
    assert selected.call_args.kwargs == dict(
        transformed=bool(log_mode), accuracy=1e-8, maxiter=expected,
        allow_chart_refinement=log_mode is None
    )
    for other in controllers:
        if other != name:
            controllers[other].assert_not_called()
    assembly.assert_called_once()
    assert assembly.call_args.args[:2] == (fitted_model, sentinel.optimized)
    assert assembly.call_args.args[2:5] == (bool(log_mode), 1e-8, expected)
    assert result is sentinel.reported


@pytest.mark.parametrize("family", range(4))
def test_joint_default_routing_preserves_explicit_controls(family):
    model, _ = prepared(family)
    from scivol._mixins import FitsMixin

    options = dict(
        method="mle",
        has_auto_components=False,
        has_exogenous_data=True,
        force_legacy_engine=False,
        kwargs={},
    )
    route = FitsMixin._should_use_native_engine
    assert route(model.spec, **options)
    for solver in ("legacy", "slsqp"):
        assert not route(model.spec, **{**options, "kwargs": {"solver": solver}})
    assert not route(model.spec, **{**options, "force_legacy_engine": True})
    assert not route(model.spec, **{**options, "method": "qmle"})
    assert not route(model.spec, **{**options, "has_auto_components": True})


@pytest.mark.parametrize("family", (0, 1, 2))
@pytest.mark.parametrize(
    "mean", (ARX(0), ARX(0, constant=False), ARX(1, constant=False), HARX([1, 3]))
)
def test_promoted_joint_with_zero_exogenous_columns(mean, family):
    data = np.random.default_rng(9218).normal(size=120)
    spec = mean + FAMILIES[family](2, 1) + DENSITIES[1]()
    with pytest.raises(ValueError, match="requires exogenous regressors"):
        spec.fit(data, maxiter=100)
    x = np.empty((len(data), 0))
    default = spec.fit(data, x=x, maxiter=100)
    explicit = spec.fit(data, x=x, solver="native", maxiter=100)
    assert default.fit_info.engine == "native-joint-linear-jet"
    assert default.fit_info.used_default_solver
    np.testing.assert_array_equal(default.params, explicit.params)
    assert default.log_likelihood == explicit.log_likelihood


@pytest.mark.parametrize("family", range(4))
def test_short_budget_returns_unchanged_estimate_with_persistent_warning(
    family, capsys
):
    model, x = prepared(family, n=100)
    result = fit_joint_mean_spec(model.spec, model.data, x=x, maxiter=1)
    assert not result.converged
    assert result.fit_info.optimizer_terminated is False
    assert result.fit_info.stationarity_status == "uncertified"
    assert result.fit_info.inference_status == "unavailable"
    assert result.std_errors is None
    assert result.warnings
    assert capsys.readouterr().out == ""
    before = result.params.copy()
    _ = result.warnings, result.warnings, result.fit_info.to_dict()
    np.testing.assert_array_equal(before, result.params)


@pytest.mark.parametrize("family", range(4))
def test_value_only_native_checkpoint_survives_result_assembly(
    family, monkeypatch, capsys
):
    import ctypes
    import time
    from scivol import _native_core
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared(family, n=100)
    initial = model.initialize()
    initial_value, gradient, *_ = model.evaluate(initial, 3)
    direction = model.units * gradient
    retained = initial - 1e-4 * model.units * direction / np.linalg.norm(direction)
    retained_value = model.evaluate(retained, 1)[0]
    assert retained_value < initial_value and model.assess(retained, None)[0]
    evaluate = model.evaluate

    def value_only(theta, request, **kwargs):
        if request != 1 and np.array_equal(theta, retained):
            raise ValueError("Derivative unavailable at the retained test point.")
        return evaluate(theta, request, **kwargs)

    def failed_native(*args):
        theta = np.ctypeslib.as_array(
            ctypes.cast(args[11], ctypes.POINTER(ctypes.c_double)), shape=(model.size,)
        )
        stats = np.ctypeslib.as_array(
            ctypes.cast(args[14], ctypes.POINTER(ctypes.c_int64)), shape=(6,)
        )
        theta[:] = retained
        stats[:] = [-101, 1, 2, 3, 2, 2]
        return False

    monkeypatch.setattr(model, "evaluate", value_only)
    monkeypatch.setattr(_native_core, "_linear_joint_fit", failed_native)
    phase = _solve_joint_phase(
        model,
        initial,
        transformed=False,
        accuracy=1e-12,
        maxiter=1,
        use_curvature=False,
    )
    assert phase.selected == "returned" and phase.jac is None and not phase.success
    assert phase.status == -101 and phase.fun == retained_value
    phase.phases, phase.selected_space = (), "theta"
    phase.curvature_prediction_pending = True
    phase.predicted_coordinate_gain = 123.0
    fitted = assemble_joint_fit(model, phase, False, 1e-12, 1, time.perf_counter())
    assert not phase.curvature_prediction_pending
    assert phase.predicted_coordinate_gain is None
    np.testing.assert_array_equal(fitted.params, retained)
    assert fitted.log_likelihood == -retained_value
    assert fitted.fit_info.optimizer_mode == -101
    assert not fitted.fit_info.optimizer_terminated and not fitted.converged
    assert fitted.fit_info.stationarity_status == "uncertified"
    assert fitted.fit_info.inference_status == "unavailable"
    assert fitted.std_errors is None and fitted.warnings
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kind,mode", [("slsqp", 0), ("native-active-face", 7)])
def test_metadata_keeps_termination_stationarity_and_inference_separate(kind, mode):
    info = FitMetadata(
        optimizer_mode=mode,
        optimizer_status_kind=kind,
        optimizer_terminated=True,
        stationarity_status="uncertified",
        inference_status="unavailable",
    )
    assert FitMetadata(**info.to_dict()).to_dict() == info.to_dict()


@pytest.mark.parametrize("units", (1e-4, 1.0, 1e4))
def test_diagonal_descent_detects_weak_quadratic_direction(units):
    from scivol._fit_credibility import diagonal_descent_prediction

    # A small gradient does not resolve a flat direction: f(x)=1e-4*x*x/2.
    gain = diagonal_descent_prediction(
        np.array([2e-4 * units]),
        np.array([1e-4 * units**2]),
        np.array([-np.inf]),
        np.array([np.inf]),
        np.empty((0, 1)),
        np.empty(0),
    )
    assert gain == pytest.approx(2e-4)


@pytest.mark.parametrize("constraint", ("bound", "linear"))
def test_diagonal_descent_respects_blocked_directions(constraint):
    from scivol._fit_credibility import diagonal_descent_prediction

    gain = diagonal_descent_prediction(
        np.array([2.0]),
        np.array([1.0]),
        np.array([0.0 if constraint == "bound" else -np.inf]),
        np.array([np.inf]),
        np.array([[1.0]]) if constraint == "linear" else np.empty((0, 1)),
        np.array([0.0]) if constraint == "linear" else np.empty(0),
    )
    assert gain == 0.0


@pytest.mark.parametrize("units", (1e-4, 1.0, 1e4))
def test_flat_inactive_coordinate_preserves_other_descent_predictions(units):
    from scivol._fit_credibility import diagonal_descent_prediction

    gain = diagonal_descent_prediction(
        np.array([0., 2e-4 * units]),
        np.array([0., 1e-4 * units**2]),
        np.array([-1., -np.inf]),
        np.array([1., np.inf]),
        np.empty((0, 2)),
        np.empty(0),
    )
    assert gain == pytest.approx(2e-4)
    assert diagonal_descent_prediction(
        np.zeros(2), np.zeros(2), np.full(2, -np.inf), np.full(2, np.inf),
        np.empty((0, 2)), np.empty(0),
    ) == 0.


def test_inactive_aparch_asymmetry_does_not_hide_a_remaining_descent():
    from benchmark_native_joint import comparison_sample
    from scivol import HARX, APARCH, Hyperbolic

    data, x = comparison_sample(2785, True, "normal")
    model = JointMeanModel(HARX([1, 3]) + APARCH(3, 2) + Hyperbolic(), data, x)
    theta = np.array([
        .007510864259074027, -.09172151319601814, .0961494541676002,
        .02659495457070226, .016843227319356745, -.07168455906367201,
        .06276873533910371, .41603081475573594, 7.0738148130150765e-6,
        0., 5.7099044755691174e-5, .9967751892768058, .0028805287239767646,
        -.961670800375402, 0., .14622685876563998, 7.890011665155125,
        -.2827106908905683, 6.252385982595102,
    ])
    _, gradient, hessian = model.evaluate(theta, 7)[:3]
    assert gradient[12] == hessian[12, 12] == 0.
    assert model.assess(theta, gradient)[0]
    gain = model.predicted_coordinate_gain(theta, gradient, hessian)
    assert gain is not None and gain > 1e-9


@pytest.mark.parametrize("curvature", (0.0, np.nan, np.inf, 1e-320))
def test_unresolved_descent_diagnostic_does_not_claim_precision(curvature):
    from scivol._fit_credibility import diagonal_descent_prediction

    assert (
        diagonal_descent_prediction(
            np.array([1.0]),
            np.array([curvature]),
            np.array([-np.inf]),
            np.array([np.inf]),
            np.empty((0, 1)),
            np.empty(0),
        )
        is None
    )


@pytest.mark.parametrize("refinement_available", (False, True))
def test_joint_refines_weak_density_direction_after_kkt_success(
    monkeypatch, refinement_available
):
    from scivol import ARX, GJRGARCH, StudentT
    from scivol._joint_mean_fit import solve_joint
    from scivol._kernels import get_routine
    from scivol._evaluation import _filter_meanx_gjr

    rng = np.random.default_rng(2786)
    y, x = rng.normal(size=400), rng.normal(size=(400, 2))
    spec = ARX(1) + GJRGARCH(1, 1) + StudentT()
    model = JointMeanModel(spec, y, x)
    # Recorded September 9 weak-nu stop, independent of the current solver path.
    initial = np.array([
        -0.06203138822102747, -0.024440873742920647, 0.04556622634817581,
        0.0222456520772501, 0.07727063837921005, 1.7957636079233487e-17,
        3.3372571108270803e-18, 0.920005882783876, 49.05828237808678,
    ])
    value, gradient, hessian, *_ = model.evaluate(initial, 7)
    feasible, residual, violation = model.assess(initial, gradient)
    assert feasible and violation == 0 and residual < 1e-6
    assert value == pytest.approx(561.6282375820467, abs=1e-9, rel=0)
    assert model.predicted_coordinate_gain(initial, gradient, hessian) > 1e-4
    if not refinement_available:
        from scivol import _joint_mean_fit as joint

        original = joint._solve_joint_phase
        retained = []

        def phase(*args, **kwargs):
            if retained:
                raise ValueError("Refinement unavailable in this regression control.")
            candidate = original(*args, **kwargs)
            retained.append(candidate.x.copy())
            return candidate

        monkeypatch.setattr(joint, "_solve_joint_phase", phase)
        result = fit_joint_mean_spec(spec, y, x=x, initial_params=initial)
        np.testing.assert_array_equal(result.params, retained[0])
        assert result.fit_info.optimizer_terminated
        assert result.fit_info.stationarity_status == "certified"
        assert any(
            "predicts further likelihood improvement" in warning
            for warning in result.warnings
        )
        return
    native = solve_joint(
        model,
        initial,
        transformed=False,
        accuracy=1e-12,
        maxiter=1000,
        allow_chart_refinement=True,
    )
    first = native.phases[0]
    assert first["mode"] == 0
    assert first["stationarity_bound"] < 1e-6
    assert first["predicted_coordinate_gain"] > 1e-4
    assert native.fun < first["nll"] - 2e-4
    assert model.assess(native.x, model.evaluate(native.x, 2)[1])[1] < 1e-6
    reference = get_routine(str(spec)).fit(
        y,
        x=x,
        initial_params=initial,
        solver="slsqp",
        log_mode=False,
        maxiter=1000,
    )
    # Retain all four scores: the old objective must agree on the improvement.
    scores = []
    for theta in (native.x, reference.params):
        canonical = model.evaluate(theta, 1)[0]
        historical = -_filter_meanx_gjr(
            spec,
            y,
            theta,
            x=x,
            compute_score=False,
            compute_hessian=False,
        ).log_likelihood
        assert canonical == pytest.approx(historical, abs=1e-8, rel=0)
        scores.append((canonical, historical))
    assert scores[0][0] < scores[1][0]
    assert scores[0][1] < scores[1][1]


def test_joint_gjr_student_default_start_preserves_verified_solution_quality():
    from scivol import GJRGARCH, StudentT
    from scivol._kernels import get_routine
    from scivol._evaluation import _filter_meanx_gjr

    rng = np.random.default_rng(2786)
    y, x = rng.normal(size=400), rng.normal(size=(400, 2))
    spec = ARX(1) + GJRGARCH() + StudentT()
    model = JointMeanModel(spec, y, x)
    initial = model.initialize()
    native = fit_joint_mean_spec(spec, y, x=x, initial_params=initial, maxiter=1000)
    reference = get_routine(str(spec)).fit(
        y, x=x, initial_params=initial, solver="slsqp", log_mode=False, maxiter=1000,
    )
    assert native.converged
    assert native.fit_info.stationarity_status == "certified"
    assert native.fit_info.jet_consistency["status"] == "consistent"
    assert native.fit_info.maximum_constraint_violation == 0
    assert -native.log_likelihood <= 561.6280335256474 + 1e-9
    scores = []
    for theta in (native.params, reference.params):
        canonical = model.evaluate(theta, 1)[0]
        historical = -_filter_meanx_gjr(
            spec, y, theta, x=x, compute_score=False, compute_hessian=False,
        ).log_likelihood
        assert canonical == pytest.approx(historical, abs=1e-9, rel=0)
        scores.append((canonical, historical))
    assert scores[0][0] < scores[1][0]
    assert scores[0][1] < scores[1][1]


@pytest.mark.parametrize("reject_at", (0, 3, None))
def test_unavailable_chart_does_not_consume_a_reserved_optimizer_phase(
    monkeypatch, reject_at
):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model, _ = prepared()
    initial = model.initialize()
    initial_value = model.evaluate(initial, 1)[0]
    calls, completed = [], []

    def phase(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["transformed"] and (reject_at is None or len(completed) == reject_at):
            raise ValueError("The joint start is outside the transformed chart.")
        completed.append(kwargs)
        return OptimizeResult(
            x=initial.copy(),
            fun=initial_value - len(completed),
            jac=np.ones(model.size),
            status=9,
            success=False,
            nit=kwargs["maxiter"],
            curvature_prepared=kwargs["use_curvature"],
            kernel_dispatch=2,
        )

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(model, "assess", lambda *args: (True, 1.0, 0.0))
    result = joint.solve_joint(
        model,
        initial,
        transformed=reject_at == 0,
        allow_chart_refinement=True,
        accuracy=1e-12,
        maxiter=80,
    )
    assert len(completed) == 4
    assert [p["maxiter"] for p in completed] == [40, 20, 10, 10]
    assert result.nit == 80
    assert result.fun == initial_value - 4
    failed = [p for p in result.phases if p["mode"] is None]
    assert len(failed) == (3 if reject_at is None else 1)
    assert all(p["iterations"] == 0 and p["space"] == "z" for p in failed)
    assert len(calls) == 4 + len(failed)
    if reject_at == 3:
        assert calls[-2]["transformed"] and not calls[-1]["transformed"]
        assert calls[-2]["maxiter"] == calls[-1]["maxiter"] == 10
        assert result.phases[-1]["space"] == "theta"


def test_unavailable_refinement_chart_retains_the_existing_fit(monkeypatch):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model, _ = prepared()
    initial = model.initialize()
    value = model.evaluate(initial, 1)[0]
    calls = []

    def phase(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise ValueError("The joint start is outside the transformed chart.")
        return OptimizeResult(
            x=initial.copy(),
            fun=value,
            status=9,
            success=False,
            nit=2,
            nfev=3,
            njev=2,
            curvature_prepared=False,
            kernel_dispatch=2,
        )

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    result = joint.solve_joint(
        model,
        initial,
        transformed=True,
        accuracy=1e-12,
        maxiter=100,
    )
    np.testing.assert_array_equal(result.x, initial)
    assert result.fun == value
    assert result.nit == 2
    assert calls[0]["accuracy"] == 1e-12
    assert calls[1]["accuracy"] == np.finfo(float).eps
    assert calls[0]["maxiter"] == 50
    assert calls[1]["maxiter"] == 49
    assert not result.success
    assert "outside the transformed chart" in result.phases[-1]["error"]

    import time

    fitted = joint.assemble_joint_fit(
        model, result, True, 1e-12, 100, time.perf_counter()
    )
    assert fitted.fit_info.optimizer_phase_modes == (9,)
    assert len(fitted.fit_info.phase_diagnostics) == 2
    assert fitted.fit_info.log_mode is True
    assert fitted.warnings


@pytest.mark.parametrize("outcome", ("better", "worse", "equal", "failed", "credible", "precision"))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("speculate", (False, True))
def test_two_route_controller_preserves_start_budget_and_best_point(outcome, transformed, speculate):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import solve_joint_two_routes

    model = SimpleNamespace(data=np.zeros(10), request_counts=np.zeros(8, dtype=int))
    initial, calls = np.array([1.0, 2.0]), []

    def controller(model, start, **options):
        calls.append((start.copy(), options))
        model.request_counts[1] += 4
        model.request_counts[2] += 2
        second = len(calls) == 2
        if second and outcome == "failed":
            raise ValueError("unavailable alternate chart")
        start[0] += len(calls)
        value = ({"better": 90.0, "worse": 110.0, "equal": 100.0,
                  "precision": 90.0}[outcome] if second else 100.0)
        bound = 0.5e-6 if not second and outcome in ("credible", "precision") else 0.5 if second else 1.0
        phase = dict(iterations=options["maxiter"], function_evaluations=4,
                     gradient_evaluations=2, hessian_evaluations=0)
        return OptimizeResult(x=start, fun=value, success=True, stationarity_bound=bound,
            predicted_coordinate_gain=1.0 if outcome == "precision" else None,
            nit=options["maxiter"], nfev=4, njev=2, nhev=0, phases=(phase,))

    result = solve_joint_two_routes(model, initial, accuracy=1e-12, maxiter=5,
        transformed=transformed, controller=controller, speculate=speculate)
    np.testing.assert_array_equal(initial, [1.0, 2.0])
    assert len(calls) == (1 if outcome == "credible" else 2)
    for start, options in calls:
        np.testing.assert_array_equal(start, initial)
        assert not options["allow_chart_refinement"] and options["speculate"] == speculate
    assert calls[0][1]["maxiter"] == 2
    assert calls[0][1]["transformed"] == transformed
    if len(calls) == 2:
        assert calls[1][1]["maxiter"] == 3
        assert calls[1][1]["transformed"] != transformed
        assert calls[1][1]["initial_curvature"]
    assert result.route_budgets == ([2] if outcome == "credible" else [2, 3])
    assert result.total_iteration_limit == 5
    assert result.nit == (2 if outcome in ("credible", "failed") else 5)
    assert result.nfev == 4*len(calls) and result.njev == 2*len(calls)
    assert result.nhev == 0
    expected_x = [3.0, 2.0] if outcome in ("better", "equal", "precision") else [2.0, 2.0]
    np.testing.assert_array_equal(result.x, expected_x)
    assert result.fun <= 100.0
    assert result.phases[0]["purpose"] == "primary"
    if outcome == "failed":
        failed = result.phases[-1]
        assert failed["mode"] is None and failed["iterations"] == 0
        assert failed["space"] == ("theta" if transformed else "z")
        assert failed["jet_requests"]["masks"]["F"] == 4
        assert failed["error"] == "unavailable alternate chart"


@pytest.mark.parametrize("maxiter,allow_chart_refinement", ((1, True), (1, False), (5, False)))
def test_two_route_controller_honors_single_route_requests(maxiter, allow_chart_refinement):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import solve_joint_two_routes

    calls = []

    def controller(model, start, **options):
        calls.append(options)
        return OptimizeResult(x=start, fun=1.0, success=False, stationarity_bound=1.0,
                              phases=(), nit=maxiter, nfev=1, njev=1, nhev=0)

    result = solve_joint_two_routes(SimpleNamespace(data=np.zeros(10)), np.ones(2),
        accuracy=1e-12, maxiter=maxiter, transformed=True,
        allow_chart_refinement=allow_chart_refinement, controller=controller)
    assert len(calls) == 1 and calls[0]["maxiter"] == maxiter
    assert result.route_budgets == [maxiter] and result.nit == maxiter


@pytest.mark.parametrize("outcome", ("better", "worse", "equal", "failed"))
@pytest.mark.parametrize("first_mode", (0, 9, -101))
@pytest.mark.parametrize("transformed", (False, True))
def test_same_start_retry_keeps_primary_work_and_accounts_for_every_route(
    monkeypatch, outcome, first_mode, transformed
):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    initial, calls = np.array([1.0, 2.0]), []

    def controller(model, start, **options):
        calls.append((start.copy(), options))
        model.request_counts[1] += 4
        model.request_counts[3] += 2
        model.request_counts[4] += 1
        second = len(calls) == 2
        if second and outcome == "failed":
            raise ValueError("unavailable retry chart")
        start[0] += len(calls)
        value = {"better": 90.0, "worse": 110.0, "equal": 100.0}.get(outcome) if second else 100.0
        iterations = options["maxiter"] if second else 3
        bound = 0.5 if second else 1.0
        phase = dict(mode=first_mode, stationarity_bound=bound, iterations=iterations,
                     function_evaluations=6, gradient_evaluations=2, hessian_evaluations=1)
        return OptimizeResult(x=start, fun=value, success=True, status=0,
            stationarity_bound=bound, nit=iterations, nfev=6, njev=2, nhev=1,
            phases=(phase,), selected_space="z" if options["transformed"] else "theta")

    monkeypatch.setattr(joint, "solve_joint", controller)
    result = joint.solve_joint_with_retry(model, initial, accuracy=1e-12, maxiter=11,
        primary_maxiter=7, transformed=transformed)
    np.testing.assert_array_equal(initial, [1.0, 2.0])
    assert len(calls) == 2
    for start, options in calls:
        np.testing.assert_array_equal(start, initial)
        assert options["allow_chart_refinement"]
    assert calls[0][1]["maxiter"] == 7
    assert calls[1][1]["maxiter"] == 8
    assert calls[0][1]["transformed"] == transformed
    assert calls[1][1]["transformed"] == (not transformed)
    assert calls[1][1]["initial_curvature"]
    assert calls[1][1]["speculate"] == (first_mode == 0)
    assert result.route_budgets == [7, 8]
    assert result.total_iteration_limit == 11
    assert result.nit == (3 if outcome == "failed" else 11)
    assert (result.nfev, result.njev, result.nhev) == (12, 4, 2)
    assert {key: result.jet_requests[key] for key in ("F", "G", "H")} == dict(F=12, G=4, H=2)
    expected_x = [3.0, 2.0] if outcome in ("better", "equal") else [2.0, 2.0]
    np.testing.assert_array_equal(result.x, expected_x)
    assert result.fun <= 100.0
    assert result.optimizer_controller == "joint-same-start-retry"
    assert result.phases[-1]["purpose"] == "same-start-metric-retry"
    if outcome == "failed":
        assert result.phases[-1]["error"] == "unavailable retry chart"
        assert result.phases[-1]["iterations"] == 0
        assert not result.phases[-1]["refinement_phases"]
    else:
        assert result.phases[-1]["refinement_phases"][0]["iterations"] == 8


@pytest.mark.parametrize("transformed", (False, True))
def test_retry_metric_unavailable_curvature_preserves_unscaled_phase(monkeypatch, transformed):
    model, _ = prepared(3, 2)
    initial = model.initialize()
    expected = _solve_joint_phase(model, initial.copy(), transformed=transformed,
        accuracy=1e-12, maxiter=20, use_curvature=False, speculate=False)
    evaluate = model.evaluate
    requested = []

    def unavailable_curvature(point, request, **kwargs):
        if request == 4:
            requested.append((point.copy(), kwargs["transformed"]))
            model.request_counts[4] += 1
            raise ValueError("The requested Hessian is unavailable")
        return evaluate(point, request, **kwargs)

    monkeypatch.setattr(model, "evaluate", unavailable_curvature)
    actual = _solve_joint_phase(model, initial.copy(), transformed=transformed,
        accuracy=1e-12, maxiter=20, use_curvature=True, speculate=False)
    assert len(requested) == 1 and requested[0][1] == transformed
    np.testing.assert_array_equal(requested[0][0], model.inverse(initial) if transformed else initial)
    np.testing.assert_array_equal(actual.x, expected.x)
    assert actual.fun == expected.fun and actual.status == expected.status
    assert actual.nit == expected.nit and not actual.curvature_prepared
    for request in ("F", "G"):
        assert actual.jet_requests[request] == expected.jet_requests[request]
    assert actual.jet_requests["H"] == expected.jet_requests["H"] + 1


@pytest.mark.parametrize("reason", ("first-credible", "kink", "exhausted", "explicit"))
@pytest.mark.parametrize("transformed", (False, True))
def test_same_start_retry_skips_unnecessary_or_disallowed_work(monkeypatch, reason, transformed):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    calls = []

    def controller(model, start, **options):
        calls.append(options)
        first_bound = 0.5e-4 if reason == "first-credible" else 1.0
        # The trigger follows caller accuracy, not a fixed 1e-6 threshold.
        return OptimizeResult(x=start.copy(), fun=1.0, success=True,
            stationarity_bound=1.0, nearby_kink_evidence={"supports_kink_stop": reason == "kink"},
            phases=(dict(mode=0, stationarity_bound=first_bound),),
            nit=10 if reason == "exhausted" else 2, nfev=0, njev=0, nhev=0)

    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    monkeypatch.setattr(joint, "solve_joint", controller)
    result = joint.solve_joint_with_retry(model, np.ones(2), accuracy=1e-8, maxiter=10,
        primary_maxiter=10 if reason == "exhausted" else 7, transformed=transformed,
        allow_chart_refinement=reason != "explicit")
    assert len(calls) == 1
    assert calls[0]["maxiter"] == (10 if reason in ("explicit", "exhausted") else 7)
    assert calls[0]["transformed"] == transformed
    assert result.stationarity_bound == 1.0
    assert len(result.route_budgets) == 1


@pytest.mark.parametrize("bound_refinement", (False, True))
def test_chart_handoff_failure_belongs_to_final_route_estimate(monkeypatch, bound_refinement):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    calls = []
    model = SimpleNamespace(family=3, p_vol=1, code=4, columns=0,
        data=np.ones(10), request_counts=np.zeros(8, dtype=int), inverse=lambda point: point.copy())
    model.assess = lambda *args: (True, 1., 0.)

    def phase(model, point, **options):
        calls.append(options)
        return OptimizeResult(x=point.copy(), jac=np.ones(2), fun=10., nit=1,
            success=len(calls) == 1, status=0 if len(calls) == 1 else -101,
            curvature_prepared=options["use_curvature"], kernel_dispatch=2)

    def bound_face(model, fitted, **options):
        if bound_refinement:
            fitted = OptimizeResult(fitted)
            fitted.fun = 9.
            fitted.selected_space = "theta"
        return fitted

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    monkeypatch.setattr(joint, "_refine_joint_active_bounds", bound_face)
    result = joint.solve_joint(model, np.ones(2), transformed=True,
        accuracy=1e-12, maxiter=20, allow_chart_refinement=True)
    assert len(calls) == 2
    assert result.chart_handoff_failure == (
        None if bound_refinement else dict(space="theta", mode=-101))


@pytest.mark.parametrize("outcome", ("better", "worse", "missing-gradient", "infeasible", "unavailable"))
@pytest.mark.parametrize("space", ("theta", "z"))
def test_chart_recovery_preserves_best_estimate_budget_and_request_accounting(monkeypatch, outcome, space):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    fitted = OptimizeResult(x=np.ones(2), fun=10., jac=np.ones(2), success=True,
        stationarity_bound=1., selected_space=space, nit=3, nfev=5, njev=4, nhev=1,
        phases=(), route_budgets=[7, 8], total_iteration_limit=11,
        chart_handoff_failure=dict(space="theta" if space == "z" else "z", mode=-101))
    model = SimpleNamespace(request_counts=np.zeros(8, dtype=int))
    model.assess = lambda *args: (outcome != "infeasible", .1, 0.)

    def evaluate(point, request):
        assert request == 2
        model.request_counts[2] += 1
        return None, np.full(2, .1), None

    model.evaluate = evaluate

    def phase(model, start, **options):
        np.testing.assert_array_equal(start, fitted.x)
        assert options["maxiter"] == 8 and options["use_curvature"]
        assert options["transformed"] == (space == "z")
        assert options["accuracy"] == np.finfo(float).eps
        assert options["stationarity_tolerance"] == 1e-6
        start[0] = 2.
        model.request_counts[3] += 2
        model.request_counts[4] += 1
        if outcome == "unavailable":
            raise ValueError("unavailable chart")
        return OptimizeResult(x=start, fun=11. if outcome == "worse" else 9.,
            jac=None if outcome == "missing-gradient" else np.full(2, .1),
            nit=4, success=False, status=-101, curvature_prepared=True, kernel_dispatch=2)

    monkeypatch.setattr(joint, "_solve_joint_phase", phase)
    result = joint._recover_failed_chart_handoff(model, fitted, accuracy=1e-12, maxiter=11)
    np.testing.assert_array_equal(fitted.x, np.ones(2))
    improved = outcome in ("better", "missing-gradient")
    assert result.fun == (9. if improved else 10.)
    assert result.success == (not improved)
    assert result.nit == (3 if outcome == "unavailable" else 7)
    assert (result.nfev, result.njev, result.nhev) == (7, 6+(outcome == "missing-gradient"), 2)
    assert result.route_budgets == fitted.route_budgets and result.total_iteration_limit == 11
    assert result.chart_handoff_failure is None
    assert fitted.chart_handoff_failure is not None
    record = result.phases[-1]
    assert record["purpose"] == "failed-handoff-recovery"
    assert record["iteration_budget"] == 8 and record["space"] == space
    assert record["failed_mode"] == -101
    assert record["jet_requests"]["masks"]["H"] == 1
    assert record["jet_requests"]["masks"]["F|G"] == 2
    if outcome == "unavailable":
        assert record["error"] == "unavailable chart"
    if improved:
        np.testing.assert_array_equal(result.jac, np.full(2, .1))
        assert result.stationarity_bound == .1


@pytest.mark.parametrize("reason", ("no-handoff", "exhausted", "credible", "kink"))
def test_chart_recovery_skips_unnecessary_work(monkeypatch, reason):
    from scipy.optimize import OptimizeResult
    from scivol import _joint_mean_fit as joint

    def unexpected(*args, **kwargs):
        pytest.fail("unexpected recovery")

    monkeypatch.setattr(joint, "_solve_joint_phase", unexpected)
    fitted = OptimizeResult(success=True, nit=10 if reason == "exhausted" else 2,
        stationarity_bound=0. if reason == "credible" else 1.,
        nearby_kink_evidence=dict(supports_kink_stop=reason == "kink"),
        chart_handoff_failure=None if reason == "no-handoff" else dict(space="theta", mode=-101))
    assert joint._recover_failed_chart_handoff(None, fitted, accuracy=1e-12, maxiter=10) is fitted


@pytest.mark.parametrize("selected_child", (False, True))
def test_chart_recovery_provenance_follows_selected_route(selected_child):
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import _retain_joint_refinement

    common = dict(success=True, stationarity_bound=1., nit=2, nfev=3, njev=4, nhev=1)
    primary = OptimizeResult(x=np.ones(2), fun=10., chart_handoff_failure=None, phases=(), **common)
    child = OptimizeResult(x=np.zeros(2), fun=9. if selected_child else 11.,
        chart_handoff_failure=dict(space="theta", mode=-101), **common)
    phase = dict(iterations=2, function_evaluations=3, gradient_evaluations=4, hessian_evaluations=1)
    result = _retain_joint_refinement(primary, child, phase)
    assert result.chart_handoff_failure == (child.chart_handoff_failure if selected_child else None)


@pytest.mark.parametrize("field,value", (
    ("maxiter", True), ("maxiter", 0), ("maxiter", 1.5), ("maxiter", np.inf),
    ("primary_maxiter", False), ("primary_maxiter", -1), ("primary_maxiter", np.nan),
    ("accuracy", 0), ("accuracy", -1), ("accuracy", np.inf), ("accuracy", np.nan),
))
def test_same_start_retry_rejects_invalid_controls_before_work(monkeypatch, field, value):
    from scivol import _joint_mean_fit as joint

    def unexpected(*args, **kwargs):
        pytest.fail("invalid controls reached the solver")

    monkeypatch.setattr(joint, "solve_joint", unexpected)
    options = dict(accuracy=1e-12, maxiter=10, primary_maxiter=7)
    options[field] = value
    with pytest.raises(ValueError):
        joint.solve_joint_with_retry(None, np.ones(2), **options)


@pytest.mark.parametrize("seed,har,sample,density", (
    pytest.param(2785, False, "normal", 3, id="narrow-ged-normal-2785"),
    pytest.param(2786, False, "persistent", 1, id="narrow-student-persistent-2786"),
    pytest.param(2786, True, "persistent", 1, id="wide-student-persistent-2786"),
    pytest.param(2785, True, "persistent", 3, id="wide-ged-persistent-2785"),
    pytest.param(2785, True, "persistent", 2, id="wide-skewt-persistent-2785"),
))
def test_public_joint_aparch_uses_portfolio_and_preserves_assembled_evidence(
    seed, har, sample, density, capsys
):
    import time
    from benchmark_native_joint import comparison_sample
    import scivol._joint_mean_fit as joint

    data, x = comparison_sample(seed, har, sample)
    p, q = (3, 2) if har else (1, 1)
    spec = (HARX([1, 3]) if har else ARX(1)) + FAMILIES[3](p, q) + DENSITIES[density]()
    model = JointMeanModel(spec, data, x)
    started = time.perf_counter()
    initial = model.initialize()
    unchanged = initial.copy()
    optimized = joint.solve_joint_with_metric_portfolio(model, initial, accuracy=1e-12, maxiter=2000)
    expected = joint.assemble_joint_fit(model, optimized, False, 1e-12, 2000, started)
    actual = fit_joint_mean_spec(spec, data, x=x, initial_params=initial, maxiter=2000)

    np.testing.assert_array_equal(initial, unchanged)
    for field in ("params", "sigma2", "resid", "std_resid", "conditional_mean",
                  "hessian", "cov_matrix", "std_errors"):
        np.testing.assert_equal(getattr(actual, field), getattr(expected, field))
    assert actual.log_likelihood == expected.log_likelihood
    assert actual.converged == expected.converged
    assert actual.n_iter == expected.n_iter <= 2000
    assert actual.convergence_message == expected.convergence_message
    np.testing.assert_equal(actual.fit_info.to_dict(), expected.fit_info.to_dict())
    assert actual.fit_info.optimizer_controller == "joint-primary-preserved-metric-portfolio"
    assert actual.fit_info.maxiter == actual.optimization_result.total_iteration_limit == 2000
    assert actual.fit_info.route_iteration_budgets[0] == 800
    assert np.isfinite(actual.params).all() and model.assess(actual.params, None)[0]
    assert np.isfinite(actual.sigma2).all() and np.all(actual.sigma2 > 0)
    warnings = actual.warnings
    assert warnings == expected.warnings
    assert actual.warnings == warnings
    if not actual.converged:
        assert actual.fit_info.warnings and warnings
        assert actual.fit_info.inference_status == "unavailable"
    if actual.fit_info.inference_status == "unavailable":
        assert actual.cov_matrix is None and actual.std_errors is None
    assert not capsys.readouterr().out


def test_same_start_retry_reaches_retained_ged_basin_with_honest_counts():
    from benchmark_native_joint import comparison_sample
    from scivol import APARCH
    from scivol._joint_mean_fit import solve_joint_with_retry

    data, x = comparison_sample(2785, False, "normal")
    model = JointMeanModel(ARX(1) + APARCH(1, 1) + DENSITIES[3](), data, x)
    initial = model.initialize()
    unchanged = initial.copy()
    fitted = solve_joint_with_retry(model, initial, accuracy=1e-12, maxiter=2000)
    np.testing.assert_array_equal(initial, unchanged)
    assert abs(fitted.fun - 536.8635178817815) <= 1e-9
    assert fitted.success and fitted.stationarity_bound <= 1e-6
    assert model.assess(fitted.x, fitted.jac)[0]
    assert fitted.nit <= 2000 and fitted.route_budgets[0] == 1000
    assert any(p.get("purpose") == "same-start-metric-retry" for p in fitted.phases)
    for field, request in (("nfev", "F"), ("njev", "G"), ("nhev", "H")):
        assert fitted[field] == fitted.jet_requests[request]


def test_same_start_retry_recovers_student_basin_after_uncertified_termination():
    pytest.importorskip("jax")
    from benchmark_native_joint import comparison_sample
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess
    from scivol._joint_mean_fit import solve_joint_with_retry

    data, x = comparison_sample(2786, False, "persistent")
    model = JointMeanModel(ARX(1) + FAMILIES[3]() + DENSITIES[1](), data, x)
    target = np.array([
        .035029255760839685, .22872031320933772, -.0827562471779584,
        .20564233378159835, .04955890905312122, .18948740740868877,
        -.0011063218071860662, .7324144196872795, 2.6413395424843857,
        8.854561424187475,
    ])
    expected = joint_value_grad_hess(target, model.data, model.features, model.h0,
        1, 1, 3, 1, request=3, exact_mean_anchor=True)
    value, gradient = model.evaluate(target, 3)[:2]
    assert abs(value-expected[0]) <= 1e-9
    np.testing.assert_allclose(gradient, expected[1], rtol=1e-8, atol=1e-8)
    assert model.assess(target, gradient)[0]
    initial = model.initialize()
    unchanged = initial.copy()
    fitted = solve_joint_with_retry(model, initial, accuracy=1e-12, maxiter=2000)
    np.testing.assert_array_equal(initial, unchanged)
    assert fitted.fun <= expected[0]+1e-9
    assert fitted.success and fitted.stationarity_bound <= 1e-6
    assert model.assess(fitted.x, fitted.jac)[0]
    assert fitted.nit <= 2000
    assert any(phase["space"] == "z" for phase in fitted.phases)


def test_same_start_retry_preserves_retained_gh_likelihood_and_honest_reporting():
    import time
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import assemble_joint_fit, solve_joint_with_retry

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3](1, 1) + DENSITIES[4](), data, x)
    initial = model.initialize()
    fitted = solve_joint_with_retry(model, initial, accuracy=1e-12, maxiter=2000)
    assert fitted.fun <= 561.0290623250537+1e-9
    assert model.assess(fitted.x, fitted.jac)[0]
    assert fitted.nit <= 2000
    assert fitted.chart_handoff_failure is None
    assert fitted.success and fitted.stationarity_bound <= 1e-6
    for field, request in (("nfev", "F"), ("njev", "G"), ("nhev", "H")):
        assert fitted[field] == fitted.jet_requests[request]

    theta, nll = fitted.x.copy(), fitted.fun
    reported = assemble_joint_fit(model, fitted, False, 1e-12, 2000, time.perf_counter())
    np.testing.assert_array_equal(fitted.x, theta)
    assert reported.log_likelihood == pytest.approx(-nll, abs=1e-9, rel=0)
    assert reported.fit_info.jet_consistency["status"] == "consistent"
    assert reported.converged and reported.fit_info.kkt_residual <= 1e-6
    assert reported.fit_info.stationarity_status == "certified"
    assert reported.fit_info.inference_status == "unavailable"
    assert reported.fit_info.warnings
    assert np.isfinite(reported.sigma2).all() and np.all(reported.sigma2 > 0)


def test_retained_high_power_gh_gradients_match_independent_ad():
    pytest.importorskip("jax")
    from benchmark_native_joint import comparison_sample
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess
    from scivol._fit_credibility import jet_agreement

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3]() + DENSITIES[4](), data, x)
    theta = np.array([
        -.059682973480150336, -.021534820952918598, .05496812949122731,
        .014125966377041822, .0019685032213102244, 0., .9999998806322236,
        .8916132001177535, 217.6418930995166, .2274801548580271, 25., 6.,
    ])
    expected = joint_value_grad_hess(theta, model.data, model.features, model.h0,
        1, 1, 3, 4, request=3, exact_mean_anchor=True)
    assert abs(expected[0]-561.0290623250537) <= 1e-9
    # Independent 70-digit differentiation gives this outward alpha score.
    # The AD Bessel integral also checks the complete gradient, including shape.
    alpha_score = 8.546212084577906e151
    for fixed in (True, False):
        engine = JointMeanModel(model.spec, data, x, fixed=fixed)
        full = engine.evaluate(theta, 7)
        for mask in range(1, 8):
            value, gradient, hessian = engine.evaluate(theta, mask)[:3]
            if mask & 1:
                assert abs(value-expected[0]) <= 1e-9
            else:
                assert value is None
            if mask & 2:
                np.testing.assert_allclose(gradient, expected[1], rtol=2e-12, atol=1e-9)
                assert abs(gradient[model.columns+1]/alpha_score-1) < 4e-15
                agreement = jet_agreement(full[0], full[1], full[0], gradient,
                    model.units, len(model.data), 1e-12)
                assert agreement["status"] == "consistent"
                assert engine.assess(theta, gradient)[1] < 1e-6
            else:
                assert gradient is None
            if mask & 4:
                assert np.isfinite(hessian).all()
                np.testing.assert_allclose(hessian, full[2], rtol=2e-12, atol=1e-9)
            else:
                assert hessian is None


def test_retained_uncertified_aparch_gh_point_is_reported_without_parameter_changes():
    import time
    from scipy.optimize import OptimizeResult
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import assemble_joint_fit

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + FAMILIES[3]() + DENSITIES[4](), data, x)
    theta = np.array([
        -.05954495312732262, -.021290926423204393, .05485222662103962,
        .013454069422175337, .002522053115910467, 2.4166846e-317,
        .9999998652967536, .8834863705286036, 215.4632640594723,
        .22911577118769738, 24.999999999999993, 5.999999998267437,
    ])
    value, gradient = model.evaluate(theta, 3)[:2]
    feasible, bound, violation = model.assess(theta, gradient)
    assert feasible and violation == 0 and 1e-6 < bound < 1e-3
    assert abs(value-561.0294622517131) <= 1e-9
    optimized = OptimizeResult(x=theta.copy(), fun=value, jac=gradient.copy(),
        success=True, status=0, selected_space="z", stationarity_bound=bound,
        nit=1, phases=(dict(space="z", mode=0, iterations=1),))
    reported = assemble_joint_fit(model, optimized, True, 1e-12, 2000, time.perf_counter())
    np.testing.assert_array_equal(reported.params, theta)
    assert abs(reported.log_likelihood+value) <= 1e-9
    assert reported.fit_info.optimizer_terminated
    assert reported.fit_info.stationarity_status == "uncertified"
    assert reported.fit_info.inference_status == "unavailable"
    assert not reported.converged and reported.fit_info.warnings


def test_same_start_retry_does_not_refine_an_already_certified_box_limit():
    from benchmark_native_joint import comparison_sample
    from scivol._joint_mean_fit import solve_joint_with_retry

    data, x = comparison_sample(2786, True, "normal")
    model = JointMeanModel(HARX([1, 3]) + FAMILIES[3](3, 2) + DENSITIES[1](), data, x)
    fitted = solve_joint_with_retry(model, model.initialize(), accuracy=1e-12, maxiter=2000)
    retry = next(p for p in fitted.phases if p.get("purpose") == "same-start-metric-retry")
    assert retry["stationarity_bound"] <= 1e-6
    assert len(retry["refinement_phases"]) == 1
    assert abs(fitted.fun-558.5411055274267) <= 1e-9
    assert fitted.success and fitted.stationarity_bound <= 1e-6
    assert fitted.fun < retry["nll"]


@pytest.mark.parametrize("evidence", (None, {}, {"supports_kink_stop": False},
                                     {"supports_kink_stop": True}))
def test_two_route_controller_respects_kink_stop_without_certifying(evidence):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import solve_joint_two_routes

    calls = []

    def controller(model, start, **options):
        calls.append(options)
        return OptimizeResult(x=start.copy(), fun=1.0, success=False,
            stationarity_bound=0.1, nearby_kink_evidence=evidence,
            phases=(), nit=1, nfev=1, njev=1, nhev=0)

    stopped = bool(evidence and evidence.get("supports_kink_stop"))
    model = SimpleNamespace(data=np.zeros(10), request_counts=np.zeros(8, dtype=int))
    result = solve_joint_two_routes(model, np.ones(2),
        accuracy=1e-12, maxiter=10, controller=controller)
    assert len(calls) == (1 if stopped else 2)
    assert not result.success and result.stationarity_bound == 0.1
    assert result.nearby_kink_evidence == evidence
    assert result.nit == result.nfev == result.njev == len(calls)
    assert result.route_budgets == [5] * len(calls)


def test_two_route_kink_stop_keeps_warning_and_unavailable_inference():
    import time
    from benchmark_native_joint import comparison_sample
    from scivol import EGARCH
    from scivol._joint_mean_fit import solve_joint_two_routes, assemble_joint_fit

    data, x = comparison_sample(2785, True, "persistent")
    model = JointMeanModel(HARX([1, 3]) + EGARCH(3, 2) + DENSITIES[0](), data, x)
    optimized = solve_joint_two_routes(model, model.initialize(), accuracy=1e-12, maxiter=600)
    result = assemble_joint_fit(model, optimized, False, 1e-12, 600, time.perf_counter())
    assert optimized.nearby_kink_evidence["supports_kink_stop"]
    assert {phase["purpose"] for phase in optimized.phases} == {"primary"}
    assert abs(-result.log_likelihood - 514.3320395720434) <= 1e-9
    assert not result.converged
    assert result.fit_info.stationarity_status == "uncertified"
    assert result.fit_info.inference_status == "unavailable"
    assert result.std_errors is None
    assert any("nearby residual kink" in warning for warning in result.warnings)


@pytest.mark.parametrize("success,bound,stalled,stops", (
    (True, 1e-8, True, True), (True, 1e-8, False, False),
    (False, 1e-8, True, False), (True, 1e-2, True, False),
))
def test_two_route_honors_stalled_refinement_only_at_certified_points(
    success, bound, stalled, stops
):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import solve_joint_two_routes

    calls = []
    model = SimpleNamespace(data=np.zeros(10), request_counts=np.zeros(8, dtype=int))

    def controller(model, start, **options):
        calls.append(options)
        return OptimizeResult(x=start.copy(), fun=1.0, success=success,
            stationarity_bound=bound, predicted_coordinate_gain=1.0,
            refinement_stalled=stalled, phases=(), nit=1, nfev=1, njev=1, nhev=0)

    result = solve_joint_two_routes(model, np.ones(2), accuracy=1e-12,
                                    maxiter=10, controller=controller)
    assert len(calls) == (1 if stops else 2)
    assert result.success == success and result.stationarity_bound == bound
    assert result.predicted_coordinate_gain == 1.0


def test_two_route_stalled_curvature_keeps_precision_warning():
    import time
    from benchmark_native_joint import comparison_sample
    from scivol import EGARCH
    from scivol._joint_mean_fit import solve_joint_two_routes, assemble_joint_fit

    data, x = comparison_sample(2786, False, "normal")
    model = JointMeanModel(ARX(1) + EGARCH(1, 1) + DENSITIES[2](), data, x)
    optimized = solve_joint_two_routes(model, model.initialize(), accuracy=1e-12, maxiter=600)
    assert optimized.refinement_stalled and len(optimized.phases) == 2
    assert {phase["purpose"] for phase in optimized.phases} == {"primary"}
    assert optimized.phases[-1]["curvature_scaling"]
    result = assemble_joint_fit(model, optimized, False, 1e-12, 600, time.perf_counter())
    assert abs(-result.log_likelihood - 557.9857825932063) <= 1e-9
    assert result.converged and result.fit_info.stationarity_status == "certified"
    assert any("Local curvature predicts" in warning for warning in result.warnings)


@pytest.mark.parametrize("budget", (0, -1, 1.5, np.inf, np.nan))
def test_two_route_controller_rejects_invalid_budgets_before_work(budget):
    from scivol._joint_mean_fit import solve_joint_two_routes

    with pytest.raises(ValueError, match="iteration budget"):
        solve_joint_two_routes(None, None, accuracy=1e-12, maxiter=budget)


@pytest.mark.parametrize("error", (ValueError, RuntimeError))
def test_two_route_controller_does_not_hide_primary_failure(error):
    from scivol._joint_mean_fit import solve_joint_two_routes

    calls = []

    def controller(*args, **kwargs):
        calls.append(kwargs)
        raise error("no finite feasible primary point")

    with pytest.raises(error, match="no finite feasible primary point"):
        solve_joint_two_routes(None, np.ones(2), accuracy=1e-12, maxiter=6,
                               controller=controller)
    assert len(calls) == 1


@pytest.mark.parametrize("density", (3, 4, 5, 6, 7))
def test_smooth_aparch_primary_certifies_without_redundant_refinement(density):
    from scivol import APARCH
    from scivol._joint_mean_fit import assemble_joint_fit, solve_joint_with_retry
    import time

    rng = np.random.default_rng(2785)
    data = rng.normal(size=400)
    x = rng.normal(size=(400, 2))
    model = JointMeanModel(ARX(1) + APARCH() + DENSITIES[density](), data, x)
    started = time.perf_counter()
    fitted = solve_joint_with_retry(model, model.initialize(), accuracy=1e-12, maxiter=1000)
    result = assemble_joint_fit(model, fitted, False, 1e-12, 1000, started)
    phases = result.fit_info.phase_diagnostics
    assert result.converged
    assert result.fit_info.kkt_residual <= result.fit_info.kkt_tolerance
    assert phases[0]["accuracy"] == 1e-12
    if phases[0]["stationarity_bound"] <= result.fit_info.kkt_tolerance:
        assert len(phases) == 1
    else:
        assert len(phases) >= 2
        assert phases[1]["accuracy"] == np.finfo(float).eps
    assert -result.log_likelihood <= phases[0]["nll"]
    assert sum(p["iterations"] for p in phases) <= result.fit_info.maxiter


def test_frozen_egarch_acceptance_panel_keeps_all_comparators():
    import benchmark_native_joint as benchmark

    cases = tuple(case for shard in range(4)
                  for case in benchmark.egarch_acceptance_cases(shard))
    assert len(cases) == len(set(cases)) == 64
    assert set(cases) == {(seed, sample, density, har)
        for seed in (2785, 2786) for sample in ("normal", "persistent")
        for density in range(8) for har in (False, True)}
    assert benchmark.EGARCH_ACCEPTANCE_VERSION == "joint-egarch-4.1-v3"
    assert benchmark.EGARCH_ACCEPTANCE_MAXITER == 300
    assert benchmark.EGARCH_ACCEPTANCE_NATIVE_MAXITER == 1800
    assert benchmark.EGARCH_ACCEPTANCE_NLL_TOLERANCE == 1e-9
    assert benchmark.EGARCH_ACCEPTANCE_LIKELIHOOD_RATE == 1.0
    assert [density.__name__ for density in benchmark.DENSITIES] == [
        "Normal", "StudentT", "SkewT", "GED", "GeneralizedHyperbolic",
        "NormalInverseGaussian", "Hyperbolic", "VarianceGamma",
    ]
    for shard in (-1, 4):
        with pytest.raises(ValueError, match="acceptance shard"):
            benchmark.egarch_acceptance_cases(shard)


@pytest.mark.parametrize("failed", (False, True))
def test_frozen_egarch_benchmark_mode_overrides_exploratory_settings(monkeypatch, tmp_path, failed):
    import json
    import sys
    import benchmark_native_joint as benchmark

    calls = []
    destination = tmp_path / "acceptance.json"

    def compare(*arguments):
        calls.append(arguments)
        if failed:
            raise RuntimeError("fit sentinel")
        return dict(nll_loss=0.0, speedup=1.0, native_certified=True)

    monkeypatch.setattr(benchmark, "compare", compare)
    monkeypatch.setattr(benchmark.signal, "alarm", lambda _: None)
    monkeypatch.setattr(benchmark.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["benchmark_native_joint.py",
        "--egarch-acceptance-shard", "3", "--density", "0", "--family", "0",
        "--maxiter", "1", "--native-maxiter", "2", "--pairs", "1", "--legacy-log", "--native-log", "z",
        "--initial-curvature", "--reference-log", "--reference-engine", "canonical",
        "--speculate-trials", "--curvature-reset", "--output", str(destination)])
    benchmark.main()
    expected = [
        (2, density, har, 3, 300, False, None, 2786, False, False,
         "legacy-matched", None, "persistent", None, 1800, False)
        for density in range(8) for har in (False, True)
    ]
    assert calls == expected
    report = json.loads(destination.read_text())
    assert report["complete"]
    assert report["acceptance"]["version"] == "joint-egarch-4.1-v3"
    assert report["acceptance"]["shard"] == 3
    assert len(report["acceptance"]["cases"]) == 16
    assert report["acceptance"]["minimum_likelihood_nonregression_rate"] == 1.0
    assert "Numerical correctness is mandatory" in report["acceptance"]["aggregate_scope"]
    assert report["settings"]["maxiter"] == 300
    assert report["settings"]["native_maxiter"] == 1800
    assert report["settings"]["aparch_retry"] is False
    if failed:
        assert all(row["iteration_limits"] == {"native": 1800, "reference": 300}
                   and "fit sentinel" in row["error"] for row in report["rows"])


@pytest.fixture
def frozen_egarch_reports():
    """Synthetic evidence tests the gate, not a numerical implementation."""
    import benchmark_native_joint as benchmark

    reports = []
    for shard, (seed, sample) in enumerate(benchmark.EGARCH_ACCEPTANCE_SHARDS):
        rows = [dict(family=2, density=density, har=har,
            iteration_limits=dict(native=1800, reference=300),
            native_nll=100.0, reference_nll=100.0,
            legacy_at_native=100.0, legacy_at_reference=100.0,
            native_feasible=True, reference_feasible=True,
            native_kkt_bound=1e-7, reference_kkt_bound=1e-7,
            native_fit_info=dict(stationarity_status="certified", kkt_tolerance=1e-6),
            comparison_domain="identical", reference_engine="legacy-matched")
            for _, _, density, har in benchmark.egarch_acceptance_cases(shard)]
        reports.append(dict(complete=True, rows=rows,
            acceptance=dict(version=benchmark.EGARCH_ACCEPTANCE_VERSION, shard=shard,
                cases=[list(case) for case in benchmark.egarch_acceptance_cases(shard)],
                nll_tolerance=1e-9, minimum_likelihood_nonregression_rate=1.0),
            settings=dict(maxiter=300, native_maxiter=1800, pairs=3, seed=seed, sample=sample,
                reference_engine="legacy-matched", native_log="auto", legacy_log=False,
                reference_log=False, initial_curvature=False, native_speculate=None,
                native_curvature_reset=None, family=[2], density=list(range(8))),
            binary_sha256="a"*64, legacy_binary_sha256="b"*64,
            source_sha256={"benchmark_native_joint.py": "c"*64}))
    return reports


@pytest.mark.parametrize("losses,expected", [(0, "review_required"), (1, "failed"), (3, "failed"), (4, "failed")])
def test_egarch_acceptance_summary_requires_64_of_64(frozen_egarch_reports, losses, expected):
    import benchmark_native_joint as benchmark

    for index in range(losses):
        row = frozen_egarch_reports[index % 4]["rows"][2*index+1]
        row.update(native_nll=100.5, legacy_at_native=100.5, nll_loss=-999)
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["status"] == expected
    assert summary["expected_cases"] == summary["observed_cases"] == 64
    assert summary["minimum_nonregressions"] == 64
    assert summary["likelihood_nonregressions"] == 64-losses
    assert summary["likelihood_rate_met"] == (losses == 0)
    assert sum(row["nll_loss"] == 0.5 for row in summary["comparisons"]) == losses
    assert summary["groups"]["width"][1]["losses"] == losses


@pytest.mark.parametrize("missing", ("shard", "row", "completion"))
def test_egarch_acceptance_summary_keeps_missing_cases_in_denominator(frozen_egarch_reports, missing):
    import benchmark_native_joint as benchmark

    if missing == "shard":
        frozen_egarch_reports.pop()
    elif missing == "row":
        frozen_egarch_reports[3]["rows"].pop()
    else:
        frozen_egarch_reports[3]["complete"] = False
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["status"] == "incomplete"
    assert summary["expected_cases"] == 64
    assert summary["likelihood_rate_met"] is None
    assert len(summary["comparisons"]) == 64
    assert summary["complete_shards"] == [0, 1, 2]


@pytest.mark.parametrize("defect", ("duplicate-shard", "duplicate-empty-shard", "duplicate-row",
    "wrong-case", "budget", "native-budget", "row-budget", "version",
    "solver", "aparch-retry", "tolerance", "native-binary", "legacy-binary", "source"))
def test_egarch_acceptance_summary_rejects_mixed_evidence(frozen_egarch_reports, defect):
    import benchmark_native_joint as benchmark

    report = frozen_egarch_reports[-1]
    if defect.startswith("duplicate") and defect.endswith("shard"):
        if defect == "duplicate-empty-shard":
            report.update(rows=[], complete=False)
        frozen_egarch_reports.append(report)
    elif defect == "duplicate-row":
        report["rows"].append(report["rows"][0])
    elif defect == "wrong-case":
        report["rows"][0]["family"] = 3
    elif defect == "budget":
        report["settings"]["maxiter"] = 1000
    elif defect == "native-budget":
        report["settings"]["native_maxiter"] = 300
    elif defect == "row-budget":
        report["rows"][0]["iteration_limits"]["native"] = 300
    elif defect == "version":
        report["acceptance"]["version"] = "joint-egarch-4.1-v1"
    elif defect == "solver":
        report["settings"]["reference_engine"] = "canonical"
    elif defect == "aparch-retry":
        report["settings"]["aparch_retry"] = True
    elif defect == "tolerance":
        report["acceptance"]["nll_tolerance"] = 1e-8
    elif defect.endswith("binary"):
        report["binary_sha256" if defect == "native-binary" else "legacy_binary_sha256"] = "d"*64
    else:
        report["source_sha256"] = {}
    with pytest.raises(ValueError):
        benchmark.summarize_egarch_acceptance(frozen_egarch_reports)


@pytest.mark.parametrize("defect", ("nan", "infinity", "missing-value", "disagreement", "failure",
    "missing-feasibility", "false-certificate", "missing-status", "changed-kkt", "wrong-domain"))
def test_egarch_acceptance_summary_does_not_count_unresolved_accuracy_as_pass(frozen_egarch_reports, defect):
    import benchmark_native_joint as benchmark

    row = frozen_egarch_reports[0]["rows"][0]
    if defect in ("nan", "infinity", "missing-value", "disagreement"):
        row["legacy_at_native"] = {"nan": np.nan, "infinity": np.inf,
            "missing-value": None, "disagreement": 101.0}[defect]
    elif defect == "failure":
        row["error"] = "Reference failed"
    elif defect == "missing-feasibility":
        del row["native_feasible"]
    elif defect == "false-certificate":
        row["native_kkt_bound"] = 1.0
    elif defect == "missing-status":
        del row["native_fit_info"]["stationarity_status"]
    elif defect == "changed-kkt":
        row["native_fit_info"]["kkt_tolerance"] = 0.1
    else:
        row["comparison_domain"] = "unmatched"
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["status"] == "incomplete"
    assert summary["likelihood_rate_met"] is None
    assert summary["comparisons"][0]["issues"]


@pytest.mark.parametrize("reference_kkt", (1e-7, 1.0, np.inf))
def test_egarch_acceptance_summary_distinguishes_certification_from_residual_ranking(frozen_egarch_reports, reference_kkt):
    import benchmark_native_joint as benchmark

    row = frozen_egarch_reports[0]["rows"][0]
    row.update(native_kkt_bound=100.0, reference_kkt_bound=reference_kkt, reference_terminated=True)
    row["native_fit_info"]["stationarity_status"] = "uncertified"
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["status"] == ("failed" if reference_kkt < 1e-6 else "review_required")
    assert summary["comparisons"][0]["certification_regression"] == (reference_kkt < 1e-6)


@pytest.mark.parametrize("native_feasible,reference_feasible", [(False, True), (True, False), (False, False)])
def test_egarch_acceptance_summary_never_hides_infeasible_points(frozen_egarch_reports, native_feasible, reference_feasible):
    import benchmark_native_joint as benchmark

    row = frozen_egarch_reports[0]["rows"][0]
    row.update(native_feasible=native_feasible, reference_feasible=reference_feasible)
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["status"] == "incomplete"
    assert summary["likelihood_rate_met"] is None
    assert summary["comparisons"][0]["feasibility_regression"] == (reference_feasible and not native_feasible)


@pytest.mark.parametrize("loss,nonregression", [(1e-9, True), (np.nextafter(1e-9, np.inf), False)])
def test_egarch_acceptance_summary_preserves_absolute_loss_tolerance(frozen_egarch_reports, loss, nonregression):
    import benchmark_native_joint as benchmark

    row = frozen_egarch_reports[0]["rows"][0]
    row.update(native_nll=float(loss), legacy_at_native=float(loss),
               reference_nll=0.0, legacy_at_reference=0.0)
    summary = benchmark.summarize_egarch_acceptance(frozen_egarch_reports)
    assert summary["comparisons"][0]["nonregression"] == nonregression


def test_egarch_acceptance_summary_cli_does_not_fit_or_overwrite_evidence(frozen_egarch_reports, monkeypatch, tmp_path, capsys):
    import json
    import sys
    import benchmark_native_joint as benchmark

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A saved-report audit must not run a fit.")

    monkeypatch.setattr(benchmark, "compare", unexpected_fit)
    paths = [tmp_path/f"shard-{shard}.json" for shard in range(4)]
    for path, report in zip(paths, frozen_egarch_reports):
        path.write_text(json.dumps(report))
    destination = tmp_path/"summary.json"
    arguments = ["benchmark_native_joint.py", "--summarize-egarch", *map(str, paths), "--output"]
    monkeypatch.setattr(sys, "argv", [*arguments, str(destination)])
    benchmark.main()
    summary = json.loads(destination.read_text())
    assert summary["status"] == "review_required"
    assert json.loads(capsys.readouterr().out) == summary
    original = paths[0].read_bytes()
    monkeypatch.setattr(sys, "argv", [*arguments, str(paths[0])])
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert paths[0].read_bytes() == original


@pytest.mark.parametrize("wide,density,reference", [
    (True, 0, [0.006180270357349499, 0.09219412121273185, 0.09899357709323367,
        -0.13041494504343182, -0.03228663010063998, 0.016552608448240327,
        0.09701585976017481, 0.006113405730778123, 0.48872508650956686,
        0.39397482991033844, -0.27732676288125985, 0.18160061864779736,
        0.0010879136208554476, -0.24961022570263564, 0.0021882143809979714,
        0.933634745597569]),
    (False, 2, [0.033902177068172876, 0.2440265941136446, -0.09605303627664827,
        0.19753991910975532, 0.00030282606658733863, 0.4181890776681622,
        0.048386157539365274, 0.94272122450994, 9.272943453334257,
        -0.08000829995881732]),
])
def test_egarch_recovers_verified_historical_neighboring_basin(wide, density, reference):
    from benchmark_native_joint import comparison_sample
    from scivol import EGARCH

    # Independently checked historical endpoints, not initial guesses or oracles.
    data, x = comparison_sample(2786, wide, "persistent")
    mean, orders = (HARX([1, 3]), (3, 2)) if wide else (ARX(1), (1, 1))
    spec = mean + EGARCH(*orders) + DENSITIES[density]()
    model = JointMeanModel(spec, data, x)
    reference = np.array(reference)
    target, reference_gradient, *_ = model.evaluate(reference, 3)
    assert model.assess(reference, reference_gradient)[0]
    fitted = fit_joint_mean_spec(spec, data, x=x, initial_params=model.initialize(), maxiter=600)
    value, gradient, *_ = model.evaluate(fitted.params, 3)
    assert value <= target + 1e-9
    assert model.assess(fitted.params, gradient)[0] and fitted.converged
    assert fitted.fit_info.optimizer_controller == "joint-continuation"
    phases = fitted.fit_info.phase_diagnostics
    assert sum(p["iterations"] for p in phases) <= 600
    assert any(p.get("purpose") == "local-region-check" for p in phases)


def test_rejection_policy_preserves_a_finite_uncertified_estimate():
    from scivol import EGARCH, HARX, VarianceGamma

    # Same jet, sample, start and budget with the historical rejection controller.
    # This is a solution-quality control, not a mathematical oracle or certificate.
    reference = np.array(
        [
            -0.029230013131622688,
            -0.02513123788578306,
            0.023240135474224894,
            0.009594256818750037,
            0.015168433521062874,
            -0.0752322535913793,
            0.009443354739201253,
            -0.00909363231614821,
            -0.32956526562601884,
            -0.06886552330431978,
            0.1903149404722953,
            -0.2682838733977858,
            0.16065218755982777,
            0.11539289561035056,
            0.6770535143367006,
            0.29083817593073047,
            -0.17101829598024995,
            3.0601418130789075,
        ]
    )
    rng = np.random.default_rng(2785)
    data, x = rng.normal(size=400), rng.normal(size=(400, 4))
    spec = HARX([1, 3]) + EGARCH(3, 2) + VarianceGamma()
    model = JointMeanModel(spec, data, x)
    fitted = fit_joint_mean_spec(spec, data, x=x, maxiter=1000)
    reference_nll = model.evaluate(reference, 1)[0]
    fitted_nll, gradient, *_ = model.evaluate(fitted.params, 3)
    assert np.isfinite(reference_nll) and np.isfinite(fitted_nll)
    assert fitted_nll <= reference_nll + 1e-9
    assert model.assess(fitted.params, gradient)[0]
    if not fitted.converged:
        assert fitted.warnings
        assert fitted.fit_info.inference_status == "unavailable"


def test_result_assembly_rejects_an_infeasible_estimate(monkeypatch):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared()
    result = OptimizeResult(x=model.initialize())
    monkeypatch.setattr(model, "assess", lambda *args: (False, np.inf, 1.0))
    with pytest.raises(RuntimeError, match="outside the model domain"):
        assemble_joint_fit(model, result, False, 1e-12, 100, time.perf_counter())


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize(
    "previous,current,expected",
    (
        (1e-3, 1e-8, 1e-3),
        (1e-8, 1e-3, 1e-3),
        (np.inf, 1e-8, np.inf),
        (np.nan, 1e-8, np.inf),
        (1e-8, np.nan, np.inf),
        (1e-8, 2e-8, 2e-8),
        (None, 2e-8, 2e-8),
    ),
)
def test_assembly_preserves_same_point_stationarity_evidence(
    family, previous, current, expected, monkeypatch, capsys
):
    import time
    from scipy.optimize import OptimizeResult
    from scivol._joint_mean_fit import assemble_joint_fit

    model, _ = prepared(family, n=100)
    theta = model.initialize()
    value = model.evaluate(theta, 7)[0]
    result = OptimizeResult(
        x=theta.copy(),
        fun=value,
        status=0,
        success=True,
        phases=(),
        selected_space="theta",
    )
    if previous is not None:
        result.stationarity_bound = previous
    monkeypatch.setattr(model, "assess", lambda *args: (True, current, 0.0))
    requests_before = tuple(model.request_counts)
    fitted = assemble_joint_fit(model, result, False, 1e-12, 100, time.perf_counter())

    np.testing.assert_array_equal(fitted.params, theta)
    assert fitted.log_likelihood == -value
    assert fitted.fit_info.optimizer_terminated
    assert fitted.fit_info.kkt_residual == expected
    assert result.stationarity_bound == expected
    assert fitted.converged == (expected <= fitted.fit_info.kkt_tolerance)
    assert sum(model.request_counts) - sum(requests_before) == 1
    if not fitted.converged:
        assert fitted.fit_info.stationarity_status == "uncertified"
        assert fitted.fit_info.inference_status == "unavailable"
        assert fitted.std_errors is None
        assert fitted.warnings
    assert capsys.readouterr().out == ""


def test_positive_covariance_diagonal_is_not_enough_for_inference():
    matrix = np.eye(3) - (11 / 3) * np.ones((3, 3))
    assert np.all(np.diag(np.linalg.inv(matrix)) > 0)
    assert observed_covariance(matrix, stationary=True) is None
    assert observed_covariance(np.eye(3), stationary=False) is None
    np.testing.assert_array_equal(
        observed_covariance(np.eye(3), stationary=True), np.eye(3)
    )


@pytest.mark.parametrize("fixed", (False, True))
def test_aparch_ged_near_zero_residual_has_consistent_request_plans(fixed):
    from benchmark_native_joint import comparison_sample
    from scivol import APARCH, GED
    from scivol._fit_credibility import jet_agreement

    data, x = comparison_sample(2785, True, "persistent")
    model = JointMeanModel(HARX([1, 3]) + APARCH(3, 2) + GED(), data, x, fixed=fixed)
    theta = np.array([
        .08754491714010511, .25660838864934593, .06485137260122179,
        -.06640820045045606, -.09484962134232898, .09198939537155856,
        .055230671525926534, .39924028777379955, .047923635732866315,
        .1375407287046691, .003526193911165149, .9966835756890273,
        -.3745339747231424, .9999999999999982, .42732753319116495,
        4.5470909691735284e-13, .08612962253941438, 1.744756962808635,
    ])
    value, gradient, hessian, *_ = model.evaluate(theta, 7)
    # Independent 50/80-digit full recursion at the exact binary64 inputs.
    assert abs(value - 509.99821499630635) <= 1e-9
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    units = model.units.copy()
    units[model.columns] = model.h0 ** (theta[model.shape_start - 1] / 2)
    for mask in range(1, 8):
        f, g, h, *_ = model.evaluate(theta, mask)
        if mask & 1:
            assert abs(f - value) <= 1e-9
        if mask & 2:
            evidence = jet_agreement(value, gradient, value, g, units, len(data), 1e-12)
            assert evidence["status"] == "consistent", evidence
        if mask & 4:
            np.testing.assert_array_equal(h, hessian)


def test_positive_but_unresolved_information_has_no_covariance():
    correlation = np.array(
        [[1.0, np.nextafter(1.0, 0.0)], [np.nextafter(1.0, 0.0), 1.0]]
    )
    np.linalg.cholesky(correlation)
    assert observed_covariance(correlation, stationary=True) is None
    units = np.array([1e-80, 1e80])
    assert (
        observed_covariance(correlation * units[:, None] * units, stationary=True)
        is None
    )


def test_covariance_equilibration_preserves_units_and_well_resolved_information():
    information = np.array([[2.0, 0.25], [0.25, 1.0]])
    units = np.array([1e-100, 1e100])
    expected = np.linalg.inv(information) / units[:, None] / units
    actual = observed_covariance(information * units[:, None] * units, stationary=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_projection_does_not_hide_free_gradient_beside_large_active_force():
    bound = projected_stationarity_bound(
        np.array([1e12, 1e-5]),
        np.array([0.0, -np.inf]),
        np.full(2, np.inf),
        np.empty((0, 2)),
        np.empty(0),
    )
    assert bound >= 1e-5


def test_projection_includes_inactive_constraint_slack():
    bound = projected_stationarity_bound(
        np.array([0.25]),
        np.array([-0.1]),
        np.array([np.inf]),
        np.empty((0, 1)),
        np.empty(0),
    )
    assert bound >= 0.1


@pytest.mark.parametrize("slack", [-1., np.nan, np.inf])
def test_unavailable_jacobian_cannot_hide_invalid_constraints(slack):
    assert projected_stationarity_bound(
        np.zeros(2), np.full(2,-np.inf), np.full(2,np.inf), None, np.array([slack])) == np.inf


def test_unavailable_jacobian_uses_a_full_problem_bound_not_a_box_projection():
    # g=(10,-1), x>=-5, and the unavailable coupled row is y<=x. Its projection
    # is (-4.5,-4.5), whose norm exceeds the box-only projection norm sqrt(26).
    bound = projected_stationarity_bound(
        np.array([10.,-1.]), np.array([-5.,-np.inf]), np.full(2,np.inf), None, np.array([0.]))
    assert bound >= 4.5*np.sqrt(2.)


@pytest.mark.parametrize("force", [1e12, 1e200, 1e300])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_projection_exact_box_multiplier_preserves_small_free_gradient(force, side):
    lower = np.array([0.0 if side > 0 else -np.inf, -np.inf])
    upper = np.array([np.inf if side > 0 else 0.0, np.inf])
    bound = projected_stationarity_bound(
        np.array([side * force, 1e-5]),
        lower,
        upper,
        np.empty((0, 2)),
        np.empty(0),
    )
    assert 1e-5 <= bound <= 1.000000000001e-5


def test_projection_does_not_treat_nearby_box_face_as_active():
    bound = projected_stationarity_bound(
        np.array([1e200, 1e-5]),
        np.array([-1e-8, -np.inf]),
        np.full(2, np.inf),
        np.empty((0, 2)),
        np.empty(0),
    )
    # The nearby face pays its full complementarity cost, unlike an active face.
    assert 1e96 <= bound <= 1.5e96


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_projection_subnormal_box_slack_preserves_the_free_gradient(side):
    lower = np.array([-1e-320 if side > 0 else -np.inf, -np.inf])
    upper = np.array([np.inf if side > 0 else 1e-320, np.inf])
    bound = projected_stationarity_bound(
        np.array([side*1e151, 1e-5]), lower, upper, np.empty((0, 2)), np.empty(0))
    assert 1e-5 <= bound <= 1.000000000001e-5


def test_box_projection_bound_covers_exact_rational_dual_gaps():
    from fractions import Fraction
    from scivol._fit_credibility import _box_projection_bound

    rng = np.random.default_rng(2786)
    for _ in range(200):
        gradient = rng.choice([-1., 1.], 4)*10.**rng.uniform(-320, 300, 4)
        lower, upper = -10.**rng.uniform(-320, 300, 4), 10.**rng.uniform(-320, 300, 4)
        lower[rng.random(4) < .1] = 0.
        bound = _box_projection_bound(gradient, lower, upper)
        squared_gap = Fraction(0)
        for force, low, high in zip(gradient, lower, upper):
            distance = -low if force > 0 else high
            g, slack = Fraction(abs(force)), Fraction(distance)
            squared_gap += 2*g*slack if distance == 0 or distance < abs(force)/2 else g*g
        assert Fraction(bound)**2 >= squared_gap


@pytest.mark.parametrize("gap", [1e-320, 1e-8, 0.1, 1., 10.])
def test_nearby_box_certificate_bounds_a_coupled_projection(gap):
    from scivol._fit_credibility import _box_projection_bound

    # Project (-10, 1) onto x>=-gap, y<=x. For gap<4.5 the solution is
    # (-gap,-gap); otherwise it is (-4.5,-4.5). The second constraint can
    # increase the norm, so the box-only projection is not a valid shortcut.
    actual = min(gap, 4.5)*np.sqrt(2.)
    bound = _box_projection_bound(
        np.array([10., -1.]), np.array([-gap, -np.inf]), np.full(2, np.inf))
    assert bound >= actual


@pytest.mark.parametrize("gap", [0.0, 0.1, 10.0])
def test_projection_box_certificate_remains_bound_with_coupled_constraints(gap):
    # Projection of (-10, 1) onto x>=0, y<=x+gap is (0,min(1,gap)).
    bound = projected_stationarity_bound(
        np.array([10.0, -1.0]),
        np.array([0.0, -np.inf]),
        np.full(2, np.inf),
        np.array([[1.0, -1.0]]),
        np.array([gap]),
    )
    assert min(1.0, gap) <= bound <= 1.000000000001


def test_projection_exactly_supported_gradient_has_zero_residual():
    bound = projected_stationarity_bound(
        np.array([1e300, -1e300]),
        np.array([0.0, -np.inf]),
        np.array([np.inf, 0.0]),
        np.array([[1.0, 1.0]]),
        np.array([0.0]),
    )
    assert bound == 0.0


@pytest.mark.parametrize("family", range(4))
def test_forecast_does_not_floor_valid_small_variances(family):
    from scivol._evaluation import FilteredState, forecast_from_state
    from scivol.roles import Role

    e, h = 2e-10, 1e-18
    if family == 0:
        theta, expected = [1e-20, 0.1, 0.8], 1e-20 + 0.1 * e**2 + 0.8 * h
    elif family == 1:
        theta, expected = [1e-20, 0.1, 0.1, 0.8], 1e-20 + 0.1 * e**2 + 0.8 * h
    elif family == 2:
        theta = [-5.0, 0.1, -0.05, 0.8]
        z = e / np.sqrt(h)
        expected = np.exp(
            -5 + 0.1 * (abs(z) - np.sqrt(2 / np.pi)) - 0.05 * z + 0.8 * np.log(h)
        )
    else:
        theta = [1e-14, 0.1, 0.1, 0.8, 1.3]
        expected = (1e-14 + 0.1 * (abs(e) - 0.1 * e) ** 1.3 + 0.8 * h**0.65) ** (
            2 / 1.3
        )
    spec = FAMILIES[family]() + Normal()
    spec.get_component(Role.VOLATILITY).unpack(np.array(theta))
    state = FilteredState(
        np.array([e]),
        np.zeros(1),
        np.array([e]),
        np.array([h]),
        0.0,
        None,
        None,
        "Normal",
    )
    forecast = forecast_from_state(spec, state, 1)
    assert forecast.residual_variance[0] < 1e-12
    np.testing.assert_allclose(
        forecast.residual_variance[0], expected, rtol=2e-14, atol=0
    )


@pytest.mark.parametrize("orders", ((1, 1), (4, 3)))
@pytest.mark.parametrize("fixed", (False, True))
def test_joint_egarch_student_terminal_collapse_preserves_the_likelihood(orders, fixed):
    from scipy.special import gammaln
    from scivol import EGARCH, StudentT

    data = np.array([1.0, 1.0, 0.0])
    x = np.array([[-0.1, 0.3], [0.4, -0.2], [0.6, 0.8]])
    model = JointMeanModel(ARX(1) + EGARCH(*orders) + StudentT(), data, x, fixed=fixed)
    theta = np.zeros(model.size)
    nu = theta[-1] = 8.0
    theta[model.columns + 1] = -1.0
    log_constant = (
        gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    )
    moment = 2 * np.exp(log_constant) * (nu - 2) / (nu - 1)
    values = []
    for omega in (-6.0, -8.0, -10.0):
        theta[model.columns] = omega
        # The final residual is exactly zero; its log-scale term still contributes.
        log_variance = np.array([
            np.log(model.h0),
            omega - (1 / np.sqrt(model.h0) - moment),
            0.0,
        ])
        log_variance[2] = omega - (np.exp(-log_variance[1] / 2) - moment)
        expected = np.sum(
            0.5 * log_variance - log_constant
            + 0.5 * (nu + 1) * np.log1p(data**2 * np.exp(-log_variance) / (nu - 2))
        )
        value, _, _, residuals, variances = model.evaluate(theta, 1, paths=True)
        full = model.evaluate(theta, 7)
        assert model.assess(theta, None)[0]
        np.testing.assert_array_equal(residuals, data)
        np.testing.assert_allclose(np.log(variances), log_variance, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose([value, full[0]], expected, rtol=2e-14, atol=2e-14)
        assert np.isfinite(full[1]).all() and np.isfinite(full[2]).all()
        values.append(value)
    assert values[2] < values[1] < values[0]
    assert 0 < variances[-1] < 1e-80


@pytest.mark.parametrize("har", (False, True))
@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_joint_aparch_forecast_skips_unrepresentable_inactive_shocks(har, p, q):
    from scivol import APARCH

    data = np.zeros(128)
    data[-1] = 1e4
    x = np.empty((len(data), 0))
    spec = (HARX([1, 3]) if har else ARX(0)) + APARCH(p, q) + Normal()
    mean = np.zeros(3 if har else 1)
    omega, delta = 1.0, 100.0
    beta = np.full(q, 0.5 / q)
    theta = np.r_[mean, omega, np.zeros(2 * p), beta, delta]
    fixed = spec.fix(data, theta, x=x)
    assert np.isfinite(fixed.log_likelihood)
    assert fixed.resid[-1] == data[-1]

    powers = list(fixed.sigma2[-q:] ** (delta / 2))
    expected = []
    for _ in range(4):
        power = omega + sum(beta[j] * powers[-1 - j] for j in range(q))
        expected.append(power ** (2 / delta))
        powers.append(power)
    with np.errstate(over="raise", invalid="raise"):
        forecast = fixed.forecast(4)
    np.testing.assert_allclose(forecast.residual_variance, expected, rtol=2e-14)


def test_aparch_forecast_retains_tiny_active_shocks_at_their_original_lags():
    from scivol import APARCH
    from scivol._evaluation import FilteredState, forecast_from_state
    from scivol.roles import Role

    spec = APARCH(2, 1) + Normal()
    spec.get_component(Role.VOLATILITY).unpack(
        np.array([1.0, 0.0, 1e-300, 0.0, 0.0, 0.0, 2.0])
    )
    residuals = np.array([1e150, 1e160])
    state = FilteredState(
        residuals,
        np.zeros(2),
        residuals,
        np.ones(2),
        0.0,
        None,
        None,
        "Normal",
    )
    with np.errstate(over="raise", invalid="raise"):
        forecast = forecast_from_state(spec, state, 1)
    np.testing.assert_allclose(forecast.residual_variance, [2.0], rtol=2e-14)


def test_egarch_second_forecast_uses_the_first_future_innovation():
    from scivol._evaluation import FilteredState, forecast_from_state
    from scivol.roles import Role

    spec = FAMILIES[2]() + Normal()
    omega, alpha, gamma, beta = -0.1, 0.25, -0.15, 0.6
    spec.get_component(Role.VOLATILITY).unpack(np.array([omega, alpha, gamma, beta]))
    e, h = 0.3, 1.7
    state = FilteredState(
        np.array([e]),
        np.zeros(1),
        np.array([e]),
        np.array([h]),
        0.0,
        None,
        None,
        "Normal",
    )
    first = (
        omega
        + alpha * (abs(e / np.sqrt(h)) - np.sqrt(2 / np.pi))
        + gamma * e / np.sqrt(h)
        + beta * np.log(h)
    )
    draws = np.random.default_rng(0).standard_normal(4000)
    expected = np.mean(
        np.exp(
            omega
            + alpha * (abs(draws) - np.sqrt(2 / np.pi))
            + gamma * draws
            + beta * first
        )
    )
    np.testing.assert_allclose(
        forecast_from_state(spec, state, 2).residual_variance,
        [np.exp(first), expected],
        rtol=2e-14,
    )


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", (0, 7))
def test_public_filter_and_fix_use_the_joint_jet(family, density):
    from scivol._validation import prepare_series_input

    model, x = prepared(family, density)
    theta = model.initialize()
    sample = prepare_series_input(model.spec, model.data, x=x)
    evaluated = JointMeanModel(model.spec, sample.data, sample.x)
    expected = evaluated.evaluate(theta, 7, paths=True)
    actual = model.spec.filter(model.data, theta, x=x)
    np.testing.assert_allclose(actual.log_likelihood, -expected[0], rtol=1e-12)
    np.testing.assert_allclose(
        model.spec.score(model.data, theta, x=x), -expected[1], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        model.spec.hessian(model.data, theta, x=x), -expected[2], rtol=1e-12, atol=1e-12
    )
    fixed = model.spec.fix(model.data, theta, x=x)
    np.testing.assert_allclose(fixed.sigma2, expected[-1], rtol=1e-12)


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("density", (0, 7))
@pytest.mark.parametrize("har", (False, True))
def test_joint_public_preparation_and_warm_start_match_the_direct_engine(
    family, density, har
):
    pd = pytest.importorskip("pandas")

    original, x = prepared(family, density, 3, 2, n=96, har=har)
    index = pd.date_range("2024-01-01", periods=len(original.data), freq="D")
    data = pd.Series(original.data, index=index, name="observations")
    hold_back, scale = 5, 2.0
    sample = np.asarray(data)[hold_back:] * scale
    regressors = x[hold_back:]
    model = JointMeanModel(original.spec, sample, regressors)
    initial = model.initialize()
    saved_initial = initial.copy()
    options = dict(initial_params=initial, maxiter=1, log_mode=False)
    expected = fit_joint_mean_spec(model.spec, sample, x=regressors, **options)
    actual = original.spec.fit(
        data, x=x, hold_back=hold_back, scale=scale, solver="native", **options
    )

    np.testing.assert_array_equal(initial, saved_initial)
    np.testing.assert_array_equal(actual.params, expected.params)
    np.testing.assert_array_equal(actual.data, sample)
    np.testing.assert_array_equal(actual._fit_x, regressors)
    np.testing.assert_array_equal(actual._index, index[hold_back:])
    assert actual.log_likelihood == expected.log_likelihood
    assert actual.fit_info.original_n_obs == len(data)
    assert actual.fit_info.effective_n_obs == len(sample)
    assert actual.fit_info.requested_hold_back == hold_back
    assert actual.fit_info.effective_hold_back == hold_back
    assert actual.fit_info.scale == scale
    assert actual.fit_info.maxiter == 1

    state = original.spec.filter(
        data, actual.params, x=x, hold_back=hold_back, scale=scale
    )
    fixed = original.spec.fix(
        data, actual.params, x=x, hold_back=hold_back, scale=scale
    )
    filtered = model.evaluate(actual.params, 1, paths=True)
    assert state.log_likelihood == -filtered[0]
    np.testing.assert_array_equal(state.residuals, filtered[3])
    np.testing.assert_array_equal(state.sigma2, filtered[4])
    np.testing.assert_array_equal(fixed.sigma2, state.sigma2)
    np.testing.assert_array_equal(fixed._index, index[hold_back:])
    # Filtering requests F; fitted inference requests F|G|H with different reductions.
    np.testing.assert_allclose(state.log_likelihood, actual.log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(state.residuals, actual.resid, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.sigma2, actual.sigma2, rtol=1e-12)
    np.testing.assert_array_equal(
        original.spec.score(data, actual.params, x=x, hold_back=hold_back, scale=scale),
        -model.evaluate(actual.params, 2)[1],
    )
    np.testing.assert_array_equal(
        original.spec.hessian(
            data, actual.params, x=x, hold_back=hold_back, scale=scale
        ),
        -model.evaluate(actual.params, 4)[2],
    )


@pytest.mark.parametrize("family", range(4))
def test_joint_normal_cross_blocks_against_independent_ad(family):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    model, _ = prepared(family, p=3, q=2, n=12)
    theta = model.initialize()
    theta[: model.columns] = 0.03
    m, p, q = model.columns, model.p_vol, model.q_vol

    def objective(parameters):
        residuals = (
            jnp.asarray(model.data) - jnp.asarray(model.features) @ parameters[:m]
        )
        alpha = parameters[m + 1 : m + 1 + p]
        gamma = parameters[m + 1 + p : m + 1 + 2 * p] if family else jnp.zeros(p)
        beta_start = m + 1 + p * (1 if family == 0 else 2)
        beta = parameters[beta_start : beta_start + q]
        delta = parameters[beta_start + q] if family == 3 else 2.0
        variances = [jnp.asarray(model.h0)]
        histories = [jnp.log(model.h0) if family == 2 else model.h0 ** (delta / 2)]
        for t in range(1, len(model.data)):
            value = parameters[m]
            for lag in range(min(p, t)):
                e = residuals[t - 1 - lag]
                if family == 2:
                    z = e / jnp.sqrt(variances[t - 1 - lag])
                    value += (
                        alpha[lag] * (jnp.abs(z) - jnp.sqrt(2 / jnp.pi))
                        + gamma[lag] * z
                    )
                elif family == 3:
                    value += alpha[lag] * (jnp.abs(e) - gamma[lag] * e) ** delta
                else:
                    value += (alpha[lag] + gamma[lag] * (e < 0)) * e**2
            for lag in range(min(q, t)):
                value += beta[lag] * histories[t - 1 - lag]
            histories.append(value)
            variances.append(jnp.exp(value) if family == 2 else value ** (2 / delta))
        variance = jnp.stack(variances)
        return 0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance) + residuals**2 / variance)

    value, gradient, hessian, *_ = model.evaluate(theta, 7)
    np.testing.assert_allclose(value, objective(theta), rtol=2e-12)
    np.testing.assert_allclose(
        gradient, jax.grad(objective)(theta), rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        hessian, jax.hessian(objective)(theta), rtol=2e-8, atol=2e-8
    )
    assert np.max(np.abs(hessian[:m, m:])) > 0.1
