"""Public-boundary regressions for the joint linear-mean native integration."""

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
    assert feasible and violation == 0 and bound == np.inf
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
    assert all(p["constraint_curvature_reset"] == (family == 3) for p in phases)
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
        )

    expected = run()
    assert any(p["constraint_curvature_resets"] for p in expected.phases)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(4)]
        results = [future.result() for future in futures]
    for actual in results:
        np.testing.assert_array_equal(actual.x, expected.x)
        assert actual.fun == expected.fun and actual.phases == expected.phases


def test_egarch_phase_keeps_independent_gradient_evidence():
    model, _ = prepared(2)
    result = _solve_joint_phase(
        model,
        model.initialize(),
        transformed=False,
        accuracy=1e-12,
        maxiter=1,
        use_curvature=False,
    )
    assert result.jac is None


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


@pytest.mark.parametrize("family", (0, 1, 3))
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


@pytest.mark.parametrize("family", (0, 1, 3))
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


def test_joint_reuses_selected_point_curvature_for_inference():
    import time
    from scivol._joint_mean_fit import solve_joint, assemble_joint_fit

    model, _ = prepared(0, 1, n=400)
    optimized = solve_joint(
        model,
        model.initialize(),
        transformed=False,
        accuracy=1e-12,
        maxiter=1000,
        allow_chart_refinement=True,
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
    from benchmark_native_joint import prepare_native_request, prepare_legacy_request

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
        np.testing.assert_allclose(legacy_output, native_output, rtol=2e-6, atol=3e-6)


def test_promotion_benchmark_retains_native_evidence_when_reference_fails(monkeypatch):
    import benchmark_native_joint as benchmark
    from types import SimpleNamespace

    def fail_reference(*args, **kwargs):
        raise RuntimeError("reference sentinel")

    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: [])
    monkeypatch.setattr(
        benchmark, "get_routine", lambda *args: SimpleNamespace(fit=fail_reference)
    )
    result = benchmark.compare(0, 0, False, pairs=3, maxiter=1)
    assert "reference sentinel" in result["error"]
    assert np.isfinite(result["native_nll"])
    assert result["theta"]
    assert result["native_fit_info"]["phase_diagnostics"]


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
            requests.setattr(model, "evaluate", lambda *args: (123.0,))
        elif failure == "mean":
            from types import SimpleNamespace

            requests.setattr(model, "blocks", [SimpleNamespace(evaluate=reject)])
        else:
            requests.setattr(model, "constraints", reject)
        result = benchmark.independent_fit(model, initial, maxiter=1)
    phase = result.phases[0]
    assert phase["invalid_objective_requests"] == int(failure != "constraint")
    assert phase["invalid_constraint_requests"] == 2 * int(failure == "constraint")
    assert phase["mean_domain_rejections"] == int(failure == "mean")


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
    assert len(points) == 2
    np.testing.assert_array_equal(points[0], result["theta"])
    np.testing.assert_array_equal(points[1], result["reference_theta"])
    np.testing.assert_allclose(
        result["legacy_at_native"], result["native_nll"], rtol=1e-12
    )
    info = result["reference_fit_info"]
    assert info["jet_requests"]["masks"]["F|G"] > 0
    assert result["reference_terminated"] == info["optimizer_terminated"]


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


@pytest.mark.parametrize("density", (0, 7))
def test_egarch_reports_scalar_precision_and_serializes_evidence(density):
    model, x = prepared(2, density, n=60)
    result = fit_joint_mean_spec(model.spec, model.data, x=x, maxiter=2)
    expected = (
        "double-double primal state; binary64 derivatives"
        if density == 7
        else "binary64"
    )
    assert result.fit_info.scalar_precision == expected
    restored = FitMetadata(**result.fit_info.to_dict())
    assert restored.scalar_precision == expected
    assert restored.to_dict() == result.fit_info.to_dict()


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


def test_egarch_vg_refinement_has_limiting_not_smooth_stationarity():
    from fractions import Fraction
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    rng = np.random.default_rng(2785)
    data, x = rng.normal(size=400), rng.normal(size=(400, 2))
    model = JointMeanModel(ARX(1) + FAMILIES[2](1, 1) + DENSITIES[7](), data, x)
    theta = np.array(
        [
            0.020710762751471757,
            -0.06020333139468085,
            -0.04212794215375844,
            0.0941897416011332,
            -0.13385337831682317,
            -0.05068053603263031,
            -0.16784771939987161,
            0.009119991210401284,
            -0.1182976959610899,
            4.8114439489744925,
        ]
    )
    row = 212
    value, gradient, _, residuals, _ = model.evaluate(theta, 3, paths=True)
    assert residuals[row] != 0
    assert model.assess(theta, gradient)[1] > 1e-6
    args = (theta, model.data, model.features, model.h0, 1, 1, 2, 7)
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


@pytest.mark.parametrize("family", (0, 1))
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
    assert route(model.spec, **options) is (family < 2)
    for solver in ("legacy", "slsqp"):
        assert not route(model.spec, **{**options, "kwargs": {"solver": solver}})
    assert not route(model.spec, **{**options, "force_legacy_engine": True})
    assert not route(model.spec, **{**options, "method": "qmle"})
    assert not route(model.spec, **{**options, "has_auto_components": True})


@pytest.mark.parametrize("family", (0, 1))
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
    initial = model.initialize()
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


@pytest.mark.parametrize("density", (3, 4, 5, 6, 7))
def test_smooth_aparch_certifies_without_redundant_refinement(density):
    from scivol import APARCH

    rng = np.random.default_rng(2785)
    data = rng.normal(size=400)
    x = rng.normal(size=(400, 2))
    result = fit_joint_mean_spec(ARX(1) + APARCH() + DENSITIES[density](), data, x=x)
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
    # The unavailable full certificate must not claim the active-face shortcut.
    assert bound >= 1e200


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
