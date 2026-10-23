"""Free-chart Python ABI contracts using mocks, without a compiler or promotion."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scivol import _aparch_fit_adapters as adapters
from scivol import _aparch_fit_policy as policy
from scivol import _native_fit as native


SHAPES = ((), (7.0,), (7.0, .23), (1.6,), (.12, 1.5, .4),
          (.12, 1.5), (.12, 1.5), (.12, 2.0))


def model_for(density=0, orders=(1, 1, 1, 1)):
    ar, ma, p, q = orders
    mean, volatility = 1 + ar + ma, 2 + 2 * p + q
    return native.NativeModel("arma_aparch", "free-chart-contract", ar, ma, p, q,
        density, mean + volatility + len(SHAPES[density]), mean, volatility, max(orders), True)


def sample(model):
    theta = np.ascontiguousarray(np.r_[.04, np.linspace(.09, -.06, model.p_ar),
        np.linspace(-.08, .04, model.q_ma), .12, np.full(model.p_vol, .06 / model.p_vol),
        np.linspace(-.16, .13, model.p_vol), np.full(model.q_vol, .6 / model.q_vol),
        1.4, SHAPES[model.density_code]])
    data = np.ascontiguousarray(np.random.default_rng(716).normal(.1, .8, 13))
    return theta, data


def request_for(model, *, theta=None, budget=3):
    if theta is None:
        theta, _ = sample(model)
    return policy.PhaseRequest("native-baseline-free-chart", "complementary",
        policy.PhaseStart(theta, np.linspace(-.4, .8, model.n_params)),
        1e-10, budget, tuple(range(model.n_params)))


def pointer_array(address, count, dtype=ctypes.c_double):
    return np.ctypeslib.as_array((dtype * count).from_address(address))


def mock_provider(model, *, strategy="free_chart", k=2, status=0, stop=0,
                  unchanged=False, identical=False, missing_gradient=False,
                  infeasible_slot=None, mutate=False, summary_change=None, output_change=None):
    calls, assessments, workspace_calls, allocations = [], [], [], []
    _, data = sample(model)
    expected_data = data.copy()
    free_chart = strategy == "free_chart"
    expected_specification = (model.p_ar, model.q_ma, model.p_vol, model.q_vol,
                              model.density_code, k, True) + ((1,) if free_chart else ())

    def workspace(*specification):
        workspace_calls.append(specification)
        assert specification == expected_specification
        return 512 if free_chart else 256

    def allocate(size):
        allocations.append(size)
        return native._aligned_workspace(size)

    def phase(*args):
        assert len(args) == (28 if free_chart else 27)
        (incoming, width, observations, n, h0, ar, ma, p, q, density, rank, steps,
         arena, arena_size, anchors, anchor_n, columns, column_n,
         continuation, continuation_n, best, best_n, summary, summary_n,
         trace, trace_n, runtime) = args[:27]
        assert (ar, ma, p, q, density, rank, runtime) == expected_specification[:7]
        assert type(runtime) is bool and runtime
        if free_chart:
            assert type(args[27]) is int and args[27] == 1
        assert steps == 3 and width == continuation_n == best_n == model.n_params
        assert anchor_n == column_n == k and summary_n == (16 if free_chart else 12)
        assert trace_n == steps * (12 + 2 * width) and h0 == 1.0
        assert arena_size == (512 if free_chart else 256) and arena % 16 == 0
        buffers = ((incoming, width * 8), (observations, n * 8), (arena, arena_size),
                   (anchors, anchor_n * np.dtype(np.uintp).itemsize),
                   (columns, column_n * np.dtype(np.uintp).itemsize),
                   (continuation, width * 8), (best, width * 8),
                   (summary, summary_n * 8), (trace, trace_n * 8))
        for index, (address, size) in enumerate(buffers):
            assert address % 8 == 0
            assert all(address + size <= other or other + length <= address
                       for other, length in buffers[:index])
        np.testing.assert_array_equal(pointer_array(observations, n), expected_data)
        theta = pointer_array(incoming, width).copy()
        calls.append(args)
        if mutate:
            pointer_array(incoming, width)[0] += 1.0
        accepted, retained = pointer_array(continuation, width), pointer_array(best, width)
        accepted[:], retained[:] = theta, theta
        if not unchanged:
            accepted[0], retained[0] = .31, np.nextafter(.33, 1.0)
        if identical:
            retained[:] = accepted
        pointer_array(anchors, k, ctypes.c_size_t)[:] = np.arange(model.maximum_lag, model.maximum_lag + k)
        pointer_array(columns, k, ctypes.c_size_t)[:] = np.arange(k)
        raw = pointer_array(summary, summary_n)
        if status != 0:
            pointer_array(anchors, k, ctypes.c_size_t)[:] = np.iinfo(np.uintp).max
            pointer_array(columns, k, ctypes.c_size_t)[:] = np.iinfo(np.uintp).max
            pointer_array(trace, trace_n)[:] = 123.0
            raw[:] = np.nan
            if status == -2:
                raw[:12] = [-2, np.inf, np.inf, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                if free_chart:
                    raw[12:] = [0, 0, 0, 1]
        else:
            raw[:12] = [stop, 10.0 if unchanged or identical else 7.0, 10.0,
                        2, 1, 11, 3, 2, 6, 9, 7, 2]
            if free_chart:
                raw[12:] = [5, 1, 1, 1]
            pointer_array(trace, trace_n)[:2 * (12 + 2 * width)] = np.arange(2 * (12 + 2 * width))
        if summary_change:
            summary_change(raw)
        if output_change:
            output_change(accepted, retained, pointer_array(anchors, k, ctypes.c_size_t),
                          pointer_array(columns, k, ctypes.c_size_t))
        return status

    def assess(model, observations, theta, solve, accuracy, *, coordinates, requests):
        assert solve.mode == -105 and solve.kkt_residual == np.inf
        assert solve.maximum_violation == np.inf and solve.iterations == 0
        assert accuracy == 1e-10
        value = 7.0 if theta[0] == np.nextafter(.33, 1.0) else 10.0
        assessments.append((theta.copy(), coordinates, solve))
        requests["F|G"] = requests.get("F|G", 0) + 1
        if missing_gradient:
            for mask in ("F", "G"):
                requests[mask] = requests.get(mask, 0) + 1
        feasible = not (infeasible_slot == "best" and value == 7.0
                        or infeasible_slot == "continuation" and value == 10.0)
        return policy.PhasePoint(theta, value,
            np.full(model.n_params, np.nan if missing_gradient else 0.0), solve,
            native.ThetaAssessment(0.0, np.inf, feasible, False), coordinates=coordinates)

    core = SimpleNamespace(_arma_aparch_multi_phase_workspace_bytes=workspace,
                           _arma_aparch_multi_phase_run=phase)
    provider = SimpleNamespace(_native_core=core, _initial_variance=lambda _: 1.0,
        _aligned_workspace=allocate, NativeSolve=native.NativeSolve,
        _assess_aparch_phase_point=assess,
        _coordinates_to_theta=lambda *a: pytest.fail("Charts return actual theta, not z."),
        _theta_to_coordinates=lambda *a: pytest.fail("Do not reconstruct original z."))
    return SimpleNamespace(data=data, provider=provider, calls=calls, assessments=assessments,
                           workspace_calls=workspace_calls, allocations=allocations)


def callback_for(model, mock, **options):
    return adapters.NativeMultiPhase(model, mock.data, mock.provider, k=2, maximum_steps=3,
        qualified_densities=(model.density_code,), force_runtime=True, **options)


@pytest.mark.parametrize("strategy", [None, "held_residuals", "free_chart"])
@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (3, 2, 2, 1)])
def test_old_and_opt_in_abis_accounting_and_independent_slots(strategy, density, orders):
    effective_strategy = strategy or "held_residuals"
    model = model_for(density, orders)
    mock = mock_provider(model, strategy=effective_strategy)
    callback = callback_for(model, mock, **({} if strategy is None else {"strategy": strategy}))
    request = request_for(model)
    original_data = mock.data.copy()
    mock.data[:] = 999.0
    result = callback(request)
    free_chart = effective_strategy == "free_chart"
    assert callback.strategy == effective_strategy
    assert len(mock.workspace_calls[0]) == (8 if free_chart else 7)
    assert len(mock.calls[0]) == (28 if free_chart else 27)
    assert mock.calls[0][23] == (16 if free_chart else 12)
    assert result.work == policy.PhaseWork(2, 11, 8 if free_chart else 3, 3, {"F|G": 2})
    assert result.evidence["requests"] == ({"F": 11, "G|H": 3, "G": 5}
                                            if free_chart else {"F": 11, "G|H": 3})
    assert result.evidence["strategy"] == effective_strategy
    assert result.evidence["certificate"] == "none"
    assert result.evidence["selection_attempts"] == (2 if free_chart else 1)
    assert result.evidence["summary"]["gradients"] == (5 if free_chart else 0)
    assert result.evidence["summary"]["best_restarts"] == int(free_chart)
    assert result.continuation.value == 10.0 and result.points[-1].value == 7.0
    assert result.continuation is not result.points[-1]
    assert result.points[-1].theta[0] == np.nextafter(.33, 1.0)
    assert all(point.coordinates is None and not point.certified for point in result.points)
    assert all(point.solve.mode == -105 for point in result.points)
    assert all(point.solve.scaled_value == point.value / model.scored_observations(original_data.size)
               for point in result.points)
    assert len(result.evidence["trace"]) == 2
    assert all(len(row) == 12 + 2 * model.n_params for row in result.evidence["trace"])
    np.testing.assert_array_equal(callback.data, original_data)
    assert not callback.data.flags.writeable
    callback(request)
    assert len(mock.workspace_calls) == len(mock.allocations) == 1
    assert mock.calls[0][12] == mock.calls[1][12] and len(mock.assessments) == 4


@pytest.mark.parametrize("unchanged,identical", [(True, False), (False, True)])
def test_identical_slots_share_assessment_and_only_unchanged_theta_keeps_original_z(unchanged, identical):
    model = model_for()
    mock = mock_provider(model, unchanged=unchanged, identical=identical)
    request = request_for(model)
    result = callback_for(model, mock, strategy="free_chart")(request)
    assert len(result.points) == len(mock.assessments) == 1
    assert result.continuation is result.points[0]
    assert result.work.validation_requests == {"F|G": 1}
    if unchanged:
        assert result.points[0].coordinates.tobytes() == request.start.coordinates.tobytes()
    else:
        assert result.points[0].coordinates is None


def test_signed_zero_change_cannot_recover_original_z_from_numeric_equality():
    model = model_for()
    theta, _ = sample(model)
    theta[1] = -0.0
    mock = mock_provider(model, unchanged=True,
        output_change=lambda continuation, best, anchors, columns: best.__setitem__(1, 0.0))
    result = callback_for(model, mock, strategy="free_chart")(request_for(model, theta=theta))
    assert len(mock.assessments) == len(result.points) == 2
    assert result.continuation.coordinates is not None and result.points[-1].coordinates is None


@pytest.mark.parametrize("stop", [0, 1, -3, -4, -5, -6, -7])
def test_all_stops_preserve_best_without_gradient_or_false_certificate(stop):
    model = model_for()
    mock = mock_provider(model, stop=stop, missing_gradient=True)
    request = request_for(model)
    baseline = policy.PhasePoint(request.start.theta, 11.0, np.zeros(model.n_params),
        native.NativeSolve(0, 11.0 / 12, 0., 0., 1, 2, 1, 0, 0, 0),
        native.ThetaAssessment(0., 0., True, True), coordinates=request.start.coordinates,
        certified=True)
    selected = policy.run_aparch_controller(request.start,
        baseline=lambda _: policy.PhaseOutcome((baseline,), baseline, policy.PhaseWork(1, 2, 1, 0)),
        released=lambda _: pytest.fail("No power provider"),
        phases=policy.ControllerPhases(complementary=(("free-chart",
            callback_for(model, mock, strategy="free_chart")),), complementary_budget=3),
        baseline_budget=2, density_code=0, power_index=7, accuracy=1e-10)
    assert selected.point.value == 7.0 and not selected.point.can_continue
    assert selected.point.solve.mode == -105 and not selected.point.certified
    assert selected.diagnostics[-1]["mode"] == stop
    assert bool(selected.diagnostics[-1]["failure"]) == (stop < 0)
    assert selected.total_work() == policy.PhaseWork(3, 13, 9, 3, {"F|G": 2, "F": 2, "G": 2})
    assert selected.diagnostics[-1]["points"][-1]["certified"] is False


def test_initial_gradient_failure_retains_unchanged_theta_and_counts_attempted_G():
    model = model_for()
    raw = [-6, 10., 10., 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    mock = mock_provider(model, unchanged=True, missing_gradient=True,
                         summary_change=lambda summary: summary.__setitem__(slice(None), raw))
    request = request_for(model)
    result = callback_for(model, mock, strategy="free_chart")(request)
    assert result.mode == -6 and "gradient" in result.failure
    assert result.continuation is result.points[0] and len(result.points) == 1
    assert result.continuation.theta.tobytes() == request.start.theta.tobytes()
    assert result.continuation.coordinates.tobytes() == request.start.coordinates.tobytes()
    assert result.work == policy.PhaseWork(0, 1, 1, 0, {"F|G": 1, "F": 1, "G": 1})
    assert result.evidence["trace"] == [] and not result.continuation.certified


@pytest.mark.parametrize("slot", ["best", "continuation"])
def test_canonical_feasibility_filters_each_slot_independently(slot):
    model = model_for()
    mock = mock_provider(model, infeasible_slot=slot)
    result = callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert len(mock.assessments) == 2 and len(result.points) == 1
    assert result.points[0].value == (10.0 if slot == "best" else 7.0)
    assert bool(result.failure) == (slot == "continuation")


@pytest.mark.parametrize("values", [0, 1])
def test_initial_unavailability_consumes_only_summary_not_poisoned_outputs(values):
    model = model_for()
    mock = mock_provider(model, status=-2,
                         summary_change=lambda summary: summary.__setitem__(5, values))
    result = callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert result.points == () and result.continuation is None and not mock.assessments
    assert result.mode == -2 and result.failure and result.work == policy.PhaseWork(0, values, 0, 0)
    assert result.evidence["summary"]["strategy"] == 1
    assert "anchors" not in result.evidence and "trace" not in result.evidence


@pytest.mark.parametrize("status", [-11, -12, -13, -14, -15])
def test_initial_selection_failure_does_not_read_or_fabricate_summary_work(status):
    model = model_for()
    mock = mock_provider(model, status=status)
    result = callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert result.mode == status and result.failure and not mock.assessments
    assert result.points == () and result.continuation is None and result.work == policy.PhaseWork()
    assert "summary" not in result.evidence and result.evidence["selection_attempts"] == 1


@pytest.mark.parametrize("status,error", [(-1, ValueError), (-6, RuntimeError), (-7, RuntimeError),
                                         (42, RuntimeError), (True, RuntimeError), (np.int64(0), RuntimeError)])
def test_invalid_completion_status_is_not_a_summary_stop(status, error):
    model = model_for()
    mock = mock_provider(model, status=status)
    with pytest.raises(error):
        callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert not mock.assessments


@pytest.mark.parametrize("field,value", [(0, -9), (1, np.inf), (1, np.nan), (1, 11.),
    (3, -1), (3, 4), (4, 3), (5, np.nan), (6, 1.5), (7, 2**53),
    (13, 2), (14, 3), (15, 0), (15, 2)] +
    [(field, value) for field in range(12, 16) for value in (-1, .5, np.nan, np.inf, 2**53)])
def test_malformed_summary_counts_and_strategy_fail_before_assessment(field, value):
    model = model_for()
    mock = mock_provider(model, summary_change=lambda summary: summary.__setitem__(field, value))
    with pytest.raises(RuntimeError):
        callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert not mock.assessments


@pytest.mark.parametrize("field,value", [(0, -6), (0, -7), (1, 10.), (2, np.nan), (5, 2)] +
    [(field, 1) for field in (3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14)] + [(15, 0)])
def test_initial_unavailability_rejects_extra_work_or_wrong_strategy(field, value):
    model = model_for()
    mock = mock_provider(model, status=-2,
                         summary_change=lambda summary: summary.__setitem__(field, value))
    with pytest.raises(RuntimeError):
        callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert not mock.assessments


@pytest.mark.parametrize("stop,field", [(-6, 12), (-7, 14)])
def test_new_failure_stops_require_their_actual_attempt_count(stop, field):
    model = model_for()
    mock = mock_provider(model, stop=stop,
                         summary_change=lambda summary: summary.__setitem__(field, 0))
    with pytest.raises(RuntimeError, match="attempted request"):
        callback_for(model, mock, strategy="free_chart")(request_for(model))
    assert not mock.assessments


@pytest.mark.parametrize("stop", [-6, -7])
def test_held_residuals_cannot_consume_free_chart_stop_semantics(stop):
    model = model_for()
    mock = mock_provider(model, strategy="held_residuals", stop=stop)
    with pytest.raises(RuntimeError, match="completed phase summary"):
        callback_for(model, mock)(request_for(model))


@pytest.mark.parametrize("shape", [(12,), (15,), (17,), (4, 4)])
def test_free_chart_parser_requires_exact_16_field_layout(shape):
    with pytest.raises(RuntimeError, match="summary layout"):
        adapters._parse_multi_phase_summary(np.zeros(shape), status=0, budget=3, strategy="free_chart")


@pytest.mark.parametrize("strategy", ["free-chart", "full", "", None, True, 1, []])
def test_unknown_strategy_fails_before_native_query_or_allocation(strategy):
    model = model_for()
    mock = mock_provider(model)
    with pytest.raises(ValueError, match="strategy"):
        callback_for(model, mock, strategy=strategy)
    assert not mock.workspace_calls and not mock.allocations and not mock.calls


def test_request_guards_keep_full_free_layout_and_explicit_budgets():
    model = model_for()
    mock = mock_provider(model)
    callback = callback_for(model, mock, strategy="free_chart")
    request = request_for(model)
    bad_requests = (replace(request, budget=0), replace(request, budget=4),
        replace(request, budget=True), replace(request, role="power"),
        replace(request, physical_power=1.4), replace(request, power_mode="fixed"),
        replace(request, free_parameters=tuple(range(1, model.n_params))),
        replace(request, start=policy.PhaseStart(np.zeros(model.n_params - 1))),
        replace(request, start=policy.PhaseStart(np.full(model.n_params, np.nan))))
    for bad in bad_requests:
        with pytest.raises(ValueError):
            callback(bad)
    assert not mock.calls


@pytest.mark.parametrize("fault", ["binding", "size", "overflow", "summary-overflow", "qualification",
                                   "data", "sample", "seed", "density", "rank", "budget", "runtime"])
def test_constructor_guards_fail_without_allocating(fault):
    model = model_for()
    mock = mock_provider(model)
    options = dict(k=2, maximum_steps=3, qualified_densities=(0,), force_runtime=True, strategy="free_chart")
    if fault == "binding":
        mock.provider._native_core._arma_aparch_multi_phase_run = None
    elif fault in ("size", "overflow", "summary-overflow"):
        width = model.n_params
        held_outputs = 2 * 2 * np.dtype(np.uintp).itemsize + (2 * width + 12 + 3 * (12 + 2 * width)) * 8
        size = 0 if fault == "size" else int(np.iinfo(np.intp).max)
        if fault == "summary-overflow":
            size -= held_outputs
        mock.provider._native_core._arma_aparch_multi_phase_workspace_bytes = lambda *args: size
    elif fault == "qualification":
        options["qualified_densities"] = ()
    elif fault == "data":
        mock.data = np.full((2, 2), .5)
    elif fault == "sample":
        mock.data = np.array([.5, .4])
    elif fault == "seed":
        mock.provider._initial_variance = lambda _: np.nan
    elif fault == "density":
        model = replace(model, density_code=True)
    elif fault == "rank":
        options["k"] = 3
    elif fault == "budget":
        options["maximum_steps"] = 0
    elif fault == "runtime":
        options["force_runtime"] = 1
    with pytest.raises(NotImplementedError if fault == "binding" else ValueError):
        adapters.NativeMultiPhase(model, mock.data, mock.provider, **options)
    assert not mock.allocations and not mock.calls


@pytest.mark.parametrize("fault", ["mutation", "nll", "theta", "anchors", "columns"])
def test_programming_errors_do_not_become_numeric_failure(fault):
    model = model_for()
    options = {}
    if fault == "mutation":
        options["mutate"] = True
    elif fault == "nll":
        options["summary_change"] = lambda summary: summary.__setitem__(1, 6.0)
    elif fault == "theta":
        options["output_change"] = lambda continuation, best, anchors, columns: best.__setitem__(0, np.nan)
    elif fault == "anchors":
        options["output_change"] = lambda continuation, best, anchors, columns: anchors.__setitem__(0, 0)
    else:
        options["output_change"] = lambda continuation, best, anchors, columns: columns.__setitem__(1, 0)
    mock = mock_provider(model, **options)
    with pytest.raises(RuntimeError):
        callback_for(model, mock, strategy="free_chart")(request_for(model))
