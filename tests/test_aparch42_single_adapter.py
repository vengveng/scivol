"""Single-anchor callback contracts, without default routing or fit promotion."""

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
    return native.NativeModel("arma_aparch", "single-anchor-contract", ar, ma, p, q,
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
    coordinates = np.linspace(-.4, .8, model.n_params)
    return policy.PhaseRequest("native-baseline-single-anchor", "complementary",
        policy.PhaseStart(theta, coordinates), 1e-10, budget, tuple(range(model.n_params)))


def pointer_array(address, count):
    return np.ctypeslib.as_array((ctypes.c_double * count).from_address(address))


def mock_provider(model, *, stop=0, unchanged=False, identical=False, missing_gradient=False,
                  infeasible_slot=None, mutate=False, raw_change=None, output_change=None):
    calls, assessments, workspace_calls = [], [], []
    _, data = sample(model)
    expected_data = data.copy()

    def workspace(*specification):
        workspace_calls.append(specification)
        assert specification == (model.p_ar, model.q_ma, model.p_vol, model.q_vol,
                                 model.density_code, True, True)
        return 256

    def refine(*args):
        assert len(args) == 21
        (incoming, width, observations, n, h0, ar, ma, p, q, density, steps, radius,
         arena, arena_size, continuation, continuation_n, best, best_n,
         runtime, joint, maximum_step) = args
        assert (ar, ma, p, q, density) == (model.p_ar, model.q_ma, model.p_vol,
                                        model.q_vol, model.density_code)
        assert steps == 3 and radius == 1024 and maximum_step == .1
        assert type(runtime) is bool and runtime and type(joint) is bool and joint
        assert width == continuation_n == best_n == model.n_params
        assert arena_size == 256 and arena % 16 == 0 and h0 == 1.0
        buffers = ((incoming, width * 8), (observations, n * 8), (arena, arena_size),
                   (continuation, width * 8), (best, width * 8))
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
        accepted[:] = theta
        retained[:] = theta
        if not unchanged:
            accepted[0], retained[0] = .31, np.nextafter(.33, 1.0)
        if identical:
            retained[:] = accepted
        # Invalid/unavailable calls deliberately poison otherwise finite slots.
        raw = ((stop, np.inf, np.inf, 0, 0, int(stop == -2), 0, 0, 0, 0)
               if stop in (-1, -2) else
               (stop, 10.0 if unchanged or identical else 7.0, 10.0, 2, 1, 11, 2, 12, 15, 3))
        if output_change:
            output_change(accepted, retained)
        return raw_change(raw) if raw_change else raw

    def assess(model, observations, theta, solve, accuracy, *, coordinates, requests):
        assert solve.mode == -105 and solve.kkt_residual == np.inf
        assert solve.maximum_violation == np.inf and solve.iterations == 0
        value = 7.0 if theta[0] == np.nextafter(.33, 1.0) else 10.0
        assessments.append((theta.copy(), coordinates, solve))
        requests["F|G"] = requests.get("F|G", 0) + 1
        if missing_gradient:
            for mask in ("F", "G"):
                requests[mask] = requests.get(mask, 0) + 1
        feasible = not (infeasible_slot == "best" and value == 7.0
                        or infeasible_slot == "continuation" and value == 10.0)
        assessment = native.ThetaAssessment(0.0, np.inf, feasible, False)
        return policy.PhasePoint(theta, value,
            np.full(model.n_params, np.nan if missing_gradient else 0.0),
            replace(solve, scaled_value=value / model.scored_observations(observations.size)),
            assessment, coordinates=coordinates)

    core = SimpleNamespace(_arma_aparch_mean_refinement_workspace_bytes=workspace,
                           _arma_aparch_mean_refine=refine)
    provider = SimpleNamespace(_native_core=core, _initial_variance=lambda _: 1.0,
        _aligned_workspace=native._aligned_workspace, NativeSolve=native.NativeSolve,
        _assess_aparch_phase_point=assess,
        _refine_aparch_mean_if_needed=lambda *a, **k: pytest.fail("Do not reuse the Normal-only adapter."),
        _coordinates_to_theta=lambda *a: pytest.fail("Do not map chart theta."),
        _theta_to_coordinates=lambda *a: pytest.fail("Do not reconstruct original z."))
    return data, provider, calls, assessments, workspace_calls


def callback_for(model, data, provider, **options):
    return adapters.NativeSingleAnchor(model, data, provider, maximum_steps=3,
        qualified_densities=(model.density_code,), force_runtime=True, **options)


@pytest.mark.parametrize("density", range(8))
def test_actual_21arg_joint_contract_and_separate_best_continuation(density):
    model = model_for(density)
    data, provider, calls, assessments, workspace_calls = mock_provider(model)
    callback = callback_for(model, data, provider)
    request = request_for(model)
    original = data.copy()
    data[:] = 999.0
    result = callback(request)
    assert result.continuation.value == 10.0 and result.points[-1].value == 7.0
    assert result.points[-1].theta[0] == np.nextafter(.33, 1.0)
    assert result.points[-1].coordinates is None and not result.points[-1].certified
    assert result.points[-1].solve.mode == -105
    assert result.work == policy.PhaseWork(2, 13, 2, 2, {"F|G": 2})
    assert result.evidence["requests"] == {"F": 11, "F|G|H": 2}
    assert result.evidence["summary"]["rejected"] == 3
    assert result.evidence["certificate"] == "none"
    assert len(assessments) == 2 and len(workspace_calls) == 1
    np.testing.assert_array_equal(callback.data, original)
    assert not callback.data.flags.writeable
    callback(request)
    assert calls[0][12] == calls[1][12] and len(workspace_calls) == 1


@pytest.mark.parametrize("unchanged,identical", [(True, False), (False, True)])
def test_identical_theta_slots_share_assessment_and_only_unchanged_preserves_z(unchanged, identical):
    model = model_for()
    data, provider, _, assessments, _ = mock_provider(model, unchanged=unchanged, identical=identical)
    request = request_for(model)
    result = callback_for(model, data, provider)(request)
    assert len(result.points) == len(assessments) == 1
    assert result.continuation is result.points[0]
    assert result.work.validation_requests == {"F|G": 1}
    if unchanged:
        assert result.points[0].coordinates.tobytes() == request.start.coordinates.tobytes()
    else:
        assert result.points[0].coordinates is None


def test_numeric_equality_does_not_preserve_z_after_signed_zero_change():
    model = model_for()
    theta, _ = sample(model)
    theta[1] = -0.0
    data, provider, _, assessments, _ = mock_provider(model, unchanged=True,
        output_change=lambda continuation, best: best.__setitem__(1, 0.0))
    result = callback_for(model, data, provider)(request_for(model, theta=theta))
    assert len(assessments) == 2 and len(result.points) == 2
    assert result.continuation.coordinates is not None and result.points[-1].coordinates is None


def test_failed_phase_retains_finite_F_without_G_and_counts_every_attempt():
    model = model_for()
    data, provider, _, _, _ = mock_provider(model, stop=-3, missing_gradient=True)
    request = request_for(model)
    baseline = policy.PhasePoint(request.start.theta, 11.0, np.zeros(model.n_params),
        native.NativeSolve(9, 11.0 / 12, 0., 1., 1, 2, 1, 0, 0, 0),
        native.ThetaAssessment(0., 1., True, False), coordinates=request.start.coordinates)
    result = policy.run_aparch_controller(request.start,
        baseline=lambda _: policy.PhaseOutcome((baseline,), baseline, policy.PhaseWork(1, 2, 1, 0)),
        released=lambda _: pytest.fail("No power provider"),
        phases=policy.ControllerPhases(complementary=(("single", callback_for(model, data, provider)),),
                                       complementary_budget=3),
        baseline_budget=2, density_code=0, power_index=7, accuracy=1e-10)
    assert result.point.value == 7.0 and not result.point.can_continue and not result.point.certified
    assert result.point.solve.mode == -105 and result.diagnostics[-1]["failure"]
    assert result.total_work() == policy.PhaseWork(3, 15, 3, 2, {"F|G": 2, "F": 2, "G": 2})


def test_initial_unavailability_counts_F_but_ignores_poisoned_theta():
    model = model_for()
    data, provider, _, assessments, _ = mock_provider(model, stop=-2)
    result = callback_for(model, data, provider)(request_for(model))
    assert result.mode == -2 and result.failure and not assessments
    assert result.points == () and result.continuation is None
    assert result.work == policy.PhaseWork(0, 1, 0, 0)


@pytest.mark.parametrize("slot", ["best", "continuation"])
def test_every_full_theta_gets_feasibility_assessment(slot):
    model = model_for()
    data, provider, _, assessments, _ = mock_provider(model, infeasible_slot=slot)
    result = callback_for(model, data, provider)(request_for(model))
    assert len(assessments) == 2 and len(result.points) == 1
    assert result.points[0].value == (10.0 if slot == "best" else 7.0)
    assert bool(result.failure) == (slot == "continuation")


@pytest.mark.parametrize("change", [
    lambda raw: raw[:-1], lambda raw: list(raw),
    lambda raw: (True, *raw[1:]), lambda raw: (99, *raw[1:]),
    lambda raw: (*raw[:3], -1, *raw[4:]),
    lambda raw: (*raw[:3], 4, *raw[4:]),
    lambda raw: (*raw[:4], 3, *raw[5:]),
    lambda raw: (*raw[:5], 100, *raw[6:]),
    lambda raw: (*raw[:6], 1.5, *raw[7:]),
    lambda raw: (*raw[:9], 16),
    lambda raw: (raw[0], np.nan, *raw[2:]),
    lambda raw: (raw[0], 11.0, *raw[2:]),
])
def test_malformed_native_results_are_not_numeric_unavailability(change):
    model = model_for()
    data, provider, _, assessments, _ = mock_provider(model, raw_change=change)
    with pytest.raises(RuntimeError):
        callback_for(model, data, provider)(request_for(model))
    assert not assessments


def test_invalid_native_stop_mutation_and_canonical_disagreement_propagate():
    model = model_for()
    for options, exception, message in (
        ({"stop": -1}, ValueError, "rejected"),
        ({"mutate": True}, RuntimeError, "mutated"),
        ({"raw_change": lambda raw: (raw[0], 6.0, *raw[2:])}, RuntimeError, "canonical theta NLL"),
        ({"output_change": lambda a, b: b.__setitem__(0, np.nan)}, RuntimeError, "finite theta"),
    ):
        data, provider, _, _, _ = mock_provider(model, **options)
        with pytest.raises(exception, match=message):
            callback_for(model, data, provider)(request_for(model))


@pytest.mark.parametrize("field,value", [("budget", 4), ("budget", 0), ("budget", True),
    ("accuracy", 0.0), ("accuracy", np.nan), ("role", "power"),
    ("free_parameters", (0, 1)), ("physical_power", 1.4), ("power_mode", "fixed-coordinate-slsqp")])
def test_request_guards_reject_before_native_call(field, value):
    model = model_for()
    data, provider, calls, _, _ = mock_provider(model)
    with pytest.raises(ValueError):
        callback_for(model, data, provider)(replace(request_for(model), **{field: value}))
    assert not calls


@pytest.mark.parametrize("options", [
    {"maximum_steps": 0}, {"maximum_steps": True}, {"maximum_steps": 2**64},
    {"radius": -1}, {"radius": True}, {"radius": 2**64},
    {"maximum_step": 0.0}, {"maximum_step": np.nan}, {"maximum_step": np.inf},
    {"maximum_step": True}, {"maximum_step": 1e308}, {"force_runtime": 1},
    {"qualified_densities": ()}, {"qualified_densities": (True,)},
])
def test_constructor_guards_and_overflow_precede_allocation(options):
    model = model_for()
    data, provider, _, _, _ = mock_provider(model)
    provider._aligned_workspace = lambda _: pytest.fail("Invalid plan must not allocate")
    arguments = dict(maximum_steps=3, qualified_densities=(0,), force_runtime=True)
    arguments.update(options)
    with pytest.raises(ValueError):
        adapters.NativeSingleAnchor(model, data, provider, **arguments)


@pytest.mark.parametrize("data", [np.array([[1.0, 2.0]]), np.array([np.nan, 1.0]), np.array([1.0])])
def test_invalid_observations_precede_allocation(data):
    model = model_for()
    _, provider, _, _, _ = mock_provider(model)
    provider._aligned_workspace = lambda _: pytest.fail("Invalid sample must not allocate")
    with pytest.raises(ValueError):
        callback_for(model, data, provider)


def test_missing_binding_and_invalid_workspace_are_explicit():
    model = model_for()
    data, provider, _, _, _ = mock_provider(model)
    provider._native_core._arma_aparch_mean_refine = None
    with pytest.raises(NotImplementedError):
        callback_for(model, data, provider)
    data, provider, _, _, _ = mock_provider(model)
    for size in (0, -1, True, 1.5, int(np.iinfo(np.intp).max)):
        provider._native_core._arma_aparch_mean_refinement_workspace_bytes = lambda *_, size=size: size
        provider._aligned_workspace = lambda _: pytest.fail("Invalid workspace must not allocate")
        with pytest.raises(ValueError):
            callback_for(model, data, provider)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("orders,runtime", [((1, 1, 1, 1), False), ((2, 1, 3, 2), True)])
def test_real_small_image_callback_matches_original_single_anchor(density, orders, runtime):
    model = model_for(density, orders)
    theta, data = sample(model)
    coordinates = native._theta_to_coordinates(model, theta)
    theta = native._coordinates_to_theta(model, coordinates)
    request = replace(request_for(model, theta=theta, budget=1),
                      start=policy.PhaseStart(theta, coordinates))
    callback = adapters.NativeSingleAnchor(model, data, native, maximum_steps=1,
        qualified_densities=(density,), radius=2, maximum_step=.1, force_runtime=runtime)
    result = callback(request)
    continuation, best = np.full_like(theta, np.nan), np.full_like(theta, np.nan)
    arena = native._aligned_workspace(callback.arena.size)
    raw = native._native_core._arma_aparch_mean_refine(theta.ctypes.data, theta.size,
        data.ctypes.data, data.size, callback.h0, *orders, density, 1, 2, arena.address, arena.size,
        continuation.ctypes.data, continuation.size, best.ctypes.data, best.size, runtime, True, .1)
    assert result.evidence["summary"] == adapters.SingleAnchorSummary(*raw)._asdict()
    assert result.mode in (-3, 0, 1) and result.continuation.finite_feasible
    assert result.work.function_evaluations == raw[5] + raw[6]
    assert result.work.gradient_evaluations == result.work.hessian_evaluations == raw[6]
    assert result.points and all(point.solve.mode == -105 and not point.certified for point in result.points)
    assert result.continuation.theta.tobytes() == continuation.tobytes()
    assert result.points[-1].theta.tobytes() == best.tobytes()
    for point in result.points:
        value, _, _ = native._evaluate_theta_request(model, data, point.theta, request=1)
        assert abs(value - point.value) <= 2e-11
        unchanged = point.theta.tobytes() == request.start.theta.tobytes()
        assert (point.coordinates is not None) == unchanged
