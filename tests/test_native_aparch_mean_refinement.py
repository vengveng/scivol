"""Checked private refinement binding; returned points are not certificates."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from scivol import _native_core
from test_native_aparch import DENSITIES, _evaluate, _forced_runtime


SENTINEL = 271828.125


def phase_sample(orders, shape=()):
    ar, ma, p, q = orders
    theta = np.r_[.03, np.full(ar, .12 / max(ar, 1)),
                  np.full(ma, -.08 / max(ma, 1)), .08,
                  np.full(p, .07 / p), np.full(p, .15),
                  np.full(q, .82 / q), 1.7, shape]
    data = np.random.default_rng(1037).normal(.1, .7, 32)
    return np.ascontiguousarray(theta), np.ascontiguousarray(data)


def call_phase(theta, data, orders=(1, 1, 1, 1), density=0,
               *, runtime=False, changes=None, optional_arguments=(),
               preserve_outputs=False, binding=None):
    required = _native_core._arma_aparch_mean_refinement_workspace_bytes(
        *orders, density, runtime, *optional_arguments[:1])
    arena = np.full(required + 32, 0xA5, dtype=np.uint8)
    continuation = np.full(theta.size + 2, SENTINEL)
    best = np.full(theta.size + 2, SENTINEL)
    arguments = dict(parameters=theta.ctypes.data, count=theta.size,
                     data=data.ctypes.data, n=data.size, h0=.5,
                     p_ar=orders[0], q_ma=orders[1], p=orders[2], q=orders[3],
                     density=density, steps=1, radius=2,
                     arena=arena.ctypes.data + 16, bytes=required,
                     continuation=continuation[1:].ctypes.data, continuation_count=theta.size,
                     best=best[1:].ctypes.data, best_count=theta.size, runtime=runtime)
    if changes:
        arguments.update(changes(arguments))
    before_theta, before_data = theta.copy(), data.copy()
    if binding is None:
        binding = _native_core._arma_aparch_mean_refine
    try:
        result = binding(*arguments.values(), *optional_arguments)
    finally:
        np.testing.assert_array_equal(theta, before_theta)
        np.testing.assert_array_equal(data, before_data)
        np.testing.assert_array_equal(arena[:16], 0xA5)
        np.testing.assert_array_equal(arena[16 + required:], 0xA5)
        for output in (continuation, best):
            np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
        if preserve_outputs:
            np.testing.assert_array_equal(arena, 0xA5)
            np.testing.assert_array_equal(continuation, SENTINEL)
            np.testing.assert_array_equal(best, SENTINEL)
    return result, continuation[1:-1], best[1:-1]


def assert_invalid_phase(result, continuation, best):
    assert result[0] == -1
    assert np.isposinf(result[1:3]).all()
    assert result[3:] == (0,) * 7
    np.testing.assert_array_equal(continuation, SENTINEL)
    np.testing.assert_array_equal(best, SENTINEL)


@pytest.mark.parametrize("name,density,shape", DENSITIES)
@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (2, 1, 3, 2)])
def test_phase_uses_complete_likelihood_and_preserves_frozen_parameters(name, density, shape, orders):
    theta, data = phase_sample(orders, shape)
    initial = _evaluate(theta, data, density, 1, orders=orders, initial_variance=.5)
    assert initial[:2] == (0, 1)
    result, continuation, best = call_phase(theta, data, orders, density)
    stop, best_value, continuation_value, iterations, accepted, values, curvatures, lattice, proposals, rejected = result
    assert stop in (0, 1)
    assert iterations == curvatures == 1
    assert accepted in (0, 1)
    assert values >= 1 and proposals >= rejected
    assert lattice > 0
    assert best_value <= initial[2]
    assert best_value <= continuation_value
    mean = 1 + orders[0] + orders[1]
    for point, value in ((continuation, continuation_value), (best, best_value)):
        np.testing.assert_array_equal(point[mean:], theta[mean:])
        evaluated = _evaluate(point, data, density, 1, orders=orders, initial_variance=.5)
        assert evaluated[:2] == (0, 1)
        assert abs(evaluated[2] - value) <= 2e-11


@pytest.mark.parametrize("name,density,shape", DENSITIES[:4])
def test_fixed_phase_agrees_with_runtime_control(name, density, shape):
    theta, data = phase_sample((1, 1, 1, 1), shape)
    fixed = call_phase(theta, data, density=density)
    runtime = call_phase(theta, data, density=density, runtime=True)
    np.testing.assert_allclose(fixed[0], runtime[0], rtol=0, atol=2e-11)
    np.testing.assert_array_equal(fixed[1], runtime[1])
    np.testing.assert_array_equal(fixed[2], runtime[2])


@pytest.mark.parametrize("change", [
    lambda a: {"bytes": a["bytes"] - 1},
    lambda a: {"count": a["count"] - 1},
    lambda a: {"continuation_count": a["count"] - 1},
    lambda a: {"best_count": a["count"] - 1},
    lambda a: {"parameters": 0},
    lambda a: {"data": 0},
    lambda a: {"arena": 0},
    lambda a: {"continuation": 0},
    lambda a: {"best": 0},
    lambda a: {"continuation": a["parameters"]},
    lambda a: {"best": a["continuation"]},
    lambda a: {"arena": a["parameters"]},
    lambda a: {"arena": a["arena"] + 1},
    lambda a: {"steps": 0},
    lambda a: {"steps": np.iinfo(np.uintp).max},
    lambda a: {"radius": np.iinfo(np.uintp).max},
    lambda a: {"n": 1},
    lambda a: {"h0": -1.},
    lambda a: {"density": 999},
])
def test_invalid_phase_arguments_preserve_outputs(change):
    theta, data = phase_sample((1, 1, 1, 1))
    assert_invalid_phase(*call_phase(theta, data, changes=change))


@pytest.mark.parametrize("orders,density", [((0, 1, 1, 1), 0), ((1, 1, 0, 1), 0),
                                           ((1, 1, 1, 1), 999)])
def test_unsupported_phase_has_no_workspace(orders, density):
    assert _native_core._arma_aparch_mean_refinement_workspace_bytes(*orders, density, False) == 0


def test_phase_python_binding_rejects_negative_sizes():
    with pytest.raises(OverflowError):
        _native_core._arma_aparch_mean_refinement_workspace_bytes(-1, 1, 1, 1, 0, False)
    with pytest.raises(TypeError):
        _native_core._arma_aparch_mean_refine()


@pytest.mark.parametrize("name,density,shape", DENSITIES)
@pytest.mark.parametrize("orders,runtime", [
    ((1, 1, 1, 1), False), ((1, 1, 1, 1), True), ((2, 1, 3, 2), True),
])
def test_workspace_optional_joint_argument(name, density, shape, orders, runtime):
    arguments = (*orders, density, runtime)
    default = _native_core._arma_aparch_mean_refinement_workspace_bytes(*arguments)
    explicit = _native_core._arma_aparch_mean_refinement_workspace_bytes(*arguments, False)
    joint = _native_core._arma_aparch_mean_refinement_workspace_bytes(*arguments, True)
    assert default == explicit > 0
    assert joint > 0


@pytest.mark.parametrize("name,density,shape", DENSITIES[:2])
@pytest.mark.parametrize("orders,runtime", [
    ((1, 1, 1, 1), False), ((1, 1, 1, 1), True), ((2, 1, 3, 2), True),
])
def test_mean_binding_defaults_equal_explicit_options(name, density, shape, orders, runtime):
    theta, data = phase_sample(orders, shape)
    default = call_phase(theta, data, orders, density, runtime=runtime)
    for optional_arguments in ((False,), (False, 1e-3)):
        explicit = call_phase(theta, data, orders, density, runtime=runtime,
                              optional_arguments=optional_arguments)
        assert explicit[0] == default[0]
        np.testing.assert_array_equal(explicit[1], default[1])
        np.testing.assert_array_equal(explicit[2], default[2])


@pytest.mark.parametrize("orders,runtime", [
    ((1, 1, 1, 1), False), ((1, 1, 1, 1), True), ((2, 1, 3, 2), False),
    ((2, 1, 3, 2), True),
])
def test_joint_binding_rescores_feasible_points_at_larger_step(orders, runtime):
    from scivol import APARCH, ARMA, Normal
    from scivol._native_fit import (
        _aparch_volatility_domain, _describe_model, _mean_parameters_are_feasible,
    )

    theta, data = phase_sample(orders)
    initial = _evaluate(theta, data, 0, 1, orders=orders, initial_variance=.5)
    assert initial[:2] == (0, 1) and np.isfinite(initial[2])
    result, continuation, best = call_phase(theta, data, orders, runtime=runtime,
                                           optional_arguments=(True, .1))
    stop, best_value, continuation_value, iterations, accepted, values, curvatures, lattice, proposals, rejected = result
    assert stop in (0, 1)
    assert iterations == curvatures == 1 and accepted in (0, 1)
    assert values >= 1 and proposals >= rejected and lattice > 0
    assert np.isfinite([best_value, continuation_value]).all()
    assert best_value <= initial[2] + 2e-11
    assert best_value <= continuation_value
    model = _describe_model(ARMA(*orders[:2]) + APARCH(*orders[2:]) + Normal())
    assert not np.array_equal(best[model.n_mean:], theta[model.n_mean:])
    small_step = call_phase(theta, data, orders, runtime=runtime,
                            optional_arguments=(True, 1e-3))
    assert not np.array_equal(best, small_step[2])
    for point, value in ((continuation, continuation_value), (best, best_value)):
        assert np.isfinite(point).all()
        assert _mean_parameters_are_feasible(model, point)
        violation, feasible, _ = _aparch_volatility_domain(model, point)
        assert feasible and violation == 0.0
        evaluated = _evaluate(point, data, 0, 1, orders=orders, initial_variance=.5)
        assert evaluated[:2] == (0, 1) and np.isfinite(evaluated[2])
        assert abs(evaluated[2] - value) <= 2e-11


def test_fixed_joint_binding_agrees_with_runtime_control(record_property):
    from scivol import APARCH, ARMA, Normal
    from scivol._native_fit import (
        _aparch_volatility_domain, _describe_model, _mean_parameters_are_feasible,
    )

    orders = (1, 1, 1, 1)
    theta, data = phase_sample(orders)
    fixed = call_phase(theta, data, optional_arguments=(True, .1))
    runtime = call_phase(theta, data, runtime=True, optional_arguments=(True, .1))
    evidence = dict(contract="Shared-point jets, objective values and feasibility; not identical solver work or trajectories",
        fixed_phase=list(fixed[0]), runtime_phase=list(runtime[0]),
        phase_difference=(np.asarray(fixed[0]) - np.asarray(runtime[0])).tolist(),
        continuation_difference=(fixed[1] - runtime[1]).tolist(),
        best_difference=(fixed[2] - runtime[2]).tolist(), shared_points=[])
    record_property("fixed_runtime_joint_evidence", evidence)
    model = _describe_model(ARMA(1, 1) + APARCH(1, 1) + Normal())
    for route, outcome in (("fixed", fixed), ("runtime", runtime)):
        for role, point, value in (("continuation", outcome[1], outcome[0][2]),
                                   ("best", outcome[2], outcome[0][1])):
            before = point.copy(), data.copy()
            mean_feasible = _mean_parameters_are_feasible(model, point)
            violation, feasible, persistence = _aparch_volatility_domain(model, point)
            checked = _evaluate(point, data, 0, 7, orders=orders, initial_variance=.5)
            general = _forced_runtime(point, data, 0, orders=orders,
                                      request=7, initial_variance=.5)
            row = dict(route=route, role=role, theta=point.tolist(), phase_value=value,
                mean_feasible=mean_feasible, volatility_feasible=feasible,
                maximum_violation=violation, persistence=persistence,
                status=checked[0], valid_mask=checked[1], comparisons={})
            evidence["shared_points"].append(row)
            assert mean_feasible and feasible and violation == 0.0
            assert checked[:2] == (0, 7)
            for bit, name, actual, control in zip((1, 2, 4), ("F", "G", "H"), checked[2:], general):
                actual, control = np.asarray(actual), np.asarray(control)
                difference = actual - control
                tolerance = 2e-11 + (0 if bit == 1 else 3e-10) * np.abs(control)
                row["comparisons"][name] = dict(fixed=actual.tolist(), runtime=control.tolist(),
                    difference=difference.tolist(), maximum_absolute_difference=float(np.max(np.abs(difference))),
                    maximum_tolerance_ratio=float(np.max(np.abs(difference) / tolerance)))
                assert np.isfinite(actual).all() and np.isfinite(control).all()
                np.testing.assert_allclose(actual, control, rtol=0 if bit == 1 else 3e-10, atol=2e-11)
                if bit == 1:
                    assert abs(float(actual) - value) <= 2e-11
                    assert abs(float(control) - value) <= 2e-11
            np.testing.assert_array_equal(point, before[0])
            np.testing.assert_array_equal(data, before[1])
    np.testing.assert_allclose(fixed[0][1:3], runtime[0][1:3], rtol=0, atol=2e-11)


@pytest.mark.parametrize("runtime", [False, True])
def test_joint_binding_default_step_equals_explicit_step(runtime):
    theta, data = phase_sample((1, 1, 1, 1))
    default = call_phase(theta, data, runtime=runtime, optional_arguments=(True,))
    explicit = call_phase(theta, data, runtime=runtime, optional_arguments=(True, 1e-3))
    assert explicit[0] == default[0]
    np.testing.assert_array_equal(explicit[1], default[1])
    np.testing.assert_array_equal(explicit[2], default[2])


@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (1, 0, 1, 2), (2, 1, 3, 2)])
@pytest.mark.parametrize("runtime", [False, True])
def test_joint_phase_reuses_mean_projection_for_volatility_trials(orders, runtime):
    theta, data = phase_sample(orders)
    radius, steps = 2, 3
    raw, _, _ = call_phase(theta, data, orders, runtime=runtime,
        changes=lambda _: dict(steps=steps, radius=radius), optional_arguments=(True, .1))
    assert raw[0] == 0 and raw[3] == steps
    mean_width = 1 + orders[0] + orders[1]
    # Five scales for two joint directions and each free mean axis, plus one
    # shared projection for every volatility-only center in this iteration.
    scans_per_iteration = 10 + 5 * (mean_width - 1) + 1
    assert 0 < raw[7] <= steps * scans_per_iteration * (2 * radius + 1)


@pytest.mark.parametrize("name,density,shape", DENSITIES)
@pytest.mark.parametrize("orders", [(1, 1, 1, 1), (2, 1, 3, 2), (3, 2, 2, 1)])
def test_supported_joint_binding_masks_and_runtime_parity(name, density, shape, orders):
    theta, data = phase_sample(orders, shape)
    initial = _evaluate(theta, data, density, 1, orders=orders, initial_variance=.5)
    assert initial[:2] == (0, 1) and np.isfinite(initial[2])
    for force_runtime in (False, True):
        required = _native_core._arma_aparch_mean_refinement_workspace_bytes(
            *orders, density, force_runtime, True)
        assert required > 0
        raw, continuation, best = call_phase(theta, data, orders, density,
            runtime=force_runtime, optional_arguments=(True, .1))
        assert raw[0] in (0, 1) and raw[3] == raw[6] == 1
        assert raw[4] in (0, 1) and raw[5] >= 1 and raw[8] >= raw[9]
        assert np.isfinite(raw[1:3]).all() and raw[1] <= initial[2] + 2e-11
        for point, value in ((continuation, raw[2]), (best, raw[1])):
            expected = _evaluate(point, data, density, 7, orders=orders, initial_variance=.5)
            assert expected[:2] == (0, 7)
            for request in range(1, 8):
                actual = _evaluate(point, data, density, request,
                    orders=orders, initial_variance=.5)
                runtime = _forced_runtime(point, data, density, request=request,
                    orders=orders, initial_variance=.5)
                assert actual[:2] == (0, request)
                for bit, observed, full, general in zip((1, 2, 4), actual[2:], expected[2:], runtime):
                    if request & bit:
                        assert np.isfinite(observed).all() and np.isfinite(general).all()
                        np.testing.assert_allclose(observed, full,
                            rtol=0 if bit == 1 else 3e-10, atol=2e-11)
                        np.testing.assert_allclose(observed, general,
                            rtol=0 if bit == 1 else 3e-10, atol=2e-11)
            assert abs(float(expected[2]) - value) <= 2e-11


@pytest.mark.parametrize("name,density,shape", [("unknown", 999, [])])
@pytest.mark.parametrize("runtime", [False, True])
def test_unsupported_joint_binding_preserves_all_buffers(name, density, shape, runtime):
    theta, data = phase_sample((1, 1, 1, 1), shape)
    assert _native_core._arma_aparch_mean_refinement_workspace_bytes(
        1, 1, 1, 1, density, runtime, True) == 0
    assert_invalid_phase(*call_phase(theta, data, density=density, runtime=runtime,
        optional_arguments=(True, .1), preserve_outputs=True))


@pytest.mark.parametrize("maximum_step", [0.0, -0.1, np.nan, np.inf])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
def test_invalid_step_binding_preserves_all_buffers(maximum_step, joint, runtime):
    theta, data = phase_sample((1, 1, 1, 1))
    assert_invalid_phase(*call_phase(theta, data, runtime=runtime,
        optional_arguments=(joint, maximum_step), preserve_outputs=True))


class InvalidTruth:
    def __bool__(self):
        raise ValueError("truth conversion failed")


@pytest.mark.parametrize("index", [5, 6])
def test_workspace_binding_propagates_truth_parsing_error(index):
    arguments = [1, 1, 1, 1, 0, False, False]
    arguments[index] = InvalidTruth()
    with pytest.raises(ValueError, match="truth conversion failed"):
        _native_core._arma_aparch_mean_refinement_workspace_bytes(*arguments)


@pytest.mark.parametrize("optional_arguments,change,error", [
    ((False, object()), None, TypeError),
    ((False, "0.1"), None, TypeError),
    ((), lambda a: {"joint": InvalidTruth()}, ValueError),
    ((True, .1), lambda a: {"runtime": InvalidTruth()}, ValueError),
    ((True, .1), lambda a: {"count": -1}, OverflowError),
])
def test_refinement_binding_parsing_errors_preserve_all_buffers(optional_arguments, change, error):
    theta, data = phase_sample((1, 1, 1, 1))
    with pytest.raises(error):
        call_phase(theta, data, optional_arguments=optional_arguments,
                   changes=change, preserve_outputs=True)


@pytest.mark.parametrize("count", [5, 8])
def test_workspace_binding_rejects_wrong_arity(count):
    with pytest.raises(TypeError):
        _native_core._arma_aparch_mean_refinement_workspace_bytes(*(0,) * count)


@pytest.mark.parametrize("count", [18, 22])
def test_refinement_binding_rejects_wrong_arity(count):
    with pytest.raises(TypeError):
        _native_core._arma_aparch_mean_refine(*(0,) * count)


class PhaseEvidence(ctypes.Structure):
    _fields_ = [("stop", ctypes.c_int), ("value", ctypes.c_double),
                ("continuation_value", ctypes.c_double)] + [
        (name, ctypes.c_size_t) for name in (
            "iterations", "accepted", "values", "curvatures", "lattice", "proposals", "rejected")]


@pytest.mark.parametrize("runtime", [False, True])
def test_legacy_c_phase_abi_matches_default_mean_binding(runtime):
    size, pointer = ctypes.c_size_t, ctypes.c_void_p
    library = ctypes.CDLL(_native_core.__file__)
    workspace = library.vlk_arma_aparch_mean_refinement_workspace_bytes
    workspace.restype = size
    workspace.argtypes = [size, size, size, size, ctypes.c_uint32, ctypes.c_bool]
    assert workspace(1, 1, 1, 1, 0, runtime) == (
        _native_core._arma_aparch_mean_refinement_workspace_bytes(1, 1, 1, 1, 0, runtime))
    run = library.vlk_arma_aparch_mean_refine
    run.restype = ctypes.c_int32
    run.argtypes = [pointer, size, pointer, size, ctypes.c_double,
                   size, size, size, size, ctypes.c_uint32, size, size,
                   pointer, size, pointer, size, pointer, size, ctypes.c_bool,
                   ctypes.POINTER(PhaseEvidence)]

    def c_binding(*arguments):
        evidence = PhaseEvidence()
        assert run(*arguments, ctypes.byref(evidence)) == 0
        return tuple(getattr(evidence, name) for name, _ in evidence._fields_)

    theta, data = phase_sample((1, 1, 1, 1))
    default = call_phase(theta, data, runtime=runtime)
    legacy = call_phase(theta, data, runtime=runtime, binding=c_binding)
    assert legacy[0] == default[0]
    np.testing.assert_array_equal(legacy[1], default[1])
    np.testing.assert_array_equal(legacy[2], default[2])


@pytest.mark.parametrize("overlap", ["parameters", "workspace", "best", "null"])
def test_c_phase_rejects_aliased_evidence_before_writing(overlap):
    size, pointer = ctypes.c_size_t, ctypes.c_void_p
    library = ctypes.CDLL(_native_core.__file__)
    run = library.vlk_arma_aparch_mean_refine
    run.restype = ctypes.c_int32
    run.argtypes = [pointer, size, pointer, size, ctypes.c_double,
                   size, size, size, size, ctypes.c_uint32, size, size,
                   pointer, size, pointer, size, pointer, size, ctypes.c_bool, pointer]
    theta, data = phase_sample((1, 1, 1, 1))
    count = _native_core._arma_aparch_mean_refinement_workspace_bytes(1, 1, 1, 1, 0, False)
    memory = np.full(count, 0xA5, dtype=np.uint8)
    continuation, best = np.full(theta.size, SENTINEL), np.full(theta.size, SENTINEL)
    arrays = (theta, data, memory, continuation, best)
    before = [array.copy() for array in arrays]
    address = {"parameters": theta.ctypes.data, "workspace": memory.ctypes.data,
               "best": best.ctypes.data, "null": 0}[overlap]
    status = run(theta.ctypes.data, theta.size, data.ctypes.data, data.size, .5,
                 1, 1, 1, 1, 0, 1, 2, memory.ctypes.data, memory.nbytes,
                 continuation.ctypes.data, theta.size, best.ctypes.data, theta.size,
                 False, address)
    assert status == 1
    for actual, expected in zip(arrays, before):
        np.testing.assert_array_equal(actual, expected)
