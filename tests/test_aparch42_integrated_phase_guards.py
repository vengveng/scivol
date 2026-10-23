"""Memory and reporting contracts for the integrated APARCH optional phases."""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


MAX_SIZE = ct.c_size_t(-1).value
SENTINEL = 271828.125
INDEX_SENTINEL = 271828
SHAPES = ((), (7.0,), (7.0, .23), (1.6,), (.12, 1.5, .4),
          (.12, 1.5), (.12, 1.5), (.12, 2.0))
INDEPENDENT_DENSITIES = (0, 3, 4, 5, 6, 7)
LAYOUTS = (((1, 1, 1, 1), 1, False), ((1, 1, 1, 1), 2, False), ((1, 1, 1, 1), 2, True),
           ((2, 1, 3, 2), 2, False), ((3, 2, 2, 1), 2, False),
           ((0, 0, 1, 1), 1, False))
HELD_POINTERS = (1, 3, 6, 7, 8, 10, 12, 14)
MULTI_POINTERS = (0, 2, 12, 14, 16, 18, 20, 22, 24)


class Specification(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("ar", "ma", "p", "q", "k")]
    _fields_ += [("density", ct.c_uint32), ("force_runtime", ct.c_bool)]


@pytest.fixture(scope="session")
def integrated_core():
    manifest = os.environ.get("SCIVOL_APARCH42_INTEGRATED_BUILD")
    if manifest:
        record = json.loads(Path(manifest).read_text())
        assert record["complete"] and record["build_qualification_passed"]
        image = Path(record["image"])
        assert hashlib.sha256(image.read_bytes()).hexdigest() == record["image_sha256"]
        spec = importlib.util.spec_from_file_location("_native_core", image)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    else:
        from scivol import _native_core as core
    for name in ("_arma_aparch_prepare", "_arma_aparch_preparation_workspace_bytes",
                 "_arma_aparch_multi_phase_run", "_arma_aparch_multi_phase_workspace_bytes"):
        assert hasattr(core, name), f"Integrated production ABI is missing {name}"
    return core


def guarded(count, dtype=np.float64, fill=SENTINEL):
    if np.issubdtype(dtype, np.integer):
        fill = INDEX_SENTINEL
    storage = np.full(count + 8, fill, dtype=dtype)
    return storage, storage[4:4 + count]


def arena(size):
    storage = np.full(size + 96, 0xA5, dtype=np.uint8)
    start = 32 + (-storage.ctypes.data % 16)
    return storage, storage[start:start + size]


@dataclass
class PhaseCall:
    core: object
    method: str
    args: list
    inputs: list
    outputs: list
    arenas: list

    def invoke(self):
        return getattr(self.core, self.method)(*self.args)

    def snapshot(self):
        return [x.tobytes() for x in self.inputs + [s for s, _ in self.outputs + self.arenas]]

    def assert_atomic(self, snapshot):
        assert self.snapshot() == snapshot

    def assert_canaries(self):
        for storage, view in self.outputs + self.arenas:
            start = (view.ctypes.data - storage.ctypes.data) // storage.itemsize
            fill = (0xA5 if storage.dtype == np.uint8 else
                    INDEX_SENTINEL if np.issubdtype(storage.dtype, np.integer) else SENTINEL)
            assert np.all(storage[:start] == fill)
            assert np.all(storage[start + view.size:] == fill)


def sample(orders, density):
    ar, ma, p, q = orders
    theta = np.ascontiguousarray(np.r_[.04, np.linspace(.09, -.06, ar),
        np.linspace(-.08, .04, ma), .12, np.full(p, .06 / p),
        np.linspace(-.16, .13, p), np.full(q, .6 / q), 1.4, SHAPES[density]])
    data = np.ascontiguousarray(np.random.default_rng(712 + sum(orders)).normal(.1, .8, 13))
    return theta, data


def multi_call(core, orders, density, k=2, runtime=False, steps=2, padding=0):
    theta, data = sample(orders, density)
    size = core._arma_aparch_multi_phase_workspace_bytes(*orders, density, k, runtime)
    assert size > 0
    scratch = arena(size)
    outputs = [guarded(k + padding, np.uintp), guarded(k + padding, np.uintp)]
    outputs += [guarded(theta.size + padding), guarded(theta.size + padding),
                guarded(12 + padding), guarded(steps * (12 + 2 * theta.size) + padding)]
    args = [theta.ctypes.data, theta.size, data.ctypes.data, data.size, .5, *orders,
            density, k, steps, scratch[1].ctypes.data, size]
    for _, output in outputs:
        args += [output.ctypes.data, output.size]
    args += [runtime]
    return PhaseCall(core, "_arma_aparch_multi_phase_run", args, [theta, data], outputs, [scratch])


def held_call(core, orders, density, runtime=False, iterations=2, *, strategy=None,
              jet_request=None):
    theta, data = sample(orders, density)
    z = np.empty_like(theta)
    map_size = core._arma_aparch_map_workspace_bytes(*orders, density)
    map_memory, map_view = arena(map_size)
    core._map_arma_aparch(theta.ctypes.data, z.ctypes.data, *orders, density, False,
                         map_view.ctypes.data, map_size)
    assert np.isfinite(z).all()
    lower, upper = np.empty_like(z), np.empty_like(z)
    core._arma_aparch_coordinate_bounds(lower.ctypes.data, upper.ctypes.data, *orders, density)
    request = (*orders, density, runtime, 1.5, 1e-8, iterations)
    if strategy is not None:
        request = (*request, strategy)
    size = core._arma_aparch_preparation_workspace_bytes(request)
    if jet_request is None:
        jet_request = 7 if strategy == 1 else 3
    jet_size = core._arma_aparch_workspace_bytes(*orders, density, 1, jet_request)
    scratch, jet = arena(size), arena(jet_size)
    outputs = [guarded(z.size), guarded(z.size)]
    args = [request, z.ctypes.data, z.size, data.ctypes.data, data.size, .5,
            lower.ctypes.data, upper.ctypes.data, scratch[1].ctypes.data, size,
            jet[1].ctypes.data, jet_size, outputs[0][1].ctypes.data, z.size,
            outputs[1][1].ctypes.data, z.size]
    return PhaseCall(core, "_arma_aparch_prepare", args, [z, data, lower, upper], outputs, [scratch, jet])


def full_strategy_report(result):
    assert type(result) is tuple and len(result) == 2
    report, diagnostics = result
    assert type(report) is tuple and len(report) == 16
    assert type(diagnostics) is dict
    assert type(diagnostics["strategy"]) is int and diagnostics["strategy"] == 1
    assert diagnostics["held_power_certified"] is False
    solver = diagnostics["solver_diagnostics_unrestricted"]
    assert type(solver) is tuple and len(solver) == 10
    assert type(solver[0]) is int
    assert all(type(value) is float for value in solver[1:4])
    assert all(type(value) is int and value >= 0 for value in solver[4:])
    for name in ("reuse_queries", "reuse_hits", "constraint_evaluations", "jacobian_evaluations",
                 "constraint_curvature_resets", "checkpoint_restarts", "stop_reason"):
        assert type(diagnostics[name]) is int and diagnostics[name] >= 0
    assert diagnostics["reuse_hits"] <= diagnostics["reuse_queries"]
    assert type(diagnostics["derivatives_current"]) is bool
    return report, diagnostics


def held_value(call, point):
    request = call.args[0]
    orders, density = request[:4], request[4]
    size = call.core._arma_aparch_workspace_bytes(*orders, density, 1, 1)
    storage, scratch = arena(size)
    value_storage, value = guarded(1)
    status = call.core._arma_aparch_jet(point.ctypes.data, point.size,
        call.inputs[1].ctypes.data, call.inputs[1].size, call.args[5], *orders, density, 1, 1,
        scratch.ctypes.data, size, value.ctypes.data, 0, 0)
    assert status == (0, 1)
    probe = PhaseCall(call.core, "_arma_aparch_jet", [], [],
                      [(value_storage, value)], [(storage, scratch)])
    probe.assert_canaries()
    return value[0]


@pytest.mark.parametrize("density", INDEPENDENT_DENSITIES)
@pytest.mark.parametrize("orders,runtime", (((1, 1, 1, 1), False), ((2, 1, 3, 2), False),
                                           ((3, 2, 2, 1), True), ((0, 0, 1, 1), False)))
def test_integrated_held_full_strategy_plans_mask7(integrated_core, density, orders, runtime):
    request = (*orders, density, runtime, 1.5, 1e-8, 1)
    raw_size = integrated_core._arma_aparch_preparation_workspace_bytes(request)
    full_size = integrated_core._arma_aparch_preparation_workspace_bytes((*request, 1))
    assert full_size > raw_size > 0
    first_order = integrated_core._arma_aparch_workspace_bytes(*orders, density, 1, 3)
    second_order = integrated_core._arma_aparch_workspace_bytes(*orders, density, 1, 7)
    assert second_order > first_order > 0


@pytest.mark.parametrize("density", INDEPENDENT_DENSITIES)
@pytest.mark.parametrize("orders,runtime", (((1, 1, 1, 1), False), ((3, 2, 2, 1), True)))
def test_integrated_held_full_strategy_nested_result_and_finite_slots(integrated_core, density, orders, runtime):
    call = held_call(integrated_core, orders, density, runtime, iterations=1, strategy=1)
    assert len(call.args[0]) == 10 and call.args[0][-1] == 1
    assert call.args[11] == integrated_core._arma_aparch_workspace_bytes(*orders, density, 1, 7)
    before = [value.tobytes() for value in call.inputs]
    incoming_value = held_value(call, call.inputs[0])
    report, diagnostics = full_strategy_report(call.invoke())
    assert report[0] == 0 and report[4] == -105
    assert np.isfinite(report[1:4]).all()
    assert report[6] >= 4 and report[7] >= 1 and report[8] >= 0
    assert report[9:11] == (0, 0) and report[13] == 1
    solver = diagnostics["solver_diagnostics_unrestricted"]
    assert report[6] >= solver[5] and report[7] >= solver[6]
    assert diagnostics["reuse_queries"] >= 1
    assert report[2] <= min(incoming_value, report[1]) + 2e-11
    for (_, point), expected in zip(call.outputs, report[1:3]):
        assert np.isfinite(point).all()
        assert abs(held_value(call, point) - expected) <= 2e-11
    power_index = call.inputs[0].size - len(SHAPES[density]) - 1
    map_size = integrated_core._arma_aparch_map_workspace_bytes(*orders, density)
    map_storage, map_view = arena(map_size)
    theta_storage, theta = guarded(call.inputs[0].size)
    integrated_core._map_arma_aparch(call.outputs[0][1].ctypes.data, theta.ctypes.data,
        *orders, density, True, map_view.ctypes.data, map_size)
    assert abs(theta[power_index] - call.args[0][6]) <= 1e-10
    map_probe = PhaseCall(integrated_core, "_map_arma_aparch", [], [],
                          [(theta_storage, theta)], [(map_storage, map_view)])
    map_probe.assert_canaries()
    assert before == [value.tobytes() for value in call.inputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", INDEPENDENT_DENSITIES)
def test_integrated_held_full_strategy_arenas_and_ranges_reject_atomically(integrated_core, density):
    orders = (3, 2, 2, 1)
    call = held_call(integrated_core, orders, density, True, iterations=1, strategy=1)
    original = list(call.args)
    faults = [(9, original[9] - 1), (11, original[11] - 1), (2, original[2] - 1),
              (4, MAX_SIZE), (9, MAX_SIZE), (11, MAX_SIZE), (13, original[13] - 1),
              (15, original[15] + 1)]
    for position in HELD_POINTERS:
        faults.extend((position, pointer) for pointer in (0, original[position] + 1, MAX_SIZE - 7))
    for left, right in itertools.combinations(HELD_POINTERS, 2):
        faults.extend((right, original[left] + offset) for offset in (0, 8))
    for position, bad in faults:
        call.args[:] = original
        call.args[position] = bad
        before = call.snapshot()
        report, diagnostics = full_strategy_report(call.invoke())
        assert report[0] == -1, (density, position, bad)
        assert np.isposinf(report[1:4]).all()
        assert report[4] == -1 and report[5:12] == (0,) * 7
        assert report[12] == -1 and report[13:] == (0, 0, 0)
        assert diagnostics["reuse_queries"] == diagnostics["reuse_hits"] == 0
        call.assert_atomic(before)
    for first_order in (False, True):
        call.args[:] = original
        arena_index, pointer_index, byte_index = (1, 10, 11) if first_order else (0, 8, 9)
        short_size = (integrated_core._arma_aparch_workspace_bytes(*orders, density, 1, 3)
                      if first_order else integrated_core._arma_aparch_preparation_workspace_bytes(original[0][:-1]))
        original_arena = call.arenas[arena_index]
        call.arenas[arena_index] = arena(short_size)
        call.args[pointer_index] = call.arenas[arena_index][1].ctypes.data
        call.args[byte_index] = short_size
        before = call.snapshot()
        report, diagnostics = full_strategy_report(call.invoke())
        assert report[0] == -1
        assert report[6:11] == (0,) * 5 and diagnostics["reuse_queries"] == 0
        call.assert_atomic(before)
        call.arenas[arena_index] = original_arena


@pytest.mark.parametrize("density", INDEPENDENT_DENSITIES)
def test_integrated_held_full_strategy_failed_retarget_retains_incoming_best(integrated_core, density):
    call = held_call(integrated_core, (1, 1, 1, 1), density, iterations=1, strategy=1)
    power_index = call.inputs[0].size - len(SHAPES[density]) - 1
    call.inputs[2][power_index] = call.inputs[0][power_index]
    call.inputs[3][power_index] = call.inputs[0][power_index]
    before = [value.tobytes() for value in call.inputs]
    incoming_value = held_value(call, call.inputs[0])
    report, diagnostics = full_strategy_report(call.invoke())
    assert report[0] == -2 and report[4] == -2
    assert np.isposinf(report[1]) and np.isposinf(report[3])
    assert abs(report[2] - incoming_value) <= 2e-11
    assert report[5:11] == (0, 1, 0, 0, 0, 0)
    assert diagnostics["solver_diagnostics_unrestricted"][4:] == (0,) * 6
    assert np.all(call.outputs[0][1] == SENTINEL)
    np.testing.assert_array_equal(call.outputs[1][1], call.inputs[0])
    assert before == [value.tobytes() for value in call.inputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", (1, 2))
def test_integrated_held_full_strategy_rejects_coupled_densities(integrated_core, density):
    call = held_call(integrated_core, (1, 1, 1, 1), density)
    call.args[0] = (*call.args[0], 1)
    before = call.snapshot()
    with pytest.raises(ValueError, match="independent power coordinate"):
        integrated_core._arma_aparch_preparation_workspace_bytes(call.args[0])
    with pytest.raises(ValueError, match="independent power coordinate"):
        call.invoke()
    call.assert_atomic(before)


@pytest.mark.parametrize("density", INDEPENDENT_DENSITIES)
def test_integrated_held_full_strategy_parser_rejects_unknown_strategy(integrated_core, density):
    call = held_call(integrated_core, (1, 1, 1, 1), density)
    original_request = call.args[0]
    for strategy, error in ((2, ValueError), (MAX_SIZE, ValueError), (-1, OverflowError), (1., TypeError)):
        call.args[0] = (*original_request, strategy)
        before = call.snapshot()
        with pytest.raises(error):
            integrated_core._arma_aparch_preparation_workspace_bytes(call.args[0])
        with pytest.raises(error):
            call.invoke()
        call.assert_atomic(before)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("orders,k,runtime", LAYOUTS)
@pytest.mark.parametrize("phase", ("held", "multi"))
def test_integrated_phase_exact_arenas_and_reuse(integrated_core, density, orders, k, runtime, phase):
    call = (held_call(integrated_core, orders, density, runtime) if phase == "held" else
            multi_call(integrated_core, orders, density, k, runtime))
    before = [x.tobytes() for x in call.inputs]
    result = call.invoke()
    call.assert_canaries()
    assert before == [x.tobytes() for x in call.inputs]
    if phase == "held":
        assert len(result) == 16 and result[0] == 0
        assert np.isfinite(result[1:4]).all()
        assert all(isinstance(x, int) for x in result[4:])
        assert result[6] >= 4 and result[7] >= 1 and result[8] == 0
        assert result[9] >= result[10] >= 0
        if density not in (1, 2):
            assert result[9:11] == (0, 0) and result[13] == 1
        else:
            assert result[9] and result[10] and result[13] == 0
        expected_map = 0 if runtime or orders not in ((1, 1, 1, 1), (2, 1, 3, 2)) else (1 if orders[0] == 1 else 2)
        assert result[11] == expected_map
        assert result[12] == integrated_core._arma_aparch_dispatch_kind(*orders, density, 1)
    else:
        assert result == 0
        summary = call.outputs[4][1]
        assert np.isfinite(summary).all()
        assert summary[3] <= 2 and summary[4] <= summary[3]
        assert summary[5] >= 1 and summary[6] >= 1
        assert summary[9] >= summary[10] >= 0 and summary[11] <= summary[10]
        assert np.equal(summary[3:], np.floor(summary[3:])).all()
        width = call.inputs[0].size
        trace = call.outputs[5][1].reshape(2, 12 + 2 * width)
        rows = int(summary[3])
        if rows:
            assert trace[rows - 1, 9] == summary[5]
            assert trace[rows - 1, 10] == summary[6]
            np.testing.assert_array_equal(trace[rows - 1, 12:12 + width], call.outputs[2][1])
            np.testing.assert_array_equal(trace[rows - 1, 12 + width:], call.outputs[3][1])
        assert np.all(trace[rows:] == SENTINEL)
    first = [view.tobytes() for _, view in call.outputs]
    second = call.invoke()
    assert result == second
    assert first == [view.tobytes() for _, view in call.outputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (3, 2, 2, 1)))
@pytest.mark.parametrize("phase", ("held", "multi"))
def test_integrated_all_pairwise_aliases_are_atomic(integrated_core, density, orders, phase):
    call = held_call(integrated_core, orders, density) if phase == "held" else multi_call(integrated_core, orders, density)
    positions = HELD_POINTERS if phase == "held" else MULTI_POINTERS
    original = list(call.args)
    for left, right in itertools.combinations(positions, 2):
        for offset in (0, 8):
            call.args[:] = original
            call.args[right] = call.args[left] + offset
            before = call.snapshot()
            result = call.invoke()
            assert (result[0] if phase == "held" else result) == -1, (phase, left, right, offset)
            call.assert_atomic(before)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("phase", ("held", "multi"))
def test_integrated_bad_ranges_and_counts_are_atomic(integrated_core, density, phase):
    orders = (3, 2, 2, 1)
    call = held_call(integrated_core, orders, density) if phase == "held" else multi_call(integrated_core, orders, density)
    original = list(call.args)
    positions = HELD_POINTERS if phase == "held" else MULTI_POINTERS
    faults = []
    for position in positions:
        for pointer in (0, original[position] + 1, MAX_SIZE - 7):
            faults.append((position, pointer))
    if phase == "held":
        faults += [(2, original[2] - 1), (4, 1), (4, MAX_SIZE), (5, 0.), (5, float("nan")),
                   (9, original[9] - 1), (11, original[11] - 1), (9, MAX_SIZE),
                   (11, MAX_SIZE), (13, original[13] - 1), (15, original[15] + 1)]
        for request_index, bad in ((0, MAX_SIZE), (2, 0), (4, 8), (6, 0.),
                                   (6, float("nan")), (7, 0.), (7, float("inf")), (8, 0), (8, -1)):
            request = list(original[0])
            request[request_index] = bad
            faults.append((0, tuple(request)))
    else:
        for position in (1, 15, 17, 19, 21, 23, 25):
            faults.append((position, original[position] - 1))
        for position in (3, 5, 11, 13, 15, 17, 19, 21, 23, 25):
            faults.append((position, MAX_SIZE))
        faults += [(3, 1), (4, 0.), (4, float("nan")), (7, 0), (10, 0), (10, 99), (11, 0), (13, original[13] - 1)]
    for position, bad in faults:
        call.args[:] = original
        call.args[position] = bad
        before = call.snapshot()
        result = call.invoke()
        assert (result[0] if phase == "held" else result) == -1, (phase, position, bad)
        call.assert_atomic(before)
        if phase == "held":
            assert np.isposinf(result[1:4]).all()
            assert result[4] == -1 and result[5:12] == (0,) * 7
            assert result[12] == -1 and result[13:] == (0, 0, 0)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("phase", ("held", "multi"))
def test_integrated_nonfinite_inputs_reject_atomically(integrated_core, density, phase):
    for index in range(4 if phase == "held" else 2):
        call = held_call(integrated_core, (1, 1, 1, 1), density) if phase == "held" else multi_call(integrated_core, (1, 1, 1, 1), density)
        # Bounds permit NaNs as unset bounds, so use invalid ordered infinities instead.
        call.inputs[index][-1] = (float("nan") if index < 2 else (float("inf") if index == 2 else -float("inf")))
        before = call.snapshot()
        result = call.invoke()
        assert (result[0] if phase == "held" else result) == -1
        call.assert_atomic(before)


@pytest.mark.parametrize("density", range(8))
def test_integrated_multi_initial_failure_reports_only_attempted_work(integrated_core, density):
    call = multi_call(integrated_core, (0, 0, 1, 1), density, k=1)
    call.inputs[1].fill(1e308)
    before = [x.tobytes() for x in call.inputs]
    assert call.invoke() == -2
    summary = call.outputs[4][1]
    assert summary[0] == -2 and np.isposinf(summary[1:3]).all()
    np.testing.assert_array_equal(summary[3:], [0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert all(np.all(view == (INDEX_SENTINEL if i < 2 else SENTINEL))
               for i, (_, view) in enumerate(call.outputs) if i != 4)
    assert before == [x.tobytes() for x in call.inputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", range(8))
def test_integrated_held_failed_retarget_keeps_only_available_best(integrated_core, density):
    call = held_call(integrated_core, (1, 1, 1, 1), density)
    power_index = call.inputs[0].size - len(SHAPES[density]) - 1
    call.inputs[2][power_index] = call.inputs[0][power_index]
    call.inputs[3][power_index] = call.inputs[0][power_index]
    before = [x.tobytes() for x in call.inputs]
    result = call.invoke()
    assert result[0] == -2 and result[4] == -2
    assert np.isposinf(result[1]) and np.isfinite(result[2]) and np.isposinf(result[3])
    assert result[5:11] == (0, 1, 0, 0, 0, 0)
    assert np.all(call.outputs[0][1] == SENTINEL)
    np.testing.assert_array_equal(call.outputs[1][1], call.inputs[0])
    assert before == [x.tobytes() for x in call.inputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", range(8))
def test_integrated_held_initial_unavailable_does_not_invent_points(integrated_core, density):
    call = held_call(integrated_core, (0, 0, 1, 1), density)
    call.inputs[1].fill(1e308)
    before = [x.tobytes() for x in call.inputs]
    result = call.invoke()
    assert result[0] == -2 and result[4] == -101
    assert np.isposinf(result[1:4]).all()
    assert result[5:11] == (0, 4, 1, 0, 0, 0)
    assert result[14:] == (0, 0)
    assert all(np.all(view == SENTINEL) for _, view in call.outputs)
    assert before == [x.tobytes() for x in call.inputs]
    call.assert_canaries()


@pytest.mark.parametrize("density", range(8))
def test_integrated_multi_capacity_padding_is_not_written(integrated_core, density):
    call = multi_call(integrated_core, (1, 1, 1, 1), density, steps=1, padding=3)
    assert call.invoke() == 0
    required = (2, 2, call.inputs[0].size, call.inputs[0].size, 12, 12 + 2 * call.inputs[0].size)
    for (_, view), count in zip(call.outputs, required):
        fill = INDEX_SENTINEL if np.issubdtype(view.dtype, np.integer) else SENTINEL
        assert np.all(view[count:] == fill)
    call.assert_canaries()


def native_multi(core):
    library = ct.CDLL(core.__file__, mode=os.RTLD_LOCAL | os.RTLD_NOW)
    workspace = library.vlk_arma_aparch_multi_phase_workspace_bytes
    workspace.argtypes, workspace.restype = [ct.c_void_p], ct.c_size_t
    kind = library.vlk_arma_aparch_multi_phase_dispatch_kind
    kind.argtypes, kind.restype = [ct.c_void_p], ct.c_int32
    run = library.vlk_arma_aparch_multi_phase_run
    run.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_double, ct.c_size_t]
    run.argtypes += [item for _ in range(7) for item in (ct.c_void_p, ct.c_size_t)]
    run.restype = ct.c_int32
    return workspace, kind, run


@pytest.mark.parametrize("density", range(8))
def test_integrated_typed_metadata_ranges_and_aliases(integrated_core, density):
    workspace, kind, run = native_multi(integrated_core)
    for pointer in (0, MAX_SIZE - 7):
        assert workspace(pointer) == 0 and kind(pointer) == -1
        assert run(pointer, 0, 0, 0, 0, .5, 1, *([0, 0] * 7)) == -1
    for orders, k, runtime in LAYOUTS:
        spec = Specification(*orders, k, density, runtime)
        expected = 0 if runtime or orders not in ((1, 1, 1, 1), (2, 1, 3, 2)) else (2 if orders[0] == 1 else 1)
        assert kind(ct.addressof(spec)) == expected
        assert workspace(ct.addressof(spec)) == integrated_core._arma_aparch_multi_phase_workspace_bytes(*orders, density, k, runtime)
    call = multi_call(integrated_core, (1, 1, 1, 1), density)
    for position in MULTI_POINTERS:
        for offset in (0, 8):
            spec = Specification(1, 1, 1, 1, 2, density, False)
            args = list(call.args)
            args[position] = ct.addressof(spec) + offset
            before = call.snapshot(), bytes(spec)
            native_args = [ct.addressof(spec), *args[:5], args[11], *args[12:26]]
            assert run(*native_args) == -1
            assert before == (call.snapshot(), bytes(spec))


def test_integrated_python_parser_never_truncates_metadata(integrated_core):
    core = integrated_core
    for name in ("_arma_aparch_prepare", "_arma_aparch_preparation_workspace_bytes",
                 "_arma_aparch_multi_phase_run", "_arma_aparch_multi_phase_workspace_bytes"):
        with pytest.raises(TypeError):
            getattr(core, name)()
    for malformed in (None, [], (1,) * 8, (1,) * 11):
        with pytest.raises(TypeError):
            core._arma_aparch_preparation_workspace_bytes(malformed)
    with pytest.raises(OverflowError):
        core._arma_aparch_preparation_workspace_bytes((-1, 1, 1, 1, 0, False, 1.5, 1e-8, 2))
    class BadBoolean:
        def __bool__(self):
            raise RuntimeError("bad boolean")
    for density in (8, 2**32, MAX_SIZE):
        with pytest.raises((ValueError, OverflowError)):
            core._arma_aparch_multi_phase_workspace_bytes(1, 1, 1, 1, density, 2, False)
        assert core._arma_aparch_preparation_workspace_bytes((1, 1, 1, 1, density, False, 1.5, 1e-8, 2)) == 0
    for request in ((1, 1, 1, 1, 0, BadBoolean(), 1.5, 1e-8, 2),):
        with pytest.raises(RuntimeError, match="bad boolean"):
            core._arma_aparch_preparation_workspace_bytes(request)
    with pytest.raises(RuntimeError, match="bad boolean"):
        core._arma_aparch_multi_phase_workspace_bytes(1, 1, 1, 1, 0, 2, BadBoolean())
    for phase in ("held", "multi"):
        call = held_call(core, (1, 1, 1, 1), 0) if phase == "held" else multi_call(core, (1, 1, 1, 1), 0)
        count_index = 2 if phase == "held" else 1
        call.args[count_index] = -1
        before = call.snapshot()
        with pytest.raises(OverflowError):
            call.invoke()
        call.assert_atomic(before)
