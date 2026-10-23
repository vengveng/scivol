"""Independent multiprecision AD and checked anchored ARMA chart contracts.

The oracle filters design rows with an inverse lag polynomial, then differentiates
the chart quotient. It does not reuse the candidate's residual sensitivities.
Builds are private header-only images; no installed extension is imported.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
import pytest


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "localdev/coordinate_006_chief/anchored"
SIZE = ctypes.c_size_t
PTR = ctypes.c_void_p
SENTINEL = 271828.125
CASES = [(0, 0, 1, 5), (4, 0, 4, 7), (0, 5, 5, 6),
         (2, 3, 4, 5), (5, 4, 5, 7), (3, 2, 3, 5)]


class Spec(ctypes.Structure):
    _fields_ = [(name, SIZE) for name in
                ("p_ar", "q_ma", "start", "tail", "fixed", "order", "pivot", "count", "mutation")]

    @classmethod
    def make(cls, case=(3, 2, 3, 5), *, fixed=0, order=2, pivot=12, count=30, mutation=0):
        return cls(*case, fixed, order, pivot, count, mutation)

    @property
    def mean(self):
        return 1 + self.p_ar + self.q_ma

    @property
    def width(self):
        return self.mean + self.tail

    @property
    def packed(self):
        return self.width * (self.width + 1) // 2


class SequenceStep(ctypes.Structure):
    _fields_ = [(name, SIZE) for name in ("time", "null_gradient", "null_hessian", "reprepare")]


class PrecisionCounts(ctypes.Structure):
    _fields_ = [(name, SIZE) for name in ("replays", "observations", "unresolved")]


@pytest.fixture(scope="module", autouse=True)
def bounded_module():
    """The complete compilation and numerical shard has a 110-second cap."""
    deadline = time.monotonic() + 110
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError("anchored chart correctness shard exceeded 110 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 110)
    yield deadline
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def snapshot_sources(directory):
    pending = [ROOT / "tests/native/transfer_006_anchored_mean_chart.cpp", Path(__file__).resolve(),
               ROOT / "tests/conformance/oracles/mean_coordinates_006.py"]
    sources = {}
    while pending:
        path = pending.pop()
        relative = str(path.relative_to(ROOT))
        if relative in sources:
            continue
        content = path.read_bytes()
        sources[relative] = hashlib.sha256(content).hexdigest()
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', content.decode(), re.M):
            candidate = path.parent / include
            if not candidate.exists():
                candidate = ROOT / "core/src" / include
            if not candidate.exists():
                candidate = ROOT / "core/src" / include
            assert candidate.exists(), include
            pending.append(candidate.resolve())
    return sources


@pytest.fixture(scope="module")
def bridge(bounded_module):
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not shutil.which(compiler[0]):
        pytest.fail("A C++20 compiler is required")
    # Unique per invocation: the parent's concurrent header edits cannot change
    # the image being tested. Hashes below pin the exact copied owners.
    directory = REVIEW / f"tracked-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = snapshot_sources(directory)
    source = directory / "tests/native/transfer_006_anchored_mean_chart.cpp"
    frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
             if "clang" in compiler[0] or sys.platform == "darwin"
             else ["-Werror=frame-larger-than=16384"])
    common = [*compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
              "-march=native", "-fno-exceptions", "-fno-rtti", *frame,
              "-I", str(directory / "core/src"),
              "-I", str(directory / "core/src"),
              "-I", str(ROOT / "core/src/math/power/third_party/boost"),
              "-I", str(ROOT / "core/src/math/power/third_party/tlfloat")]
    library, executable, assembly = (directory / name for name in
                                     ("chart.so", "chart_contract", "chart.s"))
    commands = [
        [*common, "-fPIC", "-shared", str(source), "-o", str(library)],
        [*common, "-DANCHORED_CHART_EXECUTABLE", str(source), "-o", str(executable)],
        [*common, "-fPIC", "-S", str(source), "-o", str(assembly)],
    ]
    results = []
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=max(1, bounded_module - time.monotonic()))
        results.append(dict(command=command, returncode=result.returncode, stderr=result.stderr))
        (directory / "build.json").write_text(json.dumps(dict(
            source_hashes=hashes, builds=results,
            scope="Independent header-only chart contract; no installed image or solver"), indent=2) + "\n")
        assert result.returncode == 0, result.stderr
    changed = [path for path, digest in hashes.items()
               if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest]
    assert not changed, f"Header changed during compilation; rerun this shard: {changed}"
    lib = ctypes.CDLL(str(library))
    lib.anchor_workspace.argtypes = [ctypes.POINTER(Spec)]
    lib.anchor_workspace.restype = SIZE
    lib.anchor_run.argtypes = [ctypes.POINTER(Spec), PTR, PTR, PTR, SIZE, PTR, PTR, PTR, PTR]
    lib.anchor_run.restype = ctypes.c_int
    lib.anchor_sequence_run.argtypes = [ctypes.POINTER(Spec), PTR, PTR, PTR, SIZE, PTR,
                                       SIZE, ctypes.POINTER(SequenceStep), SIZE,
                                       PTR, PTR, PTR, PTR, PTR]
    lib.anchor_sequence_run.restype = ctypes.c_int
    lib.anchor_prepared_basis.argtypes = [ctypes.POINTER(Spec), PTR, PTR, PTR, SIZE, PTR]
    lib.anchor_prepared_basis.restype = ctypes.c_int
    lib.anchor_adaptive_workspace.argtypes = [ctypes.POINTER(Spec)]
    lib.anchor_adaptive_workspace.restype = SIZE
    lib.anchor_adaptive_sequence_run.argtypes = [ctypes.POINTER(Spec), PTR, SIZE, PTR, SIZE,
        PTR, SIZE, PTR, SIZE, PTR, ctypes.POINTER(SequenceStep), SIZE,
        ctypes.POINTER(PrecisionCounts), PTR, PTR, PTR, PTR, PTR, PTR]
    lib.anchor_adaptive_sequence_run.restype = ctypes.c_int
    record = json.loads((directory / "build.json").read_text())
    record["image_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    result = SimpleNamespace(lib=lib, directory=directory, executable=executable,
                             library=library, assembly=assembly, evidence=[])
    yield result
    (directory / "numerical_evidence.json").write_text(json.dumps(result.evidence, indent=2) + "\n")


def pointer(array):
    return None if array is None else array.ctypes.data


def sample(spec):
    rng = np.random.default_rng(1030 + 100 * spec.p_ar + spec.q_ma)
    data = np.ascontiguousarray(rng.normal(.2, .8, spec.count), dtype=np.float64)
    theta = np.r_[.07, rng.uniform(-.12, .12, spec.p_ar),
                  rng.uniform(-.15, .15, spec.q_ma), np.ones(spec.tail)]
    return np.ascontiguousarray(theta, dtype=np.float64), data


def call(bridge, spec, theta, data, *, short=0, misaligned=False, null_memory=False,
         null_gradient=False, null_hessian=False):
    required = bridge.lib.anchor_workspace(ctypes.byref(spec))
    memory = np.full(required + 64, 0xA5, dtype=np.uint8)
    offset = 17 if misaligned else 16
    values = np.full(spec.count + 2, SENTINEL, dtype=np.float64)
    gradient = np.full(spec.count * spec.width + 2, SENTINEL, dtype=np.float64)
    hessian = np.full(spec.count * spec.packed + 2, SENTINEL, dtype=np.float64)
    evidence = np.full(8, SENTINEL, dtype=np.float64)
    status = bridge.lib.anchor_run(
        ctypes.byref(spec), pointer(theta), pointer(data),
        None if null_memory else memory.ctypes.data + offset, max(0, required - short),
        values[1:].ctypes.data, None if null_gradient else gradient[1:].ctypes.data,
        None if null_hessian else hessian[1:].ctypes.data, evidence[1:].ctypes.data)
    np.testing.assert_array_equal(memory[:offset], 0xA5)
    np.testing.assert_array_equal(memory[offset + required:], 0xA5)
    for output in (values, gradient, hessian, evidence):
        np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
    return SimpleNamespace(status=status, value=values[1:-1],
                           gradient=gradient[1:-1].reshape(spec.count, spec.width),
                           hessian=hessian[1:-1].reshape(spec.count, spec.packed),
                           evidence=evidence[1:-1])


def sequence_call(bridge, spec, theta, data, steps, *, provider):
    required = bridge.lib.anchor_workspace(ctypes.byref(spec))
    memory = np.full(required + 64, 0xA5, dtype=np.uint8)
    snapshot = np.empty(required, dtype=np.uint8)
    count = len(steps)
    script = (SequenceStep * count)(*steps)
    statuses = np.full(count + 2, -17, dtype=np.int32)
    values = np.full(count + 2, SENTINEL)
    gradient = np.full(count * spec.width + 2, SENTINEL)
    hessian = np.full(count * spec.packed + 2, SENTINEL)
    marker = np.iinfo(np.uintp).max
    evidence = np.full(count * 5 + 2, marker, dtype=np.uintp)
    status = bridge.lib.anchor_sequence_run(
        ctypes.byref(spec), pointer(theta), pointer(data), memory.ctypes.data + 16,
        required, pointer(snapshot), provider, script, count, statuses[1:].ctypes.data,
        values[1:].ctypes.data, gradient[1:].ctypes.data, hessian[1:].ctypes.data,
        evidence[1:].ctypes.data)
    assert status == 0, f"sequence preparation failed at step {status}"
    np.testing.assert_array_equal(memory[:16], 0xA5)
    np.testing.assert_array_equal(memory[16 + required:], 0xA5)
    np.testing.assert_array_equal(statuses[[0, -1]], -17)
    np.testing.assert_array_equal(evidence[[0, -1]], marker)
    for output in (values, gradient, hessian):
        np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
    return SimpleNamespace(status=statuses[1:-1], value=values[1:-1],
                           gradient=gradient[1:-1].reshape(count, spec.width),
                           hessian=hessian[1:-1].reshape(count, spec.packed),
                           evidence=evidence[1:-1].reshape(count, 5))


def adaptive_call(bridge, spec, theta, data, steps=None, *, chart=False, counts=None,
                  short=0, null_arena=False):
    theta = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1, spec.width)
    points = len(theta)
    if steps is None:
        steps = [SequenceStep(t, 0, spec.order == 1, 0) for t in range(spec.start, spec.count)]
    script = (SequenceStep * len(steps))(*steps)
    rows = points * len(steps)
    primal_bytes = bridge.lib.anchor_adaptive_workspace(ctypes.byref(spec))
    chart_bytes = bridge.lib.anchor_workspace(ctypes.byref(spec)) if chart else 0
    arenas = [np.full(size + 64, 0xA5, dtype=np.uint8) for size in (primal_bytes, chart_bytes)]
    snapshot = np.empty(primal_bytes + chart_bytes, dtype=np.uint8)
    statuses = np.full(rows + 2, -17, dtype=np.int32)
    outputs = [np.full(size + 2, SENTINEL) for size in
               (rows, rows * spec.width, rows * spec.packed, rows * 2)]
    marker = np.iinfo(np.uintp).max
    evidence = np.full(rows * 6 + 2, marker, dtype=np.uintp)
    before = theta.copy(), data.copy()
    status = bridge.lib.anchor_adaptive_sequence_run(
        ctypes.byref(spec), pointer(theta), points, pointer(data), chart,
        None if null_arena else arenas[0].ctypes.data + 16, max(0, primal_bytes - short),
        arenas[1].ctypes.data + 16 if chart else None, chart_bytes, pointer(snapshot),
        script, len(steps), None if counts is None else ctypes.byref(counts),
        statuses[1:].ctypes.data, *(output[1:].ctypes.data for output in outputs),
        evidence[1:].ctypes.data)
    np.testing.assert_array_equal(theta, before[0])
    np.testing.assert_array_equal(data, before[1])
    for arena, size in zip(arenas, (primal_bytes, chart_bytes)):
        np.testing.assert_array_equal(arena[:16], 0xA5)
        np.testing.assert_array_equal(arena[16 + size:], 0xA5)
    np.testing.assert_array_equal(statuses[[0, -1]], -17)
    np.testing.assert_array_equal(evidence[[0, -1]], marker)
    for output in outputs:
        np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
    return SimpleNamespace(status=status, accepted=statuses[1:-1].reshape(points, -1),
        value=outputs[0][1:-1].reshape(points, -1),
        gradient=outputs[1][1:-1].reshape(points, -1, spec.width),
        hessian=outputs[2][1:-1].reshape(points, -1, spec.packed),
        limbs=outputs[3][1:-1].reshape(points, -1, 2),
        evidence=evidence[1:-1].reshape(points, -1, 6), primal_bytes=primal_bytes)


from oracles.mean_coordinates_006 import Jet, inverse_lag_polynomial, filtered_rows


def chart_oracle(spec, theta, data):
    with mp.workdps(100):
        origin = list(map(lambda x: mp.mpf(float(x)), theta[:spec.mean]))
        scalar_rows = filtered_rows(spec, origin, data)
        pivot_a, pivot_b = scalar_rows[spec.pivot - spec.start]
        pivot_value = pivot_a - pivot_b * origin[0]
        error = -pivot_value / pivot_b
        w = [Jet.variable(0 if i == 0 else value, spec.mean, i)
             for i, value in enumerate(origin)]
        rows = filtered_rows(spec, w, data)
        a, b = rows[spec.pivot - spec.start]
        intercept = (a - w[0]) / b + error
        residuals = [a - b * intercept for a, b in rows]
        return residuals, (error, pivot_value, pivot_b), rows


def exact_conditioned_residuals(spec, theta, data):
    values = [Fraction(0)] * spec.count
    for t in range(spec.start, spec.count):
        values[t] = Fraction(float(data[t])) - Fraction(float(theta[0]))
        for lag in range(1, spec.p_ar + 1):
            values[t] -= Fraction(float(theta[lag])) * Fraction(float(data[t - lag]))
        for lag in range(1, spec.q_ma + 1):
            values[t] -= Fraction(float(theta[spec.p_ar + lag])) * values[t - lag]
    return values


def test_mp_ad_oracle_against_closed_form_quotient():
    with mp.workdps(100):
        a, b = Jet.variable(mp.mpf(".125"), 2, 0), Jet.variable(mp.mpf(".25"), 2, 1)
        f = a * b / (1 - b)
        expected_g = [b.value / (1 - b.value), a.value / (1 - b.value)**2]
        expected_h = [0, 1 / (1 - b.value)**2, 1 / (1 - b.value)**2,
                      2 * a.value / (1 - b.value)**3]
        for actual, expected in zip(f.gradient + f.hessian, expected_g + expected_h):
            assert abs(actual - expected) < mp.mpf("1e-98")


def assert_oracle(spec, actual, residuals):
    assert actual.status == 0
    pairs = [(i, j) for i in range(spec.width) for j in range(i, spec.width)]
    expected_value = np.array([float(x.value) for x in residuals])
    expected_g = np.zeros((len(residuals), spec.width))
    expected_h = np.zeros((len(residuals), spec.packed))
    for t, residual in enumerate(residuals):
        expected_g[t, :spec.mean] = list(map(float, residual.gradient))
        for k, (i, j) in enumerate(pairs):
            if j < spec.mean:
                expected_h[t, k] = float(residual.hessian[i * spec.mean + j])
    np.testing.assert_allclose(actual.value[spec.start:], expected_value, rtol=2e-15, atol=1e-30)
    np.testing.assert_allclose(actual.gradient[spec.start:], expected_g, rtol=2e-14, atol=2e-14)
    if spec.order == 2:
        np.testing.assert_allclose(actual.hessian[spec.start:], expected_h, rtol=2e-14, atol=2e-14)
    else:
        np.testing.assert_array_equal(actual.hessian, SENTINEL)
    np.testing.assert_array_equal(actual.value[:spec.start], SENTINEL)
    np.testing.assert_array_equal(actual.gradient[:spec.start], SENTINEL)
    np.testing.assert_array_equal(actual.hessian[:spec.start], SENTINEL)
    np.testing.assert_array_equal(actual.gradient[spec.start:, spec.mean:], 0)
    return dict(value_max_abs=float(np.max(np.abs(actual.value[spec.start:] - expected_value))),
                gradient_max_scaled=float(np.max(np.abs(actual.gradient[spec.start:] - expected_g)
                                                  / (1 + np.abs(expected_g)))),
                hessian_max_scaled=(float(np.max(np.abs(actual.hessian[spec.start:] - expected_h)
                                                / (1 + np.abs(expected_h)))) if spec.order == 2 else None))


def adaptive_fixture(kind):
    spec = Spec.make((5, 4, 5, 7), pivot=18, count=28)
    data = np.random.default_rng(44120).normal(size=spec.count)
    theta = np.array([.03, .12, -.04, .02, .01, -.025,
                      .15, -.08, .03, -.01, *np.ones(spec.tail)])
    if kind == "nonzero-cancellation":
        theta[:spec.mean] = [0., 1 + 2.**-27, 0., 0., 0., 0., -.2, 0., 0., 0.]
        data[spec.start - 1], data[spec.start] = 1 - 2.**-27, 1.
    elif kind == "exact-zero":
        theta[:spec.mean] = [0., .123456789, 0., 0., 0., 0., -.123456789, 0., 0., 0.]
        data[spec.start - 1] = data[-1] = 0.
    elif kind != "ordinary":
        raise ValueError(kind)
    return spec, theta, np.ascontiguousarray(data)


def assert_mp_provider(spec, actual, theta, data, point=0):
    assert actual.status == 0
    np.testing.assert_array_equal(actual.accepted[point], 1)
    exact = exact_conditioned_residuals(spec, theta, data)[spec.start:]
    np.testing.assert_array_equal(actual.value[point], list(map(float, exact)))
    with mp.workdps(100):
        coordinates = [mp.mpf(float(x)) for x in theta[:spec.mean]]
        rows = filtered_rows(spec, coordinates, data)
        scale = max(abs(mp.mpf(float(x))) for x in data)
        errors = []
        for (a, b), limbs in zip(rows, actual.limbs[point]):
            expected = a - b * coordinates[0]
            value = mp.mpf(float(limbs[0])) + mp.mpf(float(limbs[1]))
            tolerance = mp.mpf("3e-29") * abs(expected) + mp.mpf("1e-90") * scale
            assert abs(value - expected) <= tolerance, (value, expected, tolerance)
            errors.append(float(abs(value - expected) / scale))
    np.testing.assert_array_equal(actual.gradient[point], SENTINEL)
    np.testing.assert_array_equal(actual.hessian[point], SENTINEL)
    return max(errors)


def adaptive_chart_projection(spec, actual, point=0):
    assert actual.status == 0
    np.testing.assert_array_equal(actual.accepted[point], 1)
    return SimpleNamespace(status=0,
        value=np.r_[np.full(spec.start, SENTINEL), actual.value[point]],
        gradient=np.vstack((np.full((spec.start, spec.width), SENTINEL), actual.gradient[point])),
        hessian=np.vstack((np.full((spec.start, spec.packed), SENTINEL), actual.hessian[point])))


@pytest.mark.parametrize("kind", ["ordinary", "nonzero-cancellation", "exact-zero"])
def test_shared_adaptive_provider_mp_residuals_and_chart_ad(bridge, kind):
    spec, theta, data = adaptive_fixture(kind)
    seed = (7, 11, 13)
    counter = PrecisionCounts(*seed)
    runtime = adaptive_call(bridge, spec, theta, data, counts=counter)
    residual_error = assert_mp_provider(spec, runtime, theta, data)
    delta = runtime.evidence[0, -1, 3:] - np.array(seed, dtype=np.uintp)
    assert tuple(runtime.evidence[0, -1, 3:]) == (counter.replays, counter.observations, counter.unresolved)
    assert counter.unresolved == seed[2]
    if kind == "ordinary":
        np.testing.assert_array_equal(delta, 0)
    else:
        assert delta[0] > 0 and delta[1] > 0
    if kind == "nonzero-cancellation":
        assert runtime.value[0, 0] == 2.**-54 != 0
        assert data[spec.start] - theta[1] * data[spec.start - 1] == 0
        assert runtime.evidence[0, 0, 3] > seed[0]
    if kind == "exact-zero":
        assert runtime.value[0, -1] == 0
        # Recovery is selective and happens at the zero, not every observation.
        np.testing.assert_array_equal(runtime.evidence[0, :-1, 3:], np.tile(seed, (spec.count - spec.start - 1, 1)))
    spec.fixed = 1
    fixed = adaptive_call(bridge, spec, theta, data, counts=PrecisionCounts(*seed))
    for name in ("value", "limbs", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(fixed, name), getattr(runtime, name))
    residuals, _, _ = chart_oracle(spec, theta, data)
    chart_counter = PrecisionCounts(*seed)
    second = adaptive_call(bridge, spec, theta, data, chart=True, counts=chart_counter)
    errors = assert_oracle(spec, adaptive_chart_projection(spec, second), residuals)
    np.testing.assert_array_equal(second.value, runtime.value)
    assert chart_counter.unresolved == seed[2]
    for fixed_order, order in ((0, 2), (0, 1), (1, 1)):
        spec.fixed, spec.order = fixed_order, order
        actual = adaptive_call(bridge, spec, theta, data, chart=True)
        assert_oracle(spec, adaptive_chart_projection(spec, actual), residuals)
        np.testing.assert_array_equal(actual.value, second.value)
        np.testing.assert_array_equal(actual.gradient, second.gradient)
        np.testing.assert_array_equal(actual.hessian, second.hessian if order == 2 else SENTINEL)
    bridge.evidence.append(dict(case=f"shared-provider-{kind}",
        orders=[spec.p_ar, spec.q_ma], residual_max_scaled=residual_error,
        provider_count_delta=list(map(int, delta)), chart_errors=errors))


@pytest.mark.parametrize("chart", [False, True], ids=["provider", "chart"])
@pytest.mark.parametrize("fixed", [0, 1], ids=["runtime", "fixed"])
def test_shared_adaptive_summary_reused_across_points_and_caller_counters(bridge, chart, fixed):
    spec, theta, data = adaptive_fixture("nonzero-cancellation")
    spec.fixed = fixed
    ordinary = adaptive_fixture("ordinary")[1]
    points = np.array([theta, ordinary, theta.copy()])
    points[2, 0], points[2, 1], points[2, 1 + spec.p_ar] = .015, -.09, .08
    seed = np.array([19, 23, 29], dtype=np.uintp)
    counts = PrecisionCounts(*map(int, seed))
    batch = adaptive_call(bridge, spec, points, data, chart=chart, counts=counts)
    expected_counts = seed.copy()
    for point, parameters in enumerate(points):
        fresh_counts = PrecisionCounts()
        fresh = adaptive_call(bridge, spec, parameters, data, chart=chart, counts=fresh_counts)
        expected_counts += np.array([fresh_counts.replays, fresh_counts.observations, fresh_counts.unresolved], dtype=np.uintp)
        for name in ("value", "limbs", "gradient", "hessian"):
            np.testing.assert_array_equal(getattr(batch, name)[point], getattr(fresh, name)[0])
        np.testing.assert_array_equal(batch.evidence[point, -1, 3:], expected_counts)
        if chart:
            residuals, _, _ = chart_oracle(spec, parameters, data)
            assert_oracle(spec, adaptive_chart_projection(spec, batch, point), residuals)
        else:
            assert_mp_provider(spec, batch, parameters, data, point)
    assert counts.replays > seed[0] and counts.observations > seed[1]
    assert counts.unresolved == seed[2]
    np.testing.assert_array_equal([counts.replays, counts.observations, counts.unresolved], expected_counts)


@pytest.mark.parametrize("chart", [False, True], ids=["provider", "chart"])
@pytest.mark.parametrize("fixed", [0, 1], ids=["runtime", "fixed"])
@pytest.mark.parametrize("order", [1, 2])
def test_shared_adaptive_rejections_rewind_and_failed_prepare_preserve_lifecycle(bridge, chart, fixed, order):
    spec, theta, data = adaptive_fixture("nonzero-cancellation")
    spec.fixed, spec.order = fixed, order
    control = adaptive_call(bridge, spec, theta, data, chart=chart)
    assert control.status == 0
    np.testing.assert_array_equal(control.accepted, 1)
    steps, expected = [], []
    for t in range(spec.start, spec.count):
        if t in (spec.start, spec.pivot, spec.count - 1):
            rejected = [SequenceStep(time, 0, 0, 0) for time in
                        (t + 1, t - 1, spec.start - 1, spec.count, np.iinfo(np.uintp).max)]
            if chart:
                rejected += [SequenceStep(t, 1, 0, 0)]
                if order == 2:
                    rejected += [SequenceStep(t, 0, 1, 0)]
            steps.extend(rejected)
            expected.extend([0] * len(rejected))
        steps.append(SequenceStep(t, 0, order == 1, 0))
        expected.append(1)
    # Rewind after exhaustion, reject the stale time, then recover the same prefix.
    steps.append(SequenceStep(spec.count, 0, 0, 2))
    expected.append(0)
    steps.extend(SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.pivot + 1))
    expected.extend([1] * (spec.pivot + 1 - spec.start))
    # Rewind partially consumed replay checkpoints as well as fast history.
    steps.append(SequenceStep(spec.pivot + 1, 0, 0, 2))
    expected.append(0)
    steps.extend(SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.pivot + 1))
    expected.extend([1] * (spec.pivot + 1 - spec.start))
    steps.extend([SequenceStep(spec.pivot + 1, 0, 0, 3), SequenceStep(spec.start, 0, 0, 0),
                  SequenceStep(spec.count, 0, 0, 1)])
    expected.extend([0, 0, 0])
    steps.extend(SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.count))
    expected.extend([1] * (spec.count - spec.start))
    counts = PrecisionCounts(31, 37, 41)
    actual = adaptive_call(bridge, spec, theta, data, steps, chart=chart, counts=counts)
    assert actual.status == 0
    np.testing.assert_array_equal(actual.accepted[0], expected)
    accepted = np.asarray(expected, dtype=bool)
    times = [step.time - spec.start for step, ok in zip(steps, expected) if ok]
    for name in ("value", "limbs", "gradient", "hessian"):
        output = getattr(actual, name)[0]
        np.testing.assert_array_equal(output[~accepted], SENTINEL)
        np.testing.assert_array_equal(output[accepted], getattr(control, name)[0, times])
    np.testing.assert_array_equal(actual.evidence[0, ~accepted, :3], 1)
    assert counts.replays > 31 and counts.observations > 37 and counts.unresolved == 41


@pytest.mark.parametrize("fixed", [0, 1], ids=["runtime", "fixed"])
def test_shared_adaptive_zero_ma_needs_no_arena_even_for_recovery(bridge, fixed):
    spec = Spec.make((4, 0, 4, 7), fixed=fixed, pivot=11, count=18)
    theta, data = sample(spec)
    theta[0] = 0.
    data = np.ldexp(data, -920)
    counts = PrecisionCounts(43, 47, 53)
    actual = adaptive_call(bridge, spec, theta, data, counts=counts, null_arena=True)
    assert actual.primal_bytes == 0
    assert_mp_provider(spec, actual, theta, data)
    assert counts.replays > 43 and counts.observations > 47 and counts.unresolved == 53
    chart = adaptive_call(bridge, spec, theta, data, chart=True, null_arena=True)
    residuals, _, _ = chart_oracle(spec, theta, data)
    assert_oracle(spec, adaptive_chart_projection(spec, chart), residuals)
    np.testing.assert_array_equal(chart.value, actual.value)


@pytest.mark.parametrize("chart", [False, True], ids=["provider", "chart"])
def test_shared_adaptive_short_arena_rejects_without_output_or_counter_writes(bridge, chart):
    spec, theta, data = adaptive_fixture("ordinary")
    counts = PrecisionCounts(59, 61, 67)
    actual = adaptive_call(bridge, spec, theta, data, chart=chart, counts=counts, short=1)
    assert actual.status == -1
    np.testing.assert_array_equal(actual.accepted, -17)
    for name in ("value", "limbs", "gradient", "hessian"):
        np.testing.assert_array_equal(getattr(actual, name), SENTINEL)
    assert (counts.replays, counts.observations, counts.unresolved) == (59, 61, 67)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pivot_position", ["startup", "later", "last"])
def test_residual_quotient_ad_arbitrary_orders_and_fixed_projection(bridge, case, pivot_position):
    spec = Spec.make(case)
    spec.pivot = {"startup": spec.start, "later": spec.start + 7, "last": spec.count - 1}[pivot_position]
    theta, data = sample(spec)
    residuals, evidence, _ = chart_oracle(spec, theta, data)
    full = call(bridge, spec, theta, data)
    errors = assert_oracle(spec, full, residuals)
    exact = exact_conditioned_residuals(spec, theta, data)
    np.testing.assert_array_equal(full.value[spec.start:], list(map(float, exact[spec.start:])))
    with mp.workdps(100):
        for k, expected in enumerate(evidence):
            actual = mp.mpf(float(full.evidence[2 * k])) + mp.mpf(float(full.evidence[2 * k + 1]))
            assert abs(actual - expected) <= mp.mpf("3e-29") * max(1, abs(expected))
    spec.fixed = 1
    fixed = call(bridge, spec, theta, data)
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(fixed, name), getattr(full, name))
    spec.order = 1
    first = call(bridge, spec, theta, data, null_hessian=True)
    np.testing.assert_array_equal(first.value, full.value)
    np.testing.assert_array_equal(first.gradient, full.gradient)
    np.testing.assert_array_equal(first.hessian, SENTINEL)
    bridge.evidence.append(dict(case=case, pivot=int(spec.pivot), errors=errors))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("order", [1, 2])
def test_prepared_coefficient_basis_matches_independent_filtered_design(bridge, case, order):
    spec = Spec.make(case, order=order, pivot=case[2] + 7)
    theta, data = sample(spec)
    for fixed in (0, 1):
        spec.fixed = fixed
        required = bridge.lib.anchor_workspace(ctypes.byref(spec))
        memory = np.empty(required, dtype=np.uint8)
        output = np.full(2 * (spec.p_ar + 1) + 2, SENTINEL)
        assert bridge.lib.anchor_prepared_basis(ctypes.byref(spec), theta.ctypes.data,
            data.ctypes.data, memory.ctypes.data, required, output[1:].ctypes.data) == 0
        np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
        with mp.workdps(100):
            weights = inverse_lag_polynomial(list(map(mp.mpf, theta[1 + spec.p_ar:spec.mean])),
                                             spec.pivot - spec.start + 1)
            for column in range(spec.p_ar + 1):
                expected = sum((weights[spec.pivot - t]
                                * (1 if column == 0 else mp.mpf(float(data[t - column])))
                                for t in range(spec.start, spec.pivot + 1)), 0)
                actual = mp.mpf(float(output[1 + 2 * column])) + mp.mpf(float(output[2 + 2 * column]))
                assert abs(actual - expected) <= mp.mpf("3e-29") * max(1, abs(expected))


def test_nonzero_anchor_retains_actual_residual_and_ma_curvature(bridge):
    spec = Spec.make(pivot=12)
    theta, data = sample(spec)
    actual = call(bridge, spec, theta, data)
    residuals, (error, pivot, _), rows = chart_oracle(spec, theta, data)
    assert_oracle(spec, actual, residuals)
    assert abs(error) > mp.mpf(".01") and abs(pivot) > mp.mpf(".01")
    assert actual.value[spec.pivot] != 0
    basis = rows[spec.pivot - spec.start][1]
    expected = residuals[spec.pivot - spec.start]
    with mp.workdps(100):
        for i in range(spec.mean):
            assert abs(expected.gradient[i] - (int(i == 0) - error * basis.gradient[i])) < mp.mpf("1e-90")
        for i in range(spec.mean**2):
            assert abs(expected.hessian[i] + error * basis.hessian[i]) < mp.mpf("1e-90")
    ma = 1 + spec.p_ar
    assert abs(expected.gradient[ma]) > mp.mpf(".01")
    assert abs(expected.hessian[ma * spec.mean + ma]) > mp.mpf(".01")
    # Non-pivot rho/MA curvature must also survive the nonlinear intercept map.
    assert abs(residuals[0].hessian[ma]) > mp.mpf(".01")


def test_exact_rational_nonzero_cancellation_anchor(bridge):
    spec = Spec.make((1, 1, 1, 5), pivot=1, count=4)
    theta = np.array([0., 1 + 2.**-27, -.2, *np.ones(5)])
    data = np.array([1 - 2.**-27, 1., .6, -.4])
    exact = Fraction(data[1]) - Fraction(theta[1]) * Fraction(data[0])
    assert exact == Fraction(1, 2**54)
    actual = call(bridge, spec, theta, data)
    assert actual.status == 0
    assert actual.value[1] == float(exact) != 0
    assert Fraction(actual.evidence[2]) + Fraction(actual.evidence[3]) == exact
    assert Fraction(actual.evidence[0]) + Fraction(actual.evidence[1]) == -exact
    residuals, _, _ = chart_oracle(spec, theta, data)
    assert_oracle(spec, actual, residuals)


def test_near_singular_basis_has_large_exact_sensitivities_without_padding(bridge):
    spec = Spec.make((0, 1, 1, 5), pivot=2, count=20)
    theta, data = sample(spec)
    theta[1] = 1 - 2.**-20
    actual = call(bridge, spec, theta, data)
    residuals, (_, _, basis), _ = chart_oracle(spec, theta, data)
    assert basis == mp.mpf(2)**-20
    assert max(abs(x) for row in residuals for x in row.hessian) > 1e10
    errors = assert_oracle(spec, actual, residuals)
    bridge.evidence.append(dict(case="near-singular-basis", basis=str(basis),
                                largest_hessian=str(max(abs(x) for row in residuals for x in row.hessian)),
                                errors=errors))


def test_long_horizon_near_boundary_ma_derivatives(bridge):
    spec = Spec.make((0, 1, 1, 5), pivot=177, count=192)
    theta, data = sample(spec)
    theta[1] = -.999
    residuals, _, _ = chart_oracle(spec, theta, data)
    assert max(abs(x) for row in residuals for x in row.hessian) > 1000
    second = call(bridge, spec, theta, data)
    errors = assert_oracle(spec, second, residuals)
    spec.order = 1
    first = call(bridge, spec, theta, data)
    assert_oracle(spec, first, residuals)
    np.testing.assert_array_equal(first.value, second.value)
    np.testing.assert_array_equal(first.gradient, second.gradient)
    bridge.evidence.append(dict(case="long-horizon-ma", count=192, errors=errors))


def test_density_and_volatility_parameters_do_not_enter_mean_chart(bridge):
    spec = Spec.make()
    theta, data = sample(spec)
    baseline = call(bridge, spec, theta, data)
    # Their domain validation belongs to the caller, not the mean policy.
    theta[spec.mean:] = [np.nan, np.inf, -np.inf, 1e308, -1e308]
    unrelated = call(bridge, spec, theta, data)
    assert unrelated.status == 0
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(unrelated, name), getattr(baseline, name))


def sequencing_control(bridge, spec, theta, data, provider):
    steps = [SequenceStep(t, 0, spec.order == 1, 0) for t in range(spec.start, spec.count)]
    control = sequence_call(bridge, spec, theta, data, steps, provider=provider)
    np.testing.assert_array_equal(control.status, 1)
    if not provider:
        baseline = call(bridge, spec, theta, data)
        residuals, _, _ = chart_oracle(spec, theta, data)
        assert_oracle(spec, baseline, residuals)
        for name in ("value", "gradient", "hessian"):
            np.testing.assert_array_equal(getattr(control, name), getattr(baseline, name)[spec.start:])
    return control


def assert_sequence_control(spec, actual, steps, expected, control, provider):
    np.testing.assert_array_equal(actual.status, expected)
    accepted = np.asarray(expected, dtype=bool)
    times = np.array([step.time - spec.start for step, ok in zip(steps, expected) if ok])
    for name in ("value", "gradient", "hessian"):
        output = getattr(actual, name)
        np.testing.assert_array_equal(output[~accepted], SENTINEL)
        np.testing.assert_array_equal(output[accepted], getattr(control, name)[times])
    # Evidence snapshots are taken after any intentional reprepare, before advance.
    np.testing.assert_array_equal(actual.evidence[~accepted, :2], 1)
    if provider:
        np.testing.assert_array_equal(actual.evidence[accepted, 4],
                                      [step.time + 1 for step, ok in zip(steps, expected) if ok])
        resets = np.array([bool(step.reprepare) for step in steps])
        for i in np.flatnonzero(~accepted & ~resets):
            if i:
                np.testing.assert_array_equal(actual.evidence[i, 2:], actual.evidence[i - 1, 2:])
    if spec.order == 1:
        np.testing.assert_array_equal(actual.hessian, SENTINEL)


@pytest.mark.parametrize("case", [(4, 0, 4, 7), (3, 2, 3, 5)], ids=["zero-ma", "nonzero-ma"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("fixed", [0, 1], ids=["runtime", "fixed"])
@pytest.mark.parametrize("provider", [0, 1], ids=["default", "recording-provider"])
def test_call_sequencing_rejections_preserve_state(bridge, case, order, fixed, provider):
    spec = Spec.make(case, order=order, fixed=fixed, pivot=case[2] + 4, count=14)
    theta, data = sample(spec)
    control = sequencing_control(bridge, spec, theta, data, provider)
    steps, expected = [], []
    checkpoints = {spec.start, spec.start + spec.q_ma + 1, spec.pivot, spec.count - 1}
    for t in range(spec.start, spec.count):
        if t in checkpoints:
            rejected = [SequenceStep(time, 0, 0, 0) for time in
                        (t + 1, t - 1, spec.start - 1, spec.count, np.iinfo(np.uintp).max)]
            rejected += [SequenceStep(t, 1, 0, 0), SequenceStep(t, 1, 1, 0)]
            if order == 2:
                rejected.append(SequenceStep(t, 0, 1, 0))
            steps.extend(rejected)
            expected.extend([0] * len(rejected))
        steps.append(SequenceStep(t, 0, order == 1, 0))
        expected.append(1)
    steps.extend(SequenceStep(t, 0, 0, 0) for t in (spec.count - 1, spec.count))
    expected.extend([0, 0])
    actual = sequence_call(bridge, spec, theta, data, steps, provider=provider)
    assert_sequence_control(spec, actual, steps, expected, control, provider)


@pytest.mark.parametrize("case", [(4, 0, 4, 7), (3, 2, 3, 5)], ids=["zero-ma", "nonzero-ma"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("fixed", [0, 1], ids=["runtime", "fixed"])
@pytest.mark.parametrize("provider", [0, 1], ids=["default", "recording-provider"])
def test_reprepare_resets_call_sequence(bridge, case, order, fixed, provider):
    spec = Spec.make(case, order=order, fixed=fixed, pivot=case[2] + 4, count=14)
    theta, data = sample(spec)
    control = sequencing_control(bridge, spec, theta, data, provider)
    steps = [SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.pivot + 1)]
    expected = [1] * len(steps)
    # Reset a partially consumed chart, then reject its stale next observation.
    steps.append(SequenceStep(spec.pivot + 1, 0, 0, 1))
    expected.append(0)
    steps.extend(SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.count))
    expected.extend([1] * (spec.count - spec.start))
    # Exhaustion must not prevent another reset or permit a stale in-range call.
    steps.extend([SequenceStep(spec.count, 0, 0, 1), SequenceStep(spec.count - 1, 0, 0, 0)])
    expected.extend([0, 0])
    steps.extend(SequenceStep(t, 0, order == 1, 0) for t in range(spec.start, spec.count))
    expected.extend([1] * (spec.count - spec.start))
    actual = sequence_call(bridge, spec, theta, data, steps, provider=provider)
    assert_sequence_control(spec, actual, steps, expected, control, provider)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("fixed", [0, 1])
@pytest.mark.parametrize("short", [1, 16, 10000])
def test_short_workspace_rejects_without_output_writes(bridge, order, fixed, short):
    spec = Spec.make(order=order, fixed=fixed)
    theta, data = sample(spec)
    actual = call(bridge, spec, theta, data, short=short)
    assert actual.status == -1
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(actual, name), SENTINEL)


def test_unaligned_exact_size_arena_rejects_safely(bridge):
    spec = Spec.make()
    theta, data = sample(spec)
    actual = call(bridge, spec, theta, data, misaligned=True)
    assert actual.status == -1
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(actual, name), SENTINEL)


@pytest.mark.parametrize("mutation", list(range(1, 8)) + [11, 12])
def test_invalid_mean_layouts_reject(bridge, mutation):
    spec = Spec.make(mutation=mutation)
    assert bridge.lib.anchor_workspace(ctypes.byref(spec)) == 0


@pytest.mark.parametrize("mutation", [8, 9, 10])
def test_invalid_output_layout_rejects_before_projection(bridge, mutation):
    spec = Spec.make(mutation=mutation)
    assert bridge.lib.anchor_workspace(ctypes.byref(spec)) == 0


@pytest.mark.parametrize("mutation", range(1, 13))
def test_fixed_metadata_ignores_poisoned_runtime_layout(bridge, mutation):
    spec = Spec.make(fixed=1)
    theta, data = sample(spec)
    normal = call(bridge, spec, theta, data)
    required = bridge.lib.anchor_workspace(ctypes.byref(spec))
    spec.mutation = mutation
    assert bridge.lib.anchor_workspace(ctypes.byref(spec)) == required
    poisoned = call(bridge, spec, theta, data)
    assert poisoned.status == 0
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(poisoned, name), getattr(normal, name))


@pytest.mark.parametrize("bad", ["theta", "data", "memory", "pivot-before", "pivot-after", "empty"])
def test_invalid_pointer_pivot_and_range_reject(bridge, bad):
    spec = Spec.make()
    theta, data = sample(spec)
    if bad == "pivot-before":
        spec.pivot = spec.start - 1
    if bad == "pivot-after":
        spec.pivot = spec.count
    if bad == "empty":
        spec.count = 0
    actual = call(bridge, spec, None if bad == "theta" else theta,
                  None if bad == "data" else data, null_memory=bad == "memory")
    assert actual.status == -1
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(actual, name), SENTINEL)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("location", ["theta-intercept", "theta-ar", "theta-ma", "data-before", "data-at", "data-after"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_used_inputs_reject_without_failed_row_projection(bridge, order, location, bad):
    spec = Spec.make(order=order)
    theta, data = sample(spec)
    index = {"theta-intercept": 0, "theta-ar": 1, "theta-ma": 1 + spec.p_ar,
             "data-before": spec.start - 1, "data-at": spec.pivot,
             "data-after": spec.pivot + 1}[location]
    (theta if location.startswith("theta") else data)[index] = bad
    actual = call(bridge, spec, theta, data)
    if location == "data-after":
        assert actual.status == index + 1
        np.testing.assert_array_equal(actual.value[index:], SENTINEL)
        np.testing.assert_array_equal(actual.gradient[index:], SENTINEL)
        np.testing.assert_array_equal(actual.hessian[index:], SENTINEL)
    else:
        assert actual.status == -1
        for name in ("value", "gradient", "hessian", "evidence"):
            np.testing.assert_array_equal(getattr(actual, name), SENTINEL)


def test_singular_chart_basis_rejects_instead_of_flooring(bridge):
    spec = Spec.make((0, 1, 1, 5), pivot=2, count=10)
    theta, data = sample(spec)
    theta[1] = 1.
    actual = call(bridge, spec, theta, data)
    assert actual.status == -1
    for name in ("value", "gradient", "hessian", "evidence"):
        np.testing.assert_array_equal(getattr(actual, name), SENTINEL)


def test_overflowing_primal_and_derivative_requests_reject(bridge):
    spec = Spec.make((0, 1, 1, 5), pivot=1, count=6)
    theta, data = sample(spec)
    theta[1], data[1], data[2] = 1e308, 2., 0.
    actual = call(bridge, spec, theta, data)
    assert actual.status == 3
    np.testing.assert_array_equal(actual.value[2:], SENTINEL)
    np.testing.assert_array_equal(actual.gradient[2:], SENTINEL)
    np.testing.assert_array_equal(actual.hessian[2:], SENTINEL)


def test_noheap_state_contract_and_binary_ownership(bridge, bounded_module):
    result = subprocess.run([str(bridge.executable)], capture_output=True, text=True,
                            timeout=max(1, bounded_module - time.monotonic()))
    assert result.returncode == 0, result.stdout + result.stderr
    nm = shutil.which("nm")
    if nm is None:
        pytest.fail("nm is required for binary ownership evidence")
    result = subprocess.run([nm, "-u", str(bridge.library)], capture_output=True,
                            text=True, timeout=5, check=True)
    (bridge.directory / "undefined_symbols.txt").write_text(result.stdout)
    assert not re.search(r"(?:malloc|calloc|realloc|_Znwm|_Znam|vlk_aparch|scivol_)", result.stdout)
    (bridge.directory / "numerical_evidence.json").write_text(json.dumps(bridge.evidence, indent=2) + "\n")
