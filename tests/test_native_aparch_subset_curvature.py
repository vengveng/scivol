"""Normal mean-only curvature and the seed-only power-units recovery.

The MP80 likelihood checks reuse the tracked inverse-polynomial chart Jet oracle,
with quadratic and positive-base fractional-power shocks. Volatility parameters
are frozen. The tiny seed regression uses unscaled MP400/480 full-theta AD.
No installed extension, copied header, fit, or timing benchmark is used.
"""

from __future__ import annotations

import ctypes
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
import tempfile
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from test_native_anchored_mean_chart import Jet, Spec as ChartSpec, chart_oracle


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol/_native"
SOURCE = ROOT / "tests/native_aparch_subset_curvature.cpp"
_SCRATCH = tempfile.TemporaryDirectory(prefix="scivol-test-")  # build images are scratch
EVIDENCE = Path(_SCRATCH.name)
SENTINEL = 271828.125
ORDERS = [(1, 1, 1, 1), (3, 2, 3, 2), (1, 0, 1, 2)]
FIXED = {(1, 1, 1, 1): 1, (1, 0, 1, 2): 2}


class Spec(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in ("ar", "ma", "p", "q", "fixed")]


def source_hashes():
    pending, hashes = [SOURCE, Path(__file__).resolve(),
                       ROOT / "tests/test_native_anchored_mean_chart.py"], {}
    while pending:
        path = pending.pop()
        key = str(path.relative_to(ROOT))
        if key in hashes:
            continue
        content = path.read_bytes()
        hashes[key] = hashlib.sha256(content).hexdigest()
        for name in re.findall(r'^\s*#include\s+"([^\"]+)"', content.decode(), re.M):
            candidate = path.parent / name
            if not candidate.exists():
                candidate = NATIVE / name
            if not candidate.exists():
                candidate = ROOT / "scivol/_csrc" / name
            pending.append(candidate.resolve())
    return hashes


@pytest.fixture(scope="module")
def bridge():
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not shutil.which(compiler[0]):
        pytest.skip("A C++20 compiler is required")
    directory = EVIDENCE / f"run-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True)
    deadline = time.monotonic() + 109
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError("APARCH subset correctness shard exceeded 109 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 109)
    record = dict(checks=[], scope="current owners; Normal jets and all-density layouts; no fit or promotion claim")
    try:
        frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
                 if "clang" in compiler[0] or sys.platform == "darwin"
                 else ["-Werror=frame-larger-than=16384"])
        library = directory / "subset.so"
        command = [*compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
                   "-march=native", "-fno-exceptions", "-fno-rtti", *frame,
                   "-fstack-usage", "-fPIC", "-shared", "-I", str(NATIVE),
                   "-I", str(ROOT / "scivol/_csrc"),
                   "-I", str(NATIVE / "third_party/boost"),
                   "-I", str(NATIVE / "third_party/tlfloat"), str(SOURCE), "-o", str(library)]
        record["build_attempts"] = []
        for _ in range(3):
            hashes = source_hashes()
            built = subprocess.run(command, text=True, capture_output=True,
                                   timeout=max(1, deadline - time.monotonic()))
            unchanged = hashes == source_hashes()
            record["build_attempts"].append(dict(source_hashes=hashes, unchanged=unchanged,
                returncode=built.returncode, compiler_stderr=built.stderr))
            if unchanged:
                assert built.returncode == 0, built.stderr
                break
        else:
            pytest.fail("Owners kept changing during compilation; rerun")
        record.update(command=command, source_hashes=hashes, returncode=built.returncode,
                      compiler_stderr=built.stderr)
        record["image_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        lib = ctypes.CDLL(str(library))
        size, ptr = ctypes.c_size_t, ctypes.c_void_p
        lib.subset_workspace.argtypes = [ctypes.POINTER(Spec), ctypes.c_uint, ctypes.c_bool]
        lib.subset_workspace.restype = size
        lib.subset_width.argtypes = [ctypes.POINTER(Spec), ctypes.c_bool]
        lib.subset_width.restype = size
        lib.subset_evaluate.argtypes = [ctypes.POINTER(Spec), ctypes.c_uint, ctypes.c_bool,
                                       ptr, ptr, size, ctypes.c_double, size, ptr, size,
                                       ptr, ptr, ptr, ptr]
        lib.subset_evaluate.restype = ctypes.c_int
        lib.seed_workspace.argtypes = [ctypes.POINTER(Spec), ctypes.c_uint]
        lib.seed_workspace.restype = size
        lib.seed_evaluate.argtypes = [ctypes.POINTER(Spec), ctypes.c_uint, ptr, ptr, size,
                                     ctypes.c_double, ptr, size, ptr, ptr, ptr]
        lib.seed_evaluate.restype = ctypes.c_int
        lib.seed_primitive.argtypes = [ptr, ctypes.c_double, ptr]
        lib.seed_primitive.restype = ctypes.c_int
        lib.subset_layout_contract.argtypes = [ctypes.c_uint]
        lib.subset_layout_contract.restype = ctypes.c_int
        lib.subset_slope_contract.argtypes = []
        lib.subset_slope_contract.restype = ctypes.c_int
        lib.subset_log_metadata_contract.argtypes = []
        lib.subset_log_metadata_contract.restype = ctypes.c_int
        symbols = subprocess.run(["nm", "-u", str(library)], text=True, capture_output=True,
                                 check=True, timeout=5).stdout
        (directory / "undefined_symbols.txt").write_text(symbols)
        assert not re.search(r"(?:malloc|calloc|realloc|alloca|_Zn[aw]|vlk_aparch|scivol_)", symbols)
        stack_rows = [line.split("\t") for path in directory.glob("*.su")
                      for line in path.read_text().splitlines()]
        assert stack_rows and all(row[-1] == "static" for row in stack_rows), stack_rows
        frames = [int(row[-2]) for row in stack_rows]
        assert frames and max(frames) <= 16384
        record["static_frame_count"] = len(frames)
        record["maximum_static_frame_bytes"] = max(frames)
        yield SimpleNamespace(lib=lib, record=record)
        assert record["source_hashes"] == source_hashes(), "Owners changed during numerical tests"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        (directory / "evidence.json").write_text(json.dumps(record, indent=2) + "\n")


def sample(orders, *, delta=1.6):
    ar, ma, p, q = orders
    theta = np.r_[.05, np.full(ar, .07), np.full(ma, .08), .12,
                  np.full(p, .06 / p), np.linspace(-.2, .3, p), np.full(q, .35 / q), delta]
    data = np.array([.8, -.4, 1.1, .3, -1.2, .7, .6, -.2, 1.4, -.9, .5, .2, -.6, .9])
    return theta, data


@pytest.mark.parametrize("density", range(8))
def test_every_fixed_layout_field_matches_runtime_builder(bridge, density):
    assert bridge.lib.subset_layout_contract(density) == 14
    bridge.record["checks"].append(dict(test="all-field-layout", density=density,
        layouts=14, fields=18, passed=True))


def test_fixed_slope_width_ignores_runtime_metadata(bridge):
    assert bridge.lib.subset_slope_contract() == 3
    bridge.record["checks"].append(dict(test="fixed-slope-width", widths=[3,8,11],
        poisoned_runtime_width=True, exact=True, passed=True))


def test_fixed_log_variance_metadata_survives_both_range_branches(bridge):
    assert bridge.lib.subset_log_metadata_contract() == 2
    bridge.record["checks"].append(dict(test="fixed-log-metadata", poisoned_width_and_delta=True,
        reciprocal_overflow=True, exact=True, passed=True))


def call(bridge, orders, theta, data, mask, *, reduced=True, fixed=False,
         h0=1.1, pivot=7, short=False, null_unrequested=False, theta_space=False):
    assert not theta_space or not reduced
    spec = Spec(*orders, FIXED[orders] if fixed else 0)
    theta, data = (np.ascontiguousarray(x, dtype=np.float64) for x in (theta, data))
    assert theta.ndim == data.ndim == 1
    expected_parameters = 1 + sum(orders[:2]) + 2 + 2 * orders[2] + orders[3]
    assert theta.size == expected_parameters
    required = (bridge.lib.seed_workspace(ctypes.byref(spec), mask) if theta_space else
                bridge.lib.subset_workspace(ctypes.byref(spec), mask, reduced))
    assert required > 0
    arena = np.full(required + 64, 0xA5, dtype=np.uint8)
    offset = 16 + (-arena.ctypes.data % 16)
    width = 1 + sum(orders[:2]) if reduced else theta.size
    assert bridge.lib.subset_width(ctypes.byref(spec), reduced) == width
    guarded = [np.full(count + 2, SENTINEL) for count in (1, width, width * width, 6)]
    outputs = [x[1:-1] for x in guarded]
    # Finite canaries alone let a successful no-op or partial write pass parity.
    for bit, output in zip((1, 2, 4), outputs[:3]):
        if mask & bit:
            output.fill(np.nan)
    if mask & 6 and not theta_space:
        outputs[3].fill(np.nan)
    original_outputs = [output.copy() for output in outputs]
    before = theta.copy(), data.copy()
    bits = (1, 2, 4, 0 if theta_space else 6)
    pointers = [None if null_unrequested and not mask & bit else output.ctypes.data
                for bit, output in zip(bits, outputs)]
    if theta_space:
        status = bridge.lib.seed_evaluate(ctypes.byref(spec), mask,
            theta.ctypes.data, data.ctypes.data, data.size, h0,
            arena.ctypes.data + offset, required - int(short), *pointers[:3])
    else:
        status = bridge.lib.subset_evaluate(ctypes.byref(spec), mask, reduced,
            theta.ctypes.data, data.ctypes.data, data.size, h0, pivot,
            arena.ctypes.data + offset, required - int(short), *pointers)
    np.testing.assert_array_equal(theta, before[0])
    np.testing.assert_array_equal(data, before[1])
    np.testing.assert_array_equal(arena[:offset], 0xA5)
    np.testing.assert_array_equal(arena[offset + required:], 0xA5)
    for output in guarded:
        np.testing.assert_array_equal(output[[0, -1]], SENTINEL)
    for bit, output, original in zip(bits, outputs, original_outputs):
        if not mask & bit or short:
            np.testing.assert_array_equal(output, original)
        elif status == 0:
            assert np.isfinite(output).all()
    if short:
        assert status == -1
        np.testing.assert_array_equal(arena, 0xA5)
    if status != 0:
        np.testing.assert_array_equal(outputs[3], original_outputs[3])
    bridge.record["checks"].append(dict(orders=orders, mask=mask, reduced=reduced,
        fixed=int(spec.fixed), width=width, h0=h0, pivot=pivot, short=short,
        null_unrequested=null_unrequested, status=status, guarded=True,
        requested_outputs_written=status == 0, theta_space=theta_space))
    return status, (outputs[0], outputs[1], outputs[2].reshape(width, width)), outputs[3]


def assert_projection(actual, reference, mask, width):
    for bit, output, full in zip((1, 2, 4), actual, reference):
        if not mask & bit:
            continue
        target = full if bit == 1 else full[:width] if bit == 2 else full[:width, :width]
        assert output.shape == target.shape
        assert np.isfinite(output).all() and np.isfinite(target).all()
        if bit == 1:
            np.testing.assert_allclose(output, target, rtol=0, atol=2e-11)
        else:
            np.testing.assert_allclose(output, target, rtol=3e-10, atol=2e-11)


def test_guards_reject_successful_noop_and_partial_writes():
    orders = (3, 2, 3, 2)
    theta, data = sample(orders)
    for reduced, theta_space in ((False, False), (True, False), (False, True)):
        width = 1 + sum(orders[:2]) if reduced else theta.size
        for partial in (False, True):
            def evaluate(*args):
                if partial:
                    mask = args[1]
                    bits = (1, 2, 4) if theta_space else (1, 2, 4, 6)
                    counts = (1, width, width * width, 6)
                    for bit, pointer, count in zip(bits, args[-len(bits):], counts):
                        if mask & bit:
                            output = np.ctypeslib.as_array((ctypes.c_double * count).from_address(pointer))
                            output[:-1] = 0
                return 0

            fake = SimpleNamespace(lib=SimpleNamespace(subset_workspace=lambda *_: 64,
                subset_width=lambda *_: width, subset_evaluate=evaluate,
                seed_workspace=lambda *_: 64, seed_evaluate=evaluate), record={"checks": []})
            for mask in range(1, 8):
                with pytest.raises(AssertionError):
                    call(fake, orders, theta, data, mask, reduced=reduced, theta_space=theta_space)
            assert not fake.record["checks"]


def test_workspaces_follow_request_order_and_derivative_width(bridge):
    for orders in ORDERS:
        for fixed in ([False, True] if orders in FIXED else [False]):
            spec = Spec(*orders, FIXED[orders] if fixed else 0)
            sizes = {}
            for reduced in (False, True):
                sizes[reduced] = [bridge.lib.subset_workspace(ctypes.byref(spec), mask, reduced)
                                  for mask in range(1, 8)]
                f, g, fg, h, fh, gh, fgh = sizes[reduced]
                assert 0 < f < g < h < gh
                assert g == fg and h == fh and gh == fgh
            assert sizes[True][0] == sizes[False][0]
            assert all(a < b for a, b in zip(sizes[True][1:], sizes[False][1:]))


def test_invalid_fixed_dimensions_and_masks_reject_before_writing(bridge):
    cases = [(Spec(1, 1, 1, 1, 2), 7), (Spec(1, 0, 1, 2, 1), 7),
             (Spec(1, 1, 1, 1, 3), 7), (Spec(1, 1, 1, 1, 0), 0),
             (Spec(1, 1, 1, 1, 0), 8)]
    theta, data = sample(ORDERS[0])
    for spec, mask in cases:
        for reduced in (False, True):
            assert bridge.lib.subset_workspace(ctypes.byref(spec), mask, reduced) == 0
            assert bridge.lib.subset_width(ctypes.byref(spec), reduced) == (
                (3 if reduced else 8) if spec.fixed == 0 else 0)
            memory = np.full(64, 0xA5, dtype=np.uint8)
            outputs = [np.full(count, SENTINEL) for count in (3, 10, 66, 8)]
            status = bridge.lib.subset_evaluate(ctypes.byref(spec), mask, reduced,
                theta.ctypes.data, data.ctypes.data, data.size, 1.1, 7,
                memory.ctypes.data, memory.size, *(x.ctypes.data for x in outputs))
            assert status == -1
            np.testing.assert_array_equal(memory, 0xA5)
            for output in outputs:
                np.testing.assert_array_equal(output, SENTINEL)
            bridge.record.setdefault("rejection_checks", []).append(dict(
                fixed=int(spec.fixed), mask=mask, reduced=reduced, status=status, guarded=True))


@pytest.mark.parametrize("orders", ORDERS)
@pytest.mark.parametrize("h0", [1.1, 1e-220, 1e220])
def test_sparse_mean_requests_match_full_chart(bridge, orders, h0):
    theta, data = sample(orders, delta=1.6 if h0 == 1.1 else 2.4)
    width = 1 + sum(orders[:2])
    status, full, _ = call(bridge, orders, theta, data, 7, reduced=False, h0=h0)
    assert status == 0
    for fixed in ([False, True] if orders in FIXED else [False]):
        status, whole, _ = call(bridge, orders, theta, data, 7, fixed=fixed, h0=h0)
        assert status == 0
        for mask in range(1, 8):
            status, actual, _ = call(bridge, orders, theta, data, mask, fixed=fixed, h0=h0)
            assert status == 0
            assert_projection(actual, whole, mask, width)
            assert_projection(actual, full, mask, width)
            status, full_sparse, _ = call(bridge, orders, theta, data, mask,
                                          reduced=False, fixed=fixed, h0=h0)
            assert status == 0
            assert_projection(actual, full_sparse, mask, width)
            call(bridge, orders, theta, data, mask, fixed=fixed, h0=h0, short=True)
            call(bridge, orders, theta, data, mask, reduced=False, fixed=fixed, h0=h0, short=True)
            for reduced in (False, True):
                status, unused_null, _ = call(bridge, orders, theta, data, mask,
                    reduced=reduced, fixed=fixed, h0=h0, null_unrequested=True)
                assert status == 0
                assert_projection(unused_null, full if not reduced else whole, mask,
                                  theta.size if not reduced else width)


def log_jet(x):
    width = len(x.gradient)
    return Jet(mp.log(x.value), tuple(g / x.value for g in x.gradient),
               tuple(x.hessian[i * width + j] / x.value
                     - x.gradient[i] * x.gradient[j] / x.value**2
                     for i in range(width) for j in range(width)))


def power_jet(x, exponent):
    """Scalar power chain rule on a strictly positive MP Jet."""
    assert x.value > 0
    exponent = mp.mpf(exponent)
    value = x.value**exponent
    first = exponent * value / x.value
    second = exponent * (exponent - 1) * value / x.value**2
    width = len(x.gradient)
    return Jet(value, tuple(first * g for g in x.gradient),
               tuple(first * x.hessian[i * width + j]
                     + second * x.gradient[i] * x.gradient[j]
                     for i in range(width) for j in range(width)))


def test_polynomial_ad_logarithm_matches_closed_form():
    with mp.workdps(80):
        x, y = Jet.variable(mp.mpf(".25"), 2, 0), Jet.variable(mp.mpf(".5"), 2, 1)
        actual = log_jet(x * x * y)
        assert abs(actual.value - (2 * mp.log(x.value) + mp.log(y.value))) < mp.mpf("1e-78")
        assert actual.gradient == (mp.mpf(8), mp.mpf(2))
        assert actual.hessian == (mp.mpf(-32), mp.mpf(0), mp.mpf(0), mp.mpf(-4))


def test_fractional_power_ad_matches_closed_form():
    with mp.workdps(80):
        x, y = Jet.variable(4, 2, 0), Jet.variable(9, 2, 1)
        actual = power_jet(x * y, mp.mpf("1.5"))
        assert actual.value == 216
        assert actual.gradient == (81, 36)
        assert actual.hessian == (mp.mpf("10.125"), mp.mpf("13.5"), mp.mpf("13.5"), 2)


def normal_chart_oracle(orders, theta, data, h0, pivot, *, quadratic=False):
    """MP mean-coordinate AD with frozen volatility parameters and explicit seeds."""
    ar, ma, p, q = orders
    start, width = max(orders), 1 + ar + ma
    with mp.workdps(80):
        delta = mp.mpf(float(theta[-1]))
        if quadratic:
            assert delta == 2 and np.all(theta[width + 1 + p:width + 1 + 2 * p] == 0)
        spec = ChartSpec.make((ar, ma, start, theta.size - width), pivot=pivot, count=data.size)
        residuals, anchor, _ = chart_oracle(spec, theta, data)
        residuals = [Jet.constant(0, width)] * start + residuals
        powers = [Jet.constant(mp.mpf(float(h0))**(delta / 2), width)] * start
        total = Jet.constant(0, width)
        for t in range(start, data.size):
            power = Jet.constant(float(theta[width]), width)
            for lag in range(p):
                time_index = t - lag - 1
                if time_index < start:
                    continue
                residual = residuals[time_index]
                if quadratic:
                    shock = residual * residual
                else:
                    assert residual.value != 0, "Fractional AD fixture must avoid residual kinks"
                    magnitude = residual if residual.value > 0 else -residual
                    base = magnitude - residual * float(theta[width + 1 + p + lag])
                    shock = power_jet(base, delta)
                power += shock * float(theta[width + 1 + lag])
            for lag in range(q):
                power += powers[t - lag - 1] * float(theta[width + 1 + 2 * p + lag])
            powers.append(power)
            variance = power if quadratic else power_jet(power, 2 / delta)
            total += (log_jet(variance) + mp.log(2 * mp.pi)
                      + residuals[t] * residuals[t] / variance) * mp.mpf(".5")
        return (np.array([float(total.value)]), np.array(list(map(float, total.gradient))),
                np.array(list(map(float, total.hessian))).reshape(width, width)), anchor


def assert_anchor_evidence(actual, expected):
    # Bridge order is error, basis, residual; chart_oracle returns error, residual, basis.
    with mp.workdps(80):
        for index, reference in enumerate((expected[0], expected[2], expected[1])):
            observed = mp.mpf(float(actual[2 * index])) + mp.mpf(float(actual[2 * index + 1]))
            assert abs(observed - reference) <= mp.mpf("3e-29") * max(1, abs(reference))


@pytest.mark.parametrize("orders", ORDERS)
def test_independent_mp80_polynomial_fixture(bridge, orders):
    theta, data = sample(orders, delta=2)
    width = 1 + sum(orders[:2])
    theta[width + 1 + orders[2]:width + 1 + 2 * orders[2]] = 0
    expected, anchor = normal_chart_oracle(orders, theta, data, 1.1, 7, quadratic=True)
    for fixed in ([False, True] if orders in FIXED else [False]):
        for mask in range(1, 8):
            status, actual, evidence = call(bridge, orders, theta, data, mask, fixed=fixed)
            assert status == 0
            assert_projection(actual, expected, mask, width)
            if mask & 6:
                assert_anchor_evidence(evidence, anchor)
    bridge.record.setdefault("independent_derivative_evidence", []).append(dict(
        fixture="quadratic shock, delta=2, gamma=0", digits=80, orders=orders,
        masks=list(range(1, 8)), value_atol=2e-11, derivative_atol=2e-11, derivative_rtol=3e-10))


@pytest.mark.parametrize("orders", ORDERS)
@pytest.mark.parametrize("pivot_position", ["startup", "last"])
def test_independent_mp80_fractional_normal_fixture(bridge, orders, pivot_position):
    theta, data = sample(orders, delta=1.6)
    pivot = max(orders) if pivot_position == "startup" else data.size - 1
    h0 = .03125
    width = 1 + sum(orders[:2])
    expected, anchor = normal_chart_oracle(orders, theta, data, h0, pivot)
    for fixed in ([False, True] if orders in FIXED else [False]):
        for reduced in (False, True):
            for mask in range(1, 8):
                status, actual, evidence = call(bridge, orders, theta, data, mask,
                    reduced=reduced, fixed=fixed, h0=h0, pivot=pivot)
                assert status == 0
                assert_projection(actual if reduced else
                    (actual[0], actual[1][:width], actual[2][:width, :width]), expected, mask, width)
                if mask & 6:
                    assert_anchor_evidence(evidence, anchor)
    bridge.record.setdefault("independent_derivative_evidence", []).append(dict(
        fixture="Normal fractional power, frozen volatility", digits=80, orders=orders,
        pivot=int(pivot), h0=h0, delta=1.6, masks=list(range(1, 8)),
        value_atol=2e-11, derivative_atol=2e-11, derivative_rtol=3e-10))


@pytest.mark.parametrize("delta", [1.6, 2.0])
def test_tiny_actual_pivot_residual_is_not_snapped(bridge, delta):
    orders = (1, 1, 1, 1)
    theta, data = sample(orders, delta=delta)
    theta[:3] = [.125, .25, -.125]
    data[0] = .5
    data[1] = np.nextafter(.25, np.inf)
    expected_residual = mp.mpf(float(data[1])) - mp.mpf(".25")
    if delta == 2:
        theta[5] = 0
        expected, anchor = normal_chart_oracle(orders, theta, data, 1.1, 1, quadratic=True)
        assert anchor[1] == expected_residual
    status, full, _ = call(bridge, orders, theta, data, 7, reduced=False, pivot=1)
    assert status == 0
    for fixed in (False, True):
        for mask in range(1, 8):
            status, actual, evidence = call(bridge, orders, theta, data, mask, fixed=fixed, pivot=1)
            assert status == 0
            if mask != 1:
                observed = mp.mpf(float(evidence[4])) + mp.mpf(float(evidence[5]))
                assert observed == expected_residual != 0
            assert_projection(actual, full, mask, 3)
            if delta == 2:
                assert_projection(actual, expected, mask, 3)
    bridge.record.setdefault("tiny_residual_evidence", []).append(dict(
        delta=delta, actual_pivot_residual=str(expected_residual), snapped=False,
        independent_ad=delta == 2, full_chart_parity=True, masks=list(range(1, 8))))


@pytest.mark.parametrize("delta,alpha,reduced_g,reduced_h,full_h", [
    (.6, 0., True, True, False), (1., 0., True, True, False),
    (1.6, 0., True, True, True), (.6, .06, False, False, False),
    (1.6, .06, True, False, False),
])
def test_unavailable_derivatives_are_not_zero_curvature(bridge, delta, alpha, reduced_g, reduced_h, full_h):
    orders = (1, 1, 1, 1)
    _, data = sample(orders)
    data[7] = 0
    theta = np.array([0., 0., 0., .12, alpha, -.2, .35, delta])
    for fixed in (False, True):
        for reduced in (False, True):
            for mask in range(1, 8):
                valid = not mask & 4 or (reduced_h if reduced else full_h)
                valid &= not mask & 2 or reduced_g
                status, outputs, _ = call(bridge, orders, theta, data, mask, reduced=reduced, fixed=fixed)
                assert status == (0 if valid else -3)
                if not valid:
                    for bit, output in zip((1, 2, 4), outputs):
                        if mask & bit:
                            assert np.isnan(output).all()
                elif reduced and mask & 4:
                    assert np.any(outputs[2] != 0)


def test_finite_mean_gradient_does_not_certify_overflowing_full_derivatives(bridge):
    orders = (1, 1, 1, 1)
    _, data = sample(orders)
    data *= 1e-160
    theta = np.array([0., 0., 0., 1e-310, 0., -.2, .35, 2.])
    for fixed in (False, True):
        for reduced in (False, True):
            for mask in range(1, 8):
                status, _, _ = call(bridge, orders, theta, data, mask, reduced=reduced,
                                    fixed=fixed, h0=1e-310)
                valid = mask == 1 or (reduced and not mask & 4)
                assert status == (0 if valid else -3)


SEED_ORDERS = (1, 0, 1, 2)


def seed_sample(*, high_delta=False):
    theta = np.array([.04, .12, .08, .08, .1, .3, .3, 2.4])
    data = np.array([.8, -.4, 1.1, .3, -1.2, .7, .6, -.2, 1.4, -.9, .5, .2, -.6, .9])
    if high_delta:
        theta[2:] = [1., 0., -.15, .05, .05, 2200.]
    return theta, data


def exp_jet(x):
    value, width = mp.exp(x.value), len(x.gradient)
    return Jet(value, tuple(value * g for g in x.gradient),
               tuple(value * (x.hessian[i * width + j] + x.gradient[i] * x.gradient[j])
                     for i in range(width) for j in range(width)))


def loggamma_jet(x):
    width = len(x.gradient)
    first, second = mp.digamma(x.value), mp.polygamma(1, x.value)
    return Jet(mp.loggamma(x.value), tuple(first * g for g in x.gradient),
               tuple(first * x.hessian[i * width + j] + second * x.gradient[i] * x.gradient[j]
                     for i in range(width) for j in range(width)))


def unscaled_theta_oracle(orders, theta, data, h0, *, digits, density="normal", remove_seed=False):
    """Full-theta AD of unscaled equations, with zero residual and h0 histories."""
    ar, ma, p, q = orders
    start, offset, width = max(orders), 1 + ar + ma, theta.size
    with mp.workdps(digits):
        point = [Jet.variable(float(value), width, index) for index, value in enumerate(theta)]
        delta = point[offset + 1 + 2*p + q]
        seed = (Jet.constant(0, width) if remove_seed else
                exp_jet(delta * (mp.log(mp.mpf(float(h0))) / 2)))
        residuals, powers = [Jet.constant(0, width)] * data.size, [seed] * data.size
        total = Jet.constant(0, width)
        assert density in ("normal", "studentt", "skewt")
        if density in ("studentt", "skewt"):
            nu = point[-2] if density == "skewt" else point[-1]
            constant = (loggamma_jet(nu / 2) - loggamma_jet((nu + 1) / 2)
                        + log_jet((nu - 2) * mp.pi) / 2)
            if density == "skewt":
                skew = point[-1]
                shift = 4 * skew * exp_jet(-constant) * (nu - 2) / (nu - 1)
                stretch = exp_jet(log_jet(1 + 3 * skew * skew - shift * shift) / 2)
        for t in range(start, data.size):
            residual = mp.mpf(float(data[t])) - point[0]
            for lag in range(ar):
                residual -= point[1 + lag] * mp.mpf(float(data[t - 1 - lag]))
            for lag in range(ma):
                residual -= point[1 + ar + lag] * residuals[t - 1 - lag]
            residuals[t] = residual
            power = point[offset]
            for lag in range(p):
                if t - 1 - lag < start:
                    continue
                error = residuals[t - 1 - lag]
                assert error.value != 0, "Seed fixtures must avoid residual kinks"
                magnitude = error if error.value > 0 else -error
                base = magnitude - point[offset + 1 + p + lag] * error
                power += point[offset + 1 + lag] * exp_jet(delta * log_jet(base))
            for lag in range(q):
                power += point[offset + 1 + 2*p + lag] * powers[t - 1 - lag]
            powers[t] = power
            log_variance = 2 * log_jet(power) / delta
            standardized_square = residual * residual * exp_jet(-log_variance)
            if density == "normal":
                total += (mp.log(2 * mp.pi) + log_variance + standardized_square) * mp.mpf(".5")
            elif density == "studentt":
                total += constant + log_variance / 2 + (nu + 1) * log_jet(
                    1 + standardized_square / (nu - 2)) / 2
            else:
                branch = stretch * residual * exp_jet(-log_variance / 2) + shift
                adjusted = branch / (1 + skew if branch.value >= 0 else 1 - skew)
                total += constant - log_jet(stretch) + log_variance / 2 + (nu + 1) * log_jet(
                    1 + adjusted * adjusted / (nu - 2)) / 2
        return total


def unscaled_normal_theta_oracle(orders, theta, data, h0, *, digits, remove_seed=False):
    return unscaled_theta_oracle(orders, theta, data, h0, digits=digits, remove_seed=remove_seed)


def jet_outputs(jet):
    width = len(jet.gradient)
    return (np.array([float(jet.value)]), np.array(list(map(float, jet.gradient))),
            np.array(list(map(float, jet.hessian))).reshape(width, width))


def refined_seed_oracle(theta, data, h0):
    low, high = [unscaled_normal_theta_oracle(SEED_ORDERS, theta, data, h0, digits=d)
                 for d in (400, 480)]
    with mp.workdps(480):
        refinement = max(abs(a - b) / (1 + abs(b)) for a, b in zip(
            (low.value, *low.gradient, *low.hessian), (high.value, *high.gradient, *high.hessian)))
        assert refinement < mp.mpf("1e-370")
    return high, str(refinement)


def assert_seed_projection(actual, expected, mask):
    for bit, output, target in zip((1, 2, 4), actual, expected):
        if mask & bit:
            assert output.shape == target.shape
            assert np.isfinite(output).all() and np.isfinite(target).all()
            np.testing.assert_allclose(output, target, rtol=0 if bit == 1 else 3e-10,
                                       atol=2e-11)


@pytest.mark.parametrize("h0", [1e-290, 1e-200, 1e290])
def test_seed_only_units_full_theta_ad_and_sparse_guards(bridge, h0):
    theta, data = seed_sample()
    oracle, refinement = refined_seed_oracle(theta, data, h0)
    expected = jet_outputs(oracle)
    status, runtime, _ = call(bridge, SEED_ORDERS, theta, data, 7,
        reduced=False, theta_space=True, h0=h0)
    assert status == 0
    # These blocks are nonzero, so a mean-only or block-diagonal Hessian cannot pass.
    if h0 == 1e-290:
        for rows, columns in ((slice(0, 2), slice(2, 7)), (slice(0, 2), slice(7, 8)),
                              (slice(2, 7), slice(7, 8))):
            assert np.max(np.abs(expected[2][rows, columns])) > 1e-6
    for fixed in (False, True):
        for mask in range(1, 8):
            status, actual, _ = call(bridge, SEED_ORDERS, theta, data, mask,
                reduced=False, theta_space=True, fixed=fixed, h0=h0)
            assert status == 0
            assert_seed_projection(actual, expected, mask)
            assert_seed_projection(actual, runtime, mask)
            status, null_outputs, _ = call(bridge, SEED_ORDERS, theta, data, mask,
                reduced=False, theta_space=True, fixed=fixed, h0=h0, null_unrequested=True)
            assert status == 0
            assert_seed_projection(null_outputs, expected, mask)
            call(bridge, SEED_ORDERS, theta, data, mask, reduced=False,
                 theta_space=True, fixed=fixed, h0=h0, short=True)
    bridge.record.setdefault("seed_theta_evidence", []).append(dict(
        h0=h0, digits=[400, 480], oracle_refinement=refinement,
        F=float(expected[0][0]), full_theta_width=theta.size, all_cross_blocks=True,
        masks=list(range(1, 8)), routes=["runtime", "fixed-ARMA10-APARCH12"],
        value_atol=2e-11, derivative_atol=2e-11, derivative_rtol=3e-10,
        scope="Normal internal theta equations; no domain, transformed jet, filter, or fit qualification"))


def test_seed_only_units_recovery_reaches_mean_chart_requests(bridge):
    theta, data = seed_sample()
    status, reference, _ = call(bridge, SEED_ORDERS, theta, data, 7,
                              reduced=False, h0=1e-290)
    assert status == 0
    for fixed in (False, True):
        for reduced in (False, True):
            for mask in range(1, 8):
                status, actual, _ = call(bridge, SEED_ORDERS, theta, data, mask,
                    reduced=reduced, fixed=fixed, h0=1e-290, null_unrequested=True)
                assert status == 0
                assert_projection(actual, reference, mask, 2 if reduced else theta.size)


def test_seed_only_units_preserve_high_delta_max_scale(bridge):
    theta, data = seed_sample(high_delta=True)
    h0 = 4.
    oracle, refinement = refined_seed_oracle(theta, data, h0)
    expected = jet_outputs(oracle)
    for fixed in (False, True):
        for mask in range(1, 8):
            status, actual, _ = call(bridge, SEED_ORDERS, theta, data, mask,
                reduced=False, theta_space=True, fixed=fixed, h0=h0, null_unrequested=True)
            assert status == 0
            assert_seed_projection(actual, expected, mask)
            for reduced in (False, True):
                status, _, _ = call(bridge, SEED_ORDERS, theta, data, mask,
                    reduced=reduced, fixed=fixed, h0=h0, null_unrequested=True)
                assert status == 0
    primitive = call_seed_primitive(bridge, theta[[2, 5, 6, 7]], h0)
    assert primitive[44] == .5 * np.log(h0)
    assert primitive[45] == 1.
    bridge.record.setdefault("seed_boundary_evidence", []).append(dict(
        delta=2200., h0=h0, digits=[400, 480], oracle_refinement=refinement,
        log_scale=float(primitive[44]), preserved_max_scale=True,
        scope="Successful request guard, not full-range positive-intercept storage compliance"))


def call_seed_primitive(bridge, theta, h0):
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    before = theta.copy()
    guarded = np.full(54, SENTINEL)
    output = guarded[1:-1]
    output.fill(np.nan)
    status = bridge.lib.seed_primitive(theta.ctypes.data, h0, output.ctypes.data)
    np.testing.assert_array_equal(theta, before)
    np.testing.assert_array_equal(guarded[[0, -1]], SENTINEL)
    assert status == 0 and np.isfinite(output).all()
    return output


@pytest.mark.parametrize("h0", [1e-290, 1e-200])
def test_seed_only_units_keep_positive_seed_and_low_limb(bridge, h0):
    theta = np.array([.08, .3, .3, 2.4])
    output = call_seed_primitive(bridge, theta, h0)
    assert output[43] > 0 and output[45] > 0 and output[46] != 0 and output[47] > 0
    with mp.workdps(480):
        point = [Jet.variable(float(value), 4, index) for index, value in enumerate(theta)]
        seed_log = mp.log(mp.mpf(float(h0))) / 2
        seed = exp_jet(point[3] * seed_log)
        assert seed.value > 0  # The unscaled tiny seed is not representable in binary64.
        first = point[0] + (point[1] + point[2]) * seed
        targets = [Jet.constant(2 * seed_log, 4), 2 * log_jet(first) / point[3]]
        for block, target in zip(output[:42].reshape(2, 21), targets):
            actual = block[:1], block[1:5], block[5:].reshape(4, 4)
            assert_seed_projection(actual, jet_outputs(target), 7)
        scale = mp.mpf(float(output[44]))
        scaled_seed = exp_jet(point[3] * (seed_log - scale))
        intercept = point[0] * exp_jet(-point[3] * scale)
        observed = output[[45, 46, 47, 42, 48, 49, 50, 51]]
        expected = np.array(list(map(float, (scaled_seed.value, scaled_seed.gradient[3],
            scaled_seed.hessian[15], intercept.value, intercept.gradient[0],
            intercept.gradient[3], intercept.hessian[3], intercept.hessian[15]))))
        assert np.all(expected != 0)
        np.testing.assert_allclose(observed, expected, rtol=3e-12, atol=0)
        retained = float((mp.mpf(float(theta[1])) + mp.mpf(float(theta[2]))) *
                         mp.mpf(float(output[45])))
        np.testing.assert_allclose(output[43], retained, rtol=3e-12, atol=0)
        old_scale = max(.5 * np.log(h0), np.log(theta[0]) / theta[3])
        if h0 == 1e-290:
            assert np.exp(theta[3] * (.5 * np.log(h0) - old_scale)) == 0
            assert output[44] == .5 * (.5 * np.log(h0)) + .5 * (np.log(theta[0]) / theta[3])
        else:
            assert output[44] == old_scale
    bridge.record.setdefault("seed_primitive_evidence", []).append(dict(
        h0=h0, digits=480, positive_seed=float(output[45]), retained_low=float(output[43]),
        seed_delta_gradient=float(output[46]), seed_delta_hessian=float(output[47]),
        seed_log_delta_gradient=float(output[4]), seed_log_delta_hessian=float(output[20]),
        log_scale=float(output[44]), guards=True, zero_seed_substitution=False))


def test_seed_oracle_retains_contribution_below_double_rounding(bridge):
    theta, data = seed_sample()
    retained = unscaled_normal_theta_oracle(SEED_ORDERS, theta, data, 1e-290, digits=480)
    removed = unscaled_normal_theta_oracle(SEED_ORDERS, theta, data, 1e-290,
                                         digits=480, remove_seed=True)
    with mp.workdps(480):
        differences = [max(abs(a - b) for a, b in zip(left, right)) for left, right in (
            ((retained.value,), (removed.value,)),
            (retained.gradient, removed.gradient), (retained.hessian, removed.hessian))]
        assert all(mp.mpf("1e-350") < value < mp.mpf("1e-340") for value in differences)
    for actual, target in zip(jet_outputs(retained), jet_outputs(removed)):
        np.testing.assert_array_equal(actual, target)
    bridge.record["seed_influence"] = dict(digits=480, nonzero_F_G_H=list(map(str, differences)),
        identical_rounded_doubles=True,
        scope="Rounded likelihood comparisons cannot replace the positive seed/low-limb regression")
