"""RM-02: weighted APARCH shocks, complete MP AD, and exact GARCH reductions."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import warnings

import mpmath as mp
import numpy as np
import pytest

from test_native_anchored_mean_chart import Jet, Spec as ChartSpec, filtered_rows


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol/_native"
SOURCE = ROOT / "tests/native_jet_math_aparch_range.cpp"
PRECISION_SOURCE = NATIVE / "aparch_moment_precision.cpp"
GH_SOURCE = NATIVE / "gh_math_translation.cpp"
EVIDENCE = ROOT / "localdev/jet_math_recovery_20261013/aparch_range"
SENTINEL = 271828.125
F_ATOL = 2e-11
G_RTOL, G_ATOL = 1e-10, 1e-10
H_RTOL, H_ATOL = 1e-10, 1e-10


class Spec(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in ("ar", "ma", "p", "q", "columns")]
    _fields_ += [(name, ctypes.c_uint) for name in ("mode", "fixed", "density")]

    @property
    def mean(self):
        return self.columns if self.mode == 2 else 1 + self.ar + self.ma if self.mode == 1 else 0

    @property
    def first(self):
        return max(self.ar, self.ma, self.p, self.q) if self.mode == 1 else 0


def source_identities(snapshot=None):
    pending = [SOURCE, PRECISION_SOURCE, GH_SOURCE, Path(__file__).resolve(), ROOT / "tests/test_native_anchored_mean_chart.py"]
    hashes = {}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(ROOT))
        if name in hashes:
            continue
        content = path.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        if snapshot is not None:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        for include in re.findall(r'^\s*#include\s+"([^\"]+)"', content.decode(), re.M):
            candidates = [path.parent / include, NATIVE / include, ROOT / "scivol/_csrc" / include]
            pending.append(next(candidate.resolve() for candidate in candidates if candidate.exists()))
    return hashes


def matching_image(hashes):
    compiled = {name: digest for name, digest in hashes.items() if not name.endswith(".py")}
    for path in sorted(EVIDENCE.glob("run-*/evidence.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        record = json.loads(path.read_text())
        previous = {name: digest for name, digest in record.get("source_hashes", {}).items()
                    if not name.endswith(".py")}
        image = Path(record.get("binary_path", "/nonexistent"))
        certified = record.get("immutable_snapshot") or record.get("owners_unchanged_after_build")
        if previous == compiled and certified and record.get("returncode") == 0 and image.exists():
            if hashlib.sha256(image.read_bytes()).hexdigest() == record["binary_sha256"]:
                return image, record
    return None


@pytest.fixture(scope="module")
def bridge():
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not shutil.which(compiler[0]):
        pytest.skip("A C++20 compiler is required")
    pressure = subprocess.run([str(Path.home() / "bin/mempressure")], capture_output=True,
                             text=True, check=True, timeout=5)
    memory = json.loads(pressure.stdout)
    if memory["state"] == "critical":
        pytest.skip("Standalone APARCH compilation deferred under critical memory pressure")
    if memory["state"] == "warning":
        warnings.warn("Memory pressure is yellow; compiling one small standalone shard serially")
    directory = EVIDENCE / f"run-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True)
    library = directory / "aparch_range.so"
    snapshot = directory / "source_snapshot"
    hashes = source_identities(snapshot)
    command = [*compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
               "-march=native", "-fno-exceptions", "-fno-rtti", "-fstack-usage",
               "-Wframe-larger-than=16384", "-Werror=frame-larger-than",
               "-fPIC", "-shared", "-I", str(snapshot / "scivol/_native"),
               "-I", str(snapshot / "scivol/_csrc"),
               "-I", str(NATIVE / "third_party/boost"),
               "-I", str(NATIVE / "third_party/tlfloat"),
               *(str(snapshot / path.relative_to(ROOT)) for path in (SOURCE, PRECISION_SOURCE, GH_SOURCE)),
               "-o", str(library)]
    record = dict(command=command, memory=memory, source_hashes=hashes, checks=[], immutable_snapshot=True)
    started = time.monotonic()
    try:
        cached = matching_image(record["source_hashes"])
        if cached:
            library, previous = cached
            record.update(command=previous["command"], returncode=0, compiler_stderr="",
                          compile_seconds=0., reused_image=str(library))
        else:
            built = subprocess.run(command, capture_output=True, text=True, timeout=110)
            record.update(returncode=built.returncode, compiler_stderr=built.stderr,
                          compile_seconds=time.monotonic() - started)
            assert built.returncode == 0, built.stderr
        record["binary_path"] = str(library)
        record["binary_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        record["owners_unchanged_after_build"] = source_identities() == record["source_hashes"]
        for name in ("scivol/_native/aparch_equation_program.hpp", "scivol/_native/aparch_runtime_jet.hpp"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == hashes[name]
        symbols = subprocess.run(["nm", "-u", str(library)], capture_output=True, text=True,
                                 check=True, timeout=5).stdout
        (directory / "undefined_symbols.txt").write_text(symbols)
        assert not re.search(r"(?:malloc|calloc|realloc|alloca|_Zn[aw]|vlk_aparch|_native_core)", symbols)
        frames = [line.split("\t") for path in library.parent.glob("*.su")
                  for line in path.read_text().splitlines()]
        assert frames and all(row[-1] == "static" and int(row[-2]) <= 16384 for row in frames)
        record["maximum_static_frame_bytes"] = max(int(row[-2]) for row in frames)
        lib = ctypes.CDLL(str(library))
        record["installed_extension_loaded"] = any(name in sys.modules for name in
            ("scivol._core", "scivol._native_core", "_core", "_native_core"))
        assert not record["installed_extension_loaded"]
        class Owner(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Owner)]
        dladdr.restype = ctypes.c_int
        record["symbol_owners"] = {}
        for name in ("aparch_range_evaluate", "aparch_range_filter", "aparch_range_chart_evaluate",
                     "gh_shape_value_cache_init", "gh_bessel_value_plan_init"):
            owner = Owner()
            address = ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
            assert dladdr(address, ctypes.byref(owner))
            record["symbol_owners"][name] = owner.filename.decode()
            assert Path(owner.filename.decode()).resolve() == library.resolve()
        size, ptr, spec = ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(Spec)
        lib.aparch_range_workspace.argtypes = [spec, ctypes.c_uint]
        lib.aparch_range_workspace.restype = size
        lib.aparch_range_evaluate.argtypes = [spec, ctypes.c_uint, ptr, ptr, ptr, size,
                                             ctypes.c_double, ptr, size, ptr, ptr, ptr, ptr]
        lib.aparch_range_evaluate.restype = ctypes.c_int
        lib.aparch_range_filter.argtypes = [spec, ptr, ptr, size, ctypes.c_double,
                                           ptr, size, ptr, ptr]
        lib.aparch_range_filter.restype = ctypes.c_int
        lib.aparch_range_chart_workspace.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_uint]
        lib.aparch_range_chart_workspace.restype = size
        lib.aparch_range_chart_evaluate.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_uint,
            ptr, ptr, size, ctypes.c_double, ptr, size, ptr, ptr, ptr]
        lib.aparch_range_chart_evaluate.restype = ctypes.c_int
        yield SimpleNamespace(lib=lib, record=record)
        record["owners_unchanged_after_tests"] = source_identities() == record["source_hashes"]
        live = source_identities()
        record["source_changes_since_snapshot"] = [name for name, digest in hashes.items()
                                                   if live.get(name) != digest]
        assert not any(name.startswith("scivol/_native/aparch_")
                       and name in record["source_changes_since_snapshot"]
                       for name in ("scivol/_native/aparch_equation_program.hpp",
                                    "scivol/_native/aparch_runtime_jet.hpp"))
    finally:
        record["total_seconds"] = time.monotonic() - started
        (directory / "evidence.json").write_text(json.dumps(record, indent=2) + "\n")


def unary(point, value, first, second):
    width = len(point.gradient)
    return Jet(value, tuple(first * entry for entry in point.gradient),
               tuple(first * point.hessian[i * width + j]
                     + second * point.gradient[i] * point.gradient[j]
                     for i in range(width) for j in range(width)))


def ad_log(point):
    return unary(point, mp.log(point.value), 1 / point.value, -1 / point.value**2)


def ad_exp(point):
    value = mp.exp(point.value)
    return unary(point, value, value, value)


def ad_lgamma(point):
    return unary(point, mp.loggamma(point.value), mp.digamma(point.value), mp.polygamma(1, point.value))


def assemble_residual(spec, parameters, data, design, residuals, time_index, zero):
    if not spec.mode:
        return zero + float(data[time_index])
    if spec.mode == 2:
        return float(data[time_index]) - sum(
            (parameters[column] * float(design[time_index, column]) for column in range(spec.mean)), zero)
    residual = float(data[time_index]) - parameters[0]
    for lag in range(spec.ar):
        residual -= parameters[1 + lag] * float(data[time_index - 1 - lag])
    for lag in range(spec.ma):
        residual -= parameters[1 + spec.ar + lag] * residuals[time_index - 1 - lag]
    return residual


def complete_oracle(spec, theta, data, design, h0, digits, *, garch=False, chart=False):
    with mp.workdps(digits):
        parameters = [Jet.variable(float(value), theta.size, index) for index, value in enumerate(theta)]
        zero = Jet.constant(0, theta.size)
        if chart:
            chart_spec = ChartSpec.make((spec.ar, spec.ma, spec.first, theta.size - spec.mean),
                                       pivot=spec.first, count=data.size)
            origin = [mp.mpf(float(value)) for value in theta[:spec.mean]]
            a0, b0 = filtered_rows(chart_spec, origin, data)[0]
            error = -(a0 - b0 * origin[0]) / b0
            parameters[0] = Jet.variable(0, theta.size, 0)
            a, b = filtered_rows(chart_spec, parameters[:spec.mean], data)[0]
            parameters[0] = (a - parameters[0]) / b + error
        omega, alpha = spec.mean, spec.mean + 1
        gamma, beta = alpha + spec.p, alpha + 2 * spec.p
        delta_index = beta + spec.q
        delta = zero + 2 if garch else parameters[delta_index]
        seed = zero + mp.mpf(float(h0)) if garch else ad_exp(delta * (mp.log(mp.mpf(float(h0))) / 2))
        residuals, powers, variances = [zero] * data.size, [seed] * data.size, [mp.mpf(float(h0))] * data.size
        objective = zero
        for time_index in range(spec.first, data.size):
            residual = residuals[time_index] = assemble_residual(
                spec, parameters, data, design, residuals, time_index, zero)
            power = seed
            if time_index or spec.mode == 1:
                power = parameters[omega]
                for lag in range(spec.p):
                    previous_index = time_index - 1 - lag
                    if previous_index < 0 or previous_index < spec.first:
                        continue
                    previous = residuals[previous_index]
                    if garch:
                        shock = previous * previous
                    elif previous.value:
                        base = previous * (mp.sign(previous.value) - parameters[gamma + lag])
                        shock = ad_exp(delta * ad_log(base))
                    else:
                        shock = zero
                    power += parameters[alpha + lag] * shock
                for lag in range(spec.q):
                    previous_index = time_index - 1 - lag
                    if spec.mode != 1 and previous_index < 0:
                        continue
                    previous = powers[previous_index] if previous_index >= spec.first else seed
                    power += parameters[beta + lag] * previous
            powers[time_index] = power
            log_variance = ad_log(power) if garch else 2 * ad_log(power) / delta
            variance = ad_exp(log_variance)
            variances[time_index] = variance.value
            standardized2 = residual * residual / variance
            if spec.density == 0:
                objective += (mp.log(2 * mp.pi) + log_variance + standardized2) * mp.mpf(".5")
            else:
                nu = parameters[-1]
                normalizer = ad_lgamma((nu + 1) * mp.mpf(".5")) - ad_lgamma(nu * mp.mpf(".5"))
                normalizer -= ad_log((nu - 2) * mp.pi) * mp.mpf(".5")
                objective += log_variance * mp.mpf(".5") - normalizer
                objective += (nu + 1) * ad_log(1 + standardized2 / (nu - 2)) * mp.mpf(".5")
        outputs = (np.array([float(objective.value)]), np.array(objective.gradient, dtype=float),
                   np.array(objective.hessian, dtype=float).reshape(theta.size, theta.size))
        return outputs, np.array(variances, dtype=float)


def sample(mode, wide, density, *, alpha=None):
    ar, ma, p, q = (2, 1, 3, 2) if wide else (1, 1, 1, 1)
    if mode == 0:
        ar = ma = 0
    columns = 2 if mode == 2 else 0
    spec = Spec(ar, ma, p, q, columns, mode, 0, density)
    mean = np.zeros(spec.mean)
    if mode == 2:
        mean[:] = [.03, -.04]
    data = np.array([.2, .4, -.6, 1e155 if density == 0 else 1e180, 1., -.3, 1.4, .7])
    design = np.ascontiguousarray(np.c_[np.ones(data.size), np.linspace(-.3, .3, data.size)])
    omega = 8e307 if density == 0 else 1e300
    if alpha is None:
        alpha = 1e-10 if density == 0 else 1e-120
    gamma = 0. if density == 0 else .15
    theta = np.r_[mean, omega, np.full(p, alpha / p), np.full(p, gamma), np.full(q, .1 / q),
                  2., *([] if density == 0 else [8.])]
    return spec, np.ascontiguousarray(theta), data, design, 1.


def evaluate(bridge, spec, theta, data, design, h0, mask):
    required = bridge.lib.aparch_range_workspace(ctypes.byref(spec), mask)
    assert required > 0
    arena = np.full(required + 64, 0xA5, dtype=np.uint8)
    offset = 16 + (-arena.ctypes.data % 16)
    guarded = [np.full(length + 2, SENTINEL) for length in (1, theta.size, theta.size**2)]
    outputs = [array[1:-1] for array in guarded]
    for bit, output in zip((1, 2, 4), outputs):
        if mask & bit:
            output.fill(np.nan)
    execution = ctypes.c_uint()
    original = theta.copy(), data.copy(), design.copy()
    status = bridge.lib.aparch_range_evaluate(ctypes.byref(spec), mask, theta.ctypes.data,
        data.ctypes.data, design.ctypes.data, data.size, h0, arena.ctypes.data + offset, required,
        *(output.ctypes.data if mask & bit else None for bit, output in zip((1, 2, 4), outputs)),
        ctypes.byref(execution))
    for left, right in zip(original, (theta, data, design)):
        np.testing.assert_array_equal(left, right)
    for array in guarded:
        np.testing.assert_array_equal(array[[0, -1]], SENTINEL)
    np.testing.assert_array_equal(arena[:offset], 0xA5)
    np.testing.assert_array_equal(arena[offset + required:], 0xA5)
    outputs[2] = outputs[2].reshape(theta.size, theta.size)
    for bit, output in zip((1, 2, 4), outputs):
        if not mask & bit:
            np.testing.assert_array_equal(output, SENTINEL)
    bridge.record["checks"].append(dict(mode=spec.mode, fixed=spec.fixed, density=spec.density,
        orders=[spec.ar, spec.ma, spec.p, spec.q], mask=mask, status=status,
        execution=execution.value, finite_requested=all(np.isfinite(output).all()
            for bit, output in zip((1, 2, 4), outputs) if mask & bit)))
    return status, execution.value, outputs


def assert_requested(actual, expected, mask):
    for bit, output, reference, relative, absolute in zip(
            (1, 2, 4), actual, expected, (0., G_RTOL, H_RTOL), (F_ATOL, G_ATOL, H_ATOL)):
        if mask & bit:
            assert np.isfinite(output).all()
            np.testing.assert_allclose(output, reference, rtol=relative, atol=absolute)


def error_metrics(actual, expected, mask):
    return {str(bit): float(np.max(np.abs(output - reference) / (1 + np.abs(reference))))
            for bit, output, reference in zip((1, 2, 4), actual, expected) if mask & bit}


@pytest.mark.parametrize("mode,wide", [(0, False), (0, True), (1, False), (1, True), (2, True)])
@pytest.mark.parametrize("density", [0, 1])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_large_weighted_shocks_all_masks_and_generated_orders(bridge, mode, wide, density, sign):
    spec, theta, data, design, h0 = sample(mode, wide, density)
    data[3] *= sign
    low, high = [complete_oracle(spec, theta, data, design, h0, digits) for digits in (100, 150)]
    assert_requested(low[0], high[0], 7)
    assert all(np.isfinite(output).all() for output in high[0])
    assert mp.mpf(float(np.max(np.abs(data))))**2 > np.finfo(float).max
    if density == 0:
        reduction, reduction_variance = complete_oracle(spec, theta, data, design, h0, 150, garch=True)
        common = list(range(spec.mean + 1 + spec.p)) + list(range(spec.mean + 1 + 2 * spec.p, theta.size - 1))
        assert_requested((high[0][0], high[0][1][common], high[0][2][np.ix_(common, common)]),
                         (reduction[0], reduction[1][common], reduction[2][np.ix_(common, common)]), 7)
        np.testing.assert_allclose(high[1], reduction_variance, rtol=2e-15, atol=0.)
    for fixed in (0, 1):
        spec.fixed = fixed
        full_status, execution, full = evaluate(bridge, spec, theta, data, design, h0, 7)
        assert full_status == 0
        assert execution == (2 if fixed else 1)
        assert_requested(full, high[0], 7)
        bridge.record["checks"].append(dict(test="complete-MP-AD", mode=mode, fixed=fixed, sign=sign,
            density=density, errors=error_metrics(full, high[0], 7),
            value_error=float(abs(full[0][0] - high[0][0][0])),
            raw_alpha_derivative_representable=False,
            composed_gradient_max=float(np.max(np.abs(high[0][1]))),
            composed_hessian_max=float(np.max(np.abs(high[0][2])))))
        for mask in range(1, 8):
            status, execution, outputs = evaluate(bridge, spec, theta, data, design, h0, mask)
            assert status == 0
            assert execution == (2 if fixed else 1)
            assert_requested(outputs, high[0], mask)
            assert_requested(outputs, full, mask)
        if mode != 2:
            required = bridge.lib.aparch_range_workspace(ctypes.byref(spec), 1)
            memory = np.empty(required + 64, dtype=np.uint8)
            offset = 16 + (-memory.ctypes.data % 16)
            residual, variance = np.full(data.size, np.nan), np.full(data.size, np.nan)
            assert bridge.lib.aparch_range_filter(ctypes.byref(spec), theta.ctypes.data,
                data.ctypes.data, data.size, h0, memory.ctypes.data + offset, required,
                residual.ctypes.data, variance.ctypes.data) == 0
            np.testing.assert_allclose(variance, high[1], rtol=2e-13, atol=0.)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("alpha", [0., 1e-10])
def test_stateful_anchored_charts_select_units_before_advance(bridge, reduced, alpha):
    spec, theta, data, design, h0 = sample(1, False, 0, alpha=alpha)
    low, high = [complete_oracle(spec, theta, data, design, h0, digits, chart=True)[0]
                 for digits in (100, 150)]
    width = spec.mean if reduced else theta.size
    expected = high[0], high[1][:width], high[2][:width, :width]
    assert_requested((low[0], low[1][:width], low[2][:width, :width]), expected, 7)
    for fixed in (False, True):
        for mask in range(2, 8):
            required = bridge.lib.aparch_range_chart_workspace(fixed, reduced, mask)
            assert required > 0
            arena = np.full(required + 64, 0xA5, dtype=np.uint8)
            offset = 16 + (-arena.ctypes.data % 16)
            guarded = [np.full(size + 2, SENTINEL) for size in (1, width, width**2)]
            outputs = [array[1:-1] for array in guarded]
            for bit, output in zip((1, 2, 4), outputs):
                if mask & bit:
                    output.fill(np.nan)
            status = bridge.lib.aparch_range_chart_evaluate(fixed, reduced, mask,
                theta.ctypes.data, data.ctypes.data, data.size, h0, arena.ctypes.data + offset,
                required, *(output.ctypes.data if mask & bit else None
                            for bit, output in zip((1, 2, 4), outputs)))
            assert status == 0
            outputs[2] = outputs[2].reshape(width, width)
            assert_requested(outputs, expected, mask)
            for array in guarded:
                np.testing.assert_array_equal(array[[0, -1]], SENTINEL)
            for bit, output in zip((1, 2, 4), outputs):
                if not mask & bit:
                    np.testing.assert_array_equal(output, SENTINEL)
            np.testing.assert_array_equal(arena[:offset], 0xA5)
            np.testing.assert_array_equal(arena[offset + required:], 0xA5)
            bridge.record["checks"].append(dict(test="stateful-anchored-chart", fixed=fixed,
                reduced=reduced, alpha=alpha, mask=mask, status=status,
                errors=error_metrics(outputs, expected, mask)))


@pytest.mark.parametrize("mode,wide", [(0, False), (1, False), (2, True)])
def test_zero_alpha_does_not_require_an_unweighted_value(bridge, mode, wide):
    spec, theta, data, design, h0 = sample(mode, wide, 0, alpha=0.)
    expected, _ = complete_oracle(spec, theta, data, design, h0, 150)
    for fixed in (0, 1):
        spec.fixed = fixed
        for mask in range(1, 8):
            status, _, outputs = evaluate(bridge, spec, theta, data, design, h0, mask)
            assert status == 0
            assert_requested(outputs, expected, mask)


def test_requested_hessian_can_be_unrepresentable_when_value_and_gradient_are_finite(bridge):
    spec, theta, data, design, h0 = sample(0, False, 1, alpha=1e-170)
    theta[0] = .12
    theta[3] = 0.
    data[0], data[3] = 1e160, .8
    h0 = 1e300
    expected, _ = complete_oracle(spec, theta, data, design, h0, 150)
    assert np.isfinite(expected[0]).all() and np.isfinite(expected[1]).all()
    assert not np.isfinite(expected[2]).all()
    for fixed in (0, 1):
        spec.fixed = fixed
        for mask in range(1, 8):
            status, _, outputs = evaluate(bridge, spec, theta, data, design, h0, mask)
            assert status == (-3 if mask & 4 else 0)
            if not mask & 4:
                assert_requested(outputs, expected, mask)


@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_overflowing_raw_shock_base_has_finite_complete_jet(bridge, p, q, sign):
    spec = Spec(0, 0, p, q, 0, 0, 0, 1)
    theta = np.ascontiguousarray(np.r_[.12, np.full(p, 1e-78 / p),
        np.full(p, -sign * .99), np.full(q, .01 / q), .5, 2.5])
    data = np.array([sign * 9.1e307, .3, -.4, .7, 1.])
    design = np.zeros((data.size, 2))
    h0 = 1.7e308
    with np.errstate(over="ignore"):
        raw_base = np.abs(data[0]) - theta[1 + p] * data[0]
        multiplied_base = (sign - theta[1 + p]) * data[0]
    assert np.isposinf(raw_base) and np.isposinf(multiplied_base)
    with mp.workdps(160):
        residual = mp.mpf(float(data[0]))
        gamma, delta, nu = (mp.mpf(float(theta[index])) for index in (1 + p, -2, -1))
        maximum = mp.mpf(float(np.finfo(float).max))
        base = abs(residual) - gamma * residual
        weighted_shock = mp.mpf(float(theta[1])) * base**delta
        assert base > maximum and 0 < weighted_shock < maximum
        assert abs(gamma) < 1 and 0 < delta < nu and nu > 2
        standardized2 = residual**2 / mp.mpf(h0)
        assert standardized2 / (nu - 2) < maximum
        assert (nu + 1) * standardized2 < maximum
        absolute_moment = ((nu - 2)**(delta / 2) * mp.gamma((delta + 1) / 2)
            * mp.gamma((nu - delta) / 2) / (mp.sqrt(mp.pi) * mp.gamma(nu / 2)))
        shock_moment = absolute_moment * ((1 - gamma)**delta + (1 + gamma)**delta) / 2
        persistence = sum(mp.mpf(float(value)) for value in theta[1:1 + p]) * shock_moment
        persistence += sum(mp.mpf(float(value)) for value in theta[1 + 2 * p:-2])
        assert 0 < persistence < 1
    low, high = [complete_oracle(spec, theta, data, design, h0, digits) for digits in (100, 160)]
    assert all(np.isfinite(output).all() for output in high[0])
    assert np.isfinite(high[1]).all() and np.all(high[1] > 0)
    assert_requested(low[0], high[0], 7)
    np.testing.assert_array_equal(low[1], high[1])
    for fixed in (0, 1):
        spec.fixed = fixed
        for mask in range(1, 8):
            status, execution, outputs = evaluate(bridge, spec, theta, data, design, h0, mask)
            assert status == 0
            assert execution == (2 if fixed else 1)
            assert_requested(outputs, high[0], mask)
            bridge.record["checks"].append(dict(test="base-multiplication-overflow-edge",
                sign=sign, orders=[p, q], fixed=fixed, mask=mask, status=status,
                exact_base=str(base), weighted_shock=str(weighted_shock),
                persistence=str(persistence), complete_oracle_finite=True,
                errors=error_metrics(outputs, high[0], mask),
                value_error=float(abs(outputs[0][0] - high[0][0][0])) if mask & 1 else None))
        required = bridge.lib.aparch_range_workspace(ctypes.byref(spec), 1)
        memory = np.empty(required + 64, dtype=np.uint8)
        offset = 16 + (-memory.ctypes.data % 16)
        variance = np.full(data.size, np.nan)
        assert bridge.lib.aparch_range_filter(ctypes.byref(spec), theta.ctypes.data,
            data.ctypes.data, data.size, h0, memory.ctypes.data + offset, required,
            None, variance.ctypes.data) == 0
        np.testing.assert_allclose(variance, high[1], rtol=2e-13, atol=0.)
