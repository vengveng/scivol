"""Checked moment domains and request-specific retained precision."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_moment_backend(path):
    backend = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    signatures = {
        "probe_negative_moment": ([pointer, ctypes.c_uint, ctypes.c_uint,
                                   pointer, pointer, pointer], ctypes.c_bool),
        "probe_precision_bytes": ([], ctypes.c_size_t),
        "probe_precision_reset": ([pointer], None),
        "probe_precision_prepare": ([pointer, ctypes.c_uint, pointer,
                                     ctypes.c_double, ctypes.c_double], ctypes.c_bool),
        "probe_precision_snapshot": ([pointer, pointer], None),
        "probe_precise_persistence": ([pointer, ctypes.c_uint, pointer, ctypes.c_double,
                                       pointer, pointer, ctypes.c_size_t, pointer,
                                       ctypes.c_size_t, pointer], ctypes.c_bool),
        "probe_power_persistence": ([pointer, pointer, ctypes.c_size_t,
                                     ctypes.c_size_t, pointer], ctypes.c_bool),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(backend, name)
        function.argtypes, function.restype = arguments, result
    return backend


@pytest.fixture(scope="module")
def moment_backend(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None or sys.platform == "win32":
        pytest.skip("The standalone moment test requires a Unix C++20 compiler")
    directory = tmp_path_factory.mktemp("jet_math_moment_validity")
    library = directory / ("moments.dylib" if sys.platform == "darwin" else "moments.so")
    native = ROOT / "scivol/_native"
    frame_flags = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
                   if "clang" in Path(compiler).name else ["-Werror=frame-larger-than=16384"])
    command = [compiler, "-std=c++20", "-O3", "-fno-exceptions", "-fno-rtti",
               "-ffp-contract=fast", "-funroll-loops", "-march=native", "-fPIC",
               *frame_flags, "-dynamiclib" if sys.platform == "darwin" else "-shared"]
    for include in (native, ROOT / "scivol/_csrc", native / "third_party/boost",
                    native / "third_party/tlfloat"):
        command.extend(["-I", str(include)])
    command.extend([str(ROOT / "tests/native_jet_math_moment_validity.cpp"),
                    str(native / "aparch_moment_precision.cpp"),
                    str(native / "gh_math_translation.cpp"), "-o", str(library)])
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return load_moment_backend(library)


class MomentPrecision:
    def __init__(self, backend):
        self.backend = backend
        self.storage = np.zeros((backend.probe_precision_bytes() + 7) // 8, dtype=np.float64)
        backend.probe_precision_reset(self.storage.ctypes.data)

    def prepare(self, density, shape, delta, tolerance):
        parameters = np.ascontiguousarray(shape, dtype=np.float64)
        return self.backend.probe_precision_prepare(self.storage.ctypes.data, density,
            parameters.ctypes.data, delta, tolerance)

    def snapshot(self):
        output = np.empty(8)
        self.backend.probe_precision_snapshot(self.storage.ctypes.data, output.ctypes.data)
        return output

    def persistence(self, density, shape, delta, alpha, gamma, beta):
        shape, alpha, gamma, beta = [np.ascontiguousarray(value, dtype=np.float64)
                                    for value in (shape, alpha, gamma, beta)]
        output = np.empty(4)
        valid = self.backend.probe_precise_persistence(self.storage.ctypes.data,
            density, shape.ctypes.data, delta, alpha.ctypes.data, gamma.ctypes.data,
            alpha.size, beta.ctypes.data, beta.size, output.ctypes.data)
        return valid, output

    def power_persistence(self, shape, delta, alpha, gamma, beta):
        theta = np.ascontiguousarray([.1, *alpha, *gamma, *beta, delta, *shape],
                                     dtype=np.float64)
        output = np.empty(4)
        valid = self.backend.probe_power_persistence(self.storage.ctypes.data,
            theta.ctypes.data, len(alpha), len(beta), output.ctypes.data)
        return valid, output


def unresolved_persistence_weights(backend, shape, delta):
    mp = pytest.importorskip("mpmath")
    coarse = MomentPrecision(backend)
    assert coarse.prepare(2, shape, delta, 1e-24)
    halves = coarse.snapshot()[:4]
    with mp.workdps(100):
        absolute = mp.fsum(map(mp.mpf, halves))
        alpha = np.nextafter(float(1 / absolute), 0.)
        beta = float(1 - mp.mpf(alpha) * absolute)
    assert alpha > 0. and beta > 0.
    return alpha, beta


def test_failed_tighter_real_moment_does_not_reject_weaker_request(moment_backend):
    shape = (2.0001, .99)
    delta = np.nextafter(shape[0], 0.)
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, shape, delta, 1e-24)
    coarse = retained.snapshot()
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert retained.prepare(2, shape, delta, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), coarse)


def test_failed_tighter_first_request_allows_weaker_real_preparation(moment_backend):
    shape = (2.0001, .99)
    delta = np.nextafter(shape[0], 0.)
    retained = MomentPrecision(moment_backend)
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert retained.prepare(2, shape, delta, 1e-24)
    fresh = MomentPrecision(moment_backend)
    assert fresh.prepare(2, shape, delta, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), fresh.snapshot())


def test_successful_tighter_precision_serves_weaker_request_unchanged(moment_backend):
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, (8., .2), .2, 1e-28)
    tighter = retained.snapshot()
    assert retained.prepare(2, (8., .2), .2, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), tighter)


def test_precision_cache_retains_exact_point_and_density_validity(moment_backend):
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, (8., .2), .2, 1e-24)
    assert not retained.prepare(2, (8., .2), 8., 1e-24)
    assert not retained.prepare(2, (8., 1.), .2, 1e-24)
    assert retained.prepare(1, (8.,), .2, 1e-24)
    student = retained.snapshot()
    fresh = MomentPrecision(moment_backend)
    assert fresh.prepare(1, (8.,), .2, 1e-24)
    np.testing.assert_array_equal(student, fresh.snapshot())
    assert retained.prepare(2, (8., .2), .2, 1e-24)
    assert not np.array_equal(retained.snapshot()[:4], student[:4])


@pytest.mark.parametrize("orders", [(1, 1), (3, 2)])
def test_failed_refinement_does_not_poison_changed_persistence_weights(moment_backend, orders):
    shape = (2.0001, .99)
    delta = np.nextafter(shape[0], 0.)
    alpha_value, beta_value = unresolved_persistence_weights(moment_backend, shape, delta)
    p, q = orders
    alpha, gamma, beta = np.zeros(p), np.zeros(p), np.zeros(q)
    alpha[0], beta[0] = alpha_value, beta_value
    retained = MomentPrecision(moment_backend)
    valid, output = retained.power_persistence(shape, delta, alpha, gamma, beta)
    assert not valid
    assert not np.isfinite(output[0])
    beta[0] = 0.
    recovered = retained.power_persistence(shape, delta, alpha, gamma, beta)
    fresh = MomentPrecision(moment_backend).power_persistence(shape, delta, alpha, gamma, beta)
    assert fresh[0] and recovered[0]
    assert fresh[1][0] > 0. and fresh[1][3] == 1.
    np.testing.assert_array_equal(recovered[1], fresh[1])


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("density,parameter", [
    (1, -1.), (1, 1.), (1, 2.), (1, np.nan), (1, np.inf),
    (3, -1.), (3, 0.), (3, np.nan), (3, np.inf),
    (1, None), (3, None),
])
def test_checked_symmetric_moment_rejects_invalid_density(moment_backend, density,
                                                        parameter, mask):
    shape = None if parameter is None else np.array([parameter])
    outputs = [np.full(1, 91.) for _ in range(3)]
    assert not moment_backend.probe_negative_moment(
        None if shape is None else shape.ctypes.data, density, mask,
        *(output.ctypes.data for output in outputs))
    for output in outputs:
        np.testing.assert_array_equal(output, 91.)


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("density,shape", [
    (0, []), (1, [np.nextafter(2., np.inf)]), (1, [8.]), (1, [1e16]),
    (3, [np.nextafter(0., np.inf)]), (3, [.8]), (3, [2.]), (3, [50.]),
])
def test_symmetric_moment_exact_identity_and_sparse_outputs(moment_backend, density,
                                                          shape, mask):
    parameters = np.ascontiguousarray(shape, dtype=np.float64)
    width = len(shape)
    outputs = [np.full(1, 91.), np.full(width, 92.), np.full(width * width, 93.)]
    before = [output.copy() for output in outputs]
    assert moment_backend.probe_negative_moment(parameters.ctypes.data, density, mask,
        *(output.ctypes.data for output in outputs))
    for bit, output, untouched, expected in zip((1, 2, 4), outputs, before, (.5, 0., 0.),
                                               strict=True):
        np.testing.assert_array_equal(output, expected if mask & bit else untouched)


@pytest.mark.parametrize("mask", (1, 3, 7))
@pytest.mark.parametrize("density,parameter", [
    (1, -1.), (1, 1.), (1, 2.), (1, np.nan), (1, np.inf),
    (3, -1.), (3, 0.), (3, np.nan), (3, np.inf),
    (1, None), (3, None),
])
def test_native_abi_rejects_invalid_symmetric_moment(density, parameter, mask):
    from scivol import _native_core

    shape = None if parameter is None else np.array([parameter])
    outputs = [np.full(1, 91.) for _ in range(3)]
    with pytest.raises(ValueError, match="negative-half second moment is unavailable"):
        _native_core._gjr_negative_second_moment(
            0 if shape is None else shape.ctypes.data, density, mask,
            *(output.ctypes.data for output in outputs))
    for output in outputs:
        np.testing.assert_array_equal(output, 91.)
