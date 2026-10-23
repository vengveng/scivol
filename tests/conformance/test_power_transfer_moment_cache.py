"""Retained precision-cache state and checked symmetric moment contracts.

The chief supplies a test-only bridge built from power_transfer_moment_probe.cpp.
This file neither builds it nor loads the old package. Missing transport fails.
"""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def moment_backend():
    backend = ctypes.CDLL(str(Path(os.environ["SCIVOL_POWER_TRANSFER_MOMENT_BRIDGE"]).resolve(strict=True)))
    pointer = ctypes.c_void_p
    signatures = {
        "probe_persistence": ([pointer, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint, pointer], ctypes.c_bool),
        "probe_power_moment": ([pointer, ctypes.c_uint, ctypes.c_uint, pointer, pointer, pointer], ctypes.c_bool),
        "probe_negative_moment": ([pointer, ctypes.c_uint, ctypes.c_uint, pointer, pointer, pointer], ctypes.c_bool),
        "probe_precision_bytes": ([], ctypes.c_size_t),
        "probe_precision_reset": ([pointer], None),
        "probe_precision_prepare": ([pointer, ctypes.c_uint, pointer, ctypes.c_double, ctypes.c_double], ctypes.c_bool),
        "probe_precision_snapshot": ([pointer, pointer], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(backend, name)
        function.argtypes, function.restype = arguments, result
    return backend


class MomentPrecision:
    def __init__(self, backend):
        self.backend = backend
        # The cache contains doubles and integer flags; malloc alignment suffices.
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


def test_failed_tighter_real_moment_does_not_reject_weaker_request(moment_backend):
    shape = (2.0001, 0.99)
    delta = np.nextafter(shape[0], 0.0)
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, shape, delta, 1e-24)
    coarse = retained.snapshot()
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert retained.prepare(2, shape, delta, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), coarse)


def test_failed_tighter_first_request_allows_weaker_real_preparation(moment_backend):
    shape = (2.0001, 0.99)
    delta = np.nextafter(shape[0], 0.0)
    retained = MomentPrecision(moment_backend)
    assert not retained.prepare(2, shape, delta, 1e-28)
    assert retained.prepare(2, shape, delta, 1e-24)
    fresh = MomentPrecision(moment_backend)
    assert fresh.prepare(2, shape, delta, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), fresh.snapshot())


def test_successful_tighter_precision_serves_weaker_request_unchanged(moment_backend):
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, (8.0, 0.2), 0.2, 1e-28)
    tighter = retained.snapshot()
    assert retained.prepare(2, (8.0, 0.2), 0.2, 1e-24)
    np.testing.assert_array_equal(retained.snapshot(), tighter)


def test_precision_cache_retains_exact_point_and_density_validity(moment_backend):
    retained = MomentPrecision(moment_backend)
    assert retained.prepare(2, (8.0, 0.2), 0.2, 1e-24)
    assert not retained.prepare(2, (8.0, 0.2), 8.0, 1e-24)
    assert not retained.prepare(2, (8.0, 1.0), 0.2, 1e-24)
    assert retained.prepare(1, (8.0,), 0.2, 1e-24)
    student = retained.snapshot()
    fresh = MomentPrecision(moment_backend)
    assert fresh.prepare(1, (8.0,), 0.2, 1e-24)
    np.testing.assert_array_equal(student, fresh.snapshot())
    assert retained.prepare(2, (8.0, 0.2), 0.2, 1e-24)
    assert not np.array_equal(retained.snapshot()[:4], student[:4])


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("density,parameter", (
    (1, -1.0), (1, 1.0), (1, 2.0), (1, np.nan), (1, np.inf),
    (3, -1.0), (3, 0.0), (3, np.nan), (3, np.inf), (1, None), (3, None),
))
def test_checked_symmetric_moment_rejects_invalid_density(moment_backend, density, parameter, mask):
    shape = None if parameter is None else np.array([parameter])
    outputs = [np.full(1, 91.0) for _ in range(3)]
    assert not moment_backend.probe_negative_moment(
        None if shape is None else shape.ctypes.data, density, mask,
        *(output.ctypes.data for output in outputs))
    for output in outputs:
        np.testing.assert_array_equal(output, 91.0)


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("density,shape", (
    (0, []), (1, [np.nextafter(2.0, np.inf)]), (1, [8.0]), (1, [1e16]),
    (3, [np.nextafter(0.0, np.inf)]), (3, [0.8]), (3, [2.0]), (3, [50.0]),
))
def test_symmetric_moment_exact_identity_and_sparse_outputs(moment_backend, density, shape, mask):
    parameters = np.ascontiguousarray(shape, dtype=np.float64)
    width = len(shape)
    outputs = [np.full(1, 91.0), np.full(width, 92.0), np.full(width * width, 93.0)]
    before = [output.copy() for output in outputs]
    assert moment_backend.probe_negative_moment(parameters.ctypes.data, density, mask,
        *(output.ctypes.data for output in outputs))
    for bit, output, untouched, expected in zip((1, 2, 4), outputs, before, (0.5, 0.0, 0.0), strict=True):
        np.testing.assert_array_equal(output, expected if mask & bit else untouched)
