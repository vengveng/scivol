"""Private complete-jet/path transport, not the migrated public filter API."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os

import numpy as np


SENTINEL = -987654321.125


@lru_cache(maxsize=1)
def library():
    path = os.environ.get("SCIVOLAPI_NUMERICAL_PATH_BRIDGE")
    if not path:
        raise RuntimeError("Chief must build the new-only joint_paths.cpp and set SCIVOLAPI_NUMERICAL_PATH_BRIDGE")
    result = ctypes.CDLL(path)
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    result.scivolapi_numerical_joint_paths.argtypes = [
        ctypes.c_uint, ctypes.c_uint, size, size, size, ptr, size, ptr,
        ctypes.c_double, ctypes.c_double, ptr, size, ctypes.c_uint, ctypes.c_uint,
        ptr, ptr, ptr, ptr, ptr, size, ptr, ptr]
    result.scivolapi_numerical_joint_paths.restype = ctypes.c_int
    return result


def guarded(shape):
    count = int(np.prod(shape))
    storage = np.full(count+2, SENTINEL)
    return storage, storage[1:-1].reshape(shape)


def evaluate(case, mask, *, paths=True, lane=0, path_capacity=None):
    theta, data, design = (np.ascontiguousarray(case[key], dtype=np.float64)
                           for key in ("theta", "data", "design"))
    if theta.ndim != 1 or data.ndim != 1 or design.ndim != 2 or design.shape[0] != len(data):
        raise ValueError("Expected flat theta/data and a row-major full linear design")
    width, count = len(theta), len(data)
    outputs = [guarded(shape) for shape in ((1,), (width,), (width,width), (count,), (count,))]
    arrays = [a for _,a in outputs]
    f,g,h,e,v = arrays
    valid, dispatch = ctypes.c_uint(999), ctypes.c_uint(999)
    before = tuple(a.tobytes() for a in (theta,data,design))
    status = library().scivolapi_numerical_joint_paths(
        case["family"], case["density"], design.shape[1], case["p"], case["q"],
        data.ctypes.data, count, design.ctypes.data, case["h0"], case["implicit_log_variance"],
        theta.ctypes.data, width, mask, lane, f.ctypes.data, g.ctypes.data, h.ctypes.data,
        e.ctypes.data if paths else None, v.ctypes.data if paths else None,
        count if path_capacity is None else path_capacity, ctypes.byref(valid), ctypes.byref(dispatch))
    assert before == tuple(a.tobytes() for a in (theta,data,design))
    for storage,_ in outputs:
        np.testing.assert_array_equal(storage[[0,-1]].view(np.uint64), np.full(2,SENTINEL).view(np.uint64))
    return dict(status=status, valid_mask=valid.value, dispatch=dispatch.value,
                f=f[0], g=g, h=h, residuals=e, sigma2=v, sentinel=SENTINEL)


def assert_valid(result, mask, *, paths):
    assert result["status"] == 0 and result["valid_mask"] == mask
    assert result["dispatch"] == 0
    for bit,name in ((1,"f"),(2,"g"),(4,"h")):
        actual=np.asarray(result[name])
        if mask&bit:
            assert np.isfinite(actual).all()
        else:
            np.testing.assert_array_equal(actual.view(np.uint64), np.full_like(actual,SENTINEL).view(np.uint64))
    for name in ("residuals","sigma2"):
        if paths:
            assert np.isfinite(result[name]).all()
        else:
            np.testing.assert_array_equal(result[name].view(np.uint64), np.full_like(result[name],SENTINEL).view(np.uint64))
    if paths:
        assert np.all(result["sigma2"]>0)
