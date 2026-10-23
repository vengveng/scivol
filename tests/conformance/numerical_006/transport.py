"""Test-only transport, explicit retained sample construction, no old package."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from types import SimpleNamespace

import numpy as np

from native_api import evaluate
from .historical_design import _har_history_means


@lru_cache(maxsize=1)
def numerical_library():
    path = os.environ.get("SCIVOLAPI_NUMERICAL_BRIDGE")
    if not path:
        raise RuntimeError("Chief must build the new-only numerical_006 bridge and set SCIVOLAPI_NUMERICAL_BRIDGE")
    lib = ctypes.CDLL(path)
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    lib.numerical_joint.argtypes = [ctypes.c_uint, ctypes.c_uint, size, size, size, ctypes.c_bool,
        ptr, size, ptr, ctypes.c_double, ptr, size, ctypes.c_uint, ctypes.c_bool, ptr, ptr, ptr, ptr, ptr, ptr]
    lib.numerical_joint.restype = ctypes.c_int
    lib.numerical_map.argtypes = [size, size, size, ptr, size, ctypes.c_bool, ptr]
    lib.numerical_map.restype = ctypes.c_int
    lib.numerical_density.argtypes = [ctypes.c_uint, ctypes.c_uint, ptr, ptr, ptr, ptr]
    lib.numerical_density.restype = ctypes.c_int
    lib.numerical_moment.argtypes = [ptr, ctypes.c_uint, ptr, ptr, ptr]
    lib.numerical_moment.restype = ctypes.c_int
    lib.numerical_student_series.argtypes = [ptr, ptr, size, ctypes.c_double, ptr, ptr]
    lib.numerical_student_series.restype = ctypes.c_int
    lib.numerical_skew11.argtypes = [ctypes.c_bool, ptr, ptr, size, ctypes.c_double, ptr, ptr]
    lib.numerical_skew11.restype = ctypes.c_int
    return lib


def student_series(variance, square, nu):
    variance, square = (np.ascontiguousarray(v, dtype=np.float64) for v in (variance, square))
    f, h = ctypes.c_double(), ctypes.c_double()
    status = numerical_library().numerical_student_series(variance.ctypes.data, square.ctypes.data,
        len(square), nu, ctypes.byref(f), ctypes.byref(h))
    assert status == 0
    return f.value, h.value


def forced_skew11(theta, y, fixed):
    theta, y = (np.ascontiguousarray(v, dtype=np.float64) for v in (theta, y))
    f, g = np.full(1, np.nan), np.full(5, np.nan)
    status = numerical_library().numerical_skew11(fixed, theta.ctypes.data, y.ctypes.data,
        len(y), float(np.mean(y*y)), f.ctypes.data, g.ctypes.data)
    assert status == 0 and np.isfinite(f).all() and np.isfinite(g).all()
    return f[0], g


def local_density(density, point, mask=7):
    point = np.ascontiguousarray(point, dtype=np.float64)
    f, g, h = np.full(1, np.nan), np.full(len(point), np.nan), np.full((len(point), len(point)), np.nan)
    status = numerical_library().numerical_density(density, mask, point.ctypes.data,
        f.ctypes.data, g.ctypes.data, h.ctypes.data)
    assert status == 0
    return f, g, h


def standardized(density, shape, values, mask=7):
    width = len(shape)
    f, g, h = 0., np.zeros(width), np.zeros((width, width))
    for residual in np.asarray(values).ravel():
        a, b, c = local_density(density, [residual, 1., *shape], mask)
        if mask & 1:
            f += a[0]
        if mask & 2:
            g += b[2:]
        if mask & 4:
            h += c[2:, 2:]
    return f, g, h


class DensityTransport:
    """Pointer signatures retained for assertions, backed only by new owners."""

    @staticmethod
    def array(address, count):
        return np.ctypeslib.as_array((ctypes.c_double * count).from_address(address))

    def __getattr__(self, name):
        family = next((key for key in ("gh", "nig", "vg") if name.startswith("_" + key + "_standardized_nll")), None)
        if family is None:
            raise AttributeError(name)
        density, width = {"gh": (4, 3), "nig": (5, 2), "vg": (7, 2)}[family]
        if name.endswith("_grad_hess"):
            def jet(shape, values, f, g, h, count):
                result = standardized(density, self.array(shape, width), self.array(values, count))
                self.array(f, 1)[:] = result[0]
                self.array(g, width)[:] = result[1]
                self.array(h, width * width)[:] = result[2].ravel()
            return jet
        return lambda shape, values, count: standardized(density, self.array(shape, width), self.array(values, count), 1)[0]


class JointSample:
    def __init__(self, data, features, p, q, density, *, fixed=False, family=3):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.features = np.ascontiguousarray(features, dtype=np.float64)
        self.columns = self.features.shape[1]
        self.p_vol, self.q_vol = p, q
        self.h0 = float(np.mean(self.data ** 2))
        self.family, self.density, self.fixed = family, density, fixed
        self.shape_start = self.columns + 1 + (1 if family == 0 else 2) * p + q + int(family == 3)
        self.size = self.shape_start + (2 if density in (2, 5, 6, 7) else 3 if density == 4 else 0 if density == 0 else 1)

    def evaluate(self, theta, mask, *, transformed=False, paths=False):
        point = np.ascontiguousarray(theta, dtype=np.float64)
        f, g, h = np.full(1, np.nan), np.full(self.size, np.nan), np.full((self.size, self.size), np.nan)
        residuals, variances = np.full(len(self.data), np.nan), np.full(len(self.data), np.nan)
        valid = ctypes.c_uint()
        status = numerical_library().numerical_joint(self.family, self.density, self.columns,
            self.p_vol, self.q_vol, self.fixed, self.data.ctypes.data, len(self.data),
            self.features.ctypes.data, self.h0, point.ctypes.data, point.size, mask, transformed,
            f.ctypes.data, g.ctypes.data, h.ctypes.data, ctypes.byref(valid),
            residuals.ctypes.data if paths else None, variances.ctypes.data if paths else None)
        assert status == 0 and valid.value == mask, (status, valid.value, mask)
        for bit, output in zip((1, 2, 4), (f, g, h)):
            assert np.isfinite(output).all() if bit & mask else np.isnan(output).all()
        result = (f[0] if mask & 1 else None, g if mask & 2 else None, h if mask & 4 else None)
        return result + (residuals, variances) if paths else result

    def inverse(self, theta):
        return self.map(theta, forward=False)

    def map(self, point, *, forward):
        if self.family != 3 or self.density != 7:
            raise ValueError("This retained map transport is APARCH-VG only")
        point = np.ascontiguousarray(point, dtype=np.float64)
        result = np.full_like(point, np.nan)
        status = numerical_library().numerical_map(self.columns, self.p_vol, self.q_vol,
            point.ctypes.data, len(point), forward, result.ctypes.data)
        if status:
            raise ValueError("Parameters outside the transformed chart")
        return result


def retained_sample(exogenous=4, p=3, q=2, density=7, *, fixed=False):
    rng = np.random.default_rng(432)
    x = rng.normal(size=(96, exogenous))
    data = rng.standard_t(7, size=96) * .65 + x @ np.linspace(.18, -.12, exogenous)
    for time in range(1, len(data)):
        data[time] += .18 * data[time - 1]
    # Historical physical hold-back happened before historical design construction.
    data, x = np.ascontiguousarray(data[3:]), np.ascontiguousarray(x[3:])
    features = np.column_stack((np.ones(len(data)), _har_history_means(data, 1),
                                _har_history_means(data, 3), x))
    return JointSample(data, features, p, q, density, fixed=fixed)


def skew_moment(point):
    point = np.ascontiguousarray(point, dtype=np.float64)
    results = []
    for mask in range(1, 8):
        f, g, h = np.full(1, np.nan), np.full(4, np.nan), np.full((4, 4), np.nan)
        assert numerical_library().numerical_moment(point.ctypes.data, mask, f.ctypes.data,
            g.ctypes.data, h.ctypes.data) == 0
        results.append((f, g, h))
    for mask, result in enumerate(results, 1):
        for bit, actual, full in zip((1, 2, 4), result, results[-1]):
            if mask & bit:
                np.testing.assert_allclose(actual, full, rtol=2e-10, atol=2e-11)
            else:
                assert np.isnan(actual).all()
    return results[-1][0][0], results[-1][1], results[-1][2]


def surface(theta, y, mean_kind, volatility_kind, p, q, fixed_lambda=None):
    density = "variance_gamma" if fixed_lambda == np.inf else "nig" if fixed_lambda == -.5 else "hyperbolic" if fixed_lambda == 1 else "gh"
    if volatility_kind == "constant":
        model = "arma" if mean_kind == "arma" else "linear"
    else:
        model = ("arma_" if mean_kind == "arma" else "linear_" if mean_kind == "linear" else "") + volatility_kind
    design = np.linspace(-.4, .5, len(y))[:, None] if mean_kind == "linear" else np.empty((len(y), 0))
    start = max(1, p, q) if mean_kind == "arma" else 0
    case = dict(model=model, density=density, p=p if model not in ("linear", "arma") else 1 if model == "arma" else 0,
        q=q if model not in ("linear", "arma") else 1 if model == "arma" else 0,
        p_ar=1 if mean_kind == "arma" else 0, q_ma=1 if mean_kind == "arma" else 0,
        p_vol=p, q_vol=q, design=design, intercept=True, y=y, h0=float(np.mean(y*y)),
        first=start, stop=len(y), hold_back=0, transformed=False, parameters=theta)
    actual = evaluate(case, 7)
    assert actual["status"] == 0 and actual["valid_mask"] == 7, actual["reasons"]
    return actual["f"], actual["g"], actual["h"]
