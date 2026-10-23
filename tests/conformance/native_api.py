"""Private test transport. All model preparation and mathematics belong to C++."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


_DENSITIES = {name: index for index, name in enumerate(
    ("normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "variance_gamma")
)}
_DOUBLE = ctypes.POINTER(ctypes.c_double)
_U64 = ctypes.POINTER(ctypes.c_uint64)
_U32 = ctypes.POINTER(ctypes.c_uint32)


@lru_cache(maxsize=1)
def _load_bridge():
    path = os.environ.get("SCIVOLAPI_TEST_BRIDGE")
    if not path:
        raise RuntimeError("Set SCIVOLAPI_TEST_BRIDGE to the independently built jet probe; legacy fallback is forbidden.")
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    library.scivolapi_probe_create.argtypes = [
        ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t,
        _DOUBLE, _U64, ctypes.c_size_t, _DOUBLE, ctypes.c_size_t, ctypes.c_uint, ctypes.c_double,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.scivolapi_probe_create.restype = ctypes.c_int
    library.scivolapi_probe_destroy.argtypes = [ctypes.c_void_p]
    library.scivolapi_probe_destroy.restype = None
    library.scivolapi_probe_evaluate.argtypes = [
        ctypes.c_void_p, _DOUBLE, ctypes.c_size_t, ctypes.c_uint,
        _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE, _U32, _U32,
    ]
    library.scivolapi_probe_evaluate.restype = ctypes.c_int
    library.scivolapi_probe_restrict.argtypes = [
        ctypes.c_void_p, _DOUBLE, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.scivolapi_probe_restrict.restype = ctypes.c_int
    library.scivolapi_probe_metadata.argtypes = [ctypes.c_void_p, _U64, _U64, _U32]
    library.scivolapi_probe_metadata.restype = ctypes.c_int
    return library


class NativeProblem:
    """Own one prepared test problem and its exclusive workspace."""

    def __init__(self, case):
        self.library = _load_bridge()
        self.handle = ctypes.c_void_p()
        observations = np.ascontiguousarray(case["y"], dtype=np.float64)
        if observations.ndim != 1:
            raise ValueError("Probe observations must be one-dimensional.")
        rows = np.ascontiguousarray(case.get("rows", np.arange(len(observations))), dtype=np.uint64)
        if rows.shape != observations.shape:
            raise ValueError("Probe row IDs must match the observations.")
        family = {"garch": 0, "gjr": 1, "gjr_garch": 1, "arma": 2, "linear": 3,
                  "egarch": 4, "aparch": 5, "arma_garch": 6, "arma_gjr": 7,
                  "linear_garch": 8, "linear_gjr": 9, "arma_egarch": 10,
                  "arma_aparch": 11, "linear_egarch": 12, "linear_aparch": 13}[case["model"]]
        design_key = "input_exogenous" if case.get("causal_kind") else "design"
        design = np.ascontiguousarray(case.get(design_key, np.empty((len(observations), 0))), dtype=np.float64)
        if design.ndim != 2 or design.shape[0] != len(observations):
            raise ValueError("Probe design must be an observation-by-column matrix.")
        density = _DENSITIES[case["density"]]
        error = ctypes.create_string_buffer(1024)
        create = self.library.scivolapi_probe_create
        orders = [int(case.get("p", 0)), int(case.get("q", 0))]
        if family >= 6:
            create = self.library.scivolapi_probe_create_joint
            create.argtypes = self.library.scivolapi_probe_create.argtypes[:4] + [
                ctypes.c_size_t, ctypes.c_size_t] + self.library.scivolapi_probe_create.argtypes[4:]
            create.restype = ctypes.c_int
            orders = [int(case.get(axis, default)) for axis, default in
                      (("p_ar", 0), ("q_ma", 0), ("p_vol", 1), ("q_vol", 1))]
        history_arguments = []
        unit_arguments = []
        with_units = any(key in case for key in ("public_scale", "internal_response_scale", "internal_design_scales"))
        if with_units or any(key in case for key in ("observation_history", "innovation_history", "causal_kind")):
            histories = [np.ascontiguousarray(case.get(key, ()), dtype=np.float64)
                         for key in ("observation_history", "innovation_history")]
            if any(values.ndim != 1 for values in histories):
                raise ValueError("Probe histories must be one-dimensional.")
            horizons = np.ascontiguousarray(case.get("causal_horizons", ()), dtype=np.uint64)
            if horizons.ndim != 1:
                raise ValueError("Probe HAR horizons must be one-dimensional.")
            create = self.library.scivolapi_probe_create_conditioned
            create.argtypes = (self.library.scivolapi_probe_create.argtypes[:4]
                               + [ctypes.c_size_t, ctypes.c_size_t]
                               + self.library.scivolapi_probe_create.argtypes[4:]
                               + [_DOUBLE, ctypes.c_size_t, _DOUBLE, ctypes.c_size_t,
                                  ctypes.c_uint, ctypes.c_size_t, _U64, ctypes.c_size_t])
            create.restype = ctypes.c_int
            orders = ([int(case.get("p_vol", 1)), int(case.get("q_vol", 1)),
                       int(case.get("p_ar", 0)), int(case.get("q_ma", 0))] if family >= 6
                      else [int(case.get("p", 0)), int(case.get("q", 0)), 0, 0])
            for values in histories:
                history_arguments.extend((values.ctypes.data_as(_DOUBLE), values.size))
            history_arguments.extend(({None: 0, "arx": 1, "harx": 2}[case.get("causal_kind")],
                                      int(case.get("causal_order", 0)), horizons.ctypes.data_as(_U64), horizons.size))
            if with_units:
                divisors = np.ascontiguousarray(case.get("internal_design_scales", ()), dtype=np.float64)
                if divisors.ndim != 1:
                    raise ValueError("Probe design divisors must be one-dimensional.")
                conditioned_types = create.argtypes
                create = self.library.scivolapi_probe_create_conditioned_units
                create.argtypes = conditioned_types + [ctypes.c_double, ctypes.c_double, _DOUBLE, ctypes.c_size_t]
                create.restype = ctypes.c_int
                unit_arguments = [float(case.get("public_scale", 1)), float(case.get("internal_response_scale", 1)),
                                  divisors.ctypes.data_as(_DOUBLE), divisors.size]
        status = create(
            family, density, *orders,
            observations.ctypes.data_as(_DOUBLE), rows.ctypes.data_as(_U64), len(observations),
            design.ctypes.data_as(_DOUBLE), design.shape[1], bool(case.get("intercept", True)),
            float(case.get("h0", np.nan)), int(case.get("hold_back", 0)),
            int(case.get("first", ctypes.c_size_t(-1).value)), int(case.get("stop", len(observations))),
            bool(case.get("transformed", False)), ctypes.byref(self.handle), error, len(error),
            *history_arguments, *unit_arguments,
        )
        if status:
            raise ValueError(f"Native preparation failed ({status}): {error.value.decode()}")
        self.rows = rows
        self.derivative_width = None

    def restricted(self, theta, indices):
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.uintp)
        result = NativeProblem.__new__(NativeProblem)
        result.library = self.library
        result.handle = ctypes.c_void_p()
        result.rows = self.rows.copy()
        result.derivative_width = len(indices)
        error = ctypes.create_string_buffer(1024)
        status = self.library.scivolapi_probe_restrict(
            self.handle, theta.ctypes.data_as(_DOUBLE), len(theta),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)), len(indices),
            ctypes.byref(result.handle), error, len(error),
        )
        if status:
            raise ValueError(f"Native restriction failed ({status}): {error.value.decode()}")
        return result

    def close(self):
        if self.handle.value:
            self.library.scivolapi_probe_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def evaluate(self, parameters, request):
        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        if parameters.ndim != 1:
            raise ValueError("Probe parameters must be one-dimensional.")
        width = parameters.size
        d = width if self.derivative_width is None else self.derivative_width
        sentinel = -987654321.125
        value = np.full(1, sentinel)
        gradient = np.full(d, sentinel)
        hessian = np.full((d, d), sentinel)
        theta = np.full(width, np.nan)
        valid = ctypes.c_uint32()
        reasons = np.full(3, 999, dtype=np.uint32)
        status = self.library.scivolapi_probe_evaluate(
            self.handle, parameters.ctypes.data_as(_DOUBLE), width, request,
            value.ctypes.data_as(_DOUBLE), gradient.ctypes.data_as(_DOUBLE),
            hessian.ctypes.data_as(_DOUBLE), theta.ctypes.data_as(_DOUBLE),
            ctypes.byref(valid), reasons.ctypes.data_as(_U32),
        )
        first, count, dispatch = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint32()
        metadata_status = self.library.scivolapi_probe_metadata(
            self.handle, ctypes.byref(first), ctypes.byref(count), ctypes.byref(dispatch)
        )
        if metadata_status:
            raise RuntimeError(f"Native metadata failed ({metadata_status}).")
        return dict(f=value[0], g=gradient, h=hessian, status=status, valid_mask=valid.value,
                    reasons=reasons, theta=theta, sentinel=sentinel, dispatch=dispatch.value,
                    scored_rows=self.rows[first.value:first.value + count.value].copy())


def evaluate(case, request=7):
    with NativeProblem(case) as problem:
        return problem.evaluate(case["parameters"], request)
