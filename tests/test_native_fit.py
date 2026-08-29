from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from scivol import (
    ARMA,
    AutoVol,
    EGARCH,
    GED,
    GARCH,
    GJRGARCH,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from scivol import _native_core
from scivol._evaluation import filter_spec
from scivol._distribution_moments import negative_half_second_moment
from scivol._gh import standardized_gh_rvs
from scivol._native_fit import (
    NativeModel,
    _aligned_workspace,
    _assess_theta_solution,
    _coordinate_bounds,
    _coordinates_to_theta,
    _describe_model,
    _evaluate_coordinate_request,
    _evaluate_theta_request,
    _gjr_moment,
    _initial_theta,
    _jet_workspace,
    _materialize_native_egarch_variance,
    _materialize_native_arma_egarch_state,
    _theta_to_coordinates,
)


DensityFactory = Callable[[], object]

_DENSITIES: tuple[tuple[str, DensityFactory], ...] = (
    ("normal", Normal),
    ("studentt", StudentT),
    ("skewt", SkewT),
    ("ged", GED),
    ("gh", GeneralizedHyperbolic),
    ("nig", NormalInverseGaussian),
    ("hyperbolic", Hyperbolic),
    ("vg", VarianceGamma),
)
_REQUESTS = tuple(range(1, 8))
_HALF_LOG_2PI = 0.9189385332046727


def _returns(seed: int = 20260827, n: int = 700) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 300
    innovations = generator.standard_t(7, size=n + burn) * np.sqrt(5.0 / 7.0)
    return _filter_garch_innovations(innovations, burn)


def _gh_returns(seed: int = 20260827, n: int = 1200) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 300
    innovations = standardized_gh_rvs(
        generator, n + burn, 0.15, 1.3, 0.4
    )
    return _filter_garch_innovations(innovations, burn)


def _filter_garch_innovations(
    innovations: np.ndarray, burn: int
) -> np.ndarray:
    residuals = np.empty(innovations.size, dtype=np.float64)
    variance = np.empty(innovations.size, dtype=np.float64)
    variance[0] = 2e-5 / (1.0 - 0.08 - 0.88)
    residuals[0] = np.sqrt(variance[0]) * innovations[0]
    for time_index in range(1, innovations.size):
        variance[time_index] = (
            2e-5
            + 0.08 * residuals[time_index - 1] ** 2
            + 0.88 * variance[time_index - 1]
        )
        residuals[time_index] = (
            np.sqrt(variance[time_index]) * innovations[time_index]
        )
    return np.ascontiguousarray(residuals[burn:], dtype=np.float64)


def _joint_returns(
    seed: int = 778,
    n: int = 1400,
    p_ar: int = 2,
    q_ma: int = 2,
    p_vol: int = 3,
    q_vol: int = 2,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 500
    total = n + burn
    innovations = generator.standard_t(7, total) * np.sqrt(5.0 / 7.0)
    ar = np.array([0.28, -0.12], dtype=np.float64)[:p_ar]
    ma = np.array([0.20, -0.08], dtype=np.float64)[:q_ma]
    alpha = np.full(p_vol, 0.08 / p_vol, dtype=np.float64)
    beta = np.full(q_vol, 0.85 / q_vol, dtype=np.float64)
    maximum_lag = max(p_ar, q_ma, p_vol, q_vol)
    variance = np.full(total, 2e-5 / 0.07, dtype=np.float64)
    residuals = np.zeros(total, dtype=np.float64)
    observations = np.zeros(total, dtype=np.float64)
    for time_index in range(maximum_lag, total):
        variance[time_index] = 2e-5
        for lag, coefficient in enumerate(alpha, start=1):
            variance[time_index] += coefficient * residuals[time_index - lag] ** 2
        for lag, coefficient in enumerate(beta, start=1):
            variance[time_index] += coefficient * variance[time_index - lag]
        residuals[time_index] = np.sqrt(variance[time_index]) * innovations[time_index]
        observations[time_index] = 0.001 + residuals[time_index]
        for lag, coefficient in enumerate(ar, start=1):
            observations[time_index] += coefficient * observations[time_index - lag]
        for lag, coefficient in enumerate(ma, start=1):
            observations[time_index] += coefficient * residuals[time_index - lag]
    return np.ascontiguousarray(observations[burn:], dtype=np.float64)


def _joint_gjr_returns(seed: int = 20260828, n: int = 700) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 300
    total = n + burn
    innovations = generator.standard_t(8.0, total) * np.sqrt(6.0 / 8.0)
    c, ar, ma = 0.001, 0.22, -0.12
    omega, alpha, gamma, beta = 2e-5, 0.055, 0.075, 0.83
    persistence = alpha + 0.5 * gamma + beta
    variance = np.empty(total, dtype=np.float64)
    residuals = np.empty(total, dtype=np.float64)
    observations = np.empty(total, dtype=np.float64)
    variance[0] = omega / (1.0 - persistence)
    residuals[0] = 0.0
    observations[0] = c
    for time_index in range(1, total):
        lagged_residual = residuals[time_index - 1]
        variance[time_index] = (
            omega
            + (alpha + gamma * (lagged_residual < 0.0))
            * lagged_residual**2
            + beta * variance[time_index - 1]
        )
        residuals[time_index] = (
            np.sqrt(variance[time_index]) * innovations[time_index]
        )
        observations[time_index] = (
            c
            + ar * observations[time_index - 1]
            + ma * lagged_residual
            + residuals[time_index]
        )
    return np.ascontiguousarray(observations[burn:], dtype=np.float64)


def _joint_egarch_returns(seed: int = 20260829, n: int = 700) -> np.ndarray:
    generator = np.random.default_rng(seed)
    burn = 400
    total = n + burn
    degrees_of_freedom = 7.0
    innovations = (
        generator.standard_t(degrees_of_freedom, total)
        * np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    )
    expected_absolute_innovation = (
        2.0
        * np.sqrt(degrees_of_freedom - 2.0)
        * math.gamma(0.5 * (degrees_of_freedom + 1.0))
        / (
            (degrees_of_freedom - 1.0)
            * np.sqrt(np.pi)
            * math.gamma(0.5 * degrees_of_freedom)
        )
    )
    intercept, ar, ma = 0.001, 0.22, -0.12
    omega, magnitude, asymmetry, persistence = -0.12, 0.16, -0.08, 0.88
    log_variance = np.empty(total, dtype=np.float64)
    residuals = np.empty(total, dtype=np.float64)
    observations = np.empty(total, dtype=np.float64)
    log_variance[0] = omega / (1.0 - persistence)
    residuals[0] = 0.0
    observations[0] = intercept
    for time_index in range(1, total):
        standardized_residual = residuals[time_index - 1] * np.exp(
            -0.5 * log_variance[time_index - 1]
        )
        log_variance[time_index] = (
            omega
            + magnitude
            * (abs(standardized_residual) - expected_absolute_innovation)
            + asymmetry * standardized_residual
            + persistence * log_variance[time_index - 1]
        )
        residuals[time_index] = (
            np.exp(0.5 * log_variance[time_index]) * innovations[time_index]
        )
        observations[time_index] = (
            intercept
            + ar * observations[time_index - 1]
            + ma * residuals[time_index - 1]
            + residuals[time_index]
        )
    return np.ascontiguousarray(observations[burn:], dtype=np.float64)


def _model_and_parameters(
    kind: str,
    density_factory: DensityFactory,
) -> tuple[NativeModel, np.ndarray, np.ndarray]:
    data = _returns(n=48)
    volatility = GJRGARCH(2, 1) if "gjr" in kind else GARCH(2, 1)
    spec = (
        ARMA(2, 1) + volatility + density_factory()
        if kind.startswith("arma_")
        else volatility + density_factory()
    )
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)
    if kind.startswith("arma_"):
        theta[:4] = np.array([0.002, 0.15, -0.05, -0.10])
    return model, data, theta


def _fixed_joint_model_and_parameters(
    density_factory: DensityFactory,
    *,
    leverage: bool = False,
) -> tuple[NativeModel, np.ndarray, np.ndarray]:
    data = _returns(n=48)
    volatility = GJRGARCH(1, 1) if leverage else GARCH(1, 1)
    spec = ARMA(1, 1) + volatility + density_factory()
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)
    theta[:3] = np.array([0.002, 0.15, -0.05])
    return model, data, theta


def _egarch_model_and_parameters(
    density_factory: DensityFactory,
    *,
    p: int = 2,
    q: int = 1,
) -> tuple[NativeModel, np.ndarray, np.ndarray]:
    data = _returns(n=48)
    spec = EGARCH(p, q) + density_factory()
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)

    theta[0] = -0.08
    theta[1 : 1 + p] = np.linspace(0.08, 0.04, p)
    theta[1 + p : 1 + 2 * p] = np.linspace(-0.06, 0.03, p)
    beta = {
        1: np.array([0.82]),
        2: np.array([0.45, 0.18]),
        3: np.array([0.35, 0.15, 0.05]),
    }[q]
    theta[1 + 2 * p : 1 + 2 * p + q] = beta

    shape = {
        "Normal": (),
        "StudentT": (7.5,),
        "SkewT": (8.0, 0.12),
        "GED": (1.6,),
        "GeneralizedHyperbolic": (0.12, 1.8, 0.4),
        "NormalInverseGaussian": (0.12, 1.8),
        "Hyperbolic": (0.12, 1.8),
        "VarianceGamma": (0.12, 2.0),
    }[density_factory().signature]
    if shape:
        theta[-len(shape) :] = shape
    return model, data, theta


def _arma_egarch_model_and_parameters(
    density_factory: DensityFactory,
    *,
    p_ar: int = 2,
    q_ma: int = 1,
    p_vol: int = 2,
    q_vol: int = 1,
) -> tuple[NativeModel, np.ndarray, np.ndarray]:
    model, data, theta = _egarch_model_and_parameters(
        density_factory, p=p_vol, q=q_vol
    )
    spec = ARMA(p_ar, q_ma) + EGARCH(p_vol, q_vol) + density_factory()
    model = _describe_model(spec)
    native_theta, _ = _initial_theta(spec, model, data, None)
    native_theta[0] = 0.002
    native_theta[1 : 1 + p_ar] = np.linspace(0.14, -0.05, p_ar)
    native_theta[1 + p_ar : model.n_mean] = np.linspace(-0.08, -0.08, q_ma)
    native_theta[model.n_mean :] = theta
    return model, data, native_theta


def _evaluate_mask(
    model: NativeModel,
    data: np.ndarray,
    parameters: np.ndarray,
    request: int,
    *,
    transformed: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    value = np.full(1, np.nan, dtype=np.float64)
    gradient = np.full(model.n_params, np.nan, dtype=np.float64)
    hessian = np.full((model.n_params, model.n_params), np.nan, dtype=np.float64)
    workspace = _jet_workspace(model, transformed=transformed)
    initial_variance = float(np.mean(data * data))

    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        status, valid = _native_core._garch_jet(
            parameters.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            initial_variance,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            model.leverage,
        )
    elif model.kind == "egarch":
        status, valid = _native_core._egarch_jet(
            parameters.ctypes.data,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    elif model.kind == "arma_egarch":
        status, valid = _native_core._arma_egarch_jet(
            parameters.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        status, valid = _native_core._arma_garch_jet(
            parameters.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            model.leverage,
        )

    assert status == 0
    assert valid == request
    return float(value[0]), gradient, hessian


@lru_cache(maxsize=1)
def _runtime_library() -> ctypes.CDLL:
    library = ctypes.CDLL(_native_core.__file__)
    size = ctypes.c_size_t
    unsigned = ctypes.c_uint32
    pointer = ctypes.c_void_p

    library.vlk_runtime_garch_sample_jet_workspace_bytes.argtypes = [
        size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_garch_sample_jet_workspace_bytes.restype = size
    library.vlk_runtime_garch_sample_jet_evaluate.argtypes = [
        pointer, pointer, pointer, size, ctypes.c_double, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_garch_sample_jet_evaluate.restype = ctypes.c_int32
    library.vlk_runtime_gjr_garch_sample_jet_workspace_bytes.argtypes = [
        size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_gjr_garch_sample_jet_workspace_bytes.restype = size
    library.vlk_runtime_gjr_garch_sample_jet_evaluate.argtypes = (
        library.vlk_runtime_garch_sample_jet_evaluate.argtypes
    )
    library.vlk_runtime_gjr_garch_sample_jet_evaluate.restype = ctypes.c_int32

    library.vlk_runtime_egarch_jet_workspace_bytes.argtypes = [
        size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_egarch_jet_workspace_bytes.restype = size
    library.vlk_runtime_egarch_jet_evaluate.argtypes = [
        pointer, pointer, size, ctypes.c_double, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_egarch_jet_evaluate.restype = ctypes.c_int32

    library.vlk_runtime_arma_garch_jet_workspace_bytes.argtypes = [
        size, size, size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_arma_garch_jet_workspace_bytes.restype = size
    library.vlk_runtime_arma_garch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer, pointer,
        ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_arma_garch_jet_evaluate.restype = ctypes.c_int32
    library.vlk_runtime_arma_gjr_garch_jet_workspace_bytes.argtypes = [
        size, size, size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_arma_gjr_garch_jet_workspace_bytes.restype = size
    library.vlk_runtime_arma_gjr_garch_jet_evaluate.argtypes = (
        library.vlk_runtime_arma_garch_jet_evaluate.argtypes
    )
    library.vlk_runtime_arma_gjr_garch_jet_evaluate.restype = ctypes.c_int32
    library.vlk_runtime_arma_egarch_jet_workspace_bytes.argtypes = [
        size, size, size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_arma_egarch_jet_workspace_bytes.restype = size
    library.vlk_runtime_arma_egarch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer, pointer,
        ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_arma_egarch_jet_evaluate.restype = ctypes.c_int32
    return library


def _evaluate_forced_runtime(
    model: NativeModel,
    data: np.ndarray,
    parameters: np.ndarray,
    request: int,
    *,
    transformed: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    library = _runtime_library()
    value = np.full(1, np.nan, dtype=np.float64)
    gradient = np.full(model.n_params, np.nan, dtype=np.float64)
    hessian = np.full((model.n_params, model.n_params), np.nan, dtype=np.float64)
    valid = ctypes.c_uint32(0)
    initial_variance = float(np.mean(data * data))

    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        workspace_planner = (
            library.vlk_runtime_gjr_garch_sample_jet_workspace_bytes
            if model.leverage
            else library.vlk_runtime_garch_sample_jet_workspace_bytes
        )
        evaluator = (
            library.vlk_runtime_gjr_garch_sample_jet_evaluate
            if model.leverage
            else library.vlk_runtime_garch_sample_jet_evaluate
        )
        workspace_bytes = int(
            workspace_planner(
                model.p_vol,
                model.q_vol,
                model.density_code,
                int(transformed),
                request,
            )
        )
        workspace = _aligned_workspace(workspace_bytes)
        status = evaluator(
            parameters.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            initial_variance,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            ctypes.byref(valid),
        )
    elif model.kind == "egarch":
        workspace_bytes = int(
            library.vlk_runtime_egarch_jet_workspace_bytes(
                model.p_vol,
                model.q_vol,
                model.density_code,
                int(transformed),
                request,
            )
        )
        workspace = _aligned_workspace(workspace_bytes)
        status = library.vlk_runtime_egarch_jet_evaluate(
            parameters.ctypes.data,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            ctypes.byref(valid),
        )
    elif model.kind == "arma_egarch":
        workspace_bytes = int(
            library.vlk_runtime_arma_egarch_jet_workspace_bytes(
                model.p_ar,
                model.q_ma,
                model.p_vol,
                model.q_vol,
                model.density_code,
                int(transformed),
                request,
            )
        )
        workspace = _aligned_workspace(workspace_bytes)
        status = library.vlk_runtime_arma_egarch_jet_evaluate(
            parameters.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            ctypes.byref(valid),
        )
    else:
        workspace_planner = (
            library.vlk_runtime_arma_gjr_garch_jet_workspace_bytes
            if model.leverage
            else library.vlk_runtime_arma_garch_jet_workspace_bytes
        )
        evaluator = (
            library.vlk_runtime_arma_gjr_garch_jet_evaluate
            if model.leverage
            else library.vlk_runtime_arma_garch_jet_evaluate
        )
        workspace_bytes = int(
            workspace_planner(
                model.p_ar,
                model.q_ma,
                model.p_vol,
                model.q_vol,
                model.density_code,
                int(transformed),
                request,
            )
        )
        workspace = _aligned_workspace(workspace_bytes)
        status = evaluator(
            parameters.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            initial_variance,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            ctypes.byref(valid),
        )

    assert status == 0
    assert valid.value == request
    return float(value[0]), gradient, hessian


def _checked_joint_status(
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
    request: int,
) -> tuple[int, int]:
    value = np.zeros(1, dtype=np.float64)
    gradient = np.zeros(model.n_params, dtype=np.float64)
    hessian = np.zeros((model.n_params, model.n_params), dtype=np.float64)
    workspace = _jet_workspace(model, transformed=False)
    return _native_core._arma_garch_jet(
        theta.ctypes.data,
        model.n_params,
        data.ctypes.data,
        data.size,
        float(np.mean(data * data)),
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
        0,
        request,
        workspace.address,
        workspace.size,
        value.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
        model.leverage,
    )


def _checked_arma_egarch_status(
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
    request: int,
) -> tuple[int, int]:
    value = np.zeros(1, dtype=np.float64)
    gradient = np.zeros(model.n_params, dtype=np.float64)
    hessian = np.zeros((model.n_params, model.n_params), dtype=np.float64)
    workspace = _jet_workspace(model, transformed=False)
    return _native_core._arma_egarch_jet(
        theta.ctypes.data,
        model.n_params,
        data.ctypes.data,
        data.size,
        float(np.mean(data * data)),
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
        0,
        request,
        workspace.address,
        workspace.size,
        value.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
    )


def _checked_garch_status(
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
    request: int,
) -> tuple[int, int]:
    value = np.zeros(1, dtype=np.float64)
    gradient = np.zeros(model.n_params, dtype=np.float64)
    hessian = np.zeros((model.n_params, model.n_params), dtype=np.float64)
    squared = np.ascontiguousarray(data * data, dtype=np.float64)
    workspace = _jet_workspace(model, transformed=False)
    return _native_core._garch_jet(
        theta.ctypes.data,
        data.ctypes.data,
        squared.ctypes.data,
        data.size,
        float(np.mean(squared)),
        model.p_vol,
        model.q_vol,
        model.density_code,
        0,
        request,
        workspace.address,
        workspace.size,
        value.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
        model.leverage,
    )


def _independent_theta_jet(
    kind: str,
    density_name: str,
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    from scivol._devtools.ad_oracle import (
        arma_gjr_garch_value_grad_hess,
        arma_garch_value_grad_hess,
        garch_value_grad_hess,
        gjr_garch_value_grad_hess,
        gh_surface_value_grad_hess,
        vg_surface_value_grad_hess,
    )

    if density_name in {"normal", "studentt", "skewt", "ged"}:
        if not kind.startswith("arma_"):
            if model.leverage:
                expected = gjr_garch_value_grad_hess(
                    theta, data, model.p_vol, model.q_vol, dist=density_name
                )
            else:
                expected = garch_value_grad_hess(
                    theta,
                    data * data,
                    model.p_vol,
                    model.q_vol,
                    dist=density_name,
                    resid=data,
                )
            if density_name == "normal":
                expected = (
                    expected[0] + data.size * _HALF_LOG_2PI,
                    expected[1],
                    expected[2],
                )
            return expected

        oracle = (
            arma_gjr_garch_value_grad_hess
            if model.leverage
            else arma_garch_value_grad_hess
        )
        expected = oracle(
            theta, data, model.p_ar, model.q_ma, model.p_vol, model.q_vol,
            dist=density_name,
        )
        if density_name == "normal":
            expected = (
                expected[0] + _HALF_LOG_2PI,
                expected[1],
                expected[2],
            )
    else:
        oracle = (
            vg_surface_value_grad_hess
            if density_name == "vg"
            else gh_surface_value_grad_hess
        )
        fixed_lambda = {
            "gh": None,
            "nig": -0.5,
            "hyperbolic": 1.0,
        }.get(density_name)
        kwargs = {
            "mean_kind": "none" if not kind.startswith("arma_") else "arma",
            "p_mean": model.p_ar,
            "q_mean": model.q_ma,
            "volatility_kind": "gjr" if model.leverage else "garch",
            "p_vol": model.p_vol,
            "q_vol": model.q_vol,
        }
        if density_name != "vg":
            kwargs["fixed_lambda"] = fixed_lambda
        expected = oracle(theta, data, **kwargs)

    if not kind.startswith("arma_"):
        return expected
    n_eff = data.size - model.maximum_lag
    return tuple(n_eff * quantity for quantity in expected)


def _independent_egarch_theta_jet(
    density_name: str,
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    from scivol._devtools.ad_oracle import (
        egarch_value_grad_hess,
        gh_surface_value_grad_hess,
        vg_surface_value_grad_hess,
    )

    if density_name in {"normal", "studentt", "skewt", "ged"}:
        expected = egarch_value_grad_hess(
            theta,
            data,
            dist=density_name,
            p=model.p_vol,
            q=model.q_vol,
        )
        if density_name == "normal":
            return (
                expected[0] + data.size * _HALF_LOG_2PI,
                expected[1],
                expected[2],
            )
        return expected

    oracle = (
        vg_surface_value_grad_hess
        if density_name == "vg"
        else gh_surface_value_grad_hess
    )
    kwargs = {
        "volatility_kind": "egarch",
        "p_vol": model.p_vol,
        "q_vol": model.q_vol,
    }
    if density_name != "vg":
        kwargs["fixed_lambda"] = {
            "gh": None,
            "nig": -0.5,
            "hyperbolic": 1.0,
        }[density_name]
    return oracle(theta, data, **kwargs)


def _independent_arma_egarch_theta_jet(
    density_name: str,
    model: NativeModel,
    data: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    from scivol._devtools.ad_oracle import (
        arma_egarch_value_grad_hess,
        gh_surface_value_grad_hess,
        vg_surface_value_grad_hess,
    )

    if density_name in {"normal", "studentt", "skewt", "ged"}:
        expected = arma_egarch_value_grad_hess(
            theta,
            data,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            density_name,
        )
        if density_name == "normal":
            expected = (
                expected[0] + _HALF_LOG_2PI,
                expected[1],
                expected[2],
            )
    else:
        oracle = (
            vg_surface_value_grad_hess
            if density_name == "vg"
            else gh_surface_value_grad_hess
        )
        kwargs = {
            "mean_kind": "arma",
            "p_mean": model.p_ar,
            "q_mean": model.q_ma,
            "volatility_kind": "egarch",
            "p_vol": model.p_vol,
            "q_vol": model.q_vol,
        }
        if density_name != "vg":
            kwargs["fixed_lambda"] = {
                "gh": None,
                "nig": -0.5,
                "hyperbolic": 1.0,
            }[density_name]
        expected = oracle(theta, data, **kwargs)

    n_eff = data.size - model.maximum_lag
    return tuple(n_eff * quantity for quantity in expected)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("transformed", (False, True))
def test_arma_egarch_runtime_sparse_requests_match_the_full_jet(
    density_name: str,
    density_factory: DensityFactory,
    transformed: bool,
) -> None:
    model, data, theta = _arma_egarch_model_and_parameters(density_factory)
    parameters = _theta_to_coordinates(model, theta) if transformed else theta
    full = _evaluate_mask(
        model, data, parameters, 7, transformed=transformed
    )

    for request in _REQUESTS:
        sparse = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert sparse[0] == pytest.approx(
                full[0], rel=2e-12, abs=2e-11
            )
        if request & 2:
            assert_allclose(sparse[1], full[1], rtol=2e-11, atol=2e-9)
        if request & 4:
            assert_allclose(sparse[2], full[2], rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("transformed", (False, True))
def test_arma_egarch_generated_1111_matches_forced_runtime(
    density_name: str,
    density_factory: DensityFactory,
    transformed: bool,
) -> None:
    model, data, theta = _arma_egarch_model_and_parameters(
        density_factory, p_ar=1, q_ma=1, p_vol=1, q_vol=1
    )
    parameters = _theta_to_coordinates(model, theta) if transformed else theta

    for request in _REQUESTS:
        fixed = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        runtime = _evaluate_forced_runtime(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert runtime[0] == pytest.approx(
                fixed[0], rel=2e-12, abs=2e-11
            )
        if request & 2:
            assert_allclose(runtime[1], fixed[1], rtol=2e-11, atol=2e-9)
        if request & 4:
            assert_allclose(runtime[2], fixed[2], rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_arma_egarch_theta_and_transformed_jets_match_independent_ad(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import (
        native_arma_egarch_coordinate_chain_rule_value_grad_hess,
    )

    model, data, theta = _arma_egarch_model_and_parameters(density_factory)
    theta_jet = _evaluate_mask(model, data, theta, 7, transformed=False)
    expected_theta = _independent_arma_egarch_theta_jet(
        density_name, model, data, theta
    )
    assert theta_jet[0] == pytest.approx(
        expected_theta[0], rel=3e-6, abs=3e-6
    )
    assert_allclose(theta_jet[1], expected_theta[1], rtol=3e-4, atol=3e-5)
    assert_allclose(theta_jet[2], expected_theta[2], rtol=3e-3, atol=3e-2)

    cross = np.ix_(
        np.arange(model.n_mean),
        np.arange(model.n_mean, model.n_params),
    )
    assert np.max(np.abs(theta_jet[2][cross])) > 1e-6
    assert_allclose(
        theta_jet[2][cross],
        expected_theta[2][cross],
        rtol=3e-3,
        atol=3e-2,
    )

    coordinates = _theta_to_coordinates(model, theta)
    transformed = _evaluate_mask(
        model, data, coordinates, 7, transformed=True
    )
    expected_transformed = (
        native_arma_egarch_coordinate_chain_rule_value_grad_hess(
            coordinates,
            theta,
            *theta_jet,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            density_name,
        )
    )
    assert transformed[0] == pytest.approx(
        expected_transformed[0], rel=3e-10, abs=3e-9
    )
    assert_allclose(
        transformed[1], expected_transformed[1], rtol=3e-8, atol=3e-8
    )
    assert_allclose(
        transformed[2], expected_transformed[2], rtol=5e-7, atol=5e-7
    )


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_arma_egarch_public_filter_uses_the_canonical_joint_state(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    model, data, theta = _arma_egarch_model_and_parameters(density_factory)
    spec = (
        ARMA(model.p_ar, model.q_ma)
        + EGARCH(model.p_vol, model.q_vol)
        + density_factory()
    )
    residuals, variance = _materialize_native_arma_egarch_state(
        model, data, theta
    )
    state = filter_spec(spec, data, theta)
    nll = _evaluate_mask(model, data, theta, 1, transformed=False)[0]

    assert state.log_likelihood == pytest.approx(-nll, rel=2e-12, abs=2e-11)
    assert_allclose(state.residuals, residuals, rtol=0.0, atol=0.0)
    assert_allclose(state.sigma2, variance, rtol=0.0, atol=0.0)
    assert_allclose(state.mean, data - residuals, rtol=0.0, atol=0.0)


def test_arma_egarch_coordinate_map_round_trips_arbitrary_orders() -> None:
    model, _, theta = _arma_egarch_model_and_parameters(
        StudentT, p_ar=3, q_ma=2, p_vol=2, q_vol=3
    )
    recovered = _coordinates_to_theta(
        model, _theta_to_coordinates(model, theta)
    )

    assert_allclose(recovered, theta, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize(
    "parameter_offset",
    (
        lambda model: 1,
        lambda model: 1 + model.p_ar,
        lambda model: model.n_mean + 1 + 2 * model.p_vol,
    ),
    ids=("nonstationary-ar", "noninvertible-ma", "nonstationary-beta"),
)
def test_checked_arma_egarch_rejects_unstable_recursions(
    parameter_offset: Callable[[NativeModel], int],
) -> None:
    model, data, theta = _arma_egarch_model_and_parameters(
        Normal, p_ar=1, q_ma=1, p_vol=1, q_vol=1
    )
    theta[parameter_offset(model)] = 1.0

    for request in _REQUESTS:
        assert _checked_arma_egarch_status(
            model, data, theta, request
        ) == (3, 0)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
@pytest.mark.parametrize("transformed", (False, True))
def test_every_native_request_matches_the_full_jet(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
    transformed: bool,
) -> None:
    model, data, theta = _model_and_parameters(kind, density_factory)
    parameters = _theta_to_coordinates(model, theta) if transformed else theta
    full_value, full_gradient, full_hessian = _evaluate_mask(
        model, data, parameters, 7, transformed=transformed
    )

    for request in _REQUESTS:
        value, gradient, hessian = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert value == pytest.approx(full_value, rel=2e-12, abs=2e-12)
        if request & 2:
            assert_allclose(gradient, full_gradient, rtol=2e-11, atol=2e-11)
        if request & 4:
            assert_allclose(hessian, full_hessian, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("leverage", (False, True), ids=("garch", "gjr"))
@pytest.mark.parametrize("transformed", (False, True))
def test_fixed_joint_requests_match_the_full_jet(
    density_name: str,
    density_factory: DensityFactory,
    leverage: bool,
    transformed: bool,
) -> None:
    model, data, theta = _fixed_joint_model_and_parameters(
        density_factory, leverage=leverage
    )
    parameters = _theta_to_coordinates(model, theta) if transformed else theta
    full_value, full_gradient, full_hessian = _evaluate_mask(
        model, data, parameters, 7, transformed=transformed
    )

    for request in _REQUESTS:
        value, gradient, hessian = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert value == pytest.approx(full_value, rel=2e-12, abs=2e-12)
        if request & 2:
            assert_allclose(gradient, full_gradient, rtol=2e-11, atol=2e-11)
        if request & 4:
            assert_allclose(hessian, full_hessian, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
@pytest.mark.parametrize("transformed", (False, True))
def test_fixed_and_forced_runtime_jets_match_at_the_same_point(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
    transformed: bool,
) -> None:
    if not kind.startswith("arma_"):
        model, data, theta = _model_and_parameters(kind, density_factory)
    else:
        model, data, theta = _fixed_joint_model_and_parameters(
            density_factory, leverage="gjr" in kind
        )
    parameters = _theta_to_coordinates(model, theta) if transformed else theta

    for request in _REQUESTS:
        fixed = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        runtime = _evaluate_forced_runtime(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert runtime[0] == pytest.approx(
                fixed[0], rel=2e-12, abs=2e-12
            )
        if request & 2:
            assert_allclose(runtime[1], fixed[1], rtol=2e-11, atol=2e-11)
        if request & 4:
            assert_allclose(runtime[2], fixed[2], rtol=2e-10, atol=2e-10)


def test_native_fixed_dispatch_contract() -> None:
    for density_code, _ in enumerate(_DENSITIES):
        for transformed in (0, 1):
            for p, q in ((1, 1), (1, 2), (2, 1), (2, 2)):
                assert _native_core._garch_dispatch_kind(
                    p, q, density_code, transformed
                ) == 1
            assert _native_core._garch_dispatch_kind(
                3, 2, density_code, transformed
            ) == 0
            for p, q in ((1, 1), (1, 2), (2, 1), (2, 2)):
                assert _native_core._egarch_dispatch_kind(
                    p, q, density_code, transformed
                ) == 1
            assert _native_core._egarch_dispatch_kind(
                3, 2, density_code, transformed
            ) == 2

        assert _native_core._arma_garch_dispatch_kind(
            1, 1, 1, 1, density_code
        ) == 1
        assert _native_core._arma_garch_dispatch_kind(
            1, 1, 2, 1, density_code
        ) == 0
        assert _native_core._arma_garch_workspace_bytes(
            1, 1, 1, 1, density_code, 0, 7
        ) == 1
        assert _native_core._arma_garch_dispatch_kind(
            1, 1, 1, 1, density_code, True
        ) == 1
        for transformed in (0, 1):
            assert _native_core._arma_egarch_dispatch_kind(
                1, 1, 1, 1, density_code, transformed
            ) == 1
            assert _native_core._arma_egarch_dispatch_kind(
                2, 1, 1, 1, density_code, transformed
            ) == 0

    assert _native_core._garch_dispatch_kind(1, 1, 99, 0) == -1
    assert _native_core._egarch_dispatch_kind(1, 1, 99, 0) == 0
    assert _native_core._arma_garch_dispatch_kind(1, 1, 1, 1, 99) == -1
    assert _native_core._arma_egarch_dispatch_kind(1, 1, 1, 1, 99, 0) == -1


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("transformed", (False, True))
def test_egarch_sparse_requests_and_fixed_runtime_paths_match(
    density_name: str,
    density_factory: DensityFactory,
    transformed: bool,
) -> None:
    model, data, theta = _egarch_model_and_parameters(
        density_factory, p=1, q=2
    )
    parameters = _theta_to_coordinates(model, theta) if transformed else theta
    full = _evaluate_mask(
        model, data, parameters, 7, transformed=transformed
    )
    runtime = _evaluate_forced_runtime(
        model, data, parameters, 7, transformed=transformed
    )

    assert runtime[0] == pytest.approx(full[0], rel=2e-12, abs=2e-11)
    assert_allclose(runtime[1], full[1], rtol=2e-11, atol=2e-9)
    assert_allclose(runtime[2], full[2], rtol=2e-10, atol=2e-8)
    for request in _REQUESTS:
        sparse = _evaluate_mask(
            model, data, parameters, request, transformed=transformed
        )
        if request & 1:
            assert sparse[0] == pytest.approx(
                full[0], rel=2e-12, abs=2e-11
            )
        if request & 2:
            assert_allclose(sparse[1], full[1], rtol=2e-11, atol=2e-9)
        if request & 4:
            assert_allclose(sparse[2], full[2], rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_egarch_theta_and_transformed_jets_match_independent_ad(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import (
        native_egarch_coordinate_chain_rule_value_grad_hess,
    )

    model, data, theta = _egarch_model_and_parameters(
        density_factory, p=2, q=2
    )
    theta_jet = _evaluate_mask(model, data, theta, 7, transformed=False)
    expected_theta = _independent_egarch_theta_jet(
        density_name, model, data, theta
    )
    assert theta_jet[0] == pytest.approx(
        expected_theta[0], rel=2e-7, abs=2e-7
    )
    assert_allclose(theta_jet[1], expected_theta[1], rtol=3e-5, atol=3e-5)
    assert_allclose(theta_jet[2], expected_theta[2], rtol=2e-4, atol=2e-4)

    coordinates = _theta_to_coordinates(model, theta)
    transformed_jet = _evaluate_mask(
        model, data, coordinates, 7, transformed=True
    )
    expected_transformed = native_egarch_coordinate_chain_rule_value_grad_hess(
        coordinates,
        theta,
        *theta_jet,
        model.p_vol,
        model.q_vol,
        density_name,
    )
    assert transformed_jet[0] == pytest.approx(
        expected_transformed[0], rel=3e-10, abs=3e-9
    )
    assert_allclose(
        transformed_jet[1], expected_transformed[1], rtol=3e-8, atol=3e-8
    )
    assert_allclose(
        transformed_jet[2], expected_transformed[2], rtol=5e-7, atol=5e-7
    )


@pytest.mark.parametrize(
    "density_name,density_factory,p,q",
    (
        ("normal", Normal, 3, 2),
        ("studentt", StudentT, 2, 3),
    ),
)
def test_egarch_runtime_orders_beyond_precompiled_cases_match_ad(
    density_name: str,
    density_factory: DensityFactory,
    p: int,
    q: int,
) -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import egarch_logspace_value_grad_hess

    model, data, theta = _egarch_model_and_parameters(
        density_factory, p=p, q=q
    )
    theta_jet = _evaluate_mask(model, data, theta, 7, transformed=False)
    expected_theta = _independent_egarch_theta_jet(
        density_name, model, data, theta
    )
    assert theta_jet[0] == pytest.approx(
        expected_theta[0], rel=2e-9, abs=2e-8
    )
    assert_allclose(theta_jet[1], expected_theta[1], rtol=2e-6, atol=2e-6)
    assert_allclose(theta_jet[2], expected_theta[2], rtol=3e-5, atol=3e-5)

    coordinates = _theta_to_coordinates(model, theta)
    transformed = _evaluate_mask(
        model, data, coordinates, 7, transformed=True
    )
    expected_transformed = egarch_logspace_value_grad_hess(
        coordinates, data, dist=density_name, p=p, q=q
    )
    expected_value = expected_transformed[0]
    if density_name == "normal":
        expected_value += data.size * _HALF_LOG_2PI
    assert transformed[0] == pytest.approx(expected_value, rel=2e-9, abs=2e-8)
    assert_allclose(
        transformed[1], expected_transformed[1], rtol=2e-6, atol=2e-6
    )
    assert_allclose(
        transformed[2], expected_transformed[2], rtol=3e-5, atol=3e-5
    )


def test_egarch_coordinate_map_round_trips_general_beta_order() -> None:
    model, _, theta = _egarch_model_and_parameters(StudentT, p=2, q=3)
    recovered = _coordinates_to_theta(model, _theta_to_coordinates(model, theta))

    assert_allclose(recovered, theta, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_public_gjr_moment_matches_the_native_domain_contract(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    model, _, theta = _model_and_parameters("gjr", density_factory)
    shape = theta[model.n_vol:].copy()
    if density_name in {"gh", "nig", "hyperbolic", "vg"}:
        shape[0] = 0.3
        theta[model.n_vol:] = shape

    if density_name == "normal":
        arguments = ()
    elif density_name in {"studentt", "ged"}:
        arguments = (shape[0],)
    elif density_name == "skewt":
        arguments = (shape[0], shape[1])
    elif density_name == "gh":
        arguments = (None, None, shape[0], shape[1], shape[2])
    elif density_name == "nig":
        arguments = (None, None, shape[0], shape[1], -0.5)
    elif density_name == "hyperbolic":
        arguments = (None, None, shape[0], shape[1], 1.0)
    else:
        arguments = (None, None, shape[0], None, shape[1])

    public_moment = negative_half_second_moment(
        density_factory().signature, *arguments
    )
    native_moment, _ = _gjr_moment(model, theta, gradient=False)
    assert public_moment == pytest.approx(native_moment, rel=3e-10, abs=3e-11)


def test_gjr_result_persistence_uses_the_negative_half_second_moment() -> None:
    data = _returns(n=80)
    theta = np.array([2e-5, 0.08, 0.12, 0.75, 7.0, 0.6])
    result = (GJRGARCH(1, 1) + SkewT()).fix(data, theta)
    expected_moment = negative_half_second_moment("SkewT", 7.0, 0.6)

    assert result.garch_params is not None
    assert result.garch_params.negative_second_moment == pytest.approx(
        expected_moment
    )
    assert result.garch_params.persistence == pytest.approx(
        0.08 + expected_moment * 0.12 + 0.75
    )


def test_checked_gjr_accepts_negative_gamma_when_alpha_plus_gamma_is_positive() -> None:
    model, data, theta = _model_and_parameters("gjr", Normal)
    alpha_start = 1
    gamma_start = alpha_start + model.p_vol
    theta[alpha_start:gamma_start] = 0.08
    theta[gamma_start:gamma_start + model.p_vol] = -0.03

    for request in _REQUESTS:
        assert _checked_garch_status(model, data, theta, request) == (0, request)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_joint_gjr_negative_gamma_matches_the_established_evaluator(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    model, data, theta = _model_and_parameters("arma_gjr", density_factory)
    alpha_start = model.n_mean + 1
    gamma_start = alpha_start + model.p_vol
    theta[alpha_start:gamma_start] = 0.08
    theta[gamma_start:gamma_start + model.p_vol] = -0.03
    native_value = _evaluate_mask(
        model, data, theta, 1, transformed=False
    )[0]
    spec = ARMA(model.p_ar, model.q_ma) + GJRGARCH(
        model.p_vol, model.q_vol
    ) + density_factory()
    established_value = -filter_spec(spec, data, theta).log_likelihood

    assert native_value == pytest.approx(
        established_value, rel=3e-10, abs=3e-9
    )


def test_checked_gjr_rejects_negative_shock_coefficients() -> None:
    model, data, theta = _model_and_parameters("gjr", Normal)
    alpha_start = 1
    gamma_start = alpha_start + model.p_vol
    theta[alpha_start] = 0.02
    theta[gamma_start] = -0.03

    for request in _REQUESTS:
        assert _checked_garch_status(model, data, theta, request) == (3, 0)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_checked_garch_jet_rejects_each_density_domain_boundary(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    model, data, theta = _model_and_parameters("garch", density_factory)
    if density_name == "normal":
        theta[0] = 0.0
    elif density_name == "studentt":
        theta[-1] = 2.0
    elif density_name == "skewt":
        theta[-1] = 1.0
    elif density_name == "ged":
        theta[-1] = 0.0
    elif density_name in {"gh", "nig", "hyperbolic", "vg"}:
        theta[-3 if density_name == "gh" else -2] = 0.995

    for request in _REQUESTS:
        status, valid = _checked_garch_status(model, data, theta, request)
        assert status == 3
        assert valid == 0


@pytest.mark.parametrize(
    "orders,parameter_index",
    (
        ((1, 1), 1),
        ((1, 1), 2),
        ((2, 1), 1),
    ),
    ids=("fixed-ar", "fixed-ma", "runtime-ar"),
)
def test_checked_joint_jet_rejects_nonstationary_or_noninvertible_mean(
    orders: tuple[int, int],
    parameter_index: int,
) -> None:
    p_ar, q_ma = orders
    data = _returns(n=48)
    spec = ARMA(p_ar, q_ma) + GARCH(1, 1) + Normal()
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)
    theta[parameter_index] = 1.0 if p_ar == 1 else 1.5

    for request in _REQUESTS:
        status, valid = _checked_joint_status(model, data, theta, request)
        assert status == 3
        assert valid == 0


@pytest.mark.parametrize(
    "shape,valid_requests",
    ((0.8, {1}), (1.5, {1, 2, 3})),
    ids=("nondifferentiable-first", "nondifferentiable-second"),
)
def test_joint_ged_zero_residual_has_request_specific_validity(
    shape: float,
    valid_requests: set[int],
) -> None:
    data = np.array([0.1, 0.0, 0.2, -0.1, 0.15], dtype=np.float64)
    spec = ARMA(1, 1) + GARCH(1, 1) + GED()
    model = _describe_model(spec)
    theta = np.array([0.0, 0.0, 0.0, 0.01, 0.05, 0.8, shape])

    for request in _REQUESTS:
        status, valid = _checked_joint_status(model, data, theta, request)
        if request in valid_requests:
            assert status == 0
            assert valid == request
        else:
            assert status == 3
            assert valid == 0


def test_joint_arma_sample_plan_does_not_apply_hold_back_twice() -> None:
    data = np.array([0.4, -0.2, 0.1, 0.7, -0.3], dtype=np.float64)
    theta = np.array([0.03, 0.2, -0.1, 0.08, 0.07, 0.85], dtype=np.float64)
    spec = ARMA(1, 1) + GARCH(1, 1) + Normal()

    residual = np.zeros(data.size, dtype=np.float64)
    variance = np.empty(data.size, dtype=np.float64)
    variance[0] = np.mean(data * data)
    for time in range(1, data.size):
        residual[time] = (
            data[time] - theta[0] - theta[1] * data[time - 1]
            - theta[2] * residual[time - 1]
        )
        variance[time] = (
            theta[3] + theta[4] * residual[time - 1] ** 2
            + theta[5] * variance[time - 1]
        )
    expected = -0.5 * np.sum(
        np.log(2.0 * np.pi * variance[1:])
        + residual[1:] ** 2 / variance[1:]
    )

    np.testing.assert_allclose(spec.loglikelihood(data, theta), expected)
    result = spec.fit(data, solver="native", initial_params=theta, maxiter=1)
    assert result.fit_info.effective_hold_back == 0
    assert result.fit_info.effective_n_obs == data.size


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
def test_native_theta_jet_matches_independent_ad(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
) -> None:
    pytest.importorskip("jax")

    model, data, theta = _model_and_parameters(kind, density_factory)
    if density_name == "ged":
        theta[-1] = 0.8
    value, gradient, hessian = _evaluate_mask(
        model, data, theta, 7, transformed=False
    )
    expected_value, expected_gradient, expected_hessian = _independent_theta_jet(
        kind, density_name, model, data, theta
    )

    assert value == pytest.approx(expected_value, rel=3e-9, abs=3e-8)
    assert_allclose(gradient, expected_gradient, rtol=3e-7, atol=3e-7)
    assert_allclose(hessian, expected_hessian, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
def test_native_transformed_jet_matches_independent_chain_rule_ad(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
) -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import (
        native_coordinate_chain_rule_value_grad_hess,
    )

    model, data, theta = _model_and_parameters(kind, density_factory)
    coordinates = _theta_to_coordinates(model, theta)
    theta_jet = _evaluate_mask(model, data, theta, 7, transformed=False)
    transformed_jet = _evaluate_mask(
        model, data, coordinates, 7, transformed=True
    )
    expected = native_coordinate_chain_rule_value_grad_hess(
        coordinates,
        theta,
        *theta_jet,
        model.p_vol,
        model.q_vol,
        density_name,
        p_ar=None if not kind.startswith("arma_") else model.p_ar,
        q_ma=model.q_ma,
        leverage=model.leverage,
    )

    assert transformed_jet[0] == pytest.approx(
        expected[0], rel=3e-10, abs=3e-9
    )
    assert_allclose(transformed_jet[1], expected[1], rtol=3e-8, atol=3e-8)
    assert_allclose(transformed_jet[2], expected[2], rtol=5e-7, atol=5e-7)


@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
def test_native_zero_skew_reduces_exactly_to_student_t(kind: str) -> None:
    student_model, data, student_theta = _model_and_parameters(kind, StudentT)
    volatility = GJRGARCH(2, 1) if "gjr" in kind else GARCH(2, 1)
    skew_model = _describe_model(
        (ARMA(2, 1) + volatility if kind.startswith("arma_") else volatility)
        + SkewT()
    )
    skew_theta = np.ascontiguousarray(np.r_[student_theta, 0.0])
    student = _evaluate_mask(
        student_model, data, student_theta, 7, transformed=False
    )
    skew = _evaluate_mask(skew_model, data, skew_theta, 7, transformed=False)
    common = np.arange(student_model.n_params)

    assert skew[0] == pytest.approx(student[0], rel=2e-13, abs=2e-13)
    assert_allclose(skew[1][common], student[1], rtol=2e-12, atol=2e-12)
    assert_allclose(
        skew[2][np.ix_(common, common)],
        student[2],
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize("kind", ("garch", "gjr", "arma_garch", "arma_gjr"))
@pytest.mark.parametrize(
    "special_factory,fixed_lambda",
    ((NormalInverseGaussian, -0.5), (Hyperbolic, 1.0)),
    ids=("nig", "hyperbolic"),
)
def test_native_gh_special_cases_reduce_exactly(
    kind: str,
    special_factory: DensityFactory,
    fixed_lambda: float,
) -> None:
    special_model, data, special_theta = _model_and_parameters(
        kind, special_factory
    )
    volatility = GJRGARCH(2, 1) if "gjr" in kind else GARCH(2, 1)
    general_model = _describe_model(
        (ARMA(2, 1) + volatility if kind.startswith("arma_") else volatility)
        + GeneralizedHyperbolic()
    )
    general_theta = np.ascontiguousarray(np.r_[special_theta, fixed_lambda])
    special = _evaluate_mask(
        special_model, data, special_theta, 7, transformed=False
    )
    general = _evaluate_mask(
        general_model, data, general_theta, 7, transformed=False
    )
    common = np.arange(special_model.n_params)

    assert general[0] == pytest.approx(special[0], rel=2e-13, abs=2e-13)
    assert_allclose(general[1][common], special[1], rtol=2e-12, atol=2e-12)
    assert_allclose(
        general[2][np.ix_(common, common)],
        special[2],
        rtol=2e-11,
        atol=2e-11,
    )


def test_arma_egarch_zero_skew_reduces_exactly_to_student_t() -> None:
    student_model, data, student_theta = _arma_egarch_model_and_parameters(
        StudentT
    )
    skew_model = _describe_model(
        ARMA(student_model.p_ar, student_model.q_ma)
        + EGARCH(student_model.p_vol, student_model.q_vol)
        + SkewT()
    )
    skew_theta = np.ascontiguousarray(np.r_[student_theta, 0.0])
    student = _evaluate_mask(
        student_model, data, student_theta, 7, transformed=False
    )
    skew = _evaluate_mask(
        skew_model, data, skew_theta, 7, transformed=False
    )
    common = np.arange(student_model.n_params)

    assert skew[0] == pytest.approx(student[0], rel=2e-13, abs=2e-13)
    assert_allclose(skew[1][common], student[1], rtol=2e-12, atol=2e-12)
    assert_allclose(
        skew[2][np.ix_(common, common)],
        student[2],
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize(
    "special_factory,fixed_lambda",
    ((NormalInverseGaussian, -0.5), (Hyperbolic, 1.0)),
    ids=("nig", "hyperbolic"),
)
def test_arma_egarch_gh_special_cases_reduce_exactly(
    special_factory: DensityFactory,
    fixed_lambda: float,
) -> None:
    special_model, data, special_theta = _arma_egarch_model_and_parameters(
        special_factory
    )
    general_model = _describe_model(
        ARMA(special_model.p_ar, special_model.q_ma)
        + EGARCH(special_model.p_vol, special_model.q_vol)
        + GeneralizedHyperbolic()
    )
    general_theta = np.ascontiguousarray(np.r_[special_theta, fixed_lambda])
    special = _evaluate_mask(
        special_model, data, special_theta, 7, transformed=False
    )
    general = _evaluate_mask(
        general_model, data, general_theta, 7, transformed=False
    )
    common = np.arange(special_model.n_params)

    assert general[0] == pytest.approx(special[0], rel=5e-13, abs=5e-13)
    assert_allclose(general[1][common], special[1], rtol=2e-12, atol=2e-12)
    assert_allclose(
        general[2][np.ix_(common, common)],
        special[2],
        rtol=2e-11,
        atol=2e-11,
    )


def test_garch_ged_theta_jet_matches_ad_below_legacy_shape_padding() -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import garch_value_grad_hess

    model, data, theta = _model_and_parameters("garch", GED)
    theta[-1] = 0.8
    value, gradient, hessian = _evaluate_mask(
        model, data, theta, 7, transformed=False
    )
    expected = garch_value_grad_hess(
        theta, data * data, model.p_vol, model.q_vol, dist="ged"
    )

    assert value == pytest.approx(expected[0], rel=2e-9, abs=2e-8)
    assert_allclose(gradient, expected[1], rtol=2e-7, atol=2e-7)
    assert_allclose(hessian, expected[2], rtol=3e-6, atol=3e-6)
    filtered = filter_spec(GARCH(2, 1) + GED(), data, theta)
    assert -filtered.log_likelihood == pytest.approx(value, rel=2e-10, abs=2e-9)


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
@pytest.mark.parametrize(
    "kind", ("garch", "gjr", "arma_garch", "arma_gjr")
)
def test_native_density_matrices_fit_credibly(
    density_name: str,
    density_factory: DensityFactory,
    kind: str,
) -> None:
    leverage = "gjr" in kind
    data = _joint_gjr_returns() if leverage else (
        _gh_returns() if density_name == "gh" else _returns()
    )
    volatility = GJRGARCH(2, 1) if leverage else GARCH(2, 1)
    spec = (
        ARMA(1, 1) + volatility + density_factory()
        if kind.startswith("arma_")
        else volatility + density_factory()
    )
    result = spec.fit(data, solver="native", maxiter=700)

    assert result.converged, result.convergence_message
    assert result.fit_info.optimizer_mode == 0
    assert result.fit_info.kkt_residual <= result.fit_info.kkt_tolerance
    assert result.fit_info.maximum_constraint_violation <= result.fit_info.kkt_tolerance
    assert result.fit_info.objective_parity_error <= 1e-7
    assert result.fit_info.engine == "cpp-function-jet"
    assert np.isfinite(result.log_likelihood)


def test_joint_fit_has_cross_component_curvature_and_inference() -> None:
    generator = np.random.default_rng(42)
    n, burn = 1200, 300
    omega, alpha, beta = 2e-5, 0.08, 0.85
    c, phi, theta_ma = 0.001, 0.25, -0.15
    variance = np.empty(n + burn)
    residual = np.empty(n + burn)
    observations = np.empty(n + burn)
    variance[0] = omega / (1.0 - alpha - beta)
    residual[0] = 0.0
    observations[0] = c
    for time in range(1, n + burn):
        variance[time] = (
            omega + alpha * residual[time - 1] ** 2 + beta * variance[time - 1]
        )
        residual[time] = np.sqrt(variance[time]) * generator.normal()
        observations[time] = (
            c
            + phi * observations[time - 1]
            + theta_ma * residual[time - 1]
            + residual[time]
        )

    data = np.ascontiguousarray(observations[burn:], dtype=np.float64)
    result = (ARMA(1, 1) + GARCH(1, 1) + Normal()).fit(
        data, solver="native", maxiter=1000
    )

    assert result.converged, result.convergence_message
    assert result.hessian is not None
    assert np.any(np.abs(result.hessian[:3, 3:6]) > 1e-8)
    assert result.cov_matrix is not None
    assert result.std_errors is not None
    assert np.all(np.isfinite(result.std_errors))


@pytest.mark.parametrize(
    "spec,data",
    (
        (GARCH(3, 2) + Normal(), _returns(seed=778, n=900)),
        (
            ARMA(2, 2) + GARCH(3, 2) + StudentT(),
            _joint_returns(),
        ),
        (ARMA(0, 0) + GARCH(1, 1) + Normal(), _returns(seed=779, n=900)),
    ),
    ids=("garch-runtime-fallback", "joint-runtime-fallback", "zero-mean-orders"),
)
def test_native_runtime_orders_beyond_precompiled_cases(
    spec: object, data: np.ndarray
) -> None:
    result = spec.fit(data, solver="native", maxiter=1000)

    assert result.converged, result.convergence_message
    assert result.fit_info.objective_parity_error <= 1e-7


@pytest.mark.parametrize("density_factory", (Normal, StudentT, GeneralizedHyperbolic))
def test_legacy_joint_results_report_the_complete_likelihood(
    density_factory: DensityFactory,
) -> None:
    data = _returns(seed=400, n=450)
    spec = ARMA(1, 1) + GARCH(1, 1) + density_factory()
    result = spec.fit(data, solver="slsqp", log_mode=True)
    filtered = filter_spec(spec, result.data, result.params)

    assert result.log_likelihood == pytest.approx(
        filtered.log_likelihood, rel=2e-11, abs=2e-9
    )
    assert abs(result.log_likelihood) > 100.0


@pytest.mark.parametrize("density_factory", (Normal, StudentT, SkewT, GED))
def test_legacy_joint_fit_honors_the_shared_iteration_budget(
    density_factory: DensityFactory,
) -> None:
    result = (ARMA(1, 1) + GARCH(1, 1) + density_factory()).fit(
        _joint_returns(n=350, p_ar=1, q_ma=1, p_vol=1, q_vol=1),
        solver="slsqp",
        maxiter=1,
    )

    assert result.fit_info.maxiter == 1
    assert result.n_iter <= 1


def test_native_solver_retains_a_legacy_variance_gamma_optimum() -> None:
    data = _returns()
    legacy = (ARMA(1, 1) + GARCH(2, 1) + VarianceGamma()).fit(
        data, solver="slsqp", log_mode=True
    )
    native = (ARMA(1, 1) + GARCH(2, 1) + VarianceGamma()).fit(
        data,
        solver="native",
        initial_params=legacy.params,
        maxiter=700,
    )
    legacy_at_common_sample = filter_spec(
        legacy.spec, native.data, legacy.params
    ).log_likelihood

    assert native.converged, native.convergence_message
    assert native.log_likelihood >= legacy_at_common_sample - 1e-8


def test_native_failure_is_quiet_and_persistent(capsys: pytest.CaptureFixture[str]) -> None:
    result = (ARMA(1, 1) + GARCH(2, 1) + SkewT()).fit(
        _returns(n=450), solver="native", maxiter=1
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert not result.converged
    assert result.fit_info.warnings
    assert "Strict credibility check failed" in result.convergence_message


def test_native_reports_an_open_boundary_separately_from_credibility() -> None:
    generator = np.random.default_rng(20260827)
    independent = np.ascontiguousarray(
        generator.standard_t(7, size=700) * 0.035,
        dtype=np.float64,
    )
    result = (GARCH(2, 1) + Normal()).fit(
        independent, solver="native", maxiter=700
    )

    assert result.fit_info.boundary_reached
    assert any("open model boundary" in warning for warning in result.fit_info.warnings)
    if result.converged:
        assert result.fit_info.kkt_residual <= result.fit_info.kkt_tolerance
        assert result.fit_info.maximum_constraint_violation <= result.fit_info.kkt_tolerance


def test_native_fits_are_reentrant_across_threads() -> None:
    datasets = [
        _joint_returns(
            seed=900 + seed,
            n=700,
            p_ar=1,
            q_ma=1,
            p_vol=1,
            q_vol=1,
        )
        for seed in range(4)
    ]

    def fit(data: np.ndarray) -> np.ndarray:
        result = (ARMA(1, 1) + GARCH(1, 1) + StudentT()).fit(
            data, solver="native", maxiter=700
        )
        assert result.converged, result.convergence_message
        return result.params

    sequential = [fit(data) for data in datasets]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(fit, datasets))

    for expected, actual in zip(sequential, concurrent):
        assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_curvature_polish_is_deterministic_and_never_degrades_accepted_point() -> None:
    data = _joint_returns(
        seed=11,
        n=350,
        p_ar=2,
        q_ma=2,
        p_vol=3,
        q_vol=2,
    )
    expected = np.array(
        [
            -5.44738324e-4,
            6.25850478e-2,
            3.40852054e-1,
            -8.55877305e-2,
            2.69470762e-4,
            5.05091464e-2,
            1.13079489e-9,
            6.90973746e-14,
        ],
        dtype=np.float64,
    )

    model = _describe_model(ARMA(1, 2) + GARCH(1, 2) + Normal())
    baseline_value, baseline_gradient, _ = _evaluate_theta_request(
        model, data, expected, request=3
    )
    baseline_assessment = _assess_theta_solution(
        model,
        expected,
        baseline_gradient,
        data,
        1e-10,
        boundary_reached=False,
    )

    def fit() -> tuple[np.ndarray, float, float]:
        result = (ARMA(1, 2) + GARCH(1, 2) + Normal()).fit(
            data, solver="native", maxiter=700
        )
        assert result.log_likelihood >= -baseline_value - 1e-12
        assert result.fit_info.kkt_residual <= baseline_assessment.kkt_residual
        return (
            result.params,
            result.log_likelihood,
            result.fit_info.kkt_residual,
        )

    sequential = [fit() for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(lambda _: fit(), range(2)))

    for actual in sequential + concurrent:
        assert_allclose(actual[0], sequential[0][0], rtol=0.0, atol=0.0)
        assert actual[1:] == sequential[0][1:]


def test_projected_kkt_respects_a_nearly_active_coefficient() -> None:
    model = _describe_model(GARCH(1, 1) + Normal())
    parameters = np.array([1.0, 2e-8, 0.5], dtype=np.float64)
    gradient = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    residual = _native_core._garch_projected_kkt(
        parameters.ctypes.data,
        gradient.ctypes.data,
        model.p_vol,
        model.q_vol,
        model.density_code,
        1.0,
        1.0,
    )

    assert residual == pytest.approx(2e-8, rel=0.0, abs=1e-15)


def test_native_fit_reports_conditional_theta_refinement() -> None:
    result = (GARCH(2, 1) + VarianceGamma()).fit(
        _gh_returns(seed=11, n=500),
        solver="native",
        native_accuracy=1e-14,
        maxiter=300,
    )

    assert result.fit_info.solver == "native-slsqp+projected-newton"
    assert result.fit_info.optimization_space == (
        "native-z-then-scaled-theta-space"
    )
    assert result.fit_info.optimizer_phases == (
        "native-z-slsqp",
        "scaled-theta-projected-newton",
    )
    assert result.fit_info.theta_refinement_steps is not None
    assert result.fit_info.theta_refinement_steps > 0
    assert result.fit_info.kkt_residual is not None
    assert result.fit_info.kkt_tolerance is not None
    assert result.fit_info.kkt_residual <= result.fit_info.kkt_tolerance
    assert "projected Newton" in result.convergence_message


def test_gjr_theta_refinement_matches_an_independent_coordinate_solve() -> None:
    data = _joint_gjr_returns(n=500)
    spec = GJRGARCH(2, 1) + StudentT()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    coordinates = _theta_to_coordinates(model, start)
    lower, upper = _coordinate_bounds(model)
    bounds = tuple(
        (
            None if np.isnan(lower_value) else float(lower_value),
            None if np.isnan(upper_value) else float(upper_value),
        )
        for lower_value, upper_value in zip(lower, upper)
    )
    scale = 1.0 / model.scored_observations(data.size)

    def value_and_gradient(point: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = _evaluate_coordinate_request(
            model,
            data,
            np.ascontiguousarray(point, dtype=np.float64),
            request=3,
        )
        return value * scale, gradient * scale

    native = spec.fit(
        data, solver="native", initial_params=start, maxiter=300
    )
    reference = minimize(
        value_and_gradient,
        coordinates,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
    )
    reference_theta = _coordinates_to_theta(
        model, np.ascontiguousarray(reference.x, dtype=np.float64)
    )
    native_nll, _, _ = _evaluate_theta_request(
        model, data, np.ascontiguousarray(native.params), request=1
    )
    reference_nll, _, _ = _evaluate_theta_request(
        model, data, reference_theta, request=1
    )

    assert reference.success
    assert native.converged, native.convergence_message
    assert native.fit_info.optimizer_phases in (
        ("native-z-slsqp",),
        ("native-z-slsqp", "scaled-theta-slsqp"),
    )
    assert native_nll <= reference_nll + 1e-7
    assert native.fit_info.kkt_residual <= native.fit_info.kkt_tolerance
    alpha = native.params[1:3]
    gamma = native.params[3:5]
    assert np.min(alpha + gamma) <= native.fit_info.kkt_tolerance


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_egarch_default_replacement_matches_legacy_solution_quality(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    data = np.ascontiguousarray(
        np.random.default_rng(20260828).standard_t(7, 500) * np.sqrt(5.0 / 7.0)
    )
    spec = EGARCH(2, 1) + density_factory()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)

    legacy = spec.fit(
        data,
        solver="legacy",
        log_mode=True,
        initial_params=start,
        maxiter=500,
    )
    native = spec.fit(data, initial_params=start, maxiter=500)
    legacy_native_nll = _evaluate_theta_request(
        model, data, np.ascontiguousarray(legacy.params), request=1
    )[0]
    native_native_nll = _evaluate_theta_request(
        model, data, np.ascontiguousarray(native.params), request=1
    )[0]
    legacy_production_nll = -spec.loglikelihood(data, legacy.params)
    native_production_nll = -spec.loglikelihood(data, native.params)

    assert legacy.converged, (density_name, legacy.convergence_message)
    assert native.converged, (density_name, native.convergence_message)
    assert legacy.fit_info.engine == "cpp-function-jet"
    assert legacy.fit_info.solver == "scipy-lbfgsb"
    assert legacy.fit_info.optimizer_phases[0] == "scipy-z-lbfgsb"
    assert legacy_native_nll == pytest.approx(
        legacy_production_nll, rel=2e-10, abs=2e-8
    )
    assert native_native_nll == pytest.approx(
        native_production_nll, rel=2e-10, abs=2e-8
    )
    assert native_native_nll <= legacy_native_nll + 2e-7
    assert native.fit_info.kkt_residual <= native.fit_info.kkt_tolerance
    assert native.fit_info.engine == "cpp-function-jet"
    assert native.fit_info.solver == "native-curvature+slsqp"
    assert native.fit_info.optimizer_phases == (
        "native-z-exact-curvature+slsqp",
    )
    assert native.fit_info.requested_solver is None
    assert native.fit_info.used_default_solver


def test_egarch_curvature_preconditioner_recovers_a_difficult_matched_start() -> None:
    generator = np.random.default_rng(9321)
    data = np.ascontiguousarray(
        generator.standard_t(7, 650) * np.sqrt(5.0 / 7.0)
    )
    spec = EGARCH(2, 1) + StudentT()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    legacy = spec.fit(
        data,
        solver="legacy",
        log_mode=True,
        initial_params=start,
        maxiter=500,
    )
    native = spec.fit(data, initial_params=start, maxiter=500)
    legacy_nll = _evaluate_theta_request(
        model, data, np.ascontiguousarray(legacy.params), request=1
    )[0]
    native_nll = _evaluate_theta_request(
        model, data, np.ascontiguousarray(native.params), request=1
    )[0]

    assert native.converged, native.convergence_message
    assert native_nll <= legacy_nll + 1e-8
    assert native.fit_info.curvature_refinement_steps >= 1
    assert native.fit_info.kkt_residual <= 1e-9


def test_egarch_long_horizon_near_persistence_matches_ad() -> None:
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import egarch_logspace_value_grad_hess

    generator = np.random.default_rng(9321)
    data = np.ascontiguousarray(
        generator.standard_t(7, 650) * np.sqrt(5.0 / 7.0)
    )
    model = _describe_model(EGARCH(2, 1) + StudentT())
    theta = np.array([-0.01, -0.05, 0.03, -0.03, 0.01, 0.98, 10.0])
    coordinates = _theta_to_coordinates(model, theta)
    native = _evaluate_coordinate_request(
        model, data, coordinates, request=7
    )
    expected = egarch_logspace_value_grad_hess(
        coordinates, data, dist="studentt", p=2, q=1
    )

    assert native[0] == pytest.approx(expected[0], rel=2e-11, abs=2e-9)
    assert_allclose(native[1], expected[1], rtol=2e-9, atol=2e-8)
    assert_allclose(native[2], expected[2], rtol=2e-8, atol=2e-5)


def test_arma_egarch_long_horizon_near_persistence_matches_ad() -> None:
    pytest.importorskip("jax")
    data = _joint_egarch_returns(seed=9321, n=650)
    spec = ARMA(1, 1) + EGARCH(2, 1) + StudentT()
    model = _describe_model(spec)
    theta, _ = _initial_theta(spec, model, data, None)
    theta[:] = np.array([
        0.001, 0.18, -0.10,
        -0.01, -0.05, 0.03, -0.03, 0.01, 0.98,
        10.0,
    ])
    native = _evaluate_mask(model, data, theta, 7, transformed=False)
    expected = _independent_arma_egarch_theta_jet(
        "studentt", model, data, theta
    )

    assert native[0] == pytest.approx(expected[0], rel=2e-10, abs=2e-8)
    assert_allclose(native[1], expected[1], rtol=3e-7, atol=3e-5)
    assert_allclose(native[2], expected[2], rtol=3e-6, atol=3e-3)


def test_arma_egarch_default_matches_scipy_solution_quality() -> None:
    data = _joint_egarch_returns(n=600)
    spec = ARMA(1, 1) + EGARCH(1, 1) + StudentT()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    reference = spec.fit(
        data,
        solver="legacy",
        log_mode=True,
        initial_params=start,
        maxiter=500,
    )
    native = spec.fit(data, initial_params=start, maxiter=500)
    reference_at_reference = _evaluate_theta_request(
        model, data, np.ascontiguousarray(reference.params), request=1
    )[0]
    native_at_native = _evaluate_theta_request(
        model, data, np.ascontiguousarray(native.params), request=1
    )[0]
    reference_at_native = -spec.loglikelihood(data, native.params)
    native_at_reference = -spec.loglikelihood(data, reference.params)

    assert reference.converged, reference.convergence_message
    assert native.converged, native.convergence_message
    assert reference_at_reference == pytest.approx(
        native_at_reference, rel=2e-11, abs=2e-9
    )
    assert native_at_native == pytest.approx(
        reference_at_native, rel=2e-11, abs=2e-9
    )
    assert native_at_native <= reference_at_reference + 2e-7
    assert native.fit_info.kkt_residual <= native.fit_info.kkt_tolerance
    assert native.fit_info.engine == "cpp-function-jet"
    assert native.fit_info.requested_solver is None
    assert native.fit_info.used_default_solver


def test_native_egarch_result_filter_does_not_clip_small_valid_variances() -> None:
    data = np.ascontiguousarray([1.0, 0.0, 0.0, 0.0])
    model = _describe_model(EGARCH(1, 1) + Normal())
    theta = np.ascontiguousarray([-40.0, 0.0, 0.0, 0.0])

    variance = _materialize_native_egarch_variance(model, data, theta)
    public_state = filter_spec(EGARCH(1, 1) + Normal(), data, theta)

    assert variance[0] == pytest.approx(np.mean(data * data))
    assert_allclose(variance[1:], np.exp(-40.0), rtol=2e-15, atol=0.0)
    assert np.all(variance[1:] < 1e-12)
    assert_allclose(public_state.sigma2, variance, rtol=0.0, atol=0.0)


def test_explicit_scipy_egarch_path_rejects_the_incomplete_theta_domain() -> None:
    with pytest.raises(ValueError, match="log_mode=False"):
        (EGARCH(2, 1) + Normal()).fit(
            _returns(n=100), solver="legacy", log_mode=False
        )


def test_explicit_slsqp_egarch_path_uses_the_canonical_jet() -> None:
    result = (EGARCH(1, 1) + Normal()).fit(
        _returns(n=200), solver="slsqp", maxiter=300
    )

    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver == "scipy-slsqp"
    assert result.fit_info.optimization_space == "scipy-z-space"


@pytest.mark.parametrize(
    "spec,data",
    (
        (GARCH(1, 1) + Normal(), _returns(n=400)),
        (GARCH(1, 1) + GED(), _returns(n=400)),
    ),
    ids=("normal", "ged"),
)
def test_supported_fixed_mle_defaults_to_native(
    spec: object, data: np.ndarray
) -> None:
    result = spec.fit(data)

    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver.startswith("native-slsqp")
    assert result.fit_info.requested_solver is None
    assert result.fit_info.used_default_solver


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_gjr_density_matrix_defaults_to_native(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    result = (GJRGARCH(2, 1) + density_factory()).fit(
        _joint_gjr_returns(n=500)
    )

    assert result.converged, (
        density_name,
        result.convergence_message,
    )
    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver.startswith("native-slsqp")
    assert result.fit_info.requested_solver is None
    assert result.fit_info.used_default_solver


@pytest.mark.parametrize(
    "density_name,density_factory",
    _DENSITIES,
    ids=(f"garch-{density_name}" for density_name, _ in _DENSITIES),
)
def test_joint_arma_garch_density_matrix_defaults_to_native(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    result = (ARMA(1, 1) + GARCH(1, 1) + density_factory()).fit(
        _joint_returns(n=500, p_ar=1, q_ma=1, p_vol=1, q_vol=1)
    )

    assert result.converged, (density_name, result.convergence_message)
    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver.startswith("native-slsqp")
    assert result.fit_info.jet_request_plan == (
        "fused F|G during optimization; theta-space F|G|H at the solution"
    )
    assert result.fit_info.requested_solver is None
    assert result.fit_info.used_default_solver


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_joint_arma_gjr_density_matrix_defaults_to_native(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    result = (ARMA(1, 1) + GJRGARCH(1, 1) + density_factory()).fit(
        _joint_gjr_returns(n=500)
    )

    assert result.converged, (density_name, result.convergence_message)
    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver.startswith("native-slsqp")
    assert result.fit_info.jet_request_plan == (
        "fused F|G during optimization; theta-space F|G|H at the solution"
    )
    assert result.fit_info.requested_solver is None
    assert result.fit_info.used_default_solver


@pytest.mark.parametrize("density_name,density_factory", _DENSITIES)
def test_joint_arma_egarch_density_matrix_defaults_to_native(
    density_name: str,
    density_factory: DensityFactory,
) -> None:
    result = (ARMA(1, 1) + EGARCH(1, 1) + density_factory()).fit(
        _joint_egarch_returns(n=500)
    )

    assert result.converged, (density_name, result.convergence_message)
    assert result.fit_info.engine == "cpp-function-jet"
    assert result.fit_info.solver.startswith("native-")
    assert result.fit_info.requested_solver is None
    assert result.fit_info.used_default_solver
    if density_name == "gh":
        assert result.fit_info.scientific_projection_steps is not None
        assert result.fit_info.scientific_projection_steps > 0
        assert "scientific-theta-box-projection" in (
            result.fit_info.optimizer_phases
        )
        assert result.fit_info.boundary_reached
        assert result.params[-2] == pytest.approx(0.1, rel=0.0, abs=1e-14)


@pytest.mark.parametrize(
    "fit_kwargs,requested_solver,optimization_space",
    (
        ({"solver": "slsqp"}, "slsqp", "z-space"),
        ({"solver": "legacy"}, "legacy", "z-space"),
        ({"log_mode": False}, None, "theta-space"),
    ),
)
def test_explicit_legacy_paths_remain_independent(
    fit_kwargs: dict[str, object],
    requested_solver: str | None,
    optimization_space: str,
) -> None:
    result = (GARCH(1, 1) + Normal()).fit(_returns(n=400), **fit_kwargs)

    assert result.fit_info.engine is None
    assert result.fit_info.solver != "native-slsqp"
    assert result.fit_info.requested_solver == requested_solver
    assert result.fit_info.optimization_space == optimization_space


def test_promoted_gjr_keeps_the_explicit_legacy_path_independent() -> None:
    result = (GJRGARCH(1, 1) + Normal()).fit(
        _joint_gjr_returns(n=400), solver="legacy"
    )

    assert result.fit_info.engine is None
    assert result.fit_info.solver != "native-slsqp"
    assert result.fit_info.requested_solver == "legacy"
    assert not result.fit_info.used_default_solver


def test_qmle_retains_legacy_while_auto_uses_promoted_native_candidates() -> None:
    data = _returns(n=400)
    qmle = (GARCH(1, 1) + Normal()).fit(data, method="qmle")
    selected = (
        AutoVol(candidates=["GARCH"], max_p=1, max_q=1) + Normal()
    ).fit(data, n_jobs=1)

    assert qmle.fit_info.engine is None
    assert selected.fit_info.engine == "cpp-function-jet"
    assert all(
        candidate.result is None
        or candidate.result.fit_info.engine == "cpp-function-jet"
        for candidate in selected.selection.candidates
    )
