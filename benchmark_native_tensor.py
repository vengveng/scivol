"""Deterministic promotion benchmark for the current native tensor slice.

The routine benchmark covers the promoted GARCH, GJR-GARCH, and EGARCH families,
including genuine joint ARMA-GARCH and ARMA-EGARCH models. Sparse requests use a representative
5,000-observation series and attribute the marginal cost from 500 observations.
Complete fits compare matched starts and domains: established adapters where
possible, otherwise an independent SciPy optimizer over the exact native jet and
coordinate chart.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from scivol import (
    ARMA,
    EGARCH,
    GARCH,
    GJRGARCH,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from scivol import _core, _native_core
from scivol._native_fit import (
    NativeModel,
    _assess_theta_solution,
    _coordinate_bounds,
    _coordinates_to_theta,
    _describe_model,
    _evaluate_coordinate_request,
    _evaluate_theta_request,
    _initial_theta,
    _theta_to_coordinates,
)


REQUESTS = (("F", 1), ("G", 2), ("H", 4))
DENSITIES = (
    Normal,
    StudentT,
    SkewT,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    NormalInverseGaussian,
    VarianceGamma,
)
Pointer = ctypes.POINTER(ctypes.c_double)
MaskPointer = ctypes.POINTER(ctypes.c_uint32)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    model_kind: str
    density_factory: type

    @property
    def label(self) -> str:
        return f"{self.model_kind}+{self.density_factory.__name__}"

    @property
    def is_joint(self) -> bool:
        return self.model_kind.startswith("ARMA")

    @property
    def leverage(self) -> bool:
        return "GJR-GARCH" in self.model_kind

    @property
    def is_egarch(self) -> bool:
        return "EGARCH" in self.model_kind

    @property
    def supports_request_benchmark(self) -> bool:
        if not self.is_egarch:
            return True
        if not self.is_joint:
            return self.density_factory is Normal
        return self.density_factory in (Normal, StudentT, SkewT, GED)

    @property
    def supports_matched_fit(self) -> bool:
        return not (self.is_joint and self.is_egarch) or (
            self.density_factory is StudentT
        )

    def build(self):
        density = self.density_factory()
        volatility = (
            EGARCH(2, 1)
            if self.is_egarch
            else GJRGARCH(2, 1)
            if self.leverage
            else GARCH(2, 1)
        )
        if self.is_joint:
            volatility = (
                EGARCH(1, 1)
                if self.is_egarch
                else GJRGARCH(1, 1)
                if self.leverage
                else GARCH(1, 1)
            )
            return ARMA(1, 1) + volatility + density
        return volatility + density


@dataclass(frozen=True, slots=True)
class RequestResult:
    case: str
    request: str
    native_us: float
    production_us: float
    baseline_native_us: float
    baseline_production_us: float

    @property
    def ratio(self) -> float:
        return self.native_us / self.production_us

    @property
    def marginal_ratio(self) -> float:
        native = self.native_us - self.baseline_native_us
        production = self.production_us - self.baseline_production_us
        return native / production if production > 0.0 else np.inf


@dataclass(frozen=True, slots=True)
class FitResult:
    case: str
    reference: str
    reference_success: bool
    native_seconds: float
    reference_seconds: float
    native_nfev: int
    native_njev: int
    reference_nfev: int
    reference_njev: int
    native_at_native: float
    established_at_native: float
    native_at_reference: float
    established_at_reference: float
    native_feasible: bool
    reference_feasible: bool
    native_kkt: float
    reference_kkt: float

    @property
    def ratio(self) -> float:
        return self.native_seconds / self.reference_seconds


@dataclass(frozen=True, slots=True)
class IndependentFit:
    params: NDArray[np.float64]
    nfev: int
    njev: int
    success: bool


class NativePointEvaluator:
    """Own all buffers needed to time prevalidated C-to-C jet requests."""

    def __init__(
        self,
        model: NativeModel,
        data: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> None:
        self.model = model
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.squared = np.ascontiguousarray(self.data * self.data)
        self.theta = np.ascontiguousarray(theta, dtype=np.float64)
        self.initial_variance = float(np.mean(self.squared))
        self.value = np.empty(1, dtype=np.float64)
        self.gradient = np.empty(model.n_params, dtype=np.float64)
        self.hessian = np.empty((model.n_params, model.n_params), dtype=np.float64)
        self.valid_mask = np.empty(1, dtype=np.uint32)
        workspace_bytes = 64
        if model.kind == "arma_egarch":
            workspace_bytes = max(
                workspace_bytes,
                int(_native_core._arma_egarch_workspace_bytes(
                    model.p_ar,
                    model.q_ma,
                    model.p_vol,
                    model.q_vol,
                    model.density_code,
                    0,
                    7,
                )),
            )
        self.workspace = np.empty(workspace_bytes, dtype=np.uint8)
        self.library = ctypes.CDLL(_native_core.__file__)
        self.theta_pointer = self._pointer(self.theta)
        self.data_pointer = self._pointer(self.data)
        self.squared_pointer = self._pointer(self.squared)
        self.value_pointer = self._pointer(self.value)
        self.gradient_pointer = self._pointer(self.gradient)
        self.hessian_pointer = self._pointer(self.hessian)
        self.valid_mask_pointer = self.valid_mask.ctypes.data_as(MaskPointer)
        self._evaluate = self._bind_evaluator()

    @staticmethod
    def _pointer(array: NDArray[np.float64]) -> Pointer:
        return array.ctypes.data_as(Pointer)

    def _bind_evaluator(self) -> Callable[[int], None]:
        if self.model.kind == "arma_egarch":
            function = self.library.vlk_arma_egarch_jet_evaluate_prevalidated
            function.argtypes = [
                Pointer, ctypes.c_size_t, Pointer, ctypes.c_size_t,
                ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_size_t, Pointer, Pointer, Pointer, MaskPointer,
            ]
            function.restype = ctypes.c_int32

            def evaluate(request: int) -> None:
                status = function(
                    self.theta_pointer, self.model.n_params,
                    self.data_pointer, self.data.size,
                    self.initial_variance, self.model.p_ar, self.model.q_ma,
                    self.model.p_vol, self.model.q_vol,
                    self.model.density_code, 0, request,
                    ctypes.c_void_p(self.workspace.ctypes.data),
                    self.workspace.nbytes, self.value_pointer,
                    self.gradient_pointer, self.hessian_pointer,
                    self.valid_mask_pointer,
                )
                self._require_success(status, request)

            return evaluate

        if self.model.kind == "egarch":
            function = self.library.vlk_egarch_jet_evaluate_prevalidated
            function.argtypes = [
                Pointer, Pointer, ctypes.c_size_t, ctypes.c_double,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_size_t, Pointer, Pointer, Pointer, MaskPointer,
            ]
            function.restype = ctypes.c_int32

            def evaluate(request: int) -> None:
                status = function(
                    self.theta_pointer, self.data_pointer, self.data.size,
                    self.initial_variance, self.model.p_vol, self.model.q_vol,
                    self.model.density_code, 0, request,
                    ctypes.c_void_p(self.workspace.ctypes.data),
                    self.workspace.nbytes, self.value_pointer,
                    self.gradient_pointer, self.hessian_pointer,
                    self.valid_mask_pointer,
                )
                self._require_success(status, request)

            return evaluate

        if self.model.kind == "garch":
            name = (
                "vlk_evaluate_generated_gjr_sample_prevalidated"
                if self.model.leverage
                else "vlk_evaluate_generated_garch_sample_prevalidated"
            )
            function = getattr(self.library, name)
            function.argtypes = [
                Pointer, Pointer, Pointer, ctypes.c_size_t, ctypes.c_double,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, Pointer, Pointer, Pointer,
                MaskPointer,
            ]
            function.restype = ctypes.c_int32

            def evaluate(request: int) -> None:
                status = function(
                    self.theta_pointer, self.data_pointer,
                    self.squared_pointer, self.data.size,
                    self.initial_variance, self.model.p_vol, self.model.q_vol,
                    self.model.density_code, 0, request,
                    self.value_pointer, self.gradient_pointer,
                    self.hessian_pointer, self.valid_mask_pointer,
                )
                self._require_success(status, request)

            return evaluate

        name = (
            "vlk_arma_gjr_garch_jet_evaluate_prevalidated"
            if self.model.leverage
            else "vlk_arma_garch_jet_evaluate_prevalidated"
        )
        function = getattr(self.library, name)
        function.argtypes = [
            Pointer, ctypes.c_size_t, Pointer, ctypes.c_size_t, ctypes.c_double,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_size_t, Pointer, Pointer, Pointer, MaskPointer,
        ]
        function.restype = ctypes.c_int32

        def evaluate(request: int) -> None:
            status = function(
                self.theta_pointer, self.model.n_params,
                self.data_pointer, self.data.size,
                self.initial_variance, self.model.p_ar, self.model.q_ma,
                self.model.p_vol, self.model.q_vol, self.model.density_code,
                0, request, ctypes.c_void_p(self.workspace.ctypes.data),
                self.workspace.nbytes, self.value_pointer,
                self.gradient_pointer, self.hessian_pointer,
                self.valid_mask_pointer,
            )
            self._require_success(status, request)

        return evaluate

    def _require_success(self, status: int, request: int) -> None:
        if status != 0 or int(self.valid_mask[0]) != request:
            raise RuntimeError(
                f"native request failed: status={status}, "
                f"valid={int(self.valid_mask[0])}, request={request}"
            )

    def __call__(self, request: int) -> None:
        self._evaluate(request)


class ProductionPointEvaluator:
    """Call established C workhorses through the same raw ABI as the jets."""

    def __init__(
        self,
        model: NativeModel,
        data: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> None:
        self.model = model
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.squared = np.ascontiguousarray(self.data * self.data)
        self.theta = np.ascontiguousarray(theta, dtype=np.float64)
        self.initial_variance = float(np.mean(self.squared))
        self.residuals = np.empty_like(self.data)
        self.variance = np.empty_like(self.data)
        self.variance[0] = self.initial_variance
        self.gradient = np.empty(model.n_params, dtype=np.float64)
        self.hessian = np.empty((model.n_params, model.n_params), dtype=np.float64)
        self.maximum_lag = max(
            model.p_ar, model.q_ma, model.p_vol, model.q_vol, 1
        )
        self.initial_residuals = np.zeros(self.maximum_lag, dtype=np.float64)
        self.initial_variances = np.full(
            self.maximum_lag, self.initial_variance, dtype=np.float64
        )
        self.empty_features = np.empty(1, dtype=np.float64)
        self.library = ctypes.CDLL(_core.__file__)
        self.theta_pointer = self._pointer(self.theta)
        self.data_pointer = self._pointer(self.data)
        self.squared_pointer = self._pointer(self.squared)
        self.residuals_pointer = self._pointer(self.residuals)
        self.variance_pointer = self._pointer(self.variance)
        self.gradient_pointer = self._pointer(self.gradient)
        self.hessian_pointer = self._pointer(self.hessian)
        self.initial_residuals_pointer = self._pointer(self.initial_residuals)
        self.initial_variances_pointer = self._pointer(self.initial_variances)
        self.empty_features_pointer = self._pointer(self.empty_features)
        self._requests = self._build_requests()

    @staticmethod
    def _pointer(array: NDArray[np.float64]) -> Pointer:
        return array.ctypes.data_as(Pointer)

    def _bind(
        self,
        name: str,
        arguments: list[type],
        result: type | None,
    ):
        function = getattr(self.library, name)
        function.argtypes = arguments
        function.restype = result
        return function

    def _build_requests(self) -> dict[int, Callable[[], object]]:
        if self.model.kind == "arma_egarch":
            return self._build_arma_egarch_requests()
        if self.model.kind == "egarch":
            return self._build_egarch_normal_requests()
        if self.model.kind == "garch":
            if self.model.leverage:
                return self._build_gjr_requests()
            return self._build_garch_requests()
        return self._build_joint_requests()

    def _build_arma_egarch_requests(self) -> dict[int, Callable[[], object]]:
        suffix = {
            0: "normal",
            1: "studentt",
            2: "skewt",
            3: "ged",
        }[self.model.density_code]
        arguments = [
            Pointer,
            Pointer,
            Pointer,
            Pointer,
            Pointer,
            Pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        value_function = self._bind(
            f"arma_egarch_nll_pq_{suffix}", arguments, ctypes.c_double
        )
        derivative_arguments = [*arguments[:6], Pointer, *arguments[6:]]
        first_function = self._bind(
            f"arma_egarch_nll_grad_pq_{suffix}",
            derivative_arguments,
            None,
        )
        second_function = self._bind(
            f"arma_egarch_hess_pq_{suffix}",
            derivative_arguments,
            None,
        )
        common = (
            self.theta_pointer,
            self.data_pointer,
            self.residuals_pointer,
            self.variance_pointer,
            self.initial_residuals_pointer,
            self.initial_variances_pointer,
        )
        orders = (
            self.data.size,
            self.model.p_ar,
            self.model.q_ma,
            self.model.p_vol,
            self.model.q_vol,
        )
        return {
            1: lambda: value_function(*common, *orders),
            2: lambda: first_function(
                *common, self.gradient_pointer, *orders
            ),
            4: lambda: second_function(
                *common, self.hessian_pointer, *orders
            ),
        }

    def _build_egarch_normal_requests(self) -> dict[int, Callable[[], object]]:
        theta = self.theta_pointer
        residuals = self.data_pointer
        variance = self.variance_pointer
        gradient = self.gradient_pointer
        hessian = self.hessian_pointer
        n, p, q = self.data.size, self.model.p_vol, self.model.q_vol
        value_function = self._bind(
            "egarch_ll_pq_normal",
            [Pointer, Pointer, Pointer, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t],
            ctypes.c_double,
        )
        gradient_function = self._bind(
            "egarch_ll_grad_pq_normal",
            [Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t],
            None,
        )
        hessian_function = self._bind(
            "egarch_ll_hess_pq_normal",
            [Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t],
            None,
        )
        return {
            1: lambda: value_function(theta, residuals, variance, n, p, q),
            2: lambda: gradient_function(
                theta, residuals, variance, gradient, n, p, q
            ),
            4: lambda: hessian_function(
                theta, residuals, variance, hessian, n, p, q
            ),
        }

    def _build_garch_requests(self) -> dict[int, Callable[[], object]]:
        theta = self.theta_pointer
        residuals = self.data_pointer
        squared = self.squared_pointer
        variance = self.variance_pointer
        gradient = self.gradient_pointer
        hessian = self.hessian_pointer
        n, p, q = self.data.size, self.model.p_vol, self.model.q_vol
        density = self.model.density_code

        if density == 2:
            variance_function = self._bind(
                "garch_variance_pq",
                [Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                None,
            )
            value_function = self._bind(
                "skewt_nll",
                [Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_double, ctypes.c_double],
                ctypes.c_double,
            )
            first_function = self._bind(
                "garch_ll_grad_pq_skewt",
                [Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                None,
            )
            second_function = self._bind(
                "garch_ll_hess_pq_skewt",
                [Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                None,
            )

            def value() -> float:
                variance_function(theta, squared, variance, n, p, q)
                return value_function(
                    residuals, variance, n,
                    float(self.theta[-2]), float(self.theta[-1]),
                )

            def first() -> None:
                first_function(
                    theta, residuals, variance, gradient, n, p, q
                )

            def second() -> None:
                second_function(
                    theta, residuals, variance, hessian, n, p, q
                )

            return {1: value, 2: first, 4: second}

        if density in (0, 1, 3):
            suffix = {0: "normal", 1: "studentt", 3: "ged"}[density]
            value_function = self._bind(
                f"garch_ll_pq_{suffix}",
                [Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                ctypes.c_double,
            )
            derivative_arguments = [
                Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t,
            ]
            first_function = self._bind(
                f"garch_ll_grad_pq_{suffix}", derivative_arguments, None
            )
            second_function = self._bind(
                f"garch_ll_hess_pq_{suffix}", derivative_arguments, None
            )

            def value() -> float:
                return value_function(theta, squared, variance, n, p, q)

            def first() -> None:
                first_function(theta, squared, variance, gradient, n, p, q)

            def second() -> None:
                second_function(theta, squared, variance, hessian, n, p, q)

            return {1: value, 2: first, 4: second}

        fixed_lambda = {
            4: np.nan,
            5: -0.5,
            6: 1.0,
            7: np.inf,
        }[density]
        value_function = self._bind(
            "garch_gh_nll_pq",
            [Pointer, Pointer, Pointer, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double],
            ctypes.c_double,
        )
        derivative_arguments = [
            Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
        ]
        first_function = self._bind(
            "garch_gh_nll_grad_pq", derivative_arguments, None
        )
        second_function = self._bind(
            "garch_gh_nll_hess_pq", derivative_arguments, None
        )

        def value() -> float:
            return value_function(
                theta, residuals, variance, n, p, q, fixed_lambda
            )

        def first() -> None:
            first_function(
                theta, residuals, variance, gradient, n, p, q, fixed_lambda
            )

        def second() -> None:
            second_function(
                theta, residuals, variance, hessian, n, p, q, fixed_lambda
            )

        return {1: value, 2: first, 4: second}

    def _build_gjr_requests(self) -> dict[int, Callable[[], object]]:
        theta = self.theta_pointer
        residuals = self.data_pointer
        variance = self.variance_pointer
        gradient = self.gradient_pointer
        hessian = self.hessian_pointer
        n, p, q = self.data.size, self.model.p_vol, self.model.q_vol
        density = self.model.density_code

        if density == 2:
            variance_function = self._bind(
                "gjr_garch_variance_pq",
                [Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                None,
            )
            value_function = self._bind(
                "skewt_nll",
                [Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_double, ctypes.c_double],
                ctypes.c_double,
            )
            derivative_arguments = [
                Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t,
            ]
            first_function = self._bind(
                "gjr_garch_ll_grad_pq_skewt", derivative_arguments, None
            )
            second_function = self._bind(
                "gjr_garch_ll_hess_pq_skewt", derivative_arguments, None
            )

            def value() -> float:
                variance_function(theta, residuals, variance, n, p, q)
                return value_function(
                    residuals, variance, n,
                    float(self.theta[-2]), float(self.theta[-1]),
                )

            return {
                1: value,
                2: lambda: first_function(
                    theta, residuals, variance, gradient, n, p, q
                ),
                4: lambda: second_function(
                    theta, residuals, variance, hessian, n, p, q
                ),
            }

        if density in (0, 1):
            suffix = {0: "normal", 1: "studentt"}[density]
            value_function = self._bind(
                f"gjr_garch_ll_pq_{suffix}",
                [Pointer, Pointer, Pointer, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t],
                ctypes.c_double,
            )
            derivative_arguments = [
                Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t,
            ]
            first_function = self._bind(
                f"gjr_garch_ll_grad_pq_{suffix}", derivative_arguments, None
            )
            second_function = self._bind(
                f"gjr_garch_ll_hess_pq_{suffix}", derivative_arguments, None
            )
            return {
                1: lambda: value_function(theta, residuals, variance, n, p, q),
                2: lambda: first_function(
                    theta, residuals, variance, gradient, n, p, q
                ),
                4: lambda: second_function(
                    theta, residuals, variance, hessian, n, p, q
                ),
            }

        if density == 3:
            value_function = self._bind(
                "linear_mean_gjr_garch_nll_pq_ged",
                [Pointer, Pointer, Pointer, Pointer, Pointer,
                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                 ctypes.c_size_t],
                ctypes.c_double,
            )
            derivative_arguments = [
                Pointer, Pointer, Pointer, Pointer, Pointer, Pointer,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            first_function = self._bind(
                "linear_mean_gjr_garch_nll_grad_pq_ged",
                derivative_arguments,
                None,
            )
            second_function = self._bind(
                "linear_mean_gjr_garch_hess_pq_ged",
                derivative_arguments,
                None,
            )
            arguments = (
                theta, residuals, self.empty_features_pointer,
                self.residuals_pointer, variance,
            )
            orders = (n, 0, p, q)
            return {
                1: lambda: value_function(*arguments, *orders),
                2: lambda: first_function(
                    *arguments, gradient, *orders
                ),
                4: lambda: second_function(
                    *arguments, hessian, *orders
                ),
            }

        fixed_lambda = {
            4: np.nan,
            5: -0.5,
            6: 1.0,
            7: np.inf,
        }[density]
        value_function = self._bind(
            "gjr_garch_gh_nll_pq",
            [Pointer, Pointer, Pointer, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double],
            ctypes.c_double,
        )
        derivative_arguments = [
            Pointer, Pointer, Pointer, Pointer, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
        ]
        first_function = self._bind(
            "gjr_garch_gh_nll_grad_pq", derivative_arguments, None
        )
        second_function = self._bind(
            "gjr_garch_gh_nll_hess_pq", derivative_arguments, None
        )
        return {
            1: lambda: value_function(
                theta, residuals, variance, n, p, q, fixed_lambda
            ),
            2: lambda: first_function(
                theta, residuals, variance, gradient, n, p, q, fixed_lambda
            ),
            4: lambda: second_function(
                theta, residuals, variance, hessian, n, p, q, fixed_lambda
            ),
        }

    def _build_joint_requests(self) -> dict[int, Callable[[], object]]:
        theta = self.theta_pointer
        data = self.data_pointer
        residuals = self.residuals_pointer
        variance = self.variance_pointer
        gradient = self.gradient_pointer
        hessian = self.hessian_pointer
        initial_residuals = self.initial_residuals_pointer
        initial_variances = self.initial_variances_pointer
        n = self.data.size
        density = self.model.density_code

        if density in (0, 1, 2, 3):
            family = "arma_gjr_garch" if self.model.leverage else "arma_garch"
            suffix = {
                0: "normal", 1: "studentt", 2: "skewt", 3: "ged"
            }[density]
            value_function = self._bind(
                f"{family}_nll_11_{suffix}",
                [Pointer, Pointer, Pointer, Pointer,
                 ctypes.c_double, ctypes.c_size_t],
                ctypes.c_double,
            )
            first_function = self._bind(
                f"{family}_nll_grad_11_{suffix}",
                [Pointer, Pointer, Pointer, Pointer, Pointer,
                 ctypes.c_double, ctypes.c_size_t],
                ctypes.c_double,
            )
            second_function = self._bind(
                f"{family}_hess_11_{suffix}",
                [Pointer, Pointer, Pointer, Pointer, Pointer,
                 ctypes.c_double, ctypes.c_size_t],
                None,
            )
            return {
                1: lambda: value_function(
                    theta, data, residuals, variance,
                    self.initial_variance, n,
                ),
                2: lambda: first_function(
                    theta, data, residuals, variance, gradient,
                    self.initial_variance, n,
                ),
                4: lambda: second_function(
                    theta, data, residuals, variance, hessian,
                    self.initial_variance, n,
                ),
            }

        fixed_lambda = {
            4: np.nan,
            5: -0.5,
            6: 1.0,
            7: np.inf,
        }[density]
        value_function = self._bind(
            "arma_vol_gh_nll_pq",
            [Pointer, Pointer, Pointer, Pointer, Pointer, Pointer,
             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
             ctypes.c_double],
            ctypes.c_double,
        )
        derivative_arguments = [
            Pointer, Pointer, Pointer, Pointer, Pointer, Pointer, Pointer,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_double,
        ]
        first_function = self._bind(
            "arma_vol_gh_nll_grad_pq", derivative_arguments, None
        )
        second_function = self._bind(
            "arma_vol_gh_nll_hess_pq", derivative_arguments, None
        )
        order_arguments = (
            n, self.model.p_ar, self.model.q_ma,
            self.model.p_vol, self.model.q_vol,
            1 if self.model.leverage else 0,
            fixed_lambda,
        )
        return {
            1: lambda: value_function(
                theta, data, residuals, variance,
                initial_residuals, initial_variances, *order_arguments,
            ),
            2: lambda: first_function(
                theta, data, residuals, variance,
                initial_residuals, initial_variances, gradient,
                *order_arguments,
            ),
            4: lambda: second_function(
                theta, data, residuals, variance,
                initial_residuals, initial_variances, hessian,
                *order_arguments,
            ),
        }

    def __call__(self, request: int) -> object:
        return self._requests[request]()


def simulate_sample(observations: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(20260828)
    innovations = rng.standard_t(8.0, size=observations) * np.sqrt(6.0 / 8.0)
    data = np.empty(observations, dtype=np.float64)
    variance = 1.0
    previous = 0.0
    for time_index, innovation in enumerate(innovations):
        variance = 0.04 + 0.07 * previous * previous + 0.89 * variance
        data[time_index] = np.sqrt(variance) * innovation
        previous = data[time_index]
    return np.ascontiguousarray(data)


def simulate_arma_egarch_sample(observations: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(20260829)
    burn = 400
    total = observations + burn
    degrees_of_freedom = 7.0
    innovations = (
        rng.standard_t(degrees_of_freedom, total)
        * np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    )
    expected_absolute = (
        2.0
        * np.sqrt(degrees_of_freedom - 2.0)
        * np.exp(
            math.lgamma(0.5 * (degrees_of_freedom + 1.0))
            - math.lgamma(0.5 * degrees_of_freedom)
        )
        / ((degrees_of_freedom - 1.0) * np.sqrt(np.pi))
    )
    log_variance = np.empty(total, dtype=np.float64)
    residuals = np.empty(total, dtype=np.float64)
    data = np.empty(total, dtype=np.float64)
    log_variance[0] = -0.12 / (1.0 - 0.88)
    residuals[0] = 0.0
    data[0] = 0.001
    for time_index in range(1, total):
        standardized = residuals[time_index - 1] * np.exp(
            -0.5 * log_variance[time_index - 1]
        )
        log_variance[time_index] = (
            -0.12
            + 0.16 * (abs(standardized) - expected_absolute)
            - 0.08 * standardized
            + 0.88 * log_variance[time_index - 1]
        )
        residuals[time_index] = (
            np.exp(0.5 * log_variance[time_index]) * innovations[time_index]
        )
        data[time_index] = (
            0.001
            + 0.22 * data[time_index - 1]
            - 0.12 * residuals[time_index - 1]
            + residuals[time_index]
        )
    return np.ascontiguousarray(data[burn:])


def benchmark_cases(family: str) -> tuple[BenchmarkCase, ...]:
    model_kinds = {
        "garch": ("GARCH(2,1)", "ARMA(1,1)+GARCH(1,1)"),
        "gjr": ("GJR-GARCH(2,1)", "ARMA(1,1)+GJR-GARCH(1,1)"),
        "egarch": ("EGARCH(2,1)", "ARMA(1,1)+EGARCH(1,1)"),
        "arma-egarch": ("ARMA(1,1)+EGARCH(1,1)",),
        "all": (
            "GARCH(2,1)",
            "ARMA(1,1)+GARCH(1,1)",
            "GJR-GARCH(2,1)",
            "ARMA(1,1)+GJR-GARCH(1,1)",
            "EGARCH(2,1)",
            "ARMA(1,1)+EGARCH(1,1)",
        ),
    }[family]
    return tuple(
        BenchmarkCase(model_kind, density)
        for density in DENSITIES
        for model_kind in model_kinds
    )


def _average_call_microseconds(
    function: Callable[[], object], repetitions: int
) -> float:
    started = time.perf_counter_ns()
    for _ in range(repetitions):
        function()
    return (time.perf_counter_ns() - started) / repetitions / 1e3


def paired_median_call_microseconds(
    native: Callable[[], object], production: Callable[[], object]
) -> tuple[float, float]:
    native()
    production()
    initial_started = time.perf_counter_ns()
    native()
    native_ns = max(1, time.perf_counter_ns() - initial_started)
    initial_started = time.perf_counter_ns()
    production()
    production_ns = max(1, time.perf_counter_ns() - initial_started)
    initial_ns = max(native_ns, production_ns)
    repetitions = max(1, min(5000, int(40_000_000 / initial_ns)))
    samples = []
    for sample in range(9):
        if sample % 2 == 0:
            native_us = _average_call_microseconds(native, repetitions)
            production_us = _average_call_microseconds(
                production, repetitions
            )
        else:
            production_us = _average_call_microseconds(
                production, repetitions
            )
            native_us = _average_call_microseconds(native, repetitions)
        samples.append((native_us, production_us))
    median_ratio = float(np.median([
        native_us / production_us
        for native_us, production_us in samples
    ]))
    return min(
        samples,
        key=lambda sample: abs(sample[0] / sample[1] - median_ratio),
    )


def benchmark_requests(
    cases: Iterable[BenchmarkCase],
    data: NDArray[np.float64],
    baseline_data: NDArray[np.float64],
) -> list[RequestResult]:
    results = []
    for case in cases:
        spec = case.build()
        model = _describe_model(spec)
        theta, _ = _initial_theta(spec, model, data, None)
        native = NativePointEvaluator(model, data, theta)
        production = ProductionPointEvaluator(model, data, theta)
        baseline_theta, _ = _initial_theta(
            spec, model, baseline_data, None
        )
        baseline_native = NativePointEvaluator(
            model, baseline_data, baseline_theta
        )
        baseline_production = ProductionPointEvaluator(
            model, baseline_data, baseline_theta
        )
        for name, request in REQUESTS:
            native_us, production_us = paired_median_call_microseconds(
                lambda r=request: native(r),
                lambda r=request: production(r),
            )
            baseline_native_us, baseline_production_us = (
                paired_median_call_microseconds(
                    lambda r=request: baseline_native(r),
                    lambda r=request: baseline_production(r),
                )
            )
            results.append(RequestResult(
                case=case.label,
                request=name,
                native_us=native_us,
                production_us=production_us,
                baseline_native_us=baseline_native_us,
                baseline_production_us=baseline_production_us,
            ))
    return results


def timed_fit(function: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def paired_median_fits(
    native: Callable[[], object],
    reference: Callable[[], object],
) -> tuple[object, float, object, float]:
    samples = []
    for sample in range(7):
        if sample % 2 == 0:
            native_result, native_seconds = timed_fit(native)
            reference_result, reference_seconds = timed_fit(reference)
        else:
            reference_result, reference_seconds = timed_fit(reference)
            native_result, native_seconds = timed_fit(native)
        samples.append((
            native_result,
            native_seconds,
            reference_result,
            reference_seconds,
        ))
    median_ratio = float(np.median([
        native_seconds / reference_seconds
        for _, native_seconds, _, reference_seconds in samples
    ]))
    return min(
        samples,
        key=lambda sample: abs(
            sample[1] / sample[3] - median_ratio
        ),
    )


def warm_complete_fit_paths(
    case: BenchmarkCase,
    data: NDArray[np.float64],
) -> None:
    """Pay one-time Python, solver, and BLAS initialization before timing."""
    if case.is_joint and case.is_egarch:
        data = simulate_arma_egarch_sample(data.size)
    spec = case.build()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    spec.fit(data, solver="native", initial_params=start, maxiter=1)
    if case.leverage:
        independent_coordinate_fit(spec, model, data, start, 1)
    else:
        spec.fit(
            data, solver="legacy", log_mode=True,
            initial_params=start, maxiter=1,
        )


def independent_coordinate_fit(
    spec: object,
    model: NativeModel,
    data: NDArray[np.float64],
    start: NDArray[np.float64],
    maximum_iterations: int,
) -> IndependentFit:
    """Fit the identical jet and chart through an independent optimizer."""
    from scipy.optimize import minimize

    coordinates = _theta_to_coordinates(model, start)
    lower, upper = _coordinate_bounds(model)
    bounds = tuple(
        (
            None if np.isnan(lower_value) else float(lower_value),
            None if np.isnan(upper_value) else float(upper_value),
        )
        for lower_value, upper_value in zip(lower, upper)
    )
    scale = 1.0 / float(model.scored_observations(data.size))

    def value_and_gradient(
        point: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        point = np.ascontiguousarray(point, dtype=np.float64)
        value, gradient, _ = _evaluate_coordinate_request(
            model, data, point, request=3
        )
        return value * scale, gradient * scale

    result = minimize(
        value_and_gradient,
        coordinates,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": maximum_iterations,
            "ftol": 1e-14,
            "gtol": 1e-9,
            "maxls": 50,
        },
    )
    theta = _coordinates_to_theta(
        model, np.ascontiguousarray(result.x, dtype=np.float64)
    )
    _evaluate_theta_request(model, data, theta, request=3)
    _evaluate_theta_request(model, data, theta, request=4)
    spec.loglikelihood(data, theta)
    return IndependentFit(
        params=theta,
        nfev=int(result.nfev),
        njev=int(result.njev),
        success=bool(result.success),
    )


def function_counts(result: object) -> tuple[int, int]:
    if isinstance(result, IndependentFit):
        return result.nfev, result.njev
    fit_info = result.fit_info
    if fit_info.function_evaluations is not None:
        return int(fit_info.function_evaluations), int(fit_info.gradient_evaluations)
    optimization = result.optimization_result
    return int(getattr(optimization, "nfev", 0)), int(getattr(optimization, "njev", 0))


def assess(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[float, bool, float]:
    value, gradient, _ = _evaluate_theta_request(model, data, theta, request=3)
    assessment = _assess_theta_solution(
        model, theta, gradient, data, 1.0e-10, boundary_reached=False
    )
    return value, assessment.model_feasible, assessment.kkt_residual


def benchmark_fits(
    cases: Iterable[BenchmarkCase],
    data: NDArray[np.float64],
    maximum_iterations: int,
) -> list[FitResult]:
    results = []
    for case in cases:
        case_data = (
            simulate_arma_egarch_sample(data.size)
            if case.is_joint and case.is_egarch
            else data
        )
        spec = case.build()
        model = _describe_model(spec)
        start, _ = _initial_theta(spec, model, case_data, None)

        def native_fit():
            return spec.fit(
                case_data, solver="native", initial_params=start,
                maxiter=maximum_iterations,
            )

        if case.leverage:
            reference_name = "SciPy L-BFGS-B / identical native jet"

            def reference_fit():
                return independent_coordinate_fit(
                    spec, model, case_data, start, maximum_iterations
                )
        else:
            reference_name = "established SciPy adapter"

            def reference_fit():
                return spec.fit(
                    case_data, solver="legacy", log_mode=True,
                    initial_params=start, maxiter=maximum_iterations,
                )

        native, native_seconds, reference, reference_seconds = (
            paired_median_fits(native_fit, reference_fit)
        )
        native_theta = np.ascontiguousarray(native.params, dtype=np.float64)
        reference_theta = np.ascontiguousarray(
            reference.params, dtype=np.float64
        )
        native_at_native, native_feasible, native_kkt = assess(
            model, case_data, native_theta
        )
        native_at_reference, reference_feasible, reference_kkt = assess(
            model, case_data, reference_theta
        )
        established_at_native = -float(
            spec.loglikelihood(case_data, native_theta)
        )
        established_at_reference = -float(
            spec.loglikelihood(case_data, reference_theta)
        )
        native_nfev, native_njev = function_counts(native)
        reference_nfev, reference_njev = function_counts(reference)
        results.append(FitResult(
            case=case.label,
            reference=reference_name,
            reference_success=(
                reference.success
                if isinstance(reference, IndependentFit)
                else reference.converged
            ),
            native_seconds=native_seconds,
            reference_seconds=reference_seconds,
            native_nfev=native_nfev,
            native_njev=native_njev,
            reference_nfev=reference_nfev,
            reference_njev=reference_njev,
            native_at_native=native_at_native,
            established_at_native=established_at_native,
            native_at_reference=native_at_reference,
            established_at_reference=established_at_reference,
            native_feasible=native_feasible,
            reference_feasible=reference_feasible,
            native_kkt=native_kkt,
            reference_kkt=reference_kkt,
        ))
    return results


def print_request_results(
    results: Iterable[RequestResult], observations: int, baseline: int
) -> None:
    print(
        "\nSparse requests "
        f"(n={observations}; baseline={baseline}; microseconds)"
    )
    print(
        f"{'case':45} {'req':>3} {'native':>10} {'production':>12} "
        f"{'ratio':>8} {'marginal':>9}"
    )
    for result in results:
        print(
            f"{result.case:45} {result.request:>3} "
            f"{result.native_us:10.2f} {result.production_us:12.2f} "
            f"{result.ratio:8.3f} {result.marginal_ratio:9.3f}"
        )


def print_fit_results(results: Iterable[FitResult], observations: int) -> None:
    print(
        f"\nMatched-start fits (n={observations}; ratio=native/reference)"
    )
    print(
        f"{'case':45} {'native':>8} {'ref':>8} {'ratio':>7} "
        f"{'nF/nG':>11} {'rF/rG':>11} {'native KKT':>12} {'ref KKT':>12}"
    )
    for result in results:
        print(
            f"{result.case:45} {result.native_seconds:8.4f} "
            f"{result.reference_seconds:8.4f} {result.ratio:7.3f} "
            f"{result.native_nfev:4}/{result.native_njev:<4} "
            f"{result.reference_nfev:4}/{result.reference_njev:<4} "
            f"{result.native_kkt:12.3e} {result.reference_kkt:12.3e}"
        )
        print(
            f"  reference={result.reference}; cross NLL: "
            f"native@native={result.native_at_native:.9g}, "
            f"established@native={result.established_at_native:.9g}, "
            f"native@reference={result.native_at_reference:.9g}, "
            f"established@reference={result.established_at_reference:.9g}; "
            f"feasible={result.native_feasible}/{result.reference_feasible}"
        )


def enforce_gate(
    requests: Iterable[RequestResult],
    fits: Iterable[FitResult],
) -> None:
    failures = []
    failures.extend(
        f"{result.case} {result.request} ratio={result.ratio:.3f}"
        for result in requests
        if result.ratio > 1.05
    )
    failures.extend(
        f"{result.case} {result.request} "
        f"marginal ratio={result.marginal_ratio:.3f}"
        for result in requests
        if result.marginal_ratio > 1.05
    )
    for result in fits:
        tolerance = 1.0e-8 * (1.0 + abs(result.established_at_reference))
        native_parity = abs(
            result.native_at_native - result.established_at_native
        )
        reference_parity = abs(
            result.native_at_reference - result.established_at_reference
        )
        if result.reference == "established SciPy adapter" and result.ratio > 1.05:
            failures.append(f"{result.case} fit ratio={result.ratio:.3f}")
        if not result.native_feasible:
            failures.append(f"{result.case} native estimate is infeasible")
        if result.native_kkt > 1.0e-5:
            failures.append(
                f"{result.case} native KKT={result.native_kkt:.3e}"
            )
        if result.reference.startswith("SciPy L-BFGS-B"):
            if not result.reference_success or not result.reference_feasible:
                failures.append(
                    f"{result.case} independent reference is not credible"
                )
            if result.reference_kkt > 1.0e-5:
                failures.append(
                    f"{result.case} reference KKT={result.reference_kkt:.3e}"
                )
        if native_parity > tolerance or reference_parity > tolerance:
            failures.append(f"{result.case} matched-point evaluator mismatch")
        if result.native_at_native > result.native_at_reference + tolerance:
            failures.append(f"{result.case} native estimate has worse common NLL")
    if failures:
        raise SystemExit("Native promotion gate failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("garch", "gjr", "egarch", "arma-egarch", "all"),
        default="garch",
    )
    parser.add_argument("--observations", type=int, default=600)
    parser.add_argument("--request-observations", type=int, default=5000)
    parser.add_argument("--request-baseline-observations", type=int, default=500)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--requests-only", action="store_true")
    parser.add_argument("--fits-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.requests_only and args.fits_only:
        parser.error("--requests-only and --fits-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cases = benchmark_cases(args.family)
    request_data = simulate_sample(args.request_observations)
    baseline_data = simulate_sample(args.request_baseline_observations)
    request_cases = tuple(
        case for case in cases if case.supports_request_benchmark
    )
    request_results = [] if args.fits_only else benchmark_requests(
        request_cases, request_data, baseline_data
    )
    fit_data = simulate_sample(args.observations)
    fit_cases = tuple(
        case for case in cases if case.supports_matched_fit
    )
    if fit_cases and not args.requests_only:
        warm_complete_fit_paths(fit_cases[0], fit_data)
    fit_results = [] if args.requests_only else benchmark_fits(
        fit_cases, fit_data, args.maxiter
    )
    if request_results:
        print_request_results(
            request_results,
            args.request_observations,
            args.request_baseline_observations,
        )
    if fit_results:
        print_fit_results(fit_results, args.observations)
    print(f"\nElapsed: {time.perf_counter() - started:.2f}s")
    if args.check:
        enforce_gate(request_results, fit_results)


if __name__ == "__main__":
    main()
