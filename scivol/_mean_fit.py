"""Canonical constant-variance mean models and independent optimization paths."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, minimize

from . import _native_core
from .components.mean import ARMA, ARX, HARX
from .components.density import (
    Normal,
    StudentT,
    SkewT,
    GED,
    GeneralizedHyperbolic,
    NormalInverseGaussian,
    Hyperbolic,
    VarianceGamma,
)
from .roles import Role
from ._validation import validate_data, validate_regressors
from ._kernels.linear_mean_normal_common import build_linear_mean_features

DENSITIES = (
    Normal,
    StudentT,
    SkewT,
    GED,
    GeneralizedHyperbolic,
    NormalInverseGaussian,
    Hyperbolic,
    VarianceGamma,
)
FloatArray = NDArray[np.float64]
if TYPE_CHECKING:
    from .result import EstimationResult
    from ._evaluation import FilteredState


def supports_mean(spec: Any) -> bool:
    return (
        isinstance(spec.get_component(Role.MEAN), (ARMA, ARX, HARX))
        and spec.get_component(Role.VOLATILITY) is None
        and type(spec.get_component(Role.DENSITY)) in DENSITIES
    )


def _pointer(array: np.ndarray | None) -> int:
    return 0 if array is None else array.ctypes.data


def _workspace(size: int) -> NDArray[np.uint8]:
    if size <= 0:
        raise ValueError("Invalid or overflowing mean-model workspace.")
    return np.empty(int(size), dtype=np.uint8)


@dataclass
class StabilityBlock:
    offset: int
    basis: np.ndarray

    def __post_init__(self) -> None:
        self.basis = np.ascontiguousarray(self.basis, dtype=np.float64)
        self.order, self.width = self.basis.shape
        self.workspace = _workspace(
            _native_core._mean_stability_size(
                _pointer(self.basis), self.order, self.width
            )
        )
        self.values = np.empty(2 * self.order)
        self.jacobian = np.empty((2 * self.order, self.width))

    def evaluate(
        self, theta: FloatArray, *, derivatives: bool = True, stable: bool = False
    ) -> tuple[FloatArray, FloatArray | None]:
        point = np.ascontiguousarray(theta[self.offset : self.offset + self.width])
        valid = _native_core._mean_stability(
            _pointer(self.basis),
            self.order,
            self.width,
            _pointer(point),
            _pointer(self.values),
            _pointer(self.jacobian) if derivatives else 0,
            _pointer(self.workspace),
            self.workspace.nbytes,
            stable,
        )
        if not valid:
            raise ValueError(
                "Mean polynomial is outside the requested stability domain."
            )
        return self.values.copy(), self.jacobian.copy() if derivatives else None


class PreparedMean:
    """Sample, design and stability restrictions shared by joint and mean-only fits."""

    def __init__(self, spec: Any, data: Any, x: Any = None) -> None:
        self.spec = deepcopy(spec)
        self.mean = self.spec.get_component(Role.MEAN)
        self.density = self.spec.get_component(Role.DENSITY)
        if (
            not isinstance(self.mean, (ARMA, ARX, HARX))
            or type(self.density) not in DENSITIES
        ):
            raise NotImplementedError(
                "Expected ARMA, ARX or HARX with a supported density."
            )
        self.data = np.array(
            validate_data(data), dtype=np.float64, order="C", copy=True
        )
        self.data.flags.writeable = False
        self.code = DENSITIES.index(type(self.density))
        self.blocks = []
        self.har_basis = None
        if isinstance(self.mean, ARMA):
            if x is not None:
                raise ValueError("ARMA does not accept exogenous regressors.")
            self.p, self.q = self.mean.p, self.mean.q
            if any(
                not isinstance(v, (int, np.integer)) or v < 0 for v in (self.p, self.q)
            ):
                raise ValueError("ARMA orders must be non-negative integers.")
            self.columns, self.first = 1 + self.p, max(1, self.p, self.q)
            self.features = None
            self.ar_offset = 1
            if self.p:
                self.blocks.append(StabilityBlock(1, np.eye(self.p)))
            if self.q:
                self.blocks.append(StabilityBlock(self.columns, -np.eye(self.q)))
        else:
            x = None if x is None else validate_regressors(x, len(self.data))
            self.mean.set_n_exog(0 if x is None else x.shape[1])
            self.columns, self.q, self.first = self.mean.n_params, 0, 0
            self.features = build_linear_mean_features(self.mean, self.data, x)
            if not np.isfinite(self.features).all():
                raise ValueError("The prepared mean design contains non-finite values.")
            self.features.flags.writeable = False
            self.ar_offset = int(self.mean.constant)
            self.p = self.mean.lags if isinstance(self.mean, ARX) else 0
            if self.p:
                self.blocks.append(StabilityBlock(self.ar_offset, np.eye(self.p)))
            if isinstance(self.mean, HARX):
                horizons = np.asarray(self.mean.horizons)
                self.har_basis = np.ascontiguousarray(
                    (np.arange(1, max(horizons) + 1)[:, None] <= horizons) / horizons
                )
                self.blocks.append(StabilityBlock(self.ar_offset, self.har_basis))
        if self.first >= len(self.data):
            raise ValueError(
                "Mean model requires observations beyond its conditioning lags."
            )
        largest = np.max(np.abs(self.data))
        self.data_scale = (
            float(largest * np.sqrt(np.mean((self.data / largest) ** 2)))
            if largest
            else 1.0
        )
        self.mean_units = np.ones(self.columns + self.q)
        if isinstance(self.mean, ARMA) or self.mean.constant:
            self.mean_units[0] = self.data_scale
        if self.features is not None and self.mean.n_exog:
            exog = self.features[:, -self.mean.n_exog :]
            magnitude = np.max(np.abs(exog), axis=0)
            magnitude[magnitude == 0] = 1.0
            rms = magnitude * np.sqrt(np.mean((exog / magnitude) ** 2, axis=0))
            rms[rms == 0] = 1.0
            self.mean_units[self.columns - self.mean.n_exog : self.columns] = (
                self.data_scale / rms
            )


class MeanModel(PreparedMean):
    """Constant-variance jet and workspaces for a prepared mean model."""

    def __init__(
        self, spec: Any, data: Any, x: Any = None, *, fixed: bool = True
    ) -> None:
        if not supports_mean(spec):
            raise NotImplementedError("Expected a standalone ARMA, ARX or HARX model.")
        super().__init__(spec, data, x)
        self.request_counts = {"F": 0, "G": 0, "H": 0}
        self.phase_diagnostics = []
        self.width = self.columns + self.q + 1
        self.size = self.width + self.density.n_params
        self.n_eff = len(self.data) - self.first
        har_order, har_width = (
            (0, 0) if self.har_basis is None else self.har_basis.shape
        )
        self.descriptor = (
            len(self.data),
            self.columns,
            self.q,
            self.first,
            self.ar_offset,
            self.p,
            self.code,
            self.features is not None,
            bool(fixed),
            har_order,
            har_width,
            self.ar_offset,
        )
        sizes = _native_core._mean_sizes(self.descriptor, _pointer(self.har_basis))
        self.kernel_workspace, self.coordinate_workspace, self.solve_workspace = map(
            _workspace, sizes
        )
        self.lower, self.upper = np.empty(self.size), np.empty(self.size)
        _native_core._mean_bounds(
            self.descriptor, _pointer(self.lower), _pointer(self.upper)
        )
        self.units = np.ones(self.size)
        self.units[: self.width - 1] = self.mean_units
        self.units[self.width - 1] = self.data_scale * self.data_scale
        self.value, self.gradient, self.hessian = (
            np.empty(1),
            np.empty(self.size),
            np.empty((self.size, self.size)),
        )

    def evaluate(
        self,
        theta: FloatArray,
        request: int = 7,
        *,
        coordinates: bool = False,
        residuals: FloatArray | None = None,
    ) -> tuple[float | None, FloatArray | None, FloatArray | None]:
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (self.size,):
            raise ValueError(
                f"Expected {self.size} mean, variance and density parameters."
            )
        if residuals is not None and (
            residuals.shape != self.data.shape
            or residuals.dtype != np.float64
            or not residuals.flags.c_contiguous
            or not residuals.flags.writeable
        ):
            raise ValueError(
                "Residual output must be a writable contiguous float64 sample-length array."
            )
        if residuals is not None and any(
            np.shares_memory(residuals, source)
            for source in (theta, self.data, self.features)
            if source is not None
        ):
            raise ValueError(
                "Residual output must not alias parameters or sample inputs."
            )
        arena = self.coordinate_workspace if coordinates else self.kernel_workspace
        for bit, name in ((1, "F"), (2, "G"), (4, "H")):
            if request & bit:
                self.request_counts[name] += 1
        valid = _native_core._mean_jet(
            self.descriptor,
            _pointer(theta),
            _pointer(self.data),
            _pointer(self.features),
            request,
            coordinates,
            _pointer(arena),
            arena.nbytes,
            _pointer(self.value) if request & 1 else 0,
            _pointer(self.gradient) if request & 2 else 0,
            _pointer(self.hessian) if request & 4 else 0,
            _pointer(residuals),
            False,
        )
        if not valid:
            raise ValueError(
                "Requested mean jet is outside its numerical or differentiability domain."
            )
        return (
            float(self.value[0]) if request & 1 else None,
            self.gradient.copy() if request & 2 else None,
            self.hessian.copy() if request & 4 else None,
        )

    def map(self, point: FloatArray, *, forward: bool) -> FloatArray:
        point = np.ascontiguousarray(point, dtype=np.float64)
        if point.shape != (self.size,) or not np.isfinite(point).all():
            raise ValueError("Invalid mean-model parameter vector.")
        result = np.empty(self.size)
        arena = self.coordinate_workspace
        if not _native_core._mean_map(
            self.descriptor,
            forward,
            _pointer(point),
            _pointer(result),
            _pointer(arena),
            arena.nbytes,
        ):
            raise ValueError(
                "Parameters are outside the invertible mean coordinate chart."
            )
        return result

    def constraints(self, theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        values, rows = [], []
        for block in self.blocks:
            value, jacobian = block.evaluate(theta)
            full = np.zeros((len(value), self.size))
            full[:, block.offset : block.offset + block.width] = jacobian
            values.append(value)
            rows.append(full)
        return (
            (np.concatenate(values), np.vstack(rows))
            if values
            else (np.empty(0), np.empty((0, self.size)))
        )

    def feasible(self, theta: FloatArray) -> bool:
        if (
            not np.isfinite(theta).all()
            or np.any(theta < self.lower)
            or np.any(theta > self.upper)
        ):
            return False
        try:
            return all(
                np.all(block.evaluate(theta, derivatives=False, stable=True)[0] > 0)
                for block in self.blocks
            )
        except ValueError:
            return False

    def initialize(self, initial_params: FloatArray | None) -> FloatArray:
        if initial_params is not None:
            point = np.asarray(initial_params, dtype=np.float64)
            if self.code == 0 and point.shape == (self.size - 1,):
                point = np.r_[point, 1.0]
                residuals = np.empty_like(self.data)
                self.evaluate(point, 1, residuals=residuals)
                point[-1] = np.mean(residuals[self.first :] ** 2)
            if point.shape != (self.size,) or not self.feasible(point):
                raise ValueError(
                    "initial_params must satisfy the declared mean and density domain."
                )
            return point.copy()
        design = self.features
        if design is None:
            design = np.column_stack(
                [np.ones(self.n_eff)]
                + [
                    self.data[self.first - lag : len(self.data) - lag]
                    for lag in range(1, self.p + 1)
                ]
            )
            response = self.data[self.first :]
        else:
            response = self.data
        beta = (
            np.linalg.lstsq(design, response, rcond=None)[0]
            if self.columns
            else np.empty(0)
        )
        shape = np.asarray(self.density.default_start(self.data), dtype=np.float64)
        point = np.r_[beta, np.zeros(self.q), 1.0, shape]
        # Initialization chooses a feasible start; it never changes a fitted point.
        for block in self.blocks:
            try:
                block.evaluate(point, derivatives=False, stable=True)
            except ValueError:
                point[block.offset : block.offset + block.width] = 0.0
        residuals = np.empty_like(self.data)
        self.evaluate(point, 1, residuals=residuals)
        point[self.width - 1] = np.mean(residuals[self.first :] ** 2)
        if not self.feasible(point):
            raise ValueError(
                "No positive finite innovation variance at the initial mean."
            )
        return point

    def describe_domain(self) -> dict[str, Any]:
        return {
            "mean": "stationary implied AR polynomial; invertible MA polynomial",
            "variance": "sigma2 > 0",
            "conditioning": "zero residual history; supplied linear design",
            "first_scored": self.first,
            "numerical_bound_units": "internal scaled theta; see internal_data_scale",
            "numerical_lower": [
                None if not np.isfinite(v) else float(v) for v in self.lower
            ],
            "numerical_upper": [
                None if not np.isfinite(v) else float(v) for v in self.upper
            ],
            "density_support": "existing validated density support, not convergence padding",
            "kkt_metric": "original parameters in dimensionless measurement units",
        }


def assess_solution(
    model: MeanModel, theta: FloatArray, gradient: FloatArray
) -> tuple[bool, float, float]:
    """Project the theta gradient onto the local feasible set, without activity cutoffs."""
    if not model.feasible(theta):
        return False, np.inf, np.inf
    gradient = gradient * model.units / model.n_eff
    theta_units = theta / model.units
    lower, upper = model.lower / model.units, model.upper / model.units
    step = np.clip(theta_units - gradient, lower, upper) - theta_units
    values, jacobian = model.constraints(theta)
    jacobian = jacobian * model.units
    if len(values) and np.any(values + jacobian @ step < 0):
        projection = minimize(
            lambda d: 0.5 * np.dot(d + gradient, d + gradient),
            step,
            jac=lambda d: d + gradient,
            method="SLSQP",
            bounds=Bounds(lower - theta_units, upper - theta_units),
            constraints=[LinearConstraint(jacobian, -values, np.inf)],
            options={"ftol": 1e-14, "maxiter": 100},
        )
        if not projection.success:
            return True, np.inf, 0.0
        step = projection.x
    return True, float(np.max(np.abs(step), initial=0.0)), 0.0


def _solve_native_phase(
    model: MeanModel,
    initial: FloatArray,
    accuracy: float,
    maxiter: int,
    *,
    curvature_scaling: bool = False,
    transformed: bool = True,
) -> OptimizeResult:
    started = time.perf_counter()
    z = model.map(initial, forward=False) if transformed else initial.copy()
    theta, gradient = np.empty(model.size), np.empty(model.size)
    arena = model.solve_workspace
    scales = model.units.copy()
    if transformed:
        jacobian = np.empty((model.size, model.size))
        if not _native_core._mean_map(
            model.descriptor,
            True,
            _pointer(z),
            _pointer(theta),
            _pointer(model.coordinate_workspace),
            model.coordinate_workspace.nbytes,
            _pointer(jacobian),
        ):
            raise ValueError("Initial coordinate map is not differentiable.")
        sensitivity = np.linalg.norm(jacobian / model.units[:, None], axis=0)
        scales = 1 / np.maximum(sensitivity, np.sqrt(np.finfo(float).eps))
    if curvature_scaling:
        try:
            hessian = model.evaluate(z, 4, coordinates=transformed)[2] / model.n_eff
            scales = 1 / np.sqrt(
                np.maximum(np.abs(np.diag(hessian)), np.finfo(float).eps)
            )
        except ValueError:
            pass
    raw = _native_core._mean_fit(
        model.descriptor,
        _pointer(model.data),
        _pointer(model.features),
        _pointer(z),
        _pointer(arena),
        arena.nbytes,
        _pointer(theta),
        _pointer(gradient),
        _pointer(model.har_basis),
        _pointer(scales),
        maxiter,
        accuracy,
        transformed,
    )
    model.request_counts["F"] += raw[3]
    model.request_counts["G"] += raw[4]
    model.phase_diagnostics.append(
        {
            "space": "z" if transformed else "theta",
            "accuracy": accuracy,
            "curvature_scaling": curvature_scaling,
            "elapsed": time.perf_counter() - started,
            "iterations": raw[2],
            "F": raw[3],
            "G": raw[4],
            "mode": raw[1],
        }
    )
    value = raw[0]
    if (
        not np.isfinite(theta).all()
        or not np.isfinite(gradient).all()
        or not np.isfinite(value)
    ):
        theta = initial.copy()
        value, gradient, _ = model.evaluate(theta, 3)
    return OptimizeResult(
        x=theta,
        fun=value,
        status=raw[1],
        success=raw[1] == 0,
        nit=raw[2],
        nfev=raw[3],
        njev=raw[4],
        jac=gradient,
    )


def solve_native(
    model: MeanModel,
    initial: FloatArray,
    accuracy: float,
    maxiter: int,
    *,
    transformed: bool = True,
) -> OptimizeResult:
    budget = max(1, int(0.6 * maxiter))
    result = _solve_native_phase(
        model,
        initial,
        max(np.finfo(float).eps, accuracy * 0.01),
        budget,
        transformed=transformed,
    )
    space = "z" if transformed else "theta"
    phases = [f"native-{space}-slsqp"]
    modes = [int(result.status)]
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    feasible, kkt, _ = assess_solution(model, result.x, result.jac)
    for refinement in range(2):
        remaining = maxiter - result.nit
        if not feasible or remaining <= 0 or (result.success and kkt <= tolerance):
            break
        budget = max(1, remaining // 2) if refinement == 0 else remaining
        phase_accuracy = max(np.finfo(float).eps, accuracy * 0.01 ** (refinement + 2))
        refined = _solve_native_phase(
            model,
            result.x,
            phase_accuracy,
            budget,
            curvature_scaling=True,
            transformed=transformed,
        )
        phases.append(f"native-{space}-curvature-rescaled-slsqp")
        modes.append(int(refined.status))
        total = (
            result.nit + refined.nit,
            result.nfev + refined.nfev,
            result.njev + refined.njev,
        )
        if model.feasible(refined.x) and refined.fun <= result.fun:
            result = refined
        result.nit, result.nfev, result.njev = total
        feasible, kkt, _ = assess_solution(model, result.x, result.jac)
    result.phases, result.modes = tuple(phases), tuple(modes)
    return result


def solve_scipy(
    model: MeanModel,
    initial: FloatArray,
    accuracy: float,
    maxiter: int,
    *,
    transformed: bool,
) -> OptimizeResult:
    start = model.map(initial, forward=False) if transformed else initial
    lower, upper = model.lower, model.upper
    if transformed:
        lower, upper = np.empty(model.size), np.empty(model.size)
        _native_core._mean_bounds(
            model.descriptor, _pointer(lower), _pointer(upper), True
        )
    initial_value = model.evaluate(initial, 1)[0] / model.n_eff
    best = [initial_value, start.copy()]
    rejection = initial_value + max(1.0, abs(initial_value)) * 1e6

    def objective(point):
        try:
            theta = model.map(point, forward=True) if transformed else point
            if not model.feasible(theta):
                raise ValueError("Infeasible optimizer trial")
            value = model.evaluate(point, 1, coordinates=transformed)[0] / model.n_eff
        except ValueError:
            delta = point - best[1]
            return rejection + float(delta @ delta)
        if value < best[0]:
            best[:] = [value, point.copy()]
        return value

    def gradient(point):
        try:
            theta = model.map(point, forward=True) if transformed else point
            if not model.feasible(theta):
                raise ValueError("Infeasible optimizer trial")
            return model.evaluate(point, 2, coordinates=transformed)[1] / model.n_eff
        except ValueError:
            return 2 * (point - best[1])

    constraints = []
    if model.blocks:
        # Reflection charts encode AR/MA restrictions; tied HAR coefficients stay explicit.
        blocks = (
            model.blocks
            if not transformed
            else (model.blocks if model.har_basis is not None else [])
        )
        for block in blocks:

            def fun(point, block=block):
                try:
                    return block.evaluate(point)[0]
                except ValueError:
                    return np.full(2 * block.order, -1.0)

            def jac(point, block=block):
                full = np.zeros((2 * block.order, model.size))
                try:
                    full[:, block.offset : block.offset + block.width] = block.evaluate(
                        point
                    )[1]
                except ValueError:
                    pass
                return full

            constraints.append({"type": "ineq", "fun": fun, "jac": jac})
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"ftol": max(np.finfo(float).eps, accuracy * 0.01), "maxiter": maxiter},
    )
    for _ in range(2):
        try:
            point = model.map(result.x, forward=True) if transformed else result.x
            _, theta_gradient, _ = model.evaluate(point, 2)
            feasible, kkt, _ = assess_solution(model, point, theta_gradient)
        except ValueError:
            break
        remaining = maxiter - result.nit
        if (
            not feasible
            or kkt <= max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
            or remaining <= 0
        ):
            break
        refined = minimize(
            objective,
            result.x,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"ftol": np.finfo(float).eps, "maxiter": remaining},
        )
        totals = (
            result.nit + refined.nit,
            result.nfev + refined.nfev,
            result.njev + refined.njev,
        )
        if refined.fun <= result.fun:
            result = refined
        result.nit, result.nfev, result.njev = totals
    try:
        point = model.map(result.x, forward=True) if transformed else result.x
        if not model.feasible(point):
            raise ValueError("Infeasible termination")
        result.fun, result.jac, _ = model.evaluate(point, 3)
        result.x = point
    except ValueError:
        result.x = model.map(best[1], forward=True) if transformed else best[1]
        result.fun, result.jac, _ = model.evaluate(result.x, 3)
        result.success = False
        result.message = "Solver returned an invalid point; reporting the best finite evaluated point."
    return result


def prepare_mean_fit(
    spec: Any, data: Any, x: Any, initial_params: FloatArray | None
) -> tuple[MeanModel, FloatArray, float, FloatArray, FloatArray]:
    data = np.ascontiguousarray(validate_data(data))
    largest = np.max(np.abs(data))
    scale = float(largest * np.sqrt(np.mean((data / largest) ** 2))) if largest else 1.0
    if not np.isfinite(scale * scale) or scale * scale == 0:
        raise ValueError(
            "Innovation variance is not representable in the supplied measurement units."
        )
    x_scales = None
    if x is not None:
        x = validate_regressors(x, len(data))
        largest_x = np.max(np.abs(x), axis=0)
        largest_x[largest_x == 0] = 1.0
        x_scales = largest_x * np.sqrt(np.mean((x / largest_x) ** 2, axis=0))
        x_scales[x_scales == 0] = 1.0
    model = MeanModel(spec, data / scale, None if x is None else x / x_scales)
    units = np.ones(model.size)
    if isinstance(model.mean, ARMA) or model.mean.constant:
        units[0] = scale
    if x_scales is not None and len(x_scales):
        units[model.columns - len(x_scales) : model.columns] = scale / x_scales
    units[model.width - 1] = scale * scale
    if initial_params is not None:
        initial_params = np.asarray(initial_params, dtype=np.float64)
        expected = (
            model.size - 1
            if model.code == 0 and initial_params.size == model.size - 1
            else model.size
        )
        if initial_params.shape != (expected,):
            raise ValueError("Wrong number of initial mean-model parameters.")
        initial_params = initial_params / units[:expected]
    return model, data, scale, units, model.initialize(initial_params)


def fit_mean_spec(
    spec: Any,
    data: Any,
    *,
    x: Any = None,
    solver: str = "native",
    log_mode: bool | None = None,
    initial_params: FloatArray | None = None,
    maxiter: int = 500,
    native_accuracy: float = 1e-12,
    verbose: bool = False,
    **kwargs: Any,
) -> "EstimationResult":
    if kwargs:
        raise TypeError(f"Unsupported mean fit options: {', '.join(sorted(kwargs))}")
    if (
        not np.isfinite(native_accuracy)
        or native_accuracy <= 0
        or int(maxiter) != maxiter
        or maxiter < 1
    ):
        raise ValueError("Optimizer accuracy and iteration budget must be positive.")
    if solver not in {"native", "slsqp", "legacy"}:
        raise NotImplementedError(
            "Standalone native means support native and independent SciPy SLSQP."
        )
    started = time.perf_counter()
    model, reported_data, data_scale, units, initial = prepare_mean_fit(
        spec, data, x, initial_params
    )
    transformed = (
        bool(log_mode)
        if log_mode is not None
        else (model.code < 4 or model.code == 7) and solver == "native"
    )
    result = (
        solve_native(
            model, initial, native_accuracy, int(maxiter), transformed=transformed
        )
        if solver == "native"
        else solve_scipy(
            model, initial, native_accuracy, int(maxiter), transformed=transformed
        )
    )
    return assemble_mean_fit(
        model,
        reported_data,
        data_scale,
        units,
        result,
        solver=solver,
        transformed=transformed,
        maxiter=maxiter,
        accuracy=native_accuracy,
        started=started,
    )


def assemble_mean_fit(
    model: MeanModel,
    reported_data: FloatArray,
    data_scale: float,
    units: FloatArray,
    result: OptimizeResult,
    *,
    solver: str,
    transformed: bool,
    maxiter: int,
    accuracy: float,
    started: float,
) -> "EstimationResult":
    from .result import EstimationResult
    from ._native_fit import _covariance_from_hessian

    theta = result.x
    residuals = np.empty_like(model.data)
    try:
        value, gradient, hessian = model.evaluate(theta, 7, residuals=residuals)
    except ValueError:
        value, gradient, _ = model.evaluate(theta, 3, residuals=residuals)
        hessian = None
    feasible, kkt, violation = assess_solution(model, theta, gradient)
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    credible = result.success and feasible and kkt <= tolerance
    sigma2 = np.full(len(model.data), theta[model.width - 1])
    if model.code == 0:
        # Exact nuisance profiling preserves the public Normal mean parameter layout.
        theta[-1] = np.mean(residuals[model.first :] ** 2)
        value, gradient, hessian = model.evaluate(theta, 7)
        feasible, kkt, violation = assess_solution(model, theta, gradient)
        credible = result.success and feasible and kkt <= tolerance
        sigma2.fill(theta[-1])
        hessian = (
            hessian[:-1, :-1]
            - np.outer(hessian[:-1, -1], hessian[-1, :-1]) / hessian[-1, -1]
        )
        result.x = theta[:-1].copy()
    public_units = units[:-1] if model.code == 0 else units
    result.x = result.x * public_units
    value += model.n_eff * np.log(data_scale)
    residuals *= data_scale
    sigma2 *= data_scale * data_scale
    information_warning = None
    if hessian is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            hessian = hessian / public_units[:, None] / public_units[None, :]
        if not np.isfinite(hessian).all():
            hessian = None
            information_warning = (
                "Observed information is not representable in the supplied measurement "
                "units; rescale the data for inference."
            )
    result.fun, result.success = value, bool(credible)
    result.jac = (gradient[:-1] if model.code == 0 else gradient) / public_units
    result.message = (
        "Converged with original-parameter KKT credibility."
        if credible
        else f"Optimizer mode {result.status}; theta KKT {kkt:.3g}, required {tolerance:.3g}."
    )
    model.mean.unpack(theta[: model.width - 1] * units[: model.width - 1])
    model.density.unpack(theta[model.width :])
    covariance = _covariance_from_hessian(hessian)
    fitted = EstimationResult(
        spec=model.spec,
        optimization_result=result,
        data=reported_data,
        sigma2=sigma2,
        hessian=hessian,
        cov_matrix=covariance,
        time_elapsed=time.perf_counter() - started,
        fit_info={
            "solver": "native-slsqp" if solver == "native" else "scipy-slsqp",
            "engine": "native-mean-jet",
            "log_mode": transformed,
            "optimization_space": "z-space" if transformed else "theta-space",
            "optimizer_mode": int(result.status),
            "optimizer_phases": getattr(result, "phases", ("scipy-slsqp",)),
            "optimizer_phase_modes": getattr(result, "modes", (int(result.status),)),
            "kkt_residual": kkt,
            "kkt_tolerance": tolerance,
            "maximum_constraint_violation": violation,
            "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev),
            "maxiter": int(maxiter),
            "jet_request_plan": "streaming F; F/G jointly prepared",
            "parameter_domain": model.describe_domain(),
            "likelihood_n_obs": model.n_eff,
            "profiled_parameters": int(model.code == 0),
            "internal_data_scale": data_scale,
            "evaluation_counts": model.request_counts.copy(),
            "phase_diagnostics": tuple(model.phase_diagnostics),
            "kernel_dispatch": _native_core._mean_dispatch_kind(model.descriptor),
        },
    )
    fitted._resid = residuals
    if not credible:
        fitted.fit_info.add_warning(result.message)
    if covariance is None:
        fitted.fit_info.add_warning(
            information_warning
            or "Observed information does not support reliable standard errors at this estimate."
        )
    return fitted


def filter_mean_spec(
    spec: Any,
    data: Any,
    theta: FloatArray,
    *,
    x: Any = None,
    compute_score: bool = False,
    compute_hessian: bool = False,
) -> "FilteredState":
    from ._evaluation import FilteredState

    model = MeanModel(spec, data, x)
    point = np.asarray(theta, dtype=np.float64)
    residuals = np.empty_like(model.data)
    if model.code == 0:
        point = np.r_[point, 1.0]
        model.evaluate(point, 1, residuals=residuals)
        point[-1] = np.mean(residuals[model.first :] ** 2)
    if not model.feasible(point):
        raise ValueError("Parameters violate the declared mean or density domain.")
    request = 1 | (2 if compute_score else 0) | (4 if compute_hessian else 0)
    value, gradient, hessian = model.evaluate(point, request, residuals=residuals)
    if model.code == 0:
        if gradient is not None:
            gradient = gradient[:-1]
        if hessian is not None:
            hessian = (
                hessian[:-1, :-1]
                - np.outer(hessian[:-1, -1], hessian[-1, :-1]) / hessian[-1, -1]
            )
    shape = point[model.width :]
    metadata = {}
    if model.code in (1, 2, 3):
        metadata["nu"] = float(shape[0])
        if model.code == 2:
            metadata["lam"] = float(shape[1])
    if model.code >= 4:
        metadata["rho"] = float(shape[0])
        if model.code != 7:
            metadata["zeta"] = float(shape[1])
        metadata["shape_lambda"] = (
            float(shape[-1])
            if model.code in (4, 7)
            else (-0.5 if model.code == 5 else 1.0)
        )
    return FilteredState(
        model.data,
        model.data - residuals,
        residuals,
        np.full(len(model.data), point[model.width - 1]),
        -value,
        None if gradient is None else -gradient,
        None if hessian is None else -hessian,
        model.density.signature,
        **metadata,
    )
