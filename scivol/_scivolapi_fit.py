"""Thin adapter between the public components and the native scivolapi library.

Translates a composition into a native model specification, runs the native
fit, and presents the native result through the existing EstimationResult.
No mathematics lives here: likelihoods, derivatives, paths, stationarity and
inference all come from scivolapi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from .result import EstimationResult, FitMetadata
from .roles import Role

_COORDINATES = {
    None: api.Coordinates.automatic,
    "automatic": api.Coordinates.automatic,
    "z": api.Coordinates.z,
    "theta": api.Coordinates.theta,
}


@dataclass(frozen=True, slots=True)
class NativeOptimizationResult:
    x: NDArray[np.float64]
    fun: float
    success: bool
    nit: int
    message: str
    nfev: int
    njev: int


# Densities scivolapi covers, by component class name.
_DENSITIES = {
    "Normal": api.DensityKind.normal,
    "StudentT": api.DensityKind.student_t,
    "SkewT": api.DensityKind.skew_t,
    "GED": api.DensityKind.ged,
    "GeneralizedHyperbolic": api.DensityKind.generalized_hyperbolic,
    "NormalInverseGaussian": api.DensityKind.nig,
    "Hyperbolic": api.DensityKind.hyperbolic,
    "VarianceGamma": api.DensityKind.variance_gamma,
}


def native_spec(spec: Any) -> Optional[api.ModelSpec]:
    """The native specification for a composition, or None if not yet covered."""
    from .components.vol import GARCH

    if spec.get_component(Role.MEAN) is not None:
        return None
    volatility = spec.get_component(Role.VOLATILITY)
    density = spec.get_component(Role.DENSITY)
    if type(volatility) is not GARCH or getattr(volatility, "_is_auto", False):
        return None
    kind = _DENSITIES.get("Normal" if density is None else type(density).__name__)
    if kind is None:
        return None
    native = api.ModelSpec()
    native.volatility = api.VolatilityKind.garch
    native.p, native.q = int(volatility.p), int(volatility.q)
    native.density = kind
    return native


def fit_spec(
    spec: Any,
    data: NDArray[np.float64],
    *,
    coordinates: Optional[str] = None,
    maxiter: int = 1000,
    accuracy: float = 1e-10,
    initial_params: Optional[NDArray[np.float64]] = None,
    index: Any = None,
    name: Optional[str] = None,
) -> EstimationResult:
    """Fit a covered composition with scivolapi."""
    native = native_spec(spec)
    if native is None:
        raise NotImplementedError(f"scivolapi does not cover {spec.uid} yet")
    if coordinates not in _COORDINATES:
        raise ValueError("coordinates must be 'automatic', 'z' or 'theta'")

    model = api.Model(native)
    options = api.FitOptions()
    options.coordinates = _COORDINATES[coordinates]
    options.maximum_iterations = int(maxiter)
    options.accuracy = float(accuracy)
    if initial_params is not None:
        options.start = [float(v) for v in np.asarray(initial_params, dtype=np.float64)]

    y = np.ascontiguousarray(data, dtype=np.float64)
    started = time.perf_counter()
    native_result = api.fit(model, y, options)
    elapsed = time.perf_counter() - started
    if not native_result.estimate_available:
        raise RuntimeError("Fit failed: " + "; ".join(native_result.warnings))

    theta = np.array(native_result.theta)
    phases = tuple(native_result.phases)
    message = "; ".join(native_result.warnings) or "Converged and stationary."
    optimization = NativeOptimizationResult(
        x=theta,
        fun=-native_result.log_likelihood,
        success=native_result.success,
        nit=sum(phase.iterations for phase in phases),
        message=message,
        nfev=sum(phase.value_requests + phase.gradient_requests for phase in phases),
        njev=sum(phase.gradient_requests for phase in phases),
    )
    fit_info = FitMetadata(
        engine="scivolapi",
        solver="+".join(phase.name for phase in phases),
        optimization_space="+".join(phase.name.split(":")[0] for phase in phases),
        optimizer_phases=tuple(phase.name for phase in phases),
        optimizer_terminated=native_result.termination == api.Termination.converged,
        stationarity_status="stationary" if native_result.stationary else "not_stationary",
        inference_status=native_result.inference.name,
        kkt_residual=float(native_result.kkt_residual),
        kkt_tolerance=float(native_result.kkt_tolerance),
        function_evaluations=optimization.nfev,
        gradient_evaluations=optimization.njev,
        hessian_evaluations=sum(phase.hessian_requests for phase in phases),
        maxiter=int(maxiter),
        internal_data_scale=float(native_result.data_scale),
        likelihood_n_obs=int(native_result.scored),
        warnings=tuple(native_result.warnings),
    )
    covariance = np.array(native_result.covariance) if native_result.covariance.size else None
    hessian = np.array(native_result.hessian) if native_result.hessian.size else None
    result = EstimationResult(
        spec=spec,
        optimization_result=optimization,
        data=y,
        sigma2=np.array(native_result.variance),
        time_elapsed=elapsed,
        hessian=hessian,
        cov_matrix=covariance,
        fit_info=fit_info,
        index=index,
        name=name,
    )
    result._resid = np.array(native_result.residual)
    return result
