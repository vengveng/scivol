"""Fitting: a composition becomes a native model specification, the native
library estimates it, and the result is presented as an EstimationResult.

No mathematics lives here: likelihoods, derivatives, paths, stationarity and
inference all come from the native library.
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


# Native density kinds by component class name.
DENSITIES = {
    "Normal": api.DensityKind.normal,
    "StudentT": api.DensityKind.student_t,
    "SkewT": api.DensityKind.skew_t,
    "GED": api.DensityKind.ged,
    "GeneralizedHyperbolic": api.DensityKind.generalized_hyperbolic,
    "NormalInverseGaussian": api.DensityKind.nig,
    "Hyperbolic": api.DensityKind.hyperbolic,
    "VarianceGamma": api.DensityKind.variance_gamma,
}


def native_mean(mean: Any, exog: int) -> Optional[api.MeanSpec]:
    """The native mean for a mean component (None: residuals are the data)."""
    from .components.mean import ARMA, ARX, HARX

    native = api.MeanSpec()
    if mean is None:
        return native if exog == 0 else None
    if isinstance(mean, ARMA):
        if exog:
            return None
        native.constant, native.ar, native.ma = bool(mean.constant), int(mean.p), int(mean.q)
    elif isinstance(mean, ARX):
        native.constant, native.ar, native.exog = bool(mean.constant), int(mean.lags), exog
    elif isinstance(mean, HARX):
        native.constant, native.har, native.exog = bool(mean.constant), [int(h) for h in mean.horizons], exog
    else:
        return None
    return native


def native_spec(spec: Any, exog: int = 0) -> Optional[api.ModelSpec]:
    """The native specification for a composition, or None if not yet covered."""
    from .components.vol import APARCH, EGARCH, GARCH, GJRGARCH

    if hasattr(spec, "_native_definition"):
        return spec._native_definition

    mean = native_mean(spec.get_component(Role.MEAN), exog)
    if mean is None:
        return None
    volatility = spec.get_component(Role.VOLATILITY)
    density = spec.get_component(Role.DENSITY)
    volatility_kind = {
        GARCH: api.VolatilityKind.garch, GJRGARCH: api.VolatilityKind.gjr, EGARCH: api.VolatilityKind.egarch,
        APARCH: api.VolatilityKind.aparch, type(None): api.VolatilityKind.constant,
    }.get(type(volatility))
    if volatility_kind is None:
        return None
    kind = DENSITIES.get("Normal" if density is None else type(density).__name__)
    if kind is None:
        return None
    native = api.ModelSpec()
    native.mean = mean
    native.volatility = volatility_kind
    native.p, native.q = (0, 0) if volatility is None else (int(volatility.p), int(volatility.q))
    if volatility is not None:
        native.stationarity = getattr(api.Stationarity, volatility.stationarity)
    native.density = kind
    return native


@dataclass(frozen=True)
class PreparedSample:
    """The rows a model sees: y and x from row `skip` on (scaled), so that the
    first scored row is max(hold_back, lags)."""

    native: Any
    y: NDArray[np.float64]
    x: Optional[NDArray[np.float64]]
    skip: int
    hold_back: int
    original_n_obs: int
    scale: float


def fit_spec_on_plan(spec: Any, plan: api.ComparisonPlan, *,
                     options: Optional[api.FitOptions] = None,
                     exog: Optional[int] = None) -> api.FitResult:
    """Fit a candidate on an immutable native comparison plan.

    Returns the native fit for restriction testing. ``exog`` selects the number
    of leading plan columns retained by this candidate, defaulting to all.
    Single-fit sample preparation and result presentation are unchanged.
    """
    native = native_spec(spec, plan.columns if exog is None else exog)
    if native is None:
        raise NotImplementedError(f"{spec} cannot be estimated: unsupported component combination")
    return api.fit(api.Model(native), plan, api.FitOptions() if options is None else options)


def prepare_sample(spec: Any, data: Any, x: Any = None, hold_back: int = 0,
                   scale: Optional[float] = None) -> PreparedSample:
    from ._validation import validate_data, validate_hold_back, validate_regressors, validate_scale

    y = validate_data(np.asarray(data, dtype=np.float64))
    held = validate_hold_back(hold_back)
    if held >= len(y):
        raise ValueError("hold_back leaves no observations to estimate on")
    scale_factor = validate_scale(scale)
    x_all = None if x is None else validate_regressors(x, len(y))
    if hasattr(spec, "_native_definition") and x_all is not None:
        columns = spec._native_definition.mean.exog
        if x_all.shape[1] < columns:
            raise ValueError("not enough exogenous columns for this specification")
        x_all = np.ascontiguousarray(x_all[:, :columns]) if columns else None
    mean = spec.get_component(Role.MEAN)
    if x_all is not None and hasattr(mean, "set_n_exog"):
        mean.set_n_exog(x_all.shape[1])
    elif x_all is None and getattr(mean, "n_exog", 0):
        raise ValueError(f"{mean.signature} requires exogenous regressors via `x=`.")
    native = native_spec(spec, 0 if x_all is None else x_all.shape[1])
    if native is None:
        raise NotImplementedError(f"{spec} cannot be estimated: unsupported component combination")
    skip = max(held - native.mean.conditioning, 0)  # rows before the lag rows of the first scored row
    return PreparedSample(
        native=native,
        y=np.ascontiguousarray(y[skip:] * scale_factor, dtype=np.float64),
        x=None if x_all is None else np.ascontiguousarray(x_all[skip:]),
        skip=skip,
        hold_back=held,
        original_n_obs=len(y),
        scale=scale_factor,
    )


class FitError(RuntimeError):
    """A fit that produced no finite, feasible estimate."""


def failure_message(warnings: Any, y: Optional[NDArray[np.float64]] = None) -> str:
    """The reasons for a fit failure, and the sample's share of exact zeros when it is large."""
    message = "no finite, feasible estimate: " + "; ".join(warnings)
    if y is not None and y.size:
        zeros = float(np.mean(np.asarray(y) == 0.0))
        if zeros >= 0.05:
            message += (f". {zeros:.0%} of the observations are exactly zero; with exact zeros, peaked densities"
                        " and collapsing variances can leave the likelihood without a maximum")
    return message


def selected_path(native_result: api.FitResult) -> list[str]:
    """The phases of the start that produced the estimate (fit_info.solver); every
    start's phases, labelled "start k: ", stay in optimizer_phases."""
    chosen = next((k for k, s in enumerate(native_result.starts) if s.selected), 0)
    label = f"start {chosen + 1}: "
    names = [phase.name for phase in native_result.phases]
    if chosen == 0:
        return [n for n in names if not n.startswith("start ")]
    return [n[len(label):] for n in names if n.startswith(label)]


def set_starts(options: api.FitOptions, starts: Any, model: Any, explicit: bool = False) -> None:
    """Apply a `starts` choice to native fit options (see fit_spec)."""
    if isinstance(starts, str):
        if starts != "auto":
            raise ValueError("starts must be 'auto', a positive number of starts, or a list of parameter vectors")
        return
    if explicit:
        raise ValueError("initial_params is a single explicit start; pass starts='auto' with it, or use starts alone")
    if isinstance(starts, (int, np.integer)) and not isinstance(starts, bool):
        if starts < 1:
            raise ValueError("the number of starts must be at least 1")
        options.start_count = int(starts)
        return
    points = [np.asarray(getattr(s, "params", s), dtype=np.float64).ravel() for s in starts]
    if not points:
        raise ValueError("starts needs at least one parameter vector")
    for p in points:
        if p.size != model.dimension:
            raise ValueError(f"each start needs {model.dimension} values ({', '.join(model.parameter_names)}), "
                             f"received {p.size}.")
    options.starts = [[float(v) for v in p] for p in points]


def fit_spec(
    spec: Any,
    data: NDArray[np.float64],
    *,
    x: Any = None,
    hold_back: int = 0,
    scale: Optional[float] = None,
    method: str = "mle",
    coordinates: Optional[str] = None,
    maxiter: int = 1000,
    accuracy: float = 1e-10,
    initial_params: Optional[NDArray[np.float64]] = None,
    starts: Any = "auto",
    index: Any = None,
    name: Optional[str] = None,
    **unknown: Any,
) -> EstimationResult:
    """Fit a composition.

    The likelihood scores the rows from max(hold_back, lags) on, where lags is
    the mean's largest y lag: held-back rows are not scored but still supply the
    mean's lags, so every model with the same hold_back scores the same rows.
    Paths and the index cover the scored rows. `method="qmle"` fits the mean and variance
    parameters by Gaussian quasi-maximum likelihood with sandwich covariance;
    with a non-Normal density the shape parameters follow in a second stage.

    `starts` chooses the starting points: "auto" (the model's candidate starts,
    each beyond the first probed briefly and finished only if the probe already
    beats the best finished start), a number n (the first n candidates, each in
    full; 1 is a single start), or explicit parameter vectors (each in full).
    `initial_params` is a single explicit start.
    """
    if unknown:
        raise TypeError(f"fit() got unexpected keyword argument(s) {', '.join(map(repr, unknown))}; its estimation "
                        "options are method, coordinates, maxiter, accuracy, initial_params and starts")
    sample = prepare_sample(spec, data, x, hold_back, scale)
    native, y, x_arr, skip, held = sample.native, sample.y, sample.x, sample.skip, sample.hold_back
    original_n_obs, scale_factor = sample.original_n_obs, sample.scale
    if coordinates not in _COORDINATES:
        raise ValueError("coordinates must be 'automatic', 'z' or 'theta'")
    estimator = {"mle": api.Estimator.mle, "qmle": api.Estimator.qmle}.get(str(method).lower())
    if estimator is None:
        raise ValueError("method must be 'mle' or 'qmle'")

    model = api.Model(native)
    options = api.FitOptions()
    options.estimator = estimator
    options.coordinates = _COORDINATES[coordinates]
    options.maximum_iterations = int(maxiter)
    options.accuracy = float(accuracy)
    if initial_params is not None:  # parameters, or a previous result to warm start from
        source = getattr(initial_params, "spec", None)
        if source is not None and str(source) != str(spec):
            raise ValueError(f"Warm-start result is for {source}; this fit is {spec}.")
        start = np.asarray(getattr(initial_params, "params", initial_params), dtype=np.float64).ravel()
        if start.size != model.dimension:
            raise ValueError(f"initial_params needs {model.dimension} values ({', '.join(model.parameter_names)}), "
                             f"received {start.size}.")
        options.start = [float(v) for v in start]
    set_starts(options, starts, model, explicit=initial_params is not None)

    started = time.perf_counter()
    native_result = api.fit(model, y, options, x_arr)
    elapsed = time.perf_counter() - started
    return present_fit(spec, sample, native_result, maxiter=maxiter,
                       warm_started=initial_params is not None, elapsed=elapsed,
                       index=index, name=name)


def spec_from_native(definition: api.ModelSpec):
    """Present a native child with its ordinary component representation."""
    from . import components
    from .spec import CompositeSpec

    model = api.Model(definition)
    parts = [getattr(components, name)(*args, **kwargs)
             for name, args, kwargs in model.components]
    spec = CompositeSpec(*parts)
    spec._native_definition = definition
    mean = spec.get_component(Role.MEAN)
    if mean is not None and hasattr(mean, "set_n_exog"):
        mean.set_n_exog(definition.mean.exog)
    return spec


def present_fit(spec, sample: PreparedSample, native_result: api.FitResult, *,
                maxiter: int, warm_started: bool, elapsed: float,
                index=None, name=None) -> EstimationResult:
    """One result presenter for ordinary fits and identical comparison-plan fits."""
    native, y, x_arr = sample.native, sample.y, sample.x
    skip, held = sample.skip, sample.hold_back
    original_n_obs, scale_factor = sample.original_n_obs, sample.scale
    model = api.Model(native)
    if not native_result.estimate_available:
        raise FitError(failure_message(native_result.warnings, y))

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
        solver="+".join(selected_path(native_result)),
        optimization_space="+".join(name.removeprefix("probe ").split(":")[0] for name in selected_path(native_result)),
        optimizer_phases=tuple(phase.name for phase in phases),
        optimizer_terminated=native_result.termination == api.Termination.converged,
        stationarity_status="stationary" if native_result.stationary else "not_stationary",
        inference_status=native_result.inference.name,
        kink_point=bool(native_result.kink_point),
        active_kink_rows=tuple(skip+row for row in native_result.active_kink_rows),
        excluded_boundary=tuple(native_result.excluded_boundary),
        starts=tuple(dict(theta=np.array(s.theta), candidate=s.candidate, finished=s.finished,
                          log_likelihood=s.log_likelihood, success=s.success, selected=s.selected)
                     for s in native_result.starts),
        kkt_residual=float(native_result.kkt_residual),
        kkt_tolerance=float(native_result.kkt_tolerance),
        function_evaluations=optimization.nfev,
        gradient_evaluations=optimization.njev,
        hessian_evaluations=sum(phase.hessian_requests for phase in phases),
        maxiter=int(maxiter),
        warm_started=warm_started,
        internal_data_scale=float(native_result.data_scale),
        likelihood_n_obs=int(native_result.scored),
        requested_hold_back=held,
        effective_hold_back=skip + native.mean.conditioning,  # rows before the first scored row
        original_n_obs=original_n_obs,
        effective_n_obs=len(y),
        scale=scale_factor,
        warnings=tuple(native_result.warnings),
    )
    def matrix(values: Any) -> Optional[NDArray[np.float64]]:
        return np.array(values) if values.size else None

    qmle = native_result.estimator == api.Estimator.qmle
    from ._evaluation import unpack_components

    unpack_components(spec, theta)
    first = skip + native.mean.conditioning
    result = EstimationResult(
        spec=spec,
        optimization_result=optimization,
        data=y[native.mean.conditioning:],
        sigma2=np.array(native_result.variance),
        time_elapsed=elapsed,
        hessian=matrix(native_result.hessian),
        cov_matrix=matrix(native_result.covariance),
        opg=matrix(native_result.opg),
        cov_robust=matrix(native_result.robust_covariance),
        fit_info=fit_info,
        method="QMLE" if qmle else "MLE",
        index=None if index is None else index[first:],
        name=name,
    )
    result._resid = np.array(native_result.residual)
    result._fit_x = x_arr
    result.attach_sample(model, y, x_arr, skip)
    return result
