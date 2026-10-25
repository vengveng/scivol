"""Fixed-parameter evaluation, filtering, forecasting and simulation through scivolapi.

The likelihood, its derivatives, the filtered paths, forecasts and simulated
paths all come from the native library; this module only translates arguments
and draws the standardized innovations a simulation consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from dataclasses import dataclass

from ._evaluation import FilteredState, ForecastResult, SimulationResult, _coerce_params
from ._gh import fixed_gh_lambda
from .result import EstimationResult
from .roles import Role


def _native_model(spec: Any, x: Optional[NDArray[np.float64]]) -> "api.Model":
    from ._scivolapi_fit import native_spec

    mean = spec.get_component(Role.MEAN)
    if x is not None and hasattr(mean, "set_n_exog"):
        mean.set_n_exog(x.shape[1])
    native = native_spec(spec, 0 if x is None else x.shape[1])
    if native is None:
        raise NotImplementedError(f"{spec} is not available for fixed-parameter evaluation")
    return api.Model(native)


def _regressors(x: Any, rows: int, columns: Optional[int] = None) -> Optional[NDArray[np.float64]]:
    if x is None:
        return None
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] != rows or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"x must have shape ({rows}, {columns if columns is not None else 'k'}).")
    return np.ascontiguousarray(array)


def _density(spec: Any) -> str:
    density = spec.get_component(Role.DENSITY)
    return "Normal" if density is None else density.signature


def _shapes(model: "api.Model", theta: NDArray[np.float64], distribution: str) -> dict[str, Optional[float]]:
    """Density shapes by the names the forecast and innovation helpers use."""
    values = dict(zip(model.parameter_names, theta))
    shape_lambda = fixed_gh_lambda(distribution)
    return {
        "nu": values.get("nu"),
        "lam": values.get("lam"),
        "rho": values.get("rho"),
        "zeta": values.get("zeta"),
        "shape_lambda": shape_lambda if shape_lambda is not None else values.get("lambda"),
    }


_SHAPE_COUNT = {"normal": 0, "student_t": 1, "skew_t": 2, "ged": 1, "generalized_hyperbolic": 3, "nig": 2,
                "hyperbolic": 2, "variance_gamma": 2}


def innovation_density(spec: Any, theta: NDArray[np.float64]) -> "api.MarginalDensity":
    """The standardized innovation density of a spec at theta (its shapes are theta's trailing entries)."""
    from ._scivolapi_fit import _DENSITIES

    density = spec.get_component(Role.DENSITY)
    kind = _DENSITIES["Normal" if density is None else type(density).__name__]
    count = _SHAPE_COUNT[kind.name]
    return api.MarginalDensity(kind, [float(v) for v in theta[len(theta) - count:]] if count else [])


def uniforms(rng: np.random.Generator, size: Any) -> NDArray[np.float64]:
    """Uniform draws on the midpoints of a 2^-53 grid: strictly inside (0, 1)."""
    return (rng.integers(0, 2 ** 53, size=size) + 0.5) * 2.0 ** -53


def _innovations(density: "api.MarginalDensity", paths: int, horizon: int, seed: Any) -> NDArray[np.float64]:
    """paths x horizon standardized innovations by inversion of the native distribution function."""
    rng = np.random.default_rng(seed)
    u = uniforms(rng, paths * horizon)
    return np.ascontiguousarray(api.quantiles(density.density, density.shape, u).reshape(paths, horizon))


@dataclass(slots=True)
class NativeForecastResult(ForecastResult):
    """Forecasts whose quantiles come from the model itself.

    One step ahead y_{T+1} = mean + sqrt(h_{T+1}) z exactly, so its quantiles are
    the innovation density's. Beyond one step y_{T+k} mixes over the random
    path of h and the mean's lags; its quantiles are those of `simulations`
    simulated paths (`seed`).
    """

    native_density: Any = None
    _paths: Any = None

    def quantile(self, level: float | NDArray[np.float64]) -> NDArray[np.float64]:
        levels = np.asarray(level, dtype=np.float64)
        flat = np.atleast_1d(levels)
        if np.any((flat <= 0.0) | (flat >= 1.0)):
            raise ValueError("Quantile levels must lie strictly between 0 and 1.")
        z = api.quantiles(self.native_density.density, self.native_density.shape, np.ascontiguousarray(flat))
        out = np.empty((len(self.mean), flat.size))
        out[0] = self.mean[0] + np.sqrt(self.residual_variance[0]) * z
        if len(self.mean) > 1:
            out[1:] = np.quantile(self._paths()[:, 1:], flat, axis=0).T
        return out[:, 0] if levels.ndim == 0 else out


_DISPLAY = {"phi": "ar", "theta": "ma", "x": "exog", "lambda": "lam"}


def _parameters(spec: Any, model: "api.Model", params: Any) -> NDArray[np.float64]:
    """theta in the model's order from an array, a mapping by parameter name, or (None) the components'
    fitted values."""
    names = list(model.parameter_names)
    if isinstance(params, Mapping):
        flat = {}
        for key, value in params.items():
            base, bracket, rest = str(key).partition("[")
            native = _DISPLAY.get(base, base) + bracket + rest
            if native == "lam" and "lam" not in names and "lambda" in names:
                native = "lambda"
            flat[native] = value
        if all(name in flat for name in names):
            theta = np.array([float(flat[name]) for name in names])
        else:
            theta = np.asarray(_coerce_params(spec, params), dtype=np.float64)
    elif params is None:
        theta = np.asarray(_coerce_params(spec, None), dtype=np.float64)
    else:
        theta = np.asarray(params, dtype=np.float64).ravel()
    if theta.size != len(names):
        raise ValueError(f"Expected {len(names)} parameters ({', '.join(names)}), received {theta.size}.")
    return np.ascontiguousarray(theta)


class _Bound:
    """A spec at fixed parameters on the rows fit() would use."""

    def __init__(self, spec: Any, data: Any, params: Any, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None) -> None:
        from ._scivolapi_fit import prepare_sample

        self.sample = prepare_sample(spec, data, x, hold_back, scale)
        self.y, self.x = self.sample.y, self.sample.x
        self.model = api.Model(self.sample.native)
        self.theta = _parameters(spec, self.model, params)
        self.jet = api.Jet(self.model, self.y, self.x)

    def evaluate(self, request: int) -> dict[str, Any]:
        out = self.jet.theta(self.theta, request)
        if out["status"] != "ok":
            raise ValueError(f"The likelihood is not available at these parameters ({out['status']}).")
        return out


def loglikelihood(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                  scale: Optional[float] = None) -> float:
    return -float(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.VALUE)["value"])


def score(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
          scale: Optional[float] = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.GRADIENT)["gradient"])


def hessian(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
            scale: Optional[float] = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.HESSIAN)["hessian"])


def filter_state(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None) -> FilteredState:
    bound = _Bound(spec, data, params, x, hold_back, scale)
    return _state(spec, bound.model, bound.jet, bound.y, bound.theta)


def fixed_result(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None, index: Any = None, name: Optional[str] = None
                 ) -> "NativeEstimationResult":
    """A result at supplied parameters: the fitted paths, likelihood and Hessian without optimization."""
    from ._evaluation import FixedResult, _FixedOptResult
    from ._scivolapi_fit import unpack_components
    from .result import FitMetadata

    bound = _Bound(spec, data, params, x, hold_back, scale)
    state = _state(spec, bound.model, bound.jet, bound.y, bound.theta)
    unpack_components(spec, bound.theta)
    first = bound.sample.skip + bound.jet.first_scored
    result = NativeFixedResult(
        spec, _FixedOptResult(bound.theta.copy(), -state.log_likelihood), state.data,
        sigma2=state.sigma2.copy(), hessian=None if state.hessian is None else -state.hessian,
        fit_info=FitMetadata(engine="scivolapi", likelihood_n_obs=len(state.data),
                             requested_hold_back=bound.sample.hold_back, effective_hold_back=first,
                             original_n_obs=bound.sample.original_n_obs, effective_n_obs=len(bound.y),
                             scale=bound.sample.scale),
        method="FIXED", parameter_source="supplied", index=None if index is None else index[first:], name=name,
    )
    result._resid = state.residuals.copy()
    result._fit_x = bound.x
    result._attach(bound.model, bound.y, bound.x, bound.sample.skip)
    result._filtered_state_cache = state
    return result


def _state(spec: Any, model: "api.Model", jet: "api.Jet", y: NDArray[np.float64],
           theta: NDArray[np.float64]) -> FilteredState:
    """Paths, log-likelihood, score and Hessian of the log-likelihood on the scored rows."""
    out = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.PATHS)
    if not out["valid"] & api.PATHS:
        raise ValueError(f"The model cannot be filtered at these parameters ({out['status']}).")
    scored = y[jet.first_scored:]
    distribution = _density(spec)
    return FilteredState(
        data=scored.copy(),
        mean=scored - out["residual"],
        residuals=np.asarray(out["residual"]),
        sigma2=np.asarray(out["variance"]),
        log_likelihood=-float(out["value"]),
        score=-np.asarray(out["gradient"]) if "gradient" in out else None,
        hessian=-np.asarray(out["hessian"]) if "hessian" in out else None,
        distribution=distribution,
        **_shapes(model, theta, distribution),
    )


def forecast(spec: Any, model: "api.Model", theta: NDArray[np.float64], y: NDArray[np.float64],
             x: Optional[NDArray[np.float64]], horizon: int, future_x: Any = None, *,
             simulations: int = 10_000, seed: Any = 0) -> ForecastResult:
    """Forecasts after the sample (y, x); see scivolapi::forecast for which parts are exact.

    APARCH variance forecasts beyond one step average `simulations` paths drawn
    with `seed`.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    columns = 0 if x is None else x.shape[1]
    future = _regressors(future_x, horizon, columns) if columns else None
    if columns and future is None:
        raise ValueError(f"Forecasts need the exogenous regressors of the horizon: x with shape ({horizon}, {columns}).")
    distribution = _density(spec)
    density = innovation_density(spec, theta)
    from .components.vol import APARCH

    def draws() -> NDArray[np.float64]:
        return _innovations(density, int(simulations), horizon, seed)

    aparch = horizon > 1 and isinstance(spec.get_component(Role.VOLATILITY), APARCH)
    out = api.forecast(model, theta, y, horizon, x, future, draws() if aparch else None)
    cache: dict[str, NDArray[np.float64]] = {}

    def paths() -> NDArray[np.float64]:  # simulated y paths, for quantiles beyond one step
        if "y" not in cache:
            cache["y"] = api.simulate(model, theta, y, draws(), x, future)["y"]
        return cache["y"]

    return NativeForecastResult(
        mean=np.asarray(out["mean"]),
        variance=np.asarray(out["variance"]),
        residual_variance=np.asarray(out["residual_variance"]),
        distribution=distribution,
        **_shapes(model, theta, distribution),
        native_density=density,
        _paths=paths,
    )


def simulate(spec: Any, n_obs: int, params: Any, *, burn: int = 500, seed: Any = None,
             x: Any = None) -> SimulationResult:
    """A series of n_obs rows after `burn` discarded rows, started at the stationary values."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    burn = max(int(burn), 0)
    total = n_obs + burn
    regressors = _regressors(x, total)
    model = _native_model(spec, regressors)
    theta = _parameters(spec, model, params)
    z = _innovations(innovation_density(spec, theta), 1, total, seed)
    out = api.simulate_series(model, theta, z, regressors)
    keep = slice(burn, total)
    return SimulationResult(
        data=np.asarray(out["y"][0, keep]),
        mean=np.asarray(out["mean"][0, keep]),
        residuals=np.asarray(out["residual"][0, keep]),
        sigma2=np.asarray(out["variance"][0, keep]),
        innovations=np.asarray(z[0, keep]),
    )


def oos_losses(result: "NativeEstimationResult", full_data: Any, *, full_x: Any, start: int, stop: int,
               loss: str) -> NDArray[np.float64]:
    """Losses of rows [start, stop) of `full_data` (indexed like the data passed to fit) at the fitted parameters.

    The sample is filtered from the fit's first row through `stop`. logscore is
    each row's negative log density (the likelihood's own terms); qlike and mse
    compare the squared residual with the conditional variance.
    """
    skip = result._native_skip
    scale = float(result.fit_info.scale or 1.0)
    y = np.ascontiguousarray(np.asarray(full_data, dtype=np.float64)[skip:stop] * scale)
    x = None if full_x is None else _regressors(np.asarray(full_x, dtype=np.float64)[skip:stop], len(y))
    jet = api.Jet(result._native_model, y, x)
    first = skip + jet.first_scored
    if start < first or stop <= start:
        raise ValueError("The OOS slice must lie after the rows that condition the fit.")
    out = jet.theta(np.ascontiguousarray(result.params), api.TERMS | api.PATHS)
    if out["status"] != "ok":
        raise ValueError(f"The model cannot be filtered over the OOS slice ({out['status']}).")
    rows = slice(start - first, stop - first)
    key = loss.lower()
    if key == "logscore":
        return np.ascontiguousarray(out["terms"][rows])
    e2, h = out["residual"][rows] ** 2, out["variance"][rows]
    if key == "qlike":
        return np.log(h) + e2 / h
    if key == "mse":
        return (e2 - h) ** 2
    raise ValueError("oos_loss must be 'logscore', 'qlike', or 'mse'.")


class NativeEstimationResult(EstimationResult):
    """An estimation result whose filter and forecasts come from scivolapi.

    Keeps the sample the model was fitted on, including the rows that only
    supply the mean's lags.
    """

    def _attach(self, model: "api.Model", y: NDArray[np.float64], x: Optional[NDArray[np.float64]],
                skip: int) -> None:
        self._native_model = model
        self._native_y = y  # after the hold-back rows, including the mean's lag rows
        self._native_x = x
        self._native_skip = skip

    def forecast(self, horizon: int = 10, *, x: Any = None, simulations: int = 10_000, seed: Any = 0,
                 return_variance: bool = True, return_volatility: bool = True) -> ForecastResult:
        """Forecast the mean, its variance and the residual variance h steps ahead.

        Exact for the mean and for GARCH, GJR and EGARCH variances at every
        horizon; APARCH variances beyond one step average `simulations` paths.
        EGARCH forecasts beyond one step do not exist under densities whose
        exponential moments are infinite (Student-t, skew-t); those raise.
        """
        return forecast(self.spec, self._native_model, np.ascontiguousarray(self.params), self._native_y,
                        self._native_x, horizon, x, simulations=simulations, seed=seed)

    def _get_param_names(self) -> list[str]:
        """The native parameter names in the summary's display convention."""
        from ._settings import settings

        renames = {"ar": "phi", "ma": "theta", "exog": "x", "lam": "lambda"}
        names = []
        for name in self._native_model.parameter_names:
            base, bracket, rest = name.partition("[")
            names.append(settings.names.resolve(renames.get(base, base) + bracket + rest))
        return names

    def _get_filtered_state(self) -> FilteredState:
        if self._filtered_state_cache is None:
            jet = api.Jet(self._native_model, self._native_y, self._native_x)
            self._filtered_state_cache = _state(self.spec, self._native_model, jet, self._native_y,
                                                np.ascontiguousarray(self.params))
        return self._filtered_state_cache


class NativeFixedResult(NativeEstimationResult):
    def __repr__(self) -> str:
        return f"FixedResult({self.spec}, LL={self.log_likelihood:.4f})"
