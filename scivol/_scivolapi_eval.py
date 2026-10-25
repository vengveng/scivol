"""Fixed-parameter evaluation, filtering, forecasting and simulation through scivolapi.

The likelihood, its derivatives, the filtered paths, forecasts and simulated
paths all come from the native library; this module only translates arguments
and draws the standardized innovations a simulation consumes.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from ._evaluation import FilteredState, ForecastResult, SimulationResult, _coerce_params, _draw_standardized_innovations
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


def _innovations(distribution: str, shapes: dict[str, Optional[float]], paths: int, horizon: int,
                 seed: Any) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    draws = _draw_standardized_innovations(rng, paths * horizon, distribution, shapes["nu"], shapes["lam"],
                                           shapes["rho"], shapes["zeta"], shapes["shape_lambda"])
    return np.ascontiguousarray(draws.reshape(paths, horizon))


class _Bound:
    """A spec at fixed parameters on a prepared sample."""

    def __init__(self, spec: Any, data: Any, params: Any, x: Any = None) -> None:
        self.y = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        self.x = _regressors(x, len(self.y))
        self.model = _native_model(spec, self.x)
        self.theta = np.ascontiguousarray(_coerce_params(spec, params))
        self.jet = api.Jet(self.model, self.y, self.x)

    def evaluate(self, request: int) -> dict[str, Any]:
        out = self.jet.theta(self.theta, request)
        if out["status"] != "ok":
            raise ValueError(f"The likelihood is not available at these parameters ({out['status']}).")
        return out


def loglikelihood(spec: Any, data: Any, params: Any, *, x: Any = None) -> float:
    return -float(_Bound(spec, data, params, x).evaluate(api.VALUE)["value"])


def score(spec: Any, data: Any, params: Any, *, x: Any = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x).evaluate(api.GRADIENT)["gradient"])


def hessian(spec: Any, data: Any, params: Any, *, x: Any = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x).evaluate(api.HESSIAN)["hessian"])


def filter_state(spec: Any, data: Any, params: Any, *, x: Any = None) -> FilteredState:
    bound = _Bound(spec, data, params, x)
    return _state(spec, bound.model, bound.jet, bound.y, bound.theta)


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
    shapes = _shapes(model, theta, distribution)
    z = None
    from .components.vol import APARCH

    if horizon > 1 and isinstance(spec.get_component(Role.VOLATILITY), APARCH):
        z = _innovations(distribution, shapes, simulations, horizon, seed)
    out = api.forecast(model, theta, y, horizon, x, future, z)
    return ForecastResult(
        mean=np.asarray(out["mean"]),
        variance=np.asarray(out["variance"]),
        residual_variance=np.asarray(out["residual_variance"]),
        distribution=distribution,
        **shapes,
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
    theta = np.ascontiguousarray(_coerce_params(spec, params))
    distribution = _density(spec)
    z = _innovations(distribution, _shapes(model, theta, distribution), 1, total, seed)
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
